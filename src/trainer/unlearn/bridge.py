import json
import logging
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from trainer.unlearn.npo import NPO
from trainer.utils import compute_dpo_loss


logger = logging.getLogger(__name__)


class BRIDGENPO(NPO):
    """NPO with retain-side prior-guided boundary DRO.

    This is the first-round BRIDGE implementation. It keeps the NPO erase loss
    from the existing trainer and adds fixed-weight global retain KL plus an
    optional instance-level DRO risk over the current retain candidate batch.
    """

    VALID_PRIORS = {
        "none",
        "uniform",
        "history",
        "refresh_gs",
        "online_gs",
        "refresh_pi",
        "online_pi",
    }

    def __init__(
        self,
        bridge_prior: str = "none",
        bridge_lambda_g: float = 0.0,
        bridge_lambda_b: float = 0.0,
        bridge_dro_temperature: float = 1.0,
        bridge_prior_temperature: float = 1.0,
        bridge_history_beta: float = 0.9,
        bridge_refresh_interval: int = 20,
        bridge_gs_score: str = "cosine",
        bridge_pi_step_size: Optional[float] = None,
        bridge_worst_k_frac: float = 0.1,
        bridge_log_every: int = 5,
        bridge_diagnostics_filename: str = "bridge_diagnostics.jsonl",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.bridge_prior = bridge_prior.lower()
        if self.bridge_prior not in self.VALID_PRIORS:
            raise ValueError(
                f"Unsupported bridge_prior={bridge_prior}. "
                f"Expected one of {sorted(self.VALID_PRIORS)}."
            )
        if bridge_dro_temperature <= 0:
            raise ValueError("bridge_dro_temperature must be positive")
        if bridge_prior_temperature <= 0:
            raise ValueError("bridge_prior_temperature must be positive")
        if bridge_refresh_interval <= 0:
            raise ValueError("bridge_refresh_interval must be positive")
        if not (0.0 < bridge_history_beta < 1.0):
            raise ValueError("bridge_history_beta must be in (0, 1)")

        self.bridge_lambda_g = float(bridge_lambda_g)
        self.bridge_lambda_b = float(bridge_lambda_b)
        self.bridge_dro_temperature = float(bridge_dro_temperature)
        self.bridge_prior_temperature = float(bridge_prior_temperature)
        self.bridge_history_beta = float(bridge_history_beta)
        self.bridge_refresh_interval = int(bridge_refresh_interval)
        self.bridge_gs_score = bridge_gs_score.lower()
        self.bridge_pi_step_size = (
            float(bridge_pi_step_size)
            if bridge_pi_step_size is not None
            else float(getattr(self.args, "learning_rate", 1e-5))
        )
        self.bridge_worst_k_frac = float(bridge_worst_k_frac)
        self.bridge_log_every = int(bridge_log_every)
        self.bridge_diagnostics_filename = bridge_diagnostics_filename

        self._bridge_call_idx = 0
        self._history_ema: Dict[int, float] = {}
        self._last_kl: Dict[int, float] = {}
        self._score_cache: Dict[int, float] = {}
        self._last_score_refresh_call: Optional[int] = None
        self._last_metrics: Dict[str, float] = {}
        self._diagnostic_count = 0
        self._warned_missing_index = False
        self._warned_deepspeed_pi = False

    def _split_unlearn_inputs_with_index(self, inputs):
        forget_raw = inputs["forget"]
        retain_raw = inputs.get("retain", None)

        forget_inputs = self._model_inputs(forget_raw)
        retain_inputs = self._model_inputs(retain_raw) if retain_raw is not None else None
        retain_indices = self._retain_indices(retain_raw) if retain_raw is not None else None
        return forget_inputs, retain_inputs, retain_indices

    @staticmethod
    def _model_inputs(batch):
        if batch is None:
            return None
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _retain_indices(self, retain_raw) -> Optional[List[int]]:
        if retain_raw is None or "index" not in retain_raw:
            if not self._warned_missing_index and self.bridge_prior in {
                "history",
                "refresh_gs",
                "refresh_pi",
            }:
                logger.warning(
                    "BRIDGE prior %s works best when the collator keeps the "
                    "retain dataset index. Falling back to batch-local scores.",
                    self.bridge_prior,
                )
                self._warned_missing_index = True
            return None
        indices = retain_raw["index"]
        if isinstance(indices, torch.Tensor):
            return [int(x) for x in indices.detach().cpu().tolist()]
        return [int(x) for x in indices]

    @staticmethod
    def _trainable_params(model) -> List[torch.nn.Parameter]:
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _per_sample_nll(self, model, inputs) -> Tuple[torch.Tensor, object]:
        outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()
        vocab_size = logits.shape[-1]

        token_loss = F.cross_entropy(
            logits.view(-1, vocab_size).float(),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)
        mask = labels.ne(-100)
        denom = mask.sum(dim=-1).clamp_min(1)
        losses = (token_loss * mask).sum(dim=-1) / denom
        return losses, outputs

    def _per_sample_retain_kl(self, model, retain_inputs) -> Tuple[torch.Tensor, object]:
        with torch.no_grad():
            ref_outputs = self.ref_model(**retain_inputs)
            ref_logits = ref_outputs.logits[..., :-1, :].contiguous().float()
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        outputs = model(**retain_inputs)
        current_logits = outputs.logits[..., :-1, :].contiguous().float()
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        labels = retain_inputs["labels"][..., 1:].contiguous()
        mask = labels.ne(-100)
        if "attention_mask" in retain_inputs:
            mask = mask & retain_inputs["attention_mask"][..., 1:].bool()

        token_kl = F.kl_div(
            current_log_probs,
            ref_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        denom = mask.sum(dim=-1).clamp_min(1)
        per_sample_kl = (token_kl * mask).sum(dim=-1) / denom
        return per_sample_kl, outputs

    @staticmethod
    def _standardize(scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() <= 1:
            return torch.zeros_like(scores)
        std = scores.float().std(unbiased=False)
        if not torch.isfinite(std) or float(std.detach().cpu()) < 1e-8:
            return torch.zeros_like(scores)
        return (scores - scores.mean()) / (std + 1e-8)

    def _prior_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        z_scores = self._standardize(scores.float())
        prior = torch.softmax(z_scores / self.bridge_prior_temperature, dim=0)
        return prior.detach()

    @staticmethod
    def _uniform_prior(size: int, device) -> torch.Tensor:
        return torch.full((size,), 1.0 / max(size, 1), device=device)

    def _history_scores(self, retain_indices, current_kl: torch.Tensor) -> torch.Tensor:
        if retain_indices is None:
            return torch.zeros_like(current_kl.detach())
        values = [self._history_ema.get(int(idx), 0.0) for idx in retain_indices]
        return torch.tensor(values, device=current_kl.device, dtype=current_kl.dtype)

    def _update_history(self, retain_indices, current_kl: torch.Tensor) -> None:
        if retain_indices is None:
            return
        current_values = current_kl.detach().float().cpu().tolist()
        for idx, value in zip(retain_indices, current_values):
            idx = int(idx)
            previous = self._last_kl.get(idx, value)
            delta = float(value) - float(previous)
            old_ema = self._history_ema.get(idx, 0.0)
            self._history_ema[idx] = (
                self.bridge_history_beta * old_ema
                + (1.0 - self.bridge_history_beta) * delta
            )
            self._last_kl[idx] = float(value)

    def _cached_scores(
        self, retain_indices, size: int, device, dtype: torch.dtype
    ) -> torch.Tensor:
        if retain_indices is None:
            return torch.zeros(size, device=device, dtype=dtype)
        values = [self._score_cache.get(int(idx), 0.0) for idx in retain_indices]
        return torch.tensor(values, device=device, dtype=dtype)

    def _store_scores(self, retain_indices, scores: torch.Tensor) -> None:
        if retain_indices is None:
            return
        for idx, score in zip(retain_indices, scores.detach().float().cpu().tolist()):
            self._score_cache[int(idx)] = float(score)

    def _should_refresh_scores(self, call_idx: int) -> bool:
        if self.bridge_prior in {"online_gs", "online_pi"}:
            return True
        if self._last_score_refresh_call is None:
            return True
        if call_idx % self.bridge_refresh_interval == 0:
            return self._last_score_refresh_call != call_idx
        return False

    @staticmethod
    def _dot_and_norms(left: Iterable[Optional[torch.Tensor]], right) -> Tuple[float, float, float]:
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_grad, right_grad in zip(left, right):
            if left_grad is None or right_grad is None:
                continue
            left_float = left_grad.detach().float()
            right_float = right_grad.detach().float()
            dot += float((left_float * right_float).sum().cpu())
            left_norm += float((left_float * left_float).sum().cpu())
            right_norm += float((right_float * right_float).sum().cpu())
        return dot, left_norm, right_norm

    def _forget_grads(self, model, forget_inputs):
        params = self._trainable_params(model)
        if not params:
            return [], 0.0
        forget_loss = self._compute_forget_loss(model, forget_inputs)
        grads = torch.autograd.grad(
            forget_loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        detached = tuple(grad.detach() if grad is not None else None for grad in grads)
        return detached, float(forget_loss.detach().float().cpu())

    def _compute_gs_scores(self, model, forget_inputs, retain_inputs) -> Tuple[torch.Tensor, Dict[str, float]]:
        started = time.time()
        params = self._trainable_params(model)
        if not params:
            size = retain_inputs["input_ids"].shape[0]
            return torch.zeros(size, device=retain_inputs["input_ids"].device), {
                "bridge_prior_seconds": 0.0,
                "bridge_forget_probe_loss": 0.0,
            }

        forget_grads, forget_probe_loss = self._forget_grads(model, forget_inputs)
        retain_losses, _ = self._per_sample_nll(model, retain_inputs)
        scores = []
        for row_idx, retain_loss in enumerate(retain_losses):
            retain_graph = row_idx < retain_losses.shape[0] - 1
            retain_grads = torch.autograd.grad(
                retain_loss,
                params,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=True,
            )
            dot, retain_norm, forget_norm = self._dot_and_norms(retain_grads, forget_grads)
            if self.bridge_gs_score == "dot":
                score = -dot
            elif self.bridge_gs_score == "cosine":
                score = -dot / (math.sqrt(retain_norm) * math.sqrt(forget_norm) + 1e-12)
            else:
                raise ValueError("bridge_gs_score must be 'cosine' or 'dot'")
            scores.append(score)

        return torch.tensor(
            scores,
            device=retain_inputs["input_ids"].device,
            dtype=retain_losses.dtype,
        ), {
            "bridge_prior_seconds": time.time() - started,
            "bridge_forget_probe_loss": forget_probe_loss,
        }

    def _compute_pi_scores(self, model, forget_inputs, retain_inputs) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.is_deepspeed_enabled:
            if not self._warned_deepspeed_pi:
                logger.warning(
                    "BRIDGE PI temporary parameter updates are disabled under "
                    "DeepSpeed. Falling back to GS-style first-order scores."
                )
                self._warned_deepspeed_pi = True
            return self._compute_gs_scores(model, forget_inputs, retain_inputs)

        started = time.time()
        params = self._trainable_params(model)
        if not params:
            size = retain_inputs["input_ids"].shape[0]
            return torch.zeros(size, device=retain_inputs["input_ids"].device), {
                "bridge_prior_seconds": 0.0,
                "bridge_forget_probe_loss": 0.0,
            }

        forget_grads, forget_probe_loss = self._forget_grads(model, forget_inputs)

        with torch.no_grad():
            base_losses, _ = self._per_sample_nll(model, retain_inputs)
            applied_updates = []
            try:
                for param, grad in zip(params, forget_grads):
                    if grad is None:
                        applied_updates.append(None)
                        continue
                    update = grad.to(device=param.device, dtype=param.dtype)
                    param.add_(update, alpha=-self.bridge_pi_step_size)
                    applied_updates.append(update)

                perturbed_losses, _ = self._per_sample_nll(model, retain_inputs)
            finally:
                for param, update in zip(params, applied_updates):
                    if update is not None:
                        param.add_(update, alpha=self.bridge_pi_step_size)

            scores = (perturbed_losses - base_losses).detach()

        return scores.to(device=retain_inputs["input_ids"].device), {
            "bridge_prior_seconds": time.time() - started,
            "bridge_forget_probe_loss": forget_probe_loss,
        }

    def _prior_scores(
        self,
        model,
        forget_inputs,
        retain_inputs,
        retain_indices,
        current_kl: torch.Tensor,
        call_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if self.bridge_prior == "uniform":
            return None, {"bridge_prior_seconds": 0.0, "bridge_forget_probe_loss": 0.0}
        if self.bridge_prior == "history":
            return self._history_scores(retain_indices, current_kl), {
                "bridge_prior_seconds": 0.0,
                "bridge_forget_probe_loss": 0.0,
            }
        if self.bridge_prior in {"refresh_gs", "online_gs"}:
            if self._should_refresh_scores(call_idx):
                scores, meta = self._compute_gs_scores(model, forget_inputs, retain_inputs)
                self._store_scores(retain_indices, scores)
                self._last_score_refresh_call = call_idx
                return scores.detach(), meta
            return self._cached_scores(
                retain_indices,
                current_kl.shape[0],
                current_kl.device,
                current_kl.dtype,
            ), {"bridge_prior_seconds": 0.0, "bridge_forget_probe_loss": 0.0}
        if self.bridge_prior in {"refresh_pi", "online_pi"}:
            if self._should_refresh_scores(call_idx):
                scores, meta = self._compute_pi_scores(model, forget_inputs, retain_inputs)
                self._store_scores(retain_indices, scores)
                self._last_score_refresh_call = call_idx
                return scores.detach(), meta
            return self._cached_scores(
                retain_indices,
                current_kl.shape[0],
                current_kl.device,
                current_kl.dtype,
            ), {"bridge_prior_seconds": 0.0, "bridge_forget_probe_loss": 0.0}
        return None, {"bridge_prior_seconds": 0.0, "bridge_forget_probe_loss": 0.0}

    def _boundary_dro(self, per_sample_kl: torch.Tensor, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.bridge_dro_temperature
        safe_prior = prior.clamp_min(1e-12)
        logits = safe_prior.log() + per_sample_kl / tau
        risk = tau * torch.logsumexp(logits, dim=0)
        dro_weights = torch.softmax(logits.detach(), dim=0)
        return risk, dro_weights

    def _metric_dict(
        self,
        loss,
        forget_loss,
        retain_loss,
        global_kl,
        boundary_dro,
        per_sample_kl,
        prior,
        dro_weights,
        prior_scores,
        prior_meta,
        call_idx,
    ) -> Dict[str, float]:
        detached_kl = per_sample_kl.detach().float()
        k = max(1, int(math.ceil(detached_kl.numel() * self.bridge_worst_k_frac)))
        worst_k = detached_kl.topk(k=min(k, detached_kl.numel())).values.mean()
        metrics = {
            "bridge_call_idx": float(call_idx),
            "bridge_loss": float(loss.detach().float().cpu()),
            "bridge_forget_loss": float(forget_loss.detach().float().cpu()),
            "bridge_retain_loss": float(retain_loss.detach().float().cpu())
            if isinstance(retain_loss, torch.Tensor)
            else float(retain_loss),
            "bridge_global_kl": float(global_kl.detach().float().cpu()),
            "bridge_boundary_dro": float(boundary_dro.detach().float().cpu()),
            "bridge_mean_retain_kl": float(detached_kl.mean().cpu()),
            "bridge_worst_k_retain_kl": float(worst_k.cpu()),
            "bridge_prior_weighted_kl": float((prior.detach().float() * detached_kl).sum().cpu()),
            "bridge_dro_weighted_kl": float((dro_weights.detach().float() * detached_kl).sum().cpu()),
            "bridge_prior_entropy": float(
                (-(prior.detach().float().clamp_min(1e-12) * prior.detach().float().clamp_min(1e-12).log()).sum()).cpu()
            ),
            "bridge_dro_weight_entropy": float(
                (-(dro_weights.detach().float().clamp_min(1e-12) * dro_weights.detach().float().clamp_min(1e-12).log()).sum()).cpu()
            ),
        }
        if prior_scores is not None:
            scores = prior_scores.detach().float()
            metrics["bridge_prior_score_mean"] = float(scores.mean().cpu())
            metrics["bridge_prior_score_std"] = float(scores.std(unbiased=False).cpu())
        metrics.update({key: float(value) for key, value in prior_meta.items()})
        return metrics

    def _write_diagnostics(self, metrics: Dict[str, float]) -> None:
        if not self.accelerator.is_local_main_process:
            return
        if self.bridge_log_every <= 0:
            return
        call_idx = int(metrics["bridge_call_idx"])
        if call_idx % self.bridge_log_every != 0:
            return

        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, self.bridge_diagnostics_filename)
        row = {
            "global_step": int(getattr(self.state, "global_step", 0)),
            "bridge_prior": self.bridge_prior,
            "bridge_lambda_g": self.bridge_lambda_g,
            "bridge_lambda_b": self.bridge_lambda_b,
            **metrics,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._diagnostic_count += 1

    def _write_bridge_summary(self, output_dir: Optional[str] = None) -> None:
        if not self.accelerator.is_local_main_process:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            "trainer": self.__class__.__name__,
            "bridge_prior": self.bridge_prior,
            "bridge_lambda_g": self.bridge_lambda_g,
            "bridge_lambda_b": self.bridge_lambda_b,
            "bridge_dro_temperature": self.bridge_dro_temperature,
            "bridge_prior_temperature": self.bridge_prior_temperature,
            "bridge_history_beta": self.bridge_history_beta,
            "bridge_refresh_interval": self.bridge_refresh_interval,
            "bridge_gs_score": self.bridge_gs_score,
            "bridge_pi_step_size": self.bridge_pi_step_size,
            "bridge_worst_k_frac": self.bridge_worst_k_frac,
            "bridge_calls": self._bridge_call_idx,
            "bridge_diagnostic_rows": self._diagnostic_count,
            "last_metrics": self._last_metrics,
        }
        with open(os.path.join(output_dir, "bridge_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        self._write_bridge_summary(output_dir)
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        call_idx = self._bridge_call_idx
        self._bridge_call_idx += 1

        forget_inputs, retain_inputs, retain_indices = self._split_unlearn_inputs_with_index(inputs)

        precomputed_prior_scores = None
        precomputed_prior_meta = {
            "bridge_prior_seconds": 0.0,
            "bridge_forget_probe_loss": 0.0,
        }
        bridge_active = retain_inputs is not None and (
            self.bridge_lambda_g != 0.0 or self.bridge_lambda_b != 0.0
        )
        if (
            bridge_active
            and self.bridge_lambda_b != 0.0
            and self.bridge_prior in {"refresh_gs", "online_gs", "refresh_pi", "online_pi"}
        ):
            if self._should_refresh_scores(call_idx):
                if self.bridge_prior in {"refresh_gs", "online_gs"}:
                    precomputed_prior_scores, precomputed_prior_meta = self._compute_gs_scores(
                        model, forget_inputs, retain_inputs
                    )
                else:
                    precomputed_prior_scores, precomputed_prior_meta = self._compute_pi_scores(
                        model, forget_inputs, retain_inputs
                    )
                self._store_scores(retain_indices, precomputed_prior_scores)
                self._last_score_refresh_call = call_idx
            else:
                precomputed_prior_scores = self._cached_scores(
                    retain_indices,
                    retain_inputs["input_ids"].shape[0],
                    retain_inputs["input_ids"].device,
                    torch.float32,
                )

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        retain_loss = 0.0
        if self.use_retain_loss and retain_inputs is not None:
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        global_kl = self._zero_like_loss(forget_loss)
        boundary_dro = self._zero_like_loss(forget_loss)
        per_sample_kl = torch.zeros(1, device=forget_loss.device, dtype=forget_loss.dtype)
        prior = torch.ones(1, device=forget_loss.device, dtype=forget_loss.dtype)
        dro_weights = torch.ones(1, device=forget_loss.device, dtype=forget_loss.dtype)
        prior_scores = None
        prior_meta = {"bridge_prior_seconds": 0.0, "bridge_forget_probe_loss": 0.0}

        if bridge_active:
            per_sample_kl, _ = self._per_sample_retain_kl(model, retain_inputs)
            global_kl = per_sample_kl.mean()
            prior = self._uniform_prior(per_sample_kl.shape[0], per_sample_kl.device)
            dro_weights = prior

            if self.bridge_lambda_b != 0.0:
                if self.bridge_prior == "none":
                    prior = self._uniform_prior(per_sample_kl.shape[0], per_sample_kl.device)
                elif self.bridge_prior == "uniform":
                    prior = self._uniform_prior(per_sample_kl.shape[0], per_sample_kl.device)
                elif self.bridge_prior in {
                    "refresh_gs",
                    "online_gs",
                    "refresh_pi",
                    "online_pi",
                }:
                    prior_scores = precomputed_prior_scores
                    prior_meta = precomputed_prior_meta
                    prior = self._prior_from_scores(prior_scores.to(per_sample_kl.device))
                else:
                    prior_scores, prior_meta = self._prior_scores(
                        model,
                        forget_inputs,
                        retain_inputs,
                        retain_indices,
                        per_sample_kl.detach(),
                        call_idx,
                    )
                    prior = self._prior_from_scores(prior_scores.to(per_sample_kl.device))
                boundary_dro, dro_weights = self._boundary_dro(per_sample_kl, prior)

            if self.bridge_prior == "history":
                self._update_history(retain_indices, per_sample_kl)

        loss = (
            self.gamma * forget_loss
            + self.alpha * retain_loss
            + self.bridge_lambda_g * global_kl
            + self.bridge_lambda_b * boundary_dro
        )

        if bridge_active:
            metrics = self._metric_dict(
                loss=loss,
                forget_loss=forget_loss,
                retain_loss=retain_loss,
                global_kl=global_kl,
                boundary_dro=boundary_dro,
                per_sample_kl=per_sample_kl,
                prior=prior,
                dro_weights=dro_weights,
                prior_scores=prior_scores,
                prior_meta=prior_meta,
                call_idx=call_idx,
            )
            self._last_metrics = metrics
            self._write_diagnostics(metrics)

        return (loss, forget_outputs) if return_outputs else loss
