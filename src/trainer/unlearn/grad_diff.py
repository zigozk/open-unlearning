import copy
import logging

import torch.nn.functional as F

import torch

from trainer.utils import (
    compute_kl_divergence,
    compute_per_sample_kl_divergence,
    compute_per_sample_nll,
)
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class GradDiff(UnlearnTrainer):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        bridge_prior="none",
        bridge_lambda_g=0.0,
        bridge_lambda_b=0.0,
        bridge_tau=1.0,
        bridge_tau_p=1.0,
        bridge_history_beta=0.9,
        bridge_virtual_step_size=None,
        bridge_log_every=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.bridge_prior = str(bridge_prior or "none").lower()
        self.bridge_lambda_g = float(bridge_lambda_g)
        self.bridge_lambda_b = float(bridge_lambda_b)
        self.bridge_tau = float(bridge_tau)
        self.bridge_tau_p = float(bridge_tau_p)
        self.bridge_history_beta = float(bridge_history_beta)
        self.bridge_virtual_step_size = (
            None
            if bridge_virtual_step_size is None
            else float(bridge_virtual_step_size)
        )
        self.bridge_log_every = int(bridge_log_every)
        self.bridge_enabled = (
            self.bridge_lambda_g != 0.0 or self.bridge_lambda_b != 0.0
        )
        self._bridge_last_logged_step = None
        self._bridge_missing_index_warned = False
        self._bridge_missing_update_warned = False
        self._bridge_missing_grad_warned = False
        self._bridge_last_drift = {}
        self._bridge_history_score = {}
        self.ref_model = None
        if retain_loss_type == "KL" or self.bridge_enabled:
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def _model_inputs(self, inputs):
        return {
            key: inputs[key]
            for key in ("input_ids", "attention_mask", "labels")
            if key in inputs
        }

    def compute_retain_loss(self, model, retain_inputs):
        retain_inputs = self._model_inputs(retain_inputs)
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def _zero_bridge_loss(self, model):
        device = next(model.parameters()).device
        return torch.zeros((), device=device)

    def _bridge_indices(self, retain_inputs, batch_size):
        if "index" not in retain_inputs:
            if self.bridge_prior == "history" and not self._bridge_missing_index_warned:
                logger.warning(
                    "BRIDGE History-DRO requested but retain batch has no index; "
                    "falling back to uniform prior. Use collator=DataCollatorForSupervisedDatasetwithIndex."
                )
                self._bridge_missing_index_warned = True
            return None
        indices = retain_inputs["index"].detach().cpu().view(-1).tolist()
        if len(indices) != batch_size:
            logger.warning(
                "BRIDGE retain index length %s does not match batch size %s; falling back to uniform prior.",
                len(indices),
                batch_size,
            )
            return None
        return [int(index) for index in indices]

    def _scores_to_prior(self, scores):
        if scores.numel() == 1:
            return torch.ones_like(scores)
        scores = scores.detach()
        scores = scores - scores.mean()
        score_std = scores.std(unbiased=False)
        if torch.isfinite(score_std).item() and score_std.item() > 1e-6:
            scores = scores / score_std
        tau_p = max(self.bridge_tau_p, 1e-6)
        return F.softmax(scores / tau_p, dim=0)

    def _bridge_trainable_params(self, model):
        return [param for param in model.parameters() if param.requires_grad]

    def _bridge_step_size(self):
        if self.bridge_virtual_step_size is not None:
            return self.bridge_virtual_step_size
        learning_rate = getattr(self.args, "learning_rate", None)
        return 1e-5 if learning_rate is None else float(learning_rate)

    def _bridge_update_direction(self, model, update_loss):
        if update_loss is None or not getattr(update_loss, "requires_grad", False):
            if not self._bridge_missing_update_warned:
                logger.warning(
                    "BRIDGE %s prior needs an update loss; falling back to uniform prior.",
                    self.bridge_prior,
                )
                self._bridge_missing_update_warned = True
            return None, None

        params = self._bridge_trainable_params(model)
        if not params:
            if not self._bridge_missing_grad_warned:
                logger.warning(
                    "BRIDGE %s prior found no trainable parameters; falling back to uniform prior.",
                    self.bridge_prior,
                )
                self._bridge_missing_grad_warned = True
            return None, None

        grads = torch.autograd.grad(
            update_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        step_size = self._bridge_step_size()
        update_direction = [
            None if grad is None else (-step_size * grad.detach())
            for grad in grads
        ]
        if all(direction is None for direction in update_direction):
            if not self._bridge_missing_grad_warned:
                logger.warning(
                    "BRIDGE %s prior could not compute update gradients; falling back to uniform prior.",
                    self.bridge_prior,
                )
                self._bridge_missing_grad_warned = True
            return None, None
        return params, update_direction

    def _bridge_directional_score(self, loss, params, update_direction):
        if not getattr(loss, "requires_grad", False):
            return loss.new_zeros(())
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        score = loss.new_zeros(())
        for grad, direction in zip(grads, update_direction):
            if grad is not None and direction is not None:
                score = score + (grad * direction).sum()
        return score.detach()

    def _bridge_gs_scores(self, model, retain_inputs, params, update_direction):
        per_sample_nll, _ = compute_per_sample_nll(
            model, self._model_inputs(retain_inputs)
        )
        scores = [
            self._bridge_directional_score(sample_loss, params, update_direction)
            for sample_loss in per_sample_nll.unbind(dim=0)
        ]
        return torch.stack(scores)

    def _bridge_kl_pi_scores(self, per_sample_kl, params, update_direction):
        scores = [
            self._bridge_directional_score(sample_kl, params, update_direction)
            for sample_kl in per_sample_kl.unbind(dim=0)
        ]
        return torch.stack(scores)

    def _bridge_prior_distribution(
        self, model, retain_inputs, per_sample_kl, update_loss=None
    ):
        batch_size = per_sample_kl.shape[0]
        if self.bridge_prior in {"none", "uniform", "global_kl", "global-kl"}:
            return torch.full_like(per_sample_kl, 1.0 / batch_size)

        if self.bridge_prior == "history":
            indices = self._bridge_indices(retain_inputs, batch_size)
            if indices is None:
                return torch.full_like(per_sample_kl, 1.0 / batch_size)

            current_drifts = per_sample_kl.detach().float().cpu().tolist()
            scores = []
            for index, current_drift in zip(indices, current_drifts):
                previous_drift = self._bridge_last_drift.get(index)
                drift_delta = (
                    0.0 if previous_drift is None else current_drift - previous_drift
                )
                previous_score = self._bridge_history_score.get(index, 0.0)
                score = (
                    self.bridge_history_beta * previous_score
                    + (1.0 - self.bridge_history_beta) * drift_delta
                )
                self._bridge_last_drift[index] = current_drift
                self._bridge_history_score[index] = score
                scores.append(score)

            scores = torch.tensor(
                scores, dtype=per_sample_kl.dtype, device=per_sample_kl.device
            )
            return self._scores_to_prior(scores)

        if self.bridge_prior in {"gs", "gradient", "gradient_similarity"}:
            params, update_direction = self._bridge_update_direction(
                model, update_loss
            )
            if params is None:
                return torch.full_like(per_sample_kl, 1.0 / batch_size)
            scores = self._bridge_gs_scores(
                model, retain_inputs, params, update_direction
            )
            return self._scores_to_prior(scores)

        if self.bridge_prior in {"kl_pi", "kl-pi", "pi"}:
            params, update_direction = self._bridge_update_direction(
                model, update_loss
            )
            if params is None:
                return torch.full_like(per_sample_kl, 1.0 / batch_size)
            scores = self._bridge_kl_pi_scores(per_sample_kl, params, update_direction)
            return self._scores_to_prior(scores)

        raise NotImplementedError(f"Unsupported bridge_prior={self.bridge_prior}")

    def _bridge_boundary_dro(self, per_sample_kl, prior):
        tau = max(self.bridge_tau, 1e-6)
        log_prior = torch.log(prior.detach().clamp_min(1e-12))
        return tau * torch.logsumexp(log_prior + per_sample_kl / tau, dim=0)

    def _maybe_log_bridge(self, bridge_loss, global_kl, boundary_dro, prior):
        if self.bridge_log_every <= 0:
            return
        step = int(getattr(self.state, "global_step", 0))
        if step == self._bridge_last_logged_step or step % self.bridge_log_every != 0:
            return
        self._bridge_last_logged_step = step
        with torch.no_grad():
            log_prior = torch.log(prior.detach().clamp_min(1e-12))
            entropy = -(prior.detach() * log_prior).sum()
            self.log(
                {
                    "bridge_loss": bridge_loss.detach().float().item(),
                    "bridge_global_kl": global_kl.detach().float().item(),
                    "bridge_boundary_dro": boundary_dro.detach().float().item(),
                    "bridge_prior_entropy": entropy.detach().float().item(),
                    "bridge_prior_max": prior.detach().max().float().item(),
                }
            )

    def compute_bridge_loss(self, model, retain_inputs, update_loss=None):
        if not self.bridge_enabled or retain_inputs is None:
            return self._zero_bridge_loss(model)

        retain_model_inputs = self._model_inputs(retain_inputs)
        per_sample_kl, _ = compute_per_sample_kl_divergence(
            model, self.ref_model, retain_model_inputs
        )
        global_kl = per_sample_kl.mean()
        boundary_dro = per_sample_kl.new_zeros(())
        bridge_loss = per_sample_kl.new_zeros(())

        if self.bridge_lambda_g != 0.0:
            bridge_loss = bridge_loss + self.bridge_lambda_g * global_kl
        if self.bridge_lambda_b != 0.0:
            prior = self._bridge_prior_distribution(
                model, retain_inputs, per_sample_kl, update_loss=update_loss
            )
            boundary_dro = self._bridge_boundary_dro(per_sample_kl, prior)
            bridge_loss = bridge_loss + self.bridge_lambda_b * boundary_dro
        else:
            prior = torch.full_like(per_sample_kl, 1.0 / per_sample_kl.shape[0])

        self._maybe_log_bridge(bridge_loss, global_kl, boundary_dro, prior)
        return bridge_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        bridge_loss = self.compute_bridge_loss(
            model=model, retain_inputs=retain_inputs, update_loss=forget_loss
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss + bridge_loss

        return (loss, forget_outputs) if return_outputs else loss
