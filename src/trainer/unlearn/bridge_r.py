from __future__ import annotations

import torch

from trainer.unlearn.npo import NPO
from trainer.utils import compute_dpo_loss, compute_per_sample_kl_divergence


def grouped_mean(values: torch.Tensor, owners: torch.Tensor | None) -> torch.Tensor:
    if owners is None:
        return values.mean()
    groups = []
    for owner in torch.unique(owners, sorted=True):
        groups.append(values[owners == owner].mean())
    return torch.stack(groups).mean()


def centered_uniform_tail(values: torch.Tensor, center: torch.Tensor, tau: float) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("centered tail needs a non-empty per-sample vector")
    tau = max(float(tau), 1e-6)
    centered = values - center.detach()
    return tau * (torch.logsumexp(centered / tau, dim=0) - torch.log(values.new_tensor(float(values.numel()))))


class NPOBridgeR(NPO):
    """v10 ladder: NPO erase/retain + global KL + local mean + centered uniform tail."""

    def __init__(self, bridge_r_lambda_g=0.0, bridge_r_lambda_l=0.0, bridge_r_lambda_t=0.0, bridge_r_tau=1.0, bridge_r_kl_chunk_size=8, *args, **kwargs):
        # Disable the legacy BRIDGE path; BRIDGE-R has distinct request-local semantics.
        kwargs.update(bridge_lambda_g=0.0, bridge_lambda_b=0.0, bridge_prior="none")
        super().__init__(*args, **kwargs)
        self.bridge_r_lambda_g = float(bridge_r_lambda_g)
        self.bridge_r_lambda_l = float(bridge_r_lambda_l)
        self.bridge_r_lambda_t = float(bridge_r_lambda_t)
        self.bridge_r_tau = float(bridge_r_tau)
        self.bridge_r_kl_chunk_size = int(bridge_r_kl_chunk_size)

    def _chunked_local_kl(self, model, protected_inputs):
        model_inputs = self._model_inputs(protected_inputs)
        batch_size = model_inputs["input_ids"].shape[0]
        chunk_size = batch_size if self.bridge_r_kl_chunk_size <= 0 else self.bridge_r_kl_chunk_size
        values = []
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            chunk = {key: value[start:stop] for key, value in model_inputs.items()}
            chunk_values, _ = compute_per_sample_kl_divergence(model, self.ref_model, chunk)
            values.append(chunk_values)
        return torch.cat(values, dim=0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "protected" not in inputs and "forget" in inputs and isinstance(inputs["forget"], dict) and "protected" in inputs["forget"]:
            inputs = inputs["forget"]
        forget_inputs = self._model_inputs(inputs["forget"])
        forget_loss, forget_outputs = compute_dpo_loss(model=model, ref_model=self.ref_model, win_inputs=None, lose_inputs=forget_inputs, beta=self.beta)
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=inputs["retain"])
        total = self.gamma * forget_loss + self.alpha * retain_loss

        global_kl = total.new_zeros(())
        local_mean = total.new_zeros(())
        tail = total.new_zeros(())
        if self.bridge_r_lambda_g:
            global_values, _ = compute_per_sample_kl_divergence(model, self.ref_model, self._model_inputs(inputs["retain"]))
            global_kl = global_values.mean()
            total = total + self.bridge_r_lambda_g * global_kl
        if self.bridge_r_lambda_l or self.bridge_r_lambda_t:
            if "protected" not in inputs:
                raise ValueError("BRIDGE-R LocalMean/Tail requires a request-local protected batch")
            local_values = self._chunked_local_kl(model, inputs["protected"])
            owners = inputs.get("protected_owner")
            local_mean = grouped_mean(local_values, owners)
            if self.bridge_r_lambda_l:
                total = total + self.bridge_r_lambda_l * local_mean
            if self.bridge_r_lambda_t:
                tail = centered_uniform_tail(local_values, local_mean, self.bridge_r_tau)
                total = total + self.bridge_r_lambda_t * tail
        if int(getattr(self.state, "global_step", 0)) % 5 == 0:
            self.log({
                "bridge_r_global_kl": float(global_kl.detach()),
                "bridge_r_local_mean": float(local_mean.detach()),
                "bridge_r_centered_tail": float(tail.detach()),
                "bridge_r_logical_k": float(inputs.get("logical_k", torch.tensor([0])).float().mean()),
                "bridge_r_logical_k_requested": float(inputs.get("logical_k_requested", torch.tensor([0])).float().mean()),
                "bridge_r_protected_duplicate_rate": float(inputs.get("protected_duplicate_rate", torch.tensor([0.0])).float().mean()),
                "bridge_r_microbatch": float(inputs["forget"]["input_ids"].shape[0]),
                "bridge_r_kl_chunk_size": float(self.bridge_r_kl_chunk_size),
            })
        return (total, forget_outputs) if return_outputs else total
