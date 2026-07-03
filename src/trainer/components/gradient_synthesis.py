from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


GradDict = Dict[str, Optional[torch.Tensor]]


@dataclass
class GradientSynthesisStats:
    mode: str
    num_tensors: int
    num_conflict_tensors: int
    retain_norm_sq: float
    forget_norm_sq: float
    final_norm_sq: float


def _clone_grad(g: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if g is None:
        return None
    return g.detach().clone()


def clone_grad_dict(grad_dict: GradDict) -> GradDict:
    return {k: _clone_grad(v) for k, v in grad_dict.items()}


def zeros_like_grad_dict(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> GradDict:
    out: GradDict = {}
    for name, param in named_params:
        if not param.requires_grad:
            continue
        out[name] = torch.zeros_like(param, memory_format=torch.preserve_format)
    return out


def get_named_trainable_params(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {
        name: param
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def grad_dict_from_autograd(
    loss: torch.Tensor,
    model: torch.nn.Module,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> GradDict:
    named_params = get_named_trainable_params(model)
    params = list(named_params.values())

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )

    out: GradDict = {}
    for (name, _), grad in zip(named_params.items(), grads):
        out[name] = None if grad is None else grad.detach().clone()
    return out


def assign_grad_dict_to_model(model: torch.nn.Module, grad_dict: GradDict) -> None:
    named_params = get_named_trainable_params(model)
    for name, param in named_params.items():
        grad = grad_dict.get(name, None)
        if grad is None:
            param.grad = None
        else:
            if param.grad is None:
                param.grad = grad.detach().clone()
            else:
                param.grad.copy_(grad.detach())


def _safe_dot(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> torch.Tensor:
    if a is None or b is None:
        device = None
        if a is not None:
            device = a.device
        elif b is not None:
            device = b.device
        return torch.tensor(0.0, device=device)
    return torch.sum(a * b)


def _safe_norm_sq(a: Optional[torch.Tensor]) -> torch.Tensor:
    if a is None:
        return torch.tensor(0.0)
    return torch.sum(a * a)


def _ensure_same_keys(gf: GradDict, gr: GradDict) -> None:
    if set(gf.keys()) != set(gr.keys()):
        raise ValueError(
            f"Gradient dict keys mismatch: "
            f"forget_only={set(gf.keys()) - set(gr.keys())}, "
            f"retain_only={set(gr.keys()) - set(gf.keys())}"
        )


def combine_gradients(
    forget_grads: GradDict,
    retain_grads: GradDict,
    mode: str = "none",
    alpha: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> tuple[GradDict, GradientSynthesisStats]:
    """
    Combine forget and retain gradients.

    Args:
        forget_grads: gradient dict from forget objective, g_f
        retain_grads: gradient dict from retain objective, g_r
        mode:
            - "none":   naive weighted sum alpha*g_r + gamma*g_f
            - "pcgrad": module-wise PCGrad as in the paper
            - "sago":   element-wise sign-aligned gating
        alpha: weight for retain branch
        gamma: weight for forget branch
        eps: numerical stabilizer

    Returns:
        final_grads, stats
    """
    mode = mode.lower()
    if mode not in {"none", "pcgrad", "sago"}:
        raise ValueError(f"Unsupported gradient synthesis mode: {mode}")

    _ensure_same_keys(forget_grads, retain_grads)

    final_grads: GradDict = {}
    num_conflict_tensors = 0

    retain_norm_sq_total = 0.0
    forget_norm_sq_total = 0.0
    final_norm_sq_total = 0.0

    for name in forget_grads.keys():
        gf = forget_grads[name]
        gr = retain_grads[name]

        if gf is not None:
            forget_norm_sq_total += float(_safe_norm_sq(gf).item())
        if gr is not None:
            retain_norm_sq_total += float(_safe_norm_sq(gr).item())

        # 缺一路梯度时，退化为可用分支
        if gf is None and gr is None:
            final = None
        elif mode == "none":
            if gf is None:
                final = alpha * gr
            elif gr is None:
                final = gamma * gf
            else:
                final = alpha * gr + gamma * gf

        elif mode == "pcgrad":
            if gf is None:
                final = alpha * gr
            elif gr is None:
                final = gamma * gf
            else:
                dot = _safe_dot(gf, gr)
                gr_norm_sq = _safe_norm_sq(gr)
                gf_proj = gf
                if float(dot.item()) < 0.0 and float(gr_norm_sq.item()) > eps:
                    gf_proj = gf - (dot / (gr_norm_sq + eps)) * gr
                    num_conflict_tensors += 1
                final = alpha * gr + gamma * gf_proj

        elif mode == "sago":
            if gf is None:
                final = alpha * gr
            elif gr is None:
                final = gamma * gf
            else:
                same_sign_mask = (gf * gr) >= 0
                conflict_mask = ~same_sign_mask

                gf_tilde = gf * same_sign_mask
                gr_tilde = gr * conflict_mask

                if bool(conflict_mask.any().item()):
                    num_conflict_tensors += 1

                final = alpha * gr_tilde + gamma * gf_tilde

        else:
            raise RuntimeError("Unreachable branch")

        final_grads[name] = None if final is None else final.detach().clone()

        if final is not None:
            final_norm_sq_total += float(_safe_norm_sq(final).item())

    stats = GradientSynthesisStats(
        mode=mode,
        num_tensors=len(forget_grads),
        num_conflict_tensors=num_conflict_tensors,
        retain_norm_sq=retain_norm_sq_total,
        forget_norm_sq=forget_norm_sq_total,
        final_norm_sq=final_norm_sq_total,
    )
    return final_grads, stats