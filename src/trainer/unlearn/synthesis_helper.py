from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from trainer.components import (
    GradientSynthesisStats,
    combine_gradients,
    get_named_trainable_params,
    grad_dict_from_autograd,
)


@dataclass
class RetentionSynthesisConfig:
    use_retain_loss: bool = False
    gradient_synthesis: str = "none"   # none / pcgrad / sago
    alpha: float = 1.0
    gamma: float = 1.0


def _safe_get(obj: Any, key: str, default: Any):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj.get(key, default)
    except Exception:
        return getattr(obj, key, default)


def build_retention_synthesis_config(method_args: Any) -> RetentionSynthesisConfig:
    return RetentionSynthesisConfig(
        use_retain_loss=bool(_safe_get(method_args, "use_retain_loss", False)),
        gradient_synthesis=str(_safe_get(method_args, "gradient_synthesis", "none")).lower(),
        alpha=float(_safe_get(method_args, "alpha", 1.0)),
        gamma=float(_safe_get(method_args, "gamma", 1.0)),
    )


def _accumulate_grad_dict_to_model(
    model: torch.nn.Module,
    grad_dict,
    grad_scale: float = 1.0,
) -> None:
    named_params = get_named_trainable_params(model)
    for name, param in named_params.items():
        grad = grad_dict.get(name, None)
        if grad is None:
            continue
        grad = grad.detach() * grad_scale
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad.add_(grad)


def apply_forget_only_gradients(
    model: torch.nn.Module,
    forget_loss: torch.Tensor,
    gamma: float = 1.0,
    grad_scale: float = 1.0,
    retain_graph: bool = False,
) -> GradientSynthesisStats:
    forget_grads = grad_dict_from_autograd(
        forget_loss,
        model,
        retain_graph=retain_graph,
        create_graph=False,
    )

    final_grads = {}
    final_norm_sq = 0.0
    forget_norm_sq = 0.0

    for name, grad in forget_grads.items():
        if grad is None:
            final_grads[name] = None
            continue
        g = gamma * grad
        final_grads[name] = g
        forget_norm_sq += float(torch.sum(grad * grad).item())
        final_norm_sq += float(torch.sum(g * g).item())

    _accumulate_grad_dict_to_model(model, final_grads, grad_scale=grad_scale)

    return GradientSynthesisStats(
        mode="forget_only",
        num_tensors=len(final_grads),
        num_conflict_tensors=0,
        retain_norm_sq=0.0,
        forget_norm_sq=forget_norm_sq,
        final_norm_sq=final_norm_sq,
    )


def apply_synthesized_gradients(
    model: torch.nn.Module,
    forget_loss: torch.Tensor,
    retain_loss: Optional[torch.Tensor],
    method_args: Any,
    grad_scale: float = 1.0,
    retain_graph: bool = False,
) -> GradientSynthesisStats:
    cfg = build_retention_synthesis_config(method_args)

    if (not cfg.use_retain_loss) or (retain_loss is None):
        return apply_forget_only_gradients(
            model=model,
            forget_loss=forget_loss,
            gamma=cfg.gamma,
            grad_scale=grad_scale,
            retain_graph=retain_graph,
        )

    forget_grads = grad_dict_from_autograd(
        forget_loss,
        model,
        retain_graph=True,
        create_graph=False,
    )

    retain_grads = grad_dict_from_autograd(
        retain_loss,
        model,
        retain_graph=retain_graph,
        create_graph=False,
    )

    final_grads, stats = combine_gradients(
        forget_grads=forget_grads,
        retain_grads=retain_grads,
        mode=cfg.gradient_synthesis,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
    )

    _accumulate_grad_dict_to_model(model, final_grads, grad_scale=grad_scale)
    return stats