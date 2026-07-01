from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from trainer.unlearn.synthesis_helper import (
    apply_synthesized_gradients,
    build_retention_synthesis_config,
)


class RetentionPrioritizedMixin(ABC):
    @abstractmethod
    def _split_unlearn_inputs(self, inputs: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def _compute_forget_loss(
        self,
        model: torch.nn.Module,
        forget_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _compute_retain_loss(
        self,
        model: torch.nn.Module,
        retain_inputs: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        cfg = build_retention_synthesis_config(self.method_args)
        if cfg.gradient_synthesis == "none":
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()

        if hasattr(self, "_prepare_inputs"):
            inputs = self._prepare_inputs(inputs)

        forget_inputs, retain_inputs = self._split_unlearn_inputs(inputs)

        forget_loss = self._compute_forget_loss(model, forget_inputs)
        retain_loss = self._compute_retain_loss(model, retain_inputs)

        gas = 1
        if getattr(self, "args", None) is not None:
            gas = max(int(getattr(self.args, "gradient_accumulation_steps", 1)), 1)

        stats = apply_synthesized_gradients(
            model=model,
            forget_loss=forget_loss,
            retain_loss=retain_loss,
            method_args=self.method_args,
            grad_scale=1.0 / gas,
            retain_graph=False,
        )

        if hasattr(self, "log"):
            try:
                self.log(
                    {
                        "synth_retain_norm_sq": stats.retain_norm_sq,
                        "synth_forget_norm_sq": stats.forget_norm_sq,
                        "synth_final_norm_sq": stats.final_norm_sq,
                        "synth_conflict_tensors": stats.num_conflict_tensors,
                    }
                )
            except Exception:
                pass

        total_for_logging = cfg.gamma * forget_loss.detach()
        if cfg.use_retain_loss and retain_loss is not None:
            total_for_logging = total_for_logging + cfg.alpha * retain_loss.detach()

        total_for_logging = total_for_logging / gas
        return total_for_logging
