import copy

from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.retention_prioritized import RetentionPrioritizedMixin


class GradDiff(RetentionPrioritizedMixin, UnlearnTrainer):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        use_retain_loss=True,
        gradient_synthesis="none",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type

        self.use_retain_loss = use_retain_loss
        self.gradient_synthesis = gradient_synthesis.lower()

        # 给 synthesis_helper 统一读取
        self.method_args = {
            "use_retain_loss": self.use_retain_loss,
            "gradient_synthesis": self.gradient_synthesis,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
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

    def _split_unlearn_inputs(self, inputs):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        retain_inputs = inputs.get("retain", None)
        if retain_inputs is not None:
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }

        return forget_inputs, retain_inputs

    def _compute_forget_loss(self, model, forget_inputs):
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss
        return forget_loss

    def _compute_retain_loss(self, model, retain_inputs):
        if (not self.use_retain_loss) or (retain_inputs is None):
            return None
        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs, retain_inputs = self._split_unlearn_inputs(inputs)

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_loss = 0.0
        if self.use_retain_loss and retain_inputs is not None:
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss