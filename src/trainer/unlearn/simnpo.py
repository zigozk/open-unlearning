import torch.nn.functional as F

from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delta = delta
        self.beta = beta

    def _compute_forget_loss(self, model, forget_inputs):
        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100

        forget_loss, _ = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
        return forget_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs, retain_inputs = self._split_unlearn_inputs(inputs)

        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100

        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_loss = 0.0
        if self.use_retain_loss and retain_inputs is not None:
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss