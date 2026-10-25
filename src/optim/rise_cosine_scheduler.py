import math
from torch.optim.lr_scheduler import _LRScheduler


class RiseCosineScheduler(_LRScheduler):
    """
    Linear rise → cosine decay scheduler
    matching trainer.lr_step() logic.
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        steps_per_epoch: int,
        total_steps: int,
        rise: float | None = None,
        decay: float | None = None,
        last_epoch: int = -1,
        **kwargs,
    ):
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.rise = rise
        self.decay = decay
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs = total_steps // steps_per_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    # --------------------------------------------------
    # Core LR computation
    # --------------------------------------------------
    def get_lr(self):

        step = self.last_epoch  # global step
        scale = 1.0

        # ----------------------------
        # Linear rise
        # ----------------------------
        if self.rise is not None:
            rise_until_step = self.rise * self.steps_per_epoch
            if step < rise_until_step:
                scale = step / rise_until_step

        # ----------------------------
        # Cosine decay
        # ----------------------------
        if self.decay is not None:

            decay_start_epoch = self.decay

            if isinstance(decay_start_epoch, float):
                decay_start_epoch *= self.num_epochs

            decay_start_step = decay_start_epoch * self.steps_per_epoch

            if step >= decay_start_step:

                decay_step = step - decay_start_step
                total_decay_steps = self.total_steps - decay_start_step

                scale *= (1 + math.cos(math.pi * decay_step / total_decay_steps)) * 0.5

        return [base_lr * scale for base_lr in self.base_lrs]
