from torch import nn
from src.logger import Logger


class GradualUnfreezing:
    def __init__(self, unfreeze_at_epoch: int, model: nn.Module):
        self.enabled = unfreeze_at_epoch != 0
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.frozen_params = None
        if self.enabled:
            self.frozen_params = self.freeze(model)

    def freeze(self, model: nn.Module):
        """
        Freeze all parameters of the model except for those in unit_emb and session_emb.
        Returns list of parameters that were actually frozen.
        """
        frozen_params = []
        for name, param in model.named_parameters():
            # Only skip unit/session embedding layers
            if not (
                name.startswith("unit_emb.") or name.startswith("session_emb.")
            ) or name.startswith("task_emb."):
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params.append(param)
        return frozen_params

    def unfreeze(self, model: nn.Module, epoch: int):
        """
        Unfreeze all parameters of the model except for those in unit_emb and session_emb.
        Returns list of parameters that were actually unfrozen.
        """
        if self.enabled and (epoch == self.unfreeze_at_epoch):
            if self.frozen_params is None:
                raise RuntimeError("Model has not been frozen yet.")

            for param in self.frozen_params:
                param.requires_grad = True

            self.frozen_params = None
