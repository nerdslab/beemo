from typing import List
import torch
from torch.optim import Optimizer


class MergedOptimizer(Optimizer):
    def __init__(
        self,
        optimizers: List[torch.optim.Optimizer] | torch.optim.Optimizer,
    ):
        if isinstance(optimizers, torch.optim.Optimizer):
            self.optimizers = [optimizers]
        elif isinstance(optimizers, list):
            self.optimizers = optimizers
        else:
            raise ValueError(f"Invalid optimizers: {optimizers}")

        self.defaults = self.optimizers[0].defaults

    def step(self, closure=None):
        loss = None

        for opt in self.optimizers:
            if closure is not None:
                loss = opt.step(closure)
            else:
                opt.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        assert len(state_dicts) == len(
            self.optimizers
        ), "Mismatch in optimizer state dict count"
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def __repr__(self):
        return f"MergedOptimizer({self.optimizers})"
