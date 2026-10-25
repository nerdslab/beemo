# --------------------------------------------------
# Code adapted from ibl-bwb
# Authors: Alexander Andre, Shivasriganesh Mahato
# --------------------------------------------------

from typing import Literal

import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from torch_brain.data import Data
from torch import nn
from torch_brain.batching import chain, pad
from src.readout import prepare_readout

from src.utils import bin_spikes

Activation = Literal["relu", "gelu", "tanh"]


class MLP(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int = 2,
        out_step: float,
        ctx_duration: float = 1.0,
        bin_size: float = 0.02,
        out_stride: int = 1,
        depth: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        activation: Activation = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        self.out_step = out_step
        self.out_stride = int(out_stride)
        self.ctx_duration = ctx_duration
        self.bin_size = bin_size
        self.output_dim = output_dim

        model_layers = []
        in_dim = -1
        dim = hidden_dim
        for _ in range(depth):
            if in_dim == -1:
                model_layers.append(nn.LazyLinear(dim))
            else:
                model_layers.append(nn.Linear(in_dim, dim))
            if batch_norm:
                model_layers.append(nn.BatchNorm1d(dim))
            model_layers.append(self._get_activation(activation))
            if dropout > 0.0:
                model_layers.append(nn.Dropout(dropout))
            in_dim = dim
            dim = dim // 2
        self.net = nn.Sequential(*model_layers)
        self.final_hidden_dim = in_dim
        self.num_bins = round(self.ctx_duration / self.bin_size)
        self.num_out = min(
            round(self.ctx_duration / (self.out_step * self.out_stride)),
            self.num_bins,
        )
        self.slot_size = self.ctx_duration / self.num_out
        self.readout = nn.Linear(self.final_hidden_dim, self.num_out * output_dim)

    def tokenize(self, data: Data) -> dict:
        data = data.tgt_data
        num_units = len(data.units)
        binned_spikes = bin_spikes(
            data.spikes, num_units, self.bin_size, dtype=np.float32, domain=data.domain
        )  # (T, N)
        assert binned_spikes.shape[0] == self.num_bins

        binned_spikes = binned_spikes.T.flatten()  # (N*T,)

        behav_times, behav_gt, behav_weights, eval_mask = prepare_readout(data)

        # each behavior timestamp -> its slot on the decimated output grid
        t0 = float(data.domain.start[0])
        slot_idx = np.floor((behav_times - t0) / self.slot_size).astype(np.int64)
        slot_idx = np.clip(slot_idx, 0, self.num_out - 1)

        return {
            "model_inputs": {
                "spikes": pad(binned_spikes.astype(np.float32)),
                "output_timestamps": chain(behav_times.astype(np.float32)),
                "output_seqlen": len(behav_times),
                "output_slot_idx": chain(slot_idx),
            },
            "target_values": chain(behav_gt.astype(np.float32)),
            "target_weights": chain(behav_weights.astype(np.float32)),
            "eval_mask": chain(eval_mask),
            "absolute_start": data._absolute_start,
            "session_id": data.session.id,
        }

    def forward(
        self,
        spikes: torch.Tensor,
        output_seqlen: torch.Tensor,
        output_slot_idx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x = self.net(spikes)  # (B, N*T) -> (B, hidden)
        x = self.readout(x)  # (B, hidden) -> (B, num_out*behav_dim)
        x = x.view(-1, self.num_out, self.output_dim)

        # repeat_interleave reproduces the chain-collated target ordering
        batch_idx = torch.repeat_interleave(
            torch.arange(x.shape[0], device=x.device), output_seqlen
        )
        return x[batch_idx, output_slot_idx]  # (sum(output_seqlen), behav_dim)

    def _get_activation(self, activation: Activation) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_categorical("model.args.bin_size", [0.005, 0.01, 0.02, 0.04])
        trial.suggest_categorical("model.args.out_stride", [1, 2, 4, 8])
        trial.suggest_int("model.args.depth", 1, 3, step=1)
        trial.suggest_int("hidden_dim_log2", 5, 9, step=1)
        trial.suggest_float("model.args.dropout", 0.0, 0.6, step=0.2)
        trial.suggest_categorical("model.args.batch_norm", [True, False])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 7, step=1)
        trial.suggest_float("optimizer.args.weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("optimizer.args.lr", 1e-5, 1e-2, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        hidden_dim_log2 = tune_params.pop("hidden_dim_log2")
        hidden_dim = 2**hidden_dim_log2
        tune_params["model.args.hidden_dim"] = hidden_dim

        batch_size_log2 = tune_params.pop("batch_size_log2")
        batch_size = 2**batch_size_log2
        tune_params["batch_size"] = batch_size

        return tune_params
