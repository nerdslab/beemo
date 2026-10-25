import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from temporaldata import Data
from torch import nn
from torch_brain.data import chain, pad
from torch_brain.registry import ModalitySpec
from torch_brain.utils import prepare_for_readout

from src.utils import bin_spikes


class Linear(nn.Module):
    def __init__(
        self,
        *,
        readout_spec: ModalitySpec,
        out_step: float,
        bin_size: float = 0.02,
        out_stride: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.out_step = out_step
        self.out_stride = int(out_stride)
        self.bin_size = bin_size
        self.modality_spec = readout_spec
        self.ctx_duration = kwargs.get("ctx_duration", 1.0)
        self.num_bins = round(self.ctx_duration / self.bin_size)
        self.num_out = min(
            round(self.ctx_duration / (self.out_step * self.out_stride)),
            self.num_bins,
        )
        self.slot_size = self.ctx_duration / self.num_out
        self.linear = nn.LazyLinear(self.num_out * readout_spec.dim)

    def tokenize(self, data: Data) -> dict:
        data = data.tgt_data
        num_units = len(data.units)
        binned_spikes = bin_spikes(
            data.spikes, num_units, self.bin_size, dtype=np.float32, domain=data.domain
        )  # (T, N)
        assert binned_spikes.shape[0] == self.num_bins
        binned_spikes = binned_spikes.T.flatten()  # (N*T,)

        behav_times, behav_gt, behav_weights, eval_mask = prepare_for_readout(
            data, self.modality_spec
        )

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
        x = self.linear(spikes)  # (B, N*T) -> (B, num_out*behav_dim)
        x = x.view(-1, self.num_out, self.modality_spec.dim)

        # repeat_interleave reproduces the chain-collated target ordering
        batch_idx = torch.repeat_interleave(
            torch.arange(x.shape[0], device=x.device), output_seqlen
        )
        return x[batch_idx, output_slot_idx]  # (sum(output_seqlen), behav_dim)

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        # floored at 20ms: finer bins suit mlp/gru, not this.
        trial.suggest_categorical("model.args.bin_size", [0.02, 0.04])
        # decimates the output grid; divides the readout cost by the same factor
        trial.suggest_categorical("model.args.out_stride", [1, 2, 4, 8])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 7, step=1)
        trial.suggest_float("optimizer.args.weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("optimizer.args.lr", 1e-5, 1e-2, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        batch_size_log2 = tune_params.pop("batch_size_log2")
        tune_params["batch_size"] = 2**batch_size_log2
        return tune_params
