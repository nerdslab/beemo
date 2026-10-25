import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from temporaldata import Data
from torch import nn
from torch_brain.data import chain, pad
from torch_brain.registry import ModalitySpec
from torch_brain.utils import prepare_for_readout

# from torch_brain.utils.binning import bin_spikes
from src.utils import bin_spikes


class GRU(nn.Module):
    def __init__(
        self,
        *,
        readout_spec: ModalitySpec,
        out_step: float,
        bin_size: float = 0.02,
        depth: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert depth >= 1, f"depth must be >= 1, got {depth}"

        self.out_step = out_step
        self.bin_size = bin_size
        self.modality_spec = readout_spec
        self.ctx_duration = kwargs.get("ctx_duration", 1.0)
        self.num_bins = round(self.ctx_duration / self.bin_size)
        self.input_proj = nn.LazyLinear(hidden_dim)
        self._num_units = None

        gru_dropout = dropout if depth > 1 else 0.0
        self.backbone = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=depth,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.readout = nn.Linear(out_dim, readout_spec.dim)

    def tokenize(self, data: Data) -> dict:
        data = data.tgt_data
        num_units = len(data.units)
        binned_spikes = bin_spikes(
            data.spikes, num_units, self.bin_size, dtype=np.float32, domain=data.domain
        )
        assert binned_spikes.shape[0] == self.num_bins

        if self._num_units is None:
            self._num_units = num_units
        elif num_units != self._num_units:
            raise ValueError(
                f"GRU needs a constant unit count (units are the feature axis); "
                f"saw {num_units} after {self._num_units}."
            )

        behav_times, behav_gt, behav_weights, eval_mask = prepare_for_readout(
            data, self.modality_spec
        )

        # fractional bin position; forward interpolates rather than snapping,
        # giving continuous output resolution at no parameter cost
        t0 = float(data.domain.start[0])
        bin_pos = (behav_times - t0) / self.bin_size - 0.5
        bin_pos = np.clip(bin_pos, 0.0, self.num_bins - 1).astype(np.float32)

        return {
            "model_inputs": {
                "spikes": pad(binned_spikes.astype(np.float32)),
                "output_timestamps": chain(behav_times.astype(np.float32)),
                "output_seqlen": len(behav_times),
                "output_bin_pos": chain(bin_pos),
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
        output_bin_pos: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        x = self.input_proj(spikes)  # (B, T, N) -> (B, T, D)
        x, _ = self.backbone(x)
        x = self.readout(x)  # (B, T, behav_dim)

        # repeat_interleave reproduces the chain-collated target ordering
        batch_idx = torch.repeat_interleave(
            torch.arange(x.shape[0], device=x.device), output_seqlen
        )

        # interpolate between neighbouring bin centres; the readout is linear,
        # so doing it after is identical to, and cheaper than, before
        i0 = output_bin_pos.floor().long().clamp(0, self.num_bins - 1)
        i1 = (i0 + 1).clamp(max=self.num_bins - 1)
        w = (output_bin_pos - i0.to(output_bin_pos.dtype)).unsqueeze(-1)
        return x[batch_idx, i0] * (1 - w) + x[batch_idx, i1] * w

    @classmethod
    def create_search_space(cls, trial: optuna.Trial, cfg: DictConfig):
        # model
        trial.suggest_categorical("model.args.bin_size", [0.005, 0.01, 0.02, 0.04])
        trial.suggest_int("model.args.depth", 1, 3, step=1)
        trial.suggest_int("hidden_dim_log2", 5, 9, step=1)
        trial.suggest_float("model.args.dropout", 0.0, 0.6, step=0.2)
        trial.suggest_categorical("model.args.bidirectional", [True, False])

        # training
        trial.suggest_int("num_epochs", 100, 500, step=200)
        trial.suggest_int("batch_size_log2", 4, 7, step=1)
        trial.suggest_float("optimizer.args.weight_decay", 1e-6, 1e-1, log=True)

        # lr scheduler
        trial.suggest_float("optimizer.args.lr", 1e-5, 1e-2, log=True)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        hidden_dim_log2 = tune_params.pop("hidden_dim_log2")
        tune_params["model.args.hidden_dim"] = 2**hidden_dim_log2

        batch_size_log2 = tune_params.pop("batch_size_log2")
        tune_params["batch_size"] = 2**batch_size_log2

        return tune_params
