# --------------------------------------------------
# Code adapted from torch_brain v0.1 (Apache-2.0)
# --------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torch_brain.data import Data, Interval
from torch_brain.utils.weights import isin_interval


def prepare_readout(data: Data):
    r"""Behavior targets for a window, as set by the dataset in ``data.config["readout"]``:

    - timestamp_key, value_key: fields to read (default ``cursor.timestamps``, ``cursor.vel``)
    - normalize_mean, normalize_std: optional z-scoring, scalar or per-channel
    - weights: interval name -> loss weight; overlapping weights multiply
    - eval_interval: interval the evaluation mask is restricted to

    Returns timestamps, values, weights and eval_mask.
    """
    config = data.config["readout"]

    timestamps = data.get_nested_attribute(
        config.get("timestamp_key", "cursor.timestamps")
    )
    values = data.get_nested_attribute(config.get("value_key", "cursor.vel"))

    if "normalize_mean" in config:
        mean = config["normalize_mean"]
        values = values - (np.array(mean) if isinstance(mean, list) else mean)
    if "normalize_std" in config:
        std = config["normalize_std"]
        values = values / (np.array(std) if isinstance(std, list) else std)

    # we won't be running a model at float64 precision
    if values.dtype == np.float64:
        values = values.astype(np.float32)

    weights = np.ones_like(timestamps, dtype=np.float32)
    for interval_key, weight in config.get("weights", {}).items():
        interval = data.get_nested_attribute(interval_key)
        if not isinstance(interval, Interval):
            raise ValueError(
                f"Weight {interval_key} is of type {type(interval)}, expected an Interval"
            )
        weights[isin_interval(timestamps, interval)] *= weight

    eval_mask = np.ones(len(timestamps), dtype=np.bool_)
    if "eval_interval" in config:
        eval_mask = isin_interval(
            timestamps, data.get_nested_attribute(config["eval_interval"])
        )

    return timestamps, values, weights, eval_mask


def weighted_mse_loss(
    input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    loss = F.mse_loss(input, target, reduction="none").mean(dim=1)
    return (weights * loss).sum() / weights.sum()
