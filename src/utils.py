# --------------------------------------------------
# Code adapted from Nuclr
# Author: Vinam Arora
# --------------------------------------------------

import os
import sys
from pathlib import Path
import copy
from typing import Any, Union
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
import torch.distributed as dist
from typing import List
from temporaldata import IrregularTimeSeries
import logging


class Precision:
    def __init__(self, precision_str: str):
        self.precision_str = precision_str

        if precision_str == "fp32":
            self.dtype = torch.float32
        elif precision_str == "fp16":
            self.dtype = torch.float16
        elif precision_str == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Invalid precision string: {precision_str}")

    def __repr__(self):
        return self.precision_str

    def __str__(self):
        return self.precision_str


def move_to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def seed_everything(seed):
    import random, torch, numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(func):
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)

    return wrapper


def expand_path(path: Union[str, Path]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def get_num_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())


def get_num_trainable_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def validate_weights(module: torch.nn.Module):
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module but got {type(module)}")

    nan_weights = []
    for k, v in module.named_parameters():
        if not v.isfinite().all():
            nan_weights.append(k)
    if len(nan_weights) != 0:
        raise RuntimeError(f"Found NaN weights in layers: {nan_weights}")


def get_cli_logger(all_ranks=False):

    # Custom filter to convert absolute paths to relative paths
    class RelativePathFilter(logging.Filter):
        def filter(self, record):
            if hasattr(record, "pathname"):
                record.relpath = os.path.relpath(record.pathname)
            return True

    LOGGING_FORMAT = (
        "\033[1;33m%(levelname)s\033[0m | "
        "\033[1;36m%(asctime)s\033[0m | "
        "\033[1;32m%(relpath)s:%(lineno)d\033[0m | "
        "%(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
        force=True,
        stream=sys.stdout,
    )
    logger = logging.getLogger()
    logger.addFilter(RelativePathFilter())
    if not all_ranks:
        if dist.is_initialized() and not dist.get_rank() == 0:
            logger.setLevel(logging.ERROR)
    return logger


def get_class_from_path(path: str):
    import importlib

    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def instantiate(dictconfig: DictConfig | dict, *args, **kwargs) -> Any:
    """
    Instantiate a class from a configuration dictionary.

    Args:
        cfg: Configuration with _target_ field specifying the class path
        **kwargs: Additional arguments to pass to the class constructor

    Returns:
        An instance of the specified class
    """
    if isinstance(dictconfig, DictConfig):
        cfg_dict = OmegaConf.to_container(dictconfig)
    else:
        cfg_dict = copy.deepcopy(dictconfig)

    target = cfg_dict.pop("_target_", None)
    assert target is not None
    cls = get_class_from_path(target)
    return cls(*args, **cfg_dict, **kwargs)


def is_divisible(x: float, y: float, tol: float = 1e-6) -> bool:
    """Returns true if x is divisible by y"""
    return abs(x / y - round(x / y)) < tol


def apply_unit_mask_(data, mask):
    num_units_before = len(data.units)
    data.units = data.units.select_by_mask(mask)
    num_units_after = len(data.units)

    old_unit_idx = np.arange(num_units_before)[mask]
    unit_idx_remap = np.zeros(num_units_before, dtype=int)
    unit_idx_remap[old_unit_idx] = np.arange(num_units_after)

    spike_mask = np.isin(data.spikes.unit_index, old_unit_idx)
    data.spikes = data.spikes.select_by_mask(spike_mask)
    data.spikes.unit_index = unit_idx_remap[data.spikes.unit_index]

    return data


def unpad_sequences(padded_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Remove padding from a padded tensor without sorting.

    Args:
        padded_tensor: Tensor of shape [batch_size, max_seq_len, ...]
        lengths: Actual sequence lengths [batch_size]

    Returns:
        Concatenated tensor of shape [sum(lengths), ...]
    """
    sequences = [padded_tensor[i, : lengths[i]] for i in range(padded_tensor.size(0))]
    return torch.cat(sequences, dim=0)


def unchain_sequences(
    chained_tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Reconstruct padded tensor from a concatenated (unchained) tensor.

    Args:
        chained_tensor: Concatenated tensor [sum(lengths), ...]
        lengths: Length of each sequence [batch_size]

    Returns:
        Padded tensor of shape [batch_size, max_seq_len, ...]
    """
    sequences = torch.split(chained_tensor, lengths.tolist(), dim=0)
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)


from typing import Optional

import numpy as np
from temporaldata import IrregularTimeSeries, Interval


def bin_spikes(
    spikes: IrregularTimeSeries,
    num_units: int,
    bin_size: float,
    max_spikes: Optional[int] = None,
    right: bool = True,
    eps: float = 1e-3,
    dtype: np.dtype = np.int32,
    domain: Optional[Interval] = None,
) -> np.ndarray:
    r"""Bins spikes into time bins of size `bin_size`. If the total time spanned by
    the spikes is not a multiple of `bin_size`, the spikes are truncated to the nearest
    multiple of `bin_size`. If `right` is True, the spikes are truncated from the left
    end of the time series, otherwise they are truncated from the right end.

    Notes:
    - The number of units cannot be inferred from a subset of spikes,
      so `num_units` must be provided explicitly.
    - Floating-point roundoff can cause `(end - start) / bin_size` to be
      very close to an integer without being exact (e.g. 9.99999999).
      The `eps` parameter is added before flooring to make the bin-count
      computation numerically robust.

    Args:
        spikes: IrregularTimeSeries object containing the spikes.
        num_units: Number of units in the population.
        bin_size: Size of the time bins in seconds.
        max_spikes: Maximum number of spikes to include per unit per
            bin. If ``None``, no clipping is applied.
        right: Decide which side gets truncated when duration is not
            a multiple of ``bin_size``. If ``True``, excess spikes are truncated from the left edge.
        eps: Small numerical margin used during bin assignment.
        dtype: Data type of the output binned array. (default ``np.int32``)

    Returns:
        Binned spike counts with shape ``(T, N)``, where ``T`` is the number of
        time bins and ``N`` is ``num_units``.
    """

    if domain is not None:
        start = domain.start[0]
        end = domain.end[-1]
    else:
        start = spikes.domain.start[0]
        end = spikes.domain.end[-1]

    # Compute how much time must be discarded so that the duration
    # is an exact multiple of `bin_size`. The epsilon stabilizes
    # the floor operation under floating-point roundoff.
    discard = (end - start) - np.floor(((end - start) / bin_size) + eps) * bin_size
    # In theory, `discard` should always be non-negative.
    # Floating-point roundoff may make it slightly negative,
    # in that case, we avoid reslicing to prevent dropping the last spike.
    if discard > 0:
        if right:
            start += discard
        else:
            end -= discard
        # reslice
        spikes = spikes.slice(start, end)

    num_bins = round((end - start) / bin_size)

    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_bins, num_units), dtype=dtype)
    # Handle timestamps when the domain start is non-zero
    ts = spikes.timestamps - start
    bin_index = np.floor(ts * rate).astype(int)
    np.add.at(binned_spikes, (bin_index, spikes.unit_index), 1)
    if max_spikes is not None:
        np.clip(binned_spikes, None, max_spikes, out=binned_spikes)

    return binned_spikes


def infer_behavior_step(dataset) -> float:
    r"""Behavior sampling period (seconds), read from the first recording.

    Varies by brainset (Churchland ~1000 Hz, O'Doherty-Sabes 250 Hz), and the
    readout config may point at a field other than ``cursor`` (Flint and Gallego
    use ``hand``). Baselines are single-dataset, so recording 0 is representative.
    """
    recording = dataset.get_recording(dataset.recording_ids[0])

    timestamp_key = "cursor.timestamps"
    config = getattr(recording, "config", None)
    if isinstance(config, dict) and "readout" in config:
        timestamp_key = config["readout"].get("timestamp_key", timestamp_key)

    # "cursor.timestamps" -> the "cursor" object that carries the rate
    field = timestamp_key.rsplit(".", 1)[0]
    obj = recording.get_nested_attribute(field)

    rate = getattr(obj, "sampling_rate", None)
    if rate:
        return 1.0 / float(rate)

    # IrregularTimeSeries carries no rate; median ignores the domain-gap diffs
    timestamps = np.asarray(recording.get_nested_attribute(timestamp_key)[:100_000])
    if timestamps.size < 2:
        raise ValueError(f"{timestamp_key}: need >=2 timestamps to infer a rate")
    step = float(np.median(np.diff(timestamps)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"{timestamp_key}: inferred a non-positive period {step}")
    return step
