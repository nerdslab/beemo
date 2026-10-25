# --------------------------------------------------
# Code adapted from torch_brain
# --------------------------------------------------

from collections import defaultdict
from typing import Callable, Iterable, List, Optional
import torch
import torchmetrics

try:
    import wandb
except ImportError:
    wandb = None
from src.distributed import all_gather_object


def stitch(
    timestamps: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Pools values that share the same timestamp using mean or mode operations.

    This function is useful when you have multiple predictions or values for the same
    timestamp (e.g., from overlapping windows) and need to combine them into a single
    value per timestamp.

    Args:
        timestamps (torch.Tensor): A 1D tensor containing timestamps. Shape: (N,)
        values (torch.Tensor): A tensor of values corresponding to the timestamps.
            Shape:
                - For floating point types: (N, ...)
                - For categorical types (torch.long only): (N,)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - unique_timestamps: A 1D tensor of sorted unique timestamps
            - pooled_values: A tensor containing the pooled values for each unique timestamp
                - For continuous data (float types): Uses mean pooling
                - For categorical data (long type): Uses mode pooling

    Examples:
        >>> # Mean pooling for continuous values
        >>> timestamps = torch.tensor([1, 1, 2, 3, 3])
        >>> values = torch.tensor([0.1, 0.3, 0.2, 0.4, 0.6])
        >>> stitch(timestamps, values)
        (tensor([1, 2, 3]), tensor([0.2000, 0.2000, 0.5000]))

        >>> # Mode pooling for categorical values
        >>> timestamps = torch.tensor([1, 1, 2, 3, 3, 3])
        >>> values = torch.tensor([1, 1, 2, 3, 3, 1], dtype=torch.long)
        >>> stitch(timestamps, values)
        (tensor([1, 2, 3]), tensor([1, 2, 3]))
    """
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    if values.dtype == torch.long:
        # Mode pooling for categorical values

        if values.ndim != 1:
            raise ValueError(
                "For categorical values (long type), only 1D tensors are supported. "
                f"Got values with shape {values.shape} instead."
            )

        # 1. Construct a N x C class-wise vote tensor
        votes = values.new_zeros((len(unique_timestamps), values.max() + 1))
        votes.index_put_((indices, values), torch.ones_like(indices), accumulate=True)
        # 2. Mode class is the one with most votes
        mode_values = votes.argmax(dim=-1)
        return unique_timestamps, mode_values

    elif torch.is_floating_point(values):
        # Mean-pool for floating points
        # 1. Count occurrences of each unique timestamp
        counts = torch.zeros_like(unique_timestamps, dtype=torch.long)
        counts.index_add_(0, indices, torch.ones_like(indices))
        if values.dim() > 1:
            counts = counts.unsqueeze(-1)
        # 2. Accumulate and average values for each unique timestamp
        avg_values = values.new_zeros((len(unique_timestamps), *values.shape[1:]))
        avg_values.index_add_(0, indices, values).div_(counts)
        # Regarding division by zero: all elements of counts will be >= 1.
        # Reasoning: Since it was built using unique_timestamps, each index will have
        # atleast one timestamp attached to it.

        return unique_timestamps, avg_values

    else:
        raise TypeError(
            f"Unsupported dtype {values.dtype} for stitching. "
            "Only floating points supported for mean pooling, "
            " and torch.long type supported for mode pooling."
        )


class DecodingStitchEvaluator:

    def __init__(
        self,
        session_ids: Iterable[str],
        device: torch.device,
        dim: int = 2,
        metric_factory: Optional[Callable[[], torchmetrics.Metric]] = None,
    ):
        r"""
        Args:
            session_ids: An iterable of session IDs for which the metrics are to be computed.
            dim: Dimension of the behavior being decoded.
            metric_factory: (Optional) A callable that returns an instance of the metric to be used.
                Defaults to R2Score.
        """
        if metric_factory is None:
            metric_factory = lambda: torchmetrics.R2Score()

        self.metrics = {k: metric_factory() for k in session_ids}

        self.dim = dim
        self.device = device
        self._init_cache()

    def _init_cache(self):
        # Cache to store the predictions, targets, and timestamps for each
        # validation step. This will be coalesced at the end of the validation,
        # using the stitch function.
        self._cache = defaultdict(
            lambda: {
                "pred": [],
                "target": [],
                "timestamps": [],
            }
        )

    def update(
        self,
        timestamps: List[torch.Tensor],
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
        eval_masks: List[torch.Tensor],
        session_ids: List[str],
        absolute_starts: torch.Tensor | List[float],
    ):
        batch_size = len(timestamps)
        for i in range(batch_size):
            mask = eval_masks[i]
            session_id = session_ids[i]

            _preds = preds[i][mask]
            _targets = targets[i][mask]
            _timestamps = timestamps[i][mask] + absolute_starts[i]

            self._cache[session_id]["pred"].append(_preds.detach())
            self._cache[session_id]["target"].append(_targets.detach())
            self._cache[session_id]["timestamps"].append(_timestamps.detach())

    def compute(self, prefix="val"):
        # compute metric for each session
        metrics = {}
        for session_id, metric_fn in self.metrics.items():
            cache = self._cache[session_id]

            if len(cache["pred"]) > 0:
                pred = all_gather_object(torch.cat(cache["pred"]))
                target = all_gather_object(torch.cat(cache["target"]))
                timestamps = all_gather_object(torch.cat(cache["timestamps"]))

                sort_idx = torch.argsort(timestamps)
                timestamps = timestamps[sort_idx]
                pred = pred[sort_idx]
                target = target[sort_idx]

                stitched_pred = stitch(timestamps, pred)[1]
                stitched_target = stitch(timestamps, target)[1]

                metric_fn.to(self.device).update(stitched_pred, stitched_target)
                metrics[session_id] = metric_fn.compute().item()

            else:
                metric_fn.to(self.device).update(
                    torch.empty(0, self.dim, device=self.device),
                    torch.empty(0, self.dim, device=self.device),
                )
                metrics[session_id] = 0
            metric_fn.reset()

        return metrics

    def reset(self):
        self._init_cache()
