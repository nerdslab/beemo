# --------------------------------------------------
# Code adapted from Nuclr
# Author: Vinam Arora
# --------------------------------------------------


from typing import Dict, Iterator, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.distributed as dist
from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex


# class RandomFixedWindowSampler(torch.utils.data.Sampler):
#     """Samples pairs of temporal windows from recordings for contrastive learning.

#     Note: The number of samples may vary between epochs due to random sampling.

#     Args:
#         sampling_intervals: Dictionary mapping recording IDs to Interval objects
#             specifying valid time ranges for sampling.
#         window_length: Length of each sampled window in seconds.
#         max_time_diff: Maximum allowed time difference between paired windows.
#             If None, no limit is imposed.
#         seed: Random seed for reproducibility.
#     """

#     def __init__(
#         self,
#         sampling_intervals: Dict[str, Interval],
#         window_length: float,
#         seed: int,
#     ):

#         self.sampling_intervals = sampling_intervals
#         self.window_length = window_length
#         self.init_seed = seed
#         self.np_rng = np.random.default_rng(seed)

#         self._indices_cache = None

#     def set_epoch(self, epoch: int):
#         self.np_rng = np.random.default_rng(self.init_seed + epoch)

#     def _get_one_recording_sample_indices(
#         self,
#         rec_id: str,
#     ) -> List[Tuple[DatasetIndex, DatasetIndex]]:
#         """Returns sampling indices for a single recording"""

#         interval = self.sampling_intervals[rec_id]
#         sample_indices = []
#         for start, end in zip(interval.start, interval.end):
#             offset = self.np_rng.uniform(0, self.window_length)

#             for _start in np.arange(start + offset, end - self.window_length, self.window_length):
#                 # First view
#                 sample_indices.append(DatasetIndex(rec_id, _start, _start + self.window_length))

#         assert len(sample_indices) > 0
#         return sample_indices

#     def _prepare_indices_cache(self):
#         if self._indices_cache is not None:
#             return

#         self._indices_cache = []
#         for rec_id in self.sampling_intervals.keys():
#             self._indices_cache.extend(self._get_one_recording_sample_indices(rec_id))

#         self.np_rng.shuffle(self._indices_cache)

#     def __len__(self) -> int:
#         self._prepare_indices_cache()
#         return len(self._indices_cache)

#     def __iter__(self) -> Iterator[Tuple[DatasetIndex, DatasetIndex]]:
#         self._prepare_indices_cache()
#         for index in self._indices_cache:
#             yield index
#         self._indices_cache = None

import logging
from functools import cached_property


class RandomFixedWindowSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`sampling_intervals` parameter. :obj:`sampling_intervals` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        indices = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                # sample a random offset
                left_offset = (
                    torch.rand(1, generator=self.generator).item() * self.window_length
                )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(
                        start + left_offset,
                        end,
                        self.window_length,
                        dtype=torch.float64,
                    )
                    if t + self.window_length <= end
                ]

                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )
                    else:
                        indices.append(
                            DatasetIndex(
                                session_name, start, start + self.window_length
                            )
                        )

        # shuffle
        for idx in torch.randperm(len(indices), generator=self.generator):
            yield indices[idx]


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    """Wrapper for distributing any sampler across multiple processes.

    This wrapper takes an existing sampler and distributes its indices across
    multiple replicas (processes) in a distributed training setup. It ensures
    each replica gets a unique subset of the data by splitting the indices
    evenly across replicas.

    Note: This wrapper supports samplers that may return a different number of
    samples each epoch, as it recomputes the indices on each iteration.

    Note: If the length of the sampler is not a multiple of num_replicas, some
    samples will be dropped to ensure equal distribution across replicas.

    Args:
        sampler: The base sampler to wrap
        num_replicas: Number of processes in distributed training
        rank: Rank of the current process
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.sampler = sampler

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self._indices_cache = None

    def _prepare_indices_cache(self):
        if self._indices_cache is not None:
            return

        indices = list(self.sampler)

        # Force len(indices) to be a multiple of num_replicas
        num_samples_per_rank = math.floor(len(indices) / self.num_replicas)
        total_size = num_samples_per_rank * self.num_replicas
        indices = indices[:total_size]

        self._indices_cache = indices[self.rank : total_size : self.num_replicas]

    def __len__(self):
        self._prepare_indices_cache()
        return len(self._indices_cache)

    def __iter__(self):
        self._prepare_indices_cache()
        for idx in self._indices_cache:
            yield idx
        self._indices_cache = None

    def set_epoch(self, epoch: int):
        """Pass the epoch to the underlying sampler if it supports it."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class RandomFixedWindowBehaviorSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows randomly, given intervals defined in the
    :obj:`sampling_intervals` parameter. :obj:`sampling_intervals` is a dictionary where the keys
    are the session ids and the values are lists of tuples representing the
    start and end of the intervals from which to sample. The samples are shuffled, and
    random temporal jitter is applied.


    In one epoch, the number of samples that is generated from a given sampling interval
    is given by:

    .. math::
        N = \left\lfloor\frac{\text{interval_length}}{\text{window_length}}\right\rfloor

    Args:
        sampling_intervals (Dict[str, List[Tuple[int, int]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        generator (Optional[torch.Generator], optional): Generator for shuffling.
            Defaults to None.
        drop_short (bool, optional): Whether to drop short intervals. Defaults to True.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        generator: Optional[torch.Generator] = None,
        drop_short: bool = True,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.generator = generator
        self.drop_short = drop_short

    @cached_property
    def _estimated_len(self):
        num_samples = 0
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                num_samples += math.floor(interval_length / self.window_length)

        if self.drop_short and total_short_dropped > 0:
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")
        return num_samples

    def __len__(self):
        return self._estimated_len

    def __iter__(self):
        if len(self) == 0.0:
            raise ValueError("All intervals are too short to sample from.")

        indices = []
        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                # sample a random offset
                max_offset = interval_length - self.window_length
                if max_offset > self.window_length:
                    max_offset = self.window_length

                left_offset = (
                    torch.rand(1, generator=self.generator).item() * max_offset
                )

                indices_ = [
                    DatasetIndex(
                        session_name,
                        start + left_offset,
                        start + left_offset + self.window_length,
                    )
                ]

                indices_.extend(
                    [
                        DatasetIndex(
                            session_name,
                            t.item() - self.window_length,
                            (t + self.window_length).item(),
                        )
                        for t in torch.arange(
                            start + left_offset + self.window_length,
                            end,
                            self.window_length,
                            dtype=torch.float64,
                        )
                        if t + self.window_length <= end
                    ]
                )

                if len(indices_) > 0:
                    indices.extend(indices_)
                    right_offset = end - indices[-1].end
                else:
                    right_offset = end - start - left_offset

                # if there is one sample worth of data, add it
                # this ensures that the number of samples is always consistent
                if right_offset + left_offset >= self.window_length:
                    if right_offset > left_offset:
                        if interval_length > 2 * self.window_length:
                            indices.append(
                                DatasetIndex(
                                    session_name, end - 2 * self.window_length, end
                                )
                            )
                        else:
                            indices.append(
                                DatasetIndex(
                                    session_name, end - self.window_length, end
                                )
                            )
                    else:
                        if interval_length > 2 * self.window_length:
                            indices.append(
                                DatasetIndex(
                                    session_name, start, start + 2 * self.window_length
                                )
                            )
                        else:
                            indices.append(
                                DatasetIndex(
                                    session_name, start, start + self.window_length
                                )
                            )

        # shuffle
        for idx in torch.randperm(len(indices), generator=self.generator):
            yield indices[idx]


class SequentialFixedWindowBehaviorSampler(torch.utils.data.Sampler):
    r"""Samples fixed-length windows sequentially, always in the same order. The
    sampling intervals are defined in the :obj:`sampling_intervals` parameter.
    :obj:`sampling_intervals` is a dictionary where the keys are the session ids and the
    values are lists of tuples representing the start and end of the intervals
    from which to sample.

    If the length of a sequence is not evenly divisible by the step, the last
    window will be added with an overlap with the previous window. This is to ensure
    that the entire sequence is covered.

    Args:
        sampling_intervals (Dict[str, List[Tuple[float, float]]]): Sampling intervals for each
            session in the dataset.
        window_length (float): Length of the window to sample.
        step (float, optional): Step size between windows. If None, it
            defaults to ``window_length``.
        drop_short (bool, optional): Whether to drop windows smaller than ``window_length``.
            Defaults to False.
    """

    def __init__(
        self,
        *,
        sampling_intervals: Dict[str, List[Tuple[float, float]]],
        window_length: float,
        step: Optional[float] = None,
        drop_short=False,
    ):
        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.step = step or window_length
        self.drop_short = drop_short

        assert self.step > 0, "Step must be greater than 0."

    # we cache the indices since they are deterministic
    @cached_property
    def _indices(self) -> List[DatasetIndex]:
        indices = []
        total_short_dropped = 0.0

        for session_name, sampling_intervals in self.sampling_intervals.items():
            for start, end in zip(sampling_intervals.start, sampling_intervals.end):
                interval_length = end - start
                if interval_length < self.window_length:
                    if self.drop_short:
                        total_short_dropped += interval_length
                        continue
                    else:
                        raise ValueError(
                            f"Interval {(start, end)} is too short to sample from. "
                            f"Minimum length is {self.window_length}."
                        )

                indices_ = [
                    DatasetIndex(
                        session_name, t.item(), (t + self.window_length).item()
                    )
                    for t in torch.arange(
                        start,
                        start + self.window_length,
                        self.step,
                        dtype=torch.float64,
                    )
                    if t + self.window_length <= end
                ]
                indices_.extend(
                    [
                        DatasetIndex(
                            session_name,
                            t.item() - self.window_length,
                            (t + self.window_length).item(),
                        )
                        for t in torch.arange(
                            start + self.window_length,
                            end,
                            self.step,
                            dtype=torch.float64,
                        )
                        if t + self.window_length <= end
                    ]
                )

                indices.extend(indices_)

                # we need to make sure that the entire interval is covered

                if indices_[-1].end < end:
                    if interval_length >= 2 * self.window_length:
                        indices.append(
                            DatasetIndex(
                                session_name, end - 2 * self.window_length, end
                            )
                        )
                    else:
                        indices.append(
                            DatasetIndex(session_name, end - self.window_length, end)
                        )

        if self.drop_short and total_short_dropped > 0:
            num_samples = len(indices)
            logging.warning(
                f"Skipping {total_short_dropped} seconds of data due to short "
                f"intervals. Remaining: {num_samples * self.window_length} seconds."
            )
            if num_samples == 0:
                raise ValueError("All intervals are too short to sample from.")

        return indices

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        yield from self._indices
