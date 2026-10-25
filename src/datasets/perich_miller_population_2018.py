from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin
import numpy as np
from temporaldata import Interval


class PerichMillerPopulation2018(SpikingDatasetMixin, Dataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "perich_miller_population_2018",
        **kwargs,
    ):
        self.split_type = kwargs.pop("split_type", "random")
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )

        self.spiking_dataset_mixin_uniquify_unit_ids = True

    def get_session_ids(self):
        return [self.get_recording(rid).session.id for rid in self.recording_ids]
