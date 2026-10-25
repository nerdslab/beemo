from typing import Callable, Optional
from pathlib import Path

from torch_brain.datasets import Dataset, SpikingDatasetMixin


class MaMillerAdversarialBCI2023(SpikingDatasetMixin, Dataset):
    """Base dataset class for Ma et al. 2023 (eLife 84296).

    Covers .mat sessions (raw threshold crossings) for six macaques performing
    isometric wrist, power grasp, key grasp, center-out reaching, and random
    target reaching tasks.

    Note: Pop pkl sessions (pop_*) contain pre-smoothed firing rates stored as
    `neural_rates` rather than spike timestamps and are *not* compatible with
    SpikingDatasetMixin. Excluded from recording_ids when using this class.
    """

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        dirname: str = "ma_miller_adversarial_bci_2023",
        **kwargs,
    ):
        self.split_type = kwargs.pop("split_type", "causal")
        if recording_ids is None:
            recording_ids = [p.stem for p in (Path(root) / dirname).glob("*.h5")]

        recording_ids = [rid for rid in recording_ids if "pop" not in rid]
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
