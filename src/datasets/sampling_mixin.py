import numpy as np
import pandas as pd
from typing import Optional, Literal


class SamplingIntervalMixin:

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
        test_size: float = 0.2,
        valid_size: float = 0.1,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"

        sampling_intervals = {}
        for rid in self.recording_ids:
            if self.split_type == "causal":
                sampling_intervals[rid] = self.make_causal_splits(
                    self.get_recording(rid), test_size, valid_size
                )[domain_key]
            elif self.split_type == "random":
                sampling_intervals[rid] = getattr(self.get_recording(rid), domain_key)

        return sampling_intervals

    def get_session_ids(self):
        return [self.get_recording(rid).session.id for rid in self.recording_ids]
