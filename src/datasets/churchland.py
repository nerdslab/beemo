import copy
from brainsets.datasets import ChurchlandShenoyNeural2012
from temporaldata import Data
from copy import deepcopy
import torchmetrics
import numpy as np
from temporaldata import Interval
from src.datasets.sampling_mixin import SamplingIntervalMixin
from torch_brain.dataset.dataset import DatasetIndex
from src.datasets.perich_miller import calc_out_idx_slice
from torch_brain.data import collate as tbrain_collate
from pathlib import Path
from glob import glob
from omegaconf import ListConfig


class ChurchlandDataset(SamplingIntervalMixin, ChurchlandShenoyNeural2012):

    def __init__(self, root, transform=None, **kwargs):
        self.use_weights = kwargs.pop("use_weights", True)

        self.READOUT_CONFIG = {
            "readout": {
                "readout_id": "cursor_velocity_2d",
                "weights": {
                    "movement_phases.hold_period": 0.1 if self.use_weights else 1.0,
                    "movement_phases.reach_period": 1.0 if self.use_weights else 1.0,
                    "movement_phases.return_period": 1.0,
                    "cursor_outlier_segments": 0.0,
                },
                "metrics": [{"metric": torchmetrics.R2Score()}],
            }
        }

        dirname = kwargs.get("dirname", "churchland_shenoy_neural_2012")
        recording_ids = kwargs.pop("recording_ids", TRAIN_RECORDING_IDS)
        dataset_dir = Path(root) / dirname

        if isinstance(recording_ids, str):
            recording_ids = [
                Path(p).stem for p in glob(f"{dataset_dir}/{recording_ids}*.h5")
            ]
            recording_ids = [rid for rid in recording_ids if rid in TRAIN_RECORDING_IDS]

        if (
            type(recording_ids) == ListConfig
            and len(recording_ids) > 0
            and "*" in str(recording_ids[0])
        ):
            filtered_recording_ids = []
            for pattern in recording_ids:
                matching_ids = [
                    Path(path).stem for path in glob(f"{dataset_dir}/{pattern}*.h5")
                ]
                filtered_recording_ids.extend(
                    rid for rid in matching_ids if rid in TRAIN_RECORDING_IDS
                )
            recording_ids = filtered_recording_ids

        test_size = kwargs.pop("test_size", 0.2)
        valid_size = kwargs.pop("valid_size", 0.1)

        super().__init__(
            root, recording_ids=recording_ids, transform=transform, **kwargs
        )

        self._session_stats = {}
        for rid in self.recording_ids:
            data = copy.deepcopy(self._data_objects[rid])
            if self.split_type == "causal":
                train_domain = self.make_causal_splits(data, test_size, valid_size)[
                    "train_domain"
                ]
            else:
                train_domain = data.train_domain
            vel = np.asarray(data.cursor.select_by_interval(train_domain).vel)
            self._session_stats[data.session.id] = (
                np.mean(vel, axis=0).tolist(),
                np.std(vel, axis=0).tolist(),
            )

    def make_causal_splits(self, data, test_size=0.2, valid_size=0.1):
        """
        Create temporally ordered (causal) splits of trials for training, validation, and testing.

        The validation and test sets contain trials that occur *after* all trials in the training set,
        thus preventing any future information from leaking into the training set.

        Args:
            data: An object with an attribute `trials`, supporting `__len__` and `select_by_mask`, and a `domain` attribute.
            test_size: Proportion of total trials to use for testing (the last N trials).
            valid_size: Proportion of total trials to use for validation (the N trials before the test set).

        Returns:
            dict: A dictionary with keys 'train_domain', 'valid_domain', and 'test_domain', where
                'train_domain' is an Interval from the dataset domain start to the start of the validation set,
                'valid_domain' corresponds to selected validation trials,
                'test_domain' corresponds to selected test trials.
        """

        trials = data.trials
        if hasattr(trials, "is_valid"):
            trials = trials.select_by_mask(trials.is_valid)

        num_trials = len(trials)
        num_test = int(round(num_trials * test_size))
        num_valid = int(round(num_trials * valid_size))
        num_train = num_trials - num_test - num_valid

        # Causal split: sort by original order, take train (start), then valid, then test (end)
        idxs = np.arange(num_trials)

        train_ids = idxs[:num_train]
        valid_ids = idxs[num_train : num_train + num_valid]
        test_ids = idxs[num_train + num_valid :]

        # train_trials = trials.select_by_mask(np.isin(np.arange(num_trials), train_ids))
        valid_trials = trials.select_by_mask(np.isin(np.arange(num_trials), valid_ids))
        test_trials = trials.select_by_mask(np.isin(np.arange(num_trials), test_ids))
        train_sampling_intervals = (
            Interval(start=data.domain.start[0], end=valid_trials.start[0] - 1.0)
            & data.domain
        )

        return {
            "train_domain": train_sampling_intervals,
            "valid_domain": valid_trials,
            "test_domain": test_trials,
        }

    def get_recording_hook(self, data: Data):
        data.config = deepcopy(self.READOUT_CONFIG)
        mean, std = self._session_stats[data.session.id]
        data.config["readout"]["normalize_mean"] = mean
        data.config["readout"]["normalize_std"] = std
        return super().get_recording_hook(data)

    @staticmethod
    def process_batch(batch: dict) -> dict:
        if "input_seqlen_ctx" not in batch["model_inputs"]:
            return batch

        neural_batch_idx = batch["model_inputs"]["spike_seqlen"].nonzero(as_tuple=True)[
            0
        ]
        spike_batch_size = len(neural_batch_idx)

        batch["model_inputs"]["input_seqlen_ctx_behaviour_dropout"] = batch[
            "model_inputs"
        ]["input_seqlen_ctx"].clone()
        batch["model_inputs"]["input_seqlen_ctx_behaviour_dropout"][
            neural_batch_idx
        ] = 0

        output_query_idxs, input_idxs = calc_out_idx_slice(
            batch["model_inputs"]["input_seqlen_ctx"],
            batch["model_inputs"]["output_seqlen"],
        )
        batch["model_inputs"]["output_query_idx_slice"] = output_query_idxs
        batch["model_inputs"]["input_idx_slice"] = input_idxs

        # i need to get the hypothetical drop indices that would make the seq len of behavior go to 0

        output_query_idxs_behaviour_dropout, input_idxs_behaviour_dropout = (
            calc_out_idx_slice(
                batch["model_inputs"]["input_seqlen_ctx_behaviour_dropout"],
                batch["model_inputs"]["output_seqlen"],
            )
        )
        batch["model_inputs"][
            "output_query_idx_slice_behaviour_dropout"
        ] = output_query_idxs_behaviour_dropout
        batch["model_inputs"][
            "input_idx_slice_behaviour_dropout"
        ] = input_idxs_behaviour_dropout

        return batch

    @staticmethod
    def collate(data: list[dict]) -> Data:
        data_clean = [d for d in data if d is not None]
        if len(data) > len(data_clean):
            print("Found some empty samples")
        assert len(data_clean) > 0, data

        collated_data = tbrain_collate(data_clean)
        collated_data = ChurchlandDataset.process_batch(collated_data)
        return collated_data


TRAIN_RECORDING_IDS = [
    # # Jenkins
    # "jenkins_20090912_maze",
    # "jenkins_20090916_maze",
    # "jenkins_20090918_maze",
    # "jenkins_20090923_maze",
    # Nitschke
    "nitschke_20090812_maze",
    "nitschke_20090819_maze",
    "nitschke_20090910_maze",
    "nitschke_20090920_maze",
    "nitschke_20090922_maze",
    "nitschke_20100923_maze",
]
