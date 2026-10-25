import copy
from copy import deepcopy
from src.datasets.perich_miller_population_2018 import PerichMillerPopulation2018
from temporaldata import Data
from pathlib import Path
from torch_brain.data import collate as tbrain_collate
import torch
from src.datasets.sampling_mixin import SamplingIntervalMixin
import numpy as np
from temporaldata import Interval
from typing import Optional, Literal
from glob import glob
from omegaconf import ListConfig


def calc_out_idx_slice(ctx, out):
    total = ctx + out

    offsets = torch.cat(
        [
            torch.zeros(1, dtype=total.dtype, device=total.device),
            torch.cumsum(total[:-1], dim=0),
        ]
    )

    starts = offsets + ctx

    output_query_idxs = (
        torch.repeat_interleave(starts, out)
        + torch.arange(out.sum(), device=ctx.device)
        - torch.repeat_interleave(torch.cumsum(out, 0) - out, out)
    )

    T = total.sum()

    mask = torch.ones(T, dtype=torch.bool, device=ctx.device)
    mask[output_query_idxs] = False

    input_idxs = torch.arange(T, device=ctx.device)[mask]

    return output_query_idxs, input_idxs


class PerichMillerDataset(SamplingIntervalMixin, PerichMillerPopulation2018):

    def __init__(self, root, transform=None, **kwargs):

        self.use_weights = kwargs.pop("use_weights", True)

        self.CO_READOUT_CONFIG = {
            "readout": {
                "weights": {
                    "movement_phases.random_period": 1.0,
                    "movement_phases.hold_period": 0.1 if self.use_weights else 1.0,
                    "movement_phases.reach_period": 1.0,
                    "movement_phases.return_period": 1.0,
                    "movement_phases.invalid": 0.1 if self.use_weights else 1.0,
                    "cursor_outlier_segments": 0.0,
                },
                "eval_interval": "movement_phases.reach_period",
            }
        }

        self.RT_READOUT_CONFIG = {
            "readout": {
                "weights": {
                    "movement_phases.random_period": 1.0,
                    "movement_phases.hold_period": 0.1 if self.use_weights else 1.0,
                    "cursor_outlier_segments": 0.0,
                    "movement_phases.invalid": 0.1 if self.use_weights else 1.0,
                },
                "eval_interval": "movement_phases.random_period",
            }
        }

        dirname = kwargs.get("dirname", "perich_miller_population_2018")
        recording_ids = kwargs.pop("recording_ids", TRAIN_RECORDING_IDS)
        dataset_dir = Path(root) / dirname
        if type(recording_ids) == str:
            recording_ids = dataset_dir.glob(recording_ids)
            recording_ids = [
                Path(rid).stem
                for rid in recording_ids
                if Path(rid).stem in TRAIN_RECORDING_IDS
            ]

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
        elif isinstance(recording_ids, ListConfig) and len(recording_ids) == 1:
            pass
        else:
            filtered_recording_ids = []
            for rid in recording_ids:
                rid = rid.split("/")[-1]
                if rid in TRAIN_RECORDING_IDS:
                    filtered_recording_ids.append(rid)
            recording_ids = filtered_recording_ids

        test_size = kwargs.pop("test_size", 0.2)
        valid_size = kwargs.pop("valid_size", 0.1)

        super().__init__(
            root,
            recording_ids=recording_ids,
            transform=transform,
            **kwargs,
        )

        # Pre-compute session-wise stats from the training split only.
        # Use a throwaway deepcopy so _data_objects stays fully lazy for workers.
        self._session_stats = {}
        for rid in self.recording_ids:
            data = copy.deepcopy(self._data_objects[rid])
            if self.split_type == "causal":
                train_domain = self.make_causal_splits(data, test_size, valid_size)[
                    "train_domain"
                ]
            else:
                train_domain = data.train_domain
            vel = np.asarray(
                data.cursor.to_irregular().select_by_interval(train_domain).vel
            )
            self._session_stats[data.session.id] = (
                np.mean(vel, axis=0).tolist(),
                np.std(vel, axis=0).tolist(),
            )

    def get_recording_hook(self, data: Data):
        if data.session.id.endswith("center_out_reaching"):
            data.config = deepcopy(self.CO_READOUT_CONFIG)
        elif data.session.id.endswith("random_target_reaching"):
            data.config = deepcopy(self.RT_READOUT_CONFIG)

        mean, std = self._session_stats[data.session.id]
        data.config["readout"]["normalize_mean"] = mean
        data.config["readout"]["normalize_std"] = std

        super().get_recording_hook(data)

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
        train_sampling_intervals = Interval(
            start=data.domain.start[0], end=valid_trials.start[0] - 1.0
        )

        return {
            "train_domain": train_sampling_intervals,
            "valid_domain": valid_trials,
            "test_domain": test_trials,
        }

    @staticmethod
    def process_batch(batch: dict) -> dict:
        if "input_seqlen_ctx" not in batch["model_inputs"]:
            return batch

        if "spike_seqlen" in batch["model_inputs"]:
            neural_batch_idx = batch["model_inputs"]["spike_seqlen"].nonzero(
                as_tuple=True
            )[0]
            spike_batch_size = len(neural_batch_idx)
        else:
            neural_batch_idx = batch["model_inputs"]["bins_seqlen"].nonzero(
                as_tuple=True
            )[0]
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
            print(f"Found some empty samples")
        assert len(data_clean) > 0, data

        collated_data = tbrain_collate(data_clean)
        collated_data = PerichMillerDataset.process_batch(collated_data)
        return collated_data


TRAIN_RECORDING_IDS = [
    "c_20131003_center_out_reaching",
    "c_20131022_center_out_reaching",
    "c_20131023_center_out_reaching",
    "c_20131031_center_out_reaching",
    "c_20131101_center_out_reaching",
    "c_20131203_center_out_reaching",
    "c_20131204_center_out_reaching",
    "c_20131219_center_out_reaching",
    "c_20131220_center_out_reaching",
    "c_20150309_center_out_reaching",
    "c_20150311_center_out_reaching",
    "c_20150312_center_out_reaching",
    "c_20150313_center_out_reaching",
    "c_20150319_center_out_reaching",
    "c_20150629_center_out_reaching",
    "c_20150630_center_out_reaching",
    "c_20150701_center_out_reaching",
    "c_20150703_center_out_reaching",
    "c_20150706_center_out_reaching",
    "c_20150707_center_out_reaching",
    "c_20150708_center_out_reaching",
    "c_20150709_center_out_reaching",
    "c_20150710_center_out_reaching",
    "c_20150713_center_out_reaching",
    "c_20150714_center_out_reaching",
    "c_20150715_center_out_reaching",
    "c_20150716_center_out_reaching",
    "c_20151103_center_out_reaching",
    "c_20151104_center_out_reaching",
    "c_20151106_center_out_reaching",
    "c_20151109_center_out_reaching",
    "c_20151110_center_out_reaching",
    "c_20151112_center_out_reaching",
    "c_20151113_center_out_reaching",
    "c_20151116_center_out_reaching",
    "c_20151117_center_out_reaching",
    "c_20151119_center_out_reaching",
    "c_20151120_center_out_reaching",
    "c_20151201_center_out_reaching",
    "c_20160909_center_out_reaching",
    "c_20160912_center_out_reaching",
    "c_20160914_center_out_reaching",
    "c_20160915_center_out_reaching",
    "c_20160919_center_out_reaching",
    "c_20160921_center_out_reaching",
    "c_20160923_center_out_reaching",
    "c_20160929_center_out_reaching",
    "c_20161005_center_out_reaching",
    "c_20161006_center_out_reaching",
    "c_20161007_center_out_reaching",
    "c_20161011_center_out_reaching",
    "c_20161013_center_out_reaching",
    "c_20161021_center_out_reaching",
    # "j_20160405_center_out_reaching",
    # "j_20160406_center_out_reaching",
    # "j_20160407_center_out_reaching",
    "m_20140203_center_out_reaching",
    "m_20140217_center_out_reaching",
    "m_20140218_center_out_reaching",
    "m_20140303_center_out_reaching",
    "m_20140304_center_out_reaching",
    "m_20140306_center_out_reaching",
    "m_20140307_center_out_reaching",
    "m_20140626_center_out_reaching",
    "m_20140627_center_out_reaching",
    "m_20140929_center_out_reaching",
    "m_20141203_center_out_reaching",
    "m_20150511_center_out_reaching",
    "m_20150512_center_out_reaching",
    "m_20150610_center_out_reaching",
    "m_20150611_center_out_reaching",
    "m_20150612_center_out_reaching",
    "m_20150615_center_out_reaching",
    "m_20150616_center_out_reaching",
    "m_20150617_center_out_reaching",
    "m_20150623_center_out_reaching",
    # "m_20150625_center_out_reaching",
    # "m_20150626_center_out_reaching",
    "c_20131009_random_target_reaching",
    "c_20131010_random_target_reaching",
    "c_20131011_random_target_reaching",
    "c_20131028_random_target_reaching",
    "c_20131029_random_target_reaching",
    "c_20131209_random_target_reaching",
    "c_20131210_random_target_reaching",
    "c_20131212_random_target_reaching",
    "c_20131213_random_target_reaching",
    "c_20131217_random_target_reaching",
    "c_20131218_random_target_reaching",
    "c_20150316_random_target_reaching",
    "c_20150317_random_target_reaching",
    # "c_20150318_random_target_reaching",
    # "c_20150320_random_target_reaching",
    "m_20140114_random_target_reaching",
    "m_20140115_random_target_reaching",
    "m_20140116_random_target_reaching",
    "m_20140214_random_target_reaching",
    "m_20140221_random_target_reaching",
    "m_20140224_random_target_reaching",
]
