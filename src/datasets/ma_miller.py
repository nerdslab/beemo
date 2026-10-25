import copy
from copy import deepcopy

import numpy as np
import torchmetrics
from temporaldata import Data, Interval

from src.datasets.ma_miller_adversarial_bci_2023 import MaMillerAdversarialBCI2023
from src.datasets.sampling_mixin import SamplingIntervalMixin
from src.datasets.perich_miller import calc_out_idx_slice
from torch_brain.data import collate as tbrain_collate
from glob import glob
from pathlib import Path
from omegaconf import ListConfig

# Tasks with 2-D cursor output (center-out and random-target reaching).
REACHING_TASKS = {"center_out_reaching", "random_target_reaching"}

# Tasks with 2-D isometric force output.
FORCE_TASKS = {"isometric_wrist", "power_grasp", "key_grasp"}


def _task_from_session_id(session_id: str) -> str:
    """'chewie_20160927_center_out_reaching' -> 'center_out_reaching'"""
    return "_".join(session_id.split("_")[2:])


class MaMillerDataset(SamplingIntervalMixin, MaMillerAdversarialBCI2023):
    """POYO-compatible dataset for Ma et al. 2023 (eLife 84296).

    Supports both reaching sessions (cursor velocity readout) and isometric /
    grasp sessions (hand force readout).  Pop pkl sessions store pre-smoothed
    firing rates and must be excluded from recording_ids.
    """

    def __init__(self, root, transform=None, **kwargs):
        self.use_weights = kwargs.pop("use_weights", True)

        # Cursor velocity readout — center-out reaching
        self.CO_READOUT_CONFIG = {
            "readout": {
                "readout_id": "cursor_velocity_2d",
                "weights": {
                    "movement_phases.hold_period": 0.1 if self.use_weights else 1.0,
                    "movement_phases.reach_period": 1.0,
                },
                "metrics": [{"metric": torchmetrics.R2Score()}],
                "eval_interval": "movement_phases.reach_period",
            }
        }

        # Cursor velocity readout — random-target reaching
        self.RT_READOUT_CONFIG = {
            "readout": {
                "readout_id": "cursor_velocity_2d",
                "weights": {
                    "movement_phases.hold_period": 0.1 if self.use_weights else 1.0,
                    "movement_phases.reach_period": 1.0,
                },
                "metrics": [{"metric": torchmetrics.R2Score()}],
                "eval_interval": "movement_phases.reach_period",
            }
        }

        # Hand force readout — isometric wrist / power grasp / key grasp
        self.FORCE_READOUT_CONFIG = {
            "readout": {
                "readout_id": "cursor_velocity_2d",
                "metrics": [{"metric": torchmetrics.R2Score()}],
            }
        }

        dirname = kwargs.get("dirname", "ma_miller_adversarial_bci_2023")
        recording_ids = kwargs.pop("recording_ids", TRAIN_RECORDING_IDS)
        dataset_dir = Path(root) / dirname

        if isinstance(recording_ids, str):
            recording_ids = [
                Path(p).stem for p in glob(f"{dataset_dir}/{recording_ids}*.h5")
            ]
            recording_ids = [
                rid
                for rid in recording_ids
                if ("pop" not in rid and "greyson" not in rid)
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
            try:
                vel = np.asarray(
                    data.cursor.to_irregular().select_by_interval(train_domain).vel
                )
            except (KeyError, AttributeError):
                vel = None
            if vel is not None:
                self._session_stats[data.session.id] = (
                    np.mean(vel, axis=0).tolist(),
                    np.std(vel, axis=0).tolist(),
                )

    # ------------------------------------------------------------------
    # Recording hook — assign readout config per task type
    # ------------------------------------------------------------------

    def get_recording_hook(self, data: Data):
        task = _task_from_session_id(data.session.id)
        if task == "center_out_reaching":
            data.config = deepcopy(self.CO_READOUT_CONFIG)
        elif task == "random_target_reaching":
            data.config = deepcopy(self.RT_READOUT_CONFIG)
        elif task in FORCE_TASKS:
            data.config = deepcopy(self.FORCE_READOUT_CONFIG)
        else:
            raise ValueError(f"Unknown task '{task}' in session '{data.session.id}'")
        if data.session.id in self._session_stats:
            mean, std = self._session_stats[data.session.id]
            data.config["readout"]["normalize_mean"] = mean
            data.config["readout"]["normalize_std"] = std
        return super().get_recording_hook(data)

    # ------------------------------------------------------------------
    # Causal train / valid / test splits
    # ------------------------------------------------------------------

    def make_causal_splits(self, data, test_size=0.2, valid_size=0.1):
        """Temporally ordered splits based on rewarded trials.

        For sessions without `trials` (e.g. any continuous recording), falls
        back to a simple linspace split of the domain.
        """
        # if not hasattr(data, "trials") or data.trials is None:
        #     return self._linspace_splits(data, test_size, valid_size)

        trials = data.trials
        if hasattr(trials, "is_valid"):
            trials = trials.select_by_mask(trials.is_valid)

        num_trials = len(trials)
        # if num_trials == 0:
        #     return self._linspace_splits(data, test_size, valid_size)

        num_test = int(round(num_trials * test_size))
        num_valid = int(round(num_trials * valid_size))
        num_train = num_trials - num_test - num_valid

        idxs = np.arange(num_trials)
        valid_trials = trials.select_by_mask(
            np.isin(idxs, idxs[num_train : num_train + num_valid])
        )
        test_trials = trials.select_by_mask(
            np.isin(idxs, idxs[num_train + num_valid :])
        )
        train_sampling_intervals = (
            Interval(start=data.domain.start[0], end=valid_trials.start[0] - 1.0)
            & data.domain
        )

        return {
            "train_domain": train_sampling_intervals,
            "valid_domain": valid_trials,
            "test_domain": test_trials,
        }

    # def _linspace_splits(self, data, test_size=0.2, valid_size=0.1):
    #     """Time-based split into 10 equal blocks (fallback when no trials)."""
    #     train_size = 1.0 - test_size - valid_size
    #     intervals = Interval.linspace(data.domain.start[0], data.domain.end[-1], 10)
    #     n_train = int(train_size * len(intervals))
    #     n_valid = int(valid_size * len(intervals))
    #     n_test = len(intervals) - n_train - n_valid
    #     train_iv, valid_iv, test_iv = intervals.split(
    #         [n_train, n_valid, n_test], shuffle=False
    #     )
    #     return {
    #         "train_domain": train_iv,
    #         "valid_domain": valid_iv,
    #         "test_domain": test_iv,
    #     }

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
        collated_data = MaMillerDataset.process_batch(collated_data)
        return collated_data


TRAIN_RECORDING_IDS = [
    # Mihili
    "mihili_20140211_center_out_reaching",
    "mihili_20140225_center_out_reaching",
    "mihili_20140227_center_out_reaching",
    "mihili_20140228_center_out_reaching",
    "mihili_20131207_random_target_reaching",
    "mihili_20131208_random_target_reaching",
    "mihili_20140128_random_target_reaching",
    "mihili_20140129_random_target_reaching",
    "mihili_20140212_random_target_reaching",
    "mihili_20140222_random_target_reaching",
    # Chewie
    "chewie_20160927_center_out_reaching",
    "chewie_20160928_center_out_reaching",
    "chewie_20160930_center_out_reaching",
    "chewie_20161012_center_out_reaching",
    "chewie_20161102_center_out_reaching",
    "chewie_20161103_center_out_reaching",
    "chewie_20161104_center_out_reaching",
    # Greyson
    "greyson_20190812_key_grasp",
    "greyson_20190815_key_grasp",
    "greyson_20190819_key_grasp",
    "greyson_20190911_key_grasp",
    "greyson_20190913_key_grasp",
    "greyson_20190918_key_grasp",
    "greyson_20190920_key_grasp",
    "greyson_20190923_key_grasp",
    "greyson_20191007_key_grasp",
    # Pop
    "pop_20210602_power_grasp",
    "pop_20210616_power_grasp",
    "pop_20210618_power_grasp",
    "pop_20210625_power_grasp",
    "pop_20210630_power_grasp",
    "pop_20210702_power_grasp",
    "pop_20210709_power_grasp",
    "pop_20210716_power_grasp",
    "pop_20210723_power_grasp",
    # # Spike
    # "spike_20120821_isometric_wrist",
    # "spike_20120822_isometric_wrist",
    # "spike_20120823_isometric_wrist",
    # "spike_20120824_isometric_wrist",
    # "spike_20120829_isometric_wrist",
    # "spike_20120831_isometric_wrist",
    # "spike_20120905_isometric_wrist",
    # "spike_20120928_isometric_wrist",
    # "spike_20121015_isometric_wrist",
    # "spike_20121017_isometric_wrist",
    # "spike_20121018_isometric_wrist",
    # "spike_20121023_isometric_wrist",
    # "spike_20121025_isometric_wrist",
    # "spike_20121105_isometric_wrist",
    # "spike_20121107_isometric_wrist",
    # "spike_20121108_isometric_wrist",
    # "spike_20121109_isometric_wrist",
    # "spike_20121112_isometric_wrist",
    # Jango
    "jango_20150730_isometric_wrist",
    "jango_20150731_isometric_wrist",
    "jango_20150801_isometric_wrist",
    "jango_20150805_isometric_wrist",
    "jango_20150806_isometric_wrist",
    "jango_20150807_isometric_wrist",
    "jango_20150808_isometric_wrist",
    "jango_20150809_isometric_wrist",
    "jango_20150820_isometric_wrist",
    "jango_20150824_isometric_wrist",
    "jango_20150825_isometric_wrist",
    "jango_20150826_isometric_wrist",
    "jango_20150827_isometric_wrist",
    "jango_20150828_isometric_wrist",
    "jango_20150831_isometric_wrist",
    # "jango_20150905_isometric_wrist",
    # "jango_20150906_isometric_wrist",
    # "jango_20150908_isometric_wrist",
    # "jango_20151029_isometric_wrist",
    # "jango_20151102_isometric_wrist",
]
