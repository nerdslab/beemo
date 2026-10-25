import copy
from copy import deepcopy

import numpy as np
from torch_brain.data import Data, Interval

from src.datasets.gallego_carracedo_lfp_2022 import GallegoCarracedoLFP2022
from src.datasets.sampling_mixin import SamplingIntervalMixin
from src.datasets.perich_miller import calc_out_idx_slice
from torch_brain.batching import collate as tbrain_collate
from pathlib import Path
from glob import glob
from omegaconf import ListConfig


class GallegoDataset(SamplingIntervalMixin, GallegoCarracedoLFP2022):

    def __init__(self, root, transform=None, **kwargs):
        self.use_weights = kwargs.pop("use_weights", True)
        self.READOUT_CONFIG = {
            "readout": {
                "timestamp_key": "hand.timestamps",
                "value_key": "hand.vel",
                "weights": {
                    "movement_phases.hold_period": 0.1 if self.use_weights else 1.0,
                    "movement_phases.reach_period": 1.0,
                },
                "eval_interval": "movement_phases.reach_period",
            }
        }
        recording_ids = kwargs.pop("recording_ids", TRAIN_RECORDING_IDS)
        dataset_dir = Path(root) / "gallego_carracedo_lfp_2022"
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
            vel = np.asarray(
                data.hand.to_irregular().select_by_interval(train_domain).vel
            )
            self._session_stats[data.session.id] = (
                np.mean(vel, axis=0).tolist(),
                np.std(vel, axis=0).tolist(),
            )

    def get_recording_hook(self, data: Data):
        config = deepcopy(self.READOUT_CONFIG)
        if not self.use_weights:
            config["readout"]["weights"] = {
                k: 1.0 for k in config["readout"]["weights"]
            }
        data.config = config
        mean, std = self._session_stats[data.session.id]
        data.config["readout"]["normalize_mean"] = mean
        data.config["readout"]["normalize_std"] = std
        return super().get_recording_hook(data)

    def make_causal_splits(self, data, test_size=0.2, valid_size=0.1):
        trials = data.trials
        trials = trials.select_by_mask(trials.is_valid)

        num_trials = len(trials)
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
        collated_data = GallegoDataset.process_batch(collated_data)
        return collated_data


TRAIN_RECORDING_IDS = [
    # Chewie
    "chewie_20150313_center_out_reaching",
    "chewie_20150319_center_out_reaching",
    "chewie_20150629_center_out_reaching",
    "chewie_20150630_center_out_reaching",
    "chewie_20160929_center_out_reaching",
    "chewie_20161005_center_out_reaching",
    "chewie_20161006_center_out_reaching",
    "chewie_20161007_center_out_reaching",
    "chewie_20161014_center_out_reaching",
    "chewie_20161021_center_out_reaching",
    # Han
    "han_20171127_center_out_reaching",
    "han_20171128_center_out_reaching",
    "han_20171201_center_out_reaching",
    "han_20171204_center_out_reaching",
    "han_20171207_center_out_reaching",
    # # Lando
    # "lando_20170731_center_out_reaching",
    # "lando_20170803_center_out_reaching",
    # "lando_20170917_center_out_reaching",
    # Mihili
    "mihili_20140217_center_out_reaching",
    "mihili_20140218_center_out_reaching",
    "mihili_20140303_center_out_reaching",
    "mihili_20140304_center_out_reaching",
    "mihili_20140306_center_out_reaching",
    "mihili_20140307_center_out_reaching",
]
