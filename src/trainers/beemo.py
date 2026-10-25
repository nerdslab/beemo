import torch
from src.samplers import DistributedSamplerWrapper
from torch_brain.samplers import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch.utils.data import DataLoader
import multiprocessing as mp
from src.trainers.baseline import BaselineTrainer
import numpy as np
import re
from src.utils import Precision, get_class_from_path
from omegaconf import OmegaConf
import wandb
from torch_brain.transforms import Compose


class BeemoTrainer(BaselineTrainer):

    def _setup_model(self):
        self.precision = Precision(self.cfg.precision)
        self.neural_session_ids = self._subselect_neural_sessions(
            self.train_dataset.recording_ids,
            self.cfg.data.neural_session_pattern,
            self.cfg.model.args.num_neural_sessions,
        )
        if self.rank == 0:
            OmegaConf.set_struct(self.cfg, False)  # Temporarily allow adding new keys
            self.cfg.num_neural_sessions = (
                None
                if self.neural_session_ids is None
                else len(self.neural_session_ids)
            )
            # Remove 'mp/' prefix from neural_session_ids before updating config
            if self.neural_session_ids is not None:
                neural_session_ids_no_prefix = [
                    s[3:] if s.startswith("mp/") else s for s in self.neural_session_ids
                ]
            else:
                neural_session_ids_no_prefix = None
            self.cfg.neural_session_ids = neural_session_ids_no_prefix
            OmegaConf.set_struct(self.cfg, True)  # Reset struct to True
            wandb.config.update(OmegaConf.to_container(self.cfg, resolve=True))

        self.model = get_class_from_path(self.cfg.model.name)(
            **self.cfg.model.args,
            neural_session_ids=self.neural_session_ids,
        )

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")

        self.model.to(self.device)
        self.model.train()

        self.add_checkpoint_items(
            model=self.model,
        )

    # def _subselect_neural_sessions(self, recording_ids, pattern, num_sessions):
    #     if num_sessions is None:
    #         return None

    #     matches = [id for id in recording_ids if re.match(pattern, id)]
    #     # Remove 'mp/' prefix if present
    #     matches_no_prefix = [id[3:] if id.startswith("mp/") else id for id in matches]
    #     rng = np.random.default_rng(self.cfg.seed)
    #     num_to_sample = min(num_sessions, len(matches_no_prefix))
    #     if num_to_sample == 0:
    #         return []
    #     idxs = rng.choice(len(matches_no_prefix), size=num_to_sample, replace=False)
    #     return [matches[i] for i in idxs]

    def _subselect_neural_sessions(self, recording_ids, pattern, num_sessions):
        if num_sessions is None:
            return None

        matches = [id for id in recording_ids if re.match(pattern, id)]
        if not matches:
            return []

        # mp_prefix_found = any(id.startswith("mp/") for id in matches)
        # matches_no_prefix = [id[3:] if id.startswith("mp/") else id for id in matches]

        matches_no_prefix = matches

        matches_no_prefix.sort()
        rng = np.random.default_rng(self.cfg.seed)
        rng.shuffle(matches_no_prefix)
        num_to_sample = min(num_sessions, len(matches_no_prefix))
        selected = matches_no_prefix[:num_to_sample]
        # if mp_prefix_found:
        #     selected = [f"mp/{id}" for id in selected]
        selected = [f"{id}" for id in selected]

        return selected

    def _setup_dataset(self):
        train_ds = get_class_from_path(self.cfg.data.name)(
            **self.cfg.data.args,
            transform=Compose(
                [
                    get_class_from_path(t.name)(**t.get("args", {}))
                    for t in getattr(self.cfg, "train_transform", [])
                ]
            ),
        )

        data_args = self.cfg.data.args.copy()
        if data_args.get("dataset_list", None) is not None:
            data_args["dataset_list"] = ["mp"]

        eval_ds = get_class_from_path(self.cfg.data.name)(
            **data_args,
            transform=Compose(
                [
                    get_class_from_path(t.name)(**t.get("args", {}))
                    for t in getattr(self.cfg, "eval_transform", [])
                ]
            ),
        )

        self.train_dataset = train_ds
        self.eval_dataset = eval_ds
