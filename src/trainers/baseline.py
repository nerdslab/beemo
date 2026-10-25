import torch
from typing import Optional
from src.trainers.trainer import Trainer
from src.checkpoint import CheckpointDict
from src.utils import (
    Precision,
    get_class_from_path,
    get_num_params,
    get_num_trainable_params,
    move_to_device,
)
from src.stitcher import DecodingStitchEvaluator
from src.samplers import DistributedSamplerWrapper
from src.readout import weighted_mse_loss
from src.samplers import (
    RandomFixedWindowBehaviorSampler,
    SequentialFixedWindowBehaviorSampler,
)
from torch_brain.data.sampler import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch_brain.data import Dataset, collate
from torch_brain.transforms import Compose
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np
import multiprocessing as mp
from torch import nn
from src.finetuning_utils import GradualUnfreezing


class BaselineTrainer(Trainer):
    def setup(self, ckpt: Optional[CheckpointDict]):

        self._setup_dataset()
        self._setup_model()

        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_test_loader()

        self.link_model(self.model)
        self._setup_stitcher()

        self._apply_ckpt(ckpt)
        self._init_model_vocab(self.cfg.finetuning.enable)
        self._setup_optimizer_and_scheduler()
        self.model = self.make_ddp(self.model)
        self._setup_finetuning()

    def train_epoch(self):
        if self.cfg.finetuning.enable:
            self.gradual_unfreezing.unfreeze(self.model, self.epoch)

        self.model.train()
        for batch in self.logger.get_pbar(self.train_loader, "Train"):
            self.scheduler.step()
            batch = move_to_device(batch, self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                out = self.model(**batch["model_inputs"])

                out = out.float()
                loss = weighted_mse_loss(
                    out, batch["target_values"], batch["target_weights"]
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.log("train/loss", loss.item(), pbar="Loss")
            self.log_weight_stats()
            self.log_grad_stats()
            self.log_lr()
            self.push_logs()
            self.train_step += 1

            if self.cfg.debug == "eval":
                break

    @torch.inference_mode()
    def val_epoch(self):
        if self.cfg.debug == "train":
            return

        val_dict = self._eval_epoch(self.val_loader, "Val")
        test_dict = self._eval_epoch(self.test_loader, "Test")

        if (not hasattr(self, "best_val_avg")) or (val_dict["avg"] > self.best_val_avg):
            self.best_val_avg = val_dict["avg"]
            self.logger.log("test/best_on_val", test_dict["avg"])
            self.best_dict_val = val_dict.copy()
            self.best_dict_test = test_dict.copy()
            self.best_dict_val = {
                "best/val/" + k: v for k, v in self.best_dict_val.items()
            }
            self.best_dict_test = {
                "best/test/" + k: v for k, v in self.best_dict_test.items()
            }
            self.best_dict = (
                self.best_dict_val | self.best_dict_test | {"best/epoch": self.epoch}
            )
            self.logger.log_dict(self.best_dict)

        val_dict = {"val/" + k: v for k, v in val_dict.items()}
        test_dict = {"test/" + k: v for k, v in test_dict.items()}
        self.logger.log_dict(val_dict)
        self.logger.log_dict(test_dict)
        self.push_logs()

        self.return_value = val_dict | test_dict | self.best_dict

    def _eval_epoch(self, loader, name):
        self.model.eval()
        self.stitcher.reset()

        for batch in self.logger.get_pbar(loader, name):
            batch = move_to_device(batch, self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                out: Tensor = self.model(**batch["model_inputs"])  # pyright: ignore

            behav_seqlen = batch["model_inputs"]["output_seqlen"].tolist()
            self.stitcher.update(
                timestamps=batch["model_inputs"]["output_timestamps"].split(
                    behav_seqlen
                ),
                preds=out.float().split(behav_seqlen),  # type: ignore
                targets=batch["target_values"].split(behav_seqlen),
                eval_masks=batch["eval_mask"].split(behav_seqlen),
                session_ids=batch["session_id"],
                absolute_starts=batch["absolute_start"],
            )

        self.barrier()
        metric_dict = self.stitcher.compute()
        metric_dict = {k.split("/")[-1]: v for k, v in metric_dict.items()}
        metric_dict["avg"] = np.mean(list(metric_dict.values()))
        return metric_dict

    def _setup_dataset(self):
        if not hasattr(self.cfg, "test_size"):
            test_size = 0.2
            valid_size = 0.1
        else:
            valid_size = self.cfg.valid_size
            test_size = self.cfg.test_size
        train_ds = get_class_from_path(self.cfg.data.name)(
            **self.cfg.data.args,
            transform=Compose(
                [
                    get_class_from_path(t.name)(**t.get("args", {}))
                    for t in getattr(self.cfg, "train_transform", [])
                ]
            ),
            test_size=test_size,
            valid_size=valid_size,
        )

        eval_ds = get_class_from_path(self.cfg.data.name)(
            **self.cfg.data.args,
            transform=Compose(
                [
                    get_class_from_path(t.name)(**t.get("args", {}))
                    for t in getattr(self.cfg, "eval_transform", [])
                ]
            ),
            test_size=test_size,
            valid_size=valid_size,
        )

        self.train_dataset = train_ds
        self.eval_dataset = eval_ds

    def _setup_model(self):
        self.precision = Precision(self.cfg.precision)

        self.model = get_class_from_path(self.cfg.model.name)(**self.cfg.model.args)

        self.logger.info(f"Precision: {self.precision}")
        self.logger.info(f"Model: {self.model.__class__}")

        self.model.to(self.device)
        self.model.train()

        self.add_checkpoint_items(
            model=self.model,
        )

    def link_model(self, model: nn.Module):
        r"""Setup Dataset objects, and update a given model's embedding vocabs (session
        and unit_emb)
        """
        self.train_dataset.transform.transforms.append(model.tokenize)
        self.eval_dataset.transform.transforms.append(model.tokenize)

    def _setup_optimizer_and_scheduler(self):
        decay = []
        decay_names = []

        no_decay = []
        no_decay_names = []

        def update_params(module: torch.nn.Module):
            for name, param in module.named_parameters():
                if "norm" in name or "bias" in name or "emb" in name:
                    no_decay.append(param)
                    no_decay_names.append(name)
                else:
                    decay.append(param)
                    decay_names.append(name)

        update_params(self.model)

        params = [
            {
                "params": decay,
                "weight_decay": self.cfg.optimizer.args.weight_decay,
                "names": decay_names,  # ← extra metadata
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "names": no_decay_names,  # ← extra metadata
            },
        ]

        self.cfg.optimizer.args.lr = float(
            np.sqrt(self.cfg.batch_size).astype(np.float32)
        ) * float(self.cfg.optimizer.args.lr)

        self.optimizer: torch.optim.Optimizer = get_class_from_path(
            self.cfg.optimizer.name
        )(**self.cfg.optimizer.args, params=params)

        self.logger.info(f"Optimizer: {self.optimizer.__class__}")
        num_params = get_num_params(self.model)
        self.logger.info(f"Num params: {num_params:,}")

        num_trainable_params = get_num_trainable_params(self.model)
        self.logger.info(f"Num trainable params: {num_trainable_params:,}")

        self.add_checkpoint_items(
            optimizer=self.optimizer,
        )

        if hasattr(self.cfg, "scheduler"):

            self.scheduler = get_class_from_path(self.cfg.scheduler.name)(
                **self.cfg.scheduler.args,
                optimizer=self.optimizer,
                max_lr=self.cfg.optimizer.args.lr,
                total_steps=self.steps_per_epoch * self.cfg.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
            )
            self.logger.info(f"Scheduler: {self.scheduler.__class__}")
            self.add_checkpoint_items(
                scheduler=self.scheduler,
            )

    def _setup_train_loader(self):

        if not hasattr(self.cfg, "test_size"):
            test_size = 0.2
            valid_size = 0.1
        else:
            valid_size = self.cfg.valid_size
            test_size = self.cfg.test_size

        sampler = DistributedSamplerWrapper(
            RandomFixedWindowBehaviorSampler(
                sampling_intervals=self.train_dataset.get_sampling_intervals(
                    split="train",
                    test_size=test_size,
                    valid_size=valid_size,
                ),
                window_length=self.model.ctx_duration,  # TODO: Make this dynamic
                generator=torch.Generator().manual_seed(self.cfg.seed + 1),
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            collate_fn=self.train_dataset.collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
            drop_last=True,
        )
        self.steps_per_epoch = len(self.train_loader)

        self.logger.info(f"Train steps/epoch: {self.steps_per_epoch}")
        self.logger.info(f"Training on {len(self.get_unit_ids())} units")
        self.logger.info(f"Training on {len(self.get_session_ids())} sessions")

    def _init_model_vocab(self, finetuning: bool = False):
        # finetuning = False
        unit_embs = sorted(self.get_unit_ids())
        sess_embs = sorted(self.get_session_ids())

        if not finetuning:
            self.model.unit_emb.initialize_vocab(unit_embs)
            self.model.session_emb.initialize_vocab(sess_embs)
            # if hasattr(self.model, "task_emb"):
            #     self.model.task_emb.initialize_vocab(
            #         ["center_out_reaching", "random_target_reaching", "odoherty"]
            #     )
        else:
            self.model.unit_emb.extend_vocab(unit_embs)
            self.model.unit_emb.subset_vocab(unit_embs)
            # if hasattr(self.model, "task_emb"):
            #     self.model.task_emb.extend_vocab(["maze"])
            #     self.model.task_emb.subset_vocab(["maze"])
            self.model.session_emb.extend_vocab(sess_embs)
            self.model.session_emb.subset_vocab(sess_embs)

    def get_session_ids(self):
        return self.train_dataset.recording_ids

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def _setup_val_loader(self):
        if not hasattr(self.cfg, "test_size"):
            test_size = 0.2
            valid_size = 0.1
        else:
            valid_size = self.cfg.valid_size
            test_size = self.cfg.test_size
        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=self.eval_dataset.get_sampling_intervals(
                    "valid",
                    test_size=test_size,
                    valid_size=valid_size,
                ),
                window_length=self.model.ctx_duration,  # TODO: Make this dynamic
                step=self.model.ctx_duration / 2,
                drop_short=True,
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.val_loader = DataLoader(
            dataset=self.eval_dataset,
            sampler=sampler,
            collate_fn=self.eval_dataset.collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
            drop_last=False,
        )

        self.logger.info(f"Val steps/epoch: {len(self.val_loader)}")

    def _setup_test_loader(self):
        if not hasattr(self.cfg, "test_size"):
            test_size = 0.2
            valid_size = 0.1
        else:
            valid_size = self.cfg.valid_size
            test_size = self.cfg.test_size
        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=self.eval_dataset.get_sampling_intervals(
                    "test", test_size=test_size, valid_size=valid_size
                ),
                window_length=self.model.ctx_duration,  # TODO: Make this dynamic
                step=self.model.ctx_duration / 2,
                drop_short=True,
            )
        )

        multi_wrkrs = self.cfg.num_workers > 0
        self.test_loader = DataLoader(
            dataset=self.eval_dataset,
            sampler=sampler,
            collate_fn=self.eval_dataset.collate,
            batch_size=self.cfg.batch_size // self.world_size,
            num_workers=self.cfg.num_workers,
            multiprocessing_context=mp.get_context("fork") if multi_wrkrs else None,
            persistent_workers=multi_wrkrs,
            drop_last=False,
        )

        self.logger.info(f"Test steps/epoch: {len(self.test_loader)}")

    def _apply_ckpt(self, ckpt: Optional[CheckpointDict]):

        if ckpt is None:
            if self.cfg.finetuning.enable:
                raise ValueError("No checkpoint provided for finetuning")
            return

        self.model.load_state_dict(ckpt["model_state_dict"])
        # if self.cfg.ckpt.resume:
        #     self.loss.load_state_dict(ckpt["loss_state_dict"])
        #     self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # elif self.cfg.ckpt.model_only:
        #     self.loss.load_state_dict(ckpt["loss_state_dict"])

    def _setup_stitcher(self):
        session_ids = sorted(self.eval_dataset.recording_ids)
        self.stitcher = DecodingStitchEvaluator(
            session_ids=session_ids,
            dim=self.model.output_dim,
            device=self.device,
        )

    def _setup_finetuning(self):
        if self.cfg.finetuning.enable:
            self.gradual_unfreezing = GradualUnfreezing(
                self.cfg.finetuning.unfreeze_at_epoch, self.model
            )
            self.logger.info(
                f"Model frozen at epoch 0. "
                f"Will stay frozen until epoch {self.gradual_unfreezing.unfreeze_at_epoch}."
            )
