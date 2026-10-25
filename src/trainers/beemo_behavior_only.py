import torch
import wandb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Optional

matplotlib.use("Agg")

from src.trainers.baseline import BaselineTrainer
from src.checkpoint import CheckpointDict
from src.stitcher import DecodingStitchEvaluator, stitch
from src.distributed import all_gather_object
from src.trainers.beemo import BeemoTrainer
from src.samplers import DistributedSamplerWrapper
from torch_brain.samplers import (
    RandomFixedWindowSampler,
    SequentialFixedWindowSampler,
)
from torch.utils.data import DataLoader
import multiprocessing as mp


class VizDecodingStitchEvaluator(DecodingStitchEvaluator):
    """Like DecodingStitchEvaluator but also stores stitched pred/target for visualization."""

    def compute(self, prefix="val"):
        metrics = {}
        self._stitched_data = {}

        for session_id, metric_fn in self.metrics.items():
            cache = self._cache[session_id]

            if len(cache["pred"]) > 0:
                pred = all_gather_object(torch.cat(cache["pred"]))
                target = all_gather_object(torch.cat(cache["target"]))
                timestamps = all_gather_object(torch.cat(cache["timestamps"]))

                sort_idx = torch.argsort(timestamps)
                timestamps = timestamps[sort_idx]
                pred = pred[sort_idx]
                target = target[sort_idx]

                stitched_timestamps, stitched_pred = stitch(timestamps, pred)
                _, stitched_target = stitch(timestamps, target)

                self._stitched_data[session_id] = {
                    "timestamps": stitched_timestamps.cpu(),
                    "pred": stitched_pred.cpu(),
                    "target": stitched_target.cpu(),
                }

                metric_fn.to(self.device).update(stitched_pred, stitched_target)
            else:
                metric_fn.to(self.device).update(
                    torch.empty(0, self.dim, device=self.device),
                    torch.empty(0, self.dim, device=self.device),
                )

            metrics[session_id] = metric_fn.compute().item()
            metric_fn.reset()

        return metrics


class BeemoBehaviorOnlyTrainer(BeemoTrainer):
    """Samples 2x windows (ctx + pred) matching DataView(ctx_len, pred_len), so
    the model gets real behavior context instead of flat session-embedding
    predictions. Also logs per-session behavior plots to wandb.
    """

    def _setup_train_loader(self):
        sampler = DistributedSamplerWrapper(
            RandomFixedWindowSampler(
                sampling_intervals=self.train_dataset.get_sampling_intervals(
                    split="train"
                ),
                window_length=2 * self.model.ctx_duration,
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

    def _setup_val_loader(self):
        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=self.eval_dataset.get_sampling_intervals("valid"),
                window_length=2 * self.model.ctx_duration,
                step=self.model.ctx_duration,
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
        sampler = DistributedSamplerWrapper(
            SequentialFixedWindowSampler(
                sampling_intervals=self.eval_dataset.get_sampling_intervals("test"),
                window_length=2 * self.model.ctx_duration,
                step=self.model.ctx_duration,
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

    def _setup_stitcher(self):
        session_ids = sorted(self.eval_dataset.recording_ids)
        self.stitcher = VizDecodingStitchEvaluator(
            session_ids=session_ids,
            dim=self.model.output_dim,
            device=self.device,
        )

    @torch.inference_mode()
    def val_epoch(self):
        if self.cfg.debug == "train":
            return

        val_dict = self._eval_epoch(self.val_loader, "Val")
        val_stitched = getattr(self.stitcher, "_stitched_data", {}).copy()

        test_dict = self._eval_epoch(self.test_loader, "Test")
        test_stitched = getattr(self.stitcher, "_stitched_data", {}).copy()

        if (not hasattr(self, "best_val_avg")) or (val_dict["avg"] > self.best_val_avg):
            self.best_val_avg = val_dict["avg"]
            self.logger.log("test/best_on_val", test_dict["avg"])
            self.best_dict_val = {"best/val/" + k: v for k, v in val_dict.items()}
            self.best_dict_test = {"best/test/" + k: v for k, v in test_dict.items()}
            self.best_dict = (
                self.best_dict_val | self.best_dict_test | {"best/epoch": self.epoch}
            )
            self.logger.log_dict(self.best_dict)

        val_dict = {"val/" + k: v for k, v in val_dict.items()}
        test_dict = {"test/" + k: v for k, v in test_dict.items()}
        self.logger.log_dict(val_dict)
        self.logger.log_dict(test_dict)

        if not self.logger.dummy:
            viz = self._make_behavior_viz(val_stitched, "val")
            viz.update(self._make_behavior_viz(test_stitched, "test"))
            if viz:
                self.logger.log_dict(viz)

        self.push_logs()
        self.return_value = val_dict | test_dict | self.best_dict

    def _make_behavior_viz(
        self,
        stitched_data: Dict,
        split: str,
        max_duration: Optional[float] = 40,
    ) -> Dict:
        """Plot pred vs target time-series for all sessions; return dict of wandb.Images."""
        if not stitched_data:
            return {}

        images = {}
        for session_id in sorted(stitched_data.keys()):
            data = stitched_data[session_id]
            timestamps = data["timestamps"].numpy()
            pred = data["pred"].numpy()
            target = data["target"].numpy()

            if max_duration is not None:
                total_duration = timestamps[-1] - timestamps[0]
                max_start = max(0.0, float(total_duration - max_duration))
                start = np.random.uniform(0.0, max_start)
                mask = (timestamps - timestamps[0] >= start) & (
                    timestamps - timestamps[0] <= start + max_duration
                )
                timestamps = timestamps[mask]
                pred = pred[mask]
                target = target[mask]

            if len(timestamps) == 0:
                continue

            t = timestamps - timestamps[0]

            n_dims = pred.shape[1] if pred.ndim > 1 else 1
            dim_labels = (
                ["vx", "vy"] if n_dims == 2 else [f"dim {i}" for i in range(n_dims)]
            )

            fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims), squeeze=False)
            for i, ax in enumerate(axes[:, 0]):
                p = pred[:, i] if pred.ndim > 1 else pred
                tgt = target[:, i] if target.ndim > 1 else target
                ax.plot(t, tgt, label="target", linewidth=1.0, alpha=0.85)
                ax.plot(t, p, label="pred", linewidth=1.0, alpha=0.85, linestyle="--")
                ax.set_ylabel(dim_labels[i])
                ax.legend(loc="upper right", fontsize=8)
            axes[-1, 0].set_xlabel("time (s)")

            short_id = session_id.split("/")[-1]
            fig.suptitle(f"{split} | {short_id}", fontsize=10)
            fig.tight_layout()

            images[f"{split}/viz/{short_id}"] = wandb.Image(fig)
            plt.close(fig)

        return images
