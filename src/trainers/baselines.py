import torch
from omegaconf import OmegaConf

from src.checkpoint import CheckpointDict
from src.trainers.baseline import BaselineTrainer
from src.utils import infer_behavior_step, move_to_device


class BaselinesTrainer(BaselineTrainer):
    def setup(self, ckpt: CheckpointDict | None):
        self._setup_dataset()
        self._setup_model()

        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_test_loader()

        self.link_model(self.model)
        self._setup_stitcher()

        self._apply_ckpt(ckpt)
        # _init_model_vocab skipped — MLP has no unit/session vocab
        self._materialize_lazy_params()
        self._setup_optimizer_and_scheduler()
        self.model = self.make_ddp(self.model)
        self._setup_finetuning()

    def _setup_model(self):
        r"""Resolve ``out_step`` from the data, then build as usual.

        It is a property of the brainset, not a hyperparameter. Written back into
        the config so the resolved value is what gets logged and checkpointed.
        """
        if self.cfg.model.args.get("out_step", None) is not None:
            raise ValueError(
                f"out_step ({self.cfg.model.args.out_step}) is inferred from the "
                "data, not a hyperparameter; remove it from model.args."
            )

        out_step = infer_behavior_step(self.train_dataset, self.readout_spec)
        OmegaConf.set_struct(self.cfg.model.args, False)
        self.cfg.model.args.out_step = out_step
        OmegaConf.set_struct(self.cfg.model.args, True)
        self.logger.info(
            f"Inferred out_step={out_step:.6g}s ({1.0 / out_step:.1f} Hz behavior)"
        )

        super()._setup_model()

    def _materialize_lazy_params(self):
        batch = next(iter(self.train_loader))
        batch = move_to_device(batch, self.device)
        with torch.no_grad():
            self.model(**batch["model_inputs"])
