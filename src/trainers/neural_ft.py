import torch
from src.trainers.baseline import BaselineTrainer
from src.utils import move_to_device
from typing import Optional
from src.checkpoint import CheckpointDict


class NeuralFtTrainer(BaselineTrainer):
    """
    Fine-tunes the neural encoder of a Beemo model that was pre-trained
    on behavior only (num_neural_sessions=0).

    Freezes all behavior-trained parameters:
      - session_emb, value_embedding_layer, readout, output_ca, output_ffn
      - tf_blocks_decoder self-attention (index 0) and FFN (index 2)

    Trains only the neural encoder components that were never active:
      - unit_emb, input_ca, input_ffn, tf_blocks, latent_embs, modality_emb
      - tf_blocks_decoder cross-attention (index 1) — the neural→behavior bridge
    """

    # Prefixes of top-level modules that were trained during behavior-only run
    BEHAVIOR_PREFIXES = (
        "session_emb.",
        "value_embedding_layer.",
        "readout.",
        "output_ca.",
        "output_ffn.",
    )

    def setup(self, ckpt):
        self._setup_dataset()
        self._setup_model()
        self._setup_train_loader()
        self._setup_val_loader()
        self._setup_test_loader()
        self.link_model(self.model)
        self._setup_stitcher()
        self._apply_ckpt(ckpt)
        self._init_model_vocab(self.cfg.finetuning.enable)
        self._freeze_behavior_params()
        self._setup_optimizer_and_scheduler()
        self.model = self.make_ddp(self.model)
        self._setup_finetuning()

    def _freeze_behavior_params(self):
        self.frozen_behavior_params = []
        self.trainable_behavior_params_names = []
        for name, param in self.model.named_parameters():
            should_freeze = any(name.startswith(p) for p in self.BEHAVIOR_PREFIXES)

            if name.startswith("unit_emb.") or name.startswith("session_emb."):
                should_freeze = False

            if not should_freeze and name.startswith("tf_blocks_decoder."):
                parts = name.split(".")
                if len(parts) >= 3:
                    layer_idx = int(parts[1])
                    sub_idx = parts[2]
                    num_decoder_layers = len(self.model.tf_blocks_decoder)
                    is_last_layer = layer_idx == num_decoder_layers - 1
                    if is_last_layer:
                        # Last layer: [self_attn(0), ffn(1)] — freeze both
                        if sub_idx in ("0", "1"):
                            should_freeze = True
                    else:
                        # Non-last layers: [ca(0), self_attn(1), ffn(2)] — freeze self_attn and ffn
                        if sub_idx in ("1", "2"):
                            should_freeze = True

            if should_freeze:
                param.requires_grad = False
                self.frozen_behavior_params.append(param)
            else:
                self.trainable_behavior_params_names.append(name)

        self.logger.info(
            f"Frozen {len(self.frozen_behavior_params)} behavior parameter tensors. "
            f"Only neural encoder params will be trained."
        )
        self.logger.info(
            f"Trainable behavior params: {self.trainable_behavior_params_names}"
        )

    # Prefixes to load from checkpoint (behavior-trained weights)
    # Neural encoder keys are left as random init
    BEHAVIOR_LOAD_PREFIXES = (
        "session_emb.",
        "value_embedding_layer.",
        "readout.",
        "output_ca.",
        "output_ffn.",
        "tf_blocks_decoder.",
    )

    def _apply_ckpt(self, ckpt: Optional[CheckpointDict]):
        if ckpt is None:
            if self.cfg.finetuning.enable:
                raise ValueError("No checkpoint provided for finetuning")
            return

        full_sd = ckpt["model_state_dict"]

        # Build a filtered state dict with only behavior-trained keys.
        # Also include unit_emb keys from the current model's own (lazy/uninitialized)
        # state so the InfiniteVocabEmbedding hook doesn't KeyError — the hook checks
        # if the weight is UninitializedParameter and skips loading if so.
        behavior_sd = {
            k: v
            for k, v in full_sd.items()
            if any(k.startswith(p) for p in self.BEHAVIOR_LOAD_PREFIXES)
        }
        for k, v in self.model.state_dict().items():
            if k.startswith("unit_emb."):
                behavior_sd[k] = v  # current lazy weight — hook sees it and skips

        missing, unexpected = self.model.load_state_dict(behavior_sd, strict=False)

        # Verify all behavior keys from checkpoint were loaded
        behavior_keys_in_ckpt = {
            k
            for k in full_sd
            if any(k.startswith(p) for p in self.BEHAVIOR_LOAD_PREFIXES)
        }
        not_loaded = behavior_keys_in_ckpt & set(missing)
        if not_loaded:
            raise RuntimeError(
                f"Behavior keys not loaded (architecture mismatch?): {not_loaded}"
            )

        self.logger.info(
            f"Loaded {len(behavior_sd)} behavior tensors from checkpoint. "
            f"Neural encoder left as random init."
        )

    def _init_model_vocab(self, finetuning: bool = False):
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
            self.model.unit_emb.initialize_vocab(unit_embs)
            self.model.session_emb.extend_vocab(sess_embs)
            self.model.session_emb.subset_vocab(sess_embs)

    def train_epoch(self):
        if (
            self.frozen_behavior_params
            and self.unfreeze_at_epoch > 0
            and self.epoch >= self.unfreeze_at_epoch
        ):
            self._unfreeze_behavior_params()

        self.model.train()
        for batch in self.logger.get_pbar(self.train_loader, "Train"):
            self.scheduler.step()
            batch = move_to_device(batch, self.device)

            with torch.autocast(device_type="cuda", dtype=self.precision.dtype):
                out = self.model(**batch["model_inputs"])

                out = out.float()
                loss = self.readout_spec.loss_fn(
                    input=out,
                    target=batch["target_values"],
                    weights=batch["target_weights"],
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

    def _setup_finetuning(self):
        # self.unfreeze_at_epoch = 0
        self.unfreeze_at_epoch = self.cfg.finetuning.unfreeze_at_epoch

        if self.unfreeze_at_epoch > 0:
            self.logger.info(
                f"Behavior params frozen. Will unfreeze at epoch {self.unfreeze_at_epoch}."
            )
        else:
            self.logger.info(
                "Behavior params frozen for the entire run (unfreeze_at_epoch <= 0)."
            )

    def _unfreeze_behavior_params(self):
        for param in self.frozen_behavior_params:
            param.requires_grad = True
        self.frozen_behavior_params = []
        self.logger.info(f"Epoch {self.epoch}: behavior params unfrozen.")
