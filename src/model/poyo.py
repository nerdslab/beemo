# --------------------------------------------------
# Code adapted from altspiketok
# Author: Vinam Arora
# --------------------------------------------------

from typing import Dict, Optional
import numpy as np
import torch.nn as nn
from temporaldata import Data
import torch
from torch import nn, Tensor, LongTensor
from einops import repeat

from torch_brain.data import chain
from torch_brain.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    RotaryTimeEmbedding,
)
from src.nn.attention import (
    RotaryCrossAttention,
    RotarySelfAttention,
)
from src.nn.ffn import GeGLU
from src.readout import prepare_readout
import xformers.ops as xops
from src.utils import is_divisible


class POYO(nn.Module):
    """
    Baseline POYO model for behavior prediction. Takes in neural activity of current context and predicts behavior for the current context.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        depth: int,
        ctx_duration: float,
        output_dim: int = 2,
        latent_timestep: float,
        latents_per_timestep: int,
        lin_dropout: float,
        atn_dropout: float,
        ffn_mult: int,
        emb_init_scale: float = 0.02,
        dim_head: Optional[int] = None,
        rot_t_min: float = 1e-4,
        rot_t_max: float = 2.0627,
        rot_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.ctx_duration = ctx_duration
        self.latent_timestep = latent_timestep
        self.latents_per_timestep = latents_per_timestep
        is_divisible(self.ctx_duration, self.latent_timestep)
        self.latent_steps = round(self.ctx_duration / self.latent_timestep)
        self.total_latents = self.latent_steps * self.latents_per_timestep
        self.output_dim = output_dim

        self.dim_head = dim_head or (dim // num_heads)

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)

        self.latent_times = nn.Parameter(
            (
                (torch.arange(self.latent_steps) * self.latent_timestep)
                + (self.latent_timestep / 2)
            ),
            requires_grad=False,
        )
        self.latent_embs = nn.Embedding(self.latents_per_timestep, self.dim)
        self.latent_embs.weight.data.mul_(emb_init_scale)

        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=self.dim_head,
            rotate_dim=int(self.dim_head * rot_ratio),
            t_min=rot_t_min,
            t_max=rot_t_max,
        )

        # self.dropout = nn.Dropout(p=lin_dropout)

        # encoder layer
        self.input_ca = RotaryCrossAttention(
            dim=dim,
            ctx_dim=dim,
            heads=num_heads,
            dim_head=self.dim_head,
            atn_dropout=atn_dropout,
            rotate_value=True,
        )

        self.input_ffn = GeGLU(dim, ffn_mult, lin_dropout, pre_norm=True)

        # process layers
        self.tf_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=num_heads,
                            dim_head=self.dim_head,
                            atn_dropout=atn_dropout,
                            rotate_value=True,
                        ),
                        GeGLU(dim, ffn_mult, lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(depth)
            ]
        )

        # decoder layer
        self.output_ca = RotaryCrossAttention(
            dim=dim,
            ctx_dim=dim,
            heads=num_heads,
            dim_head=self.dim_head,
            atn_dropout=atn_dropout,
            rotate_value=True,
        )
        self.output_ffn = GeGLU(dim, ffn_mult, lin_dropout, pre_norm=True)

        # Output projections + loss
        self.readout = nn.Linear(dim, self.output_dim)

    def forward(
        self,
        spike_timestamps: Tensor,
        spike_unit_idx: LongTensor,
        spike_seqlen: LongTensor,
        output_timestamps: Tensor,
        output_query_idx: LongTensor,
        output_seqlen: LongTensor,
        **kwargs,
    ):

        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary has not been initialized, please use "
                "`model.unit_emb.initialize_vocab(unit_ids)`"
            )

        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        # set asserts
        assert len(spike_seqlen) == len(output_seqlen)
        assert spike_seqlen.sum() == len(spike_timestamps) == len(spike_unit_idx)
        assert output_seqlen.sum() == len(output_timestamps) == len(output_query_idx)

        # Set variables
        batch_size = len(spike_seqlen)
        device = spike_timestamps.device

        # Set biases
        latent_seqlen = [self.total_latents] * batch_size
        enc_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=latent_seqlen,
            kv_seqlen=spike_seqlen.tolist(),
            device=device,
        )
        dec_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=output_seqlen.tolist(),
            kv_seqlen=latent_seqlen,
            device=device,
        )

        # Initialize latents
        latent = repeat(
            self.latent_embs.weight,
            "n d -> (b t n) d",
            b=batch_size,
            t=self.latent_steps,
        )
        latent_times = repeat(
            self.latent_times, "t -> (b t n)", b=batch_size, n=self.latents_per_timestep
        )
        latent_times = latent_times + torch.tensor(
            kwargs.get("latent_offset")[0], device=device
        )
        latent_rot = self.rotary_emb(latent_times)

        # input
        spike_emb = self.unit_emb(spike_unit_idx)
        spike_rot = self.rotary_emb(spike_timestamps)

        # outputs
        query = self.session_emb(output_query_idx)
        query_rot = self.rotary_emb(output_timestamps)

        # Encode latents and inputs
        latent = latent + self.input_ca(
            latent, spike_emb, latent_rot, spike_rot, enc_bias
        )
        latent = latent + self.input_ffn(latent)
        latent = latent.view(batch_size, self.total_latents, self.dim)
        latent_rot = latent_rot.view(batch_size, self.total_latents, 2 * self.dim_head)

        # Process latents
        for attn, ffn in self.tf_blocks:  # pyright: ignore
            latent = latent + attn(latent, latent_rot)
            latent = latent + ffn(latent)

        latent = latent.view(batch_size * self.total_latents, self.dim)
        latent_rot = latent_rot.view(batch_size * self.total_latents, 2 * self.dim_head)

        # Decode latents and outputs
        query = query + self.output_ca(query, latent, query_rot, latent_rot, dec_bias)
        query = query + self.output_ffn(query)

        out = self.readout(query)
        return out

    def tokenize(self, data: Data) -> Dict:

        if hasattr(data, "ctx_data"):
            latent_offset = self.ctx_duration
        else:
            latent_offset = 0

        data = data.tgt_data

        unit_emb_idx_map = np.array(self.unit_emb.tokenizer(data.units.id))
        sess_emb_idx_map = self.session_emb.tokenizer(data.session.id)

        spike_timestamps = data.spikes.timestamps.astype(np.float32)
        spike_unit_idx = unit_emb_idx_map[data.spikes.unit_index]
        spike_seqlen = len(spike_timestamps)

        behav_times, behav_gt, behav_weights, eval_mask = prepare_readout(data)
        behav_times = behav_times.astype(np.float32)
        behav_gt = behav_gt.astype(np.float32)
        behav_weights = behav_weights.astype(np.float32)
        behav_query_idx = np.repeat(sess_emb_idx_map, len(behav_times))

        model_inputs = {
            "spike_timestamps": chain(spike_timestamps),
            "spike_unit_idx": chain(spike_unit_idx),
            "spike_seqlen": spike_seqlen,
            "output_timestamps": chain(behav_times),
            "output_query_idx": chain(behav_query_idx),
            "output_seqlen": len(behav_times),
            "latent_offset": chain(latent_offset),
        }

        data_dict = {
            "model_inputs": model_inputs,
            "target_values": chain(behav_gt),
            "target_weights": chain(behav_weights),
            "eval_mask": chain(eval_mask),
            "absolute_start": data._absolute_start,
            "session_id": data.session.id,
        }

        return data_dict

    DIMS = [64, 96, 128, 192, 256]

    @classmethod
    def create_search_space(cls, trial, cfg):
        trial.suggest_int("dim_idx", 0, len(cls.DIMS) - 1)
        trial.suggest_int("num_heads_log2", 0, 3)
        trial.suggest_int("model.args.depth", 0, 8)
        trial.suggest_float("dropout", 0.0, 0.9, step=0.1)

        trial.suggest_int("num_epochs", 100, 1000, step=100)
        trial.suggest_int("batch_size_log2", 3, 7)
        trial.suggest_float("optimizer.args.lr", 1e-5, 1e-2, log=True)
        trial.suggest_int("wd_log10", -3, -1)

    @classmethod
    def process_tunable_params(cls, tune_params: dict) -> dict:
        import optuna

        dim = cls.DIMS[tune_params.pop("dim_idx")]
        num_heads = 2 ** tune_params.pop("num_heads_log2")
        if (dim // num_heads) % 8 != 0:
            raise optuna.exceptions.TrialPruned()

        tune_params["model.args.dim"] = dim
        tune_params["model.args.num_heads"] = num_heads
        tune_params["batch_size"] = 2 ** tune_params.pop("batch_size_log2")
        tune_params["optimizer.args.weight_decay"] = 10.0 ** tune_params.pop("wd_log10")

        dropout = tune_params.pop("dropout")
        tune_params["model.args.lin_dropout"] = dropout
        tune_params["model.args.atn_dropout"] = dropout
        return tune_params
