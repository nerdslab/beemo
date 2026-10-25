from typing import Dict, List, Optional
import numpy as np
import torch.nn as nn
from torch_brain.data import Data
import torch
from torch import nn, Tensor, LongTensor
from einops import repeat

from torch_brain.batching import chain
from torch_brain.nn import (
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


class BeemoPOYOBehaviorOnly(nn.Module):
    """
    Produces neural latents. For the decoder we are using a more std decoder used in original attn is all u need paper.
    the latents are used as query for the decoder at every layer. fixed the latent timestep. adds 0.5 to the latent times.
    Make behavior optional with dropout. during test time no behaviour is used.
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
        decoder_depth: int = 1,
        behaviour_dropout: float = 0.5,
        neural_session_ids: Optional[List[str]] = None,
        use_behavior_ctx: bool = True,
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

        self.modality_emb = nn.Embedding(2, dim)

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

        # context processing layers

        self.value_embedding_layer = nn.Linear(self.output_dim, dim, bias=False)

        # self.tf_blocks_decoder = nn.ModuleList(
        #     [
        #         nn.ModuleList(
        #             [
        #                 RotarySelfAttention(
        #                     dim=dim,
        #                     heads=num_heads,
        #                     dim_head=self.dim_head,
        #                     atn_dropout=atn_dropout,
        #                     rotate_value=True,
        #                 ),
        #                 RotaryCrossAttention(
        #                     dim=dim,
        #                     ctx_dim=dim,
        #                     heads=num_heads,
        #                     dim_head=self.dim_head,
        #                     atn_dropout=atn_dropout,
        #                     rotate_value=True,
        #                 ),
        #                 GeGLU(dim, ffn_mult, lin_dropout, pre_norm=True),
        #             ]
        #         )
        #         for _ in range(decoder_depth)
        #     ]
        # )

        layers = []
        for _ in range(decoder_depth - 1):
            layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=num_heads,
                            dim_head=self.dim_head,
                            atn_dropout=atn_dropout,
                            rotate_value=True,
                        ),
                        RotaryCrossAttention(
                            dim=dim,
                            ctx_dim=dim,
                            heads=num_heads,
                            dim_head=self.dim_head,
                            atn_dropout=atn_dropout,
                            rotate_value=True,
                        ),
                        GeGLU(dim, ffn_mult, lin_dropout, pre_norm=True),
                    ]
                )
            )
        layers.append(
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
        )

        self.tf_blocks_decoder = nn.ModuleList(layers)

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

        self.behaviour_dropout = behaviour_dropout
        self.neural_session_ids = neural_session_ids
        self.use_behavior_ctx = use_behavior_ctx

    def forward(
        self,
        spike_timestamps: Tensor,
        spike_unit_idx: LongTensor,
        spike_seqlen: LongTensor,
        output_timestamps: Tensor,
        output_query_idx: LongTensor,
        output_seqlen: LongTensor,
        input_timestamps_ctx: Tensor,
        input_seqlen_ctx: LongTensor,
        input_values_ctx: Tensor,
        output_query_idx_slice: LongTensor,
        input_idx_slice: LongTensor,
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
        device = spike_timestamps.device

        # outputs
        query = self.session_emb(output_query_idx)
        query_rot = self.rotary_emb(output_timestamps)

        input_query_idx_ctx = kwargs["input_query_idx_ctx"]

        if spike_unit_idx.shape[0] > 0:
            neural_batch_idx = spike_seqlen.nonzero(as_tuple=True)[0]
            spike_batch_size = len(neural_batch_idx)

            # Set biases
            latent_seqlen = [self.total_latents] * spike_batch_size
            enc_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
                q_seqlen=latent_seqlen,
                kv_seqlen=spike_seqlen[neural_batch_idx].tolist(),
                device=device,
            )

            # Initialize latents
            latent = repeat(
                self.latent_embs.weight,
                "n d -> (b t n) d",
                b=spike_batch_size,
                t=self.latent_steps,
            )
            latent_times = repeat(
                self.latent_times,
                "t -> (b t n)",
                b=spike_batch_size,
                n=self.latents_per_timestep,
            )
            latent_times = latent_times + torch.tensor(
                kwargs.get("latent_offset")[0], device=device
            )
            latent_rot = self.rotary_emb(latent_times)

            # input
            spike_emb = self.unit_emb(spike_unit_idx)
            spike_rot = self.rotary_emb(spike_timestamps)

            # neural_batch_idx = spike_seqlen.nonzero(as_tuple=True)[0]
            # latent_block_size = torch.as_tensor(latent_seqlen, device=device)[0]
            # latent_block_starts = neural_batch_idx * latent_block_size
            # latent_flat_indices = (
            #     latent_block_starts[:, None]
            #     + torch.arange(latent_block_size, device=device)
            # ).reshape(-1)
            # latent = latent[latent_flat_indices]
            # latent_rot = latent_rot[latent_flat_indices]

            # Encode latents and inputs
            latent = latent + self.input_ca(
                latent, spike_emb, latent_rot, spike_rot, enc_bias
            )
            latent = latent + self.input_ffn(latent)
            latent = latent.view(spike_batch_size, self.total_latents, self.dim)
            latent_rot = latent_rot.view(
                spike_batch_size, self.total_latents, 2 * self.dim_head
            )

            # Process latents
            for attn, ffn in self.tf_blocks:  # pyright: ignore
                latent = latent + attn(latent, latent_rot)
                latent = latent + ffn(latent)

            latent = latent.view(spike_batch_size * self.total_latents, self.dim)
            latent_rot = latent_rot.view(
                spike_batch_size * self.total_latents, 2 * self.dim_head
            )

            if self.training and torch.rand(1, device=device) < self.behaviour_dropout:
                # we can drop behavior only when neural data is present

                # compute starts directly (no concat needed)
                ends = input_seqlen_ctx.cumsum(0)
                starts = ends - input_seqlen_ctx

                # keep only non-zero batch elements
                starts = starts[neural_batch_idx]
                lengths = input_seqlen_ctx[neural_batch_idx]

                idx = (
                    torch.repeat_interleave(starts, lengths)
                    + torch.arange(lengths.sum(), device=device)
                    - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
                )

                # ok so these indices will be dropped from the behaviour context
                # get all indxes that are not in idx
                T = input_seqlen_ctx.sum()

                mask = torch.ones(T, dtype=torch.bool, device=device)
                mask[idx] = False

                input_idxs = torch.arange(T, device=device)[mask]
                input_seqlen_ctx = kwargs["input_seqlen_ctx_behaviour_dropout"]
                output_query_idx_slice = kwargs[
                    "output_query_idx_slice_behaviour_dropout"
                ]
                input_idx_slice = kwargs["input_idx_slice_behaviour_dropout"]
                input_timestamps_ctx = input_timestamps_ctx[input_idxs]
                input_values_ctx = input_values_ctx[input_idxs]
                input_query_idx_ctx = input_query_idx_ctx[input_idxs]

        if not self.use_behavior_ctx or len(input_values_ctx) == 0 or not self.training:
            # if not self.use_behavior_ctx:
            behav_neural_ctx = query.to(dtype=query.dtype, device=device)
            behav_neural_rot_ctx = query_rot.to(dtype=query_rot.dtype, device=device)

            dec_bias_sa_ctx = xops.fmha.BlockDiagonalMask.from_seqlens(
                q_seqlen=output_seqlen.tolist(),
                kv_seqlen=output_seqlen.tolist(),
                device=device,
            )
            ctx_seqlen = output_seqlen

            if spike_unit_idx.shape[0] > 0:
                dec_bias_ca_ctx = xops.fmha.BlockDiagonalMask.from_seqlens(
                    q_seqlen=(output_seqlen)[neural_batch_idx].tolist(),
                    kv_seqlen=latent_seqlen,
                    device=device,
                )
        else:
            input_rot_ctx = self.rotary_emb(input_timestamps_ctx)
            enc_inputs_ctx = self.value_embedding_layer(input_values_ctx)
            query_ctx = self.session_emb(input_query_idx_ctx)

            behav_neural_ctx = torch.empty(
                (torch.sum(input_seqlen_ctx + output_seqlen).item(), self.dim),
                device=device,
                dtype=enc_inputs_ctx.dtype,
            )
            behav_neural_rot_ctx = torch.empty(
                (torch.sum(input_seqlen_ctx + output_seqlen).item(), 2 * self.dim_head),
                device=device,
                dtype=input_rot_ctx.dtype,
            )

            behav_neural_ctx[output_query_idx_slice] = query.to(
                dtype=behav_neural_ctx.dtype
            )
            behav_neural_ctx[input_idx_slice] = enc_inputs_ctx.to(
                dtype=behav_neural_ctx.dtype
            )
            behav_neural_rot_ctx[output_query_idx_slice] = query_rot.to(
                dtype=behav_neural_rot_ctx.dtype
            )
            behav_neural_rot_ctx[input_idx_slice] = input_rot_ctx.to(
                dtype=behav_neural_rot_ctx.dtype
            )

            dec_bias_sa_ctx = xops.fmha.BlockDiagonalMask.from_seqlens(
                q_seqlen=(input_seqlen_ctx + output_seqlen).tolist(),
                kv_seqlen=(input_seqlen_ctx + output_seqlen).tolist(),
                device=device,
            )
            ctx_seqlen = input_seqlen_ctx + output_seqlen

            if spike_unit_idx.shape[0] > 0:
                dec_bias_ca_ctx = xops.fmha.BlockDiagonalMask.from_seqlens(
                    q_seqlen=(input_seqlen_ctx + output_seqlen)[
                        neural_batch_idx
                    ].tolist(),
                    kv_seqlen=latent_seqlen,
                    device=device,
                )

        for attn, ca_attn, ffn in self.tf_blocks_decoder[:-1]:  # pyright: ignore
            behav_neural_ctx = behav_neural_ctx + attn(
                behav_neural_ctx, behav_neural_rot_ctx, dec_bias_sa_ctx
            )
            if spike_unit_idx.shape[0] > 0:

                # compute starts directly (no concat needed)
                ends = ctx_seqlen.cumsum(0)
                starts = ends - ctx_seqlen

                # keep only non-zero batch elements
                starts = starts[neural_batch_idx]
                lengths = ctx_seqlen[neural_batch_idx]

                # fully vectorized index construction
                idx = (
                    torch.repeat_interleave(starts, lengths)
                    + torch.arange(lengths.sum(), device=device)
                    - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
                )

                behav_neural_ctx_ca = behav_neural_ctx[idx]
                behav_neural_ctx_ca = behav_neural_ctx_ca + ca_attn(
                    behav_neural_ctx_ca,
                    latent,
                    behav_neural_rot_ctx[idx],
                    latent_rot,
                    dec_bias_ca_ctx,
                )
                behav_neural_ctx[idx] = behav_neural_ctx_ca

            behav_neural_ctx = behav_neural_ctx + ffn(
                behav_neural_ctx,
            )

        attn, ffn = self.tf_blocks_decoder[-1][0], self.tf_blocks_decoder[-1][1]
        behav_neural_ctx = behav_neural_ctx + attn(
            behav_neural_ctx, behav_neural_rot_ctx, dec_bias_sa_ctx
        )
        behav_neural_ctx = behav_neural_ctx + ffn(
            behav_neural_ctx,
        )

        # if not self.use_behavior_ctx:
        if not self.use_behavior_ctx or len(input_values_ctx) == 0 or not self.training:
            query = behav_neural_ctx
        else:
            query = behav_neural_ctx[output_query_idx_slice]

        out = self.readout(query)
        return out

    def tokenize(self, data: Data) -> Dict:

        if hasattr(data, "ctx_data"):
            data_ctx = data.ctx_data
            latent_offset = self.ctx_duration
        else:
            latent_offset = 0

        data_target = data.tgt_data

        unit_emb_idx_map = np.array(self.unit_emb.tokenizer(data_target.units.id))
        sess_emb_idx_map = self.session_emb.tokenizer(data_target.session.id)

        behav_times, behav_gt, behav_weights, eval_mask = prepare_readout(data_target)

        if (
            self.neural_session_ids is None
            or not getattr(self, "training", False)
            or data.session.id in self.neural_session_ids
        ) and hasattr(data_target, "spikes"):
            spike_timestamps = data_target.spikes.timestamps.astype(np.float32)
            spike_unit_idx = unit_emb_idx_map[data_target.spikes.unit_index]
            spike_seqlen = len(spike_timestamps)
        else:
            spike_timestamps = np.array([], dtype=np.float32)
            spike_unit_idx = np.array([], dtype=unit_emb_idx_map.dtype)
            spike_seqlen = 0

            # mask = behav_weights == 1.0
            # num_ones = mask.sum()
            # if num_ones > 1:
            #     decay = 0.95 ** np.arange(num_ones)
            #     # rescale to [1 → 0.2]
            #     decay = 0.2 + (decay - decay.min()) * (1.0 - 0.2) / (
            #         decay.max() - decay.min()
            #     )
            #     behav_weights[mask] = decay

        behav_times = behav_times.astype(np.float32)
        behav_gt = behav_gt.astype(np.float32)
        behav_weights = behav_weights.astype(np.float32)
        behav_query_idx = np.repeat(sess_emb_idx_map, len(behav_times))

        if not hasattr(data, "ctx_data"):
            behav_times_ctx = np.array([], dtype=np.float32)
            behav_gt_ctx = np.array([], dtype=np.float32)
            behav_query_idx_ctx = np.array([], dtype=behav_query_idx.dtype)
        else:

            behav_times_ctx, behav_gt_ctx, _, _ = prepare_readout(data_ctx)
            behav_times_ctx = behav_times_ctx.astype(np.float32)
            behav_gt_ctx = behav_gt_ctx.astype(np.float32)
            behav_query_idx_ctx = np.repeat(sess_emb_idx_map, len(behav_times_ctx))

        model_inputs = {
            "spike_timestamps": chain(spike_timestamps),
            "spike_unit_idx": chain(spike_unit_idx),
            "spike_seqlen": spike_seqlen,
            "output_timestamps": chain(behav_times),
            "output_query_idx": chain(behav_query_idx),
            "output_seqlen": len(behav_times),
            "input_timestamps_ctx": chain(behav_times_ctx),
            "input_seqlen_ctx": len(behav_times_ctx),
            "input_values_ctx": chain(behav_gt_ctx),
            "input_query_idx_ctx": chain(behav_query_idx_ctx),
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
