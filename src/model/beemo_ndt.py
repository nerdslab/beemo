from typing import Dict, List, Optional
import numpy as np
import torch.nn as nn
from temporaldata import Data
import torch
from torch import nn, Tensor, LongTensor
from einops import repeat

from torch_brain.data import chain
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


class BeemoNDT(nn.Module):
    """
    NDT encoder (bin spikes → unit embeddings → sum-pool → self-attn) feeding
    the same behavior-conditioned cross-attention decoder as BeemoNDT.
    NDT output tokens (one per bin) play the role of POYO latents.
    During test time no behaviour context is used.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        depth: int,
        ctx_duration: float,
        output_dim: int = 2,
        bin_size: float,
        lin_dropout: float,
        atn_dropout: float,
        ffn_mult: int,
        emb_init_scale: float = 0.02,
        dim_head: Optional[int] = None,
        rot_t_min: float = 1e-2,
        rot_t_max: float = 2.0627,
        rot_ratio: float = 0.5,
        decoder_depth: int = 1,
        behaviour_dropout: float = 0.5,
        neural_session_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.ctx_duration = ctx_duration
        self.neural_ctx_duration = 1.0
        self.bin_size = bin_size
        is_divisible(self.neural_ctx_duration, self.bin_size)
        self.num_bins = round(self.neural_ctx_duration / self.bin_size)
        self.output_dim = output_dim

        self.dim_head = dim_head or (dim // num_heads)

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)

        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=self.dim_head,
            rotate_dim=int(self.dim_head * rot_ratio),
            t_min=rot_t_min,
            t_max=rot_t_max,
        )

        # NDT encoder: self-attention on binned spikes
        self.self_attns = nn.ModuleList(
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

        # Output projections + loss
        self.readout = nn.Linear(dim, self.output_dim)

        self.behaviour_dropout = 0.5
        self.neural_session_ids = neural_session_ids

    def forward(
        self,
        unit_idx: LongTensor,
        unit_seqlen: LongTensor,
        bins: Tensor,
        bins_seqlen: LongTensor,
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

        assert len(unit_seqlen) == len(output_seqlen)

        device = unit_idx.device

        # outputs
        query = self.session_emb(output_query_idx)
        query_rot = self.rotary_emb(output_timestamps)

        input_query_idx_ctx = kwargs["input_query_idx_ctx"]

        # ── NDT encoder ───────────────────────────────────────────────────────
        if unit_idx.shape[0] > 0:
            neural_batch_idx = unit_seqlen.nonzero(as_tuple=True)[0]
            spike_batch_size = len(neural_batch_idx)
            bs = len(unit_seqlen)

            # unit embeddings: (total_units, dim)
            uembs = self.unit_emb(unit_idx)
            bins = bins.to(uembs.dtype) / 64.0

            tok = torch.einsum("nt,nd->ntd", bins, uembs)

            # sum unit contributions into each batch slot: (bs, num_bins, dim)
            sum_idx = torch.arange(bs, device=device).repeat_interleave(unit_seqlen)
            x = torch.zeros(bs, self.num_bins, self.dim, device=device, dtype=tok.dtype)
            x = torch.index_add(x, 0, sum_idx, tok)

            # keep only neural batch elements: (spike_batch_size, num_bins, dim)
            x = x[neural_batch_idx]

            # rotary for NDT self-attention (relative to window start)
            t_enc = (
                torch.arange(self.num_bins, device=device, dtype=torch.float32)
                * self.bin_size
            )
            x_rot_3d = repeat(
                self.rotary_emb(t_enc), "T D -> B T D", B=spike_batch_size
            )

            # enc_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            #     q_seqlen=[self.num_bins] * spike_batch_size,
            #     kv_seqlen=[self.num_bins] * spike_batch_size,
            #     device=device,
            # ) # bias is not needed here because input is batched

            for attn, ffn in self.self_attns:  # pyright: ignore
                x = x + attn(x, x_rot_3d)
                x = x + ffn(x)

            # flatten for decoder cross-attention: (spike_batch_size * num_bins, dim)
            latent = x.view(spike_batch_size * self.num_bins, self.dim)

            # latent rotary with ctx window offset
            latent_offset = kwargs.get("latent_offset")[0]
            t_lat = t_enc + latent_offset
            latent_rot = repeat(
                self.rotary_emb(t_lat), "T D -> (B T) D", B=spike_batch_size
            )
            latent_seqlen = [self.num_bins] * spike_batch_size

            if self.training and torch.rand(1, device=device) < self.behaviour_dropout:
                ends = input_seqlen_ctx.cumsum(0)
                starts = ends - input_seqlen_ctx
                starts = starts[neural_batch_idx]
                lengths = input_seqlen_ctx[neural_batch_idx]

                idx = (
                    torch.repeat_interleave(starts, lengths)
                    + torch.arange(lengths.sum(), device=device)
                    - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
                )

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

        if not self.training:
            behav_neural_ctx = query.to(dtype=query.dtype, device=device)
            behav_neural_rot_ctx = query_rot.to(dtype=query_rot.dtype, device=device)

            dec_bias_sa_ctx = xops.fmha.BlockDiagonalMask.from_seqlens(
                q_seqlen=output_seqlen.tolist(),
                kv_seqlen=output_seqlen.tolist(),
                device=device,
            )
            ctx_seqlen = output_seqlen

            if unit_idx.shape[0] > 0:
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

            if unit_idx.shape[0] > 0:
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
            if unit_idx.shape[0] > 0:

                ends = ctx_seqlen.cumsum(0)
                starts = ends - ctx_seqlen
                starts = starts[neural_batch_idx]
                lengths = ctx_seqlen[neural_batch_idx]

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

        if not self.training:
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
        ) and hasattr(data, "spikes"):
            if hasattr(data, "ctx_data"):
                spike_times = (
                    torch.tensor(data_target.spikes.timestamps, dtype=torch.float32)
                    - self.neural_ctx_duration
                )
            else:
                spike_times = torch.tensor(
                    data_target.spikes.timestamps, dtype=torch.float32
                )
            spike_unit_indices = torch.tensor(
                data_target.spikes.unit_index, dtype=torch.long
            )
            num_units = len(data_target.units.id)
            num_bins = int(self.neural_ctx_duration / self.bin_size)

            bins = np.zeros((num_units, num_bins), dtype=np.int16)
            rate = 1 / self.bin_size
            bins = torch.zeros((num_units, num_bins + 1), dtype=torch.int16)
            bins.index_put_(
                indices=(spike_unit_indices, torch.floor(spike_times * rate).long()),
                values=torch.ones(len(spike_times), dtype=torch.int16),
                accumulate=True,
            )
            bins = bins[:, :num_bins]

            unit_idx = unit_emb_idx_map
            unit_seqlen = num_units
        else:
            bins = torch.zeros((0, self.num_bins), dtype=torch.int16)
            unit_idx = np.array([], dtype=unit_emb_idx_map.dtype)
            unit_seqlen = 0
            mask = behav_weights == 1.0
            num_ones = mask.sum()
            if num_ones > 1:
                decay = 0.95 ** np.arange(num_ones)
                # rescale to [1 → 0.2]
                decay = 0.2 + (decay - decay.min()) * (1.0 - 0.2) / (
                    decay.max() - decay.min()
                )
                behav_weights[mask] = decay

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
            "unit_idx": chain(unit_idx),
            "unit_seqlen": unit_seqlen,
            "bins": chain(bins),
            "bins_seqlen": len(bins),
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
