import numpy as np
import pandas as pd
from torch_brain.data import Interval
from typing import Optional, Literal
from torch_brain.batching import collate as tbrain_collate
from torch_brain.data import Data
import torch

from torch_brain.datasets import NestedSpikingDataset as BaseNestedSpikingDataset


def calc_out_idx_slice(ctx, out):
    total = ctx + out

    offsets = torch.cat(
        [
            torch.zeros(1, dtype=total.dtype, device=total.device),
            torch.cumsum(total[:-1], dim=0),
        ]
    )

    starts = offsets + ctx

    output_query_idxs = (
        torch.repeat_interleave(starts, out)
        + torch.arange(out.sum(), device=ctx.device)
        - torch.repeat_interleave(torch.cumsum(out, 0) - out, out)
    )

    T = total.sum()

    mask = torch.ones(T, dtype=torch.bool, device=ctx.device)
    mask[output_query_idxs] = False

    input_idxs = torch.arange(T, device=ctx.device)[mask]

    return output_query_idxs, input_idxs


class NestedSpikingDataset(BaseNestedSpikingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_type = "causal"

    def get_session_ids(self):
        return [self.get_recording(rid).session.id for rid in self.recording_ids]

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
        collated_data = NestedSpikingDataset.process_batch(collated_data)
        return collated_data
