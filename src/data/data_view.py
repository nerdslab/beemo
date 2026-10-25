# --------------------------------------------------
# Code adapted from neuro4cast
# Author: Shivasriganesh Mahato and Alexandre André
# --------------------------------------------------

import math
from typing import Optional

from temporaldata import Data


class DataView:
    def __init__(
        self,
        ctx_len: int,
        pred_len: int,
        feature_fields: Optional[str] = None,
        feature_style: Optional[str] = None,
        reset_origin: bool = False,
        test_mode: bool = False,
    ):
        self.ctx_len = ctx_len
        self.pred_len = pred_len
        self.test_mode = test_mode
        self.reset_origin = reset_origin

    def __call__(self, data: Data):
        # if not math.isclose(
        #     data.end - data.start, self.ctx_len + self.pred_len, rel_tol=1e-6
        # ):  # if not enough data that means we cannot use behavior so we will just have target data for neural data
        if data.end - data.start < self.ctx_len + self.pred_len:
            tgt_data = data.slice(
                0, self.ctx_len, reset_origin=self.reset_origin
            ).materialize()

            setattr(data, "tgt_data", tgt_data)
            return data

        ctx_data = data.slice(
            0, self.ctx_len, reset_origin=self.reset_origin
        ).materialize()
        tgt_data = data.slice(
            self.ctx_len, self.ctx_len + self.pred_len, reset_origin=self.reset_origin
        ).materialize()
        # tgt_data = data.slice(self.ctx_len, self.ctx_len + self.pred_len).materialize()
        # if self.test_mode:
        #     tgt_data = data.slice(0, self.ctx_len).materialize()

        # else:
        #     tgt_data = data.slice(
        #         self.ctx_len, self.ctx_len + self.pred_len
        #     ).materialize()

        setattr(data, "ctx_data", ctx_data)
        setattr(data, "tgt_data", tgt_data)

        return data
