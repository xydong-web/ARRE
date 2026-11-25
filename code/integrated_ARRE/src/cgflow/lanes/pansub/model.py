"""Lane A / PAN-Sub: the policy network.

`PANRxnFlow` has exactly the same parameters (and therefore the same `state_dict`) as the
baseline `RxnFlow`; it only chooses which action categorical to build.  Baseline checkpoints
load into it unchanged (tests/lane_a/test_state_dict.py).
"""

from __future__ import annotations

import torch_geometric.data as gd
import torch
from torch import Tensor

from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical

from .categorical import PANActionCategorical
from .pan import PanEngine


class PANRxnFlow(RxnFlow):
    """RxnFlow + an optional PAN action categorical.

    The PAN runtime is attached *after* the trainer has deep-copied the model into the EMA
    sampling model (`attach_pan`), so that the 1.3 GB embedding cache is never duplicated.
    """

    def attach_pan(self, engine: PanEngine | None, role: str = "train") -> None:
        self._pan_engine = engine
        self._pan_role = role

    @property
    def pan_engine(self) -> PanEngine | None:
        return getattr(self, "_pan_engine", None)

    def _make_cat(self, g: gd.Batch, emb: Tensor) -> RxnActionCategorical:
        action_masks = list(torch.unbind(g.protocol_mask, dim=1))
        engine = self.pan_engine
        if engine is None or not engine.cfg.enable:
            return RxnActionCategorical(g, emb, action_masks=action_masks, model=self)
        return PANActionCategorical(
            g, emb, action_masks=action_masks, model=self, engine=engine, role=getattr(self, "_pan_role", "train")
        )
