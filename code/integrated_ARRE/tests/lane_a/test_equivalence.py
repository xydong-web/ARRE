"""T3/T4 - default-off equivalence with the baseline, and checkpoint compatibility."""

from __future__ import annotations

import copy

import torch

from gflownet.utils.misc import set_worker_rng_seed
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import set_worker_env

from cgflow.lanes.config import LaneAConfig
from cgflow.lanes.pansub.model import PANRxnFlow
from cgflow.lanes.pansub.pan import PanEngine
from cgflow.lanes.pansub.categorical import PANActionCategorical

from .realenv import FakeBatch, base_cfg, load_env, mask_for_pattern

DEV = torch.device("cpu")


def _setup(sampling_ratio=0.01):
    env, ctx = load_env()
    cfg = base_cfg(sampling_ratio)
    set_worker_env("ctx", ctx)
    set_worker_env("env", env)
    set_worker_env("action_subsampler", SubsamplingPolicy(env, cfg))
    base = RxnFlow(ctx, cfg, num_graph_out=1)
    lane = PANRxnFlow(ctx, cfg, num_graph_out=1)
    lane.load_state_dict(base.state_dict())
    return env, ctx, cfg, base, lane


def _batch(ctx, n=6, pattern=1, level=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = mask_for_pattern(ctx, pattern).unsqueeze(0).repeat(n, 1)
    lvl = torch.full((n, 1), float(level))
    emb = torch.randn((n, 128), generator=g)
    return FakeBatch(mask, lvl), emb


def test_state_dict_is_identical_to_baseline():
    env, ctx, cfg, base, lane = _setup()
    assert list(base.state_dict().keys()) == list(lane.state_dict().keys())
    for k, v in base.state_dict().items():
        assert v.shape == lane.state_dict()[k].shape
    missing, unexpected = lane.load_state_dict(base.state_dict(), strict=True)
    assert not missing and not unexpected
    # and a PAN model's state dict loads back into a pristine baseline model
    RxnFlow(ctx, cfg, num_graph_out=1).load_state_dict(lane.state_dict(), strict=True)


def _run_pair(ctx, base, lane, engine_cfg: LaneAConfig | None, seed=7):
    """Build the baseline and the lane categorical from the same RNG state and compare."""
    graphs, emb = _batch(ctx)
    out = []
    for model, use_engine in ((base, False), (lane, True)):
        set_worker_rng_seed(123)
        torch.manual_seed(seed)
        if use_engine and engine_cfg is not None:
            engine = PanEngine(engine_cfg, ctx, DEV, num_emb_block=64, seed=0)
            engine.cache.refresh(model, 0)
            model.attach_pan(engine, "train")
        elif use_engine:
            model.attach_pan(None, "train")
        cat = model._make_cat(graphs, emb)
        torch.manual_seed(seed + 1)
        actions = cat.sample()
        lp = cat.log_prob(actions)
        out.append((cat, actions, lp))
    return out


def test_disabled_lane_is_bitwise_baseline():
    env, ctx, cfg, base, lane = _setup()
    (c0, a0, l0), (c1, a1, l1) = _run_pair(ctx, base, lane, None)
    assert isinstance(c0, RxnActionCategorical) and not isinstance(c0, PANActionCategorical)
    assert type(c1) is RxnActionCategorical
    assert a0 == a1
    assert torch.equal(l0, l1)
    for x, y in zip(c0.logits, c1.logits, strict=True):
        assert torch.equal(x, y)


def test_uniform_subsample_switches_are_baseline():
    """enable=true but normalizer=uniform & behavior=subsample must still be the baseline."""
    env, ctx, cfg, base, lane = _setup()
    la = LaneAConfig(enable=True, normalizer="uniform", behavior="subsample", cache_dtype="float32",
                     table_device="cpu")
    (c0, a0, l0), (c1, a1, l1) = _run_pair(ctx, base, lane, la)
    assert isinstance(c1, PANActionCategorical) and c1.need_subsample
    assert a0 == a1
    assert torch.equal(l0, l1)
    for x, y in zip(c0.logits, c1.logits, strict=True):
        assert torch.equal(x, y)
