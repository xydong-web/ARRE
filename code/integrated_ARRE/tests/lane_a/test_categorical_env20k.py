"""T2b - the PAN categorical on the real 20k-block environment, against full enumeration."""

from __future__ import annotations

import math
from collections import Counter

import torch

from gflownet.utils.misc import set_worker_rng_seed
from rxnflow.envs.action import RxnActionType
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import set_worker_env

from cgflow.lanes.config import LaneAConfig
from cgflow.lanes.pansub.model import PANRxnFlow
from cgflow.lanes.pansub.pan import PanEngine

from .realenv import FakeBatch, base_cfg, brute_force_state_logits, load_env, mask_for_pattern

DEV = torch.device("cpu")


def _small_pattern(ctx):
    """The BiRxn state pattern with the fewest legal entries (keeps brute force cheap)."""
    sizes = {}
    for p in ctx.protocols:
        if p.action is RxnActionType.BiRxn:
            sizes[p.state_type] = sizes.get(p.state_type, 0) + sum(len(ctx.blocks[t]) for t in p.block_types)
    return min(sizes, key=sizes.get), min(sizes.values())


def _setup(la: LaneAConfig, n=4, pattern=None, level=0.3, seed=0):
    env, ctx = load_env()
    cfg = base_cfg()
    set_worker_env("ctx", ctx)
    set_worker_env("env", env)
    set_worker_env("action_subsampler", SubsamplingPolicy(env, cfg))
    set_worker_rng_seed(0)
    torch.manual_seed(seed)
    model = PANRxnFlow(ctx, cfg, num_graph_out=1)
    model.eval()
    engine = PanEngine(la, ctx, DEV, num_emb_block=cfg.model.num_emb_block, seed=seed)
    engine.cache.refresh(model, 0)
    model.attach_pan(engine, "train")
    if pattern is None:
        pattern, _ = _small_pattern(ctx)
    mask = mask_for_pattern(ctx, pattern).unsqueeze(0).repeat(n, 1)
    g = torch.Generator().manual_seed(seed + 5)
    emb = torch.randn((n, 128), generator=g)
    return ctx, model, engine, FakeBatch(mask, torch.full((n, 1), float(level))), emb, pattern


def test_exact_normalizer_on_real_env():
    la = LaneAConfig(enable=True, normalizer="pan", behavior="full", cache_dtype="float32",
                     table_device="cpu", chunk_elems=1 << 18)
    ctx, model, engine, graphs, emb, pattern = _setup(la, n=3)
    with torch.no_grad():
        pan = engine.evaluate(model, emb, graphs.protocol_mask, graphs.level, n_beh=0, n_is=0)
        for i in range(emb.shape[0]):
            lg, legal, _ = brute_force_state_logits(ctx, model, emb[i], graphs.protocol_mask[i],
                                                    float(graphs.level[i]))
            ref = torch.logsumexp(lg[legal], 0)
            print(f"state {i}: pattern={pattern} entries={lg.numel()} legal={int(legal.sum())} "
                  f"logZc={float(pan.log_Zc[i]):.5f} exact={float(ref):.5f}")
            assert abs(float(pan.log_Zc[i]) - float(ref)) < 3e-3
            assert abs(float(pan.log_Nleg[i].exp()) - float(legal.sum())) < 0.5


def test_behaviour_frequencies_on_real_env():
    la = LaneAConfig(enable=True, normalizer="pan", behavior="full", behavior_eps=0.15,
                     cache_dtype="float32", table_device="cpu", chunk_elems=1 << 18)
    ctx, model, engine, graphs, emb, pattern = _setup(la, n=1)
    lg, legal, meta = brute_force_state_logits(ctx, model, emb[0], graphs.protocol_mask[0],
                                               float(graphs.level[0]))
    lZ = torch.logsumexp(lg[legal], 0)
    nleg = int(legal.sum())
    q = torch.where(legal, 0.85 * (lg - lZ).exp() + 0.15 / nleg, torch.zeros_like(lg))
    pos = {m: k for k, m in enumerate(meta)}

    cnt = Counter()
    n_draw = 0
    with torch.no_grad():
        for _ in range(60):
            cat = model._make_cat(graphs, emb)
            acts = cat.sample_full(0.0)
            for a in acts:
                cnt[pos[(int(a[0]), int(a[1]), int(a[2]))]] += 1
                n_draw += 1
            # 400 extra draws per round, cheaply, from the same cached pass
            pan = cat._ensure_pan(n_beh=1, n_is=400)
            for p_, t_, b_ in zip(pan.proto[0, 1:], pan.bt[0, 1:], pan.block[0, 1:]):
                cnt[pos[(int(p_), int(t_), int(b_))]] += 1
                n_draw += 1
    obs = torch.zeros(lg.numel())
    for k, v in cnt.items():
        obs[k] = v
    assert float(obs[~legal].sum()) == 0.0, "sampled an action the size mask forbids"
    nb = 30
    bid = torch.arange(lg.numel()) % nb
    o = torch.zeros(nb).index_add_(0, bid, obs)
    e = torch.zeros(nb).index_add_(0, bid, q * n_draw)
    chi2 = float((((o - e) ** 2) / e.clamp_min(1e-9)).sum())
    print(f"n_draw={n_draw} chi2={chi2:.1f} (df={nb - 1})")
    assert chi2 < 3 * (nb - 1), chi2


def test_log_prob_matches_exact_when_cache_is_fresh():
    la = LaneAConfig(enable=True, normalizer="pan", behavior="full", num_samples=4096,
                     defensive_mix=0.1, cache_dtype="float32", table_device="cpu",
                     chunk_elems=1 << 18, grad_checkpoint=False, clamp_logprob=False)
    ctx, model, engine, graphs, emb, pattern = _setup(la, n=3)
    with torch.no_grad():
        cat = model._make_cat(graphs, emb)
        actions = cat.sample_full(0.0)
        cat._pan = None
        cat._pan_slots = (0, 0)
        cat.role = "train"  # force the importance-sampling estimator
        lp = cat.log_prob(actions)
        for i, a in enumerate(actions):
            lg, legal, meta = brute_force_state_logits(ctx, model, emb[i], graphs.protocol_mask[i],
                                                       float(graphs.level[i]))
            k = {m: j for j, m in enumerate(meta)}[(int(a[0]), int(a[1]), int(a[2]))]
            ref = float(lg[k] - torch.logsumexp(lg[legal], 0))
            print(f"state {i}: log_pf={float(lp[i]):.5f} exact={ref:.5f} diff={float(lp[i]) - ref:+.2e}")
            assert abs(float(lp[i]) - ref) < 0.02
    assert 0.0 < engine.stats["lane_a/ess_frac"] <= 1.0


def test_chunking_and_mixed_groups():
    """Many chunks per segment (small chunk budget) and a batch whose states have different
    protocol masks must still give the exact normaliser."""
    la = LaneAConfig(enable=True, normalizer="pan", behavior="full", cache_dtype="float32",
                     table_device="cpu", chunk_elems=4096)
    env, ctx = load_env()
    cfg = base_cfg()
    set_worker_env("ctx", ctx)
    set_worker_env("env", env)
    set_worker_env("action_subsampler", SubsamplingPolicy(env, cfg))
    torch.manual_seed(3)
    model = PANRxnFlow(ctx, cfg, num_graph_out=1)
    model.eval()
    engine = PanEngine(la, ctx, DEV, num_emb_block=cfg.model.num_emb_block, seed=3)
    engine.cache.refresh(model, 0)
    model.attach_pan(engine, "train")

    pat_small, _ = _small_pattern(ctx)
    masks = torch.stack(
        [
            mask_for_pattern(ctx, None),  # initial state: 32 FirstBlock protocols, 49.5k entries
            mask_for_pattern(ctx, pat_small),
            mask_for_pattern(ctx, 1),  # the largest pattern of the 20k env
            mask_for_pattern(ctx, None),
        ]
    )
    lvl = torch.tensor([[0.0], [0.3], [0.5], [0.7]])
    g = torch.Generator().manual_seed(9)
    emb = torch.randn((4, 128), generator=g)
    with torch.no_grad():
        pan = engine.evaluate(model, emb, masks, lvl, n_beh=1, n_is=8)
        for i in range(4):
            lg, legal, meta = brute_force_state_logits(ctx, model, emb[i], masks[i], float(lvl[i]))
            ref = torch.logsumexp(lg[legal], 0)
            print(f"state {i}: entries={lg.numel()} legal={int(legal.sum())} "
                  f"logZc={float(pan.log_Zc[i]):.5f} exact={float(ref):.5f}")
            assert abs(float(pan.log_Zc[i]) - float(ref)) < 5e-3
            assert abs(float(pan.log_Nleg[i].exp()) - float(legal.sum())) < 0.5
            pos = {m: k for k, m in enumerate(meta)}
            for s in range(pan.proto.shape[1]):
                k = pos[(int(pan.proto[i, s]), int(pan.bt[i, s]), int(pan.block[i, s]))]
                assert bool(legal[k]), "sampled an illegal action"
