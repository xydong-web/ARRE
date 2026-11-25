"""Shared fixtures for the Lane A tests that use the real 20k-block smoke environment."""

from __future__ import annotations

import os

import torch

from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.env_context import SynthesisEnvContext
from cgflow.envs.semla_env import SynthesisEnvContext3D_semla
from cgflow.config import Config

ENV20K = os.environ.get("LANE_A_ENV20K", "/path/to/ARRE/data/smoke/env20k")


class LiteSemlaCtx(SynthesisEnvContext3D_semla):
    """`SynthesisEnvContext3D_semla` without the pose model / pocket (block-side API only).

    `get_block_data` (and therefore the pairwise size mask) is INHERITED from the real class,
    so the level rule the tests check is the production one.
    """

    def __init__(self, env: SynthesisEnv, num_cond_dim: int = 0):
        SynthesisEnvContext.__init__(self, env, num_cond_dim)


class FakeBatch:
    """The attributes `RxnFlow._make_cat` / `RxnActionCategorical` read off a gd.Batch."""

    def __init__(self, protocol_mask: torch.Tensor, level: torch.Tensor):
        self.protocol_mask = protocol_mask
        self.level = level
        self.num_graphs = int(protocol_mask.shape[0])


def load_env(env_dir: str = ENV20K):
    env = SynthesisEnv(env_dir)
    ctx = LiteSemlaCtx(env, num_cond_dim=32)
    return env, ctx


def base_cfg(sampling_ratio: float = 0.01) -> Config:
    cfg = Config()
    cfg.model.num_emb = 64
    cfg.algo.action_subsampling.sampling_ratio = sampling_ratio
    cfg.algo.action_subsampling.min_sampling = 50
    cfg.algo.sampling_tau = 0.9
    return cfg


def mask_for_pattern(ctx, pattern: int | None) -> torch.Tensor:
    """pattern None -> the initial state (all FirstBlock protocols)."""
    from rxnflow.envs.action import RxnActionType

    m = torch.zeros(len(ctx.protocols), dtype=torch.bool)
    for i, p in enumerate(ctx.protocols):
        if pattern is None:
            m[i] = p.action is RxnActionType.FirstBlock
        else:
            m[i] = p.action is RxnActionType.BiRxn and p.state_type == pattern
    return m


def brute_force_state_logits(ctx, model, emb_row: torch.Tensor, mask: torch.Tensor, level_state: float,
                             chunk: int = 8192):
    """All legal (protocol, block) logits of one state, by enumeration (CPU, chunked)."""
    from rxnflow.envs.action import RxnActionType

    outs, legals, meta = [], [], []
    thr = 1.0 - level_state
    for p in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
        proto = ctx.protocols[p]
        for bt_idx, t in enumerate(proto.block_types):
            n = len(ctx.blocks[t])
            for lo in range(0, n, chunk):
                idx = torch.arange(lo, min(lo + chunk, n))
                typ, desc, fp, level = ctx.get_block_data(t, idx)
                if proto.action is RxnActionType.FirstBlock:
                    lg = model.hook_firstblock(emb_row.view(1, -1), (typ, desc, fp), proto.name).view(-1)
                else:
                    lg = model.hook_birxn(emb_row.view(1, -1), (typ, desc, fp), proto.name).view(-1)
                outs.append(lg)
                legals.append(level.view(-1) < thr)
                meta += [(p, bt_idx, int(i)) for i in idx]
    return torch.cat(outs), torch.cat(legals), meta
