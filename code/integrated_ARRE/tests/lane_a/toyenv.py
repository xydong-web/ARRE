"""A tiny synthetic environment + policy for the Lane A unit tests.

It mimics only the parts of `SynthesisEnvContext` and `RxnFlow` that `PanEngine` touches, so
the estimator can be compared against brute-force enumeration of the whole action space on
the CPU in a second.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gflownet.models.graph_transformer import mlp
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.models.gfn import BlockEmbedding


class ToyProtocol(Protocol):
    def __init__(self, name: str, block_types: list[str]):
        super().__init__(name, RxnActionType.FirstBlock, block_types=block_types)


class ToyCtx:
    """Mimics `SynthesisEnvContext3D_semla` for the block-side API only."""

    def __init__(self, type_sizes: dict[str, int], protocols: dict[str, list[str]], fp_dim=64, desc_dim=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.block_types = sorted(type_sizes)
        self.num_block_types = len(self.block_types)
        self.block_type_to_idx = {t: i for i, t in enumerate(self.block_types)}
        self.blocks = {t: [f"{t}:{i}" for i in range(type_sizes[t])] for t in self.block_types}
        self.block_fp_dim = fp_dim
        self.block_desc_dim = desc_dim
        self.block_features = {}
        for t in self.block_types:
            n = type_sizes[t]
            desc = torch.rand((n, desc_dim), generator=g)
            fp = torch.rand((n, fp_dim), generator=g) < 0.2
            self.block_features[t] = (desc, fp)
        self.protocols = [ToyProtocol(name, bts) for name, bts in protocols.items()]
        self.num_protocols = len(self.protocols)
        self.protocol_to_idx = {p.name: i for i, p in enumerate(self.protocols)}

    def get_block_data(self, block_type: str, block_indices):
        desc, fp = self.block_features[block_type]
        desc, fp = desc[block_indices], fp[block_indices]
        if fp.dim() == 1:
            desc, fp = desc.view(1, -1), fp.view(1, -1)
        typ = torch.full((fp.shape[0],), self.block_type_to_idx[block_type], dtype=torch.long)
        # same shape of rule as SynthesisEnvContext3D_semla: a scalar "size" taken from desc
        level = (desc[:, 1] / 5).view(-1, 1)
        return typ, desc, fp.to(torch.float32), level


class _Scale(nn.Module):
    def __init__(self, s: float):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x * self.s


class ToyModel(nn.Module):
    """Same bilinear structure as `RxnFlow`: l(s, p, b) = <mlp(act(emb_s + emb_p)), emb_block(b)>."""

    def __init__(self, ctx: ToyCtx, n_state: int = 16, n_emb_block: int = 8, seed: int = 0, scale: float = 1.0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb_block = BlockEmbedding(ctx.block_fp_dim, ctx.block_desc_dim, ctx.num_block_types, n_emb_block, n_emb_block, 1)
        self.emb_protocol = nn.ParameterDict({p.name: nn.Parameter(torch.randn(n_state)) for p in ctx.protocols})
        self.mlp_firstblock = nn.Sequential(mlp(n_state, n_state, n_emb_block, 1), _Scale(scale))
        self.mlp_birxn = nn.Sequential(mlp(n_state, n_state, n_emb_block, 1), _Scale(scale))
        self.act = nn.LeakyReLU()

    def perturb(self, scale: float = 0.05, seed: int = 1):
        """Move the parameters a little, to emulate a stale embedding cache."""
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn(p.shape, generator=g) * scale)


def brute_force_logits(ctx: ToyCtx, model: ToyModel, emb_row: torch.Tensor, protocols: list[int], thr: float):
    """All (protocol, block) logits of one state, plus the legality mask, by full enumeration."""
    rows = []
    legal = []
    meta = []
    for p in protocols:
        proto = ctx.protocols[p]
        u = model.mlp_firstblock(model.act(emb_row + model.emb_protocol[proto.name]))
        for bt_idx, t in enumerate(proto.block_types):
            idx = torch.arange(len(ctx.blocks[t]))
            typ, desc, fp, level = ctx.get_block_data(t, idx)
            e = model.emb_block((typ, desc, fp))
            rows.append(e @ u)
            legal.append(level.view(-1) < thr)
            meta += [(p, bt_idx, int(i)) for i in idx]
    return torch.cat(rows), torch.cat(legal), meta
