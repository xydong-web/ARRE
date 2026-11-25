"""T2a - the flat block table reproduces the context's own block features exactly."""

from __future__ import annotations

import torch

from cgflow.lanes.pansub.block_table import BlockTable
from .realenv import load_env


def test_block_table_matches_context():
    env, ctx = load_env()
    table = BlockTable(ctx, torch.device("cpu"), "cpu")
    assert table.num_blocks == sum(len(v) for v in ctx.blocks.values())

    g = torch.Generator().manual_seed(0)
    for t in [ctx.block_types[i] for i in torch.randperm(len(ctx.block_types), generator=g)[:12].tolist()]:
        n = len(ctx.blocks[t])
        k = min(n, 97)
        local = torch.randperm(n, generator=g)[:k]
        ti = ctx.block_type_to_idx[t]
        glob = table.type_offset_cpu[ti] + local
        typ_r, desc_r, fp_r, lvl_r = ctx.get_block_data(t, local)
        typ, desc, fp = table.features(glob)
        assert torch.equal(typ, typ_r), t
        assert torch.allclose(desc, desc_r), t
        assert torch.equal(fp, fp_r), t
        assert torch.allclose(table.level[glob], lvl_r.view(-1)), t


def test_segment_encode_decode_roundtrip():
    env, ctx = load_env()
    table = BlockTable(ctx, torch.device("cpu"), "cpu")
    g = torch.Generator().manual_seed(1)
    for p in torch.randperm(len(ctx.protocols), generator=g)[:25].tolist():
        segs = table.segments[p]
        if not segs:
            continue
        bt_idx, ti, start, length = segs[int(torch.randint(len(segs), (1,), generator=g))]
        blk = int(torch.randint(length, (1,), generator=g))
        glob = table.encode(p, bt_idx, blk)
        assert glob == start + blk
        assert int(table.seg_start[p, bt_idx]) + blk == glob
        bt2, blk2 = table.decode(torch.tensor([p]), torch.tensor([glob]))
        assert int(blk2) == blk
        # the type -> position map is only unambiguous if a protocol lists each type once
        assert len({s[1] for s in segs}) == len(segs)
        assert int(bt2) == bt_idx
