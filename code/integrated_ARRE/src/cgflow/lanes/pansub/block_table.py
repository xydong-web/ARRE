"""Lane A / PAN-Sub: a flat, GPU-friendly view of the whole building-block library.

The baseline keeps the block features as `{block_type: (desc float32 [N_t, 8], fp bool [N_t, 1190])}`
(12.7 GB for the 10.4M-entry catalog) and materialises `float32` features for the ~1% it
sub-samples.  PAN needs
  (a) a *cached* embedding of EVERY block (to build the proposal and to normalise exactly), and
  (b) cheap random access to the features of an arbitrary set of blocks (to re-embed the
      importance samples WITH gradient).
`BlockTable` provides both by concatenating all block types into global arrays with per-type
offsets, with the fingerprints bit-packed to `uint8[N, 149]` (1.55 GB instead of 12.4 GB).

Index conventions
-----------------
* `type_idx`  : index into `ctx.block_types` (the sorted list of block-type names), 0..T-1.
* `global_idx`: `type_offset[type_idx] + block_idx`, 0..N-1; `block_idx` is the index used by
  `ActionIndex` / `ctx.blocks[block_type]`.
* `protocol_idx`: index into `ctx.protocols`.
* a "segment" is a (protocol, block type) pair, i.e. a contiguous slice of the global array that
  one protocol can act with.  `ActionIndex(protocol_idx, block_type_idx, block_idx)` where
  `block_type_idx` is the position of the type inside `protocol.block_types`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rxnflow.envs.action import RxnActionType
from rxnflow.envs.env_context import SynthesisEnvContext

FP_DIM = 1190


def unpack_bits(packed: Tensor, n_bits: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Inverse of `np.packbits(..., axis=1)` (MSB first) for a uint8 tensor [M, B]."""
    assert packed.dtype == torch.uint8
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :n_bits].to(dtype)


class BlockTable:
    """Flat global arrays over all (block type, block) pairs plus the protocol -> segment map."""

    def __init__(
        self,
        ctx: SynthesisEnvContext,
        device: torch.device,
        table_device: str = "auto",
    ) -> None:
        self.ctx = ctx
        self.device = device
        if table_device == "auto":
            self.table_device = torch.device("cpu")
        else:
            self.table_device = torch.device(table_device)

        self.fp_dim = int(ctx.block_fp_dim)
        block_types: list[str] = list(ctx.block_types)
        self.block_types = block_types
        self.num_types = T = len(block_types)

        sizes = [len(ctx.blocks[t]) for t in block_types]
        offsets = np.zeros(T + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        self.type_offset_cpu = torch.from_numpy(offsets)
        self.num_blocks = N = int(offsets[-1])

        # ---- global feature arrays -------------------------------------------------------
        desc = torch.empty((N, ctx.block_desc_dim), dtype=torch.float32)
        fp_packed = torch.empty((N, (self.fp_dim + 7) // 8), dtype=torch.uint8)
        type_id = torch.empty((N,), dtype=torch.int16)
        level = torch.empty((N,), dtype=torch.float32)
        probe = 1 << 14  # `get_block_data` materialises float32 fingerprints -> keep slices small
        for ti, t in enumerate(block_types):
            lo, hi = int(offsets[ti]), int(offsets[ti + 1])
            if hi == lo:
                continue
            d, f = ctx.block_features[t]
            desc[lo:hi] = d.to(torch.float32)
            fp_packed[lo:hi] = torch.from_numpy(np.packbits(f.numpy(), axis=1))
            type_id[lo:hi] = ti
            # `level` is taken from the context itself so that the pairwise size mask is
            # bit-identical to the baseline (`SynthesisEnvContext3D_semla.get_block_data`).
            n_t = hi - lo
            for a in range(0, n_t, probe):
                b = min(a + probe, n_t)
                *_, lv = ctx.get_block_data(t, torch.arange(a, b, dtype=torch.long))
                level[lo + a : lo + b] = lv.view(-1).to(torch.float32)

        self.desc = desc.to(self.table_device)
        self.fp_packed = fp_packed.to(self.table_device)
        self.type_id_cpu = type_id
        self.type_id = type_id.to(device)
        self.level = level.to(device)
        self.type_offset = self.type_offset_cpu.to(device)

        # ---- protocol -> segments --------------------------------------------------------
        # segments[p] = list of (block_type_idx_in_protocol, global_type_idx, start, length)
        self.segments: list[list[tuple[int, int, int, int]]] = []
        self.is_block_protocol: list[bool] = []
        type_to_idx = {t: i for i, t in enumerate(block_types)}
        pos_of_type = torch.full((len(ctx.protocols), T), -1, dtype=torch.int16)
        for pi, protocol in enumerate(ctx.protocols):
            segs: list[tuple[int, int, int, int]] = []
            ok = protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn)
            self.is_block_protocol.append(ok)
            if ok:
                for bt_idx, t in enumerate(protocol.block_types):
                    ti = type_to_idx[t]
                    segs.append((bt_idx, ti, int(offsets[ti]), int(offsets[ti + 1] - offsets[ti])))
                    pos_of_type[pi, ti] = bt_idx
            self.segments.append(segs)
        self.pos_of_type = pos_of_type.to(device)

        # [n_protocols, max #block types per protocol] -> global start of the segment
        max_bt = max((len(segs) for segs in self.segments), default=1)
        seg_start = torch.zeros((len(ctx.protocols), max(max_bt, 1)), dtype=torch.long)
        seg_len = torch.zeros_like(seg_start)
        for pi, segs in enumerate(self.segments):
            for bt_idx, _ti, start, length in segs:
                seg_start[pi, bt_idx] = start
                seg_len[pi, bt_idx] = length
        self.seg_start = seg_start.to(device)
        self.seg_len = seg_len.to(device)

        self.is_firstblock = torch.tensor(
            [p.action is RxnActionType.FirstBlock for p in ctx.protocols], dtype=torch.bool, device=device
        )
        self.protocol_names = [p.name for p in ctx.protocols]

    # -------------------------------------------------------------------------------------
    def nbytes(self) -> dict[str, int]:
        return {
            "fp_packed": self.fp_packed.numel(),
            "desc": self.desc.numel() * 4,
            "type_id": self.type_id.numel() * 2,
            "level": self.level.numel() * 4,
        }

    def features(self, global_idx: Tensor, device: torch.device | None = None):
        """(typ long, desc float32, fp float32) for the given global indices, on `device`."""
        device = self.device if device is None else device
        idx = global_idx.to(self.table_device)
        packed = self.fp_packed.index_select(0, idx).to(device, non_blocking=True)
        desc = self.desc.index_select(0, idx).to(device, non_blocking=True)
        typ = self.type_id.index_select(0, global_idx.to(self.type_id.device)).to(device).long()
        fp = unpack_bits(packed, self.fp_dim, torch.float32)
        return typ, desc, fp

    def features_slice(self, lo: int, hi: int, device: torch.device | None = None):
        device = self.device if device is None else device
        packed = self.fp_packed[lo:hi].to(device, non_blocking=True)
        desc = self.desc[lo:hi].to(device, non_blocking=True)
        typ = self.type_id[lo:hi].to(device).long()
        fp = unpack_bits(packed, self.fp_dim, torch.float32)
        return typ, desc, fp

    def decode(self, protocol_idx: Tensor, global_idx: Tensor) -> tuple[Tensor, Tensor]:
        """(block_type_idx inside the protocol, block_idx inside that block type)."""
        ti = self.type_id[global_idx].long()
        block_idx = global_idx - self.type_offset[ti]
        bt_idx = self.pos_of_type[protocol_idx, ti].long()
        return bt_idx, block_idx

    def encode(self, protocol_idx: int, block_type_idx: int, block_idx: int) -> int:
        _, ti, start, _ = self.segments[protocol_idx][block_type_idx]
        return start + block_idx


class BlockEmbCache:
    """GPU-resident cache of `emb_block` over every block, refreshed every R iterations."""

    def __init__(self, table: BlockTable, num_emb_block: int, dtype: torch.dtype = torch.float16):
        self.table = table
        self.dtype = dtype
        self.emb = torch.zeros((table.num_blocks, num_emb_block), dtype=dtype, device=table.device)
        self.version: int = -1
        self.n_refresh: int = 0
        self.last_refresh_seconds: float = 0.0
        self.last_absmax: float = 0.0
        self.n_overflow: int = 0

    @torch.no_grad()
    def refresh(self, model, iteration: int, chunk: int = 65536) -> float:
        import time

        t0 = time.time()
        was_training = model.training
        model.eval()
        N = self.table.num_blocks
        limit = float(torch.finfo(self.dtype).max)
        absmax = 0.0
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            typ, desc, fp = self.table.features_slice(lo, hi)
            e = model.emb_block((typ, desc, fp))
            absmax = max(absmax, float(e.abs().max()))
            # fp16 saturates at 65504; clamping keeps the proposal finite (the importance
            # weights then correct the - larger - mismatch) instead of poisoning it with inf.
            self.emb[lo:hi] = e.clamp(-limit, limit).to(self.dtype)
        self.last_absmax = absmax
        if absmax > limit:
            self.n_overflow += 1
            print(f"[lane_a] WARNING: block embedding |max|={absmax:.1f} exceeds the cache dtype range", flush=True)
        if was_training:
            model.train()
        if self.emb.device.type == "cuda":
            torch.cuda.synchronize(self.emb.device)
        self.version = iteration
        self.n_refresh += 1
        self.last_refresh_seconds = time.time() - t0
        return self.last_refresh_seconds
