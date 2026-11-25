"""Lane A / PAN-Sub core: the policy-adaptive normaliser.

Notation (one state `s`, action `a = (p, b)` with `p` a protocol and `b` a building block):

    l_theta(s, p, b) = < u_p(s) , e_theta(b) >            (the bilinear logit of the baseline)
    P_theta(p, b | s) = exp l_theta(s, p, b) / Z_theta(s),  Z_theta(s) = sum over LEGAL (p, b)

The baseline never evaluates `Z_theta`.  It draws a *uniform* 1% subset `S` of every block type
and uses  log Z_hat = log sum_{b in S} (1/rho) exp l_b, which has relative variance
(1/rho - 1) / M_eff with M_eff = 1 / sum_b P_b^2.

PAN replaces that by importance sampling against a proposal built from a *cached* copy of the
block tower (`BlockEmbCache`, refreshed every R iterations) combined with the current state
embeddings:

    l_c(s, p, b)   = < u_p(s).detach() , E_cache[b] >
    q_c(p, b | s)  = exp l_c / Z_c(s)                      (Z_c computed EXACTLY, full pass)
    q(p, b | s)    = (1 - lambda) q_c + lambda / |legal(s)|            (defensive mixture)
    b_1..b_K ~iid~ q
    log Z_hat(s)   = logsumexp_k ( l_theta(s, b_k) - log q(b_k) ) - log K
    log P_F(a | s) = l_theta(s, a) - log Z_hat(s)

`Z_hat` is unbiased for `Z_theta` for ANY proposal with full support; its relative variance is
chi^2(P_theta || q) / K, which goes to 0 as the cache becomes fresh (q -> P_theta), and the
defensive component bounds the weights by |legal| / lambda.  The gradient of `log Z_hat` is the
self-normalised IS estimator  sum_k w_k grad l_theta(s, b_k),  w_k = softmax_k(l_theta - log q).

Sampling from `q` over up to 6.0M legal entries without materialising them is done in two
chunked passes over the cached table: pass 1 accumulates the per-chunk log-sum-exp `m[s, c]`
and the per-chunk legal count, which gives `Z_c` exactly and a categorical over chunks; pass 2
re-computes the logits of the selected chunks and draws the position inside the chunk by
inverse CDF.  The mixture is sampled by first picking the component, so the uniform draws need
no matrix product at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from rxnflow.envs.action import RxnActionType

from cgflow.lanes.config import LaneAConfig
from .block_table import BlockTable, BlockEmbCache

NEG_INF = float("-inf")
EMB_CHUNK = 1 << 16  # blocks per on-the-fly `emb_block` call (unpacked fp is ~4.8 kB/block)


@dataclass
class _Chunk:
    slot: int  # index of the protocol inside the group's legal-protocol list
    proto: int  # protocol index in ctx.protocols
    bt: int  # block_type_idx inside that protocol
    lo: int  # global block index range
    hi: int


@dataclass
class _Plan:
    protocols: list[int]
    chunks: list[_Chunk]
    n_entries: int


@dataclass
class PanSample:
    """Result of one full-space cached pass over a batch of states."""

    log_Zc: Tensor  # [Ns]      exact log normaliser of the cached proposal
    log_Nleg: Tensor  # [Ns]      log |legal(s)|
    proto: Tensor  # [Ns, S]   protocol index of every drawn entry
    bt: Tensor  # [Ns, S]   block_type_idx inside that protocol
    block: Tensor  # [Ns, S]   block index inside that block type
    glob: Tensor  # [Ns, S]   global block index
    log_q: Tensor  # [Ns, S]   log proposal density of the drawn entry
    u_sel: Tensor  # [Ns, S, D] u_p(s) of the drawn entry (WITH grad)
    l_c: Tensor  # [Ns, S]   cached-proposal logit of the drawn entry (no grad)
    n_beh: int
    n_is: int
    n_degenerate: int = 0

    @property
    def is_slice(self) -> slice:
        return slice(self.n_beh, self.n_beh + self.n_is)


class PanEngine:
    """Owns the block table, the embedding cache and the chunk plans."""

    def __init__(
        self,
        cfg: LaneAConfig,
        ctx,
        device: torch.device,
        num_emb_block: int,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.device = device
        self.table = BlockTable(ctx, device, cfg.table_device)
        dtype = torch.float16 if cfg.cache_dtype == "float16" else torch.float32
        self.cache = BlockEmbCache(self.table, num_emb_block, dtype)
        self._plans: dict[tuple[bytes, int], _Plan] = {}
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed) * 1_000_003 + 20_251_018)
        self.stats: dict[str, float] = {}
        self._timers: dict[str, float] = {}
        self.iteration: int = 0
        self._last_diag_it: int = -1

    # ---------------------------------------------------------------- plans
    def _plan(self, mask_row: Tensor, n_states: int) -> _Plan:
        key_bucket = 1 << max(0, (n_states - 1).bit_length())
        key = (mask_row.detach().cpu().numpy().tobytes(), key_bucket)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        protocols = [int(i) for i in torch.nonzero(mask_row, as_tuple=False).view(-1).tolist()]
        max_len = max(4096, self.cfg.chunk_elems // max(key_bucket, 1))
        chunks: list[_Chunk] = []
        n_entries = 0
        for slot, p in enumerate(protocols):
            for bt_idx, _ti, start, length in self.table.segments[p]:
                n_entries += length
                for a in range(0, length, max_len):
                    chunks.append(_Chunk(slot, p, bt_idx, start + a, start + min(a + max_len, length)))
        plan = _Plan(protocols, chunks, n_entries)
        self._plans[key] = plan
        return plan

    # ---------------------------------------------------------------- u_p(s)
    def _u_of_group(self, model, emb_g: Tensor, protocols: list[int]) -> Tensor:
        """[n, P, D] state-side vectors u_p(s), WITH grad."""
        names = [self.table.protocol_names[p] for p in protocols]
        pe = torch.stack([model.emb_protocol[nm] for nm in names], dim=0)  # [P, F]
        x = model.act(emb_g.unsqueeze(1) + pe.unsqueeze(0))  # [n, P, F]
        first = [self.ctx.protocols[p].action is RxnActionType.FirstBlock for p in protocols]
        if all(first):
            return model.mlp_firstblock(x)
        if not any(first):
            return model.mlp_birxn(x)
        raise RuntimeError("a state must not mix FirstBlock and BiRxn protocols")

    def u_of_actions(self, model, emb: Tensor, proto_idx: Tensor) -> Tensor:
        """[Ns, D] u_{p_i}(s_i) for one protocol per state, WITH grad."""
        names = [self.table.protocol_names[int(p)] for p in proto_idx.tolist()]
        pe = torch.stack([model.emb_protocol[nm] for nm in names], dim=0)  # [Ns, F]
        x = model.act(emb + pe)
        is_first = self.table.is_firstblock[proto_idx]
        out = torch.empty((emb.shape[0], self.cache.emb.shape[1]), device=emb.device, dtype=x.dtype)
        if bool(is_first.any()):
            out[is_first] = model.mlp_firstblock(x[is_first])
        if bool((~is_first).any()):
            out[~is_first] = model.mlp_birxn(x[~is_first])
        return out

    # ---------------------------------------------------------------- pass 1
    def _pass1(self, plan: _Plan, u_det: Tensor, thr: Tensor):
        """m[n, C] = log sum exp of the cached logits in each chunk, nleg[n, C] = legal count."""
        n = u_det.shape[0]
        C = len(plan.chunks)
        dev = self.device
        m = torch.full((n, C), NEG_INF, device=dev)
        nleg = torch.zeros((n, C), device=dev)
        emb_c = self.cache.emb
        level = self.table.level
        for ci, ch in enumerate(plan.chunks):
            legal = level[ch.lo : ch.hi].unsqueeze(0) < thr.unsqueeze(1)  # [n, L]
            cnt = legal.sum(1)
            nleg[:, ci] = cnt.to(nleg.dtype)
            if not bool(legal.any()):
                continue
            logits = u_det[:, ch.slot] @ emb_c[ch.lo : ch.hi].to(u_det.dtype).T  # [n, L]
            logits = logits.masked_fill(~legal, NEG_INF)
            m[:, ci] = torch.logsumexp(logits, dim=1)
        return m, nleg

    # ---------------------------------------------------------------- pass 2
    def _pass2(self, plan: _Plan, u_det: Tensor, thr: Tensor, m: Tensor, c_sel: Tensor, is_unif: Tensor):
        """Draw the position inside each selected chunk.  Returns global idx / proto / bt / block."""
        n, S = c_sel.shape
        dev = self.device
        glob = torch.zeros((n, S), dtype=torch.long, device=dev)
        slot = torch.zeros((n, S), dtype=torch.long, device=dev)
        proto = torch.zeros((n, S), dtype=torch.long, device=dev)
        bt = torch.zeros((n, S), dtype=torch.long, device=dev)
        emb_c = self.cache.emb
        level = self.table.level
        for ci, ch in enumerate(plan.chunks):
            sel = c_sel == ci
            if not bool(sel.any()):
                continue
            rows = torch.nonzero(sel.any(dim=1), as_tuple=False).view(-1)
            legal = level[ch.lo : ch.hi].unsqueeze(0) < thr[rows].unsqueeze(1)  # [R, L]
            sel_q = sel & ~is_unif
            sel_u = sel & is_unif
            if bool(sel_q.any()):
                logits = u_det[rows, ch.slot] @ emb_c[ch.lo : ch.hi].to(u_det.dtype).T
                wq = torch.where(legal, (logits - m[rows, ci].unsqueeze(1)).exp(), torch.zeros_like(logits))
            else:
                wq = None
            wu = legal.to(torch.float32)
            for use_unif, mask in ((False, sel_q), (True, sel_u)):
                if not bool(mask.any()):
                    continue
                w = wu if use_unif else wq
                cum = torch.cumsum(w, dim=1)
                total = cum[:, -1:].clamp_min(1e-30)
                cum = cum / total
                ii, kk = torch.nonzero(mask, as_tuple=True)  # sorted by row then slot
                cnt = torch.bincount(ii, minlength=n)
                starts = torch.cumsum(cnt, 0) - cnt
                rank = torch.arange(ii.numel(), device=dev) - starts[ii]
                mmax = int(cnt.max())
                row_of = torch.full((n,), -1, dtype=torch.long, device=dev)
                row_of[rows] = torch.arange(rows.numel(), device=dev)
                u01 = torch.rand((rows.numel(), mmax), device=dev, generator=self.gen)
                pos = torch.searchsorted(cum.contiguous(), u01.contiguous(), right=True)
                pos = pos.clamp_(max=w.shape[1] - 1)
                chosen = pos[row_of[ii], rank]
                glob[ii, kk] = chosen + ch.lo
                slot[ii, kk] = ch.slot
                proto[ii, kk] = ch.proto
                bt[ii, kk] = ch.bt
        return glob, slot, proto, bt

    # ---------------------------------------------------------------- main entry
    def evaluate(
        self,
        model,
        emb: Tensor,
        protocol_mask: Tensor,
        level_state: Tensor,
        n_beh: int,
        n_is: int,
        beh_lam: Tensor | None = None,
    ) -> PanSample:
        """Full-space cached pass + S = n_beh + n_is draws per state from the defensive mixture."""
        t0 = time.time()
        cfg = self.cfg
        Ns = emb.shape[0]
        draw = (n_beh + n_is) > 0
        S = max(1, n_beh + n_is)
        dev = self.device
        D = self.cache.emb.shape[1]

        log_Zc = torch.zeros(Ns, device=dev)
        log_Nleg = torch.zeros(Ns, device=dev)
        o_glob = torch.zeros((Ns, S), dtype=torch.long, device=dev)
        o_proto = torch.zeros((Ns, S), dtype=torch.long, device=dev)
        o_bt = torch.zeros((Ns, S), dtype=torch.long, device=dev)
        o_block = torch.zeros((Ns, S), dtype=torch.long, device=dev)
        o_logq = torch.zeros((Ns, S), device=dev)
        o_lc = torch.zeros((Ns, S), device=dev)
        o_u = torch.zeros((Ns, S, D), device=dev)

        lam = torch.full((Ns, S), float(cfg.defensive_mix), device=dev)
        if n_beh > 0:
            lam[:, 0] = float(cfg.behavior_eps) if beh_lam is None else beh_lam
        if n_is == 0 and n_beh == 0:
            lam[:, 0] = float(cfg.defensive_mix)

        thr_all = 1.0 - level_state.view(-1)
        uniq, inverse = torch.unique(protocol_mask, dim=0, return_inverse=True)
        n_degenerate = 0
        for g in range(uniq.shape[0]):
            rows = torch.nonzero(inverse == g, as_tuple=False).view(-1)
            plan = self._plan(uniq[g], int(rows.numel()))
            u = self._u_of_group(model, emb[rows], plan.protocols)  # [n, P, D], grad
            u_det = u.detach()
            thr = thr_all[rows]
            m, nleg = self._pass1(plan, u_det, thr)
            bad = nleg.sum(1) == 0
            if bool(bad.any()):  # the size mask excluded every block -> ignore it for those rows
                n_degenerate += int(bad.sum())
                thr = torch.where(bad, torch.full_like(thr, float("inf")), thr)
                m, nleg = self._pass1(plan, u_det, thr)
            lZc = torch.logsumexp(m, dim=1)
            nleg_tot = nleg.sum(1)
            lN = nleg_tot.clamp_min(1.0).log()
            log_Zc[rows] = lZc
            log_Nleg[rows] = lN
            if not draw:
                continue

            lam_g = lam[rows]
            is_unif = torch.rand(lam_g.shape, device=dev, generator=self.gen) < lam_g
            pq = torch.softmax(m, dim=1)
            pu = nleg / nleg_tot.clamp_min(1.0).unsqueeze(1)
            c_q = torch.multinomial(pq, S, replacement=True, generator=self.gen)
            c_u = torch.multinomial(pu, S, replacement=True, generator=self.gen)
            c_sel = torch.where(is_unif, c_u, c_q)
            glob, slot, proto, bt = self._pass2(plan, u_det, thr, m, c_sel, is_unif)

            # exact proposal density of the drawn entries
            u_sel = torch.gather(u, 1, slot.unsqueeze(-1).expand(-1, -1, D))  # [n, S, D], grad
            e_c = self.cache.emb[glob.reshape(-1)].to(u_sel.dtype).view(glob.shape[0], S, D)
            l_c = (u_sel.detach() * e_c).sum(-1)
            log_q = torch.logaddexp(
                torch.log1p(-lam_g) + l_c - lZc.unsqueeze(1),
                torch.log(lam_g) - lN.unsqueeze(1),
            )
            ti = self.table.type_id[glob].long()
            o_glob[rows] = glob
            o_proto[rows] = proto
            o_bt[rows] = bt
            o_block[rows] = glob - self.table.type_offset[ti]
            o_logq[rows] = log_q
            o_lc[rows] = l_c
            o_u[rows] = u_sel

        self._timers["pan_pass"] = self._timers.get("pan_pass", 0.0) + (time.time() - t0)
        return PanSample(
            log_Zc=log_Zc,
            log_Nleg=log_Nleg,
            proto=o_proto,
            bt=o_bt,
            block=o_block,
            glob=o_glob,
            log_q=o_logq,
            u_sel=o_u,
            l_c=o_lc,
            n_beh=n_beh,
            n_is=n_is,
            n_degenerate=n_degenerate,
        )

    # ---------------------------------------------------------------- grad-side block embeddings
    def embed_blocks(self, model, global_idx: Tensor, chunk: int = 32768) -> Tensor:
        """e_theta(b) for the given global block indices, WITH grad (optionally checkpointed)."""
        outs = []

        def _fwd(idx: Tensor) -> Tensor:
            typ, desc, fp = self.table.features(idx)
            return model.emb_block((typ, desc, fp))

        for lo in range(0, int(global_idx.numel()), chunk):
            part = global_idx[lo : lo + chunk]
            if self.cfg.grad_checkpoint and torch.is_grad_enabled():
                outs.append(checkpoint(_fwd, part, use_reentrant=False))
            else:
                outs.append(_fwd(part))
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    def logits_of(self, model, u_sel: Tensor, glob: Tensor) -> Tensor:
        """l_theta(s, b) for a [Ns, S] block-index matrix, WITH grad."""
        flat = glob.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        e = self.embed_blocks(model, uniq)
        e_sel = e[inv].view(*glob.shape, -1)
        return (u_sel * e_sel).sum(-1)

    def logits_of_actions(self, model, u_a: Tensor, glob: Tensor) -> Tensor:
        uniq, inv = torch.unique(glob, return_inverse=True)
        e = self.embed_blocks(model, uniq)
        return (u_a * e[inv]).sum(-1)


    # ---------------------------------------------------------------- exact diagnostics
    def should_diag(self, iteration: int, role: str) -> bool:
        if role != "train" or self.cfg.diag_every <= 0:
            return False
        if iteration % int(self.cfg.diag_every) != 0:
            return False
        return self._last_diag_it != iteration

    @torch.no_grad()
    def exact_stats(self, model, emb: Tensor, protocol_mask: Tensor, level_state: Tensor):
        """Exact log Z_theta, M_eff and top-k mass for a handful of states.

        The block tower of the CURRENT model is evaluated on the fly over the whole legal set,
        so this is the ground truth the PAN estimate is compared against (it costs one full
        embedding pass and is therefore only run every `diag_every` iterations).
        """
        uniq, inverse = torch.unique(protocol_mask, dim=0, return_inverse=True)
        counts = torch.bincount(inverse, minlength=uniq.shape[0])
        g = int(counts.argmax())
        rows = torch.nonzero(inverse == g, as_tuple=False).view(-1)[: int(self.cfg.diag_num_states)]
        plan = self._plan(uniq[g], int(rows.numel()))
        u = self._u_of_group(model, emb[rows], plan.protocols)
        thr = (1.0 - level_state.view(-1)[rows])
        n = rows.numel()
        dev = self.device
        run_max = torch.full((n,), NEG_INF, device=dev)
        run_s1 = torch.zeros(n, device=dev)
        run_s2 = torch.zeros(n, device=dev)
        topk = min(100, 100)
        run_top = torch.full((n, topk), NEG_INF, device=dev)
        was_training = model.training
        model.eval()
        for ch in plan.chunks:
            legal = self.table.level[ch.lo : ch.hi].unsqueeze(0) < thr.unsqueeze(1)
            if not bool(legal.any()):
                continue
            # the unpacked float fingerprints are 4.8 kB per block -> embed in small slices
            parts = []
            for a in range(ch.lo, ch.hi, EMB_CHUNK):
                b = min(a + EMB_CHUNK, ch.hi)
                typ, desc, fp = self.table.features_slice(a, b)
                parts.append(u[:, ch.slot] @ model.emb_block((typ, desc, fp)).T)
            logits = (torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]).masked_fill(~legal, NEG_INF)
            new_max = torch.maximum(run_max, logits.max(dim=1).values)
            scale = (run_max - new_max).exp()
            scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
            shifted = (logits - new_max.unsqueeze(1)).exp()
            run_s1 = run_s1 * scale + shifted.sum(1)
            run_s2 = run_s2 * scale * scale + (shifted * shifted).sum(1)
            run_max = new_max
            k = min(topk, logits.shape[1])
            cand = torch.cat([run_top, logits.topk(k, dim=1).values], dim=1)
            run_top = cand.topk(topk, dim=1).values
        if was_training:
            model.train()
        log_Z = run_max + run_s1.clamp_min(1e-30).log()
        log_s2 = 2 * run_max + run_s2.clamp_min(1e-300).log()
        log_meff = 2 * log_Z - log_s2
        p_top = (run_top - log_Z.unsqueeze(1)).exp()
        return {
            "rows": rows,
            "log_Z": log_Z,
            "log_meff": log_meff,
            "top1": p_top[:, 0],
            "top10": p_top[:, :10].sum(1),
            "top100": p_top.sum(1),
            "n_entries": plan.n_entries,
        }
