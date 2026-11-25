"""Lane A / PAN-Sub: the action categorical.

`PANActionCategorical` subclasses the baseline `RxnActionCategorical` and overrides only
`sample()` (A2, `behavior="full"`) and `log_prob()` (A1, `normalizer="pan"`).  Whichever of the
two is switched off falls through to the baseline implementation, so the four combinations
(uniform/pan) x (subsample/full) are all available and `uniform+subsample` is exactly the
baseline (the expensive 1% sub-sample is only built when some code path still needs it).
"""

from __future__ import annotations

import math

import torch
import torch_geometric.data as gd
from torch import Tensor

from gflownet.envs.graph_building_env import ActionIndex
from rxnflow.policy.action_categorical import RxnActionCategorical

from .pan import PanEngine, PanSample


class PANActionCategorical(RxnActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        emb: Tensor,
        action_masks: list[Tensor],
        model: torch.nn.Module,
        engine: PanEngine,
        role: str = "train",
    ):
        self.engine = engine
        self.pan_cfg = engine.cfg
        self.role = role
        self._pan: PanSample | None = None
        self._pan_slots: tuple[int, int] = (0, 0)
        self.diag: dict[str, float] = {}

        self.use_pan_norm = self.pan_cfg.normalizer == "pan"
        self.use_full_behavior = self.pan_cfg.behavior == "full"
        self.need_subsample = (not self.use_pan_norm) or (not self.use_full_behavior)

        if self.need_subsample:
            super().__init__(graphs, emb, action_masks, model)
        else:
            # the baseline constructor materialises the 1% sub-sample and its logits; PAN needs
            # neither, so set up only what the rest of the code reads.
            from rxnflow.utils.misc import get_worker_env

            self.ctx = get_worker_env("ctx")
            self.model = model
            self.graphs = graphs
            self.level_state = graphs.level
            self.num_graphs = graphs.num_graphs
            self.emb = emb
            self.dev = emb.device
            self._epsilon = 1e-38
            self._action_masks = action_masks
            self._masked_logits = []
            self.raw_logits = []
            self.weighted_logits = []
            self.subsamples = []
            self.subsample_size = []
            self._weights = []
            self.batch = [torch.arange(self.num_graphs, device=self.dev)] * self.ctx.num_protocols
            self.slice = [torch.arange(self.num_graphs + 1, device=self.dev)] * self.ctx.num_protocols
            self.logprobs = None
            self.log_n = None

        self.protocol_mask = torch.stack(action_masks, dim=1)  # [Ngraph, Nprotocol]

    # ------------------------------------------------------------------ helpers
    def _no_grad(self) -> bool:
        return self.role == "sample"

    def _ensure_pan(self, n_beh: int, n_is: int, beh_lam: Tensor | None = None) -> PanSample:
        have_beh, have_is = self._pan_slots
        if self._pan is not None and have_beh >= n_beh and have_is >= n_is:
            return self._pan
        ctx = torch.no_grad() if self._no_grad() else torch.enable_grad()
        with ctx:
            self._pan = self.engine.evaluate(
                self.model,
                self.emb,
                self.protocol_mask,
                self.level_state,
                n_beh=n_beh,
                n_is=n_is,
                beh_lam=beh_lam,
            )
        self._pan_slots = (n_beh, n_is)
        return self._pan

    def _action_global_idx(self, actions) -> tuple[Tensor, Tensor]:
        """(protocol index, global block index); `actions` is a list of ActionIndex (sampling)
        or the [N, 3] long tensor that `construct_batch` builds (training)."""
        table = self.engine.table
        if not isinstance(actions, Tensor):
            actions = torch.tensor([[int(a[0]), int(a[1]), int(a[2])] for a in actions], dtype=torch.long)
        a = actions.to(self.dev).long()
        proto, bt, blk = a[:, 0], a[:, 1], a[:, 2]
        glob = table.seg_start[proto, bt] + blk
        return proto, glob

    # ------------------------------------------------------------------ A2: behaviour
    def sample_full(self, random_action_prob: float = 0.0) -> list[ActionIndex]:
        """a ~ (1 - eps) q_cached(.|s) + eps * Uniform(legal(s)) over the FULL legal action set."""
        from gflownet.utils.misc import get_worker_rng

        eps = float(self.pan_cfg.behavior_eps)
        lam = torch.full((self.num_graphs,), eps, device=self.dev)
        if random_action_prob > 0:
            rng = get_worker_rng()
            is_random = torch.as_tensor(
                rng.uniform(size=self.num_graphs) < random_action_prob, device=self.dev
            )
            lam = torch.where(is_random, torch.ones_like(lam), lam)
            self.diag["lane_a/behavior_random_frac"] = float(is_random.float().mean())
            self.engine.stats.update(self.diag)
        n_is = int(self.pan_cfg.num_samples) if self.pan_cfg.sampling_logprob == "pan" else 0
        pan = self._ensure_pan(n_beh=1, n_is=n_is, beh_lam=lam)
        proto = pan.proto[:, 0].tolist()
        bt = pan.bt[:, 0].tolist()
        blk = pan.block[:, 0].tolist()
        return [ActionIndex(int(p), int(t), int(b)) for p, t, b in zip(proto, bt, blk, strict=True)]

    def sample(self) -> list[ActionIndex]:
        if self.use_full_behavior:
            return self.sample_full(0.0)
        return super().sample()

    # ------------------------------------------------------------------ A1: normaliser
    def log_prob(self, actions, logprobs: Tensor | None = None, batch: Tensor | None = None) -> Tensor:
        if not self.use_pan_norm:
            return super().log_prob(actions, logprobs, batch)
        if self._no_grad():
            # sampling: the value is only logged (the TB loss re-evaluates the policy)
            with torch.no_grad():
                return self._log_prob_pan(actions)
        return self._log_prob_pan(actions)

    def _log_prob_pan(self, actions) -> Tensor:
        engine = self.engine
        proto_a, glob_a = self._action_global_idx(actions)
        u_a = engine.u_of_actions(self.model, self.emb, proto_a)
        l_a = engine.logits_of_actions(self.model, u_a, glob_a)

        if self._no_grad() and self.pan_cfg.sampling_logprob == "cached_exact":
            pan = self._ensure_pan(n_beh=0, n_is=0)
            log_Z = pan.log_Zc
            self.diag["lane_a/logZ_source_cached"] = 1.0
        else:
            K = int(self.pan_cfg.num_samples)
            pan = self._ensure_pan(n_beh=0, n_is=K)
            sl = pan.is_slice
            u_sel = pan.u_sel[:, sl]
            glob = pan.glob[:, sl]
            log_q = pan.log_q[:, sl]
            l_s = engine.logits_of(self.model, u_sel, glob)
            w = l_s - log_q  # log importance weights
            log_Z = torch.logsumexp(w, dim=1) - math.log(max(K, 1))
            with torch.no_grad():
                wn = torch.softmax(w, dim=1)
                ess = 1.0 / (wn * wn).sum(1).clamp_min(1e-30)
                self.diag["lane_a/ess_frac"] = float((ess / K).mean())
                self.diag["lane_a/logZ_hat_minus_logZc"] = float((log_Z - pan.log_Zc).mean())
                self.diag["lane_a/logw_std"] = float(w.std(dim=1).mean())
                self.diag["lane_a/q_mass_top1"] = float(
                    (pan.log_q[:, sl].max(dim=1).values).exp().mean()
                )
        log_pf = l_a - log_Z
        with torch.no_grad():
            self.diag["lane_a/clamp_frac"] = float((log_pf > 0).float().mean())
            self.diag["lane_a/log_Nlegal"] = float(pan.log_Nleg.mean())
            self.diag["lane_a/degenerate_states"] = float(pan.n_degenerate)
            self.diag["lane_a/cache_age"] = float(engine.iteration - engine.cache.version)
            if pan.n_is > 0:
                drift = (l_s - pan.l_c[:, pan.is_slice]).detach()
                self.diag["lane_a/cache_drift_mean"] = float(drift.mean())
                self.diag["lane_a/cache_drift_std"] = float(drift.std())
            if engine.should_diag(engine.iteration, self.role):
                engine._last_diag_it = engine.iteration
                ex = engine.exact_stats(self.model, self.emb, self.protocol_mask, self.level_state)
                r = ex["rows"]
                err = (log_Z[r] - ex["log_Z"]).abs()
                self.diag["lane_a/abs_logZ_err"] = float(err.mean())
                self.diag["lane_a/abs_logZ_err_max"] = float(err.max())
                self.diag["lane_a/log_Meff"] = float(ex["log_meff"].mean())
                self.diag["lane_a/top1_mass"] = float(ex["top1"].mean())
                self.diag["lane_a/top10_mass"] = float(ex["top10"].mean())
                self.diag["lane_a/top100_mass"] = float(ex["top100"].mean())
                self.diag["lane_a/diag_n_entries"] = float(ex["n_entries"])
        if self.pan_cfg.clamp_logprob:
            log_pf = log_pf.clamp(max=0.0)
        engine.stats.update(self.diag)
        return log_pf
