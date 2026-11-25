"""Lane B - oracle-reuse learning engine.

Three coupled parts, all gated by `cfg.lane_b.enable` (default False):

B1  replay of docked trajectories: every valid sampled trajectory (states keep their sampling-time pose
    features, raw `obj_props`, sampling-time condition) is kept in a bounded buffer.
B2  rank-prioritised sampling: p_i ~ 1/(kappa*|D| + rank_i) mixed with a uniform part, rank by the
    evaluation criterion (Vina among QED>0.5 first, then the rest).
B3  Bayesian condition-hindsight: a replayed object x is relabelled with c' ~ p(c|x), where for an
    idealised conditional GFlowNet  p(c|x) = p(c) R_c(x)/Z(c) / sum_c' p(c') R_c'(x)/Z(c').
    The posterior is approximated by self-normalised importance sampling with the prior p(c) as proposal
    (M candidates from `task.sample_conditional_information`, weights exp(log R_c(x) - log Z_theta(c)),
    resample one).  See docs/lane_b/REPORT.md for the derivation and the failure modes of the idealisation.
B4  updates-to-data ratio: after the baseline step on the fresh batch, `utd - 1` extra optimizer steps on
    replay batches built by B2 + B3.  lr / optimizer / clipping are untouched.

With `enable=False` nothing of the above is constructed, no hook is registered, no RNG is consumed, and
`VinaMOOTrainer_laneB.train_batch` is `VinaMOOTrainer_semla.train_batch`.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch_geometric.data as gd
from torch import Tensor

from gflownet.data.replay_buffer import detach_and_cpu
from gflownet.online_trainer import model_grad_norm

from cgflow.config import Config
from cgflow.tasks.semla_vina import VinaMOOTrainer_semla

# `BaseTask.reward_to_obj_properties` clips the objectives at 1e-4; a failed UniDock call gives vina
# score 0.0 -> reward clip(-0.1*0.0, min=1e-5) = 1e-5 -> clipped to exactly 1e-4.
FAILED_DOCKING_REWARD = 1e-4
QED_THRESHOLD = 0.5  # the evaluation protocol keeps molecules with QED > 0.5


def _rss_gb() -> float:
    """Resident set size of this process in GB (Linux, no extra dependency)."""
    try:
        with open("/proc/self/statm", "r", encoding="utf8") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / 1024**3
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------------------------------
# B3a: the TB-implied log-partition, estimated by kernel regression (hindsight="posterior_cal")
# ----------------------------------------------------------------------------------------------------
def nw_log_partition(
    q_beta: Tensor,
    q_pref: Tensor,
    d_beta: Tensor,
    d_pref: Tensor,
    d_v: Tensor,
    h_beta: float,
    h_pref: float,
    beta_max: float = 64.0,
    min_kernel: float = 1e-6,
) -> tuple[Tensor, Tensor, float]:
    """Nadaraya-Watson estimate of E[v | c] with a Gaussian kernel in (beta/beta_max, w_0).

    At the TB optimum every trajectory satisfies  log Z(c) = log R_c(x) - sum_t log P_F(a_t|s_t),
    so the least-squares-optimal value of the log-partition head at a condition c is the conditional mean
    of `v = log R - sum log P_F` given c.  Estimating that conditional mean directly from the stored
    trajectories gives the same target without waiting for the head to converge.

    Returns (estimate [Q], usable mask [Q], mean effective neighbour count).
    """
    qb = (q_beta.double() / beta_max).view(-1, 1)
    db = (d_beta.double() / beta_max).view(1, -1)
    qp = q_pref.double().view(-1, 1)
    dp = d_pref.double().view(1, -1)
    log_k = -0.5 * (((qb - db) / h_beta) ** 2 + ((qp - dp) / h_pref) ** 2)
    mx = log_k.max(dim=1, keepdim=True).values
    w = torch.exp(log_k - mx)
    denom = w.sum(dim=1)
    est = (w * d_v.double().view(1, -1)).sum(dim=1) / denom
    n_eff = float((denom.pow(2) / w.pow(2).sum(dim=1)).mean())
    ok = mx.squeeze(1) > math.log(min_kernel)  # at least one data point within ~5 bandwidths
    return est.float(), ok, n_eff


# ----------------------------------------------------------------------------------------------------
# B3: Bayesian condition-hindsight
# ----------------------------------------------------------------------------------------------------
def posterior_hindsight(
    obj_props: Tensor,
    task: Any,
    logZ_fn: Callable[[Tensor], Tensor],
    num_candidates: int,
    train_it: int,
    generator: torch.Generator,
    relabel_mask: Tensor | None = None,
    base_cond: dict[str, Tensor] | None = None,
    min_ess: float = 0.0,
) -> tuple[dict[str, Tensor], Tensor, dict[str, float]]:
    """Relabel the conditions of a batch of already-evaluated objects with a draw from p(c|x).

    For an idealised conditional GFlowNet trained to p_theta(x|c) = R_c(x)/Z(c), the joint over
    (condition, object) is p(c) R_c(x)/Z(c) and therefore

        p(c|x) = p(c) exp( log R_c(x) - log Z(c) ) / normaliser.

    The normaliser is intractable (c is continuous: beta ~ U(0,64), w ~ Dirichlet(1,1)), so we use
    self-normalised importance sampling with the prior as proposal:
    draw c_1..c_M ~ p(c), weight w_m ∝ exp(log R_{c_m}(x) - log Z_theta(c_m)), resample one.
    `log Z_theta` is the model's own learned log-partition head.

    Parameters
    ----------
    obj_props: Tensor [N, K]
        Raw objectives of the stored objects (NOT the scalarised/tempered log reward).
    task:
        Needs `sample_conditional_information(n, train_it)` (the prior p(c), returning the
        'encoding' plus its raw parts) and `cond_info_to_logreward(cond_info, obj_props)`.
    logZ_fn:
        Maps a condition-information dict (as the task produces it, i.e. with 'encoding', 'beta',
        'preferences') to log Z [B] (no grad).  Either the model's logZ head or the kernel-regressed
        TB-implied log-partition (`make_calibrated_logZ`).
    num_candidates: int
        M, the number of prior draws per object.
    generator: torch.Generator
        CPU generator used for the resampling step (and nothing else), so the lane never perturbs the
        baseline RNG streams through this function.
    relabel_mask: Tensor [N] of bool or None
        Rows with False keep `base_cond` (requires `base_cond`).
    base_cond: dict or None
        The sampling-time conditions, used for the untouched rows and for the diagnostics.
    min_ess: float
        Rows whose normalised effective sample size is below this keep their own condition (requires
        `base_cond`).  0.0 disables the safeguard.

    Returns
    -------
    cond_info: dict[str, Tensor]
        The new per-object conditions (same keys/shapes as the task's conditional information).
    log_rewards: Tensor [N]
        log R_{c'}(x) under the new conditions (recomputed through the task, so it is exactly the
        quantity the TB loss needs).
    info: dict[str, float]
        Diagnostics (effective sample size, how the temperature moved, ...).
    """
    n = int(obj_props.shape[0])
    m = int(num_candidates)
    assert n > 0 and m > 0

    cand = task.sample_conditional_information(n * m, train_it)
    rep_obj = obj_props.repeat_interleave(m, dim=0)
    log_r = task.cond_info_to_logreward(cand, rep_obj).reshape(n, m).double()
    log_z = logZ_fn(cand).reshape(n, m).double()

    log_w = log_r - log_z
    log_w = log_w - log_w.max(dim=1, keepdim=True).values
    w = torch.softmax(log_w, dim=1)
    ess = (1.0 / (w * w).sum(dim=1)) / m  # normalised effective sample size in [1/m, 1]

    pick = torch.multinomial(w.float(), 1, generator=generator).squeeze(1)
    flat = torch.arange(n, dtype=torch.long) * m + pick
    out = {k: v[flat].clone() for k, v in cand.items()}

    low_ess = ess < float(min_ess) if min_ess > 0.0 else torch.zeros(n, dtype=torch.bool)
    if relabel_mask is not None or min_ess > 0.0:
        assert base_cond is not None, "relabel_mask / min_ess require base_cond"
        do = torch.ones(n, dtype=torch.bool) if relabel_mask is None else relabel_mask.bool()
        keep = (~do) | low_ess
        if bool(keep.any()):
            for k in out:
                out[k][keep] = base_cond[k][keep].to(out[k].dtype)

    log_rewards = task.cond_info_to_logreward(out, obj_props)

    info: dict[str, float] = {
        "hs_ess": float(ess.mean()),
        "hs_logw_spread": float((log_w.max(dim=1).values - log_w.min(dim=1).values).mean()),
        "hs_skipped_lowess_frac": float(low_ess.float().mean()),
    }
    if base_cond is not None and "beta" in base_cond and "beta" in out:
        db = (out["beta"] - base_cond["beta"]).float()
        info["hs_beta_up"] = float((db > 0).float().mean())
        info["hs_beta_down"] = float((db < 0).float().mean())
        info["hs_dbeta_mean"] = float(db.mean())
        info["hs_beta_mean"] = float(out["beta"].float().mean())
        info["hs_beta_mean_old"] = float(base_cond["beta"].float().mean())
    if base_cond is not None and "preferences" in base_cond and "preferences" in out:
        info["hs_pref_vina_mean"] = float(out["preferences"][:, 0].float().mean())
        info["hs_pref_vina_mean_old"] = float(base_cond["preferences"][:, 0].float().mean())
    return out, log_rewards, info


def make_calibrated_logZ(
    cal_beta: Tensor,
    cal_pref: Tensor,
    cal_v: Tensor,
    head_fn: Callable[[dict[str, Tensor]], Tensor],
    h_beta: float,
    h_pref: float,
    beta_max: float = 64.0,
    stats: dict[str, float] | None = None,
) -> Callable[[dict[str, Tensor]], Tensor]:
    """log Z(c) estimator for `hindsight="posterior_cal"`: kernel regression of the TB-implied
    `v = log R - sum log P_F` over the recent trajectories, with the model's head as a fallback for
    query conditions that have no data within ~5 bandwidths."""

    def fn(cond: dict[str, Tensor]) -> Tensor:
        pref = cond["preferences"][:, 0] if "preferences" in cond else torch.ones_like(cond["beta"])
        est, ok, n_eff = nw_log_partition(
            cond["beta"], pref, cal_beta, cal_pref, cal_v, h_beta, h_pref, beta_max
        )
        if not bool(ok.all()):
            est = torch.where(ok, est, head_fn(cond).to(est.dtype))
        if stats is not None:
            stats["cal_neff"] = n_eff
            stats["cal_head_fallback_frac"] = float((~ok).float().mean())
            stats["cal_logz_mean"] = float(est.mean())
        return est

    return fn


# ----------------------------------------------------------------------------------------------------
# B1 + B2: the buffer
# ----------------------------------------------------------------------------------------------------
class ReplayEngine:
    """Bounded buffer of docked trajectories with rank-prioritised sampling.

    Stored objects are *references* to the trajectory dicts produced by the sampler (so the MolGraph
    states keep their sampling-time cached node features / poses; nothing is re-run through the pose
    model at replay time).  It is registered as a `DataSource` sampling hook, i.e. it is called with the
    full trajectory dicts right after the oracle has been queried and the log-rewards computed.
    """

    def __init__(self, cfg: Config, ctx: Any = None):
        c = cfg.lane_b
        self.cfg = cfg
        self.c = c
        self.ctx = ctx  # only needed for the (optional) feature fingerprints
        self.capacity = int(c.capacity)
        # dedicated stream: never touches `get_worker_rng()` / the global torch RNG
        self.rng = np.random.default_rng(1_000_003 * (int(cfg.seed) + 1) + 11)
        self.items: list[dict] = []
        self.current_iter: int = 0
        self.n_inserted = 0
        self.n_skipped_invalid = 0
        self.n_skipped_failed = 0
        self.n_evicted = 0
        self._order: np.ndarray | None = None  # items sorted best-first
        self._obj: np.ndarray | None = None

    # -- bookkeeping ----------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def _invalidate(self) -> None:
        self._order = None
        self._obj = None

    def objectives(self) -> np.ndarray:
        if self._obj is None:
            if len(self.items) == 0:
                self._obj = np.zeros((0, 2), dtype=np.float64)
            else:
                self._obj = np.stack([r["obj"] for r in self.items])
        return self._obj

    def order(self) -> np.ndarray:
        """Indices of the stored trajectories, best first.

        Ranking criterion == the evaluation criterion: molecules with QED > 0.5 first, inside each group
        by Vina (i.e. by the vina reward, which is -0.1 * vina, descending).
        """
        if self._order is None:
            obj = self.objectives()
            if obj.shape[0] == 0:
                self._order = np.zeros((0,), dtype=np.int64)
            else:
                group = (obj[:, 1] <= QED_THRESHOLD).astype(np.int64)  # 0 = passes the QED filter
                self._order = np.lexsort((-obj[:, 0], group))
        return self._order

    def ranks(self) -> np.ndarray:
        order = self.order()
        r = np.empty(order.shape[0], dtype=np.float64)
        r[order] = np.arange(order.shape[0], dtype=np.float64)
        return r

    def priorities(self) -> np.ndarray:
        n = len(self.items)
        if n == 0:
            return np.zeros((0,), dtype=np.float64)
        if self.c.priority == "uniform":
            return np.full(n, 1.0 / n, dtype=np.float64)
        if self.c.priority != "rank":
            raise ValueError(f"lane_b.priority must be 'rank' or 'uniform', got {self.c.priority!r}")
        denom = max(float(self.c.rank_kappa) * n, 1.0)
        p = 1.0 / (denom + self.ranks())
        p = p / p.sum()
        u = float(self.c.uniform_mix)
        if u > 0.0:
            p = (1.0 - u) * p + u / n
            p = p / p.sum()
        return p

    # -- insertion ------------------------------------------------------------------------------
    def __call__(self, trajs, rewards, obj_props, cond_info) -> dict[str, Any]:
        """DataSource sampling hook signature."""
        self.insert(trajs)
        return {}

    def insert(self, trajs) -> None:
        protected = self._protected_ids()
        for slot, t in enumerate(trajs):
            if not bool(t.get("is_valid", True)):
                self.n_skipped_invalid += 1
                continue
            obj = np.asarray(t["obj_props"].detach().cpu(), dtype=np.float64).reshape(-1)
            if self.c.skip_failed_docking and obj[0] <= FAILED_DOCKING_REWARD * (1.0 + 1e-6):
                self.n_skipped_failed += 1
                continue
            rec: dict[str, Any] = {"traj": t, "obj": obj, "it": int(self.current_iter), "slot": int(slot)}
            # data for the TB-implied log-partition estimate (hindsight="posterior_cal"):
            # v = log R_c(x) - sum_t log P_F(a_t|s_t) at the sampling-time condition c.
            ci = t.get("cond_info", {})
            if "fwd_logprob" in t and "beta" in ci:
                rec["cal_beta"] = float(ci["beta"])
                rec["cal_pref"] = float(ci["preferences"][0]) if "preferences" in ci else 1.0
                rec["cal_v"] = float(t["log_reward"]) - float(t["fwd_logprob"])
            if self.c.selfcheck_replay_data and self.ctx is not None:
                rec["fp"] = self.fingerprint(t)
            if len(self.items) >= self.capacity:
                self._evict_one(protected)
            self.items.append(rec)
            self.n_inserted += 1
        self._invalidate()

    def _protected_ids(self) -> set[int]:
        k = int(self.c.protected_elites)
        if k <= 0 or len(self.items) == 0:
            return set()
        return {id(self.items[i]) for i in self.order()[:k]}

    def _evict_one(self, protected: set[int]) -> None:
        """FIFO eviction by age; the protected elites are skipped (unless everything is protected)."""
        for i, rec in enumerate(self.items):
            if id(rec) not in protected:
                self.items.pop(i)
                self.n_evicted += 1
                self._invalidate()
                return
        self.items.pop(0)
        self.n_evicted += 1
        self._invalidate()

    # -- sampling -------------------------------------------------------------------------------
    def sample(self, k: int) -> tuple[list[dict], np.ndarray]:
        """Draw `k` distinct stored trajectories without replacement, ~ `priorities()`.

        Uses the Gumbel-top-k trick (exact weighted sampling without replacement, Efraimidis-Spirakis)
        on the engine's own RNG stream.
        """
        n = len(self.items)
        k = int(min(k, n))
        if k <= 0:
            return [], np.zeros((0,), dtype=np.float64)
        p = self.priorities()
        keys = np.log(p) + self.rng.gumbel(size=n)
        if k < n:
            idx = np.argpartition(-keys, k - 1)[:k]
        else:
            idx = np.arange(n)
        idx = idx[np.argsort(-keys[idx])]
        return [self.items[i] for i in idx], p[idx]

    def priority_ess(self) -> float:
        """Normalised effective sample size of the priority distribution (1 = uniform)."""
        p = self.priorities()
        if p.size == 0:
            return float("nan")
        return float(1.0 / np.square(p).sum() / p.size)

    def calibration_data(self, limit: int) -> tuple[Tensor, Tensor, Tensor]:
        """(beta, w_0, v) of the `limit` most recent stored trajectories (the list is chronological:
        entries are appended in sampling order and eviction only removes older ones)."""
        recs = [r for r in self.items[-int(limit) :] if "cal_v" in r]
        if not recs:
            empty = torch.zeros(0)
            return empty, empty, empty
        beta = torch.tensor([r["cal_beta"] for r in recs], dtype=torch.float32)
        pref = torch.tensor([r["cal_pref"] for r in recs], dtype=torch.float32)
        v = torch.tensor([r["cal_v"] for r in recs], dtype=torch.float32)
        return beta, pref, v

    # -- feature-identity self-check ------------------------------------------------------------
    def fingerprint(self, traj: dict) -> str:
        """sha1 of every tensor `ctx.graph_to_Data` produces for every state of the trajectory, plus the
        action indices - i.e. exactly the inputs `algo.construct_batch` derives from a stored trajectory."""
        assert self.ctx is not None
        h = hashlib.sha1()
        for g, a in traj["traj"]:
            d = self.ctx.graph_to_Data(g)
            for key in sorted(d.keys()):
                v = d[key]
                if torch.is_tensor(v):
                    h.update(key.encode())
                    h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
            try:
                h.update(str(tuple(self.ctx.GraphAction_to_ActionIndex(d, a))).encode())
            except Exception:  # pragma: no cover - action encoding is not available for every ctx
                pass
        return h.hexdigest()

    def verify_stored_features(self) -> tuple[int, int]:
        """Re-derive the features of every fingerprinted trajectory; returns (mismatches, checked)."""
        bad = 0
        n = 0
        for rec in self.items:
            if "fp" not in rec:
                continue
            n += 1
            if self.fingerprint(rec["traj"]) != rec["fp"]:
                bad += 1
        return bad, n

    # -- diagnostics ----------------------------------------------------------------------------
    def stats(self) -> dict[str, float]:
        obj = self.objectives()
        out = {
            "buffer_size": float(len(self.items)),
            "buffer_inserted": float(self.n_inserted),
            "buffer_evicted": float(self.n_evicted),
            "buffer_skipped_failed": float(self.n_skipped_failed),
            "buffer_skipped_invalid": float(self.n_skipped_invalid),
            "rss_gb": _rss_gb(),
        }
        if obj.shape[0] > 0:
            out["buffer_vina_mean"] = float(-10.0 * obj[:, 0].mean())
            out["buffer_vina_best"] = float(-10.0 * obj[:, 0].max())
            out["buffer_qed_pass_frac"] = float((obj[:, 1] > QED_THRESHOLD).mean())
        return out


# ----------------------------------------------------------------------------------------------------
# B4: the trainer
# ----------------------------------------------------------------------------------------------------
class VinaMOOTrainer_laneB(VinaMOOTrainer_semla):
    """Baseline trainer + (optional) oracle-reuse learning engine.

    `cfg.lane_b.enable=False` -> this class *is* the baseline: `setup` registers nothing and
    `train_batch` only forwards to the parent.
    """

    def setup(self):
        super().setup()
        self._lane_b_engine: ReplayEngine | None = None
        c = getattr(self.cfg, "lane_b", None)
        if c is None or not bool(c.enable):
            return
        if int(self.cfg.num_workers) != 0:
            raise ValueError(
                "lane_b requires num_workers=0: the replay engine is a sampling hook and must live in the "
                "same process as train_batch (the baseline runs with num_workers=0)."
            )
        if c.priority not in ("rank", "uniform"):
            raise ValueError(f"lane_b.priority must be 'rank' or 'uniform', got {c.priority!r}")
        if c.hindsight not in ("none", "posterior", "posterior_cal"):
            raise ValueError(
                f"lane_b.hindsight must be 'none', 'posterior' or 'posterior_cal', got {c.hindsight!r}"
            )
        if int(c.utd) < 1:
            raise ValueError(f"lane_b.utd must be >= 1, got {c.utd}")
        if not (0.0 <= float(c.hindsight_prob) <= 1.0):
            raise ValueError(f"lane_b.hindsight_prob must be in [0,1], got {c.hindsight_prob}")
        if not (0.0 <= float(c.uniform_mix) <= 1.0):
            raise ValueError(f"lane_b.uniform_mix must be in [0,1], got {c.uniform_mix}")

        engine = ReplayEngine(self.cfg, ctx=self.ctx if bool(c.selfcheck_replay_data) else None)
        self._lane_b_engine = engine
        self.sampling_hooks.append(engine)  # picked up by build_training_data_loader
        self._lane_b_gen = torch.Generator()
        self._lane_b_gen.manual_seed(7919 * (int(self.cfg.seed) + 1) + 13)
        self._lane_b_nonfinite = 0
        print(
            f"[lane_b] enabled: utd={c.utd} replay_batch={c.replay_batch} capacity={c.capacity} "
            f"priority={c.priority}(kappa={c.rank_kappa}, mix={c.uniform_mix}) hindsight={c.hindsight}"
            f"(M={c.hindsight_candidates}, p={c.hindsight_prob}) warmup_iters={c.warmup_iters} "
            f"protected_elites={c.protected_elites} ema_on_extra={c.ema_on_extra} "
            f"lr_sched_on_extra={c.lr_sched_on_extra} selfcheck={c.selfcheck_replay_data}",
            flush=True,
        )

    # -- one optimizer step on an extra (replay) batch -------------------------------------------
    def _lane_b_step(self, loss: Tensor) -> dict[str, float]:
        """Exactly `StandardOnlineTrainer.step` (same optimizers, same clipping, same EMA); the LR
        schedulers are only advanced if `lane_b.lr_sched_on_extra` is set, so that by default the lane
        adds gradient steps without changing the baseline's learning-rate schedule."""
        loss.backward()
        with torch.no_grad():
            g0 = model_grad_norm(self.model)
            self.clip_grad_callback(self.model.parameters())
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        if bool(self.cfg.lane_b.lr_sched_on_extra):
            self.lr_sched.step()
            self.lr_sched_Z.step()
        if bool(self.cfg.lane_b.ema_on_extra) and self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
        return {"grad_norm": float(g0)}

    def _lane_b_logZ(self, cond_info: dict[str, Tensor]) -> Tensor:
        """log Z(c) from the model's own head (the literal lane-spec estimator)."""
        with torch.no_grad():
            return self.model.logZ(cond_info["encoding"].to(self.device))[:, 0].detach().float().cpu()

    def _lane_b_build_batch(self, trajs: list[dict], cond_info: dict[str, Tensor], log_rewards: Tensor) -> gd.Batch:
        """Mirror of `DataSource.create_batch` for replayed trajectories.

        The stored trajectory dicts are never mutated: the per-trajectory dict is shallow-copied and only
        `cond_info` / `log_reward` are replaced, so the buffer keeps its sampling-time condition and reward.
        """
        items = []
        for i, t in enumerate(trajs):
            t2 = dict(t)
            t2["cond_info"] = {k: v[i] for k, v in cond_info.items()}
            t2["log_reward"] = log_rewards[i]
            items.append(t2)
        items = detach_and_cpu(items)
        ci = torch.stack([t["cond_info"]["encoding"] for t in items])
        lr = torch.stack([t["log_reward"] for t in items])
        batch = self.algo.construct_batch(items, ci, lr)
        batch.num_online = 0
        batch.num_offline = len(items)
        batch.extra_info = {}
        if "preferences" in items[0]["cond_info"]:
            batch.preferences = torch.stack([t["cond_info"]["preferences"] for t in items])
        if "focus_dir" in items[0]["cond_info"]:
            batch.focus_dir = torch.stack([t["cond_info"]["focus_dir"] for t in items])
        if self.ctx.has_n() and self.cfg.algo.tb.do_predict_n:
            log_ns = [self.ctx.traj_log_n(t["traj"]) for t in items]
            batch.log_n = torch.tensor([i[-1] for i in log_ns], dtype=torch.float32)
            batch.log_ns = torch.tensor(sum(log_ns, start=[]), dtype=torch.float32)
        batch.obj_props = torch.stack([t["obj_props"] for t in items])
        return batch

    def _lane_b_replay_batch(self, train_it: int) -> tuple[gd.Batch, dict[str, float]]:
        engine = self._lane_b_engine
        assert engine is not None
        c = self.cfg.lane_b
        recs, probs = engine.sample(int(c.replay_batch))
        trajs = [r["traj"] for r in recs]
        obj_props = torch.stack([t["obj_props"] for t in trajs]).float()
        base_cond = {k: torch.stack([t["cond_info"][k] for t in trajs]) for k in trajs[0]["cond_info"]}
        info: dict[str, float] = {
            "replay_age_mean": float(np.mean([train_it - r["it"] for r in recs])),
            "replay_age_max": float(np.max([train_it - r["it"] for r in recs])),
            "replay_vina_mean": float(-10.0 * np.mean([r["obj"][0] for r in recs])),
            "replay_qed_mean": float(np.mean([r["obj"][1] for r in recs])),
            "replay_prob_mean_ratio": float(np.mean(probs) * len(engine)),
        }
        if c.hindsight in ("posterior", "posterior_cal"):
            logz_fn: Callable[[dict[str, Tensor]], Tensor] = self._lane_b_logZ
            if c.hindsight == "posterior_cal":
                cal_beta, cal_pref, cal_v = engine.calibration_data(int(c.cal_history))
                if cal_v.numel() >= int(c.cal_min_samples):
                    cal_stats: dict[str, float] = {}
                    logz_fn = make_calibrated_logZ(
                        cal_beta,
                        cal_pref,
                        cal_v,
                        self._lane_b_logZ,
                        float(c.cal_bandwidth_beta),
                        float(c.cal_bandwidth_pref),
                        beta_max=float(self.task.temperature_conditional.upper_bound),
                        stats=cal_stats,
                    )
                    info["cal_used"] = 1.0
                    info["cal_n"] = float(cal_v.numel())
                    info["cal_v_mean"] = float(cal_v.mean())
                else:  # too few stored trajectories -> fall back to the model's head
                    info["cal_used"] = 0.0
                    info["cal_n"] = float(cal_v.numel())
                    cal_stats = {}
            mask = torch.rand(len(trajs), generator=self._lane_b_gen) < float(c.hindsight_prob)
            cond_info, log_rewards, hs_info = posterior_hindsight(
                obj_props,
                self.task,
                logz_fn,
                int(c.hindsight_candidates),
                train_it,
                self._lane_b_gen,
                relabel_mask=mask,
                base_cond=base_cond,
                min_ess=float(c.hindsight_min_ess),
            )
            info.update(hs_info)
            info.update(cal_stats if c.hindsight == "posterior_cal" else {})
            info["hs_relabelled_frac"] = float(mask.float().mean())
        else:
            cond_info = base_cond
            log_rewards = torch.stack([t["log_reward"] for t in trajs])
        return self._lane_b_build_batch(trajs, cond_info, log_rewards), info

    # -- the training iteration ------------------------------------------------------------------
    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> dict[str, Any]:
        info = super().train_batch(batch, epoch_idx, batch_idx, train_it)
        engine = getattr(self, "_lane_b_engine", None)
        if engine is None:
            return info  # lane off -> baseline behaviour

        c = self.cfg.lane_b
        engine.current_iter = train_it
        t_start = time.time()
        t_build = 0.0
        t_grad = 0.0
        extra_losses: list[float] = []
        replay_info: dict[str, float] = {}

        ready = len(engine) >= max(1, int(c.replay_batch)) and train_it > int(c.warmup_iters)
        if ready and int(c.utd) > 1:
            self.model.train()
            for k in range(1, int(c.utd)):
                t0 = time.time()
                rbatch, rinfo = self._lane_b_replay_batch(train_it)
                rbatch = rbatch.to(self.device)
                t_build += time.time() - t0
                t0 = time.time()
                loss, _ = self.algo.compute_batch_losses(self.model, rbatch)
                if torch.isfinite(loss):
                    self._lane_b_step(loss)
                    extra_losses.append(float(loss))
                    info[f"lane_b/loss_extra_{k}"] = float(loss)
                else:
                    # never kill a 19 h run because of one pathological replay batch
                    self._lane_b_nonfinite += 1
                    self.opt.zero_grad()
                    self.opt_Z.zero_grad()
                t_grad += time.time() - t0
                if k == 1:
                    replay_info = rinfo
                del rbatch, loss

        if bool(c.selfcheck_replay_data) and engine.ctx is not None:
            bad, checked = engine.verify_stored_features()
            info["lane_b/selfcheck_mismatch"] = float(bad)
            info["lane_b/selfcheck_checked"] = float(checked)

        for k, v in engine.stats().items():
            info[f"lane_b/{k}"] = v
        for k, v in replay_info.items():
            info[f"lane_b/{k}"] = v
        if extra_losses:
            info["lane_b/loss_extra_mean"] = float(np.mean(extra_losses))
            info["lane_b/loss_extra_last"] = float(extra_losses[-1])
        info["lane_b/n_extra_steps"] = float(len(extra_losses))
        info["lane_b/nonfinite_extra"] = float(self._lane_b_nonfinite)
        info["lane_b/prio_ess"] = engine.priority_ess()
        info["lane_b/time_build"] = t_build
        info["lane_b/time_grad"] = t_grad
        info["lane_b/time_total"] = time.time() - t_start
        if torch.cuda.is_available():
            info["lane_b/gpu_peak_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
        return info
