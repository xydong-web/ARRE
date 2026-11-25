"""Lane D - pose-aware proxy gating: propose F*64 candidates, dock the best 64.

Mechanism (see docs/lane_d/REPORT.md for the full write-up)
-----------------------------------------------------------
D1  Per iteration the sampler draws F chunks of ``algo.num_from_policy``
    trajectories (F = ``lane_d.oversample``), i.e. a candidate pool of F*64,
    each chunk exactly as the baseline draws its single batch.
D2  Every finished candidate is described by a cheap deterministic pose-aware
    feature vector (``lane_d_features``): protein-ligand contact shells from
    the SemlaFlow-predicted pose, clash / burial geometry, 2D descriptors and a
    Morgan count fingerprint.
D3  A bootstrapped ensemble of M small MLPs (``lane_d_proxy``) predicts the
    UniDock reward r = clip(-0.1*vina, 1e-5) and is refit online, warm started,
    after every batch of fresh labels.
D4  Acquisition.  Until ``warmup_labels`` labels exist and the *unbiased*
    prequential Spearman of the proxy reaches ``min_rho``, the first 64
    candidates are taken - which is exactly an on-policy batch, so the lane
    degrades to the baseline by construction while the proxy is poor.
    Afterwards the pool is split into ``strata`` equal-count beta quantile
    bins; inside each bin a fixed fraction ``random_frac`` of the quota is
    drawn uniformly at random and the rest greedily by
        a(x) = w_vina * mu(x) + w_qed * QED(x) + kappa * w_vina * sigma(x)
    (``score_mode="scalarized"``) or  mu(x) + kappa*sigma(x) (``"vina"``),
    with candidates below ``qed_floor`` pushed behind every eligible one.

Exactly ``algo.num_from_policy`` trajectories are handed to the oracle every
iteration, so the docking budget and the reward definition are untouched.
"""

from __future__ import annotations

import pathlib
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from gflownet.data.data_source import DataSource
from rxnflow.base.gflownet.sqlite_log import CustomSQLiteLogHook

from cgflow.config import Config
from cgflow.tasks.semla_vina import VinaMOOTrainer_semla
from cgflow.utils import extract_pocket

from .lane_d_config import LaneDConfig
from .lane_d_features import FEATURE_DIM, IDX_QED, featurize_batch, load_pocket_atoms
from .lane_d_proxy import LabelBuffer, ProxyEnsemble, pearson, spearman

_BIG = 1.0e6  # soft QED floor: eligible candidates always outrank ineligible ones


def decide_gate(gate_on: bool, eligible: bool, rho: float, min_rho: float, hysteresis: float) -> bool:
    """State machine of the warm-up gate (pure, so it can be unit tested).

    ``eligible`` collects the hard preconditions (acting enabled, a fitted
    proxy, a pool larger than the batch, enough labels).  Once those hold the
    gate follows the *unbiased* prequential Spearman ``rho`` measured on the
    uniformly drawn picks, with hysteresis so it cannot flap: it switches on at
    ``min_rho`` and only back off below ``min_rho - hysteresis``.  A NaN rho
    (not enough random picks yet) always means "off".
    """
    if not eligible:
        return False
    if gate_on:
        return not (np.isnan(rho) or rho < float(min_rho) - float(hysteresis))
    return (not np.isnan(rho)) and rho >= float(min_rho)


def split_counts(total: int, k: int) -> list[int]:
    """Split ``total`` into ``k`` almost-equal integer parts that sum to total."""
    base, rem = divmod(int(total), int(k))
    return [base + (1 if i < rem else 0) for i in range(int(k))]


class PoseProxyGate:
    """Holds the proxy, the label buffer and the acquisition rule of lane D."""

    def __init__(self, cfg: Config, ctx, device: torch.device, seed: int):
        self.cfg = cfg
        self.dcfg: LaneDConfig = cfg.lane_d
        d = self.dcfg
        if d.use_vina_local:
            raise NotImplementedError(
                "lane_d.use_vina_local is not implemented: a Vina local-opt feature needs a "
                "pdbqt round trip per candidate and cannot be produced inside the 2 s/iteration "
                "budget for F*64 candidates.  See docs/lane_d/REPORT.md (limitations)."
            )
        if int(d.oversample) < 1:
            raise ValueError(f"lane_d.oversample must be >= 1, got {d.oversample}")
        if int(d.strata) < 1:
            raise ValueError(f"lane_d.strata must be >= 1, got {d.strata}")
        if d.score_mode not in ("scalarized", "vina"):
            raise ValueError(f"lane_d.score_mode must be 'scalarized' or 'vina', got {d.score_mode!r}")
        if not (0.0 <= float(d.random_frac) < 1.0):
            raise ValueError(f"lane_d.random_frac must be in [0, 1), got {d.random_frac}")

        objectives = list(cfg.task.moo.objectives)
        if "vina" not in objectives:
            raise ValueError(f"lane D needs a 'vina' objective, got {objectives}")
        self.i_vina = objectives.index("vina")
        self.i_qed = objectives.index("qed") if "qed" in objectives else None
        if d.score_mode == "scalarized" and set(objectives) != {"vina", "qed"}:
            raise ValueError(
                f"lane_d.score_mode='scalarized' assumes objectives == [vina, qed]; got {objectives}. "
                "Use lane_d.score_mode=vina for other objective sets."
            )

        self.ctx = ctx
        self.device = torch.device(device)
        # dedicated RNG stream: seeded from cfg.seed, never touches the worker RNG that
        # drives the baseline sampler, so turning the lane on cannot perturb the baseline
        # stream through a shared generator.
        self.rng = np.random.default_rng(int(seed) * 7919 + 20_251_018)

        pocket_pdb = extract_pocket.extract_pocket_crossdocked_v1(ctx.protein_path, ctx.ref_ligand_path)
        self.pocket = load_pocket_atoms(pocket_pdb, ctx.pocket_center)

        if d.fit:
            self.buffer: LabelBuffer | None = LabelBuffer(
                FEATURE_DIM, int(d.buffer_capacity), int(d.ensemble), float(d.bootstrap_p), self.rng
            )
            self.proxy: ProxyEnsemble | None = ProxyEnsemble(
                FEATURE_DIM,
                n_members=int(d.ensemble),
                hidden=int(d.hidden),
                lr=float(d.lr),
                lambda_rank=float(d.lambda_rank),
                rank_tau=float(d.rank_tau),
                device=self.device,
                seed=int(seed),
            )
        else:
            self.buffer, self.proxy = None, None

        self.gate_on = False
        self.rho_ema_val = float("nan")          # diagnostic EMA over selected batches
        self.rho_rand = float("nan")             # gate statistic: unbiased, windowed
        self._rand_pred: deque[float] = deque(maxlen=int(d.rand_window))
        self._rand_true: deque[float] = deque(maxlen=int(d.rand_window))
        self._pending: dict | None = None
        self._log: dict[str, float] = {}

    # ---------------------------------------------------------------- selection
    def select(self, pool: list[dict], n_select: int) -> list[int]:
        """Score the pool and return the indices of the ``n_select`` candidates to dock."""
        d = self.dcfg
        n_pool = len(pool)
        t0 = time.time()

        mols, valid = [], np.zeros(n_pool, dtype=bool)
        for i, tr in enumerate(pool):
            mol = None
            if bool(tr.get("is_valid", True)):
                try:
                    mol = self.ctx.graph_to_obj(tr["result"])
                    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
                        mol = None
                except Exception:
                    mol = None
            valid[i] = mol is not None
            mols.append(mol)
        X = featurize_batch(mols, self.pocket)
        t_feat = time.time() - t0

        qed = X[:, IDX_QED].astype(np.float64)
        beta = np.asarray([float(tr["cond_info"]["beta"]) for tr in pool], dtype=np.float64)
        prefs = np.stack([np.asarray(tr["cond_info"]["preferences"], dtype=np.float64) for tr in pool])

        t1 = time.time()
        have_proxy = self.proxy is not None and self.buffer is not None and len(self.buffer) >= 8
        if have_proxy:
            mu, sigma = self.proxy.predict(X)
        else:
            mu = np.zeros(n_pool, dtype=np.float32)
            sigma = np.zeros(n_pool, dtype=np.float32)
        t_score = time.time() - t1

        if d.score_mode == "scalarized":
            w_v = prefs[:, self.i_vina]
            w_q = prefs[:, self.i_qed] if self.i_qed is not None else np.zeros(n_pool)
            acq = w_v * mu + w_q * qed + float(d.kappa) * w_v * sigma
        else:
            acq = mu + float(d.kappa) * sigma
        acq = np.nan_to_num(np.asarray(acq, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        acq = np.where(valid, acq, -np.inf)

        n_labels = self.buffer.n_total if self.buffer is not None else 0
        eligible = bool(d.act) and have_proxy and n_pool > n_select and n_labels >= int(d.warmup_labels)
        self.gate_on = decide_gate(
            self.gate_on, eligible, self.rho_rand, float(d.min_rho), float(d.rho_hysteresis)
        )

        if self.gate_on:
            sel, is_rand = self._stratified_select(acq, qed, beta, n_select)
        else:
            # the first chunk is a complete i.i.d. on-policy batch -> baseline behaviour,
            # and it doubles as the unbiased sample for the proxy diagnostics
            sel = list(range(min(n_select, n_pool)))
            is_rand = [True] * len(sel)

        order = np.argsort(np.asarray(sel, dtype=np.int64), kind="mergesort")
        sel = [int(sel[i]) for i in order]
        is_rand_arr = np.asarray([is_rand[i] for i in order], dtype=bool)

        self._pending = dict(
            X=X[sel],
            mu=np.asarray(mu, dtype=np.float64)[sel],
            sigma=np.asarray(sigma, dtype=np.float64)[sel],
            is_rand=is_rand_arr,
            t_feat=t_feat,
            t_score=t_score,
            pool_stats=dict(
                pool_size=float(n_pool),
                pool_invalid_frac=float(1.0 - valid.mean()) if n_pool else 0.0,
                frac_qed_fail=float(np.mean(qed <= float(d.qed_floor))) if n_pool else 0.0,
                beta_mean_pool=float(beta.mean()) if n_pool else 0.0,
                beta_mean_sel=float(beta[sel].mean()) if sel else 0.0,
                n_labels=float(n_labels),
                gate_on=float(self.gate_on),
                n_random_picks=float(is_rand_arr.sum()),
            ),
        )
        return sel

    def _stratified_select(self, acq, qed, beta, n_select: int) -> tuple[list[int], list[bool]]:
        d = self.dcfg
        n_pool = acq.shape[0]
        k = max(1, min(int(d.strata), int(n_select)))

        order = np.argsort(beta, kind="mergesort")
        stratum_of = np.empty(n_pool, dtype=np.int64)
        pos = 0
        for s, c in enumerate(split_counts(n_pool, k)):
            stratum_of[order[pos: pos + c]] = s
            pos += c
        quotas = split_counts(n_select, k)

        floor_bonus = _BIG * (qed > float(d.qed_floor)).astype(np.float64)
        key_all = acq + floor_bonus

        sel: list[int] = []
        is_rand: list[bool] = []
        for s in range(k):
            members = np.nonzero(stratum_of == s)[0]
            q = min(quotas[s], members.size)
            if q <= 0:
                continue
            n_r = int(round(float(d.random_frac) * q))
            if float(d.random_frac) > 0.0:
                n_r = max(1, n_r)  # >=1 random pick per stratum keeps the support argument exact
            n_r = min(n_r, q)
            perm = self.rng.permutation(members)
            rand_pick, rest = perm[:n_r], perm[n_r:]
            n_g = q - n_r
            if n_g > 0 and rest.size > 0:
                greedy = rest[np.argsort(-key_all[rest], kind="mergesort")[:n_g]]
            else:
                greedy = rest[:0]
            sel.extend(int(i) for i in rand_pick)
            is_rand.extend([True] * len(rand_pick))
            sel.extend(int(i) for i in greedy)
            is_rand.extend([False] * len(greedy))

        if len(sel) < n_select:  # safety net (a stratum was short)
            taken = set(sel)
            rest = np.asarray([i for i in range(n_pool) if i not in taken], dtype=np.int64)
            if rest.size:
                need = n_select - len(sel)
                extra = rest[np.argsort(-key_all[rest], kind="mergesort")[:need]]
                sel.extend(int(i) for i in extra)
                is_rand.extend([False] * len(extra))
        return sel, is_rand

    # ---------------------------------------------------------------- learning
    def observe(self, trajs: list[dict], n_dock_calls: int, t_dock: float, t_sample: float) -> None:
        """Prequential scoring of the fresh labels, then buffer insert + online refit."""
        p = self._pending
        self._pending = None
        log: dict[str, float] = {}
        if p is None:
            return
        d = self.dcfg

        ok = np.asarray([bool(tr.get("is_valid", True)) for tr in trajs], dtype=bool)
        y = np.asarray(
            [float(tr["obj_props"][self.i_vina]) if bool(tr.get("is_valid", True)) else 0.0 for tr in trajs],
            dtype=np.float64,
        )
        mu, sigma, is_rand = p["mu"], p["sigma"], p["is_rand"]
        n = min(len(trajs), mu.shape[0])
        ok, y = ok[:n], y[:n]
        mu, sigma, is_rand = mu[:n], sigma[:n], is_rand[:n]

        have_proxy = self.proxy is not None and self.buffer is not None and len(self.buffer) >= 8
        if have_proxy and ok.sum() >= 3:
            rho_sel = spearman(mu[ok], y[ok])
            r_sel = pearson(mu[ok], y[ok])
            if not np.isnan(rho_sel):
                self.rho_ema_val = (
                    rho_sel
                    if np.isnan(self.rho_ema_val)
                    else float(d.rho_ema) * self.rho_ema_val + (1.0 - float(d.rho_ema)) * rho_sel
                )
            resid = y[ok] - mu[ok]
            denom = float(np.mean(sigma[ok] ** 2))
            log["lane_d/rho_sel"] = float(rho_sel)
            log["lane_d/pearson_sel"] = float(r_sel)
            log["lane_d/rho_ema"] = float(self.rho_ema_val)
            log["lane_d/rmse"] = float(np.sqrt(np.mean(resid**2)))
            log["lane_d/disp_ratio"] = float(np.mean(resid**2) / denom) if denom > 1e-12 else float("nan")

        # unbiased window: only the uniformly drawn picks (all 64 while the gate is off)
        rand_mask = is_rand & ok
        greedy_mask = (~is_rand) & ok
        if have_proxy:
            for a, b in zip(mu[rand_mask].tolist(), y[rand_mask].tolist()):
                self._rand_pred.append(a)
                self._rand_true.append(b)
            if len(self._rand_pred) >= 16:
                self.rho_rand = spearman(np.asarray(self._rand_pred), np.asarray(self._rand_true))
        log["lane_d/rho_rand"] = float(self.rho_rand)
        log["lane_d/rand_window"] = float(len(self._rand_pred))
        if rand_mask.any():
            log["lane_d/rand_mean_r"] = float(y[rand_mask].mean())
        if greedy_mask.any():
            log["lane_d/sel_mean_r"] = float(y[greedy_mask].mean())
        if rand_mask.any() and greedy_mask.any():
            log["lane_d/lift"] = float(y[greedy_mask].mean() - y[rand_mask].mean())

        t_fit = 0.0
        if self.buffer is not None and ok.any():
            self.buffer.add(p["X"][:n][ok], y[ok])
            if self.proxy is not None:
                t2 = time.time()
                loss = self.proxy.fit(
                    self.buffer,
                    epochs=int(d.fit_epochs),
                    max_samples=int(d.fit_max_samples),
                    batch_size=int(d.fit_batch_size),
                    rng=self.rng,
                )
                t_fit = time.time() - t2
                log["lane_d/fit_loss"] = float(loss)
                log["lane_d/fit_steps"] = float(self.proxy.n_fit_steps)

        log.update({f"lane_d/{k}": v for k, v in p["pool_stats"].items()})
        log["lane_d/n_valid_sel"] = float(ok.sum())
        log["lane_d/n_dock_calls"] = float(n_dock_calls)
        log["lane_d/t_feat"] = float(p["t_feat"])
        log["lane_d/t_score"] = float(p["t_score"])
        log["lane_d/t_fit"] = float(t_fit)
        log["lane_d/t_sample"] = float(t_sample)
        log["lane_d/t_oracle"] = float(t_dock)
        log["lane_d/t_overhead"] = float(p["t_feat"] + p["t_score"] + t_fit)
        self._log = log

    def pop_log(self) -> dict[str, float]:
        out, self._log = self._log, {}
        return out


class LaneDDataSource(DataSource):
    """``DataSource`` with one extra iterator: oversample -> gate -> oracle.

    It mirrors :meth:`DataSource.do_sample_model` call for call; the only
    differences are the F chunks and the selection step between sampling and
    ``compute_properties``.
    """

    def do_sample_model_gated(self, model, num_from_policy: int, gate: PoseProxyGate):
        assert self.replay_buffer is None, "lane D does not support the replay buffer"

        def iterator():
            while self.active:
                t = self.current_iter
                p = self.algo.get_random_action_prob(t)
                n_chunks = max(1, int(gate.dcfg.oversample))

                t_sample0 = time.time()
                pool: list[dict] = []
                for _ in range(n_chunks):
                    cond_info = self.task.sample_conditional_information(num_from_policy, t)
                    chunk = self.algo.create_training_data_from_own_samples(
                        model, num_from_policy, cond_info["encoding"], p
                    )
                    self.set_traj_cond_info(chunk, cond_info)
                    pool.extend(chunk)
                t_sample = time.time() - t_sample0

                if n_chunks == 1 and not gate.dcfg.fit:
                    # nothing to gate and no proxy to train: identity selection, zero overhead
                    sel = list(range(min(num_from_policy, len(pool))))
                    gate._pending = None
                else:
                    # with F == 1 select() is the identity (the gate needs n_pool > n_select),
                    # so the proxy is still trained and measured on an untouched on-policy batch
                    sel = gate.select(pool, num_from_policy)
                trajs = [pool[i] for i in sel]

                n_dock_calls = _count_fresh_docking(self.task, self.ctx, trajs)
                t_dock0 = time.time()
                self.compute_properties(trajs, mark_as_online=True)
                t_dock = time.time() - t_dock0
                self.compute_log_rewards(trajs)
                batch_info = self.call_sampling_hooks(trajs)

                gate.observe(trajs, n_dock_calls, t_dock, t_sample)
                batch_info.update(gate.pop_log())
                yield (trajs, batch_info)

        self.iterators.append(iterator)
        return self


def _count_fresh_docking(task, ctx, trajs: list[dict]) -> int:
    """Number of selected molecules that will really invoke UniDock.

    ``cgflow.tasks.semla_vina._run_redocking_caching`` skips any molecule whose
    canonical SMILES is already in ``task.topn_vina`` (the best-2000 cache), so
    the realised oracle cost can be below the 64 candidates handed over -- in
    the baseline as well as here.  Logged so the budget stays auditable.
    """
    try:
        from rdkit import Chem

        n = 0
        for tr in trajs:
            if not bool(tr.get("is_valid", True)):
                continue
            mol = ctx.graph_to_obj(tr["result"])
            if task.topn_vina.get(Chem.MolToSmiles(mol), 0.0) >= 0.0:
                n += 1
        return n
    except Exception:
        return -1


class VinaMOOTrainer_laneD(VinaMOOTrainer_semla):
    """Baseline trainer + the lane D data loader (inert unless ``lane_d.enable``)."""

    lane_d_gate: PoseProxyGate | None = None

    def build_training_data_loader(self) -> DataLoader:
        if not self.cfg.lane_d.enable:
            # byte-for-byte the baseline path
            return super().build_training_data_loader()

        if self.cfg.replay.use:
            raise ValueError("lane D is incompatible with replay.use=true")
        if self.cfg.num_workers != 0:
            raise ValueError(
                "lane D keeps the proxy in the training process; run with num_workers=0 (the project default)"
            )
        n_drawn = self.cfg.algo.num_from_policy
        if not n_drawn:
            raise ValueError("lane D needs algo.num_from_policy > 0")
        model = self._wrap_for_mp(self.sampling_model)

        self.lane_d_gate = PoseProxyGate(self.cfg, self.ctx, self.device, int(self.cfg.seed))
        print(
            f"[lane_d] enabled: F={self.cfg.lane_d.oversample} M={self.cfg.lane_d.ensemble} "
            f"act={self.cfg.lane_d.act} fit={self.cfg.lane_d.fit} strata={self.cfg.lane_d.strata} "
            f"score_mode={self.cfg.lane_d.score_mode} warmup={self.cfg.lane_d.warmup_labels} "
            f"min_rho={self.cfg.lane_d.min_rho} pocket_atoms={len(self.lane_d_gate.pocket)}",
            flush=True,
        )

        src = LaneDDataSource(self.cfg, self.ctx, self.algo, self.task, replay_buffer=None)
        src.do_sample_model_gated(model, n_drawn, self.lane_d_gate)
        if self.cfg.log_dir:
            src.add_sampling_hook(CustomSQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)
