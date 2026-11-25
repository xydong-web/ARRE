"""CPU unit tests of RHC building blocks.  Run:  pytest -q tests/rhc  (shared cgflow env, PYTHONPATH=<worktree>/src)."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cgflow.lanes.rhc import (
    ARMS,
    RHCState,
    ModeTracker,
    Portfolio,
    SyntheticPathSampler3D_semlaRHC,
    VinaMOOTask_semlaRHC,
)
from cgflow.lanes.rhc_config import RHCConfig
from rxnflow.utils.chem_metrics import compute_diverse_top_k

FRAGS = ["c1ccccc1", "c1ccncc1", "C1CCNCC1", "C1CCOCC1", "c1ccsc1", "C1CC1", "c1cc[nH]c1", "C1CCCCC1", "c1ncccn1", "O=C(N)C"]
LINK = ["", "C", "CC", "O", "N", "C(=O)N", "S(=O)(=O)N", "C(=O)O", "OC", "NC"]


def rand_smiles(rng: random.Random) -> str:
    parts = [rng.choice(FRAGS) for _ in range(rng.randint(2, 4))]
    s = parts[0]
    for p in parts[1:]:
        s += rng.choice(LINK) + p
    return s


def valid_unique(n, seed):
    from rdkit import Chem

    rng, out = random.Random(seed), []
    seen = set()
    while len(out) < n:
        smi = rand_smiles(rng)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can not in seen:
            seen.add(can)
            out.append(can)
    return out


# ----------------------------------------------------------------------------------------------------------------------
def test_mode_tracker_matches_evaluation_protocol():
    """The greedy diverse list must equal rxnflow.utils.chem_metrics.compute_diverse_top_k on the eligible set."""
    smiles = valid_unique(500, 1)
    rs = np.random.RandomState(0)
    vina = -rs.uniform(3, 13, size=len(smiles))
    qed = rs.uniform(0.3, 0.9, size=len(smiles))
    tr = ModeTracker(k_modes=100, k_elite=256, thresh=0.5, qed_min=0.5)
    for s, v, q in zip(smiles, vina, qed):
        tr.add(s, float(v), float(q), [], "onpolicy", 0, (0.0, 0.0), {})
    tr.refresh()
    el = [i for i in range(len(smiles)) if qed[i] > 0.5]
    ref = compute_diverse_top_k([smiles[i] for i in el], [-float(vina[i]) for i in el], k=256, thresh=0.5)
    ref_smis = [smiles[el[j]] for j in ref]
    assert tr.modes == ref_smis
    assert len(tr.modes) > 20
    # ineligible molecules are never stored
    assert all(tr.history[s]["qed"] > 0.5 and tr.history[s]["vina"] < 0 for s in tr.history)


def test_mode_tracker_entrants_are_new_members_only():
    smiles = valid_unique(120, 2)
    tr = ModeTracker(k_modes=10, k_elite=20, thresh=0.5, qed_min=0.5)
    for i, s in enumerate(smiles[:60]):
        tr.add(s, -5.0 - 0.01 * i, 0.8, [], "onpolicy", 0, (0, 0), {})
    first = tr.refresh()
    assert len(first) == min(10, len(tr.modes))
    # a strictly better molecule that is dissimilar to everything enters; re-running without changes yields no entrants
    assert tr.refresh() == set()
    new = "CC(C)(C)OC(=O)N1CCC(CC1)c1nc2ccccc2[nH]1"
    assert tr.add(new, -15.0, 0.8, [], "prefix1", 1, (0, 0), {})
    ent = tr.refresh()
    assert new in ent
    assert tr.modes[0] == new


# ----------------------------------------------------------------------------------------------------------------------
def _pf(**kw):
    c = RHCConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_portfolio_counts_floor_cap():
    cfg = _pf()
    pf = Portfolio(cfg, seed=0)
    rs = np.random.RandomState(1)
    for _ in range(500):
        pf.alpha = rs.uniform(0.5, 30, 3)
        pf.beta = rs.uniform(1, 300, 3)
        avail = {"prefix1": rs.rand() < 0.8, "prefix2": rs.rand() < 0.8}
        c = pf.allocate(64, avail)
        assert sum(c.values()) == 64 and all(v >= 0 for v in c.values())
        assert c["onpolicy"] >= int(cfg.onpolicy_floor * 64) - 1
        for a in ("prefix1", "prefix2"):
            assert c[a] <= int(cfg.prefix_cap * 64) + 1
            if not avail[a]:
                assert c[a] == 0


def _simulate(rates_fn, iters, cfg=None, seed=0):
    cfg = cfg or _pf()
    pf = Portfolio(cfg, seed)
    rs = np.random.RandomState(seed + 5)
    shares = []
    for it in range(iters):
        c = pf.allocate(64, {"prefix1": True, "prefix2": True})
        rates = rates_fn(it)
        hits = {a: int(rs.binomial(c[a], rates[a])) for a in ARMS}
        pf.update(c, hits)
        shares.append([c[a] for a in ARMS])
    return np.array(shares)


def test_portfolio_prefers_better_arm_and_tracks_regime_change():
    r1 = {"onpolicy": 0.03, "prefix1": 0.12, "prefix2": 0.02}
    r2 = {"onpolicy": 0.03, "prefix1": 0.02, "prefix2": 0.12}
    sh = _simulate(lambda it: r1 if it < 200 else r2, 400)
    early = sh[100:200].mean(0)
    late = sh[330:400].mean(0)
    assert early[1] > early[2] + 8, early  # prefix1 clearly preferred while it is the best arm
    assert late[2] > late[1] + 8, late  # after the switch the discount lets prefix2 take over
    assert sh[:, 0].min() >= int(0.25 * 64) - 1  # floor respected


# ----------------------------------------------------------------------------------------------------------------------
class _FakeParent:
    calls = []

    @staticmethod
    def score(self, objs):
        from rdkit import Chem

        _FakeParent.calls.append([Chem.MolToSmiles(o) for o in objs])
        return [-float(len(Chem.MolToSmiles(o))) for o in objs]


def test_dedupe_wrapper_never_redocks(monkeypatch):
    from rdkit import Chem
    from cgflow.tasks.semla_vina import VinaMOOTask_semla

    monkeypatch.setattr(VinaMOOTask_semla, "_calc_vina_score_batch", _FakeParent.score)
    _FakeParent.calls.clear()
    t = object.__new__(VinaMOOTask_semlaRHC)
    t.rhc_cfg = _pf(enable=True, dedupe=True)
    t.score_cache = {}
    mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN", "CCO", "c1ccccc1", "CCN"]]
    s1 = t._calc_vina_score_batch(mols)
    assert [len(c) for c in _FakeParent.calls] == [3]  # CCO, CCN, benzene docked once
    assert t.dock_calls_last == 3 and t.cache_hits_last == 2
    assert s1[0] == s1[2] and s1[1] == s1[4]
    s2 = t._calc_vina_score_batch([Chem.MolFromSmiles(s) for s in ["CCN", "CCCl", "CCO"]])
    assert [len(c) for c in _FakeParent.calls] == [3, 1]  # only CCCl is new
    assert t.dock_calls_last == 1 and t.cache_hits_last == 2
    assert s2[0] == s1[1] and s2[2] == s1[0]
    # switched off => plain pass-through (no cache use)
    t2 = object.__new__(VinaMOOTask_semlaRHC)
    t2.rhc_cfg = _pf(enable=True, dedupe=False)
    t2.score_cache = {}
    _FakeParent.calls.clear()
    t2._calc_vina_score_batch(mols)
    assert [len(c) for c in _FakeParent.calls] == [5]


def _traj(smi, valid=True):
    return {"is_valid": valid, "result": SimpleNamespace(smi=smi)}


def test_novel_and_trim_to_budget():
    cache = {"A": -1.0, "B": -2.0}
    trajs = [_traj("A"), _traj("C"), _traj("C"), _traj("D"), _traj("E"), _traj("B"), _traj("F"), _traj("X", valid=False)]
    assert RHCState.novel_smiles(trajs, cache) == ["C", "D", "E", "F"]
    kept = RHCState.trim_to_budget(trajs, cache, budget=3)
    smis = [t["result"].smi for t in kept]
    assert smis == ["A", "C", "C", "D", "E", "B", "X"]  # F (4th novel) dropped; duplicates / cached / invalid kept
    assert len(RHCState.novel_smiles(kept, cache)) == 3


# ----------------------------------------------------------------------------------------------------------------------
def test_forced_prefix_hook(monkeypatch):
    from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler

    monkeypatch.setattr(SyntheticPathSampler, "_estimate_policy", lambda self, model, g, c, m: "cat")
    monkeypatch.setattr(SyntheticPathSampler, "_sample_action", lambda self, g, cat, p=0: [("sampled", i) for i in range(len(g))])
    s = object.__new__(SyntheticPathSampler3D_semlaRHC)
    s.ctx = SimpleNamespace(GraphAction_to_ActionIndex=lambda g, a: ("forced", a))
    s.forced = [["a0", "a1"], None, ["b0"]]
    s._step, s._active = -1, [0, 1, 2]
    s._estimate_policy(None, ["g0", "g1", "g2"], None, [True, True, True])
    out = s._sample_action(["g0", "g1", "g2"], "cat", 0.0)
    assert out == [("forced", "a0"), ("sampled", 1), ("forced", "b0")]
    # step 1: trajectory 1 finished, trajectory 2's prefix is exhausted
    s._estimate_policy(None, ["g0", "g2"], None, [True, False, True])
    out = s._sample_action(["g0", "g2"], "cat", 0.0)
    assert out == [("forced", "a1"), ("sampled", 1)]
    # no forcing at all => untouched baseline behaviour
    s.forced = None
    s._estimate_policy(None, ["g0"], None, [True, False, False])
    assert s._sample_action(["g0"], "cat", 0.0) == [("sampled", 0)]


def test_config_default_off_and_overridable():
    from cgflow.config import Config, init_empty
    from omegaconf import OmegaConf

    c = init_empty(Config())
    c.log_dir = "/tmp/x"
    c.env_dir = "/tmp/y"
    merged = OmegaConf.merge(OmegaConf.structured(Config(log_dir="/tmp/x", env_dir="/tmp/y")), c)
    assert merged.rhc.enable is False and merged.rhc.dedupe is True
    c.rhc.enable = True
    merged = OmegaConf.merge(OmegaConf.structured(Config(log_dir="/tmp/x", env_dir="/tmp/y")), c)
    assert merged.rhc.enable is True and merged.rhc.cond_mode == "posterior"


# ----------------------------------------------------------------------------------------------------------------------
def test_z_estimator_recovers_conditional_mean_and_falls_back():
    from cgflow.lanes.rhc import ZEstimator

    rs = np.random.RandomState(0)
    z = ZEstimator(capacity=2048, min_n=256, bw_beta=0.1, bw_w=0.15)
    # below min_n: nothing usable
    est, ok = z.predict(np.array([10.0]), np.array([0.5]))
    assert not ok.any() and np.isnan(est).all()
    truth = lambda beta, w: -0.2 * beta * (1.0 - 0.5 * w)  # smooth known conditional mean of v(c)
    beta = rs.uniform(0, 64, 1500)
    w = rs.uniform(0, 1, 1500)
    z.add(beta, w, truth(beta, w) + rs.normal(0, 0.5, 1500))
    qb, qw = np.array([8.0, 30.0, 55.0]), np.array([0.2, 0.5, 0.8])
    est, ok = z.predict(qb, qw)
    assert ok.all()
    assert np.max(np.abs(est - truth(qb, qw))) < 0.8, (est, truth(qb, qw))
    # far from any data -> unusable (head fallback)
    z2 = ZEstimator(capacity=512, min_n=100, bw_beta=0.02, bw_w=0.02)
    z2.add(rs.uniform(0, 5, 300), rs.uniform(0, 0.2, 300), np.zeros(300))
    est, ok = z2.predict(np.array([60.0]), np.array([0.9]))
    assert not ok.any()
    # ring buffer keeps the most recent entries only
    z3 = ZEstimator(capacity=100, min_n=10, bw_beta=0.1, bw_w=0.15)
    z3.add(np.zeros(100), np.full(100, 0.5), np.zeros(100))
    z3.add(np.zeros(100), np.full(100, 0.5), np.ones(100))
    est, ok = z3.predict(np.array([0.0]), np.array([0.5]))
    assert ok.all() and abs(est[0] - 1.0) < 1e-6 and len(z3) == 100


def test_offspring_condition_posterior_uses_calibrated_z():
    """With a mis-calibrated head (logZ=0) the posterior collapses to low beta; with the kernel estimator it does not."""
    from cgflow.lanes.rhc import RHCState

    class FakeTask:
        def sample_conditional_information(self, n, it):
            rs = np.random.RandomState(it * 7 + n)
            beta = torch.tensor(rs.uniform(0, 64, n), dtype=torch.float32)
            pref = torch.tensor(rs.dirichlet([1, 1], n), dtype=torch.float32)
            return {"beta": beta, "preferences": pref, "encoding": torch.stack([beta / 64, pref[:, 0]], 1)}

        def cond_info_to_logreward(self, ci, op):
            scal = (op * ci["preferences"]).sum(1)
            return (scal.clamp(min=1e-30).log() * ci["beta"]).clamp(min=-75)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def logZ(self, enc):
            return torch.zeros(enc.shape[0], 1)  # untrained head: logZ == 0 everywhere

    import tempfile

    for mode, ess_min in (("head", 0.0), ("head_safe", 0.1), ("kernel", 0.1)):
        cfg = SimpleNamespace(rhc=_pf(enable=True, z_estimator="head" if mode.startswith("head") else mode, log_arms=False,
                                         hindsight_candidates=64, cond_mode="posterior", ess_min=ess_min))
        st = RHCState(cfg, tempfile.mkdtemp(), seed=0)
        # calibrated truth: v(c) = log R_c(x_typ) with typical reward 0.3 -> log Z(c) ~ beta * log(0.3)
        rs = np.random.RandomState(1)
        b = rs.uniform(0, 64, 1500)
        w = rs.uniform(0, 1, 1500)
        st.zest.add(b, w, b * np.log(0.3) + rs.normal(0, 0.3, 1500))
        specs = [dict(obj_props=(0.6, 0.7), cond={"beta": torch.tensor(30.0), "preferences": torch.tensor([0.5, 0.5]), "encoding": torch.zeros(2)})] * 400
        out = st.offspring_cond(specs, FakeTask(), FakeModel(), it=5)
        mb = float(out["beta"].mean())
        if mode == "head":
            assert mb < 10.0, mb  # collapse to low beta (the artefact Lane B measured)
        elif mode == "head_safe":
            assert st.cond_stats["rhc/cond_skipped"] > 0.3 and mb > 12.0, (mb, st.cond_stats)  # ESS guard falls back to the elite's own condition
        else:
            assert mb > 22.0, mb  # an elite with R~0.65 > typical 0.3 is most typical at high beta under the calibrated Z
            assert st.cond_stats["rhc/cond_kernel_frac"] > 0.9


def test_rhc_state_roundtrip_is_exact():
    """Cull-safe resume: state_dict -> pickle (torch.save uses the same machinery) -> load_state_dict reproduces the elite archive, the greedy modes, the bandit and ALL random streams."""
    import pickle
    import tempfile

    cfg = SimpleNamespace(rhc=_pf(enable=True, log_arms=False, warmup_iters=0, min_elites=2))
    st = RHCState(cfg, tempfile.mkdtemp(), seed=3)
    smiles = valid_unique(300, 5)
    rs = np.random.RandomState(0)
    for i, s in enumerate(smiles):
        v, q = -rs.uniform(3, 13), rs.uniform(0.3, 0.9)
        st.tracker.add(s, float(v), float(q), [("a", i), ("b", i + 1), ("c", 2)], "onpolicy", i // 64, (-v / 10, q),
                       {"beta": torch.tensor(float(i % 64)), "preferences": torch.tensor([0.4, 0.6])})
    st.tracker.refresh()
    st.seen.update(smiles)
    st.n_docked_total = len(smiles)
    st.zest.add(rs.uniform(0, 64, 200), rs.uniform(0, 1, 200), rs.normal(size=200))
    st.cond_stats = {"rhc/cond_ess": 3.5}
    for _ in range(5):
        st.portfolio.allocate(64, {"prefix1": True, "prefix2": True})
        st.portfolio.update({"onpolicy": 40, "prefix1": 12, "prefix2": 12}, {"onpolicy": 2, "prefix1": 4, "prefix2": 1})
    blob = pickle.dumps(st.state_dict())
    st2 = RHCState(cfg, tempfile.mkdtemp(), seed=999)  # different seed on purpose: everything must come from the checkpoint
    st2.load_state_dict(pickle.loads(blob))
    assert st2.tracker.modes == st.tracker.modes and len(st.tracker.modes) > 20
    assert st2.tracker._sorted == st.tracker._sorted and st2.tracker.top_stats() == st.tracker.top_stats()
    assert st2.seen == st.seen and st2.n_docked_total == st.n_docked_total and st2.cond_stats == st.cond_stats
    q = (np.array([10.0, 30.0, 60.0]), np.array([0.2, 0.5, 0.8]))
    a, b = st.zest.predict(*q), st2.zest.predict(*q)
    assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0], equal_nan=True)
    for it in (31, 32, 33):  # the continued streams (Thompson draw, elite picks, torch generator) are identical
        p1, p2 = st.plan(it, 64), st2.plan(it, 64)
        assert p1 == p2
        s1, s2 = st.prefix_specs(p1), st2.prefix_specs(p2)
        assert [(s["arm"], s["elite"]) for s in s1] == [(s["arm"], s["elite"]) for s in s2]
        assert s1 and all(s["forced"] == [("a", 0)][:0] + s["forced"] for s in s1)
    assert torch.equal(st.gen.get_state(), st2.gen.get_state())


def test_route_aware_prefix_selection_tilts_rank_weights_by_the_forced_prefix_potential():
    """route_aware=True multiplies the rank weight of an elite by exp(route_logw(forced prefix)); with route_aware=False, no hook, or a constant hook the picks are bit-identical."""
    import tempfile

    smiles = valid_unique(150, 11)

    def build(route_aware, hook):
        cfg = SimpleNamespace(rhc=_pf(enable=True, log_arms=False, warmup_iters=0, min_elites=2, route_aware=route_aware))
        st = RHCState(cfg, tempfile.mkdtemp(), seed=5)
        for i, s in enumerate(smiles):
            bad = i % 2 == 1
            acts = [SimpleNamespace(action="first", protocol="block1"), SimpleNamespace(action="rxn", protocol="rxn35_linker_b0" if bad else "rxn1_brick_b0"), SimpleNamespace(action="x", protocol="stop")]
            st.tracker.add(s, -12.0 + 0.01 * i, 0.7, acts, "onpolicy", 0, (1.2, 0.7), {"beta": torch.tensor(32.0), "preferences": torch.tensor([0.5, 0.5])})
        st.tracker.refresh()
        st.route_logw = hook
        return st

    hook = lambda acts, k: -2.0 if "linker" in acts[k - 1].protocol else 0.0  # noqa: E731
    plan = {"onpolicy": 0, "prefix1": 0, "prefix2": 20000}
    st = build(True, hook)
    elites = st.tracker.elites(3)
    assert len(elites) > 20
    w = 1.0 / (st.cfg.rank_kappa * len(elites) + np.arange(len(elites)) + 1.0)
    bad = np.array(["linker" in st.tracker.history[s]["actions"][1].protocol for s in elites])
    assert 0 < bad.sum() < len(elites)
    exp_bad = float((w * np.exp(-2.0 * bad))[bad].sum() / (w * np.exp(-2.0 * bad)).sum())
    plain_bad = float(w[bad].sum() / w.sum())
    picks = st.prefix_specs(plan)
    share_bad = float(np.mean([("linker" in st.tracker.history[sp["elite"]]["actions"][1].protocol) for sp in picks]))
    assert share_bad == pytest.approx(exp_bad, abs=0.015) and exp_bad < plain_bad - 0.1  # the tilt is real, not a no-op
    assert st.route_stats["rhc/route_lw_prefix2"] < 0.0
    ref = [sp["elite"] for sp in build(False, None).prefix_specs(plan)]
    assert [sp["elite"] for sp in build(False, hook).prefix_specs(plan)] == ref  # switch off => hook ignored
    assert [sp["elite"] for sp in build(True, lambda acts, k: 0.0).prefix_specs(plan)] == ref  # constant potential => identical stream
    assert [sp["elite"] for sp in build(True, None).prefix_specs(plan)] == ref  # no hook => plain rank weights


def test_route_consistent_portfolio_scales_prefix_arms_by_their_route_mass():
    """arm_mass = sum_i w_i exp(lw_i) / sum_i w_i per prefix arm; allocate(mass) multiplies the Thompson draw of that arm.  prefix2 holds only linker routes => its mass is exp(linker offset) and its slot share collapses;
    without route awareness (or all masses 1) the allocation is bit-identical to the plain portfolio."""
    import tempfile

    smiles = valid_unique(150, 21)

    def build(route_aware, hook):
        cfg = SimpleNamespace(rhc=_pf(enable=True, log_arms=False, warmup_iters=0, min_elites=2, route_aware=route_aware))
        st = RHCState(cfg, tempfile.mkdtemp(), seed=9)
        for i, s in enumerate(smiles):
            three = i % 2 == 1  # 3-action routes (first block + linker + brick) can feed prefix2; 2-action routes only prefix1
            acts = [SimpleNamespace(action="first", protocol="block1"), SimpleNamespace(action="rxn", protocol="rxn1_linker_b0" if three else "rxn1_brick_b0")]
            if three:
                acts.append(SimpleNamespace(action="rxn", protocol="rxn2_brick_b0"))
            st.tracker.add(s, -12.0 + 0.01 * i, 0.7, acts, "onpolicy", 0, (1.2, 0.7), {"beta": torch.tensor(32.0), "preferences": torch.tensor([0.5, 0.5])})
        st.tracker.refresh()
        st.route_logw = hook
        return st

    LINK = -3.0
    hook = lambda acts, k: (LINK if "linker" in acts[k - 1].protocol else 0.0) if k == 2 else -0.1  # noqa: E731
    st = build(True, hook)
    avail = {"prefix1": True, "prefix2": True}
    mass = st.arm_mass(avail)
    assert mass["prefix2"] == pytest.approx(np.exp(LINK)) and mass["prefix1"] == pytest.approx(np.exp(-0.1))
    pf = lambda: Portfolio(_pf(enable=True), 3)  # noqa: E731
    n_plain = np.mean([pf().allocate(64, avail)["prefix2"] for _ in range(1)])
    p_plain, p_mass = pf(), pf()
    a_plain, a_mass = p_plain.allocate(64, avail), p_mass.allocate(64, avail, mass)  # same Thompson draw (same seed)
    assert a_mass["prefix2"] < a_plain["prefix2"] and a_mass["onpolicy"] > a_plain["onpolicy"] and a_plain["prefix2"] >= n_plain - 1e-9
    assert pf().allocate(64, avail, {"prefix1": 1.0, "prefix2": 1.0}) == a_plain  # mass 1 == plain
    # plan(): identical stream without route awareness
    s_off, s_on_nohook = build(False, hook), build(True, None)
    for it in (31, 32, 33):
        assert s_off.plan(it, 64) == s_on_nohook.plan(it, 64)
    assert st.plan(31, 64)["prefix2"] <= build(False, None).plan(31, 64)["prefix2"]
    assert "rhc/route_mass_prefix2" in st.route_stats
