"""Lane B / T1: the prioritised replay buffer (rank formula, determinism, eviction, filtering)."""

import numpy as np
import torch

from cgflow.config import Config
from cgflow.lanes.lane_b import FAILED_DOCKING_REWARD, ReplayEngine


def make_cfg(**lane_kwargs) -> Config:
    cfg = Config()
    cfg.seed = 0
    for k, v in lane_kwargs.items():
        setattr(cfg.lane_b, k, v)
    return cfg


def make_traj(r_vina: float, qed: float, is_valid: bool = True) -> dict:
    return {
        "obj_props": torch.tensor([r_vina, qed], dtype=torch.float32),
        "is_valid": is_valid,
        "log_reward": torch.tensor(0.0),
        "cond_info": {"beta": torch.tensor(1.0), "encoding": torch.zeros(4)},
        "traj": [],
    }


def test_rank_is_the_evaluation_criterion():
    eng = ReplayEngine(make_cfg(capacity=100))
    # (r_vina, qed):   qed>0.5 group first, inside a group by r_vina descending
    eng.insert([make_traj(0.9, 0.2), make_traj(0.5, 0.9), make_traj(1.2, 0.8), make_traj(2.0, 0.4)])
    order = eng.order().tolist()
    assert order == [2, 1, 3, 0], order
    ranks = eng.ranks()
    assert ranks.tolist() == [3.0, 1.0, 0.0, 2.0]


def test_priority_formula_and_uniform_mix():
    n = 500
    kappa, mix = 0.01, 0.2
    eng = ReplayEngine(make_cfg(capacity=n, rank_kappa=kappa, uniform_mix=mix, priority="rank"))
    rng = np.random.default_rng(0)
    eng.insert([make_traj(float(v), 0.9) for v in rng.uniform(0.5, 1.5, n)])
    p = eng.priorities()
    ranks = eng.ranks()
    raw = 1.0 / (max(kappa * n, 1.0) + ranks)
    raw = raw / raw.sum()
    expect = (1 - mix) * raw + mix / n
    expect = expect / expect.sum()
    assert np.allclose(p, expect, rtol=1e-12, atol=1e-15)
    assert abs(p.sum() - 1.0) < 1e-12
    # the best element must be the most likely one, the worst the least likely
    assert p[eng.order()[0]] == p.max()
    assert p[eng.order()[-1]] == p.min()

    uni = ReplayEngine(make_cfg(capacity=n, priority="uniform"))
    uni.insert([make_traj(float(v), 0.9) for v in rng.uniform(0.5, 1.5, n)])
    assert np.allclose(uni.priorities(), 1.0 / n)


def test_sampling_is_deterministic_distinct_and_biased_to_the_top():
    n, k = 400, 32
    kw = dict(capacity=n, rank_kappa=0.01, uniform_mix=0.0, priority="rank")
    objs = [(float(i) / n + 0.5, 0.9) for i in range(n)]  # better r_vina for larger i

    def build(seed):
        cfg = make_cfg(**kw)
        cfg.seed = seed
        e = ReplayEngine(cfg)
        e.insert([make_traj(a, b) for a, b in objs])
        return e

    e1, e2, e3 = build(0), build(0), build(1)
    s1, _ = e1.sample(k)
    s2, _ = e2.sample(k)
    s3, _ = e3.sample(k)
    assert [id(r) for r in s1] != [id(r) for r in s2]  # different objects ...
    assert [r["obj"][0] for r in s1] == [r["obj"][0] for r in s2]  # ... but the same draw
    assert [r["obj"][0] for r in s1] != [r["obj"][0] for r in s3]  # a different seed differs
    assert len({id(r) for r in s1}) == k  # without replacement

    # the mean rank of a prioritised draw must be clearly better than uniform (rank 0 = best).
    # p ~ 1/(kappa*n + rank) with kappa*n = 4 has mean rank ~83; drawing 32 of 400 *without replacement*
    # pushes that up (the top entries cannot be drawn twice), uniform would give (n-1)/2 = 199.5.
    ranks = e1.ranks()
    idx = {id(r): i for i, r in enumerate(e1.items)}
    mean_rank = float(np.mean([ranks[idx[id(r)]] for r in s1]))
    assert mean_rank < 0.7 * (n - 1) / 2, mean_rank


def test_invalid_and_failed_docking_are_never_stored():
    eng = ReplayEngine(make_cfg(capacity=10, skip_failed_docking=True))
    eng.insert(
        [
            make_traj(0.9, 0.9),
            make_traj(0.9, 0.9, is_valid=False),
            make_traj(FAILED_DOCKING_REWARD, 0.9),  # UniDock failure -> vina reward clipped to 1e-4
        ]
    )
    assert len(eng) == 1
    assert eng.n_skipped_invalid == 1 and eng.n_skipped_failed == 1
    keep = ReplayEngine(make_cfg(capacity=10, skip_failed_docking=False))
    keep.insert([make_traj(FAILED_DOCKING_REWARD, 0.9)])
    assert len(keep) == 1


def test_capacity_fifo_and_protected_elites():
    eng = ReplayEngine(make_cfg(capacity=20, protected_elites=2))
    elite = make_traj(9.9, 0.9)  # by far the best
    eng.insert([elite] + [make_traj(0.5, 0.9) for _ in range(19)])
    assert len(eng) == 20
    for _ in range(5):
        eng.insert([make_traj(0.4, 0.9) for _ in range(10)])
    assert len(eng) == 20
    assert eng.n_evicted == 50
    stored = [r["traj"] for r in eng.items]
    assert any(t is elite for t in stored), "a protected elite was evicted"
    # without protection the elite is evicted by FIFO
    eng2 = ReplayEngine(make_cfg(capacity=20, protected_elites=0))
    elite2 = make_traj(9.9, 0.9)
    eng2.insert([elite2] + [make_traj(0.5, 0.9) for _ in range(19)])
    eng2.insert([make_traj(0.4, 0.9) for _ in range(10)])
    assert not any(r["traj"] is elite2 for r in eng2.items)


def test_priority_ess_and_stats():
    eng = ReplayEngine(make_cfg(capacity=100, priority="uniform"))
    eng.insert([make_traj(1.0, 0.9) for _ in range(100)])
    assert abs(eng.priority_ess() - 1.0) < 1e-9
    eng2 = ReplayEngine(make_cfg(capacity=100, priority="rank", uniform_mix=0.0, rank_kappa=0.001))
    # +1 so that no entry looks like a failed docking (vina reward == 1e-4), which would be filtered out
    eng2.insert([make_traj(float(i) + 1.0, 0.9) for i in range(100)])
    assert eng2.priority_ess() < 0.9
    st = eng2.stats()
    assert st["buffer_size"] == 100.0 and st["buffer_qed_pass_frac"] == 1.0
    assert st["buffer_vina_best"] == -10.0 * 100.0
