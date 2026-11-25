"""T3 - the acquisition rule: warm-up gate, strata, random fraction, QED floor."""

import numpy as np
import pytest

from cgflow.lanes.lane_d import PoseProxyGate, decide_gate, split_counts
from cgflow.lanes.lane_d_config import LaneDConfig

N_SELECT = 64
N_POOL = 128


def _gate(**kw) -> PoseProxyGate:
    """A PoseProxyGate with only the fields ``_stratified_select`` needs.

    Built without __init__ on purpose: the selector is pure and must be
    testable without a pocket file, a GPU or the pose model.
    """
    g = object.__new__(PoseProxyGate)
    g.dcfg = LaneDConfig(**kw)
    g.rng = np.random.default_rng(0)
    return g


def _pool(rng, n=N_POOL):
    acq = rng.normal(size=n)
    qed = rng.uniform(0.5, 0.9, size=n)
    beta = rng.uniform(0.0, 64.0, size=n)
    return acq, qed, beta


def test_gate_stays_off_while_ineligible():
    assert decide_gate(False, False, 0.9, 0.2, 0.05) is False
    assert decide_gate(True, False, 0.9, 0.2, 0.05) is False


def test_gate_needs_rho_above_min_rho():
    assert decide_gate(False, True, 0.19, 0.2, 0.05) is False
    assert decide_gate(False, True, 0.20, 0.2, 0.05) is True
    assert decide_gate(False, True, float("nan"), 0.2, 0.05) is False


def test_gate_hysteresis():
    assert decide_gate(True, True, 0.17, 0.2, 0.05) is True
    assert decide_gate(True, True, 0.14, 0.2, 0.05) is False
    assert decide_gate(True, True, float("nan"), 0.2, 0.05) is False


def test_split_counts():
    assert split_counts(64, 4) == [16, 16, 16, 16]
    assert sum(split_counts(64, 5)) == 64
    assert split_counts(5, 2) == [3, 2]


@pytest.mark.parametrize("strata", [1, 2, 4, 8])
def test_selects_exactly_n_select_unique(strata):
    rng = np.random.default_rng(1)
    g = _gate(strata=strata)
    acq, qed, beta = _pool(rng)
    sel, is_rand = g._stratified_select(acq, qed, beta, N_SELECT)
    assert len(sel) == N_SELECT
    assert len(set(sel)) == N_SELECT
    assert len(is_rand) == N_SELECT


def test_random_fraction_is_respected_per_stratum():
    rng = np.random.default_rng(2)
    g = _gate(strata=4, random_frac=0.1)
    acq, qed, beta = _pool(rng)
    _, is_rand = g._stratified_select(acq, qed, beta, N_SELECT)
    assert sum(is_rand) == 8  # round(0.1 * 16) = 2 per stratum


def test_at_least_one_random_pick_per_stratum_keeps_support():
    rng = np.random.default_rng(3)
    g = _gate(strata=8, random_frac=0.01)  # round() would give 0
    acq, qed, beta = _pool(rng)
    _, is_rand = g._stratified_select(acq, qed, beta, N_SELECT)
    assert sum(is_rand) == 8


def test_random_frac_zero_gives_pure_greedy():
    rng = np.random.default_rng(4)
    g = _gate(strata=4, random_frac=0.0)
    acq, qed, beta = _pool(rng)
    _, is_rand = g._stratified_select(acq, qed, beta, N_SELECT)
    assert sum(is_rand) == 0


def test_beta_marginal_is_preserved_by_stratification():
    rng = np.random.default_rng(5)
    k = 4
    g = _gate(strata=k, random_frac=0.1)
    acq, qed, beta = _pool(rng)
    sel, _ = g._stratified_select(acq, qed, beta, N_SELECT)
    order = np.argsort(beta, kind="mergesort")
    stratum = np.empty(N_POOL, dtype=int)
    pos = 0
    for s, c in enumerate(split_counts(N_POOL, k)):
        stratum[order[pos: pos + c]] = s
        pos += c
    counts = np.bincount(stratum[np.asarray(sel)], minlength=k)
    assert counts.tolist() == split_counts(N_SELECT, k)


def test_unstratified_selection_shifts_beta_but_stratified_does_not():
    """The point of the stratification: without it the gate over-selects high beta."""
    rng = np.random.default_rng(6)
    beta = rng.uniform(0.0, 64.0, size=N_POOL)
    acq = beta / 64.0 + 0.05 * rng.normal(size=N_POOL)  # better molecules at high beta
    qed = np.full(N_POOL, 0.8)
    flat = _gate(strata=1, random_frac=0.1)._stratified_select(acq, qed, beta, N_SELECT)[0]
    strat = _gate(strata=4, random_frac=0.1)._stratified_select(acq, qed, beta, N_SELECT)[0]
    pool_mean = beta.mean()
    flat_shift = beta[np.asarray(flat)].mean() - pool_mean
    strat_shift = beta[np.asarray(strat)].mean() - pool_mean
    assert flat_shift > 10.0, "unstratified top-k must drift to high beta"
    # stratification bounds the residual drift by roughly one stratum width (64/k);
    # it does not remove it, inside a stratum the gate still prefers the better half.
    assert abs(strat_shift) < 64.0 / 4.0
    assert abs(strat_shift) < 0.4 * flat_shift


def test_greedy_picks_are_the_best_scoring_ones():
    rng = np.random.default_rng(7)
    g = _gate(strata=1, random_frac=0.0)
    acq = rng.normal(size=N_POOL)
    qed = np.full(N_POOL, 0.9)
    beta = rng.uniform(0, 64, N_POOL)
    sel, _ = g._stratified_select(acq, qed, beta, N_SELECT)
    assert set(sel) == set(np.argsort(-acq)[:N_SELECT].tolist())


def test_qed_floor_outranks_the_score():
    g = _gate(strata=1, random_frac=0.0, qed_floor=0.45)
    acq = np.concatenate([np.full(64, 10.0), np.full(64, -10.0)])
    qed = np.concatenate([np.full(64, 0.1), np.full(64, 0.9)])
    beta = np.linspace(0, 64, N_POOL)
    sel, _ = g._stratified_select(acq, qed, beta, N_SELECT)
    assert set(sel) == set(range(64, 128))


def test_qed_floor_is_soft_when_the_pool_cannot_fill_the_batch():
    g = _gate(strata=1, random_frac=0.0, qed_floor=0.45)
    acq = np.linspace(0.0, 1.0, N_POOL)
    qed = np.concatenate([np.full(10, 0.9), np.full(N_POOL - 10, 0.1)])
    beta = np.linspace(0, 64, N_POOL)
    sel, _ = g._stratified_select(acq, qed, beta, N_SELECT)
    assert len(sel) == N_SELECT
    assert set(range(10)).issubset(set(sel))


def test_invalid_candidates_are_never_picked_greedily():
    rng = np.random.default_rng(8)
    g = _gate(strata=2, random_frac=0.0)
    acq = rng.normal(size=N_POOL)
    acq[:40] = -np.inf  # invalid trajectories
    qed = np.full(N_POOL, 0.8)
    beta = rng.uniform(0, 64, N_POOL)
    sel, _ = g._stratified_select(acq, qed, beta, N_SELECT)
    assert not np.isneginf(acq[np.asarray(sel)]).any()


def test_selection_is_reproducible_for_a_given_rng_seed():
    rng = np.random.default_rng(9)
    acq, qed, beta = _pool(rng)
    a = _gate(strata=4)._stratified_select(acq, qed, beta, N_SELECT)[0]
    b = _gate(strata=4)._stratified_select(acq, qed, beta, N_SELECT)[0]
    assert a == b
