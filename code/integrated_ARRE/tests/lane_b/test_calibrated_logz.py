"""Lane B / T2b: the TB-implied log-partition estimator used by `hindsight="posterior_cal"`.

Toy with a KNOWN Z(c): a finite object set X, condition c = (beta, w), R_c(x) = (w.f(x))^beta, so
log Z(c) = log sum_x (w.f(x))^beta is analytic.  A trajectory of the toy "sampler" q(x|c) contributes
    v = log R_c(x) - log q(x|c)
(the single-step analogue of `log R - sum_t log P_F`).  For the TB-calibrated sampler q = R_c/Z(c) every
v equals log Z(c) exactly, which is the identity `posterior_cal` relies on; for an uncalibrated sampler
E_q[v] = log Z(c) - KL(q || R_c/Z) <= log Z(c) (Jensen), which is the caveat documented in the report.
"""

import numpy as np
import torch

from cgflow.lanes.lane_b import make_calibrated_logZ, nw_log_partition, posterior_hindsight

K = 40  # number of toy objects
BETA_MAX = 64.0
_rng = np.random.default_rng(0)
F = torch.tensor(_rng.uniform(0.2, 1.1, size=(K, 2)), dtype=torch.float64)  # objective vectors


def log_R(beta: torch.Tensor, w0: torch.Tensor) -> torch.Tensor:
    """[N, K] log R_c(x) for every condition / object pair."""
    w = torch.stack([w0, 1.0 - w0], dim=1).double()
    s = (w[:, None, :] * F[None, :, :]).sum(-1).clamp(min=1e-30)
    return beta.double()[:, None] * s.log()


def true_logZ(beta: torch.Tensor, w0: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(log_R(beta, w0), dim=1)


def sample_v(n: int, seed: int, temper: float = 1.0):
    """Draw conditions from the prior and one object from q ~ R_c^temper; return (beta, w0, v).

    temper=1 -> the TB-calibrated sampler (v == log Z(c) exactly).
    temper<1 -> an uncalibrated, flatter sampler (Jensen gap).
    """
    g = torch.Generator().manual_seed(seed)
    beta = torch.rand(n, generator=g) * BETA_MAX
    w0 = torch.rand(n, generator=g)
    lr = log_R(beta, w0)
    log_q = torch.log_softmax(temper * lr, dim=1)
    idx = torch.multinomial(log_q.exp().float(), 1, generator=g).squeeze(1)
    rows = torch.arange(n)
    v = lr[rows, idx] - log_q[rows, idx]
    return beta, w0, v.float()


def test_nw_recovers_logZ_for_a_tb_calibrated_sampler():
    d_beta, d_w0, d_v = sample_v(2048, seed=1, temper=1.0)
    q_beta, q_w0, _ = sample_v(512, seed=2, temper=1.0)
    est, ok, n_eff = nw_log_partition(q_beta, q_w0, d_beta, d_w0, d_v, 0.1, 0.15, BETA_MAX)
    truth = true_logZ(q_beta, q_w0).float()
    assert bool(ok.all())
    assert n_eff > 20.0, n_eff
    err = (est - truth).abs()
    spread = float(truth.std())
    # with the production bandwidths (0.1, 0.15) the estimator is dominated by its smoothing bias;
    # it tracks log Z(c) to ~15% of the spread of log Z over the prior
    assert float(err.mean()) < 0.2 * spread, (float(err.mean()), spread)
    assert float(torch.corrcoef(torch.stack([est, truth]))[0, 1]) > 0.97
    # ... and it is consistent: narrower kernels (more data per unit volume would be needed in general,
    # here 2048 points are enough) reduce the error
    est2, _, _ = nw_log_partition(q_beta, q_w0, d_beta, d_w0, d_v, 0.05, 0.08, BETA_MAX)
    err2 = (est2 - truth).abs()
    assert float(err2.mean()) < 0.6 * float(err.mean()), (float(err2.mean()), float(err.mean()))
    assert float(torch.corrcoef(torch.stack([est2, truth]))[0, 1]) > 0.99


def test_jensen_gap_for_an_uncalibrated_sampler():
    """E[log R - log q] <= log Z with equality only for the calibrated sampler: the estimator therefore
    UNDERESTIMATES log Z while the policy is not calibrated (documented limitation of posterior_cal)."""
    d_beta, d_w0, d_v = sample_v(4096, seed=3, temper=0.3)
    q_beta = torch.linspace(5.0, 60.0, 40)
    q_w0 = torch.full((40,), 0.5)
    est, _, _ = nw_log_partition(q_beta, q_w0, d_beta, d_w0, d_v, 0.1, 0.15, BETA_MAX)
    truth = true_logZ(q_beta, q_w0).float()
    gap = (truth - est)
    assert float((gap > 0).float().mean()) > 0.9, gap
    assert float(gap.mean()) > 0.5


def test_per_query_fallback_to_the_head_when_no_data_is_nearby():
    d_beta = torch.full((300,), 1.0)  # all data at beta ~ 0
    d_w0 = torch.full((300,), 0.5)
    d_v = torch.full((300,), -3.0)
    head_called = {"n": 0}

    def head(cond):
        head_called["n"] += 1
        return torch.full_like(cond["beta"], 123.0)

    fn = make_calibrated_logZ(d_beta, d_w0, d_v, head, 0.1, 0.15, BETA_MAX)
    cond = {"beta": torch.tensor([1.0, 64.0]), "preferences": torch.tensor([[0.5, 0.5], [0.5, 0.5]])}
    out = fn(cond)
    assert abs(float(out[0]) + 3.0) < 1e-4  # near the data -> kernel estimate
    assert abs(float(out[1]) - 123.0) < 1e-4  # far from any data -> head
    assert head_called["n"] == 1


def test_posterior_cal_matches_the_exact_logZ_posterior():
    """With TB-calibrated data the kernel estimate stands in for the exact log Z: the relabelled
    temperature distribution must match the one obtained with the analytic log Z."""

    class ToyTask:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

        def sample_conditional_information(self, n, train_it):
            beta = torch.tensor(self.rng.uniform(0.0, BETA_MAX, n), dtype=torch.float32)
            w0 = torch.tensor(self.rng.uniform(0.0, 1.0, n), dtype=torch.float32)
            prefs = torch.stack([w0, 1.0 - w0], dim=1)
            return {"beta": beta, "preferences": prefs, "encoding": torch.cat([beta[:, None], prefs], 1)}

        def cond_info_to_logreward(self, ci, obj):
            s = (obj * ci["preferences"]).sum(1).clamp(min=1e-30)
            return ci["beta"] * s.log()

    def exact(cond):
        return true_logZ(cond["beta"], cond["preferences"][:, 0]).float()

    d_beta, d_w0, d_v = sample_v(4096, seed=5, temper=1.0)
    cal = make_calibrated_logZ(d_beta, d_w0, d_v, exact, 0.1, 0.15, BETA_MAX)

    obj = F[int(torch.argmax((F * torch.tensor([0.5, 0.5])).sum(1)))].float()  # a good object
    n, m = 3000, 128
    obj_props = obj.repeat(n, 1)

    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    out_e, _, _ = posterior_hindsight(obj_props, ToyTask(7), exact, m, 0, g1)
    out_c, _, info = posterior_hindsight(obj_props, ToyTask(7), cal, m, 0, g2)

    bins = np.linspace(0, BETA_MAX, 9)
    he = np.histogram(out_e["beta"].numpy(), bins=bins)[0] / n
    hc = np.histogram(out_c["beta"].numpy(), bins=bins)[0] / n
    tv = 0.5 * float(np.abs(he - hc).sum())
    assert tv < 0.06, (tv, he, hc)
    assert abs(float(out_e["beta"].mean()) - float(out_c["beta"].mean())) < 2.0
    assert info["hs_skipped_lowess_frac"] == 0.0  # min_ess defaults to 0 here


def test_min_ess_safeguard_keeps_the_original_condition():
    class ToyTask:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)

        def sample_conditional_information(self, n, train_it):
            beta = torch.tensor(self.rng.uniform(0.0, BETA_MAX, n), dtype=torch.float32)
            w0 = torch.tensor(self.rng.uniform(0.0, 1.0, n), dtype=torch.float32)
            prefs = torch.stack([w0, 1.0 - w0], dim=1)
            return {"beta": beta, "preferences": prefs, "encoding": torch.cat([beta[:, None], prefs], 1)}

        def cond_info_to_logreward(self, ci, obj):
            s = (obj * ci["preferences"]).sum(1).clamp(min=1e-30)
            return ci["beta"] * s.log()

    def exact(cond):
        return true_logZ(cond["beta"], cond["preferences"][:, 0]).float()

    task = ToyTask(3)
    n, m = 512, 16
    obj_props = torch.tensor([0.3, 0.3]).repeat(n, 1)  # a bad object -> very peaked posterior, tiny ESS
    base = task.sample_conditional_information(n, 0)
    gen = torch.Generator().manual_seed(4)
    out, log_rewards, info = posterior_hindsight(
        obj_props, task, exact, m, 0, gen, base_cond=base, min_ess=0.99
    )
    assert info["hs_skipped_lowess_frac"] > 0.5
    kept = torch.isclose(out["beta"], base["beta"])
    assert bool(kept.any())
    assert torch.allclose(log_rewards[kept], task.cond_info_to_logreward(base, obj_props)[kept], atol=1e-5)
