"""Lane B / T2: the Bayesian condition-hindsight step.

A toy conditional model with a KNOWN log Z(c) is used, so the exact posterior

    p(c|x)  ~  p(c) exp( log R_c(x) - log Z(c) ),   c = (beta, w),  p(c) = U(0,B) x Dirichlet(1,1)

can be computed by a very large self-normalised Monte-Carlo integration and compared, with a chi-square
test, with the empirical distribution of the conditions the implementation produces.
"""

import numpy as np
import torch
from scipy import stats

from cgflow.lanes.lane_b import posterior_hindsight


class ToyTask:
    """Mimics `BaseTask` for the two methods `posterior_hindsight` needs.

    prior:   beta ~ U(0, beta_max),  w ~ Dirichlet(1,1)  (== w0 ~ U(0,1), w1 = 1-w0)
    reward:  log R_c(x) = beta * log( w . f(x) )           (the real task's scalarisation, no clamp)
    encoding: [beta, w0, w1] so that the toy log Z can read the condition back.
    """

    def __init__(self, seed: int, beta_max: float = 8.0):
        self.rng = np.random.default_rng(seed)
        self.beta_max = beta_max

    def sample_conditional_information(self, n: int, train_it: int) -> dict:
        beta = torch.tensor(self.rng.uniform(0.0, self.beta_max, n), dtype=torch.float32)
        w0 = torch.tensor(self.rng.uniform(0.0, 1.0, n), dtype=torch.float32)
        prefs = torch.stack([w0, 1.0 - w0], dim=1)
        enc = torch.cat([beta[:, None], prefs], dim=1)
        return {"beta": beta, "preferences": prefs, "encoding": enc}

    def cond_info_to_logreward(self, cond_info: dict, obj_props: torch.Tensor) -> torch.Tensor:
        s = (obj_props * cond_info["preferences"]).sum(1).clamp(min=1e-30)
        return cond_info["beta"] * s.log()


# log Z(c) of the toy model: the partition function of a reference object (analytic, known)
F_REF = torch.tensor([1.0, 0.8])


def toy_logZ(cond_info: dict) -> torch.Tensor:
    """log Z of the toy model; takes the condition dict, like the real estimators do."""
    encoding = cond_info["encoding"]
    beta = encoding[:, 0]
    w = encoding[:, 1:]
    return beta * (w * F_REF).sum(1).clamp(min=1e-30).log()


def reference_posterior_hist(obj, beta_max: float, bins: np.ndarray, n_mc: int = 4_000_000):
    """Exact (up to 1/sqrt(n_mc)) bin probabilities of p(beta|x) by self-normalised MC over the prior."""
    rng = np.random.default_rng(12345)
    beta = rng.uniform(0.0, beta_max, n_mc)
    w0 = rng.uniform(0.0, 1.0, n_mc)
    s = w0 * float(obj[0]) + (1 - w0) * float(obj[1])
    s_ref = w0 * float(F_REF[0]) + (1 - w0) * float(F_REF[1])
    log_u = beta * (np.log(s) - np.log(s_ref))
    log_u -= log_u.max()
    u = np.exp(log_u)
    idx = np.clip(np.digitize(beta, bins) - 1, 0, len(bins) - 2)
    num = np.bincount(idx, weights=u, minlength=len(bins) - 1)
    return num / num.sum()


def test_relabelled_conditions_follow_the_posterior_chi2():
    beta_max = 8.0
    n, m = 4000, 512
    obj = torch.tensor([0.75, 0.55])  # a mediocre object: the posterior should prefer small beta
    task = ToyTask(seed=7, beta_max=beta_max)
    gen = torch.Generator()
    gen.manual_seed(20240918)

    obj_props = obj.repeat(n, 1)
    out, log_rewards, info = posterior_hindsight(obj_props, task, toy_logZ, m, 0, gen)

    # the returned log rewards must be the ones of the returned conditions
    assert torch.allclose(log_rewards, task.cond_info_to_logreward(out, obj_props), atol=1e-6)
    assert 0.0 < info["hs_ess"] <= 1.0

    bins = np.linspace(0.0, beta_max, 9)
    ref = reference_posterior_hist(obj, beta_max, bins)
    emp = np.histogram(out["beta"].numpy(), bins=bins)[0]
    expected = ref * emp.sum()
    keep = expected > 5  # chi-square validity
    chi2 = float(((emp[keep] - expected[keep]) ** 2 / expected[keep]).sum())
    pval = float(stats.chi2.sf(chi2, df=int(keep.sum()) - 1))
    tv = 0.5 * float(np.abs(emp / emp.sum() - ref).sum())
    assert tv < 0.05, f"total variation {tv:.4f} too large (emp={emp / emp.sum()}, ref={ref})"
    assert pval > 1e-3, f"chi2={chi2:.2f} p={pval:.2e} tv={tv:.4f}"


def test_better_objects_are_relabelled_to_higher_temperature():
    """The mechanism's purpose: an object that is good relative to the model's own Z(c) is explained by a
    high-beta condition, a bad one by a low-beta condition."""
    n, m = 3000, 256
    task = ToyTask(seed=3)
    gen = torch.Generator()
    gen.manual_seed(5)
    good = torch.tensor([1.3, 0.9]).repeat(n, 1)
    bad = torch.tensor([0.4, 0.3]).repeat(n, 1)
    out_g, _, _ = posterior_hindsight(good, task, toy_logZ, m, 0, gen)
    out_b, _, _ = posterior_hindsight(bad, task, toy_logZ, m, 0, gen)
    assert float(out_g["beta"].mean()) > 3.0 * float(out_b["beta"].mean())


def test_preferences_are_relabelled_towards_the_objective_the_object_is_good_at():
    n, m = 3000, 256
    task = ToyTask(seed=11)
    gen = torch.Generator()
    gen.manual_seed(17)
    obj = torch.tensor([1.5, 0.05]).repeat(n, 1)
    out, _, _ = posterior_hindsight(obj, task, toy_logZ, m, 0, gen)
    assert float(out["preferences"][:, 0].mean()) > 0.8


def test_mask_keeps_the_original_condition_and_diagnostics_are_consistent():
    n, m = 256, 16
    task = ToyTask(seed=1)
    gen = torch.Generator()
    gen.manual_seed(2)
    obj_props = torch.tensor([1.0, 0.6]).repeat(n, 1)
    base = task.sample_conditional_information(n, 0)
    mask = torch.zeros(n, dtype=torch.bool)  # relabel nothing
    out, log_rewards, info = posterior_hindsight(
        obj_props, task, toy_logZ, m, 0, gen, relabel_mask=mask, base_cond=base
    )
    for k in base:
        assert torch.allclose(out[k], base[k])
    assert torch.allclose(log_rewards, task.cond_info_to_logreward(base, obj_props), atol=1e-6)
    assert info["hs_beta_up"] == 0.0 and info["hs_beta_down"] == 0.0

    mask = torch.ones(n, dtype=torch.bool)  # relabel everything
    out2, _, info2 = posterior_hindsight(
        obj_props, task, toy_logZ, m, 0, gen, relabel_mask=mask, base_cond=base
    )
    assert info2["hs_beta_up"] + info2["hs_beta_down"] > 0.99
    assert not torch.allclose(out2["beta"], base["beta"])


def test_single_candidate_is_a_prior_draw():
    """M=1 degenerates to 'keep a fresh prior sample' - a sanity check of the SNIS bookkeeping."""
    n = 2000
    task = ToyTask(seed=4, beta_max=10.0)
    gen = torch.Generator()
    gen.manual_seed(9)
    obj_props = torch.tensor([1.1, 0.7]).repeat(n, 1)
    out, _, info = posterior_hindsight(obj_props, task, toy_logZ, 1, 0, gen)
    assert abs(float(out["beta"].mean()) - 5.0) < 0.4  # ~ U(0,10)
    assert abs(info["hs_ess"] - 1.0) < 1e-6
