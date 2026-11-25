"""T1 - Lane A estimator maths on a toy action space, against brute-force enumeration.

Checks
  * the exact cached normaliser produced by the chunked pass == full enumeration
  * the two-stage sampler really draws from the defensive mixture (frequencies + log q)
  * Z_hat is unbiased for Z_theta even with a STALE cache, and log Z_hat is biased low
  * PAN has a much smaller variance than the baseline's uniform-rho sub-sample estimator
  * E[grad log Z_hat] == grad log Z
"""

from __future__ import annotations

import math

import pytest
import torch

from cgflow.lanes.config import LaneAConfig
from cgflow.lanes.pansub.pan import PanEngine

from .toyenv import ToyCtx, ToyModel, brute_force_logits

DEV = torch.device("cpu")
TYPES = {"a": 600, "b": 1200, "c": 80}
PROTOS = {"p0": ["a", "b"], "p1": ["c", "a"]}


def make(scale: float = 1.0, K: int = 512, lam: float = 0.1, seed: int = 0, cfg_kw=None):
    ctx = ToyCtx(TYPES, PROTOS, fp_dim=64, seed=seed)
    model = ToyModel(ctx, seed=seed, scale=scale)
    cfg = LaneAConfig(
        enable=True,
        normalizer="pan",
        behavior="full",
        num_samples=K,
        defensive_mix=lam,
        cache_dtype="float32",
        grad_checkpoint=False,
        chunk_elems=4096,
        **(cfg_kw or {}),
    )
    engine = PanEngine(cfg, ctx, DEV, num_emb_block=8, seed=seed)
    engine.cache.refresh(model, 0)
    return ctx, model, engine


def state(n=3, level=0.9, seed=0):
    g = torch.Generator().manual_seed(100 + seed)
    emb = torch.randn((n, 16), generator=g)
    mask = torch.ones((n, 2), dtype=torch.bool)
    lvl = torch.full((n, 1), float(level))
    return emb, mask, lvl


def exact(ctx, model, emb, lvl, protocols=(0, 1)):
    out = []
    for i in range(emb.shape[0]):
        lg, legal, meta = brute_force_logits(ctx, model, emb[i], list(protocols), 1.0 - float(lvl[i]))
        out.append((lg.detach(), legal, meta))
    return out


def test_exact_normalizer_matches_enumeration():
    ctx, model, engine = make()
    emb, mask, lvl = state()
    pan = engine.evaluate(model, emb, mask, lvl, n_beh=0, n_is=0)
    for i, (lg, legal, _) in enumerate(exact(ctx, model, emb, lvl)):
        ref = torch.logsumexp(lg[legal], 0)
        assert abs(float(pan.log_Zc[i]) - float(ref)) < 2e-3, (i, float(pan.log_Zc[i]), float(ref))
        assert abs(float(pan.log_Nleg[i].exp()) - float(legal.sum())) < 0.5


@pytest.mark.parametrize("lam", [0.0, 0.25, 1.0])
def test_sampler_matches_proposal(lam):
    """Empirical frequencies of the drawn entries == the mixture density, and log_q is right."""
    ctx, model, engine = make(scale=2.0, K=20000, lam=lam)
    emb, mask, lvl = state(n=1)
    pan = engine.evaluate(model, emb, mask, lvl, n_beh=0, n_is=20000)
    lg, legal, meta = exact(ctx, model, emb, lvl)[0]
    lZ = torch.logsumexp(lg[legal], 0)
    nleg = int(legal.sum())
    q = torch.where(
        legal,
        (1 - lam) * (lg - lZ).exp() + lam / nleg,
        torch.zeros_like(lg),
    )
    assert abs(float(q.sum()) - 1.0) < 1e-4

    flat = {(p, b, i): k for k, (p, b, i) in enumerate(meta)}
    keys = [flat[(int(p), int(t), int(x))] for p, t, x in zip(pan.proto[0], pan.bt[0], pan.block[0])]
    idx = torch.tensor(keys)
    # log_q reported by the engine == brute-force mixture density
    assert torch.allclose(pan.log_q[0], q[idx].log(), atol=2e-3), (pan.log_q[0][:5], q[idx][:5].log())
    # empirical frequencies: binned chi-square (so every cell has a large expected count)
    n_draw = idx.numel()
    cnt = torch.bincount(idx, minlength=lg.numel()).float()
    assert int((cnt[~legal] > 0).sum()) == 0, "an illegal action was sampled"
    nb = 40
    bin_id = torch.arange(lg.numel()) % nb
    obs = torch.zeros(nb).index_add_(0, bin_id, cnt)
    exp = torch.zeros(nb).index_add_(0, bin_id, q * n_draw)
    chi2 = float((((obs - exp) ** 2) / exp.clamp_min(1e-9)).sum())
    print(f"lam={lam} chi2={chi2:.1f} (df={nb - 1})")
    assert chi2 < 3 * (nb - 1), chi2
    # and the individual heavy hitters within 5 sigma
    top = torch.topk(q, 20).indices
    z = (cnt[top] - q[top] * n_draw).abs() / (q[top] * n_draw).sqrt()
    assert float(z.max()) < 5.0, float(z.max())


def _zhat_samples(engine, model, emb, mask, lvl, K, reps):
    out = []
    for _ in range(reps):
        pan = engine.evaluate(model, emb, mask, lvl, n_beh=0, n_is=K)
        with torch.no_grad():
            l = engine.logits_of(model, pan.u_sel, pan.glob)
            out.append(float(torch.logsumexp(l - pan.log_q, dim=1) - math.log(K)))
    return torch.tensor(out)


@pytest.mark.parametrize("drift", [0.0, 0.05, 0.3])
def test_unbiased_with_stale_cache(drift):
    """Z_hat is unbiased for ANY proposal; log Z_hat is biased low by Jensen, by ~chi2/(2K)."""
    ctx, model, engine = make(scale=2.0, K=128, lam=0.1)
    if drift:
        model.perturb(drift)  # the cache now belongs to an older network
    emb, mask, lvl = state(n=1)
    lg, legal, _ = exact(ctx, model, emb, lvl)[0]
    logZ = float(torch.logsumexp(lg[legal], 0))

    z = _zhat_samples(engine, model, emb, mask, lvl, 128, 400)
    r = (z - logZ).exp()
    se = float(r.std()) / math.sqrt(r.numel())
    bias = logZ - float(z.mean())
    pred = 0.5 * float(r.var())  # second-order Jensen gap
    print(f"drift={drift}: E[Zhat]/Z={float(r.mean()):.4f} (se {se:.4f}) "
          f"log-bias={bias:.4f} predicted={pred:.4f} sd(logZhat)={float(z.std()):.4f}")
    assert abs(float(r.mean()) - 1.0) < max(4 * se, 0.02)
    assert bias >= -3 * float(z.std()) / math.sqrt(z.numel())  # Jensen: never biased high
    if drift <= 0.05:
        assert bias < 0.05


def test_log_bias_scales_like_one_over_K():
    ctx, model, engine = make(scale=2.0, K=64, lam=0.1)
    model.perturb(0.3)
    emb, mask, lvl = state(n=1)
    lg, legal, _ = exact(ctx, model, emb, lvl)[0]
    logZ = float(torch.logsumexp(lg[legal], 0))
    b = {}
    for K in (64, 256, 1024):
        z = _zhat_samples(engine, model, emb, mask, lvl, K, 250)
        b[K] = logZ - float(z.mean())
    print("log-bias vs K:", {k: round(v, 4) for k, v in b.items()})
    assert b[1024] < b[64] / 3.0
    assert b[256] < b[64] / 1.8


def test_variance_better_than_uniform_subsample():
    """Baseline: log Z_hat = log sum_{b in S} (1/rho) exp l_b with S uniform of size rho*N."""
    ctx, model, engine = make(scale=3.0, K=64, lam=0.1)
    emb, mask, lvl = state(n=1)
    lg, legal, _ = exact(ctx, model, emb, lvl)[0]
    lg_leg = lg[legal]
    logZ = float(torch.logsumexp(lg_leg, 0))
    n = lg_leg.numel()
    rho = 64 / n
    g = torch.Generator().manual_seed(7)
    unif = []
    for _ in range(400):
        sel = torch.randperm(n, generator=g)[:64]
        unif.append(float(torch.logsumexp(lg_leg[sel] - math.log(rho), 0)))
    pan = []
    for _ in range(400):
        s = engine.evaluate(model, emb, mask, lvl, n_beh=0, n_is=64)
        with torch.no_grad():
            l = engine.logits_of(model, s.u_sel, s.glob)
            pan.append(float(torch.logsumexp(l - s.log_q, dim=1) - math.log(64)))
    u = torch.tensor(unif)
    p = torch.tensor(pan)
    print(f"logZ={logZ:.3f} uniform: bias={u.mean() - logZ:+.3f} sd={u.std():.3f} | "
          f"pan: bias={p.mean() - logZ:+.3f} sd={p.std():.4f}")
    assert float(p.std()) < float(u.std()) / 5
    assert abs(float(p.mean()) - logZ) < abs(float(u.mean()) - logZ)


def test_gradient_matches_exact():
    ctx, model, engine = make(scale=2.0, K=256, lam=0.1)
    emb, mask, lvl = state(n=1)

    lg, legal, _ = brute_force_logits(ctx, model, emb[0], [0, 1], 1.0 - float(lvl[0]))
    exact_logZ = torch.logsumexp(lg[legal], 0)
    gex = torch.autograd.grad(exact_logZ, list(model.parameters()), allow_unused=True)

    acc = [torch.zeros_like(p) for p in model.parameters()]
    R = 60
    for _ in range(R):
        s = engine.evaluate(model, emb, mask, lvl, n_beh=0, n_is=256)
        l = engine.logits_of(model, s.u_sel, s.glob)
        lz = torch.logsumexp(l - s.log_q, dim=1) - math.log(256)
        gs = torch.autograd.grad(lz.sum(), list(model.parameters()), allow_unused=True)
        for a, b in zip(acc, gs, strict=True):
            if b is not None:
                a += b / R
    num = sum(float((a * b).sum()) for a, b in zip(acc, gex, strict=True) if b is not None)
    na = math.sqrt(sum(float((a * a).sum()) for a, b in zip(acc, gex, strict=True) if b is not None))
    nb = math.sqrt(sum(float((b * b).sum()) for b in gex if b is not None))
    cos = num / (na * nb)
    rel = math.sqrt(
        sum(float(((a - b) ** 2).sum()) for a, b in zip(acc, gex, strict=True) if b is not None)
    ) / nb
    print(f"grad cosine={cos:.5f} rel_err={rel:.4f}")
    assert cos > 0.995, cos
    assert rel < 0.12, rel


def test_uniform_error_grows_with_peakedness():
    """relvar(Z_hat_uniform) = (1/rho - 1) / M_eff  ->  the baseline degrades as the policy peaks,
    PAN does not.  This is the quantity the Step-0 study measures on real checkpoints."""
    rows = []
    for scale in (1.0, 3.0, 6.0, 10.0):
        ctx, model, engine = make(scale=scale, K=64, lam=0.1)
        emb, mask, lvl = state(n=1)
        lg, legal, _ = exact(ctx, model, emb, lvl)[0]
        lgl = lg[legal]
        logZ = float(torch.logsumexp(lgl, 0))
        p = (lgl - logZ).exp()
        meff = float(1.0 / (p * p).sum())
        n = lgl.numel()
        rho = 64 / n
        g = torch.Generator().manual_seed(11)
        u = torch.tensor(
            [
                float(torch.logsumexp(lgl[torch.randperm(n, generator=g)[:64]] - math.log(rho), 0))
                for _ in range(300)
            ]
        )
        z = _zhat_samples(engine, model, emb, mask, lvl, 64, 200)
        rows.append((scale, n, meff, float(u.mean()) - logZ, float(u.std()), float(z.mean()) - logZ, float(z.std())))
        print(
            f"scale={scale:4.1f} N={n} M_eff={meff:9.1f} | uniform bias={rows[-1][3]:+.3f} sd={rows[-1][4]:.3f}"
            f" | pan bias={rows[-1][5]:+.4f} sd={rows[-1][6]:.4f}"
        )
    # the uniform estimator degrades monotonically with peakedness; PAN stays exact
    assert rows[-1][4] > 5 * rows[0][4]
    assert all(r[6] < 0.05 for r in rows)
    assert all(abs(r[5]) < 0.05 for r in rows)
