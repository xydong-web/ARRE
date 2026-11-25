"""T2 - rank statistics, the label buffer and the online ensemble."""

import numpy as np
import torch

from cgflow.lanes.lane_d_proxy import LabelBuffer, ProxyEnsemble, pearson, rankdata_average, spearman


def test_rankdata_matches_scipy():
    from scipy.stats import rankdata

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.integers(0, 5, size=50).astype(float)  # lots of ties
        assert np.allclose(rankdata_average(a), rankdata(a, method="average"))


def test_spearman_pearson_match_scipy():
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 0.6 * x + 0.8 * rng.normal(size=200)
    assert abs(spearman(x, y) - spearmanr(x, y).statistic) < 1e-9
    assert abs(pearson(x, y) - pearsonr(x, y)[0]) < 1e-9


def test_degenerate_inputs_are_nan_not_crash():
    assert np.isnan(spearman(np.ones(10), np.ones(10)))
    assert np.isnan(spearman(np.zeros(2), np.zeros(2)))
    assert np.isnan(pearson(np.arange(2.0), np.arange(2.0)))


def test_buffer_fifo_and_stats():
    rng = np.random.default_rng(2)
    buf = LabelBuffer(dim=4, capacity=10, n_members=3, bootstrap_p=0.8, rng=rng)
    X = rng.normal(size=(25, 4)).astype(np.float32)
    y = rng.normal(size=25).astype(np.float32)
    buf.add(X, y)
    assert len(buf) == 10 and buf.n_total == 25
    assert buf.mask.any(axis=1).all(), "no label may be invisible to every member"
    mean, std = buf.feature_stats()
    assert np.allclose(mean, X.mean(axis=0), atol=1e-5)
    assert np.allclose(std, X.std(axis=0) + 1e-3, atol=1e-4)
    ym, _ = buf.label_stats()
    assert abs(ym - float(y.mean())) < 1e-5


def test_buffer_bootstrap_members_differ():
    rng = np.random.default_rng(3)
    buf = LabelBuffer(dim=2, capacity=500, n_members=5, bootstrap_p=0.8, rng=rng)
    buf.add(rng.normal(size=(400, 2)).astype(np.float32), rng.normal(size=400).astype(np.float32))
    frac = buf.mask[: len(buf)].mean(axis=0)
    assert np.all(np.abs(frac - 0.8) < 0.08)
    assert (buf.mask[:, 0] == buf.mask[:, 1]).mean() < 0.95


def _synthetic(n, d, rng):
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (np.tanh(X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.3 * X[:, 2] * X[:, 3]).astype(np.float32)
    return X, y


def _train_and_score(lambda_rank: float, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    d = 8
    Xtr, ytr = _synthetic(1500, d, rng)
    Xte, yte = _synthetic(400, d, rng)
    buf = LabelBuffer(dim=d, capacity=4000, n_members=3, bootstrap_p=0.8, rng=rng)
    buf.add(Xtr, ytr)
    ens = ProxyEnsemble(d, n_members=3, hidden=64, lr=3e-3, lambda_rank=lambda_rank, device="cpu", seed=seed)
    for _ in range(25):
        ens.fit(buf, epochs=1, max_samples=1500, batch_size=128, rng=rng)
    mu, _ = ens.predict(Xte)
    return spearman(mu, yte)


def test_proxy_learns_a_synthetic_function():
    assert _train_and_score(lambda_rank=0.0) > 0.85


def test_ranking_loss_also_learns_the_order():
    assert _train_and_score(lambda_rank=2.0) > 0.85


def test_ranking_loss_term_is_active_and_finite():
    ens = ProxyEnsemble(4, n_members=1, hidden=8, lambda_rank=1.0, device="cpu", seed=0)
    target = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    good = ens._loss(target.clone(), target)
    bad = ens._loss(-target.clone(), target)
    assert torch.isfinite(good) and torch.isfinite(bad)
    assert float(bad) > float(good)


def test_sigma_grows_off_support():
    rng = np.random.default_rng(11)
    d = 6
    Xtr, ytr = _synthetic(800, d, rng)
    buf = LabelBuffer(dim=d, capacity=2000, n_members=5, bootstrap_p=0.8, rng=rng)
    buf.add(Xtr, ytr)
    ens = ProxyEnsemble(d, n_members=5, hidden=64, lr=3e-3, lambda_rank=1.0, device="cpu", seed=1)
    for _ in range(20):
        ens.fit(buf, epochs=1, max_samples=800, batch_size=128, rng=rng)
    mu_in, sig_in = ens.predict(Xtr[:200])
    far = (rng.normal(size=(200, d)) * 6.0 + 12.0).astype(np.float32)
    _, sig_far = ens.predict(far)
    assert np.isfinite(mu_in).all() and np.isfinite(sig_in).all()
    assert sig_far.mean() > sig_in.mean(), "disagreement must grow away from the data"


def test_predict_is_deterministic_and_empty_safe():
    rng = np.random.default_rng(12)
    ens = ProxyEnsemble(5, n_members=2, hidden=16, device="cpu", seed=3)
    X = rng.normal(size=(7, 5)).astype(np.float32)
    a, _ = ens.predict(X)
    b, _ = ens.predict(X)
    assert np.array_equal(a, b)
    mu, sig = ens.predict(np.zeros((0, 5), dtype=np.float32))
    assert mu.shape == (0,) and sig.shape == (0,)


def test_fit_with_too_few_labels_is_a_noop():
    rng = np.random.default_rng(13)
    buf = LabelBuffer(dim=3, capacity=100, n_members=2, bootstrap_p=0.8, rng=rng)
    buf.add(rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=4).astype(np.float32))
    ens = ProxyEnsemble(3, n_members=2, hidden=8, device="cpu", seed=0)
    assert np.isnan(ens.fit(buf, rng=rng))
    assert ens.n_fit_steps == 0
