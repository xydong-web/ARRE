"""Lane D - D3: online bootstrapped MLP ensemble that ranks docking candidates.

The ensemble predicts the UniDock reward  r = clip(-0.1 * vina, 1e-5)  of a
candidate from its pose-aware features (``lane_d_features``).  It is trained
*prequentially*: every iteration the current ensemble first scores the new
candidate pool, the selected 64 are docked, and only afterwards are those 64
labels added to the buffer and used for a few warm-started gradient steps.
The correlation reported in ``lane_d/rho_*`` is therefore always an honest
out-of-sample number.

Loss per member, on standardised targets  y~ = (y - mean) / std :

    L = MSE(f(x), y~)  +  lambda_rank * E_{(i,j)} [ w_ij * softplus(-sign(d_ij) * s_ij / tau) ]

with pairs (i, j = perm(i)) drawn inside the minibatch,
s_ij = f(x_i) - f(x_j), d_ij = y~_i - y~_j and w_ij = |d_ij| / mean|d|.
The weight makes near-ties (which are dominated by docking noise) contribute
little, so the ranking term concentrates on pairs the oracle can actually
separate.  MSE alone is a poor surrogate objective here because only the
*order* of the candidates is used downstream; the ranking term alone has no
scale and would make the ensemble spread meaningless.

Uncertainty is the ensemble disagreement, sigma(x) = std_m f_m(x), i.e. an
epistemic estimate only: it says "the members disagree here", not "docking is
noisy here".  Diagnostics report that honestly (`lane_d/disp_ratio`).

Diversity between members comes from (i) independent initialisation and
(ii) a fixed per-sample bootstrap mask ~ Bernoulli(bootstrap_p) drawn once when
the sample enters the buffer, so each member sees its own stable subsample.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# rank statistics (implemented here so the lane needs no extra dependency)
# --------------------------------------------------------------------------------------
def rankdata_average(a: np.ndarray) -> np.ndarray:
    """``scipy.stats.rankdata(a, method='average')`` for 1-D float arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan")
    return pearson(rankdata_average(x), rankdata_average(y))


# --------------------------------------------------------------------------------------
# label buffer
# --------------------------------------------------------------------------------------
class LabelBuffer:
    """FIFO ring buffer of (features, label, per-member bootstrap mask).

    FIFO (rather than reservoir) eviction is deliberate: the candidate
    distribution is non-stationary because the policy keeps moving, and the
    proxy only has to rank *the molecules the current policy proposes*.
    """

    def __init__(self, dim: int, capacity: int, n_members: int, bootstrap_p: float, rng: np.random.Generator):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.n_members = int(n_members)
        self.bootstrap_p = float(bootstrap_p)
        self.rng = rng
        self.X = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self.y = np.zeros((self.capacity,), dtype=np.float32)
        self.mask = np.zeros((self.capacity, self.n_members), dtype=bool)
        self.n = 0          # number of valid rows
        self._cursor = 0
        self.n_total = 0    # labels ever inserted (monotone; used by the warm-up gate)
        # running statistics over *all* inserted rows (not only the live window)
        self._sum = np.zeros(self.dim, dtype=np.float64)
        self._sumsq = np.zeros(self.dim, dtype=np.float64)
        self._ysum = 0.0
        self._ysumsq = 0.0

    def __len__(self) -> int:
        return self.n

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32).reshape(-1, self.dim)
        y = np.asarray(y, dtype=np.float32).ravel()
        assert X.shape[0] == y.shape[0]
        if X.shape[0] == 0:
            return
        masks = self.rng.random((X.shape[0], self.n_members)) < self.bootstrap_p
        # make sure no sample is invisible to every member
        empty = ~masks.any(axis=1)
        if empty.any():
            masks[empty, self.rng.integers(0, self.n_members, int(empty.sum()))] = True
        for i in range(X.shape[0]):
            c = self._cursor
            self.X[c] = X[i]
            self.y[c] = y[i]
            self.mask[c] = masks[i]
            self._cursor = (c + 1) % self.capacity
            self.n = min(self.n + 1, self.capacity)
        self.n_total += int(X.shape[0])
        self._sum += X.sum(axis=0, dtype=np.float64)
        self._sumsq += (X.astype(np.float64) ** 2).sum(axis=0)
        self._ysum += float(y.sum())
        self._ysumsq += float((y.astype(np.float64) ** 2).sum())

    def feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        n = max(self.n_total, 1)
        mean = self._sum / n
        var = np.maximum(self._sumsq / n - mean**2, 0.0)
        std = np.sqrt(var) + 1e-3
        return mean.astype(np.float32), std.astype(np.float32)

    def label_stats(self) -> tuple[float, float]:
        n = max(self.n_total, 1)
        mean = self._ysum / n
        var = max(self._ysumsq / n - mean**2, 0.0)
        return float(mean), float(np.sqrt(var) + 1e-6)


# --------------------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------------------
def _mlp(d_in: int, hidden: int, gen: torch.Generator) -> nn.Sequential:
    layers: list[nn.Module] = []
    sizes = [d_in, hidden, hidden, 1]
    for i in range(3):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        bound = 1.0 / float(np.sqrt(sizes[i]))
        with torch.no_grad():
            lin.weight.uniform_(-bound, bound, generator=gen)
            lin.bias.uniform_(-bound, bound, generator=gen)
        layers.append(lin)
        if i < 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class ProxyEnsemble:
    """M small MLPs trained online; all tensors live on ``device``."""

    def __init__(
        self,
        dim: int,
        n_members: int = 5,
        hidden: int = 128,
        lr: float = 1e-3,
        lambda_rank: float = 1.0,
        rank_tau: float = 1.0,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ):
        self.dim = int(dim)
        self.n_members = int(n_members)
        self.lambda_rank = float(lambda_rank)
        self.rank_tau = float(rank_tau)
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed) * 1000 + 7717)
        self.members = [_mlp(self.dim, hidden, self.gen).to(self.device) for _ in range(self.n_members)]
        self.opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.members]
        self.x_mean = torch.zeros(self.dim, device=self.device)
        self.x_std = torch.ones(self.dim, device=self.device)
        self.y_mean, self.y_std = 0.0, 1.0
        self.n_fit_steps = 0

    # -- normalisation ------------------------------------------------------------------
    def set_stats(self, x_mean: np.ndarray, x_std: np.ndarray, y_mean: float, y_std: float) -> None:
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32, device=self.device)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float32, device=self.device)
        self.y_mean, self.y_std = float(y_mean), float(y_std)

    def _norm(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.x_mean) / self.x_std

    # -- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mu, sigma) in *label* units (the same units as r_vina)."""
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        xb = self._norm(torch.as_tensor(X, dtype=torch.float32, device=self.device))
        preds = []
        for m in self.members:
            m.eval()
            preds.append(m(xb).squeeze(-1))
        P = torch.stack(preds, dim=0)  # (M, B)
        mu = P.mean(dim=0) * self.y_std + self.y_mean
        sigma = P.std(dim=0, unbiased=False) * self.y_std
        return mu.cpu().numpy().astype(np.float32), sigma.cpu().numpy().astype(np.float32)

    # -- training -----------------------------------------------------------------------
    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        if self.lambda_rank <= 0.0 or pred.numel() < 2:
            return mse
        perm = torch.randperm(pred.shape[0], generator=self.gen).to(pred.device)
        s = pred - pred[perm]
        d = target - target[perm]
        w = d.abs()
        w = w / (w.mean() + 1e-6)
        rank = (w * F.softplus(-torch.sign(d) * s / self.rank_tau)).mean()
        return mse + self.lambda_rank * rank

    def fit(self, buf: LabelBuffer, epochs: int = 2, max_samples: int = 4096, batch_size: int = 256,
            rng: np.random.Generator | None = None) -> float:
        """A few warm-started passes over a subsample of the buffer.  Returns the last loss."""
        if len(buf) < 8:
            return float("nan")
        rng = rng if rng is not None else np.random.default_rng(0)
        x_mean, x_std = buf.feature_stats()
        y_mean, y_std = buf.label_stats()
        self.set_stats(x_mean, x_std, y_mean, y_std)

        n = len(buf)
        take = min(n, int(max_samples))
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        X = torch.as_tensor(buf.X[idx], dtype=torch.float32, device=self.device)
        y = torch.as_tensor((buf.y[idx] - y_mean) / y_std, dtype=torch.float32, device=self.device)
        Xn = self._norm(X)
        masks = buf.mask[idx]

        last = float("nan")
        for m_i, (model, opt) in enumerate(zip(self.members, self.opts)):
            sub = np.nonzero(masks[:, m_i])[0]
            if sub.size < 8:
                continue
            sub_t = torch.as_tensor(sub, dtype=torch.long, device=self.device)
            model.train()
            for _ in range(int(epochs)):
                order = torch.randperm(sub_t.numel(), generator=self.gen).to(self.device)
                shuffled = sub_t[order]
                for start in range(0, shuffled.numel(), batch_size):
                    sel = shuffled[start: start + batch_size]
                    if sel.numel() < 4:
                        continue
                    pred = model(Xn[sel]).squeeze(-1)
                    loss = self._loss(pred, y[sel])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    self.n_fit_steps += 1
                    last = float(loss.detach().item())
        return last
