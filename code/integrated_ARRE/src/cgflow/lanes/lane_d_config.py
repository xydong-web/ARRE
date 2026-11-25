"""Lane D configuration.  Kept free of any ``cgflow`` import so that
``cgflow.config`` can import it without a cycle.  ALL switches default to the
baseline behaviour: with ``enable=False`` the trainer calls straight into
``VinaMOOTrainer_semla.build_training_data_loader`` and nothing in this lane
is constructed.
"""

from dataclasses import dataclass


@dataclass
class LaneDConfig:
    """Pose-aware proxy gating: propose F*64 candidates, dock the best 64.

    Attributes
    ----------
    enable : bool
        Master switch.  False -> the baseline data loader, bit for bit.
    oversample : int
        F.  Candidates proposed per iteration are F * algo.num_from_policy,
        drawn as F sequential chunks of num_from_policy so that per-call GPU
        memory and the F=1 code path are exactly the baseline's.
        The dominant cost of the lane: each extra chunk costs one extra pass
        of the SemlaFlow pose model (~25-30 s/iteration on the full env).
    act : bool
        Use the proxy to *select*.  False -> the first 64 candidates are taken
        (an unbiased on-policy sample) while the proxy is still trained and
        its diagnostics logged; this is the honest "proxy quality without
        feedback loop" ablation.
    fit : bool
        Train the proxy online.  False -> no proxy at all (pure oversample +
        take-first-64 control).
    ensemble : int
        M, number of bootstrapped MLP members.
    hidden : int
        Width of the two hidden layers of each member.
    warmup_labels : int
        Minimum number of docked labels before the gate may fire.
    min_rho : float
        Minimum running (EMA) prequential Spearman before the gate may fire.
    rho_ema : float
        EMA coefficient for the running Spearman used by the gate.
    rho_hysteresis : float
        The gate switches off again below ``min_rho - rho_hysteresis``.
    qed_floor : float
        Candidates with QED <= floor are selected only if the pool cannot fill
        the batch above it (soft floor, never a hard drop).
    kappa : float
        Optimism: candidates are ranked by mu + kappa * sigma.
    random_frac : float
        Fraction of the batch picked uniformly at random from the pool.  It
        keeps the selection distribution full-support (TB stays consistent)
        and provides the unbiased sample used for the lift/calibration
        diagnostics.  Must be > 0 for the theory in docs/lane_d/REPORT.md.
    strata : int
        Number of equal-count beta quantile strata the selection is run inside
        (1 = the unstratified gate of the lane spec).  Stratification keeps the
        beta marginal of the training batch equal to the U(0, 64) prior.
    score_mode : str
        "scalarized" (default): rank by w1*mu_vina + w2*QED under the
        candidate's own Dirichlet preference, sigma scaled by w1.
        "vina": rank by mu_vina + kappa*sigma_vina (the lane spec's variant).
    fit_epochs, fit_max_samples, fit_batch_size, lr, lambda_rank, rank_tau
        Online-fit hyper-parameters of the ensemble.
    bootstrap_p : float
        Per-member Bernoulli inclusion probability of each label.
    buffer_capacity : int
        FIFO capacity of the label buffer (~4.5 kB per label).
    rand_window : int
        Window (in labels) of the unbiased random-pick diagnostics.
    use_vina_local : bool
        NOT IMPLEMENTED - raises.  See docs/lane_d/REPORT.md (limitations):
        a Vina local-opt feature cannot be produced inside the 2 s/iteration
        budget for 128 candidates with the pdbqt round trip this repo uses.
    """

    enable: bool = False
    oversample: int = 2
    act: bool = True
    fit: bool = True

    ensemble: int = 5
    hidden: int = 128

    warmup_labels: int = 2000
    min_rho: float = 0.2
    rho_ema: float = 0.9
    rho_hysteresis: float = 0.05

    qed_floor: float = 0.45
    kappa: float = 1.0
    random_frac: float = 0.1
    strata: int = 4
    score_mode: str = "scalarized"

    fit_epochs: int = 2
    fit_max_samples: int = 4096
    fit_batch_size: int = 256
    lr: float = 1e-3
    lambda_rank: float = 1.0
    rank_tau: float = 1.0
    bootstrap_p: float = 0.8
    buffer_capacity: int = 20000
    rand_window: int = 256

    use_vina_local: bool = False
