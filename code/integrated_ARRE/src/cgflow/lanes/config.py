"""Lane A (PAN-Sub) + Lane B (oracle-reuse learning engine) configuration (integration branch: both live in this module).

Kept in its own module so that `cgflow.config` can import it without pulling in torch or the
lane implementation.  Every switch defaults to OFF: with the defaults the lane trainer
reproduces the baseline exactly (tests/lane_a/test_equivalence.py).
"""

from dataclasses import dataclass

from gflownet.utils.misc import StrictDataClass


@dataclass
class LaneAConfig(StrictDataClass):
    """PAN-Sub: policy-adaptive normaliser sub-sampling.

    Attributes
    ----------
    enable : bool
        Master switch.  False -> the lane trainer is the baseline.
    normalizer : str
        "uniform" (baseline: log Z_hat from a uniform 1% sub-sample with importance weights)
        or "pan" (A1: self-normalised importance sampling against the cached-policy proposal).
    behavior : str
        "subsample" (baseline: softmax restricted to a random 1% sub-sample) or
        "full" (A2: (1-eps) * cached policy over the FULL legal action set + eps * uniform(legal)).
    num_samples : int
        K, the number of importance samples per state used by the PAN normaliser.
    defensive_mix : float
        lambda, weight of the uniform(legal) component of the PAN *proposal*.  Bounds the
        importance weight by |legal| / lambda.
    behavior_eps : float
        epsilon, weight of the uniform(legal) component of the A2 *behaviour* policy.
    refresh_every : int
        Refresh the cached block-embedding table every this many training iterations.
    diag_every : int
        Emit the (expensive) PAN diagnostics every this many iterations.  0 disables them.
    diag_num_states : int
        Number of states used for the exact-normaliser diagnostic.
    cache_dtype : str
        dtype of the cached block embeddings ("float16" or "float32").
    table_device : str
        Where the packed fingerprints / descriptors live: "auto", "cuda" or "cpu".
    chunk_elems : int
        Upper bound on n_states * n_blocks materialised by one chunk of the full-space pass.
    grad_checkpoint : bool
        Re-materialise the unpacked fingerprints in the backward pass instead of storing them.
    sampling_logprob : str
        How log P_F is reported while *sampling* (it is not used by the TB loss, which always
        re-evaluates the policy): "cached_exact" (l_theta(a) - log Z_cached, essentially exact
        for the sampling model) or "pan" (the same estimator as at training time).
    clamp_logprob : bool
        Keep the baseline clamp(max=0) safety net on log P_F.  The firing rate is logged.
    """

    enable: bool = False
    normalizer: str = "uniform"
    behavior: str = "subsample"
    num_samples: int = 1024
    defensive_mix: float = 0.1
    behavior_eps: float = 0.1
    refresh_every: int = 4
    diag_every: int = 20
    diag_num_states: int = 16
    cache_dtype: str = "float16"
    table_device: str = "auto"
    chunk_elems: int = 8388608
    grad_checkpoint: bool = True
    sampling_logprob: str = "cached_exact"
    clamp_logprob: bool = True


@dataclass
class LaneBConfig(StrictDataClass):
    """Lane B - oracle-reuse learning engine (prioritised replay + Bayesian condition-hindsight + multi-update).

    Every switch is off / neutral by default: with `enable=False` the lane trainer reduces to the baseline
    `VinaMOOTrainer_semla` (no hook is registered, `train_batch` only calls `super()`).

    Attributes
    ----------
    enable : bool
        Master switch of the lane.
    utd : int
        Updates-to-data ratio.  Per oracle batch: 1 baseline step on the fresh batch + (utd - 1) replay steps.
        utd=1 -> replay buffer is filled and logged but no extra gradient step is taken.
    replay_batch : int
        Number of trajectories per replay batch (baseline fresh batch is 64).
    capacity : int
        Maximum number of stored trajectories (FIFO eviction, elites protected).
    warmup_iters : int
        No extra steps before (and including) this training iteration.
    priority : str
        "rank" (p_i ~ 1/(rank_kappa*|D| + rank_i), rank by the evaluation criterion) or "uniform".
    rank_kappa : float
        Rank-offset of the prioritisation, as a fraction of the buffer size.
    uniform_mix : float
        Mixture weight of the uniform distribution: p = (1-uniform_mix) * p_rank + uniform_mix / |D|.
    hindsight : str
        "posterior_cal" (default; Bayesian condition relabeling with the kernel-regressed, TB-implied
        log-partition), "posterior" (same but using the model's logZ head, i.e. the literal lane spec)
        or "none".  See docs/lane_b/REPORT.md.
    hindsight_candidates : int
        Number M of conditions drawn from the prior p(c) for the self-normalised importance sampling of p(c|x).
    hindsight_prob : float
        Fraction of the replayed trajectories that are relabelled (the rest keep their sampling-time condition).
    hindsight_min_ess : float
        A sample is only relabelled when the normalised effective sample size of its importance sampler is at
        least this; otherwise it keeps its own condition (set 0.0 to disable the safeguard).
    cal_history : int
        Number of most recent stored trajectories used by the kernel regression of log Z(c).
    cal_min_samples : int
        Below this many stored trajectories the kernel estimate is replaced by the model's logZ head.
    cal_bandwidth_beta : float
        Gaussian kernel bandwidth along beta/64.
    cal_bandwidth_pref : float
        Gaussian kernel bandwidth along the vina preference weight w_0.
    protected_elites : int
        The top-`protected_elites` trajectories (by the ranking criterion) are never evicted.
    skip_failed_docking : bool
        Do not store trajectories whose UniDock call failed (score 0.0 -> vina reward clipped to 1e-4).
    ema_on_extra : bool
        Run the sampling-model EMA after every extra optimizer step (as the baseline `step` does).
    lr_sched_on_extra : bool
        Advance the LR schedulers on the extra steps as well.  Default False so that the learning-rate
        schedule stays exactly the baseline's (the lane adds gradient steps, not a new LR schedule).
    selfcheck_replay_data : bool
        Fingerprint every stored trajectory's sampling-time features and re-check them at replay time
        (diagnostic `lane_b/selfcheck_mismatch`).  Costs one extra `graph_to_Data` pass per stored trajectory.
    """

    enable: bool = False
    utd: int = 4
    replay_batch: int = 64
    capacity: int = 12000
    warmup_iters: int = 20
    priority: str = "rank"
    rank_kappa: float = 0.01
    uniform_mix: float = 0.2
    hindsight: str = "posterior_cal"
    hindsight_candidates: int = 32
    hindsight_prob: float = 1.0
    hindsight_min_ess: float = 0.1
    cal_history: int = 1024
    cal_min_samples: int = 256
    cal_bandwidth_beta: float = 0.1
    cal_bandwidth_pref: float = 0.15
    protected_elites: int = 200
    skip_failed_docking: bool = True
    ema_on_extra: bool = True
    lr_sched_on_extra: bool = False
    selfcheck_replay_data: bool = False
