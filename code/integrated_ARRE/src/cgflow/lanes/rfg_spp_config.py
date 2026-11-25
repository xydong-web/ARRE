"""RFG/SPP (critic-tilted behaviour policy) configuration.  Every switch defaults to OFF: with the defaults the lane trainer is the baseline."""

from dataclasses import dataclass

from gflownet.utils.misc import StrictDataClass


@dataclass
class RFGSPPConfig(StrictDataClass):
    """Critic-tilted behaviour policy (CTB).

    Attributes
    ----------
    enable : bool
        Master switch.  False -> baseline behaviour (the critic is never built, nothing is tilted).
    tilt : float
        lambda.  1.0 = the behaviour logits get +beta*(w_v*c_v(b)+w_q*c_q(b))/s_bar, i.e. the first-order term of sampling proportionally
        to R_c(x)=s(x)^beta under the critic's posterior-mean block contributions c(.).  0 disables the tilt.
    min_labels : int
        No tilt before the critic has seen this many docked molecules.
    ridge_grid : str
        Comma-separated L2 prior precisions of the (standardised) block-feature weights of the Bayesian-ridge critic.  One ridge solution is kept per value and the one with the
        best PREQUENTIAL Spearman (moving average; prediction before learning) is used - online model selection, no hand-tuned regularisation.
    kl_target : float
        Trust region of the tilt: mean KL(tilted behaviour || untilted policy) per step, in nats.  A multiplicative controller scales lambda_eff in [kl_min, 1]
        so that the KL stays near this value (0 disables).  Bounds how far the behaviour policy may move away from the policy (exploration / diversity guard).
    rho_lo, rho_hi : float
        Reliability gate.  lambda_eff = tilt * clip((rho_ema - rho_lo)/(rho_hi - rho_lo), 0, 1), where rho_ema is the exponential moving
        average of the PREQUENTIAL Spearman correlation (prediction made before the batch is used for training) of the critic's r_vina prediction.
    rho_ema : float
        Decay of that moving average.
    max_bonus : float
        |bonus| is clipped to this many nats (bounds the dynamic range of the tilt to 2*max_bonus).
    use_qed : bool
        Also tilt with the QED critic (weighted by the trajectory's QED preference).
    mode : str
        "behavior" (CTB: the critic only tilts the sampler's behaviour logits; the training side is untouched) or
        "prior" (CPP: the critic bonus is part of the POLICY logits l' = l_theta + bonus in BOTH the sampling and the training categorical,
        so sampling is on-policy and the network only learns the residual w.r.t. the critic prior; TB is parameterisation-invariant).
    roles : str
        "both" | "first" | "reagent": which synthesis steps are tilted (ablation).
    refit_every : int
        Solve the ridge system every k iterations (sufficient statistics are updated every iteration).
    std_floor, z_clip : float
        Feature standardisation (statistics of the block population): sd floor and |z| clip.
    stat_blocks_per_type : int
        Number of blocks per block type used to estimate the standardisation statistics.
    calibrate : bool
        Empirical-Bayes de-shrinkage (v3).  The ridge posterior mean is biased towards 0 (offline: the prequential OLS slope of the true reward on the prediction is 4-8 for the
        selected precision at n<=2k labels and ~1.2-2 at n>=8k), so the first-order tilt exp(beta*c/s_bar) built from it is far too weak early in the run although its RANKING
        (Spearman ~0.5 already at n=500) is useful.  With calibrate=True the block scores are multiplied by the prequential OLS slope s_o of y_o on the prediction
        (exponentially weighted, per output, clipped to [calib_min, calib_max]), i.e. the critic's amplitude is matched to the observed effect sizes.
    calib_decay : float
        Per-batch decay of the exponentially weighted moments used for the slope (0.9 ~ the last 10 batches ~ 640 molecules; offline toy: the estimate lags the improving critic and over-states the slope by ~15 %, i.e. the calibrated amplitude is ~10-15 % too large rather than 4-8x too small).
    calib_min, calib_max : float
        Clip range of the slope.
    group_norm : bool
        Structure-preserving tilt (v4).  The v3 bonus is added to the logits of the block actions WITHOUT normalisation, so besides re-weighting blocks inside a protocol it also moves
        probability mass BETWEEN protocols (reaction families, linker vs brick) and between "add another block" and "stop" (a positive mean bonus raises the continue probability by
        ~exp(var/2)).  Measured consequence at it=85: the share of 3-action routes among the top-100 rises from 5 % (baseline) to 20 % (E3) / 58 % (BE) and AiZynth success of these routes is
        0.41 vs 0.78 for 2-action routes.  With group_norm=True every protocol's bonus is shifted by  -log sum_b pi_p(b) exp(bonus_p(b))  (pi_p = softmax of the protocol's own logits over
        the sub-sample) so that the tilted policy has EXACTLY the untilted protocol / stop marginals and only the conditional distribution of the block given the protocol is tilted
        (conditional exponential tilt).
    syn_prior : str
        Path of a JSON route-level retrosynthesis-success model {family: {...}, gamma3: g, ...} (see tmp/lead_analysis/syn_profile.py).  Empty = off.
    syn_kappa : float
        Strength of the retrosynthesis-aware route prior: every reaction protocol gets the extra behaviour logit  kappa * log sigmoid(b0 + theta_family + gamma3 * [linker]),  i.e. the behaviour
        policy samples routes in proportion to P(AiZynth solves the route)^kappa on top of the (group-normalised) critic tilt.  0 = off.
    syn_lookahead : bool
        First-decision look-ahead of the route prior (Doob h-transform of the first decision).  The per-reaction offset alone is MYOPIC: the type K of the first block fixes which reactions can follow (e.g. type 2 -> only
        rxn35, a family with AiZynth success ~0.1 and a QED>0.5 rate of 0.15), so penalising rxn35 at the second decision merely moves mass between rxn35_linker and rxn35_brick (measured in the first 30 iterations with
        kappa=2: rxn35_brick share 12 % -> 34 %).  With syn_lookahead=true every first-block protocol 'blockK' gets  log( sum_p w_p exp(syn_bias(p)) / sum_p w_p )  over the BiRxn protocols p with state_type K
        (w_p = number of building blocks of p = the continuation distribution of the untrained policy).  Requires syn_kappa > 0.
    """

    enable: bool = False
    tilt: float = 1.0
    min_labels: int = 128
    ridge_grid: str = "300,3000,30000,300000"
    kl_target: float = 4.0
    kl_gain: float = 0.5
    kl_min: float = 0.1
    rho_lo: float = 0.05
    rho_hi: float = 0.30
    rho_ema: float = 0.9
    max_bonus: float = 6.0
    use_qed: bool = True
    mode: str = "behavior"
    roles: str = "both"
    refit_every: int = 1
    std_floor: float = 0.05
    z_clip: float = 10.0
    stat_blocks_per_type: int = 200
    calibrate: bool = True
    calib_decay: float = 0.9
    calib_min: float = 0.25
    calib_max: float = 8.0
    group_norm: bool = False
    syn_prior: str = ""
    syn_kappa: float = 0.0
    syn_lookahead: bool = False
