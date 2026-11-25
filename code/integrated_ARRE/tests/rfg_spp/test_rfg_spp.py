"""Unit tests of RFG/SPP (CPU).  Run:  PYTHONPATH=<worktree>/src pytest -q tests/rfg_spp"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rxnflow.base.trainer import mogfn_trainer  # noqa: F401  (import-cycle workaround, see rfg_spp.py)
from gflownet.utils.transforms import thermometer
from rxnflow.envs import RxnActionType
from cgflow.lanes.rfg_spp import BlockCritic, RFGSPPState, spearman
from cgflow.lanes.rfg_spp_config import RFGSPPConfig

DEV = torch.device("cpu")


def make_cfg():
    return SimpleNamespace(
        cond=SimpleNamespace(
            temperature=SimpleNamespace(sample_dist="uniform", num_thermometer_dim=32, dist_params=[0, 64]),
            moo=SimpleNamespace(num_thermometer_dim=16, num_objectives=2),
        )
    )


def make_state(cfg_e=None, D=24, fp=20):
    cfg_e = cfg_e or RFGSPPConfig(enable=True, ridge_grid="1.0", min_labels=10)
    critic = BlockCritic(cfg_e, torch.zeros(D), torch.ones(D), DEV)
    protos = [SimpleNamespace(name="first", action=RxnActionType.FirstBlock), SimpleNamespace(name="rxn", action=RxnActionType.BiRxn)]
    return RFGSPPState(cfg_e, make_cfg(), critic, SimpleNamespace(protocols=protos)), critic


def test_decode_roundtrip_matches_task_encoding():
    st, _ = make_state()
    g = torch.Generator().manual_seed(0)
    beta = torch.rand(200, generator=g) * 64
    w = torch.distributions.Dirichlet(torch.full((2,), 1.5)).sample((200,))
    enc = torch.cat([thermometer(beta, 32, 0, 64), thermometer(w, 16, 0, 1).reshape(200, -1)], 1)
    b2, w2 = st.decode(enc)
    assert (b2 - beta).abs().max() < 1e-3, (b2 - beta).abs().max()
    assert (w2 - w).abs().max() < 1e-4


def _toy(n_blocks=300, D=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n_blocks, D, generator=g)
    w_true = torch.randn(2, D, 2, generator=g) * 0.3  # [role, D, out]
    return phi, w_true, g


def test_critic_recovers_a_block_additive_function():
    st, critic = make_state(RFGSPPConfig(enable=True, ridge_grid="3.0", min_labels=10), D=24)
    phi, w_true, g = _toy()
    desc, fp = phi[:, :4], phi[:, 4:]  # critic.phi() concatenates [fp, desc] -> reorder so that phi((desc,fp)) == original columns
    desc, fp = phi[:, 20:], phi[:, :20]
    rhos = []
    for it in range(30):
        m = 64
        b = torch.randint(0, phi.shape[0], (m, 3), generator=g)
        mol = torch.arange(m).repeat_interleave(3)
        role = torch.tensor([0, 1, 1]).repeat(m)
        idx = b.reshape(-1)
        x = critic.build_x(desc[idx], fp[idx], mol, role, m)
        contrib = torch.einsum("nd,ndo->no", phi[idx], w_true[role])
        y = torch.zeros(m, 2).index_add_(0, mol, contrib) + 0.05 * torch.randn(m, 2, generator=g)
        if critic.fitted:
            rhos.append(spearman(critic.predict(x)[:, 0], y[:, 0]))
        critic.update(x, y)
        critic.fit()
    assert np.mean(rhos[-10:]) > 0.9, rhos[-10:]
    # block scores of a candidate set correlate with the true role-wise contributions
    sc = critic.block_scores((None, desc, fp), role=1)[:, 0]
    true = (phi @ w_true[1][:, 0])
    assert spearman(sc, true) > 0.9


def test_gate_ramps_with_prequential_rho():
    st, critic = make_state(RFGSPPConfig(enable=True, tilt=2.0, rho_lo=0.1, rho_hi=0.3, min_labels=10), D=8)
    assert st.lam_eff() == 0.0  # not fitted
    critic.fitted, critic.n = True, 100
    assert st.lam_eff() == 0.0  # rho unknown
    st.rho_ema = 0.05
    assert st.lam_eff() == 0.0
    st.rho_ema = 0.2
    assert abs(st.lam_eff() - 1.0) < 1e-6
    st.rho_ema = 0.9
    assert abs(st.lam_eff() - 2.0) < 1e-6
    critic.n = 5
    assert st.lam_eff() == 0.0


def test_taus_scale_with_beta_and_preference():
    st, _ = make_state(RFGSPPConfig(enable=True, use_qed=True), D=8)
    # same preference, beta 0 / 32 / 64  ->  tau proportional to beta
    beta = torch.tensor([0.0, 32.0, 64.0])
    w = torch.tensor([[0.5, 0.5]] * 3)
    tv, tq = st.taus(beta, w, lam=1.0)
    assert tv[0] == 0 and tq[0] == 0
    assert abs(float(tv[2] / tv[1]) - 2.0) < 1e-5 and abs(float(tq[2] / tq[1]) - 2.0) < 1e-5
    # same beta, different preferences -> the objective with the larger weight gets the larger tau
    w2 = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    tv2, tq2 = st.taus(torch.tensor([40.0, 40.0]), w2, lam=1.0)
    assert tv2[0] > tv2[1] and tq2[1] > tq2[0]
    # lambda is a pure multiplier
    tv3, _ = st.taus(torch.tensor([40.0, 40.0]), w2, lam=0.5)
    assert torch.allclose(tv3, 0.5 * tv2)
    st2, _ = make_state(RFGSPPConfig(enable=True, use_qed=False), D=8)
    _, tq_off = st2.taus(beta, w, lam=1.0)
    assert float(tq_off.abs().sum()) == 0.0


class FakeCat:
    def __init__(self, raw):
        self.raw_logits = raw


def test_bonus_shapes_clip_and_masked_entries_stay_masked():
    st, _ = make_state(RFGSPPConfig(enable=True, max_bonus=2.0), D=8)
    g = torch.Generator().manual_seed(1)
    raw = [torch.randn(5, 7, generator=g), torch.randn(5, 11, generator=g)]
    raw[1][:, 3] = -float("inf")
    scores = {"first": torch.randn(7, 2, generator=g) * 5, "rxn": torch.randn(11, 2, generator=g) * 5}
    tv, tq = torch.full((5,), 3.0), torch.full((5,), 1.0)
    b = st.bonuses(raw, scores, tv, tq)
    assert [x.shape for x in b] == [x.shape for x in raw]
    assert all(float(x.abs().max()) <= 2.0 + 1e-6 for x in b)
    assert torch.isinf((raw[1] + b[1])[:, 3]).all()
    # a protocol without a score (or with a mismatching one) gets zeros
    b2 = st.bonuses(raw, {"first": scores["first"][:3]}, tv, tq)
    assert float(b2[0].abs().sum()) == 0 and float(b2[1].abs().sum()) == 0
    # role ablation
    st3, _ = make_state(RFGSPPConfig(enable=True, roles="first"), D=8)
    b3 = st3.bonuses(raw, scores, tv, tq)
    assert float(b3[1].abs().sum()) == 0 and float(b3[0].abs().sum()) > 0


def test_tilt_moves_probability_mass_towards_high_critic_score_and_lowers_entropy():
    st, _ = make_state(RFGSPPConfig(enable=True, max_bonus=20.0), D=8)
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(1, 5000, generator=g)
    score = torch.randn(5000, 2, generator=g) * 0.1
    scores = {"first": score}
    prev_mean, prev_h = None, None
    for tau in (0.0, 5.0, 20.0, 60.0):
        b = st.bonuses([logits, torch.zeros(1, 0)], {"first": score, "rxn": torch.zeros(0, 2)}, torch.tensor([tau]), torch.tensor([0.0]))
        p = torch.softmax(logits + b[0], 1)[0]
        mean_score = float((p * score[:, 0]).sum())
        h = float(-(p * p.clamp_min(1e-30).log()).sum())
        if prev_mean is not None:
            assert mean_score > prev_mean and h < prev_h
        prev_mean, prev_h = mean_score, h


def test_default_config_is_off():
    assert RFGSPPConfig().enable is False


def test_chunked_block_scores_equal_unchunked():
    st, critic = make_state(RFGSPPConfig(enable=True, ridge_grid="1.0"), D=24)
    g = torch.Generator().manual_seed(5)
    critic.W = torch.randn(2, 24, 2, generator=g)
    desc, fp = torch.randn(40000, 4, generator=g), torch.randn(40000, 20, generator=g)
    full = critic.phi(desc, fp) @ critic.W[1]
    chunked = critic.block_scores((None, desc, fp), 1)
    assert chunked.shape == (40000, 2) and torch.allclose(full, chunked, atol=1e-5)


def test_kl_of_matches_manual_and_is_zero_without_bonus():
    g = torch.Generator().manual_seed(7)
    l1, l2 = torch.randn(3, 50, generator=g), torch.randn(3, 70, generator=g)
    l2[:, 5] = -float("inf")
    z1, z2 = torch.zeros_like(l1), torch.zeros_like(l2)
    assert RFGSPPState.kl_of([l1, l2], [z1, z2]) < 1e-6
    b1, b2 = torch.randn(3, 50, generator=g), torch.randn(3, 70, generator=g)
    kl = RFGSPPState.kl_of([l1, l2], [b1, b2])
    lb, lt = torch.cat([l1, l2], 1), torch.cat([l1 + b1, l2 + b2], 1)
    p, q = torch.softmax(lb, 1), torch.softmax(lt, 1)
    ref = torch.where(q > 0, q * (q.log() - p.log()), torch.zeros_like(q)).sum(1).mean()
    assert abs(kl - float(ref)) < 1e-4 and kl > 0


def test_kl_trust_region_controller_shrinks_and_recovers():
    st, _ = make_state(RFGSPPConfig(enable=True, kl_target=2.0, kl_gain=0.5, kl_min=0.1), D=8)
    for _ in range(5):
        st.kl_acc = [8.0]
        st.update_kl()
    shrunk = st.kl_scale
    assert 0.1 <= shrunk < 0.5, shrunk
    for _ in range(30):
        st.kl_acc = [0.2]
        st.update_kl()
    assert st.kl_scale > shrunk and st.kl_scale <= 1.0
    # disabled
    st2, _ = make_state(RFGSPPConfig(enable=True, kl_target=0.0), D=8)
    st2.kl_acc = [100.0]
    st2.update_kl()
    assert st2.kl_scale == 1.0
    # lam_eff includes the scale
    st.critic.fitted, st.critic.n, st.rho_ema, st.kl_scale = True, 500, 0.9, 0.5
    assert abs(st.lam_eff() - 0.5 * st.cfg.tilt) < 1e-6


def test_ridge_grid_selection_prefers_strong_regularisation_with_few_noisy_labels():
    cfg = RFGSPPConfig(enable=True, ridge_grid="1,100000", min_labels=10)
    st, critic = make_state(cfg, D=24)
    phi, w_true, g = _toy()
    desc, fp = phi[:, 20:], phi[:, :20]
    rho_k = [[], []]
    for it in range(12):
        m = 20
        b = torch.randint(0, phi.shape[0], (m, 3), generator=g)
        mol = torch.arange(m).repeat_interleave(3)
        role = torch.tensor([0, 1, 1]).repeat(m)
        idx = b.reshape(-1)
        x = critic.build_x(desc[idx], fp[idx], mol, role, m)
        contrib = torch.einsum("nd,ndo->no", phi[idx], w_true[role])
        y = torch.zeros(m, 2).index_add_(0, mol, contrib) + 2.0 * torch.randn(m, 2, generator=g)  # very noisy labels
        if critic.fitted:
            for k, p in enumerate(critic.predict_all(x)):
                rho_k[k].append(spearman(p[:, 0], y[:, 0]))
        critic.update(x, y)
        critic.fit()
    # with 240 very noisy labels and 49 weights the weakly regularised solution must not beat the strongly regularised one on average
    assert np.nanmean(rho_k[1]) >= np.nanmean(rho_k[0]) - 0.05
    critic.select(0)
    assert critic.k == 0 and torch.equal(critic.coef, critic.coefs[0])


def test_critic_state_dict_roundtrip_is_exact():
    cfg_e = RFGSPPConfig(enable=True, ridge_grid="1.0,10.0,100.0", min_labels=10)
    _, c1 = make_state(cfg_e, D=24)
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        x = torch.randn(32, 2 * 24 + 1, generator=g)
        x[:, -1] = 1.0
        c1.update(x, torch.randn(32, 2, generator=g))
        c1.fit()
    c1.select(2)
    _, c2 = make_state(cfg_e, D=24)
    c2.load_state_dict(c1.state_dict())
    xt = torch.randn(10, 49, generator=g)
    assert c2.k == 2 and c2.n == c1.n and c2.fitted
    assert torch.allclose(c1.predict(xt), c2.predict(xt))
    assert all(torch.allclose(a, b) for a, b in zip(c1.predict_all(xt), c2.predict_all(xt), strict=True))


def test_calibration_matches_effect_size_and_state_roundtrip():
    cfg_e = RFGSPPConfig(enable=True, ridge_grid="3000.0", min_labels=10)
    st, crit = make_state(cfg_e, D=24)
    g = torch.Generator().manual_seed(3)
    w_true = torch.randn(2 * 24, 2, generator=g) * 0.2

    def batch(m=64):
        x = torch.randn(m, 2 * 24 + 1, generator=g)
        x[:, -1] = 1.0
        return x, x[:, :-1] @ w_true + 0.3 * torch.randn(m, 2, generator=g)

    for _ in range(40):  # prequential: predict, record calibration moments, then learn
        x, y = batch()
        if crit.fitted:
            st.cal_update(0, crit.predict_all(x)[0], y)
        crit.update(x, y)
        crit.fit()
    sc = st.cal_scale(0)
    assert (sc > 1.05).all(), sc  # strong ridge on 2.5k samples under-states the effect sizes
    xt, yt = batch(4000)
    raw = (xt[:, :-1] @ crit.coef[:-1])
    crit.set_scale(sc)
    cal = xt[:, :-1] @ crit.W.reshape(-1, 2)
    for o in range(2):
        slope = lambda p: float(((p[:, o] - p[:, o].mean()) * (yt[:, o] - yt[:, o].mean())).sum() / ((p[:, o] - p[:, o].mean()) ** 2).sum())
        assert slope(raw) > 1.1 and 0.7 < slope(cal) < 1.15, (slope(raw), slope(cal))  # the prequential estimate lags the improving critic: mild over-correction
    # off switch -> neutral
    st.cfg.calibrate = False
    assert np.allclose(st.cal_scale(0), 1.0)
    st.cfg.calibrate = True
    # the scale survives the state round trip
    _, c2 = make_state(cfg_e, D=24)
    c2.load_state_dict(crit.state_dict())
    assert torch.allclose(c2.scale, crit.scale) and torch.allclose(c2.W, crit.W)


def test_group_normalize_preserves_protocol_mass_and_masked_entries():
    g = torch.Generator().manual_seed(11)
    logit = torch.randn(4, 50, generator=g)
    logit[:, 7] = -float("inf")
    logit[2, :] = -float("inf")  # a state for which the whole protocol is masked
    b = torch.randn(4, 50, generator=g) * 3
    bn = RFGSPPState.group_normalize(logit, b)
    assert bn.shape == b.shape and torch.isfinite(bn).all()
    for r in (0, 1, 3):
        pi = torch.softmax(logit[r], 0)
        assert abs(float((pi * torch.exp(bn[r])).sum()) - 1.0) < 1e-5  # protocol mass preserved ...
        assert torch.allclose(torch.softmax(logit[r] + b[r], 0), torch.softmax(logit[r] + bn[r], 0), atol=1e-6)  # ... conditional distribution identical to the plain tilt
        assert float((pi * torch.exp(b[r])).sum()) != pytest.approx(1.0, abs=1e-3)  # whereas the plain bonus does move the mass
    assert torch.equal(bn[2], b[2])  # fully masked row untouched


def test_bonuses_normalize_keeps_joint_protocol_marginals():
    st, _ = make_state(RFGSPPConfig(enable=True, max_bonus=20.0), D=8)
    g = torch.Generator().manual_seed(12)
    raw = [torch.randn(6, 30, generator=g), torch.randn(6, 40, generator=g) + 1.0]
    scores = {"first": torch.randn(30, 2, generator=g), "rxn": torch.randn(40, 2, generator=g) + 0.5}  # the reagent protocol has a positive MEAN score
    tv, tq = torch.full((6,), 2.0), torch.full((6,), 0.5)

    def masses(bon):
        p = torch.softmax(torch.cat([l + b for l, b in zip(raw, bon)], 1), 1)
        return p[:, :30].sum(1), p[:, 30:].sum(1)

    m0 = masses([torch.zeros_like(r) for r in raw])
    m_plain = masses(st.bonuses(raw, scores, tv, tq, normalize=False))
    m_norm = masses(st.bonuses(raw, scores, tv, tq, normalize=True))
    assert torch.allclose(m_norm[0], m0[0], atol=1e-5) and torch.allclose(m_norm[1], m0[1], atol=1e-5)
    assert float((m_plain[1] - m0[1]).abs().max()) > 0.02  # the unnormalised tilt shifts mass between protocols


def test_syn_bias_charges_reaction_protocols_and_linkers_only(tmp_path):
    import json

    f = tmp_path / "syn.json"
    f.write_text(json.dumps({"family": {"rxn3_brick": 0.6, "rxn3_linker": 0.9, "rxn35_brick": -3.0}, "gamma3": -2.0, "default": 0.0, "b0": 1.0}))
    cfg_e = RFGSPPConfig(enable=True, syn_prior=str(f), syn_kappa=0.5)
    critic = BlockCritic(cfg_e, torch.zeros(8), torch.ones(8), DEV)
    protos = [
        SimpleNamespace(name="block2", action=RxnActionType.FirstBlock),
        SimpleNamespace(name="rxn3_brick_b0", action=RxnActionType.BiRxn),
        SimpleNamespace(name="rxn3_linker_b1", action=RxnActionType.BiRxn),
        SimpleNamespace(name="rxn35_brick_b0", action=RxnActionType.BiRxn),
        SimpleNamespace(name="rxn99_brick_b0", action=RxnActionType.BiRxn),
        SimpleNamespace(name="stop", action=RxnActionType.Stop),
    ]
    st = RFGSPPState(cfg_e, make_cfg(), critic, SimpleNamespace(protocols=protos))
    got = [st.syn_bias(p) for p in protos]
    ls = lambda z: float(-np.log1p(np.exp(-z)))  # log sigmoid
    assert got == pytest.approx([0.0, 0.5 * ls(1.0 + 0.6), 0.5 * ls(1.0 + 0.9 - 2.0), 0.5 * ls(1.0 - 3.0), 0.5 * ls(1.0), 0.0])
    assert got[3] < got[2] < got[1] < 0.0  # bad family < linker < good brick
    assert RFGSPPState.family("rxn3_brick_b1") == "rxn3_brick" and RFGSPPState.family("block2") == "block2"
    st_off, _ = make_state(RFGSPPConfig(enable=True, syn_kappa=0.0, syn_prior=str(f)), D=8)
    assert st_off.syn is None and st_off.syn_bias(protos[1]) == 0.0


def test_syn_lookahead_penalises_first_blocks_whose_only_continuations_are_bad(tmp_path):
    import json

    f = tmp_path / "syn.json"
    f.write_text(json.dumps({"family": {"rxn35_brick": -3.0, "rxn35_linker": -2.4, "rxn1_brick": 1.7, "rxn1_linker": 1.9}, "gamma3": -2.3, "default": 0.0, "b0": 1.0}))
    protos = [
        SimpleNamespace(name="block2", action=RxnActionType.FirstBlock, block_types=["2"]),
        SimpleNamespace(name="block1", action=RxnActionType.FirstBlock, block_types=["1"]),
        SimpleNamespace(name="block9", action=RxnActionType.FirstBlock, block_types=["9"]),  # no follower at all
        SimpleNamespace(name="rxn35_brick_b0", action=RxnActionType.BiRxn, state_type=2, block_types=["7"]),
        SimpleNamespace(name="rxn35_linker_b0", action=RxnActionType.BiRxn, state_type=2, block_types=["1-7", "2-7"]),
        SimpleNamespace(name="rxn1_brick_b0", action=RxnActionType.BiRxn, state_type=1, block_types=["3"]),
        SimpleNamespace(name="rxn1_linker_b0", action=RxnActionType.BiRxn, state_type=1, block_types=["1-10"]),
    ]
    blocks = {"7": list(range(100)), "1-7": [0] * 300, "2-7": [0] * 300, "3": [0] * 100, "1-10": [0] * 200}
    ctx = SimpleNamespace(protocols=protos, blocks=blocks)
    ls = lambda z: float(-np.log1p(np.exp(-z)))

    def build(lookahead):
        cfg_e = RFGSPPConfig(enable=True, syn_prior=str(f), syn_kappa=1.0, syn_lookahead=lookahead)
        return RFGSPPState(cfg_e, make_cfg(), BlockCritic(cfg_e, torch.zeros(8), torch.ones(8), DEV), ctx)

    st = build(True)
    exp2 = np.log((100 * np.exp(ls(1.0 - 3.0)) + 600 * np.exp(ls(1.0 - 2.4 - 2.3))) / 700)
    exp1 = np.log((100 * np.exp(ls(1.0 + 1.7)) + 200 * np.exp(ls(1.0 + 1.9 - 2.3))) / 300)
    b2, b1, b9 = (st.syn_bias(p) for p in protos[:3])
    assert b2 == pytest.approx(exp2) and b1 == pytest.approx(exp1) and b9 == 0.0
    assert b2 < b1 < 0.0  # the type whose only follower family is bad is the least attractive first block
    assert all(build(False).syn_bias(p) == 0.0 for p in protos[:3])  # look-ahead is off by default


def test_prefix_logw_is_the_log_density_ratio_of_the_tilted_prefix_up_to_a_constant(tmp_path):
    """Doob bookkeeping behind RHC's route-aware prefix proposals: for the forced prefix (a1) the tilted/untilted probability ratio is exp(look-ahead bias of a1) / Z1, and for
    the forced prefix (a1, p2) it is exp(s(p2)) / Z1 - the bias of a1 cancels against the normaliser of the second decision.  Checked by explicit enumeration of a two-level tree."""
    import json

    f = tmp_path / "syn.json"
    f.write_text(json.dumps({"family": {"rxn35_brick": -3.0, "rxn35_linker": -2.4, "rxn1_brick": 1.7, "rxn1_linker": 1.9}, "gamma3": -2.3, "default": 0.0, "b0": 1.0}))
    protos = [
        SimpleNamespace(name="block2", action=RxnActionType.FirstBlock, block_types=["2"]),
        SimpleNamespace(name="block1", action=RxnActionType.FirstBlock, block_types=["1"]),
        SimpleNamespace(name="rxn35_brick_b0", action=RxnActionType.BiRxn, state_type=2, block_types=["7"]),
        SimpleNamespace(name="rxn35_linker_b0", action=RxnActionType.BiRxn, state_type=2, block_types=["1-7", "2-7"]),
        SimpleNamespace(name="rxn1_brick_b0", action=RxnActionType.BiRxn, state_type=1, block_types=["3"]),
        SimpleNamespace(name="rxn1_linker_b0", action=RxnActionType.BiRxn, state_type=1, block_types=["1-10"]),
    ]
    blocks = {"7": list(range(100)), "1-7": [0] * 300, "2-7": [0] * 300, "3": [0] * 100, "1-10": [0] * 200}
    cfg_e = RFGSPPConfig(enable=True, syn_prior=str(f), syn_kappa=2.0, syn_lookahead=True)
    st = RFGSPPState(cfg_e, make_cfg(), BlockCritic(cfg_e, torch.zeros(8), torch.ones(8), DEV), SimpleNamespace(protocols=protos, blocks=blocks))
    act = lambda p: SimpleNamespace(action=p.action, protocol=p.name)
    first = {p.name: p for p in protos[:2]}
    followers = {"block2": [protos[2], protos[3]], "block1": [protos[4], protos[5]]}
    w = {p.name: float(sum(len(blocks[t]) for t in p.block_types)) for p in protos[2:]}
    pi1 = {"block2": 0.5, "block1": 0.5}  # untilted first decision
    s = {p.name: st.syn_bias(p) for p in protos[2:]}
    b = {n: st.syn_bias(first[n]) for n in first}
    z1 = sum(pi1[n] * np.exp(b[n]) for n in pi1)
    mu1 = {n: pi1[n] * np.exp(b[n]) / z1 for n in pi1}
    # k == 1: log mu(a1)/pi(a1) differences between first blocks equal differences of prefix_logw
    r1 = {n: np.log(mu1[n] / pi1[n]) for n in pi1}
    lw1 = {n: st.prefix_logw([act(first[n])], 1) for n in pi1}
    assert (r1["block2"] - r1["block1"]) == pytest.approx(lw1["block2"] - lw1["block1"])
    # k == 2: enumerate the tilted second decision (softmax of log pi(p|a1) + s(p) over the followers of a1)
    r2, lw2 = {}, {}
    for n, fl in followers.items():
        tot = sum(w[p.name] for p in fl)
        pi2 = {p.name: w[p.name] / tot for p in fl}
        z2 = sum(pi2[p.name] * np.exp(s[p.name]) for p in fl)
        assert np.log(z2) == pytest.approx(b[n])  # the look-ahead IS the normaliser of the second decision
        for p in fl:
            mu2 = pi2[p.name] * np.exp(s[p.name]) / z2
            r2[(n, p.name)] = np.log(mu1[n] * mu2 / (pi1[n] * pi2[p.name]))
            lw2[(n, p.name)] = st.prefix_logw([act(first[n]), act(p)], 2)
    keys = list(r2)
    base = keys[0]
    for k_ in keys[1:]:
        assert (r2[k_] - r2[base]) == pytest.approx(lw2[k_] - lw2[base])
    # off by default / without prior / non-reaction k-th action
    assert RFGSPPState(RFGSPPConfig(enable=True), make_cfg(), BlockCritic(RFGSPPConfig(enable=True), torch.zeros(8), torch.ones(8), DEV), SimpleNamespace(protocols=protos, blocks=blocks)).prefix_logw([act(first["block2"])], 1) == 0.0
    assert st.prefix_logw([act(first["block2"]), SimpleNamespace(action=RxnActionType.Stop, protocol="stop")], 2) == 0.0
