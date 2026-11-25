"""Lane B / T3 (code level): with every switch off the lane trainer IS the baseline trainer.

The end-to-end evidence is the pair of 5-iteration cluster runs documented in docs/lane_b/REPORT.md
(baseline trainer vs. lane trainer with `lane_b.enable=false`, same seed -> identical train logs).
These tests pin the code paths that would silently break that equivalence.
"""

from unittest import mock

import torch

from cgflow.config import Config
from cgflow.lanes.config import LaneBConfig
from cgflow.lanes.lane_b import ReplayEngine, VinaMOOTrainer_laneB
from cgflow.tasks.semla_vina import VinaMOOTrainer_semla


def test_all_switches_default_to_off():
    assert LaneBConfig().enable is False
    assert Config().lane_b.enable is False


def test_subclass_does_not_re_decorate_the_mogfn_trainer():
    """`mogfn_trainer` wraps setup/train_batch/_save_state at class-creation time; applying it twice
    would run the focus-model / pareto hooks twice."""
    assert VinaMOOTrainer_laneB.__mro__[1] is VinaMOOTrainer_semla
    assert VinaMOOTrainer_laneB._save_state is VinaMOOTrainer_semla._save_state


def test_setup_registers_nothing_when_disabled():
    tr = VinaMOOTrainer_laneB.__new__(VinaMOOTrainer_laneB)
    tr.cfg = Config()
    tr.cfg.lane_b.enable = False
    tr.sampling_hooks = []
    with mock.patch.object(VinaMOOTrainer_semla, "setup", lambda self: None):
        VinaMOOTrainer_laneB.setup(tr)
    assert tr.sampling_hooks == []
    assert tr._lane_b_engine is None
    assert not hasattr(tr, "_lane_b_gen")


def test_setup_registers_the_engine_when_enabled():
    tr = VinaMOOTrainer_laneB.__new__(VinaMOOTrainer_laneB)
    tr.cfg = Config()
    tr.cfg.lane_b.enable = True
    tr.cfg.num_workers = 0
    tr.ctx = None
    tr.sampling_hooks = []
    with mock.patch.object(VinaMOOTrainer_semla, "setup", lambda self: None):
        VinaMOOTrainer_laneB.setup(tr)
    assert len(tr.sampling_hooks) == 1 and isinstance(tr.sampling_hooks[0], ReplayEngine)
    assert tr._lane_b_engine is tr.sampling_hooks[0]


def test_setup_rejects_impossible_settings():
    def build(**kw):
        tr = VinaMOOTrainer_laneB.__new__(VinaMOOTrainer_laneB)
        tr.cfg = Config()
        tr.cfg.lane_b.enable = True
        tr.cfg.num_workers = 0
        tr.ctx = None
        tr.sampling_hooks = []
        for k, v in kw.items():
            setattr(tr.cfg.lane_b, k, v)
        return tr

    for kw in [{"priority": "nope"}, {"hindsight": "nope"}, {"utd": 0}, {"hindsight_prob": 2.0}]:
        tr = build(**kw)
        with mock.patch.object(VinaMOOTrainer_semla, "setup", lambda self: None):
            try:
                VinaMOOTrainer_laneB.setup(tr)
            except ValueError:
                continue
        raise AssertionError(f"lane_b accepted an invalid configuration: {kw}")

    tr = build()
    tr.cfg.num_workers = 2  # the hook would live in another process than train_batch
    with mock.patch.object(VinaMOOTrainer_semla, "setup", lambda self: None):
        try:
            VinaMOOTrainer_laneB.setup(tr)
        except ValueError:
            return
    raise AssertionError("lane_b accepted num_workers != 0")


def test_train_batch_is_a_pure_pass_through_when_disabled():
    tr = VinaMOOTrainer_laneB.__new__(VinaMOOTrainer_laneB)
    tr.cfg = Config()
    tr.cfg.lane_b.enable = False
    tr._lane_b_engine = None
    tr.sampling_hooks, tr.to_terminate = [], []  # GFNTrainer.__del__ touches these
    base = {"loss": 1.0, "logZ": 2.0}
    torch.manual_seed(0)
    before = torch.rand(3)
    with mock.patch.object(VinaMOOTrainer_semla, "train_batch", return_value=dict(base)) as m:
        out = VinaMOOTrainer_laneB.train_batch(tr, batch=None, epoch_idx=0, batch_idx=0, train_it=7)
    torch.manual_seed(0)
    after = torch.rand(3)
    assert out == base, out
    assert m.call_count == 1
    assert torch.equal(before, after)  # the lane consumed no torch RNG
    assert not any(k.startswith("lane_b/") for k in out)


def test_no_extra_step_during_warmup():
    """With the lane on but the buffer still cold, train_batch must not touch the optimizer."""
    tr = VinaMOOTrainer_laneB.__new__(VinaMOOTrainer_laneB)
    tr.cfg = Config()
    tr.cfg.lane_b.enable = True
    tr.cfg.lane_b.warmup_iters = 20
    tr.cfg.lane_b.replay_batch = 64
    tr._lane_b_engine = ReplayEngine(tr.cfg)
    tr._lane_b_nonfinite = 0
    tr.sampling_hooks, tr.to_terminate = [], []  # GFNTrainer.__del__ touches these
    tr.algo = mock.Mock()
    tr.model = mock.Mock()
    with mock.patch.object(VinaMOOTrainer_semla, "train_batch", return_value={"loss": 1.0}):
        out = VinaMOOTrainer_laneB.train_batch(tr, batch=None, epoch_idx=0, batch_idx=0, train_it=3)
    assert out["lane_b/n_extra_steps"] == 0.0
    assert out["lane_b/buffer_size"] == 0.0
    tr.algo.compute_batch_losses.assert_not_called()
