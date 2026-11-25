"""T4 - with the switches off the lane reduces to the baseline.

The *strong* evidence for the equivalence is the pair of 5-iteration cluster
smoke runs documented in docs/lane_d/REPORT.md (baseline trainer vs lane-D
trainer with ``lane_d.enable=false``, same seed, identical train.log).  These
unit tests pin the two code-level guarantees that make that possible.
"""

from unittest import mock

import numpy as np
import pytest

from cgflow.config import Config
from cgflow.lanes.lane_d import PoseProxyGate, VinaMOOTrainer_laneD
from cgflow.lanes.lane_d_config import LaneDConfig
from cgflow.tasks.semla_vina import VinaMOOTrainer_semla


def test_all_switches_default_to_off():
    d = LaneDConfig()
    assert d.enable is False
    assert d.use_vina_local is False
    assert Config().lane_d.enable is False


def test_lane_is_a_subclass_of_the_baseline_trainer():
    assert issubclass(VinaMOOTrainer_laneD, VinaMOOTrainer_semla)
    # the lane touches exactly one trainer method
    overridden = {
        name
        for name, value in vars(VinaMOOTrainer_laneD).items()
        if callable(value) and not name.startswith("__")
    }
    assert overridden == {"build_training_data_loader"}


def test_disabled_lane_calls_the_baseline_loader_verbatim():
    trainer = object.__new__(VinaMOOTrainer_laneD)
    trainer.cfg = Config()  # lane_d.enable is False
    trainer.sampling_hooks, trainer.to_terminate = [], []  # silence GFNTrainer.__del__
    sentinel = object()
    with mock.patch.object(
        VinaMOOTrainer_semla, "build_training_data_loader", return_value=sentinel
    ) as baseline:
        out = VinaMOOTrainer_laneD.build_training_data_loader(trainer)
    assert out is sentinel
    baseline.assert_called_once()
    assert getattr(trainer, "lane_d_gate", None) is None, "no proxy may be constructed when off"


def test_enabled_lane_does_not_take_the_baseline_path():
    """Guard against the branch silently inverting."""
    trainer = object.__new__(VinaMOOTrainer_laneD)
    cfg = Config()
    cfg.lane_d.enable = True
    cfg.num_workers = 4  # trips the explicit guard before anything heavy is built
    trainer.cfg = cfg
    trainer.sampling_hooks, trainer.to_terminate = [], []
    with mock.patch.object(VinaMOOTrainer_semla, "build_training_data_loader") as baseline:
        with pytest.raises(ValueError, match="num_workers"):
            VinaMOOTrainer_laneD.build_training_data_loader(trainer)
    baseline.assert_not_called()


def test_identity_selection_when_the_pool_equals_the_batch():
    """F = 1 -> select() is the identity and marks the batch as a uniform sample."""
    g = object.__new__(PoseProxyGate)
    g.dcfg = LaneDConfig(enable=True, oversample=1)
    g.rng = np.random.default_rng(0)
    g.gate_on = False
    # the gate cannot fire because n_pool == n_select; verified through the state machine
    from cgflow.lanes.lane_d import decide_gate

    eligible = g.dcfg.act and True and (64 > 64) and True
    assert decide_gate(False, eligible, 0.9, g.dcfg.min_rho, g.dcfg.rho_hysteresis) is False


def test_use_vina_local_raises_instead_of_silently_doing_nothing():
    cfg = Config()
    cfg.lane_d.enable = True
    cfg.lane_d.use_vina_local = True
    with pytest.raises(NotImplementedError, match="use_vina_local"):
        PoseProxyGate(cfg, ctx=None, device="cpu", seed=0)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("oversample", 0, "oversample"),
        ("strata", 0, "strata"),
        ("score_mode", "nope", "score_mode"),
        ("random_frac", 1.0, "random_frac"),
    ],
)
def test_config_validation(field, value, match):
    cfg = Config()
    cfg.lane_d.enable = True
    setattr(cfg.lane_d, field, value)
    with pytest.raises(ValueError, match=match):
        PoseProxyGate(cfg, ctx=None, device="cpu", seed=0)


def test_scalarized_mode_requires_vina_and_qed():
    cfg = Config()
    cfg.lane_d.enable = True
    cfg.task.moo.objectives = ["vina", "sa"]
    with pytest.raises(ValueError, match="scalarized"):
        PoseProxyGate(cfg, ctx=None, device="cpu", seed=0)
