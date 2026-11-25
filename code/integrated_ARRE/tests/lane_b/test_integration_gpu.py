"""Lane B / T4 + T3(runtime): integration test on a GPU node (needs the 20k-block env and a pose checkpoint).

Run through `scripts_lane/test_lane_b_gpu.sbatch`; skipped unless CGFLOW_LANE_B_GPU_TEST=1.

What it proves
--------------
* a trajectory stored in the replay buffer still produces EXACTLY the tensors it produced at sampling time,
  after further sampling iterations / pose-model calls / optimizer steps have happened
  (fingerprint over every `graph_to_Data` tensor of every state + the encoded actions),
* the graph tensors a replayed trajectory contributes to a training batch are identical, element by element,
  to the ones the same trajectory contributed to the fresh (sampling-time) batch,
* the hindsight relabeling does not mutate the stored trajectories,
* the extra (replay) optimizer steps run, produce finite losses and move the parameters.
"""

import os
import shutil
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("CGFLOW_LANE_B_GPU_TEST") != "1",
    reason="needs a GPU node, the block env and a SemlaFlow checkpoint (set CGFLOW_LANE_B_GPU_TEST=1)",
)

ROOT = Path(os.environ.get("CGFLOW_ROOT", "/path/to/ARRE"))
ENV_DIR = os.environ.get("CGFLOW_TEST_ENV_DIR", str(ROOT / "data/smoke/env20k"))
CKPT = os.environ.get("CGFLOW_TEST_CKPT", str(ROOT / "weights/smoke_stateflow.ckpt"))
TARGET = os.environ.get("CGFLOW_TEST_TARGET", "ADRB2")
NUM_FROM_POLICY = int(os.environ.get("CGFLOW_TEST_NFP", "16"))


def _limit_cpu_count() -> None:
    """Same systems-only fix as run/run_opt.py: unidock_tools forks os.cpu_count() workers twice per
    docking call, which on these nodes is the machine total (104) instead of the Slurm allocation."""
    v = os.environ.get("SLURM_CPUS_PER_TASK", "")
    n = int(v) if v.isdigit() else (os.cpu_count() or 1)
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        pass
    os.cpu_count = lambda: n


_limit_cpu_count()


def _build_trainer(tmp_path: Path):
    from cgflow.config import Config, init_empty
    from cgflow.lanes.lane_b import VinaMOOTrainer_laneB

    pocket = tmp_path / "pocket" / TARGET
    pocket.mkdir(parents=True)
    for f in ("protein.pdb", "ligand.mol2"):  # private copy: UniDock writes a pdbqt next to the protein
        shutil.copy(ROOT / "data/LIT-PCBA" / TARGET / f, pocket / f)

    cfg = init_empty(Config())
    cfg.desc = "lane_b integration test"
    cfg.env_dir = ENV_DIR
    cfg.num_training_steps = 2
    cfg.num_validation_gen_steps = 0
    cfg.num_final_gen_steps = 0
    cfg.print_every = 1
    cfg.seed = 0
    cfg.task.docking.protein_path = str(pocket / "protein.pdb")
    cfg.task.docking.ref_ligand_path = str(pocket / "ligand.mol2")
    cfg.task.docking.redocking = True
    cfg.semlaflow.ckpt_path = CKPT
    cfg.semlaflow.num_inference_steps = 20
    cfg.log_dir = str(tmp_path / "run")
    cfg.algo.num_from_policy = NUM_FROM_POLICY
    cfg.lane_b.enable = True
    cfg.lane_b.utd = 2
    cfg.lane_b.warmup_iters = 0
    cfg.lane_b.replay_batch = 8
    cfg.lane_b.capacity = 64
    cfg.lane_b.protected_elites = 2
    cfg.lane_b.hindsight = "posterior_cal"
    cfg.lane_b.hindsight_candidates = 8
    cfg.lane_b.cal_min_samples = 8  # the toy run only stores a handful of trajectories per iteration
    cfg.lane_b.selfcheck_replay_data = True
    trainer = VinaMOOTrainer_laneB(cfg)
    # `GFNTrainer.run()` does this before touching the data loader; we drive the loop by hand here
    trainer.model.to(trainer.device)
    trainer.sampling_model.to(trainer.device)
    return trainer


def _graph_slices(batch):
    """(start, stop) row index of every graph of a collated batch."""
    if hasattr(batch, "x_ptr"):
        ptr = batch.x_ptr.tolist()
    else:  # pragma: no cover
        counts = torch.bincount(batch.x_batch).tolist()
        ptr = [0]
        for c in counts:
            ptr.append(ptr[-1] + c)
    return list(zip(ptr[:-1], ptr[1:]))


def test_replayed_trajectories_reproduce_their_sampling_time_features(tmp_path):
    trainer = _build_trainer(tmp_path)
    engine = trainer._lane_b_engine
    assert engine is not None

    dl = trainer.build_training_data_loader()
    data_iter = iter(dl)

    fresh = next(data_iter)  # iteration 1: sampling + pose model + docking, the hook fills the buffer
    assert len(engine) > 0, "nothing was stored in the replay buffer"
    stored = list(engine.items)
    slots = [r["slot"] for r in stored]

    # --- (a) the rows a stored trajectory contributes to a training batch are the sampling-time rows ----
    replay = trainer._lane_b_build_batch(
        [r["traj"] for r in stored],
        {k: torch.stack([r["traj"]["cond_info"][k] for r in stored]) for k in stored[0]["traj"]["cond_info"]},
        torch.stack([r["traj"]["log_reward"] for r in stored]),
    )
    f_sl, r_sl = _graph_slices(fresh), _graph_slices(replay)
    f_off = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(fresh.traj_lens, 0)])
    r_off = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(replay.traj_lens, 0)])
    assert replay.traj_lens.tolist() == [int(fresh.traj_lens[s]) for s in slots]
    for j, s in enumerate(slots):
        for t in range(int(replay.traj_lens[j])):
            gf, gr = int(f_off[s]) + t, int(r_off[j]) + t
            a, b = f_sl[gf], r_sl[gr]
            assert torch.equal(fresh.x[a[0] : a[1]], replay.x[b[0] : b[1]])
            assert torch.equal(fresh.pos[a[0] : a[1]], replay.pos[b[0] : b[1]])
            for key in ("graph_attr", "protocol_mask", "level"):
                assert torch.equal(getattr(fresh, key)[gf], getattr(replay, key)[gr]), key
            assert torch.equal(fresh.actions[gf], replay.actions[gr])
        assert torch.equal(fresh.log_rewards[s], replay.log_rewards[j])

    # --- (b) nothing mutates the stored states later on -------------------------------------------------
    info1 = trainer.train_batch(fresh.to(trainer.device), 0, 0, 1)
    assert info1["lane_b/selfcheck_mismatch"] == 0.0
    assert info1["lane_b/selfcheck_checked"] > 0.0
    next(data_iter)  # a second sampling iteration: the pose model and its caches run again
    bad, checked = engine.verify_stored_features()
    assert bad == 0 and checked > 0, f"{bad}/{checked} stored trajectories changed their features"


def test_extra_steps_run_and_do_not_corrupt_the_buffer(tmp_path):
    trainer = _build_trainer(tmp_path)
    engine = trainer._lane_b_engine
    dl = trainer.build_training_data_loader()
    data_iter = iter(dl)
    fresh = next(data_iter)

    rec = engine.items[0]
    enc_before = rec["traj"]["cond_info"]["encoding"].clone()
    beta_before = rec["traj"]["cond_info"]["beta"].clone()
    lr_before = rec["traj"]["log_reward"].clone()
    p_before = [p.detach().clone() for p in trainer.model.parameters()]

    info = trainer.train_batch(fresh.to(trainer.device), 0, 0, 1)

    assert info["lane_b/n_extra_steps"] == 1.0, info
    assert "lane_b/loss_extra_1" in info and torch.isfinite(torch.tensor(info["lane_b/loss_extra_1"]))
    assert info["lane_b/nonfinite_extra"] == 0.0
    assert 0.0 < info["lane_b/hs_ess"] <= 1.0
    assert info["lane_b/cal_used"] == 1.0 and info["lane_b/cal_n"] >= 8.0
    assert 0.0 <= info["lane_b/cal_head_fallback_frac"] <= 1.0
    assert info["lane_b/buffer_size"] == float(len(engine))
    # the stored trajectory keeps its sampling-time condition and reward
    assert torch.equal(rec["traj"]["cond_info"]["encoding"], enc_before)
    assert torch.equal(rec["traj"]["cond_info"]["beta"], beta_before)
    assert torch.equal(rec["traj"]["log_reward"], lr_before)
    # and the extra step actually moved the parameters
    assert any(not torch.equal(a, b.detach()) for a, b in zip(p_before, trainer.model.parameters()))
