#!/usr/bin/env python3
"""run_opt.py + cull-safe checkpoint/resume (see rs_support.py).  Same CLI as run_opt.py plus --ckpt_every (default 20, env RS_EVERY).
If <log_dir>/rs_state.pt exists the run is resumed from it, otherwise a fresh run starts (the log dir must then be absent, as before).
Everything else (config, systems-only patches, shutdown) is run_opt.py verbatim, imported from the same directory."""
import argparse
import importlib
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

if any(a.startswith("cgflow.lanes2d") for a in sys.argv):  # round 4 (WP3): the 2D RxnFlow trainers run through the same launchers / watchdog / resume machinery
    import runpy

    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_opt2d_rs.py"), run_name="__main__")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_opt as base  # noqa: E402
import rs_support as rs  # noqa: E402


def main():
    ncpu = base.limit_cpu_count()
    print(f"[run_opt_rs] os.cpu_count() capped to {ncpu}", flush=True)
    if os.environ.get("FAST_ETKDG", "1") == "1":
        base.start_etkdg_pool(max(ncpu - 1, 1))
        print(f"[run_opt_rs] ETKDG worker pool started ({max(ncpu - 1, 1)} workers)", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--env_dir", required=True)
    ap.add_argument("--pocket_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--redocking", type=int, default=1)
    ap.add_argument("--num_inference_steps", type=int, default=80)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--desc", default="Vina-QED optimization using 3D information")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--trainer", default="cgflow.tasks.semla_vina:VinaMOOTrainer_semla")
    ap.add_argument("--ckpt_every", type=int, default=int(os.environ.get("RS_EVERY", "20")))
    a = ap.parse_args()

    from cgflow.config import Config, init_empty

    if os.environ.get("FAST_ETKDG", "1") == "1":
        base.patch_fast_etkdg()
    mod_name, cls_name = base.trainer_path(a.trainer)
    TrainerCls = getattr(importlib.import_module(mod_name), cls_name)
    TrainerCls = rs.make_resumable(TrainerCls, a.ckpt_every)

    pocket_dir = Path(a.pocket_dir)
    config = init_empty(Config())
    config.desc = a.desc
    config.env_dir = a.env_dir
    config.num_training_steps = a.steps
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.print_every = 1
    config.seed = a.seed
    config.task.docking.protein_path = str(pocket_dir / a.target / "protein.pdb")
    config.task.docking.ref_ligand_path = str(pocket_dir / a.target / "ligand.mol2")
    config.task.docking.redocking = bool(a.redocking)
    config.semlaflow.ckpt_path = a.ckpt
    config.semlaflow.num_inference_steps = a.num_inference_steps
    config.log_dir = a.log_dir
    base.apply_overrides(config, a.sets)

    st = rs.begin_resume(a.log_dir)  # None -> fresh start (log_dir must not exist)
    trainer = TrainerCls(config)
    if st is not None:
        rs.finish_resume(trainer, st)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(trainer.cfg)), Path(a.log_dir).parent / f"config.seed{a.seed}.yaml")
    trainer.run()
    try:
        trainer.terminate()
    except Exception as e:  # noqa: BLE001
        print(f"[run_opt_rs] terminate() raised {e!r}", flush=True)
    try:
        trainer._summary_writer.flush()
    except Exception:  # noqa: BLE001
        pass
    if base._ETKDG_POOL is not None:
        base._ETKDG_POOL.terminate()
    if os.environ.get("RS_KEEP") != "1":  # checkpoints are only useful for an unfinished run
        for f in ("rs_state.pt", "rs_state.prev.pt", "rs_state.pt.tmp"):
            try:
                os.remove(os.path.join(a.log_dir, f))
            except OSError:
                pass
    print("[run_opt_rs] finished cleanly", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
