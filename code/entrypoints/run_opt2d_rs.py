#!/usr/bin/env python3
"""2D (RxnFlow) counterpart of run_opt_rs.py: pocket-specific Vina-QED optimisation WITHOUT a pose model (RxnFlow's Uni-Dock MOO trainer, oracle of record = Uni-Dock 'balance',
conformer/docking seed 1, 20 A box), same CLI, same cull-safe checkpoint/resume (rs_support), same systems-only patches (CPU cap, parallel ETKDG) as the 3D runs.
Trainers: cgflow.lanes2d.trainers2d:UniDockMOOTrainer_oracle (baseline RxnFlow) | :UniDockMOOTrainer_RFGSPP (RFG/SPP: --set rfg_spp.enable=true ...)   [--ckpt / --redocking / --num_inference_steps are accepted and ignored]"""
import argparse
import importlib
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_opt as base  # noqa: E402
import rs_support as rs  # noqa: E402


def patch_inline_retro():
    """Systems-only patch. RxnFlow's MultiRetroSyntheticAnalyzer forks a ProcessPoolExecutor AFTER the 17 GB building-block environment is loaded and every worker
    deep-copies the analyzer (measured +1.6 GB PSS per worker, 4 workers => 27 GB for the whole process tree, above the 26 GB task limit). The analysis (analyzer.run) is a pure
    function of (mol, max_step, known_branches) apart from its internal cache, so it is run in-process instead: same trees, no worker memory (login-node probe in LEAD_LOG)."""
    import rxnflow.envs.retrosynthesis as R

    class InlineRetroAnalyzer:
        def __init__(self, analyzer, num_workers: int = 4):
            self.analyzer = analyzer
            self.futures = []

        def init(self):
            self.result()

        def result(self):
            out, self.futures = self.futures, []
            return out

        def submit(self, key, mol, max_rxns, known_branches):
            self.futures.append((key, self.analyzer.run(mol, max_rxns, known_branches)))

    R.MultiRetroSyntheticAnalyzer = InlineRetroAnalyzer
    import rxnflow.algo.synthetic_path_sampling as S

    S.MultiRetroSyntheticAnalyzer = InlineRetroAnalyzer
    print("[run_opt2d_rs] retrosynthesis analysis runs in-process (RETRO_INLINE=1)", flush=True)


def main():
    ncpu = base.limit_cpu_count()
    print(f"[run_opt2d_rs] os.cpu_count() capped to {ncpu}", flush=True)
    if os.environ.get("FAST_ETKDG", "1") == "1":
        base.start_etkdg_pool(max(ncpu - 1, 1))
        print(f"[run_opt2d_rs] ETKDG worker pool started ({max(ncpu - 1, 1)} workers)", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--env_dir", required=True)
    ap.add_argument("--pocket_dir", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--redocking", type=int, default=1)
    ap.add_argument("--num_inference_steps", type=int, default=80)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--desc", default="Vina-QED optimisation, RxnFlow 2D")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--trainer", default="cgflow.lanes2d.trainers2d:UniDockMOOTrainer_oracle")
    ap.add_argument("--ckpt_every", type=int, default=int(os.environ.get("RS_EVERY", "20")))
    a = ap.parse_args()

    if os.environ.get("FAST_ETKDG", "1") == "1":
        base.patch_fast_etkdg()  # must precede the import of rxnflow.tasks.unidock (it binds `docking` by name)
    if os.environ.get("RETRO_INLINE", "1") == "1":
        patch_inline_retro()
    from cgflow.lanes2d.config2d import Config2D
    from rxnflow.config import init_empty

    mod_name, cls_name = base.trainer_path(a.trainer)
    TrainerCls = getattr(importlib.import_module(mod_name), cls_name)
    TrainerCls = rs.make_resumable(TrainerCls, a.ckpt_every)

    pocket_dir = Path(a.pocket_dir)
    config = init_empty(Config2D())
    config.desc = a.desc
    config.env_dir = a.env_dir
    config.num_training_steps = a.steps
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.print_every = 1
    config.seed = a.seed
    config.task.docking.protein_path = str(pocket_dir / a.target / "protein.pdb")
    config.task.docking.ref_ligand_path = str(pocket_dir / a.target / "ligand.mol2")
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
        print(f"[run_opt2d_rs] terminate() raised {e!r}", flush=True)
    try:
        trainer._summary_writer.flush()
    except Exception:  # noqa: BLE001
        pass
    if base._ETKDG_POOL is not None:
        base._ETKDG_POOL.terminate()
    if os.environ.get("RS_KEEP") != "1":
        for f in ("rs_state.pt", "rs_state.prev.pt", "rs_state.pt.tmp"):
            try:
                os.remove(os.path.join(a.log_dir, f))
            except OSError:
                pass
    print("[run_opt2d_rs] finished cleanly", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
