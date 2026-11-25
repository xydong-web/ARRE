#!/usr/bin/env python3
"""One LIT-PCBA pocket-specific optimisation run (CGFlow / 3DSynthFlow), argparse front-end to the repo's own
experiments/scripts/exp1_litpcba_moo.py recipe (wandb sweep replaced by CLI because compute nodes have no internet).

Settings mirrored from the tag's experiments/scripts/exp1_litpcba_moo.py + sweep/redock.yaml:
  num_training_steps=1000, num_validation_gen_steps=0, num_final_gen_steps=0, print_every=1,
  redocking=True (UniDock 'balance'), semlaflow.num_inference_steps=80, Vina+QED MOO.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf


from component_names import COMPONENT_NAMES, trainer_path


def apply_overrides(cfg, sets):
    """Apply `a.b.c=value` overrides (value parsed as YAML) on the dataclass config. Used for lane switches and for
    infrastructure-only knobs (checkpoint_every, ...). Unknown paths raise, so a typo can never silently run baseline."""
    for kv in sets or []:
        key, val = kv.split("=", 1)
        parts = key.split(".")
        parts[0] = COMPONENT_NAMES.get(parts[0], parts[0])
        *path, last = parts
        obj = cfg
        for p in path:
            obj = getattr(obj, p)
        if not hasattr(obj, last):
            raise AttributeError(f"unknown config key: {key}")
        setattr(obj, last, yaml.safe_load(val))


def limit_cpu_count() -> int:
    """Systems-only fix (no algorithmic effect).

    unidock_tools builds `multiprocess.Pool(os.cpu_count())` twice per docking call (topology build, post-processing).
    On the compute nodes os.cpu_count() is the machine total (104) rather than the Slurm allocation (8), and every
    os.fork() of this ~20 GB-RSS parent costs ~0.2 s, i.e. ~45 s per iteration spent only on forking idle workers
    (measured with cProfile: 1664 forks = 356 s over 8 iterations). Cap the reported CPU count to the allocation.
    """
    n = None
    for var in ("SLURM_CPUS_PER_TASK",):
        v = os.environ.get(var, "")
        if v.isdigit():
            n = int(v)
    try:
        aff = len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        aff = os.cpu_count() or 1
    n = min(n, aff) if n else aff
    os.cpu_count = lambda: n
    return n


# ---- systems-only speed-up: parallel ETKDG conformer generation inside the Uni-Dock wrapper ------------------------------------------------
# cgflow.utils.unidock.docking embeds the 64 molecules of every oracle call one after another (~9 s / iteration of a ~65 s iteration).
# The embedding of each molecule is deterministic (srETKDGv3 with a fixed seed), so running it in a persistent worker pool yields identical
# conformers; only wall-clock changes.  The pool is forked at start-up, while the process is still small (forking a 20 GB-RSS parent costs 0.2 s per worker).
_ETKDG_POOL = None


def _etkdg_worker(args):
    mol_pkl, sdf_path = args
    from rdkit import Chem
    from cgflow.utils.unidock import run_etkdg

    return run_etkdg(Chem.Mol(mol_pkl), sdf_path)


def start_etkdg_pool(n: int):
    global _ETKDG_POOL
    import multiprocessing as mp

    if n > 1 and _ETKDG_POOL is None:
        _ETKDG_POOL = mp.get_context("fork").Pool(n)
    return _ETKDG_POOL


def patch_fast_etkdg():
    """Replace cgflow.utils.unidock.docking by a copy whose ETKDG loop runs in the worker pool (everything else identical)."""
    import tempfile
    from pathlib import Path as _P

    from rdkit import Chem

    import cgflow.utils.unidock as ud

    if _ETKDG_POOL is None:
        return

    def docking(rdmols, protein_path, center, seed=1, size=20.0, search_mode="balance"):
        protein_path = _P(protein_path)
        protein_pdbqt_path = protein_path.parent / (protein_path.name + "qt")
        if not protein_pdbqt_path.exists():
            ud.pdb2pdbqt(protein_path, protein_pdbqt_path)
        with tempfile.TemporaryDirectory() as out_dir:
            out_dir = _P(out_dir)
            files = [out_dir / f"{i}.sdf" for i in range(len(rdmols))]
            flags = _ETKDG_POOL.map(_etkdg_worker, [(m.ToBinary(Chem.PropertyPickleOptions.AllProps), str(f)) for m, f in zip(rdmols, files)], chunksize=2)
            sdf_list = [f for f, ok in zip(files, flags) if ok]
            if len(sdf_list) > 0:
                runner = ud.UniDock(protein_pdbqt_path, sdf_list, center[0], center[1], center[2], size, size, size, out_dir / "workdir")
                runner.docking(out_dir / "savedir", search_mode=search_mode, num_modes=1, seed=seed)
            res = []
            for i in range(len(rdmols)):
                try:
                    docked_rdmol = list(Chem.SDMolSupplier(str(out_dir / "savedir" / f"{i}.sdf")))[0]
                    assert docked_rdmol is not None
                    docking_score = float(docked_rdmol.GetProp("docking_score"))
                except Exception:
                    docked_rdmol, docking_score = None, 0.0
                res.append((docked_rdmol, docking_score))
            return res

    ud.docking = docking


def main():
    ncpu = limit_cpu_count()
    print(f"[run_opt] os.cpu_count() capped to {ncpu}", flush=True)
    if os.environ.get("FAST_ETKDG", "1") == "1":
        start_etkdg_pool(max(ncpu - 1, 1))
        print(f"[run_opt] ETKDG worker pool started ({max(ncpu - 1, 1)} workers)", flush=True)
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
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="config override a.b.c=value (repeatable); lane switches and infrastructure knobs")
    ap.add_argument("--trainer", default="cgflow.tasks.semla_vina:VinaMOOTrainer_semla",
                    help="module:Class of the trainer; innovation lanes ship their own subclass (default = paper baseline)")
    a = ap.parse_args()

    from cgflow.config import Config, init_empty
    if os.environ.get("FAST_ETKDG", "1") == "1":
        patch_fast_etkdg()
    mod_name, cls_name = trainer_path(a.trainer)
    TrainerCls = getattr(importlib.import_module(mod_name), cls_name)

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
    apply_overrides(config, a.sets)

    trainer = TrainerCls(config)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(trainer.cfg)), Path(a.log_dir).parent / f"config.seed{a.seed}.yaml")
    trainer.run()
    # Graceful, deterministic shutdown.  The Pareto-accumulation daemon thread of gflownet.utils.multiobjective_hooks races with interpreter
    # finalisation and sporadically aborts the process ("Fatal Python error: _enter_buffered_busy ... daemon threads", core dump, non-zero exit),
    # which would skip the evaluation steps that follow.  Everything is written by now (SQLite rows are committed per batch, model_state.pt is closed).
    try:
        trainer.terminate()  # stops + joins the Pareto thread (and any other hook / wrapper)
    except Exception as e:  # noqa: BLE001
        print(f"[run_opt] terminate() raised {e!r}", flush=True)
    try:
        trainer._summary_writer.flush()
    except Exception:  # noqa: BLE001
        pass
    if _ETKDG_POOL is not None:
        _ETKDG_POOL.terminate()
    print("[run_opt] finished cleanly", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
