"""Cull/crash-safe training state for the CGFlow trainers (infrastructure only - no algorithmic change).

Why: the shared cluster's root user has cancelled our long runs (sacct "CANCELLED by 0") several times; a 1000-iteration run
takes 13-19 h and the baseline trainer keeps no resumable state (model_state.pt has no optimiser/RNG and is only written at the end).

What: every `every` iterations the training process writes <log_dir>/rs_state.pt (atomic rename; previous copy kept as
rs_state.prev.pt) containing
  GFN model + EMA sampling model + both Adam optimisers + both LR schedules + all RNG streams (torch cpu/cuda, numpy global,
  python, gflownet worker RNG) + the reward cache task.topn_vina + oracle call counter + number of SQLite rows committed so far
  + optional lane state (trainer.rs_extra_state() / trainer.rs_load_extra_state()).
A restarted process (same command line) finds the file, builds the trainer through its normal constructor on a fresh directory,
swaps the old run directory back in (SQLite table truncated to the checkpointed row count, so no molecule is counted twice),
restores everything and continues at iteration it+1 (DataSource.current_iter included).

Equivalence test: tests/rs (batch it+1 of a resumed process == batch it+1 of the uninterrupted process, SMILES and cond-info).
"""
from __future__ import annotations

import os
import random
import shutil
import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from component_names import PREVIOUS_RHC, checkpoint_extra

STATE = "rs_state.pt"
PREV = "rs_state.prev.pt"
TMP = "rs_state.pt.tmp"
VERSION = 1


def _rows(log_dir) -> int:
    p = Path(log_dir) / "train" / "generated_objs_0.db"
    if not p.exists():
        return 0
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=120)
    try:
        return int(con.execute("select count(*) from results").fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        con.close()


def rng_snapshot() -> dict:
    from gflownet.utils import misc

    return {
        "worker": {int(k): v.get_state() for k, v in misc._worker_rngs.items()},
        "np": np.random.get_state(),
        "py": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def rng_apply(snap: dict) -> None:
    from gflownet.utils import misc

    for wid, s in snap["worker"].items():
        r = misc._worker_rngs.get(wid)
        if r is None:
            r = np.random.RandomState()
            misc._worker_rngs[wid] = r
        r.set_state(s)
    np.random.set_state(snap["np"])
    random.setstate(snap["py"])
    torch.set_rng_state(snap["torch"].cpu())
    if snap["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in snap["cuda"]])


def collect(tr, it: int) -> dict:

    task = tr.task
    st = {
        "version": VERSION,
        "it": int(it),
        "wall": time.time(),
        "model": tr.model.state_dict(),
        "sampling_model": None if tr.sampling_model is tr.model else tr.sampling_model.state_dict(),
        "opt": tr.opt.state_dict(),
        "opt_Z": tr.opt_Z.state_dict(),
        "lr_sched": tr.lr_sched.state_dict(),
        "lr_sched_Z": tr.lr_sched_Z.state_dict(),
        "rng": rng_snapshot(),
        "topn_vina": list(task.topn_vina.items()) if hasattr(task, "topn_vina") else None,
        "oracle_idx": getattr(task, "oracle_idx", None),
        "n_rows": _rows(tr.cfg.log_dir),
        "extra": tr.rs_extra_state() if callable(getattr(tr, "rs_extra_state", None)) else None,
    }
    return st


def save(tr, it: int) -> float:
    t0 = time.time()
    d = Path(tr.cfg.log_dir)
    st = collect(tr, it)
    tmp = d / TMP
    torch.save(st, tmp)
    cur = d / STATE
    if cur.exists():
        os.replace(cur, d / PREV)
    os.replace(tmp, cur)
    return time.time() - t0


def _try_load(p: Path):
    try:
        st = torch.load(p, map_location="cpu", weights_only=False)
        assert st.get("version") == VERSION and "it" in st and "model" in st and "n_rows" in st
        return st
    except Exception as e:  # noqa: BLE001 - partially written / corrupt file
        print(f"[rs] cannot use {p.name}: {e!r}", flush=True)
        return None


def load_best(log_dir) -> dict | None:
    """Newest valid state among rs_state.pt, rs_state.pt.tmp (complete but not yet renamed) and rs_state.prev.pt."""
    d = Path(log_dir)
    best = None
    for name in (STATE, TMP, PREV):
        p = d / name
        if p.exists():
            st = _try_load(p)
            if st is not None and (best is None or st["it"] > best["it"]):
                best = st
    return best


def _aside(log_dir) -> Path:
    return Path(str(log_dir) + ".rs_old")


def begin_resume(log_dir):
    """Call BEFORE constructing the trainer.  Returns the state to restore (and moves the old run dir aside so that the
    trainer's constructor, which insists on creating the directory itself, works), or None for a fresh start."""
    log_dir = Path(log_dir)
    aside = _aside(log_dir)
    if aside.exists():  # a previous resume was interrupted between the two renames -> put the old dir back first
        if log_dir.exists():
            if (log_dir / "train").exists() or (log_dir / STATE).exists():
                raise RuntimeError(f"both {log_dir} and {aside} look like real run dirs - resolve manually")
            shutil.rmtree(log_dir)
        os.rename(aside, log_dir)
    if not log_dir.exists():
        return None
    st = load_best(log_dir)
    if st is None:
        return None
    os.rename(log_dir, aside)
    return st


def finish_resume(tr, st: dict) -> None:
    """Call AFTER the trainer has been constructed (it created a fresh <log_dir> containing only config.yaml/pose)."""
    log_dir = Path(tr.cfg.log_dir)
    aside = _aside(log_dir)
    assert aside.exists(), aside
    shutil.rmtree(log_dir)
    os.rename(aside, log_dir)
    if (log_dir / PREVIOUS_RHC).is_dir() and not (log_dir / "rhc").exists():
        os.rename(log_dir / PREVIOUS_RHC, log_dir / "rhc")

    # SQLite: drop everything written after the checkpoint
    db = log_dir / "train" / "generated_objs_0.db"
    if db.exists():
        con = sqlite3.connect(db, timeout=300)
        n0 = con.execute("select count(*) from results").fetchone()[0]
        if n0 < st["n_rows"]:
            raise RuntimeError(f"db has {n0} rows but checkpoint expects {st['n_rows']}")
        con.execute("delete from results where rowid > ?", (st["n_rows"],))
        con.commit()
        n1 = con.execute("select count(*) from results").fetchone()[0]
        con.close()
        print(f"[rs] db rows {n0} -> {n1} (checkpoint it={st['it']})", flush=True)
    elif st["n_rows"] > 0:
        raise RuntimeError("checkpoint expects db rows but train/generated_objs_0.db is missing")

    dev = tr.device
    tr.model.to(dev)
    if tr.sampling_model is not tr.model:
        tr.sampling_model.to(dev)
    tr.model.load_state_dict(st["model"])
    if st["sampling_model"] is not None:
        tr.sampling_model.load_state_dict(st["sampling_model"])
    tr.opt.load_state_dict(st["opt"])
    tr.opt_Z.load_state_dict(st["opt_Z"])
    tr.lr_sched.load_state_dict(st["lr_sched"])
    tr.lr_sched_Z.load_state_dict(st["lr_sched_Z"])

    # NOTE: building the DataLoader iterator draws a base seed from the global torch RNG, so the RNG state must be (re-)applied once more
    # right after that (see make_resumable.build_training_data_loader); applying it here as well keeps everything consistent until then.
    rng_apply(st["rng"])
    tr._rs_rng = st["rng"]

    task = tr.task
    if st["topn_vina"] is not None and hasattr(task, "topn_vina"):
        task.topn_vina = OrderedDict(st["topn_vina"])
    if st["oracle_idx"] is not None and hasattr(task, "oracle_idx"):
        task.oracle_idx = st["oracle_idx"]
    if st.get("extra") is not None:
        if not hasattr(tr, "rs_load_extra_state"):
            raise RuntimeError("checkpoint carries lane state but the trainer has no rs_load_extra_state()")
        tr.rs_load_extra_state(checkpoint_extra(st["extra"]))

    tr._rs_start = int(st["it"])
    tr.cfg.start_at_step = int(st["it"])
    print(f"[rs] resumed: continue at iteration {st['it'] + 1}; ckpt written {time.time() - st['wall']:.0f}s ago", flush=True)


def make_resumable(cls, every: int = 20):
    """Subclass `cls` (any trainer of this code base) with periodic checkpointing + DataSource step restore.

    Innovation-lane trainers (module cgflow.lanes.*) keep extra training state (replay buffers, running statistics, critics, ...).  They must
    provide rs_extra_state()/rs_load_extra_state(); otherwise checkpointing is disabled for them, because a resume that silently dropped
    that state would no longer be the algorithm under test (a cull then restarts the run from scratch, as before)."""
    if cls.__module__.startswith("cgflow.lanes") and getattr(cls, "rs_extra_state", None) is None and every > 0:  # a class may opt out with rs_extra_state = None
        print(f"[rs] {cls.__module__}.{cls.__name__} has no rs_extra_state(): checkpointing DISABLED (a cull restarts the run from scratch)", flush=True)
        every = 0

    class _Resumable(cls):  # type: ignore[misc, valid-type]
        _rs_every = int(every)
        _rs_start = 0
        _rs_rng = None

        def build_training_data_loader(self):
            dl = super().build_training_data_loader()
            if self._rs_start:
                src = dl.dataset  # DataSource (num_workers=0: iterated in this process)
                src.global_step_count += self._rs_start
                src.current_iter = self._rs_start
            if self._rs_rng is not None:
                snap, base_cls = self._rs_rng, type(dl)

                class _RngRestoringLoader(base_cls):  # type: ignore[misc, valid-type]
                    def __iter__(self_inner):
                        it = super().__iter__()  # draws a base seed from the global torch RNG ...
                        rng_apply(snap)  # ... which the uninterrupted run never did at this point -> undo
                        return it

                dl.__class__ = _RngRestoringLoader
            return dl

        def train_batch(self, batch, epoch_idx, batch_idx, train_it):
            out = super().train_batch(batch, epoch_idx, batch_idx, train_it)
            try:
                (Path(self.cfg.log_dir) / "rs_alive").write_text(f"{time.time():.0f} {train_it} {os.environ.get('SLURM_JOB_ID', '')}\n")
            except OSError:
                pass
            if self._rs_every > 0 and train_it % self._rs_every == 0 and train_it < self.cfg.num_training_steps:
                try:  # a failing checkpoint (disk quota, NFS hiccup, ...) must never kill the training run
                    dt = save(self, train_it)
                    print(f"[rs] checkpoint it={train_it} saved in {dt:.1f}s ({(Path(self.cfg.log_dir) / STATE).stat().st_size / 1e6:.0f} MB)", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[rs] checkpoint FAILED at it={train_it}: {e!r}", flush=True)
            return out

    _Resumable.__name__ = cls.__name__
    _Resumable.__qualname__ = cls.__qualname__
    return _Resumable
