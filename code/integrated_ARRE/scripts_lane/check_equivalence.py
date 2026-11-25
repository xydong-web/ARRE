#!/usr/bin/env python3
"""Lane-D equivalence check at the level that is actually deterministic.

``generated_objs_*.db`` stores, for every sampled trajectory, the full action
sequence, the final SMILES, the sampled condition (preferences, beta) and the
two objective values.  Everything up to and including the pose model is
deterministic given the seed; the UniDock oracle is NOT (Uni-Dock's GPU
Monte-Carlo search gives different scores for the same ligand across runs even
though cgflow.utils.unidock.docking passes seed=1).  So the equivalence claim
is checked per field, and a baseline-vs-baseline pair is used as the noise
floor for the oracle column.

    python scripts_lane/check_equivalence.py RUN_A RUN_B [--iters 1]
"""
import argparse
import glob
import json
import sqlite3
import sys

FIELDS = ["traj", "smi", "pref_0", "pref_1", "ci_beta", "fr_1", "fr_0"]
LABEL = {
    "traj": "trajectory (protocol+block per step)",
    "smi": "final SMILES",
    "pref_0": "preference w_vina",
    "pref_1": "preference w_qed",
    "ci_beta": "temperature beta",
    "fr_1": "QED reward       (deterministic)",
    "fr_0": "Vina reward      (UniDock: NOT deterministic)",
}


def load(run_dir: str):
    f = sorted(glob.glob(f"{run_dir}/train/generated_objs_*.db"))
    if not f:
        raise SystemExit(f"no generated_objs db under {run_dir}/train")
    con = sqlite3.connect(f[0])
    rows = con.execute(f"select {','.join(FIELDS)} from results order by rowid").fetchall()
    con.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1,
                    help="how many leading iterations to compare (later ones inherit oracle noise)")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    n = min(len(A), len(B), args.batch * args.iters)
    out = {"run_a": args.a, "run_b": args.b, "trajectories_compared": n, "identical": {}}
    for i, f in enumerate(FIELDS):
        out["identical"][f] = sum(1 for k in range(n) if A[k][i] == B[k][i])

    det = [f for f in FIELDS if f != "fr_0"]
    ok = all(out["identical"][f] == n for f in det)
    out["verdict"] = (
        "PASS - every deterministic field is bit-identical" if ok
        else "FAIL - a deterministic field differs"
    )
    print(json.dumps(out, indent=2))
    print()
    for f in FIELDS:
        print(f"  {out['identical'][f]:>4}/{n}  {LABEL[f]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
