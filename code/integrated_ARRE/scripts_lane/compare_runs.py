#!/usr/bin/env python3
"""Compare two CGFlow run directories iteration by iteration.

Used for the lane-D equivalence evidence: a baseline run and a
``lane_d.enable=false`` run with the same seed must agree exactly on every
scalar the trainer logs (loss, logZ, sampled rewards, ...).

    python scripts_lane/compare_runs.py RUN_A RUN_B [--keys loss,logZ]
"""
import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"iteration (\d+) : (.*)$")


def parse(run_dir: Path) -> dict[int, dict[str, float]]:
    log = run_dir / "train.log"
    out: dict[int, dict[str, float]] = {}
    for raw in log.read_text(errors="replace").splitlines():
        m = LINE.search(raw)
        if not m:
            continue
        it = int(m.group(1))
        vals: dict[str, float] = {}
        for tok in m.group(2).split():
            if ":" not in tok:
                continue
            k, _, v = tok.rpartition(":")
            try:
                vals[k] = float(v)
            except ValueError:
                pass
        out[it] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--keys", default="")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--skip", default="time_spent,train_time,eval_time",
                    help="comma separated keys that are wall-clock and must differ")
    args = ap.parse_args()

    A, B = parse(Path(args.a)), parse(Path(args.b))
    skip = {k for k in args.skip.split(",") if k}
    its = sorted(set(A) & set(B))
    if not its:
        print("NO COMMON ITERATIONS -> FAIL")
        return 2
    wanted = [k for k in args.keys.split(",") if k]

    diffs, compared = [], 0
    for it in its:
        keys = wanted or sorted((set(A[it]) | set(B[it])) - skip)
        for k in keys:
            if k in skip:
                continue
            va, vb = A[it].get(k), B[it].get(k)
            if va is None or vb is None:
                diffs.append((it, k, va, vb))
                continue
            compared += 1
            if abs(va - vb) > args.atol:
                diffs.append((it, k, va, vb))

    print(json.dumps({
        "run_a": args.a, "run_b": args.b,
        "iterations": its, "scalars_compared": compared,
        "mismatches": len(diffs),
        "verdict": "PASS (identical)" if not diffs else "FAIL (differs)",
    }, indent=2))
    for it, k, va, vb in diffs[:40]:
        print(f"  it={it} {k}: {va} vs {vb}")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
