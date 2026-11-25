#!/usr/bin/env python3
"""Parse aizynthcli json(.gz) (pandas table orient) into success rate / mean steps over solved targets."""
import argparse
import gzip
import json
from pathlib import Path


def read_table(p: Path):
    op = gzip.open if p.name.endswith(".gz") else open
    with op(p, "rt") as f:
        d = json.load(f)
    return d["data"] if isinstance(d, dict) and "data" in d else d


def truthy(x):
    return x is True or str(x).lower() in ("true", "1", "1.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aizynth", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = read_table(Path(a.aizynth))
    solved = [truthy(r.get("is_solved")) for r in rows]
    steps = [float(r["number_of_steps"]) for r, s in zip(rows, solved) if s and r.get("number_of_steps") is not None]
    m = {"n_targets": len(rows), "n_solved": int(sum(solved)),
         "success_rate": 100.0 * sum(solved) / max(len(rows), 1),
         "mean_steps_solved": (sum(steps) / len(steps)) if steps else float("nan")}
    Path(a.out).write_text(json.dumps(m, indent=1))
    print(json.dumps(m))


if __name__ == "__main__":
    main()
