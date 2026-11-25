#!/usr/bin/env python3
"""Top-K diverse mode extraction for one LIT-PCBA optimisation run (paper protocol).

Protocol (CGFlow/RxnFlow papers): all generated molecules -> keep QED > 0.5 -> rank by docking score ->
top-K (K=100) diverse modes with the repo's own selector `rxnflow.utils.chem_metrics.compute_diverse_top_k`
(RDKit topological fingerprint, Tanimoto >= 0.5 excluded).
Per-molecule values are decoded from the SQLite sampling log written by the trainer:
    fr_0 = max(-0.1 * vina, 1e-4)   (vina reward; failed docking -> 1e-4)
    fr_1 = QED
Writes <out_dir>/topk.csv, <out_dir>/topk.smi, <out_dir>/metrics.json
"""
import argparse
import csv
import glob
import json
import sqlite3
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")
from rxnflow.utils.chem_metrics import compute_diverse_top_k  # noqa: E402

FAIL_EPS = 1.5e-4  # reward clipped at 1e-4 <=> docking failed / score 0


def read_run(run_dir: Path):
    paths = sorted(glob.glob(str(run_dir / "train" / "generated_objs_*.db")))
    if not paths:
        raise FileNotFoundError(f"no generated_objs_*.db under {run_dir}/train")
    rows = []
    for p in paths:
        con = sqlite3.connect(p, timeout=120)
        cur = con.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info(results)")]
        i_smi, i_fr0, i_fr1 = cols.index("smi"), cols.index("fr_0"), cols.index("fr_1")
        for row in cur.execute("SELECT * FROM results ORDER BY rowid"):
            rows.append((row[i_smi], float(row[i_fr0]), float(row[i_fr1])))
        con.close()
    return rows


def rows_per_iteration(run_dir: Path, n_rows: int, default: int = 64):
    """number of logged trajectories of every training iteration (RHC logs a variable batch size in train.log)"""
    import re

    prefixes = ("rhc", "_".join(("lane", "c")))
    sizes = {}
    log = run_dir / "train.log"
    if log.exists():
        for ln in log.read_text(errors="replace").splitlines():
            m = re.search(r"iteration (\d+) :", ln)
            if not m:
                continue
            b = re.search(r"(?<![a-z_])(?:" + "|".join(prefixes) + r")/batch_size:(\d+)", ln)
            sizes[int(m.group(1))] = int(b.group(1)) if b else default
    if not sizes:
        return [default] * (n_rows // default)
    return [sizes.get(i, default) for i in range(1, max(sizes) + 1)]


def curve_metrics(smis, vina, qed, bounds, k, thresh, vthr):
    """sample-efficiency curve: evaluation-protocol metrics on the molecules logged up to each budget boundary (iterations)"""
    out = {}
    for it, nrows in bounds:
        s, v, q = smis[:nrows], vina[:nrows], qed[:nrows]
        keep = [i for i in range(len(s)) if q[i] > 0.5 and v[i] < 0]
        if not keep:
            out[str(it)] = {"rows": len(s), "n_selected": 0, "vina_mean": None, "n_modes_vina_lt_thr": 0}
            continue
        idx = compute_diverse_top_k([s[i] for i in keep], [-v[i] for i in keep], k=k, thresh=thresh)
        sel = [keep[j] for j in idx]
        out[str(it)] = {
            "rows": len(s),
            "n_selected": len(sel),
            "vina_mean": float(np.mean([v[i] for i in sel])),
            "n_modes_vina_lt_thr": num_modes(s, v, q, vthr, thresh),
        }
    return out


def online_mode_counts(smis, vina, qed, thr, budgets, thresh=0.5):
    """Paper 'discovered modes' (sample-efficiency table / figure): molecules are visited in generation order; a molecule with Vina < thr and
    QED > 0.5 is a new mode iff its RDKit-fingerprint Tanimoto similarity to every already accepted mode is < thresh.  Returns {budget: #modes after
    the first `budget` generated molecules} for the budgets that do not exceed the number of logged molecules."""
    budgets = sorted(b for b in budgets if b <= len(smis))
    fps, out, bi = [], {}, 0
    for i, (s, v, q) in enumerate(zip(smis, vina, qed)):
        while bi < len(budgets) and i >= budgets[bi]:
            out[budgets[bi]] = len(fps)
            bi += 1
        if v < thr and q > 0.5:
            fp = Chem.RDKFingerprint(Chem.MolFromSmiles(s))
            if not fps or max(DataStructs.BulkTanimotoSimilarity(fp, fps)) < thresh:
                fps.append(fp)
    while bi < len(budgets):
        out[budgets[bi]] = len(fps)
        bi += 1
    return out


def num_modes(smis, vina, qed, vthr, thresh=0.5):
    """paper 'sample efficiency' metric: diverse modes with Vina < vthr, QED > 0.5, sim < thresh"""
    keep = [i for i in range(len(smis)) if vina[i] < vthr and qed[i] > 0.5]
    if not keep:
        return 0
    idx = compute_diverse_top_k([smis[i] for i in keep], [-vina[i] for i in keep], k=10**9, thresh=thresh)
    return len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--mode_vina", type=float, default=-10.0, help="Vina threshold for the #modes metric")
    ap.add_argument("--curve", default="16,78,157,250,469,500,750,1000", help="iteration budgets for the sample-efficiency curve ('' = off)")
    ap.add_argument("--target", default=None, help="LIT-PCBA target name; FEN1 uses the paper's looser mode threshold (Vina < -7)")
    ap.add_argument("--mode_budgets", default="1000,5000,10000,30000,64000", help="#molecules explored at which discovered modes are counted (paper table)")
    a = ap.parse_args()
    if a.target == "FEN1" and a.mode_vina == -10.0:
        a.mode_vina = -7.0  # paper footnote: FEN1 is evaluated with Vina < -7 to keep a similar number of modes

    rows = read_run(Path(a.run_dir))
    smis, vina, qed = [], [], []
    n_total = len(rows)
    for smi, fr0, fr1 in rows:
        if not smi:
            continue
        smis.append(smi)
        vina.append(0.0 if fr0 <= FAIL_EPS else -10.0 * fr0)
        qed.append(fr1)
    n_valid = len(smis)
    n_dock_ok = sum(v < 0 for v in vina)
    keep = [i for i in range(n_valid) if qed[i] > 0.5 and vina[i] < 0]
    idx = compute_diverse_top_k([smis[i] for i in keep], [-vina[i] for i in keep], k=a.k, thresh=a.thresh) if keep else []
    sel = [keep[j] for j in idx]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for i in sel:
        mol = Chem.MolFromSmiles(smis[i])
        ha = mol.GetNumHeavyAtoms()
        recs.append({"smiles": smis[i], "vina": vina[i], "qed": qed[i], "heavy_atoms": ha, "ligand_eff": -vina[i] / ha})
    with open(out / "topk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["smiles", "vina", "qed", "heavy_atoms", "ligand_eff"])
        w.writeheader()
        w.writerows(recs)
    (out / "topk.smi").write_text("\n".join(r["smiles"] for r in recs) + "\n")

    v = np.array([r["vina"] for r in recs]) if recs else np.array([np.nan])
    m = {
        "n_logged": n_total, "n_valid": n_valid, "n_dock_ok": n_dock_ok, "n_qed_pass": len(keep), "n_selected": len(sel),
        "vina_mean": float(np.mean(v)), "vina_median": float(np.median(v)),
        "qed_mean": float(np.mean([r["qed"] for r in recs])) if recs else float("nan"),
        "ligand_eff_mean": float(np.mean([r["ligand_eff"] for r in recs])) if recs else float("nan"),
        "heavy_atoms_mean": float(np.mean([r["heavy_atoms"] for r in recs])) if recs else float("nan"),
        "n_modes_vina_lt_thr": num_modes(smis, vina, qed, a.mode_vina),
        "mode_vina_thr": a.mode_vina,
        "best_vina": float(min(vina)) if vina else float("nan"),
    }
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m))

    if a.curve:
        cum = np.cumsum(rows_per_iteration(Path(a.run_dir), n_total))
        # raw rows (invalid trajectories have an empty smiles: give them an impossible QED so they are never selected)
        smi_all = [r[0] if r[0] else "C" for r in rows]
        qed_all = [r[2] if r[0] else -1.0 for r in rows]
        vina_all = [0.0 if r[1] <= FAIL_EPS else -10.0 * r[1] for r in rows]
        bounds = [(it, int(cum[it - 1])) for it in (int(x) for x in a.curve.split(",") if x) if it <= len(cum)]
        cv = curve_metrics(smi_all, vina_all, qed_all, bounds, a.k, a.thresh, a.mode_vina)
        (out / "curve.json").write_text(json.dumps(cv, indent=1))
        print("curve", json.dumps(cv))
    mb = [int(x) for x in a.mode_budgets.split(",") if x]
    if mb:
        smi_all = [r[0] if r[0] else "C" for r in rows]
        qed_all = [r[2] if r[0] else -1.0 for r in rows]
        vina_all = [0.0 if r[1] <= FAIL_EPS else -10.0 * r[1] for r in rows]
        mc = online_mode_counts(smi_all, vina_all, qed_all, a.mode_vina, mb, a.thresh)
        (out / "modes.json").write_text(json.dumps({"threshold": a.mode_vina, "online_modes": {str(k): v for k, v in mc.items()}}, indent=1))
        print("modes", json.dumps(mc))


if __name__ == "__main__":
    main()
