#!/usr/bin/env python3
"""WP1 report: full ablation of F = r2CE4L4 at the final budget, with the decision rules of docs/ROUND4_PLAN.md section 1 applied mechanically.
usage: ablation_report.py [--out docs/ABLATION_TABLE.md] [--min_seeds 2]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import report_lib as R  # noqa: E402

F = "r2CE4L4"
B = "base_ep30"
VARIANTS = [  # tag, label, role
    ("base_ep30", "baseline B", "reference"),
    ("r2CE4L4", "F = C + E + P(4) + LA", "model"),
    ("r2E4L4", "F - C (leave-C-out)", "loo"),
    ("r4CP4", "F - critic (C + prior only)", "loo"),
    ("r4CE4", "F - prior (kappa 0)", "loo"),
    ("r4CE4S4", "F - look-ahead (myopic prior)", "loo"),
    ("r4C", "only C", "single"),
    ("r4E4", "only E", "single"),
    ("r4P4", "only prior (+LA)", "single"),
    ("r4C0", "C, frozen policy (exploit-only 1)", "exploit"),
    ("r4CE0", "F, frozen policy (exploit-only 2)", "exploit"),
    ("r4B0", "B, frozen policy (null model)", "null"),
    ("r2CE4L2", "F with kappa 2", "kappa"),
    ("r4CE4L1", "F with kappa 1", "kappa"),
    ("r4CE4L8", "F with kappa 8", "kappa"),
    ("r2E4L2", "E + P(2) + LA (round 2)", "reference"),
]
KAPPA = {"r4CE4": 0, "r4CE4L1": 1, "r2CE4L2": 2, "r2CE4L4": 4, "r4CE4L8": 8}
MARGIN_VINA, MARGIN_AIZ = 0.10, 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--min_seeds", type=int, default=2, help="a target counts for a variant when it has at least this many seeds")
    a = ap.parse_args()
    D = {t: R.load(t) for t, _, _ in VARIANTS}

    def pt(tag, key):
        d = {t: s for t, s in D[tag].items() if len(s) >= a.min_seeds}
        return R.per_target(d, key)

    L = ["# Ablation of F at the final budget (WP1)", "", f"M1 = mean top-100 diverse Vina (kcal/mol, lower better), M2 = AiZynth success (%); per-target seed means over targets with >= {a.min_seeds} seeds; tests are paired over the common targets.", ""]
    L += ["| tag | variant | runs | targets | M1 | M2 | dM1 vs F | dM2 vs F | M1 wins vs F | Wilcoxon p (M1) | Wilcoxon p (M2) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    FV, FA = pt(F, "vina"), pt(F, "aiz")
    stat = {}
    for tag, label, role in VARIANTS:
        v, ai = pt(tag, "vina"), pt(tag, "aiz")
        pv, pa = R.paired(v, FV, "low"), R.paired(ai, FA, "high")
        stat[tag] = dict(vina=v, aiz=ai, p1=pv, p2=pa, role=role)
        wins = "-" if pv["wins"] is None else f"{pv['n'] - pv['wins']}/{pv['n']}"  # targets where the variant is WORSE than F (= F wins)
        L.append(f"| `{tag}` | {label} | {R.n_runs(D[tag])} | {len(v)} | {R.mean_over(v):.3f} | {R.mean_over(ai):.2f} | {pv['mean_diff']:+.3f} | {pa['mean_diff']:+.2f} | {wins} F wins | {pv['wilcoxon_p']:.2g} | {pa['wilcoxon_p']:.2g} |")
    L += ["", "(dM1 = variant - F, positive = variant worse; dM2 = variant - F, negative = variant worse.)", ""]
    # primary family
    loo = [t for t, _, r in VARIANTS if r == "loo"]
    ps, names = [], []
    for t in loo:
        for m, key in (("M1", "p1"), ("M2", "p2")):
            ps.append(stat[t][key]["wilcoxon_p"])
            names.append((t, m))
    adj = dict(zip(names, R.holm(ps)))
    L += ["## Primary family: four leave-one-out contrasts x two metrics (two-sided Wilcoxon, Holm over 8 tests)", "", "| contrast | metric | difference F - variant | Wilcoxon p | Holm p | verdict |", "|---|---|---|---|---|---|"]
    for t in loo:
        p1, p2 = stat[t]["p1"], stat[t]["p2"]
        h1, h2 = adj[(t, "M1")], adj[(t, "M2")]
        # F better on M1 <=> variant M1 higher (mean_diff > 0); F better on M2 <=> variant M2 lower (mean_diff < 0)
        sig1 = (h1 == h1) and h1 < 0.05 and p1["mean_diff"] > 0
        sig2 = (h2 == h2) and h2 < 0.05 and p2["mean_diff"] < 0
        ok1 = p1["mean_diff"] > -MARGIN_VINA  # F not worse than the variant on M1 by more than the margin
        ok2 = p2["mean_diff"] < MARGIN_AIZ
        contributes = (sig1 and ok2) or (sig2 and ok1)
        verdict = "pending (data incomplete)" if (h1 != h1 and h2 != h2) else ("component CONTRIBUTES" if contributes else "not individually necessary at this budget")
        L.append(f"| {t} | M1 | {-p1['mean_diff']:+.3f} | {p1['wilcoxon_p']:.2g} | {h1:.3g} | |")
        L.append(f"| {t} | M2 | {-p2['mean_diff']:+.2f} | {p2['wilcoxon_p']:.2g} | {h2:.3g} | **{verdict}** |")
    # sufficiency
    L += ["", "## Sufficiency (a single-component / exploit-only variant within 0.10 kcal/mol of F on M1 and not more than 3 points below F on M2 explains the gain)", ""]
    for t in [t for t, _, r in VARIANTS if r in ("single", "exploit")]:
        p1, p2 = stat[t]["p1"], stat[t]["p2"]
        if p1["n"] < 3:
            L.append(f"* `{t}`: not enough data yet")
            continue
        suff = p1["mean_diff"] <= MARGIN_VINA and p2["mean_diff"] >= -MARGIN_AIZ
        L.append(f"* `{t}`: dM1 {p1['mean_diff']:+.3f}, dM2 {p2['mean_diff']:+.2f} -> {'SUFFICIENT (explains the gain)' if suff else 'falls short of F'}")
    # not just exploitation
    L += ["", "## 'The gain is not just more exploitation' (F must beat r4C0 and r4CE0 on M1 in >= 10/15 targets and on the mean, paired Wilcoxon p < 0.05)", ""]
    for t in ("r4C0", "r4CE0"):
        p1 = stat[t]["p1"]
        if p1["n"] < 3:
            L.append(f"* vs `{t}`: not enough data yet")
            continue
        fw = p1["n"] - p1["wins"]
        ok = fw >= 10 and p1["mean_diff"] > 0 and p1["wilcoxon_p"] < 0.05
        L.append(f"* vs `{t}`: F wins {fw}/{p1['n']}, mean advantage {p1['mean_diff']:+.3f} kcal/mol, Wilcoxon p {p1['wilcoxon_p']:.2g} -> {'F beats the exploit-only control' if ok else 'NOT demonstrated: the paper must say the gain comes from sampler engineering'}")
    # null model
    b0, bb = stat["r4B0"]["vina"], stat[B]["vina"]
    pn = R.paired(b0, bb, "low")
    if pn["n"] >= 3:
        L += ["", f"## Null model: B vs B with frozen policy (r4B0): mean difference {pn['mean_diff']:+.3f} kcal/mol over {pn['n']} targets (Wilcoxon p {pn['wilcoxon_p']:.2g}); "
              + ("|B - B0| < 0.15: the baseline is statistically indistinguishable from random search over the action space at 64k calls." if abs(pn['mean_diff']) < 0.15 else "|B - B0| >= 0.15: the baseline does learn something measurable at this budget.")]
    # kappa
    ks = [(k, t) for t, k in KAPPA.items() if stat.get(t) and len(stat[t]["vina"]) >= 3]
    if len(ks) >= 4:
        kk = [k for k, _ in sorted(ks)]
        m1 = [R.mean_over(stat[t]["vina"]) for _, t in sorted(ks)]
        m2 = [R.mean_over(stat[t]["aiz"]) for _, t in sorted(ks)]
        r1, r2 = stats.spearmanr(kk, m1), stats.spearmanr(kk, m2)
        L += ["", "## kappa sweep", "", "| kappa | M1 | M2 |", "|---|---|---|"] + [f"| {k} | {a_:.3f} | {b_:.2f} |" for k, a_, b_ in zip(kk, m1, m2)]
        L += ["", f"Spearman(kappa, M1) = {r1.statistic:+.2f} (p {r1.pvalue:.2g}); Spearman(kappa, M2) = {r2.statistic:+.2f} (p {r2.pvalue:.2g}).  kappa = 4 stays the model of record."]
    # curves
    L += ["", "## Sample-efficiency (mean top-100 Vina over targets at 250 / 500 / 750 / 1000 iterations)", "", "| tag | @250 | @500 | @750 | @1000 |", "|---|---|---|---|---|"]
    for tag, _, _ in VARIANTS:
        cells = []
        for it in R.CURVE_ITS:
            v = R.per_target({t: s for t, s in D[tag].items() if len(s) >= a.min_seeds}, f"vina@{it}")
            cells.append(f"{R.mean_over(v):.3f}" if v else "-")
        L.append(f"| `{tag}` | " + " | ".join(cells) + " |")
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
