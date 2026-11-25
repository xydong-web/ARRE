#!/usr/bin/env python3
"""Aggregate per-run metrics into per-target / overall tables and apply the PRE-REGISTERED acceptance rules.

Layout read:   <results_root>/<tag>/<target>/seed<k>/{metrics.json, aizynth_metrics.json, curve.json}
Paper targets: paper_targets/paper_tables_raw.json (parsed from the paper's appendix tables, CGFlow row = 3-seed means)

Rules (see docs/PROMOTION_RULES.md)
  reproduction : 15-target mean Vina of the tag must reach >= (1-tol) of the paper's mean (tol = 2%: ours <= paper * 0.98, both negative),
                 AiZynth success >= (1-tol) * paper.  Per-target numbers are reported (wins vs paper, ligand efficiency, steps).
  innovation   : candidate tag vs baseline tag, seed-averaged per-target values on the common targets.
                 PASS iff  (a) Vina wins on >= ceil(min_win_frac * n) targets,
                           (b) mean Vina delta <= -min_delta kcal/mol,
                           (c) one-sided paired t-test p < alpha on the per-target deltas,
                           (d) AiZynth success not worse than the baseline by more than aiz_noninf points (mean over targets),
                           (e) beats the PAPER's per-target Vina on a majority of the targets (user requirement).
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

TARGETS = ["ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1",
           "PKM2", "PPARG", "TP53", "VDR"]
CURVE_ITS = ["250", "500", "750", "1000"]


def load_tag(root: Path, tag: str):
    """-> {target: {seed: {metric: value}}}"""
    out = {}
    for t in TARGETS:
        d = root / tag / t
        if not d.exists():
            continue
        for sd in sorted(d.glob("seed*")):
            m = {}
            f1, f2, f3 = sd / "metrics.json", sd / "aizynth_metrics.json", sd / "curve.json"
            if f1.exists():
                j = json.loads(f1.read_text())
                m.update({"vina": j["vina_mean"], "qed": j["qed_mean"], "ligand_eff": j["ligand_eff_mean"],
                          "n_selected": j["n_selected"], "n_modes": j.get("n_modes_vina_lt_thr"),
                          "heavy_atoms": j.get("heavy_atoms_mean")})
            if f2.exists():
                j = json.loads(f2.read_text())
                m.update({"aizynth": j["success_rate"], "steps": j["mean_steps_solved"]})
            f4 = sd / "modes.json"
            if f4.exists():
                for b, c in json.loads(f4.read_text()).get("online_modes", {}).items():
                    m[f"omodes@{b}"] = float(c)
            if f3.exists():
                cj = json.loads(f3.read_text())
                for it in CURVE_ITS:
                    if it in cj and cj[it].get("vina_mean") is not None:
                        m[f"vina@{it}"] = cj[it]["vina_mean"]
                        m[f"modes@{it}"] = cj[it]["n_modes_vina_lt_thr"]
            if m:
                out.setdefault(t, {})[sd.name] = m
    return out


def per_target(data, key):
    res = {}
    for t, seeds in data.items():
        v = [m[key] for m in seeds.values() if key in m and m[key] is not None and not math.isnan(m[key])]
        if v:
            res[t] = (st.mean(v), (st.pstdev(v) if len(v) > 1 else float("nan")), len(v))
    return res


def paper_dict(paper, metric):
    return {t: v[0] for t, v in paper.get(metric, {}).get("CGFlow", {}).items()}


def mean_of(d, keys=None):
    ks = [k for k in (keys if keys is not None else d) if k in d]
    return st.mean(d[k][0] for k in ks) if ks else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/path/to/ARRE/results")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--paper", default="/path/to/ARRE/paper_targets/paper_tables_raw.json")
    ap.add_argument("--baseline_tag", default=None, help="compare against this tag (innovation rule)")
    ap.add_argument("--paper_v1", default="/path/to/ARRE/paper_targets/paper_tables_raw_v1.json",
                    help="arXiv v1 (submission-era) numbers of the paper, reported next to the camera-ready (v2) numbers that gate acceptance")
    ap.add_argument("--paper_modes", default="/path/to/ARRE/paper_targets/paper_modes_v2.json")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--min_win_frac", type=float, default=0.6)
    ap.add_argument("--min_delta", type=float, default=0.05, help="kcal/mol, required mean improvement")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--aiz_noninf", type=float, default=3.0, help="percentage points")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.results)
    data = load_tag(root, a.tag)
    paper = json.loads(Path(a.paper).read_text()) if Path(a.paper).exists() else {}
    pv, pa, pl = (paper_dict(paper, k) for k in ("vina", "aizynth_success", "ligand_eff"))
    paper1 = json.loads(Path(a.paper_v1).read_text()) if Path(a.paper_v1).exists() else {}
    pv1, pa1 = (paper_dict(paper1, k) for k in ("vina", "aizynth_success"))
    V, A, L, Q, S = (per_target(data, k) for k in ("vina", "aizynth", "ligand_eff", "qed", "steps"))
    nm = per_target(data, "n_modes")
    rep = {"tag": a.tag, "targets_done": sorted(data), "n_targets": len(data), "per_target": {}}

    print(f"tag = {a.tag}")
    print(f"{'target':11s} {'seeds':>5s} {'vina':>14s} {'paper':>8s} {'v1':>7s} {'d%':>6s} | {'aiz%':>13s} {'paper':>7s} {'v1':>6s} | {'LE':>6s} {'paperLE':>7s} | {'QED':>5s} {'modes<-10':>9s}")
    for t in TARGETS:
        if t not in V:
            continue
        v, vs, n = V[t]
        d = 100 * (v - pv[t]) / abs(pv[t]) if t in pv else float("nan")
        aa = A.get(t, (float("nan"),) * 3)
        le = L.get(t, (float("nan"),) * 3)
        q = Q.get(t, (float("nan"),) * 3)
        mm = nm.get(t, (float("nan"),) * 3)
        print(f"{t:11s} {n:5d} {v:8.3f}±{vs:5.3f} {pv.get(t, float('nan')):8.2f} {pv1.get(t, float('nan')):7.2f} {d:+6.1f} | {aa[0]:7.2f}±{aa[1]:5.2f} {pa.get(t, float('nan')):7.2f} {pa1.get(t, float('nan')):6.1f} | {le[0]:6.3f} {pl.get(t, float('nan')):7.3f} | {q[0]:5.3f} {mm[0]:9.1f}")
        rep["per_target"][t] = {"vina": v, "vina_std": vs, "n_seeds": n, "aizynth": aa[0], "ligand_eff": le[0], "qed": q[0],
                                "paper_vina": pv.get(t), "paper_aizynth": pa.get(t), "paper_ligand_eff": pl.get(t)}
    if V:
        mv, pm = mean_of(V), st.mean(pv[t] for t in V if t in pv)
        rep.update(mean_vina=mv, paper_mean_vina_same_targets=pm, vina_ratio_to_paper=mv / pm, vina_pass_2pct=bool(mv <= pm * (1 - a.tol)),
                   vina_wins_vs_paper=int(sum(V[t][0] < pv[t] for t in V if t in pv)), n_targets_vina=len(V))
        print(f"\nmean Vina over {len(V)} targets: ours {mv:.3f} vs paper(v2, camera-ready) {pm:.3f}  ratio {mv/pm:.4f}  pass(>={100*(1-a.tol):.0f}%)={rep['vina_pass_2pct']}"
              f"   per-target wins vs paper: {rep['vina_wins_vs_paper']}/{len(V)}")
        if pv1:
            pm1 = st.mean(pv1[t] for t in V if t in pv1)
            rep.update(paper_v1_mean_vina_same_targets=pm1, vina_ratio_to_paper_v1=mv / pm1, vina_pass_2pct_v1=bool(mv <= pm1 * (1 - a.tol)))
            print(f"                 vs paper(v1, arXiv April 2025) {pm1:.3f}  ratio {mv/pm1:.4f}  pass(>={100*(1-a.tol):.0f}%)={rep['vina_pass_2pct_v1']}")
    if A:
        ma, pam = mean_of(A), st.mean(pa[t] for t in A if t in pa)
        rep.update(mean_aizynth=ma, paper_mean_aizynth_same_targets=pam, aizynth_pass_2pct=bool(ma >= pam * (1 - a.tol)))
        print(f"mean AiZynth success over {len(A)} targets: ours {ma:.2f} vs paper(v2) {pam:.2f}  pass(>={100*(1-a.tol):.0f}%)={rep['aizynth_pass_2pct']}"
              + (f"   | paper(v1) {st.mean(pa1[t] for t in A if t in pa1):.2f}" if pa1 else "") + "   (NOTE: the paper's own CGFlow AiZynth number changed 48.1 -> 68.6 between v1 and v2, i.e. it depends on the evaluator config)")
    if S:
        print(f"mean AiZynth steps: {mean_of(S):.2f} (paper 2.39)")
    if L:
        ml, pml = mean_of(L), st.mean(pl[t] for t in L if t in pl)
        rep.update(mean_ligand_eff=ml, paper_mean_ligand_eff=pml)
        print(f"mean ligand efficiency: ours {ml:.3f} vs paper {pml:.3f}")
    # sample-efficiency curve (mean over targets)
    cur = {}
    for it in CURVE_ITS:
        c = per_target(data, f"vina@{it}")
        if c:
            cur[it] = (mean_of(c), len(c), mean_of(per_target(data, f"modes@{it}")))
    if cur:
        print("curve (mean top-100 Vina over targets @ iterations; #modes<-10):",
              "  ".join(f"@{it}: {v[0]:.3f} (n={v[1]}, modes {v[2]:.1f})" for it, v in cur.items()))
        rep["curve"] = {it: {"vina": v[0], "n": v[1], "modes": v[2]} for it, v in cur.items()}

    # discovered modes vs the paper's sample-efficiency table (5 targets; Vina<-10, FEN1<-7; QED>0.5; Tanimoto<0.5; paper = 2 seeds)
    if Path(a.paper_modes).exists():
        pm_ = json.loads(Path(a.paper_modes).read_text())
        bud = pm_["budgets"]
        rows_ = []
        for t, ref in pm_["CGFlow"].items():
            if t == "Average" or t not in data:
                continue
            ours = per_target(data, "omodes@%d" % bud[0])  # just to know which targets have modes.json
            vals = [per_target(data, "omodes@%d" % b).get(t) for b in bud]
            if any(v is not None for v in vals):
                rows_.append((t, vals, ref))
        if rows_:
            print("\ndiscovered modes (online, ours vs paper CGFlow)   budgets: " + " / ".join(str(b) for b in bud))
            sums_o, sums_p, cnt = [0.0] * len(bud), [0.0] * len(bud), [0] * len(bud)
            for t, vals, ref in rows_:
                cells = []
                for i, v in enumerate(vals):
                    if v is None:
                        cells.append("   -  ")
                    else:
                        cells.append(f"{v[0]:7.1f}/{ref[i]:7.1f}")
                        sums_o[i] += v[0]; sums_p[i] += ref[i]; cnt[i] += 1
                print(f"  {t:11s} " + "  ".join(cells))
            avg = "  ".join((f"{sums_o[i]/cnt[i]:7.1f}/{sums_p[i]/cnt[i]:7.1f}" if cnt[i] else "   -  ") for i in range(len(bud)))
            print(f"  {'mean':11s} " + avg + f"   (ours/paper; mean over the {len(rows_)} targets with data)")
            rep["modes_vs_paper"] = {str(bud[i]): {"ours": (sums_o[i] / cnt[i] if cnt[i] else None), "paper": (sums_p[i] / cnt[i] if cnt[i] else None)} for i in range(len(bud))}

    if a.baseline_tag:
        from scipy import stats

        base = load_tag(root, a.baseline_tag)
        BV, BA, BL = per_target(base, "vina"), per_target(base, "aizynth"), per_target(base, "ligand_eff")
        common = [t for t in TARGETS if t in V and t in BV]
        n = len(common)
        if n < 3:
            print(f"\nvs baseline '{a.baseline_tag}': only {n} common targets - no decision")
        else:
            dv = np.array([V[t][0] - BV[t][0] for t in common])  # negative = better
            wins = int((dv < 0).sum())
            need = math.ceil(a.min_win_frac * n)
            tt = stats.ttest_1samp(dv, 0.0, alternative="less")
            try:
                wx = stats.wilcoxon(dv, alternative="less").pvalue if np.any(dv != 0) else 1.0
            except ValueError:
                wx = float("nan")
            rs = np.random.RandomState(0)
            boot = [dv[rs.randint(0, n, n)].mean() for _ in range(5000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ca = [t for t in common if t in A and t in BA]
            da = np.array([A[t][0] - BA[t][0] for t in ca]) if ca else np.array([0.0])
            wins_a = int((da > 0).sum()) if ca else 0
            wins_l = sum(L[t][0] > BL[t][0] for t in common if t in L and t in BL)
            paper_wins = sum(V[t][0] < pv[t] for t in common if t in pv)
            crit = {
                "a_wins>=need": wins >= need,
                "b_mean_delta<=-min": float(dv.mean()) <= -a.min_delta,
                "c_paired_t_p<alpha": bool(tt.pvalue < a.alpha),
                "d_aizynth_noninferior": (float(da.mean()) >= -a.aiz_noninf) if ca else None,
                "e_beats_paper_majority": paper_wins >= math.ceil(0.5 * n + 0.01),
            }
            decided = all(v for v in crit.values() if v is not None)
            rep["vs_baseline"] = {"baseline": a.baseline_tag, "n_common": n, "vina_wins": wins, "need": need, "mean_dvina": float(dv.mean()),
                                  "ci95": [float(lo), float(hi)], "t_p": float(tt.pvalue), "wilcoxon_p": float(wx), "aizynth_wins": wins_a,
                                  "mean_daizynth": float(da.mean()), "ligand_eff_wins": int(wins_l), "paper_wins": int(paper_wins),
                                  "criteria": crit, "PROMOTE": bool(decided)}
            print(f"\n== vs baseline '{a.baseline_tag}' on {n} targets ==")
            print(f"   Vina: wins {wins}/{n} (need {need}); mean delta {dv.mean():+.3f} kcal/mol, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]; "
                  f"paired-t p={tt.pvalue:.4f}, Wilcoxon p={wx:.4f}; beats paper on {paper_wins}/{n}")
            print(f"   AiZynth: wins {wins_a}/{len(ca)}, mean delta {da.mean():+.2f} pts;  ligand-eff wins {wins_l}/{n}")
            for it in CURVE_ITS:
                bc = per_target(base, f"vina@{it}")
                lc = per_target(data, f"vina@{it}")
                cc = [t for t in common if t in bc and t in lc]
                if cc:
                    print(f"   curve @{it}: lane {st.mean(lc[t][0] for t in cc):.3f} vs base {st.mean(bc[t][0] for t in cc):.3f} "
                          f"(delta {st.mean(lc[t][0]-bc[t][0] for t in cc):+.3f}, wins {sum(lc[t][0] < bc[t][0] for t in cc)}/{len(cc)})")
            print("   criteria:", crit)
            print(f"   ==> PROMOTE = {decided}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
