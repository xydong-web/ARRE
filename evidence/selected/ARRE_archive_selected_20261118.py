#!/usr/bin/env python3
"""Export the selected molecules and S19 statistics from frozen ARRE outputs."""

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import statistics
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "submission_delivery"
OLD_DATA = (DELIVERY / "ARRE_20261117/ARRE_GitHub_Code_20261117"
            / "manuscript/source/data")
NEW = DELIVERY / "ARRE_20261118"
PAPER = NEW / "ARRE_AdvancedScience_Manuscript_20261118/manuscript/source"
EXPORT = NEW / "ARRE_Selected_Molecules_20261118"
PLANNER = ROOT / "results/independent_retro_20261025"
EXCLUDED = "RG1_RAUSE"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def json_lines(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def gzip_text(path):
    with path.open("wb") as raw, gzip.GzipFile(
        fileobj=raw, filename="", mode="wb", mtime=0, compresslevel=5
    ) as stream, io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
        yield text


def planner_index():
    index = {}
    for folder, file, entries, hash_key, candidate_file, candidate_hash in (
        ("p3_full", "input_manifest.json", "source_files", "source_sha256",
         "top100_all_methods_frozen.csv", "candidates_sha256"),
        *((f"{folder}/run", f"{folder}/input_manifest_extra.json", "sources", "sha256",
           f"{folder}/frozen_extra.csv", "csv_sha256")
          for folder in ("trial", "s1_held20", "s1_dev15")),
    ):
        manifest = read(PLANNER / file)
        path = PLANNER / candidate_file
        check(sha(path) == manifest[candidate_hash], f"{path}: frozen CSV hash")
        rows = {}
        for row in csv_rows(path):
            unit = (row["method"], row["target"], int(row["seed"]))
            rows.setdefault(unit, []).append(row)
        for item in manifest[entries]:
            key = (item["tag"], item["target"], int(item["seed"]))
            if key[0] in ("tp26b", "tp26f", "x4newtB", "x4newtF"):
                continue
            unit = (item["label"], key[1], key[2])
            if unit not in rows:
                continue
            stem = f"unit_{unit[0]}_{unit[1]}_s{unit[2]}"
            check(key not in index, f"{key}: duplicated planner source")
            check((PLANNER / folder / f"{stem}.done").is_file(), f"{stem}: not done")
            index[key] = (item[hash_key], rows[unit], PLANNER / folder / f"{stem}.jsonl")
    return index


def unit_inventory(trial, port):
    held = {**trial["tags"], "RX": port["held20"]["tags"]["RX"],
            "RXCE": port["held20"]["tags"]["RXCE"]}
    dev = dict(port["dev15"]["tags"])
    comparison = [row for row in csv_rows(OLD_DATA / "legacy/fig2_target_means.csv")
                  if row["budget"] == "64000"]
    dev.update({row["method"]: row["tag"] for row in comparison})
    dev_targets = sorted({row["target"] for row in comparison})
    five = port["five5"]
    five_targets = next(iter(five["contrasts"]["m1_adj"].values()))["pockets"]
    for cohort, tags, targets in (("held20", held, trial["pockets"]),
                                  ("dev15", dev, dev_targets),
                                  ("dev5_subset", {"S3": five["tags"]["S3"]}, five_targets)):
        for method, tag in sorted(tags.items()):
            for pocket in sorted(targets):
                for seed in range(3):
                    yield cohort, method, tag, pocket, seed


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def same_candidate(one, two):
    return (all(close(one[k], two[k]) for k in ("vina", "qed", "heavy_atoms"))
            and one["smiles"] == two.get("smiles_in", two.get("smiles")))


def join_aizynth(rows, aiz_data, key):
    by_smi = {row["target"]: row for row in aiz_data}
    check(len(aiz_data) == len(by_smi) == 100, f"{key}: AiZynth count")
    selected = {r["smiles"] for r in rows}
    if set(by_smi) == selected:
        return by_smi

    from rdkit import Chem

    def canonical(smiles):
        mol = Chem.MolFromSmiles(smiles)
        check(mol is not None, f"{key}: invalid SMILES in AiZynth identity check")
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    missing = {canonical(s): s for s in selected - by_smi.keys()}
    extra = {canonical(s): s for s in by_smi.keys() - selected}
    check(missing.keys() == extra.keys() and len(missing) == len(selected - by_smi.keys())
          and len(extra) == len(by_smi.keys() - selected), f"{key}: ambiguous AiZynth identity")
    # AiZynth preserves the original input index; require it to confirm each alias.
    rank = {row["smiles"]: idx for idx, row in enumerate(rows)}
    for identity, smiles in missing.items():
        record = by_smi.pop(extra[identity])
        check(record["index"] == rank[smiles], f"{key}: AiZynth alias/input index conflict")
        by_smi[smiles] = record
    return by_smi


def output_molecules(trial, port):
    planned = planner_index()
    traj_path = OLD_DATA / "method_v10/selected_trajectory.json"
    trajectory = read(traj_path)
    traj_hash = sha(traj_path)
    traj_verified = read(OLD_DATA / "method_v10/source_manifest.json")
    check(traj_verified["verified"], "Forward trajectory example is not verified")
    coverage = {}
    total = Counter()
    fields = ("candidate_id", "cohort", "method", "run_tag", "task", "seed", "rank",
              "smiles", "vina_kcal_mol", "qed", "heavy_atoms", "ligand_efficiency",
              "aizynth_solved", "aizynth_tree_count", "retrostar_solved", "retrostar_status",
              "retrostar_route_n_reactions", "forward_path_recorded")
    with ExitStack() as stack:
        molecules = csv.writer(stack.enter_context(gzip_text(EXPORT / "selected_molecules.csv.gz")))
        molecules.writerow(fields)
        routes = stack.enter_context(gzip_text(EXPORT / "aizynth_predicted_routes.jsonl.gz"))
        retro = stack.enter_context(gzip_text(EXPORT / "retrostar_outcomes.jsonl.gz"))
        forward = stack.enter_context((EXPORT / "recorded_forward_paths.jsonl").open("w"))
        provenance = stack.enter_context((EXPORT / "run_manifest.jsonl").open("w"))
        for cohort, method, tag, task, seed in unit_inventory(trial, port):
            key = (tag, task, seed)
            root = ROOT / "results" / tag / task / f"seed{seed}"
            topk, aiz, metrics = (root / name for name in
                                  ("topk.csv", "aizynth.json.gz", "metrics.json"))
            top_sha = sha(topk)
            if key == ("r2CE4L4", "ADRB2", 0):
                check(top_sha == traj_verified["source_hashes"]["topk.csv"],
                      "Forward trajectory receipt topk hash mismatch")
            rows = csv_rows(topk)
            expected_hash, frozen, result = planned[key]
            check(len(rows) == len({r["smiles"] for r in rows}) == len(frozen) == 100,
                  f"{key}: selected set count")
            check(top_sha == expected_hash, f"{key}: selected set hash")
            run_metrics = read(metrics)
            check(run_metrics["n_selected"] == 100 and all(
                close(statistics.fmean(float(row[k]) for row in rows), run_metrics[name])
                for k, name in (("vina", "vina_mean"), ("qed", "qed_mean"),
                                ("heavy_atoms", "heavy_atoms_mean"))
            ), f"{key}: topk/metrics disagreement")
            with gzip.open(aiz, "rt") as f:
                aiz_data = json.load(f)["data"]
            by_smi = join_aizynth(rows, aiz_data, key)
            aiz_metrics = root / "aizynth_metrics.json"
            labels = read(aiz_metrics)
            n_solved = sum(row["is_solved"] for row in aiz_data)
            check(n_solved == labels["n_solved"], f"{key}: AiZynth solved count")
            results = json_lines(result)
            by_rank = {r["rank"]: r for r in results}
            check(len(results) == len(by_rank) == 100, f"{key}: independent planner count")
            check(all((r["target"], r["seed"], r["method"]) ==
                      (task, seed, frozen[0]["method"]) for r in results),
                  f"{key}: independent planner unit")
            counts = coverage.setdefault((cohort, method), Counter())
            unit_counts = Counter(units=1)
            for rank, (row, frozen_row) in enumerate(zip(rows, frozen, strict=True), 1):
                ret = by_rank[rank]
                check(int(frozen_row["rank"]) == rank and same_candidate(row, frozen_row)
                      and same_candidate(row, ret), f"{key}: rank/identity mismatch {rank}")
                cid = f"{cohort}:{tag}:{task}:{seed}:{rank}"
                ar = by_smi[row["smiles"]]
                trees = ar["trees"] or []
                alias = ar["target"] != row["smiles"]
                json.dump(dict(candidate_id=cid, is_solved=ar["is_solved"], trees=trees,
                               planner_target_smiles=ar["target"],
                               identity_match="isomeric_alias" if alias else "exact"),
                          routes, separators=(",", ":"), allow_nan=False)
                routes.write("\n")
                json.dump({"candidate_id": cid, **{k: ret[k] for k in (
                    "solved", "status", "in_stock", "termination", "route_n_reactions",
                    "n_calls", "smiles_canonical")}}, retro, separators=(",", ":"))
                retro.write("\n")
                available = False
                if key == ("r2CE4L4", "ADRB2", 0):
                    for role in ("parent", "child"):
                        if row["smiles"] == trajectory[role]["smiles"]:
                            available = True
                            print(json.dumps(dict(
                                candidate_id=cid, role=role, receipt_sha256=traj_hash,
                                iteration=trajectory[role]["it"],
                                actions=trajectory[f"{role}_trajectory"])), file=forward)
                molecules.writerow((cid, cohort, method, tag, task, seed, rank,
                                    *(row[k] for k in ("smiles", "vina", "qed", "heavy_atoms",
                                                       "ligand_eff")), ar["is_solved"], len(trees),
                                    ret["solved"], ret["status"], ret["route_n_reactions"],
                                    available))
                unit_counts.update(candidates=1, aizynth_outcomes=1,
                                   aizynth_with_trees=bool(trees), aizynth_solved=ar["is_solved"],
                                   aizynth_isomeric_aliases=alias,
                                   retrostar_outcomes=1, retrostar_solved=ret["solved"],
                                   retrostar_errors=ret["status"] == "error",
                                   forward_paths=available)
            sources = {str(p.relative_to(ROOT)): sha(p) for p in
                       (aiz, metrics, aiz_metrics, result, result.with_suffix(".done"))}
            sources[str(topk.relative_to(ROOT))] = top_sha
            print(json.dumps(dict(cohort=cohort, method=method, tag=tag, task=task, seed=seed,
                                  counts=unit_counts, inputs_sha256=sources), sort_keys=True),
                  file=provenance)
            total.update(unit_counts)
            counts.update(unit_counts)
            if total["units"] % 100 == 0:
                print(f"Verified and exported {total['units']} units", flush=True)
    check((total["units"], total["candidates"]) == (795, 79500), "Archive inventory incomplete")
    check(total["forward_paths"] > 0, "Verified forward example did not join")
    with (EXPORT / "coverage.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("cohort", "method", *total))
        for (cohort, method), counts in sorted(coverage.items()):
            writer.writerow((cohort, method, *(counts[k] for k in total)))
    (EXPORT / "export_summary.json").write_text(json.dumps(total, indent=2) + "\n")
    shutil.copy2(traj_path, EXPORT / "forward_path_source_receipt.json")
    print(dict(total), flush=True)


def paired(values):
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    c95 = t.ppf(.975, n - 1) * se
    return dict(n=n, mean=mean, low95=mean-c95, high95=mean+c95,
                lower_one_sided95=mean-t.ppf(.95, n - 1)*se,
                negative=sum(v < 0 for v in values), positive=sum(v > 0 for v in values))


def estimate(vectors):
    rows = []
    for contrast, records in vectors.items():
        check(len(records) == len({r["pocket"] for r in records}) == 20
              and sum(r["pocket"] == EXCLUDED for r in records) == 1,
              f"{contrast}: need 20 unique pockets including {EXCLUDED}")
        for subset, keep in (("frozen20", lambda r: True),
                             ("exclude_RG1_RAUSE", lambda r: r["pocket"] != EXCLUDED)):
            rows.append(dict(contrast=contrast, subset=subset,
                             **paired([float(r["difference"]) for r in records if keep(r)])))
    return rows


def write_stats(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_sensitivity(trial, port):
    contrasts = {
        k: trial["P1_m1_adj_R1"][k] for k in ("F-B", "F-SynGBO", "FZ-SynGBO")
    }
    contrasts["RXCE-RX"] = port["held20"]["contrasts"]["m1_adj"]["RXCE-RX"]
    contrasts["FZ-SynGBO Retro*"] = trial["P2"]["FZ_vs_SynGBO_noninferiority"]
    vectors = {}
    with (EXPORT / "heldout_differences.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("contrast", "pocket", "difference", "units"))
        writer.writeheader()
        for contrast, source in contrasts.items():
            vectors[contrast] = [
                dict(contrast=contrast, pocket=pocket, difference=value,
                     units="percentage_points" if "Retro*" in contrast else "kcal/mol")
                for pocket, value in zip(source["pockets"], source["diffs"], strict=True)
            ]
            writer.writerows(vectors[contrast])
    rows = estimate(vectors)
    for row in rows[::2]:
        frozen = contrasts[row["contrast"]]
        check(all(close(a, b) for a, b in zip(
            (row["mean"], row["low95"], row["high95"]), (frozen["mean"], *frozen["ci95"])
        )), f"{row['contrast']}: frozen summary disagreement")
    check(close(rows[-2]["lower_one_sided95"],
                contrasts["FZ-SynGBO Retro*"]["one_sided_lower_limit_pp"]),
          "Frozen non-inferiority bound disagreement")
    write_stats(EXPORT / "heldout_sensitivity.csv", rows)
    (EXPORT / "heldout_source_sha256.json").write_text(json.dumps({
        fname: sha(OLD_DATA / fname) for fname in ("trial.json", "portability.json")
    }, indent=2) + "\n")
    labels = ("F--CGFlow", "F--SynGBO", "FZ--SynGBO", "Guided--parent RxnFlow",
              "FZ--SynGBO Retro* (pp)")
    with (PAPER / "homology_sensitivity_rows.tex").open("w") as f:
        for label, orig, new in zip(labels, rows[::2], rows[1::2], strict=True):
            values = [f"${r['mean']:.3f}$ & $[{r['low95']:.3f}, {r['high95']:.3f}]$"
                      for r in (orig, new)]
            print(" & ".join((label, *values)) + r" \\", file=f)
        print(r"Retro* one-sided 95\% lower bound (pp) & \multicolumn{2}{c}{$"
              f"{rows[-2]['lower_one_sided95']:.3f}" r"$} & \multicolumn{2}{c}{$"
              f"{rows[-1]['lower_one_sided95']:.3f}" r"$} \\", file=f)
        print(r"\bottomrule", file=f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", type=Path, metavar="HELDOUT_DIFFERENCES_CSV",
                        help="read portable vectors and print the sensitivity summaries")
    args = parser.parse_args()
    if args.recompute:
        vectors = {}
        for row in csv_rows(args.recompute):
            vectors.setdefault(row["contrast"], []).append(row)
        print(json.dumps(estimate(vectors), indent=2))
        return
    EXPORT.mkdir(exist_ok=True)
    trial, port = (read(OLD_DATA / name) for name in ("trial.json", "portability.json"))
    generate_sensitivity(trial, port)
    output_molecules(trial, port)


if __name__ == "__main__":
    main()
