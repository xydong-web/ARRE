#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the ARRE generation-route addendum from its own files (no access to the raw runs needed).

  python3 verify_generation_routes_export.py --package <this directory> \
      --archive <ARRE_Selected_Molecules_20261118 directory> [--reexec 2000]

Checks: file checksums; join keys against selected_molecules.csv.gz; per-format structure of every route;
replay-prefix consistency of the ARRE continuation arms; recomputed coverage; optional RDKit re-execution of a
random sample of SynGA/SynGBO routes (building blocks and reaction SMARTS are stored in each record).
Exit status 0 only if every check passes.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import os
import random
import sys

NA_METHODS = {"GA", "S3"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reexec_record(rec):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    bbs = iter(b["smiles"] for b in rec["building_blocks"])
    rxs = iter(rec["reactions"])
    stack = []
    for tok in rec["postfix"].split(" "):
        if tok[0] == "B":
            stack.append(next(bbs))
            continue
        rx = next(rxs)
        prod = tok[1:].split(":", 1)[1]
        rxn = AllChem.ReactionFromSmarts(rx["smarts"])
        k = rxn.GetNumReactantTemplates()
        args, stack = stack[-k:], stack[:-k]
        want = Chem.MolToSmiles(Chem.MolFromSmiles(prod))
        hit = False
        for perm in sorted(set(itertools.permutations(range(k)))):
            mols = [Chem.MolFromSmiles(args[i]) for i in perm]
            for ps in rxn.RunReactants(tuple(mols)):
                for p in ps:
                    try:
                        Chem.SanitizeMol(p)
                    except Exception:
                        continue
                    if Chem.MolToSmiles(p) == want:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            return False
        stack.append(prod)
    return len(stack) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reexec", type=int, default=0, help="re-execute this many random SynGA/SynGBO routes with RDKit")
    a = ap.parse_args()
    pkg = a.package
    failures = []

    def check(name, cond, detail=""):
        print("%-4s %s%s" % ("PASS" if cond else "FAIL", name, (" - " + detail) if detail else ""), flush=True)
        if not cond:
            failures.append(name)

    # 1. checksums
    sums = json.load(open(os.path.join(pkg, "SHA256SUMS.json")))
    bad = [rel for rel, h in sums.items() if sha256_file(os.path.join(pkg, rel)) != h]
    check("SHA256SUMS.json (%d files)" % len(sums), not bad, "mismatch: %s" % bad if bad else "")

    summary = json.load(open(os.path.join(pkg, "export_summary.json")))
    arc_csv = os.path.join(a.archive, "selected_molecules.csv.gz")
    check("selected_molecules.csv.gz hash equals the one used for the export",
          sha256_file(arc_csv) == summary["archive_selected_molecules_sha256"])

    # 2. join keys
    with gzip.open(arc_csv, "rt", newline="") as f:
        arc = {r["candidate_id"]: r for r in csv.DictReader(f)}
    with gzip.open(os.path.join(pkg, "generation_route_status.csv.gz"), "rt", newline="") as f:
        status = list(csv.DictReader(f))
    check("status table covers every archive row exactly once",
          len(status) == len(arc) == len({s["candidate_id"] for s in status}) and {s["candidate_id"] for s in status} == set(arc))
    keys_ok = all(all(s[k] == arc[s["candidate_id"]][k] for k in ("cohort", "method", "run_tag", "task", "seed", "rank")) for s in status)
    check("status table join keys equal the archive", keys_ok)
    st_by_id = {s["candidate_id"]: s for s in status}
    cnt = collections.Counter(s["route_status"] for s in status)
    na_ok = all((s["method"] in NA_METHODS) == (s["route_status"] == "not_applicable_no_route_by_design") for s in status)
    check("not-applicable status is exactly GraphGA and S3-GFN", na_ok, str(dict(cnt)))

    # 3. route records
    n = collections.Counter()
    seen = set()
    prob = collections.Counter()
    syn_for_reexec = []
    per_triple = collections.Counter()
    prefix_sel = prefix_ok = 0
    with gzip.open(os.path.join(pkg, "generation_routes.jsonl.gz"), "rt") as f:
        for ln in f:
            r = json.loads(ln)
            cid = r["candidate_id"]
            n["records"] += 1
            if cid in seen or cid not in arc:
                prob["unknown_or_duplicate_id"] += 1
                continue
            seen.add(cid)
            A = arc[cid]
            if r["smiles"] != A["smiles"]:
                prob["smiles_differs_from_archive"] += 1
            if (r["cohort"], r["method"], r["run_tag"], r["task"], str(r["seed"]), str(r["rank"])) != (
                    A["cohort"], A["method"], A["run_tag"], A["task"], A["seed"], A["rank"]):
                prob["join_key_differs"] += 1
            s = st_by_id[cid]
            if s["route_status"] != "exported" or s["route_format_or_reason"] != r["route_format"]:
                prob["status_table_disagrees"] += 1
            ck = r["checks"]
            if not (ck.get("smiles_match") and ck.get("score_consistent")):
                prob["check_flag_false"] += 1
            per_triple[(r["cohort"], r["method"], r["run_tag"])] += 1
            fmt = r["route_format"]
            if fmt == "cgflow_traj_json_v1":
                acts = r["actions"]
                if not (acts and [x["step"] for x in acts] == list(range(len(acts))) and acts[0]["smiles"] == ""
                        and all(len(x["action"]) == 3 for x in acts)):
                    prob["cgflow_structure"] += 1
                if r["n_building_blocks"] != len(acts) or r["n_reactions"] != sum(1 for x in acts if x["action"][0].startswith("rxn")):
                    prob["cgflow_counts"] += 1
                if len(acts) > 1 and acts[1]["smiles"] != acts[0]["action"][2]:
                    prob["cgflow_step1_state_not_first_block"] += 1
                if r["route_ambiguous"] != ("alt_routes" in r):
                    prob["ambiguity_flag"] += 1
                arre = r.get("arre")
                if arre and arre.get("prefix_len"):
                    k = arre["prefix_len"]
                    prefix_sel += 1
                    routes = [acts] + [x["actions"] for x in r.get("alt_routes", [])]
                    good = any(len(c) >= k and c[:k] == arre.get("parent_prefix_actions") for c in routes)
                    if good and arre.get("prefix_matches_parent") is True:
                        prefix_ok += 1
                    else:
                        prob["replay_prefix_not_confirmed"] += 1
            elif fmt == "sfn_graphaction_v1":
                acts = r["actions"]
                kinds = [x["action"][0] for x in acts]
                if not (kinds[0] == "AddFirstReactant" and acts[-1]["state_smiles"] == r["smiles"]
                        and [x["step"] for x in acts] == list(range(len(acts)))):
                    prob["sfn_structure"] += 1
                if r["n_reactions"] != sum(1 for k_ in kinds if k_ in ("ReactUni", "ReactBi")):
                    prob["sfn_counts"] += 1
            elif fmt == "synga_postfix_v1":
                toks = r["postfix"].split(" ")
                bids = [int(t[1:]) for t in toks if t[0] == "B"]
                rids = [int(t[1:].split(":", 1)[0]) for t in toks if t[0] == "R"]
                if bids != [b["id"] for b in r["building_blocks"]] or rids != [x["id"] for x in r["reactions"]]:
                    prob["synga_tokens_vs_tables"] += 1
                last = toks[-1]
                if last[0] == "R" and last[1:].split(":", 1)[1] != r["smiles"]:
                    prob["synga_root_product"] += 1
                if r["n_building_blocks"] != len(bids) or r["n_reactions"] != len(rids):
                    prob["synga_counts"] += 1
                syn_for_reexec.append(r)
            else:
                prob["unknown_format"] += 1
    exported_ids = {s["candidate_id"] for s in status if s["route_status"] == "exported"}
    check("route file holds exactly the exported ids (%d)" % len(seen), seen == exported_ids and n["records"] == len(seen))
    check("per-record structural checks", not prob, str(dict(prob)) if prob else "no problems in %d records" % n["records"])
    check("ARRE replay-prefix confirmed for every continuation-arm molecule", prefix_sel == prefix_ok,
          "%d of %d" % (prefix_ok, prefix_sel))

    # 4. coverage recomputed
    cov = {(r["cohort"], r["method"], r["run_tag"]): r for r in csv.DictReader(open(os.path.join(pkg, "coverage.csv")))}
    cov_ok = all(int(cov[k]["exported"]) == per_triple.get(k, 0) for k in cov)
    tot = sum(int(r["selected"]) for r in cov.values())
    check("coverage.csv exported counts equal the route file", cov_ok)
    check("exported + not-applicable + unavailable = all archive rows",
          cnt["exported"] + cnt["not_applicable_no_route_by_design"] + cnt["unavailable_in_logs"] == len(arc) == tot,
          "%d + %d + %d = %d" % (cnt["exported"], cnt["not_applicable_no_route_by_design"], cnt["unavailable_in_logs"], len(arc)))

    # 5. optional RDKit re-execution (self-contained)
    if a.reexec:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            print("SKIP RDKit not importable; --reexec ignored")
        else:
            rnd = random.Random(20261118)
            sample = rnd.sample(syn_for_reexec, min(a.reexec, len(syn_for_reexec)))
            okc = sum(1 for r in sample if reexec_record(r))
            check("RDKit re-execution of %d random SynGA/SynGBO routes" % len(sample), okc == len(sample), "%d reproduced" % okc)
    print("\nRESULT:", "ALL CHECKS PASSED" if not failures else "FAILED: %s" % failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
