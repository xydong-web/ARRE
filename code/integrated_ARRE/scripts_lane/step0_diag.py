"""Lane A - Step-0 study: how wrong is the baseline's uniform-sub-sample normaliser?

For a real policy checkpoint of a CGFlow run it measures, on the states the policy itself
visits (step 0 = first block, step 1 = first BiRxn):

  * the exact normaliser  log Z = log sum_{legal (p,b)} exp l_theta(s,p,b)   (full pass,
    block tower evaluated on the fly - no approximation),
  * M_eff = 1 / sum_b P_b^2, and the mass of the top-1 / top-10 / top-100 actions,
  * the distribution of the BASELINE estimator `log Z_hat` over R independent draws of its
    uniform 1% sub-sample - obtained by running the real `RxnActionCategorical`, not a model
    of it - i.e. bias, sd, and how often the `clamp(max=0)` in `log_prob` fires,
  * the probability that the top-1 / top-10 actions are inside the behaviour sub-sample,
  * the same bias/sd for the PAN estimator at an equal sample budget.

usage (inside a GPU Slurm job, see scripts_lane/step0.sbatch):
    python scripts_lane/step0_diag.py --target ADRB2 --ckpt <model_state_100.pt> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

C = Path("/path/to/ARRE")


def build_trainer(target: str, pocket_dir: Path, env_dir: str, log_dir: str, num_states: int, seed: int):
    from cgflow.config import Config, init_empty
    from cgflow.lanes.lane_a import VinaMOOTrainer_laneA

    cfg = init_empty(Config())
    cfg.desc = "lane-A step0 diagnostic"
    cfg.env_dir = env_dir
    cfg.num_training_steps = 1
    cfg.num_validation_gen_steps = 0
    cfg.num_final_gen_steps = 0
    cfg.print_every = 1
    cfg.seed = seed
    cfg.log_dir = log_dir
    cfg.overwrite_existing_exp = True
    cfg.task.docking.protein_path = str(pocket_dir / target / "protein.pdb")
    cfg.task.docking.ref_ligand_path = str(pocket_dir / target / "ligand.mol2")
    cfg.task.docking.redocking = True
    cfg.semlaflow.ckpt_path = str(C / "weights/stateflow_main_ep004.ckpt")
    cfg.semlaflow.num_inference_steps = 80
    cfg.algo.num_from_policy = num_states
    # the engine is only used for its block table / exact pass here
    cfg.lane_a.enable = True
    cfg.lane_a.normalizer = "pan"
    cfg.lane_a.behavior = "subsample"
    cfg.lane_a.diag_every = 0
    return VinaMOOTrainer_laneA(cfg)


@torch.no_grad()
def exact_full_pass(engine, model, emb, protocol_mask, level_state, topk=100):
    """Exact log Z, M_eff and the top-k actions, by enumerating the whole legal set."""
    uniq, inverse = torch.unique(protocol_mask, dim=0, return_inverse=True)
    n = emb.shape[0]
    dev = emb.device
    out = {
        "log_Z": torch.zeros(n, device=dev),
        "log_meff": torch.zeros(n, device=dev),
        "n_legal": torch.zeros(n, device=dev),
        "top_logit": torch.zeros((n, topk), device=dev),
        "top_glob": torch.zeros((n, topk), dtype=torch.long, device=dev),
        "top_proto": torch.zeros((n, topk), dtype=torch.long, device=dev),
        "n_entries": torch.zeros(n, device=dev),
    }
    model.eval()
    for g in range(uniq.shape[0]):
        rows = torch.nonzero(inverse == g, as_tuple=False).view(-1)
        plan = engine._plan(uniq[g], int(rows.numel()))
        u = engine._u_of_group(model, emb[rows], plan.protocols)
        thr = 1.0 - level_state.view(-1)[rows]
        m = len(rows)
        run_max = torch.full((m,), float("-inf"), device=dev)
        s1 = torch.zeros(m, device=dev)
        s2 = torch.zeros(m, device=dev)
        nleg = torch.zeros(m, device=dev)
        t_val = torch.full((m, topk), float("-inf"), device=dev)
        t_glob = torch.zeros((m, topk), dtype=torch.long, device=dev)
        t_proto = torch.zeros((m, topk), dtype=torch.long, device=dev)
        for ch in plan.chunks:
            legal = engine.table.level[ch.lo : ch.hi].unsqueeze(0) < thr.unsqueeze(1)
            nleg += legal.sum(1)
            if not bool(legal.any()):
                continue
            parts = []
            for a0 in range(ch.lo, ch.hi, 1 << 16):
                b0 = min(a0 + (1 << 16), ch.hi)
                typ, desc, fp = engine.table.features_slice(a0, b0)
                parts.append(u[:, ch.slot] @ model.emb_block((typ, desc, fp)).T)
            lg = (torch.cat(parts, 1) if len(parts) > 1 else parts[0]).masked_fill(~legal, float("-inf"))
            new_max = torch.maximum(run_max, lg.max(dim=1).values)
            sc = torch.nan_to_num((run_max - new_max).exp(), nan=0.0, posinf=0.0, neginf=0.0)
            sh = (lg - new_max.unsqueeze(1)).exp()
            s1 = s1 * sc + sh.sum(1)
            s2 = s2 * sc * sc + (sh * sh).sum(1)
            run_max = new_max
            k = min(topk, lg.shape[1])
            v, i = lg.topk(k, dim=1)
            cv = torch.cat([t_val, v], 1)
            cg = torch.cat([t_glob, i + ch.lo], 1)
            cp = torch.cat([t_proto, torch.full_like(i, ch.proto)], 1)
            v2, i2 = cv.topk(topk, dim=1)
            t_val, t_glob, t_proto = v2, cg.gather(1, i2), cp.gather(1, i2)
        log_Z = run_max + s1.clamp_min(1e-300).log()
        out["log_Z"][rows] = log_Z
        out["log_meff"][rows] = 2 * log_Z - (2 * run_max + s2.clamp_min(1e-300).log())
        out["n_legal"][rows] = nleg
        out["top_logit"][rows] = t_val
        out["top_glob"][rows] = t_glob
        out["top_proto"][rows] = t_proto
        out["n_entries"][rows] = float(plan.n_entries)
    return out


@torch.no_grad()
def baseline_replicas(trainer, graphs, emb, actions, exact, reps: int):
    """Run the REAL baseline categorical `reps` times and collect its log Z_hat."""
    from rxnflow.policy.action_categorical import RxnActionCategorical

    model = trainer.model
    masks = list(torch.unbind(graphs.protocol_mask, dim=1))
    n = emb.shape[0]
    logZ = torch.zeros((reps, n), device=emb.device)
    in_top1 = torch.zeros((reps, n), device=emb.device)
    in_top10 = torch.zeros((reps, n), device=emb.device)
    for r in range(reps):
        cat = RxnActionCategorical(graphs, emb, action_masks=masks, model=model)
        lg = cat.weighted_logits
        maxl = cat._compute_batchwise_max(lg).values
        ex = [(i - maxl.unsqueeze(1)).exp().clamp(cat._epsilon) for i in lg]
        logZ[r] = sum(i.sum(1) for i in ex).log() + maxl
        for i in range(n):
            hits = []
            for j in range(10):
                p = int(exact["top_proto"][i, j])
                gidx = int(exact["top_glob"][i, j])
                ti = int(trainer.pan_engine.table.type_id[gidx])
                bt = trainer.pan_engine.table.block_types[ti]
                loc = gidx - int(trainer.pan_engine.table.type_offset_cpu[ti])
                sub = cat.subsamples[p].get(bt)
                hits.append(bool(sub is not None and bool((sub == loc).any())))
            in_top1[r, i] = float(hits[0])
            in_top10[r, i] = float(any(hits))
        del cat
    return logZ, in_top1, in_top10


@torch.no_grad()
def pan_replicas(engine, model, emb, protocol_mask, level_state, K: int, reps: int):
    out = []
    for _ in range(reps):
        pan = engine.evaluate(model, emb, protocol_mask, level_state, n_beh=0, n_is=K)
        l = engine.logits_of(model, pan.u_sel, pan.glob)
        out.append(torch.logsumexp(l - pan.log_q, dim=1) - math.log(K))
    return torch.stack(out)


# ---------------------------------------------------------------------------- sub-sample study
def _group_rows(protocol_mask):
    uniq, inverse = torch.unique(protocol_mask, dim=0, return_inverse=True)
    return uniq, inverse


@torch.no_grad()
def subsample_replicas(engine, model, emb, protocol_mask, level_state, query_proto, query_glob, reps, seed=0):
    """Draw `reps` baseline uniform sub-samples S and return, per replica and state:

      log_Zhat[r, s] = log sum_{b in S} exp(l_b + log 1/rho_t)   (the baseline ESTIMATOR)
      log_ZS  [r, s] = log sum_{b in S} exp(l_b)                 (the baseline BEHAVIOUR normaliser)
      inS     [r, s, q]                                          (membership of the query actions)

    Logits come from the embedding cache (refreshed from `model`, so l_c == l_theta up to the
    cache dtype), which makes 200 replicas affordable; `subsample_replicas` draws S with the very
    same `SubsamplingPolicy` object the baseline uses.
    """
    from rxnflow.utils.misc import get_worker_env

    sampler = get_worker_env("action_subsampler")
    table = engine.table
    dev = engine.device
    N = table.num_blocks
    n = emb.shape[0]
    nq = query_glob.shape[1]
    uniq, inverse = _group_rows(protocol_mask)
    log_Zhat = torch.full((reps, n), float("-inf"), device=dev)
    log_ZS = torch.full((reps, n), float("-inf"), device=dev)
    inS = torch.zeros((reps, n, nq), dtype=torch.bool, device=dev)
    thr_all = 1.0 - level_state.view(-1)
    for g in range(uniq.shape[0]):
        rows = torch.nonzero(inverse == g, as_tuple=False).view(-1)
        plan = engine._plan(uniq[g], int(rows.numel()))
        u = engine._u_of_group(model, emb[rows], plan.protocols)
        thr = thr_all[rows]
        qk = (query_proto[rows] * N + query_glob[rows]).view(len(rows), nq)
        for r in range(reps):
            keys, lse_w, lse_p = [], [], []
            for slot, p in enumerate(plan.protocols):
                idx_list, w_list = [], []
                for bt_idx, ti, start, length in table.segments[p]:
                    sub, ratio = sampler.sampling(table.block_types[ti])
                    idx_list.append(sub.to(dev) + start)
                    w_list.append(torch.full((len(sub),), -math.log(ratio), device=dev))
                idx = torch.cat(idx_list)
                w = torch.cat(w_list)
                keys.append(p * N + idx)
                e = engine.cache.emb[idx].to(u.dtype)
                lg = u[:, slot] @ e.T  # [nrows, |S_p|]
                legal = table.level[idx].unsqueeze(0) < thr.unsqueeze(1)
                lse_p.append(lg.masked_fill(~legal, float("-inf")))
                lse_w.append((lg + w.unsqueeze(0)).masked_fill(~legal, float("-inf")))
            allk = torch.sort(torch.cat(keys)).values
            log_ZS[r, rows] = torch.logsumexp(torch.cat(lse_p, dim=1), dim=1)
            log_Zhat[r, rows] = torch.logsumexp(torch.cat(lse_w, dim=1), dim=1)
            pos = torch.searchsorted(allk, qk.reshape(-1)).clamp(max=allk.numel() - 1)
            inS[r, rows] = (allk[pos] == qk.reshape(-1)).view(len(rows), nq)
    return log_Zhat, log_ZS, inS


@torch.no_grad()
def marginal_distortion(engine, model, emb, protocol_mask, level_state, exact, reps, n_mc, seed=0):
    """TV(P_marg, P_full) and P_marg/P_full for the top-1/10/100 actions.

    P_marg(a) = E_S[ 1{a in S} exp(l_a) / sum_{b in S} exp(l_b) ]  (the baseline's marginal
    behaviour: it draws ONE sub-sample S per step and then takes a plain softmax over it,
    because `algo.action_subsampling.importance_temp` is 0 in the paper's configuration).

    Writing  g(a) = P_marg(a) / P_full(a) = E_S[ 1{a in S} * Z / Z_S ], we have
        TV = 1/2 sum_a |P_marg(a) - P_full(a)| = 1/2 * E_{a ~ P_full} |g(a) - 1|,
    which is estimated with `n_mc` actions drawn exactly from P_full.
    """
    lam = engine.cfg.defensive_mix
    engine.cfg.defensive_mix = 0.0  # draw from q_c == P_full (fresh cache), not from the mixture
    try:
        pan = engine.evaluate(model, emb, protocol_mask, level_state, n_beh=0, n_is=n_mc)
    finally:
        engine.cfg.defensive_mix = lam
    n = emb.shape[0]
    topk = exact["top_glob"].shape[1]
    q_proto = torch.cat([pan.proto, exact["top_proto"]], dim=1)
    q_glob = torch.cat([pan.glob, exact["top_glob"]], dim=1)
    _, log_ZS, inS = subsample_replicas(
        engine, model, emb, protocol_mask, level_state, q_proto, q_glob, reps, seed
    )
    # g(a) = mean_r 1{a in S_r} * Z / Z_{S_r}
    w = (exact["log_Z"].unsqueeze(0) - log_ZS).exp()  # [reps, n]
    g = (inS.float() * w.unsqueeze(-1)).mean(0)  # [n, n_mc + topk]
    g_mc = g[:, :n_mc]
    g_top = g[:, n_mc:]
    p_top = (exact["top_logit"] - exact["log_Z"].unsqueeze(1)).exp()
    out = {
        "tv_mean": float((0.5 * (g_mc - 1.0).abs().mean(1)).mean()),
        "tv_max": float((0.5 * (g_mc - 1.0).abs().mean(1)).max()),
        "ratio_top1": float(g_top[:, 0].mean()),
        "ratio_top10": float(g_top[:, :10].mean()),
        "ratio_top100": float(g_top.mean()),
        "massratio_top10": float(
            ((p_top[:, :10] * g_top[:, :10]).sum(1) / p_top[:, :10].sum(1)).mean()
        ),
        "massratio_top100": float(((p_top * g_top).sum(1) / p_top.sum(1)).mean()),
        "logZS_sd_mean": float(log_ZS.std(dim=0).mean()),
        "reps": reps,
        "n_mc": n_mc,
    }
    per_state = {"tv": (0.5 * (g_mc - 1.0).abs().mean(1)), "ratio_top1": g_top[:, 0]}
    return out, per_state


def summarize(tag, exact, logZ_base, logZ_pan, in_top1, in_top10, action_logit, betas):
    z = exact["log_Z"]
    res = {
        "tag": tag,
        "n_states": int(z.numel()),
        "n_entries_mean": float(exact["n_entries"].mean()),
        "n_legal_mean": float(exact["n_legal"].mean()),
        "log_Z_mean": float(z.mean()),
        "M_eff_median": float(exact["log_meff"].exp().median()),
        "M_eff_min": float(exact["log_meff"].exp().min()),
        "M_eff_max": float(exact["log_meff"].exp().max()),
        "top1_mass_mean": float((exact["top_logit"][:, 0] - z).exp().mean()),
        "top10_mass_mean": float((exact["top_logit"][:, :10] - z.unsqueeze(1)).exp().sum(1).mean()),
        "top100_mass_mean": float((exact["top_logit"] - z.unsqueeze(1)).exp().sum(1).mean()),
        "beta_mean": float(betas.mean()) if betas is not None else None,
        "beta_min": float(betas.min()) if betas is not None else None,
        "beta_max": float(betas.max()) if betas is not None else None,
    }
    for name, lz in (("baseline", logZ_base), ("pan", logZ_pan)):
        if lz is None:
            continue
        err = lz - z.unsqueeze(0)
        res[f"{name}_bias_mean"] = float(err.mean())
        res[f"{name}_bias_median"] = float(err.median())
        res[f"{name}_sd_mean"] = float(err.std(dim=0).mean())
        res[f"{name}_sd_max"] = float(err.std(dim=0).max())
        res[f"{name}_abs_err_mean"] = float(err.abs().mean())
        res[f"{name}_p90_abs_err"] = float(err.abs().flatten().quantile(0.9))
        if action_logit is not None:
            res[f"{name}_clamp_frac"] = float(((action_logit.unsqueeze(0) - lz) > 0).float().mean())
    if in_top1 is not None:
        res["P_top1_in_subsample"] = float(in_top1.mean())
        res["P_any_top10_in_subsample"] = float(in_top10.mean())
    return res


# ---------------------------------------------------------------------------- behaviour A/B
@torch.no_grad()
def behaviour_experiment(trainer, n_traj: int, random_action_prob: float = 0.05, seed: int = 0):
    """Sample `n_traj` trajectories with the baseline sampler and with the full-space sampler
    from the SAME checkpoint (no docking, no training) and compare exploration."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from gflownet.utils.misc import set_worker_rng_seed

    engine = trainer.pan_engine
    dev = trainer.device
    out = {}
    for mode in ("subsample", "full"):
        engine.cfg.behavior = mode
        set_worker_rng_seed(seed)
        torch.manual_seed(seed + 1)
        engine.gen.manual_seed(seed + 2)
        cond = trainer.task.sample_conditional_information(n_traj, 0)
        t0 = time.time()
        trajs = trainer.algo.create_training_data_from_own_samples(
            trainer.sampling_model, n_traj, cond["encoding"].to(dev), random_action_prob=random_action_prob
        )
        first, mols, lp, lens = [], [], [], []
        for tj in trajs:
            if not tj["is_valid"]:
                continue
            a0 = tj["traj"][0][1]
            first.append((a0.block_type, a0.block_idx))
            mols.append(tj["result"].mol)
            lp.append(float(tj["fwd_logprob"]))
            lens.append(len(tj["traj"]))
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(Chem.MolToSmiles(m)), 2, 1024) for m in mols]
        sims = []
        for i in range(len(fps)):
            if i:
                sims += DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        smis = {Chem.MolToSmiles(m) for m in mols}
        out[mode] = {
            "n_valid": len(mols),
            "n_distinct_first_blocks": len(set(first)),
            "n_distinct_molecules": len(smis),
            "mean_pairwise_tanimoto": float(sum(sims) / max(len(sims), 1)),
            "max_pairwise_tanimoto": float(max(sims)) if sims else 0.0,
            "mean_logP_full_traj": float(sum(lp) / max(len(lp), 1)),
            "mean_traj_len": float(sum(lens) / max(len(lens), 1)),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[step0] behaviour {mode}: {out[mode]}", flush=True)
    engine.cfg.behavior = "subsample"
    return out


def stratify(name, values, beta, pref0, bins=((0, 16), (16, 32), (32, 48), (48, 65))):
    """Mean of `values` per beta bin and per Vina-preference half."""
    out = {}
    for lo, hi in bins:
        m = (beta >= lo) & (beta < hi)
        out[f"{name}_beta[{lo},{hi})"] = float(values[m].mean()) if bool(m.any()) else None
        out[f"n_beta[{lo},{hi})"] = int(m.sum())
    hi_pref = pref0 >= 0.5
    out[f"{name}_pref_vina>=0.5"] = float(values[hi_pref].mean()) if bool(hi_pref.any()) else None
    out[f"{name}_pref_vina<0.5"] = float(values[~hi_pref].mean()) if bool((~hi_pref).any()) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="ADRB2")
    ap.add_argument("--ckpt", required=True, help="model_state_<it>.pt of a diagnostic run")
    ap.add_argument("--env_dir", default=str(C / "data/envs/catalog"))
    ap.add_argument("--out", default="/tmp/step0")
    ap.add_argument("--num_states", type=int, default=64, help="trajectories sampled to harvest states")
    ap.add_argument("--analyse", type=int, default=48, help="states analysed per step")
    ap.add_argument("--reps", type=int, default=48, help="REAL RxnActionCategorical replicas (log Z_hat)")
    ap.add_argument("--marg_reps", type=int, default=200, help="sub-sample replicas for the marginal study")
    ap.add_argument("--mc_actions", type=int, default=2048, help="actions ~ P_full used for the TV estimate")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--behaviour", type=int, default=1)
    a = ap.parse_args()

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"step0_{a.target}"
    pocket = tmp / "pocket"
    (pocket / a.target).mkdir(parents=True, exist_ok=True)
    for f in ("protein.pdb", "ligand.mol2"):
        dst = pocket / a.target / f
        if not dst.exists():
            dst.write_bytes((C / "data/LIT-PCBA" / a.target / f).read_bytes())

    t0 = time.time()
    trainer = build_trainer(a.target, pocket, a.env_dir, str(tmp / "log"), a.num_states, a.seed)
    print(f"[step0] trainer ready in {time.time() - t0:.0f}s", flush=True)
    state = torch.load(a.ckpt, map_location="cpu")
    trainer.model.load_state_dict(state["models_state_dict"][0])
    trainer.sampling_model.load_state_dict(state["sampling_model_state_dict"][0])
    trainer.model.to(trainer.device)
    trainer.sampling_model.to(trainer.device)
    engine = trainer.pan_engine
    print(f"[step0] checkpoint {a.ckpt} loaded (step {state.get('step')})", flush=True)

    results = {"ckpt": a.ckpt, "target": a.target, "train_step": int(state.get("step", -1)), "steps": []}

    # --- 3) behaviour A/B (uses the sampling model; must run before the cache is pinned to `model`)
    if a.behaviour:
        engine.cache.refresh(trainer.sampling_model, 0)
        results["behaviour"] = behaviour_experiment(trainer, a.num_states, 0.05, a.seed)

    # --- generate the states the policy actually visits (no docking) -----------------
    engine.cfg.behavior = "subsample"
    engine.cache.refresh(trainer.sampling_model, 0)
    cond = trainer.task.sample_conditional_information(a.num_states, 0)
    dev = trainer.device
    t1 = time.time()
    trajs = trainer.algo.create_training_data_from_own_samples(
        trainer.sampling_model, a.num_states, cond["encoding"].to(dev), random_action_prob=0.0
    )
    print(f"[step0] sampled {len(trajs)} trajectories in {time.time() - t1:.0f}s", flush=True)
    batch = trainer.algo.construct_batch(trajs, cond["encoding"], torch.zeros(len(trajs))).to(dev)
    traj_lens = batch.traj_lens
    step_of = torch.cat([torch.arange(int(l), device=dev) for l in traj_lens])
    batch_idx = torch.arange(len(trajs), device=dev).repeat_interleave(traj_lens.to(dev))
    beta_all = cond["beta"].view(-1).to(dev)[batch_idx]
    pref_all = cond["preferences"].to(dev)[batch_idx][:, 0]

    # the exact pass / PAN must be measured against the TRAINING model (that is what the TB
    # loss re-evaluates), so refresh the cache from it.
    engine.cache.refresh(trainer.model, 0)
    fwd_cat, _, _ = trainer.algo.estimate_policy(trainer.model, batch, batch.cond_info, batch_idx)
    emb_all = fwd_cat.emb.detach()
    del fwd_cat

    for step in (0, 1):
        rows = torch.nonzero(step_of == step, as_tuple=False).view(-1)[: a.analyse]
        if rows.numel() == 0:
            continue
        emb, mask, lvl = emb_all[rows], batch.protocol_mask[rows], batch.level[rows]
        acts = batch.actions.to(dev)[rows]
        beta, pref0 = beta_all[rows], pref_all[rows]
        t2 = time.time()
        exact = exact_full_pass(engine, trainer.model, emb, mask, lvl)
        print(f"[step0] step {step}: exact pass, {float(exact['n_entries'].max()):.0f} entries x "
              f"{rows.numel()} states in {time.time() - t2:.1f}s", flush=True)
        glob_a = engine.table.seg_start[acts[:, 0], acts[:, 1]] + acts[:, 2]
        u_a = engine.u_of_actions(trainer.model, emb, acts[:, 0])
        l_a = engine.logits_of_actions(trainer.model, u_a, glob_a).detach()

        t3 = time.time()
        lz_b, it1, it10 = baseline_replicas(trainer, type("G", (), {
            "protocol_mask": mask, "level": lvl, "num_graphs": int(rows.numel())})(), emb, acts, exact, a.reps)
        print(f"[step0] step {step}: {a.reps} REAL baseline replicas in {time.time() - t3:.1f}s", flush=True)
        lz_p = pan_replicas(engine, trainer.model, emb, mask, lvl, a.K, min(a.reps, 24))
        res = summarize(f"step{step}", exact, lz_b, lz_p, it1, it10, l_a, beta)

        t4 = time.time()
        marg, per_state = marginal_distortion(
            engine, trainer.model, emb, mask, lvl, exact, a.marg_reps, a.mc_actions, a.seed
        )
        print(f"[step0] step {step}: marginal study ({a.marg_reps} sub-samples) in {time.time() - t4:.1f}s",
              flush=True)
        res["marginal"] = marg
        res["strata"] = {}
        res["strata"].update(stratify("M_eff", exact["log_meff"].exp(), beta, pref0))
        res["strata"].update(stratify("logZhat_err_sd", (lz_b - exact["log_Z"].unsqueeze(0)).std(0), beta, pref0))
        res["strata"].update(stratify("TV", per_state["tv"], beta, pref0))
        res["strata"].update(stratify("ratio_top1", per_state["ratio_top1"], beta, pref0))
        res["strata"].update(stratify("top1_mass", (exact["top_logit"][:, 0] - exact["log_Z"]).exp(), beta, pref0))
        print(json.dumps(res, indent=2), flush=True)
        results["steps"].append(res)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"step0_{a.target}_{Path(a.ckpt).stem}.json"
    (out / name).write_text(json.dumps(results, indent=2))
    print(f"[step0] wrote {out / name}", flush=True)


if __name__ == "__main__":
    main()
