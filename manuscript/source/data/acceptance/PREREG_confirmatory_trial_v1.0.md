# Confirmatory comparison of ARRE operating points on held-out pockets - analysis plan v1.0

Status: **FROZEN at 2026-11-02 16:15:31 CST** by SHA-256 (this file, the analysis script, the launcher, the pocket list and the data/code manifests of Appendix B; the hashes are written to
`status/prereg_hashes.txt` before the first confirmatory run is submitted). Everything changed afterwards is a recorded deviation (Appendix C), never a silent edit.
The single-pocket pilot on a reserve pocket (section 6) is not confirmatory and decides nothing.

## 1. Questions and claims being tested

* **C2 - position under constraints.** The frozen-policy ARRE variant FZ (`r4CE0`) docks better than SynGBO at equal molecule size (P1) and is not inferior to SynGBO in the
  share of selected molecules solved by an independent retrosynthesis planner (P2, margin 5 percentage points).
* **C3 - generalisation.** The direction, and at least half of the magnitude, of the development-panel effects of F (`r2CE4L4`) and FZ against CGFlow-B, SynGA and SynGBO are retained on
  pockets that took no part in any design, tuning or selection decision.
* **Not tested here and not to be claimed from this trial:** binding affinity, hit rate, or that the compounds are better binders. All statements are about optimisation under the
  Uni-Dock/Vina oracle and the independent planner.

## 2. Arms (registered before the first run; no tuning on the new pockets)

| arm | tag (results/) | code | recipe (verbatim from the ledger entries of the stored development runs) |
|---|---|---|---|
| B (base generator) | `x4nw2B` | `repo/cgflow_baseline` | no lane arguments (ledger `base_ep30`) |
| F (docking-priority ARRE) | `x4nw2F` | `frozen/round2_integ_4` | `--trainer cgflow.lanes.combo:VinaMOOTrainer_ARRE --set rfg_spp.enable=true --set rfg_spp.group_norm=true --set rfg_spp.syn_prior=data/syn_prior_v1.frozen.json --set rfg_spp.syn_lookahead=true --set rhc.enable=true --set rfg_spp.syn_kappa=4.0` (ledger `r2CE4L4`) |
| FZ (balanced ARRE) | `x4nw2FZ` | `frozen/round2_integ_4` | F plus `--set opt.learning_rate=0.0 --set algo.tb.Z_learning_rate=0.0` (ledger `r4CE0`) |
| SynGA | `x4nw2ga64k` | `slurm/synga_launch.py --method ga` | ledger `x4synga` |
| SynGBO | `x4nw2gbo64k` | `slurm/synga_launch.py --method gbo` | ledger `x4syngbo` |

Common to all arms: pose-model checkpoint `weights/stateflow_main_ep029.ckpt`; 64,000 newly docked molecules per run (1,000 iterations x 64; SynGA/SynGBO: 64,000 oracle calls);
seeds 0, 1, 2; Uni-Dock `balance` re-docking (conda package unidock 1.1.2, whose binary reports "Uni-Dock v1.1.1"; Appendix B records both and the binary hash), box 20 A around the reference ligand, the same receptor preparation and box rules as the development panel; environment
catalog `data/envs/catalog`; top-100 diverse molecules with QED > 0.5 and Tanimoto < 0.5 (paper protocol, `eval/extract_topk.py`). Hashes of checkpoint, code trees, building-block
catalog, stock and pocket files: Appendix B. The generic-search and size-cap control arms run on other pockets and are not part of this trial.

## 3. Pocket eligibility and selection (done before any arm was run on a new pocket)

1. **Candidate pool.** The test list of the CrossDocked2020 name-based split `data/crossdock/split_by_name.pt` (the pose model was trained on the training list of the same split; pockets of the
   test list are absent from the training list by name, homology separation beyond the name is not verified here, see item 4). Test list 100 pockets; 85 pass the eligibility rules of item 2 (the first three classes); 15 are already used in `data/NEWT`; 70 eligible and unused pockets remain, 52 after the family and paralog rules (`newt2_draw.json`).
2. **Fixed rules** (written in `advancement/newt2_draw.py` before the draw, no result of any kind used): LIT-PCBA protein families, proteins that repeat by mnemonic, and ligands with
   fewer than 10 heavy atoms are excluded; pockets already in `data/NEWT` are excluded; protein families that already have a representative among the 45 used targets (protein kinases,
   nuclear receptors, class-A GPCRs, bromodomains, NOS isoforms; list in `newt2_draw.json`) are excluded; at most one pocket per paralog group (tankyrases, phospholipase A2,
   cytochrome P450, sirtuin-like deacylases).
3. **Draw.** The remaining names are sorted alphabetically and permuted with `numpy.random.RandomState(20261102)`; the first 20 are the primary pockets and the next 10 the reserve.
4. **Limitation, stated.** No sequence-clustering tool is available offline, so the 30%-identity separation from the 45 used targets is NOT verified computationally; it rests on the
   curated family exclusions above and on the name-based split. A reviewer-facing check with MMseqs2/CD-HIT will be added as a post-hoc table, not as a selection rule.
5. **Receptor preparation.** All 30 drawn pockets passed receptor preparation and a 3-iteration docking smoke run of the base generator on 2026-11-02 (30/30), so **no pocket was replaced**;
   the replacement rule (next reserve pocket, before any arm is run) would have been the only permitted exclusion.
6. The primary list (Appendix A) is hashed with this file. One reserve pocket (POL_FOAMV) is used for the pilot and is not analysed.

## 4. Endpoints and decision rules

Unit of analysis: the pocket. Seeds are computational replicates and are averaged inside a pocket (a pocket-arm cell needs >= 2 complete seeds). The analysis is produced by the hashed
script `advancement/trial_analysis.py` without manual intervention; any additional analysis is labelled exploratory.

### P1 - docking at equal size (co-primary, superiority, directional)
* Metric: `M1_adj`, definition R1 of `docs/SIZE_MATCHED_COMPARISON.md`: per pocket, ordinary least squares of Vina on heavy-atom count fitted on B's pooled top-100 molecules (all complete
  seeds); the adjusted score of a run is the mean residual of its top-100 (negative = docks better than a B molecule of the same size). The script reproduces the published development
  values (F-B -1.110, F-SynGBO -0.550, F-SynGA -0.963, F-FZ -0.385 kcal/mol, 15/15 targets each).
* Contrasts: F-B, F-SynGBO, FZ-SynGBO, FZ-B, F-SynGA, FZ-SynGA. Two-sided paired t over pockets; Holm over the six; 95% intervals reported.
* **P1 succeeds for FZ-SynGBO** if the mean is negative and the Holm-adjusted p < 0.05. The smallest effect of interest (-0.15 kcal/mol, about one third of the Uni-Dock repeat noise of a
  single molecule) is reported against the 95% interval but is not a gate: on the development panel FZ-SynGBO is -0.165 (SD of the pocket-level differences 0.21), so a point-estimate gate at
  -0.15 would pass only about 63% of the time even if the development effect transferred perfectly (power of the superiority test at N = 20: 92% for a true -0.165, 70% for -0.12, 38% for -0.08).
* Sensitivity (descriptive): raw M1, ANCOVA with arm indicator, 3-atom-bin matching, R2 (adds logP, TPSA, rotatable bonds), worst-case imputation for missing cells.

### P2 - independently planned solved fraction (co-primary)
* Metric: share of the 100 selected molecules solved by Retro* (Syntheseus commit `2776bdfa8469783be650cc9854c1579eb5f59533`) with MHNreact (checkpoint SHA-256
  `82ec9e2fbeb41e0a9070341d5d0fdab8ea866341da9cdd5a5fadd2223f422945`), at most 500 expansions and 300 s per molecule, stock = InChIKey lookup in `data/aizynth_stock/building_blocks.inchikeys`,
  `PYTHONHASHSEED=0`, speed-only caches as in `retro_common.install_speedups(2)` (pipeline `docs/arre_v2_gap_experiments_20261101/retro_*`, input built with `retro_prepare_extra.py`).
* **FZ vs SynGBO: non-inferiority**, margin 5 percentage points: the lower limit of the two-sided 90% t interval of the pocket-level paired difference (one-sided alpha 0.05) must exceed -5.
* FZ vs B: superiority, two-sided paired t. F vs B and F vs SynGBO are reported descriptively.
* **C2 is declared supported only if P1(FZ-SynGBO) AND P2(FZ vs SynGBO non-inferiority) both succeed** (intersection-union test, no alpha split).

### C3 - retention
For every P1 contrast the trial mean is compared with the same contrast on the development panel (same script and definition). **Retained** = same sign and trial/development ratio >= 0.5
(point estimate); the pocket-bootstrap 95% interval of the ratio is reported. If a development effect is shrunk by more than half, the paper reports the shrinkage and does not describe the
development effect size as typical.

### Secondary (descriptive, no claims without a pre-listed correction)
AiZynthFinder solved fraction (iteration limit 100, 120 s) on the same molecules, ligand efficiency, heavy atoms, dual success at B's best docking quartile, oracle calls needed to reach B's final
level, PoseBusters pass rates and GNINA/RTMScore rescoring of the selected molecules, a prediction interval for a new pocket.

## 5. Sample size

* P1 is not power-limited (SD of the pocket-level F-SynGBO difference 0.165 kcal/mol on the development panel: 6 pockets give 90% power for 0.30 kcal/mol).
* P2 non-inferiority, noncentral t, one-sided alpha 0.05, margin 5 pp, **N = 20 primary pockets**:

  | pocket SD (pp) | true difference 0 | -1 pp | -2 pp | -3 pp |
  |---|---|---|---|---|
  | 6.3 (F-SynGBO on the development panel) | 96% | 86% | 66% | 39% |
  | 7.8 (F-B on the development panel) | 87% | 71% | 51% | 29% |

  If the true FZ-SynGBO difference is as large as -2 pp, N = 20 gives only 51-66% power; the answer then comes as an interval estimate. A 30-pocket extension (the 10 reserve pockets,
  about 2,500 additional GPU hours) would raise these to 66-82% but is not part of this plan. A pre-planned pooled analysis of the trial and the 15 development pockets (FZ was not run on the
  external panels) is reported separately and labelled supplementary.

## 6. Conduct

* **Pilot.** One reserve pocket (POL_FOAMV), seed 0, 100 iterations (6,400 molecules) per arm (SynGA/SynGBO: 6,400 calls), tags `z4nw2p*`, was run to check the chain end to end (generation,
  extraction, AiZynthFinder array, SynGA/SynGBO registration, independent-planner input). It informs no arm decision and is not analysed.
* **Failures.** Node loss or pre-emption: the same run resumes from its last checkpoint (`rs_state`, every 20 iterations). A software crash is re-run once from scratch with the same seed and
  configuration; a second failure is recorded as missing (never dropped) and the worst-case imputation sensitivity of section 4 is added. No run is excluded because of its result.
* All arms of a pocket-seed are submitted in the same working session. To limit the load on the shared cluster the SynGA/SynGBO runs of pockets 11-20 may be submitted up to 24 h after those of pockets
  1-10 and the array throttles of the B/F/FZ arms may be raised later; no result of any arm is inspected in between. The watchdog (`slurm/watchdog_v2.py`) may re-submit lost tasks with the recorded recipe; it changes nothing else.
* **No peeking.** Health monitoring (job states, error logs, iteration counters, disk) is allowed; docking scores, selected molecules, AiZynthFinder or independent-planner results of trial runs are not inspected, and
  no figure or table is produced, before every run has finished (or been declared missing) and the hashed analysis script is run on the complete set.
* **Noise floor.** Uni-Dock is not bit-deterministic (mean absolute difference 0.476 kcal/mol between repeats of the same molecule); no statement is made about single molecules.
* **Wall-clock note.** Per-iteration cost differs between arms (ARRE F is 1.16x B per 1000 iterations on the five B-long targets, range 0.99-1.32, and about 1.24x in the median over the development panel); the claim tested
  here is at equal oracle calls. On the development panel the advantage over B at equal wall-clock is -1.06 kcal/mol against -1.15 at equal calls (`advancement/walltime_equal.py`, 5 targets x 2 seeds).

## 7. What each outcome allows the paper to say

| outcome | wording |
|---|---|
| P1 and P2 succeed for FZ-SynGBO | "A balanced operating point of ARRE docks better than the strongest search baseline at equal molecule size without loss of independently planned solved fraction (non-inferiority margin 5 pp)." |
| P1 succeeds, P2 fails | "ARRE is a docking-priority method; the synthesisability-first baseline remains preferable when route feasibility is the main criterion." |
| P1 fails, P2 succeeds | "At equal size ARRE offers no docking advantage over SynGBO on new pockets; its advantage is over the base generator." |
| both fail | restrict the claims to the base-generator comparison and to search efficiency on the development and external panels |
| F-B and FZ-B retain direction but < half magnitude | report the shrinkage; do not describe the development effect size as typical |
| P2 interval inconclusive (neither non-inferior nor inferior) | report the interval and the pooled supplementary analysis; make no non-inferiority statement |

## Appendix A - pocket list (generated from `advancement/newt2_draw.json`)
| # | role | pocket (CrossDocked mnemonic) | ligand heavy atoms | protein.pdb sha256 (first 16) | ligand.mol2 sha256 (first 16) |
|---|---|---|---|---|---|
| 1 | primary | BACE2_HUMAN | 38 | 3869a3438d73bd6b | 1d568d6b91d04448 |
| 2 | primary | DPO4_SULSO | 18 | 46ed96681402eb42 | f882f83416c9eb3f |
| 3 | primary | PYRE_BACAN | 22 | a983655e1b167253 | da563935b5d5028b |
| 4 | primary | COTA_BACSU | 16 | 8a8538dc6d53314c | cc70a08ca3a6f731 |
| 5 | primary | PHP_SULSO | 13 | b066fb93311d15ce | 6ad95c138c5e315d |
| 6 | primary | PAC_ECOLX | 13 | d31152e6eeacc083 | 1a35a95a834b230b |
| 7 | primary | BTRN_BACCI | 27 | dc7655e23ea722ab | d5fbf67699db1c9b |
| 8 | primary | GSTP1_HUMAN | 22 | 9606d66d77139767 | 05f471ac0ba85fac |
| 9 | primary | TNKS1_HUMAN | 42 | 466711b63e966f96 | 923081c010a0f18c |
| 10 | primary | PPIA_HUMAN | 18 | 8c109b076fc16a36 | 33f55f0d068b9969 |
| 11 | primary | XANLY_BACGL | 17 | 1493ca0e4bb55599 | f831bc6ad61b66b0 |
| 12 | primary | BAPA_SPHXN | 25 | 758386579a575876 | 9ad5468d047a9d6e |
| 13 | primary | NQO1_HUMAN | 21 | 249424c6a64c952e | ac3c7ad565c7bebb |
| 14 | primary | LAT_MYCTU | 15 | 144d1d35910791b1 | f371c047e23eb98d |
| 15 | primary | IPMK_HUMAN | 27 | ad4d396b3b35bf07 | cef90058f0c8a5cc |
| 16 | primary | TIAM1_HUMAN | 16 | 7f99dd72c0577d17 | d7e28967db81fa18 |
| 17 | primary | PLCD1_RAT | 20 | 51ffe39bc1062ce5 | a51c48c30cdbfb85 |
| 18 | primary | UPPS_ECOLI | 23 | c042d450bcbd38c0 | 8533e5113f29b47f |
| 19 | primary | RG1_RAUSE | 37 | e4582d8c39bcd0c6 | 8fc732b73691b1b5 |
| 20 | primary | Y635_MYCTU | 19 | 1328c8ff4b3a82d7 | 6f706fec30f0020f |
| 21 | reserve | MENE_BACSU | 23 | bcbdddaec4b1451e | 049138f0f9c13177 |
| 22 | reserve | MCCF_ECOLX | 32 | 5c7377cf6eacb31e | ec08d1cb16566ef3 |
| 23 | reserve | F16P1_HUMAN | 21 | fe272d295514925b | a173f1e35098a0e1 |
| 24 | reserve | CHIB_SERMA | 14 | 368ceb8f916ff364 | 3fd96a20f58ff43c |
| 25 | reserve | BGAT_HUMAN | 26 | 7175140c1af70f4e | d5d8f89ed58e3937 |
| 26 | reserve | PA21B_PIG | 29 | f6edc91a435f7b40 | 145b02eede27abd9 |
| 27 | reserve | SQHC_ALIAD | 31 | a0f63052da26f8da | 7a254e0053d06532 |
| 28 | reserve | IMA1_HUMAN | 15 | 392b991ebc5dbe80 | e1a6c99b71b96be7 |
| 29 | reserve | LMBL1_HUMAN | 11 | 81631f4a2c274421 | b16dd457a702d295 |
| 30 | reserve | POL_FOAMV | 35 | 5c9eb75101d05a42 | bfdee4c867aa65f0 |

Primary pockets in analysis order (file `advancement/pockets_primary.txt`): BACE2_HUMAN DPO4_SULSO PYRE_BACAN COTA_BACSU PHP_SULSO PAC_ECOLX BTRN_BACCI GSTP1_HUMAN TNKS1_HUMAN PPIA_HUMAN XANLY_BACGL BAPA_SPHXN NQO1_HUMAN LAT_MYCTU IPMK_HUMAN TIAM1_HUMAN PLCD1_RAT UPPS_ECOLI RG1_RAUSE Y635_MYCTU


## Appendix B - environment, code and data manifests with SHA-256 (generated by `prereg/prereg_freeze.py`)
| item | value (sha256 unless stated) |
|---|---|
| pose checkpoint weights/stateflow_main_ep029.ckpt | `e51704fbb5b39aeb326238a08dc7d1d0bb8abd92a26539157a519a15f6118824` |
| route prior data/syn_prior_v1.frozen.json | `6818a6129237b1a4ed1d0c1df670293357ee0e001288b7ab78081d4feffde3c5` |
| AiZynth stock InChIKeys data/aizynth_stock/building_blocks.inchikeys | `369a722f765453d76127a8a0bf74ff3a92e2f388e986064733a082ed62f08dc7` |
| environment workflow data/envs/catalog/workflow.yaml | `9a4c5cf742bdd75e1f2d4b814dfb025fbfc17273eeb224ba2560ce0380757e8c` |
| block features data/envs/catalog/bb_feature.pt | `32b2a9df7f04f756de3216bd72a3d2ca97c1d8fbdb3bcb5a9d9bed8ca560bfc0` |
| block lists data/envs/catalog/blocks (485 files, tree hash) | `b794a0e43d95a26c7145c88a2b8d08ee8536587ad8e608327030350eeb1a168a` |
| code tree frozen/round2_integ_4/src (181 .py files, tree hash) | `1d60b0c58fc5b8a843f6c710c97537f669d4180a86e883e9fa03019a1d9d62bb` |
| code tree repo/cgflow_baseline/src (162 .py files, tree hash) | `480889f0cf9d6f0ac768d79ea3e0387ac8e442c4e3c3676bf6499b224ae38821` |
| script slurm/launch_tag4.sh | `9a9e3befd8424b2baab23852d66b4cc0a0063bc4a62e625a58cc55f0480fced0` |
| script slurm/run_one_opt4.sh | `1c85b56b9f512b9bc8a72d3c4ebbe0515118dc529ed8097db1bdc44f681e9bd1` |
| script slurm/opt_array4.sbatch | `04c3a8917aebebfbc7cc546566024ee9ba57e0e95354d1993cb98bc81dfb2493` |
| script slurm/aizynth_array4.sbatch | `fde81793e850e9826f169410c5027220e9cac2ec1c89fb0f42283e7e0a7d29b0` |
| script run/run_opt_rs.py | `67750d9f1f4522b5292940c06e89600824f9d8b6dd7a1d9e3cf791ecc3978010` |
| script eval/extract_topk.py | `15bcd3efc107002304772e20ec09adf9a59c6620f00498941026bc6225ed37b7` |
| script eval/extract_topk_calls.py | `f5ed4615abf9e5b8800a92afcc92910deeff64f3d6e3e76e10c5c11d14ef32d0` |
| script slurm/synga_launch.py | `997623a65ec3f0eb1120f5f8025805a30098ce44ba3676f56c7d99b23b1f693e` |
| script slurm/synga_run.sbatch | `b74bebea6c4cba22e0f37a818bb0c76fbbd549f25d5dc8bcbb864246f21fbd77` |
| script slurm/reg_submit.py | `27dca5f0464e3cd3b607e3c7c74a32ababc95919266e7406b9dfb96d6c9b4320` |
| script advancement/trial_analysis.py | `008c53885bee5bd8cd1f8f78c12d6793d4da69bba101e971746e9034b8c3ab41` |
| script advancement/trial_launch.sh | `079d3f75a6c660f3b6531ace84b93b06772514dc40a289b83d0d10f97c3a66d4` |
| script advancement/newt2_draw.py | `b34636352c7e90575f5589b59e79ffd24f15b7306869451674ff4f1fae5d8799` |
| script advancement/newt2_draw.json | `fec98ac6e7c59cf19d49e73c3697c6bbad04f79d2fdd8db4e6efda748f7c7aaa` |
| script advancement/newt2_pool.json | `1e074b10c0d615476d34af5837f19d1b04935a4269c4cecb1f323684da4de972` |
| script advancement/newt2_manifest.json | `488c3ab373324759b40b6e5c26d1ae2433dccbfbf13f719dd609e9ed7bff6b3d` |
| script retro_common.py | `59f98a461817d29c685a03f4154a38441de87de5841e05a1dead28b639007b71` |
| script retro_run_search.py | `754ba26ba8f0fa37f38195142a78934429c5e64ceff2fc54598ac58de18aab1e` |
| script retro_array.sbatch | `c9ab325ba0a77d7b854c1b7b4323bc2b979dac7662c5eec8fc94100d326f7cd6` |
| script retro_prepare_extra.py | `dfba2ddb0df6745e015811258af6ee00f3ffe25dea860e93b9e623272f7f494c` |
| script retro_analyze_extra.py | `b790ea8e194ff3f415c5c7ab6c7c5bceb436e0ae5677aad7b69db8408a9b8578` |
| environment cgflow pip freeze | `278f25fdcf50f910e8263a3d669bd6529d005848d9b6cd2fa360129177fc557a` |
| environment syntheseus_independent_20261025 pip freeze | `2a8335f5e8f6a7795ae793d5beb03de5d1748a7fc0fb3769b01dcd86cc36a2f1` |
| Uni-Dock binary self-reported version string (the conda package is 1.1.2, see next row) | `Uni-Dock v1.1.1` |
| Uni-Dock conda package (conda-meta) | `unidock-1.1.2-cuda120_h3031e94_0.json` |
| Uni-Dock binary envs/cgflow/bin/unidock | `dc72b78b509a1d79f22d4c3d268bb21952d2de7ed693bd3eb2ec21eec92d4e3f` |


## Appendix C - deviations log (append-only)
1. (before freeze) v0.1 -> v1.0: the P1 gate "FZ-SynGBO mean <= -0.15 kcal/mol" was replaced by "mean < 0 and Holm p < 0.05" with the -0.15 value reported as a reference, because the development
   estimate (-0.165, SD 0.21) makes a point-estimate gate fail in about 37% of trials even when the development effect transfers.
2. (before freeze) the 30%-identity clustering of v0.1 section 3 could not be done offline; replaced by the curated family exclusions and the paralog rule of `newt2_draw.py`; limitation stated in section 3.
3. (before freeze) v0.1 "arm labels masked as A-E" removed: the analysis is rule-based and produced by the hashed script, masking would add nothing.
