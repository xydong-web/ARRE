# Supplementary comparator study S1 - addendum to PREREG_confirmatory_trial v1.0

Frozen by SHA-256 before any run of this study is submitted (hash in `status/prereg_hashes.txt`). Written 2026-11-04 in response to the decision to drop SynGA from the featured comparisons and to add locally matched, published same-field methods.
The v1.0 confirmatory trial is unchanged: its frozen analysis (Holm over six contrasts, including the two SynGA contrasts) is run as written and its SynGA rows go to the supplement.

## 1. Status of the data at the time of freezing
Known and already used for planning: the development-panel results of CGFlow-B, ARRE (F), FZ, SynGA, SynGBO; the 5-pocket development runs of RxnFlow (`x4rx2dB/E/CE`: RxnFlow - CGFlow-B -0.055 [-0.240, +0.129]; F - RxnFlow -1.080 [-1.430, -0.730]; RxnFlow+ARRE guidance - RxnFlow -1.078 [-1.405, -0.751]).
NOT opened: any docking, AiZynthFinder or independent-planner result of the confirmatory trial `x4nw2*`. Contrasts of this study that involve `x4nw2*` arms (E5) are computed only after the frozen trial analysis has been run.

## 2. Inclusion rule for comparators (independent of outcomes)
A published method enters the study if it (i) appears in Table 1 of the manuscript as literature context, (ii) has public code, and (iii) can be attached to the shared Uni-Dock oracle and run at the shared budget within the engineering time we allow ourselves (stated per method below).
Methods that fail (iii) stay literature-only and are reported as "not reproduced locally"; no method is added or removed after seeing any of its results. Hyper-parameters: the defaults of the official repository at the cloned commit; no tuning of any comparator.

| id | method | panel | runs | note |
|---|---|---|---|---|
| E1 | RxnFlow (2D parent, `frozen/round4_2d`) and the same code with the ARRE recipe (CE) | the 10 development pockets not yet run: GBA IDH1 KAT2A MAPK1 MTORC1 OPRK1 PKM2 PPARG TP53 VDR | 2 x 10 x 3 seeds = 60 | shared action space; completes the 15-pocket development panel |
| E5 | same two arms | 20 primary held-out pockets of the trial (`data/NEWT2`) | 2 x 20 x 3 = 120 | tags `x4nw2rxB`, `x4nw2rxCE` |
| E2 | SynFlowNet (`repo/ext/synflownet`) with the shared docking oracle | 15 development pockets | 45 | own action space; equal-budget accounting required (section 4) |
| E3 | GraphGA (Jensen 2019 settings of the PMO benchmark: population 120, offspring 70, mutation 0.067) with the shared oracle | 15 development pockets | 45 | conventional optimiser without synthesis constraint |
| E6 | S3-GFN (`repo/ext/s3gfn`) | ADRB2 ALDH1 ESR_ago ESR_antago FEN1 | 15 | strongest published method on these five tasks; N = 5, descriptive |
| E4 | validity checks of E1-E3, E5, E6 (PoseBusters dock_fast, re-docking, cross-receptor where applicable, independent planner, AiZynthFinder) | same | - | existing scripts |

## 3. Shared protocol
Uni-Dock `balance`, 20 A box centred on the reference ligand, conformers rebuilt by ETKDG from the molecular graph; budget = 64,000 distinct molecules docked by the shared oracle (first 64,000 distinct docked molecules in call order); top-100 selection by the shared extractor (QED > 0.5, Tanimoto < 0.5, diverse); snapshots at 16,000/32,000/64,000; seeds 0, 1, 2; same building-block release for methods that use the shared action space. ARRE configurations are the frozen ones: F = `r2CE4L4` recipe (CGFlow), CE = `x4rx2dCE` recipe (RxnFlow).

## 4. Equal-budget gate for external methods
A run counts as budget-matched only if the shared oracle was called on >= 64,000 distinct molecules of that run and the 64,000-molecule snapshot can be extracted. Runs below the gate are reported as such, are not used in the matched contrasts and are listed. Each of E2, E3 and E6 first runs one pilot run on the reserve pocket POL_FOAMV (not analysed), then the runner script is hashed (`status/prereg_hashes.txt`) before production runs are submitted.

## 5. Endpoints and decision rules
Unit = pocket (seeds averaged inside a pocket, >= 2 complete seeds per cell), two-sided paired t, 95% intervals. Primary metric: Vina of the diverse top-100, reported raw and as M1_adj (definition R1 against CGFlow-B's pooled top-100 of the same pocket); M1_adj is used for a comparator only if its mean top-100 heavy-atom count lies within 3 atoms of CGFlow-B's on that panel, otherwise raw and 3-atom-bin matched values are reported.
* Family P (portability, Holm over 2): P1 CE - RxnFlow on the 15 development pockets; P2 CE - RxnFlow on the 20 held-out pockets. Supported if the mean is negative and the Holm-adjusted p < 0.05; the ratio to F - CGFlow-B on the same panel is reported (retention).
* Family X (external comparators, Holm over 3): X1 F - RxnFlow (15 development pockets); X2 F - SynFlowNet (15); X3 F - GraphGA (15). Supported if the mean is negative and the Holm-adjusted p < 0.05.
* X4 F - S3-GFN on the five shared pockets: descriptive (N = 5), no test statistic is used as a gate.
* Secondary (no gate): heavy atoms, QED, independent-planner and AiZynthFinder solved share, PoseBusters pass rate, cross-receptor retention, budget-snapshot curves, wall-clock.
Allowed wording is fixed by the outcome: a contrast that is not supported is reported as such; nothing in this study supports a statement about "state of the art"; the strongest local competitor remains SynGBO (v1.0).

## 6. Conduct
Failures: node loss -> resume from the last checkpoint; CUDA out of memory -> resume from the checkpoint on a GPU with more memory; a second failure is recorded as missing and the worst-case imputation sensitivity is added; no run is excluded because of its result. Health monitoring only (job states, error logs, counters) until a panel is complete. Deviations are appended to `prereg/DEVIATIONS_confirmatory_trial.md` (section "Supplement S1").
