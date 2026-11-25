# CGFlow + ARRE reproducibility release

This repository contains the ARRE implementation, the paper's frozen numerical
evidence, figure reconstruction code and paper-named run interface. It is
intended to be uploaded as a source repository. It does not redistribute pose
model checkpoints, protein pockets, reaction catalogs, private cluster data or
third-party planner weights. The manuscript makes the same limitation explicit.

## 1. Directory layout

`code/integrated_ARRE/` is the runnable ARRE/CGFlow source tree. `code/entrypoints/`
contains the original training entrypoints and `code/ARRE_PaperAPI/` provides
the paper-named components (`ARRE`, `RHC`, `RFG`, `SPP`) and the legacy run profiles. `code/evaluation/`
contains the top-K extraction and paper claim checks. `manuscript/source/data/`
contains the frozen arrays used by the figures (local storage paths in
metadata are redacted). `manuscript/scripts/` rebuilds
the quantitative panels. The historical/ directory reconstructs the archived source trees associated with the five held-out search profiles. Its manifests and instructions describe the version binding.

## 2. Hardware and software

For the supplied environment, use Linux, Python 3.11, one CUDA GPU with at
least 12 GB memory, 8 CPU cores and at least 26 GB RAM. The verified stack was
Python 3.11.16, PyTorch 2.5.1 built for CUDA 12.1, NumPy 1.26.4,
SciPy 1.17.1, pandas 3.0.6, RDKit 2024.03.6, Matplotlib 3.10.9,
Lightning 2.6.6, Uni-Dock 1.1.1 and Uni-Dock tools 0.0.1. Match the CUDA and PyTorch build
before installing PyG extensions.

```bash
conda create -n arre python=3.11 -y
conda activate arre
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter torch-sparse torch-cluster --only-binary=:all: \
  --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
python -m pip install -r requirements-figures.txt
python -m pip install -e code/integrated_ARRE \
  --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Install Uni-Dock separately, for example `conda install -c conda-forge -c bioconda unidock=1.1.1`
when that build is available for your platform. Alternatively, use the
provider's release or an HPC module. Confirm `unidock` is on `PATH`; a Python
installation of `unidock_tools` alone does not provide the GPU executable.

```bash
python - <<'PY'
import torch, torch_scatter, torch_sparse, torch_cluster
import rdkit, scipy, cgflow, rxnflow
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("rdkit", rdkit.__version__, "scipy", scipy.__version__)
print("cgflow", cgflow.__file__)
PY
unidock --version
```

## 3. External assets

Prepare three paths outside this repository.

1. `ENV_DIR` must contain the reaction/building-block environment, including
   `workflow.yaml`, `bb_feature.pt` and the generated `blocks/` directory.
   Generate it from your licensed catalog with the included scripts below,
   or point to a compatible environment already prepared by your lab.
2. `POCKET_DIR/TARGET/` must contain `protein.pdb` and `ligand.mol2` for each
   target. On a cluster you can stage them to node-local scratch and point
   `POCKET_DIR` there when launching the run.
3. `POSE_CKPT` must point to the compatible StateFlow pose checkpoint
   (`stateflow_main_ep029.ckpt` in the original project). Do not commit this
   file to GitHub.

Check that the files exist before a run:

```bash
export ENV_DIR=/path/to/catalog-env
export POCKET_DIR=/path/to/pockets
export POSE_CKPT=/path/to/stateflow_main_ep029.ckpt

# Only run these two commands when building a new environment from a catalog.
# The destination ENV_DIR must not exist.
python code/integrated_ARRE/data/scripts/a_catalog_to_smi.py \
  -b /path/to/catalog.sdf -o /path/to/catalog.smi --cpu 8
python code/integrated_ARRE/data/scripts/b_create_env.py \
  -b /path/to/catalog.smi -p code/integrated_ARRE/data/template/real \
  -o "$ENV_DIR" --cpu 8

test -f "$ENV_DIR/workflow.yaml"
test -f "$ENV_DIR/bb_feature.pt"
test -d "$ENV_DIR/blocks"
test -f "$POCKET_DIR/BACE2_HUMAN/protein.pdb"
test -f "$POCKET_DIR/BACE2_HUMAN/ligand.mol2"
test -f "$POSE_CKPT"
```

## 4. Validate the source package

Run these commands from the repository root. They do not train a model or
invoke docking.

```bash
python code/ARRE_PaperAPI/ARRE_release_cli.py verify
python code/ARRE_PaperAPI/ARRE_verify_claims.py --json
python manuscript/scripts/rebuild_all_quantitative.py
```

`verify` checks the package SHA-256 manifest. `ARRE_verify_claims.py` recomputes
the four central paired estimates from the frozen 20-pocket vectors. The figure
command rebuilds quantitative Figures 2-8 and Supporting Figures S1-S3 under
`manuscript/source/figures/`. Run it in a disposable checkout; figure
reconstruction also updates the figure QA metadata checked by `verify`. The
curated Figure 1 and graphical abstract are in the separate manuscript package.

## 5. One-iteration GPU smoke test

Run this on a compute node with the external assets. It checks imports, model
startup, one Uni-Dock batch and output writing; it is not a reproduction of
the paper's 64,000-evaluation trials.

```bash
export PYTHONPATH="$PWD/code/ARRE_PaperAPI:$PWD/code/integrated_ARRE/src"
export WANDB_MODE=offline OMP_NUM_THREADS=4
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
python code/ARRE_PaperAPI/ARRE_run.py --profile ARRE \
  --target BACE2_HUMAN --seed 0 --steps 1 \
  --env-dir "$ENV_DIR" --pocket-dir "$POCKET_DIR" \
  --ckpt "$POSE_CKPT" --log-dir "$PWD/smoke_output" --execute
```

Expected outputs include `smoke_output/train.log`, one or more
`smoke_output/train/generated_objs_*.db` files, `smoke_output/model_state.pt`
and `config.seed0.yaml` next to the run directory. Look for `iteration 1`,
a nonzero docking-call count and `[run_opt] finished cleanly` in the terminal
output or scheduler log. The first iteration does not test RHC/RFG after
warmup.

## 6. Training a new run

The runner is dry-run by default. Remove `--execute` to inspect the complete
command without allocating a GPU. `--steps 1000` is the paper-scale training
length; it is computationally expensive and should be submitted through the
cluster scheduler.

```bash
export PYTHONPATH="$PWD/code/ARRE_PaperAPI:$PWD/code/integrated_ARRE/src"
mkdir -p "$PWD/runs/ARRE_BACE2_s0"
python code/ARRE_PaperAPI/ARRE_run.py --profile ARRE \
  --target BACE2_HUMAN --seed 0 --steps 1000 \
  --env-dir "$ENV_DIR" --pocket-dir "$POCKET_DIR" \
  --ckpt "$POSE_CKPT" --log-dir "$PWD/runs/ARRE_BACE2_s0/train_run" --execute
```

`RHC` enables reaction-history continuation. `RFG` enables reactant feedback
guidance. `SPP` enables the frozen synthesis prior. `ARRE` combines all three, `F` selects standard ARRE, and `FZ` selects policy-frozen ARRE (figure label “ARRE (frozen policy)”). In the latter, the generator policy and log-Z are held fixed while the reaction-history archive and reactant feedback continue to update. The paper prior is shipped as
`code/frozen_prior/syn_prior_v1.frozen.json`. The profiles are starting points
for new runs; in this CLI `F` and `ARRE` use the same combined settings.
Use a different parent directory for each target/seed so emitted config
files never overwrite each other.

For Slurm, place the command above in a batch script with one GPU, 8 CPUs,
26 GB memory and a wall-time appropriate to the target. Set
`SLURM_CPUS_PER_TASK` so the entrypoint can cap Uni-Dock's worker pool. Write
logs and outputs to a run-specific directory. This command starts a new run;
it is not a resume command.

## 7. Evaluate a completed run

After `train.log` and the generated SQLite files exist, extract the paper's
diverse top-100 set:

```bash
python code/evaluation/extract_topk.py \
  --run_dir "$PWD/runs/ARRE_BACE2_s0/train_run" \
  --out_dir "$PWD/results/ARRE_BACE2_s0" \
  --target BACE2_HUMAN --k 100
```

This writes `topk.csv`, `topk.smi` and `metrics.json`. The selector keeps
QED > 0.5 molecules, ranks by docking score and removes RDKit-fingerprint
duplicates at Tanimoto similarity 0.5. It does not run docking itself.

For a paper-level comparison, repeat the same command for every target and
seed, preserve the generated manifests and logs, and apply the same target
list, budget, seed set, docking executable, checkpoint and planner versions.
Do not compare a one-iteration smoke run with the manuscript's final metrics.

## 8. Troubleshooting

- `ModuleNotFoundError` for `cgflow` or `rxnflow`: export the `PYTHONPATH`
  shown above and verify the editable install points at `code/integrated_ARRE`.
- PyG import or undefined-symbol errors: reinstall the three PyG extension
  wheels from the index matching the installed torch CUDA build.
- `workflow.yaml` or `bb_feature.pt` missing: `ENV_DIR` is not a generated
  CGFlow catalog environment.
- Uni-Dock cannot start: check `which unidock`, executable permissions,
  CUDA visibility and the provider's runtime libraries.
- `protein.pdb` or `ligand.mol2` missing: the target folder is incomplete.
- The output directory already exists: use a fresh path for `--log-dir`.
- Memory pressure during docking: run on a node with at least 12 GB GPU memory,
  set `SLURM_CPUS_PER_TASK` to the allocation and keep temporary docking work
  on node-local scratch when the cluster requires it.

## 9. Reproducibility boundary

The frozen arrays and source checks support statistical and figure
reconstruction. They do not establish a fresh full-model reproduction,
independent binding validation, chemical synthesis or public redistribution
rights for third-party data. Review UPSTREAMS.md and upstream redistribution terms before publishing this repository.

## Evidence for the submission

The evidence directory contains the pocket-level contrasts, the sequence-similarity sensitivity analysis, and counts for candidate and route coverage. The full selected-molecule and generation-route archives are separate companion datasets. Their names and checksums appear in COMPANION_DATA.md. The original forward_paths column in the selected-molecule archive covers the single legacy example. The generation-routes addendum supersedes that field as a measure of trajectory coverage.

## Rebuild Figures 5, 6 and 8

Run `python manuscript/scripts/comparison_figures.py --output figures --qa-dir plate_qa` to render the three published comparisons from the included analyses. Each figure is exported as a true-text PDF, editable SVG and 600 dpi TIFF. The same renderer is used by the full quantitative rebuild. Each panel maps to input files and a statistical procedure in `manuscript/source/panel_sources.json`. Original target and seed records are retained in `manuscript/source/data`.

Alignment, PDF text, collision and source checks are archived in `manuscript/source/plate_qa/`. To repeat the PDF checks with PyMuPDF installed, run `python manuscript/scripts/figure_qc/audit_pdf_text.py figures/Figure_05.pdf --min-pt 6` and `python manuscript/scripts/figure_qc/audit_figure_collisions.py figures/Figure_05.pdf`. Substitute the corresponding filenames for Figures 6 and 8. The bundled audit programs come from [nature-skills](https://github.com/Yuan1z0825/nature-skills), with their license in `manuscript/scripts/figure_qc/LICENSE`.

The three revised figures use the same quantitative rebuild command. Figure 5 depicts the independently recorded ADRB2 parent, shared prefix and continuation from search/search_route.json. The SVGs retain a shared-core orientation. Figure 8 includes an exploratory pose-displacement comparison and a single OPRK1 example. The two coordinate files, mapped heavy-atom positions, local receptor coordinates, complete eligible IDs and source digests are in manuscript/source/data/medicinal_figures. To re-extract those small assets from the original full CGFlow project, run `python manuscript/scripts/prepare_structure_evidence.py /path/to/CGFlow` in the project RDKit environment. Redrawing the published panels uses the included data directly.

Figures 5, 6 and 8 use the bundled Liberation Sans regular and bold typefaces. Their copyright and SIL Open Font License are in `manuscript/scripts/fonts/LICENSE.txt`.

Figure 8c shows the original and redocked poses at the same scale in the prepared OPRK1 pocket. Both camera-matched images, PDB atoms, pose files and per-landmark coordinates or atom indices are in `manuscript/source/data/medicinal_figures/`. From the archive root, run `OPENBLAS_NUM_THREADS=1 python manuscript/scripts/render_oprk1_pocket.py` to regenerate the pocket images and JSON using NumPy, SciPy and Pillow, then `python manuscript/scripts/rebuild_all_quantitative.py`. The pocket PDB can be re-extracted by the structure-evidence command above.
