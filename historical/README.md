# Source versions used in the held-out comparisons

The five profiles in heldout_profiles.json come from the scheduler launch ledger. Each includes the trainer options in public naming and the original source directory. The three manifests bind the current archived source files to SHA-256 hashes. The historical scheduler recorded the directory and arguments, but did not sign every source file at launch. Duplicate source files refer to the common implementation in this release. Extra source files are stored by content hash. `source_map.json` verifies the
restorable public source. `pre_rename_source_sha256.json` records the earlier
archive hashes for affected files, indexed by public path.

From the repository root, run `python historical/restore_sources.py RxnFlow /path/to/new/rxnflow-src`. Use CGFlow_guided or CGFlow_parent for the other implementations. The destination itself is the source root to place first in PYTHONPATH. For historical 2D runs use code/entrypoints/run_opt2d_rs.py, and for 3D runs use code/entrypoints/run_opt_rs.py. Select the trainer options for the matching profile, replacing {PRIOR} with the absolute path to code/frozen_prior/syn_prior_v1.frozen.json. Both entrypoints accept --help. Their --pocket_dir argument is the parent directory containing one subdirectory per target (for example POCKET_DIR/BACE2_HUMAN/protein.pdb and ligand.mol2). Supply the subdirectory name with --target. External inputs and the training environment are described in the root README.

The source maps support the two backbone comparisons. The frozen per-pocket arrays and the figure scripts have separate, CPU-only verification entrypoints. Reproducing the upstream SynGBO search or running Retro* requires their external systems and reference inputs.
