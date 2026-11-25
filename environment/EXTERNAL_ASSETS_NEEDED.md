# External runtime assets required for full scientific reproduction

The current archive does not include:
- weights/stateflow_main_ep029.ckpt (~454 MB; the original pose-model checkpoint)
- data/envs/catalog/ (original reaction environment and building-block catalog)
- data/LIT-PCBA/{pocket}/protein.pdb and ligand.mol2
- Uni-Dock executable and full environment-specific runtime dependencies
- independent Retro*-MHNreact pretrained planner assets

The source and frozen results can be reconstructed without these, but new
pocket-guided generation requires user-provided legally accessible copies.
The paper trial used 64,000 new molecule evaluations per target and three
seeds. A 1-iteration GPU smoke test is NOT a reproduction of the paper metrics.
