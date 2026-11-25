# ARRE/RHC/RFG/SPP public names

RHC controls reaction-prefix replay and allocation. RFG controls the critic
tilt. SPP uses the frozen `syn_prior`, `syn_kappa` and `syn_lookahead`.
RFG and SPP share the `rfg_spp` configuration. ARRE combines all three.

`ARRETrainer`, `RHCTrainer`, `RFGSPPTrainer` are actual aliases to imported
classes. The examples use the public trainer and configuration names.

Standard ARRE uses the legacy profile `F`. Policy-frozen ARRE uses `FZ` and sets `opt.learning_rate` and `algo.tb.Z_learning_rate` to zero. The archive and the reactant critic continue to update during search. See ARRE_run_manifest.csv.

Example using the original, archived runner, after installing `code/integrated_ARRE`:

```
PYTHONPATH=code/ARRE_PaperAPI:code/integrated_ARRE/src \
 python code/entrypoints/run_opt.py \
 --trainer arre_paper_names:ARRETrainer \
 --target ADRB2 --seed 0 --env_dir /PATH/TO/ENV \
 --pocket_dir /PATH/TO/POCKETS --ckpt /PATH/TO/POSE_MODEL \
 --log_dir /PATH/TO/NEW/OUTPUT \
 --set rhc.enable=true --set rfg_spp.enable=true \
 --set rfg_spp.group_norm=true --set rfg_spp.syn_kappa=4.0 \
 --set rfg_spp.syn_lookahead=true \
 --set rfg_spp.syn_prior=/PATH/TO/syn_prior_v1.frozen.json
```

This is a **new test run**, not a literal recreation of a historical job;
other settings must be reproduced from the matching frozen config, including
budget, sampling state, pretrained network and input data. Never overwrite
historical logs or assume this example reproduces all published numbers.
