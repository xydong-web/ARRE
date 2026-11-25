# Scientific scope and code mapping

| Paper term | Actual archived implementation |
|---|---|
| ARRE | `cgflow.lanes.combo:VinaMOOTrainer_ARRE` |
| RHC | `cgflow.lanes.rhc` — elite-prefix continuation and bandit allocation |
| RFG | `cgflow.lanes.rfg_spp` — online Bayesian-ridge reactant critic, tilted block choices |
| SPP | `cgflow.lanes.rfg_spp` — `syn_prior`, `syn_kappa`, `syn_lookahead` (frozen) |
| Standard ARRE (legacy key `F`) | Archived `x4nw2F` docking-priority arm |
| Policy-frozen ARRE (figure label “ARRE (frozen policy)”, legacy key `FZ`) | Archived `x4nw2FZ` balanced frozen generator policy |

Policy-frozen versus standard ARRE in the archived configurations: optimizer learning rate 0.0001 -> 0;
log-Z learning rate 0.001 -> 0.

Proof strength: superiority of size-adjusted Vina for two respective parents;
policy-frozen ARRE noninferiority for independent retrosynthesis planning vs SynGBO;
descriptive/exploratory for the heavy-atom cap, alternative CNNaffinity
and RxnFlow geometric pass-rate changes. Synthesis and binding are computed,
not experimentally confirmed. Distributed historical configurations use the same public component names.
