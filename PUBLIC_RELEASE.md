# Public naming revision

RHC is the reaction-history continuation component. RFG and SPP share
the `rfg_spp` module and configuration. The public release uses these
names in code, configurations and provenance references.

`SHA256SUMS.json` verifies the distributed files. The original submission
archive is retained in the project records. Its SHA-256 is
`76e9a0a980ee3a0eaf52567f6ecda1f27f7294ec5d4117b06e2e4ada9865d0d0`. Historical content and deployment
hashes embedded in the provenance records refer to the recorded run files.

The public runners accept earlier trainer arguments and component state keys.
The source version, configuration and external inputs must still match the run.
The public source emits the published configuration, metric and checkpoint keys.

The confirmatory plan and deviation log in `manuscript/source/data/acceptance/`
are public copies with updated terminology. Recorded dates and decisions follow
the original records.
