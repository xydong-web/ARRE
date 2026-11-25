# Upstream snapshots and licenses

The runnable `code/integrated_ARRE/` tree retains the upstream CGFlow
`LICENSE` and citation. External components and checked-out references used in
the project are listed below. The table is not a claim that every comparator
has been independently rebuilt in this repository.

| Component | Source | Commit or provenance |
|---|---|---|
| CGFlow | `tsa87/cgflow` | integrated source under `code/integrated_ARRE/` |
| RxnFlow | `SeonghwanSeo/RxnFlow` | upstream reference; archived trial source reconstructed by historical/restore_sources.py |
| SynGA | `alstonlo/synga` | external |
| S3-GFN / SynFlowNet / Syntheseus | `hyeonahkimm/s3gfn`, `mirunacrt/synflownet`, `microsoft/syntheseus` | external |
| GraphGA comparator | `wenhao-gao/mol_opt` | external |

Each redistributed upstream tree retains its own license. Review upstream
licenses and benchmark-provider terms before a public GitHub upload.
