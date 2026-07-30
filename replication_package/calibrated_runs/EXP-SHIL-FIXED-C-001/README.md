# EXP-SHIL-FIXED-C-001

Date/time: 2026-07-30 19:53 +03:00
Tool: Codex
Model, if known: GPT-5.6 Extra High (xhigh, user-attested)
Operation ID: `shil-ijmlc-public-repo-sync-20260730-1953`

`EXP-SHIL-FIXED-C-001` is the pre-specified fixed-`C=1.0` Dry Bean cost
sensitivity linked to `EXP-SHIL-SC-001`.

The run contains ten completed outer seeds and 20 contemporary method rows. Its
locked analysis reports the arithmetic mean of ten paired selection-time ratios
and a 10,000-resample percentile interval. The interpretation is limited to the
evaluated procedure and cost design.

## Layout

- `src/`: fixed-`C` experiment, predecessor implementation, partition runner,
  and analysis source.
- `config/`: the full configuration and ten seed-specific configurations.
- `protocol/`: the pre-specified cost-sensitivity protocol.
- `inputs/`: the historical tuned-run metrics used only for the labeled
  descriptive comparison.
- `dry_bean/`: combined and per-seed outputs plus locked analysis files.
- `verification/`: reconciled pullback and locked-analysis checks.
- `commands/`: the recorded VPS launch wrappers. They retain the execution
  environment's original Python path.

The repository-root `run_smoke.ps1` verifies the final row counts, the ten
successful per-seed statuses, the selected analysis values, and the curated
SHA-256 manifest without rerunning the experiment.
