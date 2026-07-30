# EXP-SHIL-SC-001

Date/time: 2026-07-30 19:53 +03:00
Tool: Codex
Model, if known: GPT-5.6 Extra High (xhigh, user-attested)
Operation ID: `shil-ijmlc-public-repo-sync-20260730-1953`

`EXP-SHIL-SC-001` evaluates stability-calibrated SHIL and matched L1 routes under
the hash-locked protocol in `protocol/`.

The full synthetic matrix contains 120 dataset runs across twelve scenarios and
ten generator seeds. Its `metrics.csv` has 720 method rows, and the recorded
analysis decision is `PROTOCOL_ONLY`. The Dry Bean recovery contains ten
completed outer seeds and 60 method rows. Each seed directory provides a
completion record, resolved configuration, run identity, source and protocol
hashes, and output hashes.

## Layout

- `src/`: experiment, partition runner, and analysis source.
- `tests/`: source and partition-runner unit tests.
- `config/`: the full configuration and ten seed-specific configurations.
- `protocol/`: the pre-specified method-extension protocol.
- `synthetic/`: final full-synthetic outputs and analysis.
- `dry_bean/`: final combined and per-seed Dry Bean outputs and analysis.
- `commands/`: the recorded VPS launch wrappers. They contain the execution
  environment's original Python path.

`full_synthetic_run_manifest.json` and
`dry_bean_recovery_run_manifest.json` are immutable run-time snapshots. The
latter records the launch boundary; `release_closure.json`, the combined
outputs, and the ten `partition_status.json` files record the verified completed
state.

## Commands

From this directory, a new synthetic run can be written to a fresh output path:

```powershell
python src/sc_shil_experiment.py --config config/full_config.json --mode full-synthetic --output reproduced/full-synthetic
python src/analyze_results.py --input reproduced/full-synthetic --output reproduced/full-synthetic-analysis
```

Dry Bean is downloaded by the supplied loader and checked against the locked UCI
archive hash. Raw dataset files are not included. The complete synthetic and
Dry Bean runs are long CPU jobs; `run_smoke.ps1` at the repository root performs
integrity checks without rerunning them.
