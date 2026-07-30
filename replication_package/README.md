# Replication Package

Date/time: 2026-07-30 19:53 +03:00
Tool: Codex
Model, if known: GPT-5.6 Extra High (xhigh, user-attested)
Operation ID: shil-ijmlc-public-repo-sync-20260730-1953

This directory maps manuscript evidence to runnable scripts and package-local generated outputs.

Public repository: `https://github.com/canay/sparse_hypergraph_interactions`.

## Core Scripts

- `../code/shil_run_experiments.py`: base datasets, candidate interaction construction, SHIL selector, and main frozen-style experiment runner.
- `../code/shil_q1_extension.py`: post-G02 follow-up controls for direct L1 interaction baselines and boosted controls.
- `../code/shil_scale_stress.py`: registered `EXP-SHIL-Q1-002` wider-feature support-stress follow-up.
- `../code/q1_action_remediation_analysis.py`: `EXP-SHIL-Q1-003` summaries for uncertainty, budget-matched L1 top-8 ranking, and stability numeric reporting.
- `../code/q1_bounded_sensitivity.py`: `EXP-SHIL-Q1-004` bounded sensitivity rerun for correlated predictors, lower signal-to-noise, smaller samples, and class imbalance.

## Evidence Boundary

`EXP-SHIL-Q1-002` is used only as bounded wider-feature support-stress evidence. `EXP-SHIL-Q1-003` narrows the claim further: L1 top-8 coefficient ranking matches SHIL's planted-support precision and recall on the synthetic follow-up, so the manuscript may contrast SHIL with the default nonzero L1 support but must not claim superiority over budget-matched L1 ranking. `EXP-SHIL-Q1-004` adds descriptive three-seed planted-pair sensitivity evidence; it is not a broad generalization benchmark.

The `calibrated_runs/` subtree contains `EXP-SHIL-SC-001` and
`EXP-SHIL-FIXED-C-001`. The synthetic stability-calibration matrix supports the
protocol-only decision recorded in its analysis summary. The Dry Bean arm has no
ground-truth interaction support and is retained as real-data prediction,
stability, and cost evidence. The fixed-`C` arm is a pre-specified
procedure-level cost sensitivity and does not establish general method
superiority.

## Commands

From the repository root:

```powershell
python -m pip install -r requirements.txt
.\run_smoke.ps1
.\run_all.ps1
```

The command wrappers verify package layout, dependency imports, SHA-256 entries,
selected evidence-table schemas, calibrated row counts, and the twenty completed
Dry Bean seed records without running an experiment. Full experiment commands
are documented with each calibrated run and are not required for a clean-clone
smoke test.
