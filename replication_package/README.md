# Replication Package

Date/time: 2026-07-16 23:22 +03:00
Tool: Codex
Model, if known: GPT-5
Operation ID: shil-github-disclosure-tier-reassessment-20260716-2312

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

## Commands

From the repository root:

```powershell
python -m pip install -r requirements.txt
.\run_smoke.ps1
.\run_all.ps1
```

The command wrappers verify package layout, dependency imports, and selected evidence-table schemas without running an experiment. Full frozen, ten-seed follow-up, scale-stress, and bounded-sensitivity experiments remain available through the scripts but are not required for a clean-clone smoke test.
