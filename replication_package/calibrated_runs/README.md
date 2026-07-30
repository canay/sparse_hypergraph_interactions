# Calibrated Run Artifacts

Date/time: 2026-07-30 19:53 +03:00
Tool: Codex
Model, if known: GPT-5.6 Extra High (xhigh, user-attested)
Operation ID: `shil-ijmlc-public-repo-sync-20260730-1953`

This directory contains the final selected artifacts for the stability-calibration
method extension and its fixed-`C` cost sensitivity.

## Artifact Map

- `EXP-SHIL-SC-001/` contains the hash-locked protocol, runnable source,
  configurations, twelve-scenario synthetic outputs, the ten-seed Dry Bean
  recovery outputs, analyses, and per-seed completion records.
- `EXP-SHIL-FIXED-C-001/` contains the pre-specified fixed-`C=1.0` protocol,
  runnable source, configurations, ten-seed Dry Bean outputs, analyses, and
  verification records.
- `manifest_sha256.csv` records each curated file's relative path, byte size, and
  SHA-256 digest.

## Evidence Boundary

The synthetic stability-calibration matrix records the project-level decision
`PROTOCOL_ONLY`. The Dry Bean analysis has no interaction ground truth and
therefore supports prediction, stability, and cost reporting only. The fixed-`C`
analysis isolates the cost of L1 hyperparameter search under the locked
sensitivity design; it does not support an unrestricted superiority claim.

Raw public datasets and local cache files are not redistributed. The run
identities and completion records preserve the execution paths recorded at run
time. These paths are provenance fields, not required installation locations.
