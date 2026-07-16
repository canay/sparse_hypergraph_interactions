# Claim Boundary

Date/time: 2026-07-16 23:22 +03:00
Tool: Codex
Model, if known: GPT-5
Operation ID: `shil-github-disclosure-tier-reassessment-20260716-2312`

This public release is aligned with the diagnostic manuscript after the
claim-trace audit, the Q1 numeric update, and the bounded sensitivity check.

Supported:

- Compact sparse support selection over pair and triple interaction candidates.
- Planted-support recovery checks on controlled synthetic tasks.
- Comparison against direct L1 pair/triple logistic interaction search under matched train/validation/test protocols.
- Budget-matched L1 top-8 coefficient-ranking check as a required fairness boundary.
- Reporting selected-edge count, support precision, support recall, and predictive metrics together.
- Bounded three-seed planted-pair sensitivity checks for correlated predictors, lower signal-to-noise, smaller samples, and class imbalance.

Not supported:

- Broad tabular state-of-the-art claims.
- General superiority over sparse polynomial logistic regression.
- Superiority over budget-matched L1 coefficient ranking.
- Causal interpretation of real-data interactions.
- General sensitivity coverage across all synthetic generators, categorical data, distribution shift, missing-data regimes, or high-dimensional real tables.

The audited `EXP-SHIL-Q1-002` evidence supports this boundary only as a bounded wider-feature support-stress result. The `EXP-SHIL-Q1-003` evidence narrows the compactness language: SHIL is more compact than the default nonzero L1 support in the promoted comparisons, but L1 top-8 ranking matches SHIL's synthetic planted-support precision and recall. The `EXP-SHIL-Q1-004` evidence extends the default-support compactness check to four planted-pair stressors, but it remains descriptive and small-sample.
