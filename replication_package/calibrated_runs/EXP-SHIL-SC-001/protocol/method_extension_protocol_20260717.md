# Stability-Calibrated SHIL Method-Extension Protocol

Date/time: 2026-07-17 09:48:13 +03:00
Tool: Codex
Model, if known: GPT-5
Operation ID: shil-stability-extension-lock-20260717-0948

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan-to-run, user pre-authorized
- Origin Date: 2026-07-17
- Verification Status: DESIGN_LOCKED_BEFORE_PILOT
- Version Label: sc_shil_protocol_v1

## Objective and Hypothesis

Objective: test whether replacing fixed top-`k` SHIL reporting with complementary-pairs stability aggregation and validation-loss calibration yields a genuinely better explicit pair/triple support than strong L1 alternatives under identical data, candidate, resampling, tuning, and refit conditions.

Primary hypothesis: SC-SHIL improves exact support F1 and false-inclusion count over both size-matched L1 and stability-calibrated L1 while remaining noninferior in test log loss.

The protocol permits a negative outcome. No scenario, seed, comparator, or endpoint may be removed after results are observed.

## Method Definition

For every outer run:

1. Generate or load the dataset without access to the outer test fold.
2. Make a stratified 60/20/20 train/validation/test split. Fit `StandardScaler` on outer training only.
3. Enumerate the identical clipped pair-plus-triple candidate family for SHIL and L1.
4. On outer training, construct `B=20` deterministic complementary half-sample pairs (40 base-selection fits). Each half-sample receives a deterministic inner 80/20 fit/early-stop split.
5. For SHIL, fit the existing supervised-moment gate model and retain its complete hyperedge ranking. For L1, select `C` from `{0.02, 0.05, 0.25, 1.0}` using inner validation log loss, refit on the full half-sample, and rank interaction blocks by class-coefficient L2 norm.
6. For each base ranking, record top-`q` membership for `q in {4, 8, 16}`. Aggregate selection frequency over all 40 half-samples.
7. Form stable-support candidates with `pi in {0.60, 0.70, 0.80, 0.90}`. Empty interaction support is allowed and means a main-effects-only refit.
8. For each distinct candidate support, refit L2 logistic regression on outer training and compute per-observation validation log loss. Select the smallest support whose mean validation loss is within one standard error of the minimum. Ties are resolved by higher `pi`, then lower `q`, then lexicographic edge order.
9. With the chosen `q` and `pi`, recompute complementary-pairs frequencies on the combined 80% development data using the same deterministic schedule, freeze the final support, refit L2 logistic regression on development data, and evaluate the locked test fold once.

The procedure is called `SC-SHIL` when the base ranking comes from SHIL and `SC-L1` when the identical wrapper uses L1. No finite-sample false-discovery-control claim will be made unless its assumptions and bound are separately verified; the complementary-pairs construction is used here as a disciplined resampling design.

## Locked Comparators

1. `L1-full`: validation-selected default nonzero L1 pair-plus-triple support.
2. `L1-top8`: fixed top-eight coefficient ranking for continuity.
3. `L1-match`: top-`k` L1 ranking with `k` equal to SC-SHIL's final support size in the same outer run.
4. `SHIL-k8`: current supervised-moment SHIL with fixed eight-edge support.
5. `SC-L1`: stability-calibrated L1 with the identical wrapper.
6. `SC-SHIL`: proposed method extension.

All six use the same outer split, scaling, clipped interaction products, candidate family, development/test boundary, and L2 logistic refit where applicable.

## Locked Synthetic Scenario Matrix

Ten independent outer generator seeds will be used: `1103, 1129, 1151, 1171, 1181, 1201, 1213, 1231, 1249, 1277`. Pilot seeds `901, 907` are implementation/resource checks and are not promoted.

| ID | Order/density | n | d | Noise/correlation | Heredity or ambiguity role |
|---|---|---:|---:|---|---|
| S01 | 3 pairs | 2800 | 16 | sd 0.65, independent | heredity violated; continuity core |
| S02 | 3 triples | 2800 | 16 | sd 0.65, independent | heredity violated; order stress |
| S03 | 2 pairs + 2 triples | 2800 | 16 | sd 0.65, independent | heredity violated; mixed core |
| S04 | mixed 4-edge | 900 | 16 | sd 0.65, independent | small-sample stress |
| S05 | mixed 4-edge | 6000 | 16 | sd 0.65, independent | sample-size scaling |
| S06 | mixed 4-edge | 2800 | 32 | sd 0.65, independent | dimension/candidate scaling |
| S07 | mixed 4-edge, coefficients x0.55 | 2800 | 16 | sd 1.10, independent | weak-signal/low-SNR stress |
| S08 | mixed 4-edge | 2800 | 16 | sd 0.65, AR(1) rho 0.60 | correlated-predictor stress |
| S09 | mixed 4-edge plus main effects on all participating variables | 2800 | 16 | sd 0.65, independent | strong-heredity-respected regime |
| S10 | 4 pairs + 4 triples | 4000 | 24 | sd 0.75, independent | denser mixed support |
| S11 | mixed 4-edge with correlated proxy variables | 2800 | 20 | sd 0.65, proxy correlation 0.95 | redundant-support ambiguity; report exact and equivalence-class metrics |
| S12 | mixed 4-edge | 2800 | 16 | sd 0.65, independent, 0.70 label quantile | class-imbalance stress |

Synthetic labels, planted edges, equivalence classes, and nuisance terms are generated before splitting. Planted support is never used for model fitting, threshold calibration, or stopping.

## Real-Data Resampling Study

The stronger real-data application is the UCI Dry Bean dataset: 13,611 observations, 16 numeric image-derived morphology features, seven classes, no reported missing values, DOI `10.24432/C50S4B`, CC BY 4.0. It will be reconstructed through a versioned loader and verified by shape, column names, class counts, and a local content hash. Raw data will not be redistributed unnecessarily.

Dry Bean will use ten stratified outer resplits with the locked six-method comparison. Because no interaction truth exists, endpoints are prediction, support size, outer-resplit Jaccard, chance-corrected stability, edge-frequency concentration, runtime, and peak resident memory. Selected products remain associational model diagnostics.

## Primary and Secondary Endpoints

Primary structural endpoints on synthetic data:

- exact edge-level support F1;
- false-inclusion count and false-discovery proportion;
- support recall and precision;
- selected-support size;
- pairwise Jaccard across independent outer generator draws;
- Nogueira chance-corrected stability with uncertainty.

For S11, exact metrics remain primary and equivalence-class recovery is a declared sensitivity endpoint.

Secondary predictive endpoints:

- test log loss;
- macro-F1;
- one-vs-rest ROC-AUC where defined;
- accuracy for continuity only.

Computational endpoints:

- wall-clock training and prediction time;
- peak resident memory;
- candidate count and completed base-fit count;
- failures, convergence warnings, and early-stopping counts.

## Uncertainty and Paired Comparisons

- Report every scenario/method cell; no cherry-picking.
- Summarize outer-run means, sample standard deviations, medians, and percentile intervals.
- Primary paired differences use 10,000 bootstrap resamples of outer runs, stratified by scenario.
- Report 95% confidence intervals for SC-SHIL minus each comparator.
- No equivalence or superiority language is permitted from overlapping descriptive intervals alone.

## Locked Success and Pivot Criteria

`METHOD_EXTENSION_SUPPORTED` requires all of the following on the macro-average of S01-S12:

1. the 95% CI lower bound for SC-SHIL minus `L1-match` support F1 is above 0;
2. the 95% CI lower bound for SC-SHIL minus `SC-L1` support F1 is above 0;
3. the 95% CI upper bounds for SC-SHIL minus each of those comparators in false-inclusion count are below 0;
4. the 95% CI upper bound for SC-SHIL minus each comparator in test log loss is below the noninferiority margin `0.01`;
5. SC-SHIL improves outer-run support stability over `SHIL-k8` with a positive 95% CI lower bound.

`PROTOCOL_ONLY` applies if stability calibration improves one or both fixed selectors but SC-SHIL does not satisfy the method-specific comparison against SC-L1. The paper may discuss a comparative stability-calibration protocol but may not claim a uniquely superior SHIL extension.

`NEGATIVE_EXTENSION` applies if SC-SHIL fails to improve structural recovery over `L1-match`, breaches the prediction margin, or depends on excluding preregistered scenarios. The negative result must be retained and reported.

## Execution Stages and Commands

The implementation will be self-contained under:

`experiments/2026-07-17_codex_local_stability-calibrated-shil/`

Stages:

1. Unit tests and static configuration validation.
2. Smoke run on one small scenario/seed with reduced complementary pairs.
3. Non-promoted pilot on seeds 901 and 907 to verify runtime, memory, output schema, and failure handling; scientific hyperparameters remain locked.
4. Full synthetic matrix, real-data resampling, analysis, and figures.

Exact commands, environment capture, console transcripts, configuration hashes, source hashes, start/end timestamps, exit codes, and output hashes must be stored in the run folder and registered in `experiments/EXPERIMENT_REGISTRY.csv`.

## Monitoring and Stop Rules

- Hard timeout: 24 hours per full command; terminate only at the hard timeout.
- Process-alive and progress-file checks: every 30 seconds while actively monitored.
- Output-stall advisory: no progress update for 10 consecutive checks, recognizing long matrix operations.
- No automatic retry after a crash. Diagnose, log, and rerun only if the failure is an implementation/environment fault rather than a scientific outcome.
- A convergence warning is recorded as an outcome; it may trigger a predeclared maximum-iteration escalation only if the same escalation is applied to all affected methods/cells.

