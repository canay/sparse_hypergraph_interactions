# Fixed-C Cost-Sensitivity Protocol

Date/time: 2026-07-27 15:03 +03:00  
Tool: Codex  
Model, if known: GPT-5 runtime family; exact picker not exposed  
Operation ID: `shil-fixed-c-cost-sensitivity-20260727-1503`

Status: `RESULT_UNSEEN_PROTOCOL_LOCK`  
Issue: `RA-A06b`  
Methodology change: `MCH-SHIL-002`  
Experiment ID: `EXP-SHIL-FIXED-C-001`

## Purpose and claim boundary

The completed Dry Bean comparison used the same 80 stability-ranking calls for
SC-L1 and SC-SHIL, but each tuned L1 call fitted four candidate regularization
values and then refitted the selected model. The L1 route therefore used 400
logistic fits, whereas SC-SHIL used 80 screening optimizations. This sensitivity
isolates the L1 hyperparameter-search component by fixing the regularization
value before any new outcome is observed.

The experiment is a computational-fairness diagnostic. It cannot create a new
method contribution, overturn the locked `PROTOCOL_ONLY` scientific verdict, or
support a general hardware-independent speed claim.

## Result-unseen fixed decision

- Fixed value: `C = 1.0`.
- Rationale: `1.0` is an existing member of the locked four-value grid and the
  pre-existing scale used by the downstream logistic refit. It is fixed before
  this sensitivity produces any runtime, support, or predictive result.
- A fixed-C ranking call fits one L1 logistic model on the entire complementary
  half-sample. It performs no inner C search and no post-search refit.
- Each stability-calibrated selector receives 20 complementary pairs in the
  tuning phase and 20 in the final-development phase. This yields 80 fixed-C
  L1 fits and 80 SC-SHIL screening optimizations per outer split.

## Locked evidence scope

- Dataset: UCI Dry Bean, dataset ID 602, DOI `10.24432/C50S4B`.
- Archive SHA-256:
  `0A64EFF5BE87F48C3DBBFC0A12A56C5D5B5167EF8E61CD45D69B3E7C7130C06F`.
- ARFF SHA-256:
  `B2A4A76A2AEDFB8ED415ADFC1BFC70B5F202CB00CB72E500766E364C14834014`.
- Outer split seeds: `11, 23, 37, 53, 71, 89, 101, 131, 157, 181`.
- Candidate orders: pairs and triples.
- Interaction clipping: `12.0`.
- Calibration grid: `q in {4, 8, 16}` and
  `pi in {0.6, 0.7, 0.8, 0.9}`.
- Calibration rule: the smallest support within one standard error of the best
  outer-validation log loss.
- SC-SHIL settings: 180 epochs, learning rate `0.025`, gate L1 `0.0008`, L2
  `0.0002`, and patience `28`.
- Test data remain untouched until support calibration is complete.

## Comparators

1. `SC-L1-fixed-C`: the identical stability wrapper with one fixed L1 fit per
   ranking call.
2. `SC-SHIL-contemporary`: the unchanged SHIL stability wrapper rerun in the
   same seed job and software environment.
3. `SC-L1-tuned-historical`: the completed 2026-07-18 run, used only to
   quantify the recorded four-C-plus-refit overhead. It is not treated as a
   contemporaneous timing replicate.

## Outcomes and analysis

Primary timing evidence:

- paired per-seed selection time for `SC-L1-fixed-C` and
  `SC-SHIL-contemporary`;
- paired ratio `SC-L1-fixed-C / SC-SHIL-contemporary`;
- median and mean paired ratio, plus a 10,000-resample paired bootstrap
  percentile interval;
- descriptive comparison with the historical tuned SC-L1 time.

Secondary safeguards:

- selected support size, chosen `q` and `pi`, macro-F1, log loss, accuracy, and
  convergence-warning count;
- exact fit/optimization counts and source, config, protocol, data, and output
  hashes;
- no claim of predictive equivalence from small observed differences.

## Falsifiable interpretation rule

If the 95% paired-bootstrap interval for the fixed-C L1 to contemporary SC-SHIL
time ratio includes `1.0`, the manuscript will not retain evidence for a
residual selection-time separation after removal of L1 hyperparameter search.
If the interval lies entirely above `1.0`, only the measured procedure-level
residual ratio will be reported. If it lies entirely below `1.0`, the recorded
direction will be reported without reinterpretation.

No magnitude threshold is chosen after the run. Predictive and support changes
remain descriptive safeguards and cannot be used to select another C value.

## Durability and recovery

- Local smoke must pass before the full run.
- Full evidence is written to a new run folder; prior outputs are never
  overwritten.
- Each outer seed is an atomic partition with a separate status and attempt
  folder.
- A failed seed may be retried only under the unchanged protocol and config.
- Combined analysis runs only after all ten locked seeds pass integrity checks.
- Compute, aggregation, analysis, and manuscript integration are separate
  phases.

## Stop conditions

Stop and record non-evidence if the protocol hash, source hash, input hash, seed
set, candidate grid, fixed C, or fit-count invariant differs from this lock.
Do not substitute a favorable C, omit failed seeds, or integrate partial output.

## Timing-order and bootstrap lock

Date/time: 2026-07-27 15:14 +03:00  
Tool: Codex  
Model, if known: GPT-5 runtime family; exact picker not exposed  
Operation ID: `shil-fixed-c-cost-sensitivity-20260727-1503`

To reduce a fixed method-order artifact, the two selectors alternate order by
the locked outer-seed list position. `SC-L1-fixed-C` runs first for positions
1, 3, 5, 7, and 9; `SC-SHIL-contemporary` runs first for positions 2, 4, 6, 8,
and 10. The method-specific random-seed offsets remain unchanged from the
predecessor protocol.

The bootstrap statistic is the arithmetic mean of the ten paired per-seed time
ratios. A fixed analysis seed of `20260727` generates 10,000 resamples of the
ten paired rows with replacement. The 2.5th and 97.5th percentiles form the
reported interval. The median paired ratio is descriptive and is not used for
the decision rule.
