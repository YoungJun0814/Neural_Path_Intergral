# G11 M7 V3 Confirmatory Decision

Date: 2026-07-22

Protocol: `g11-m7-confirmatory-v3`

Frozen tag: `g11-m7-confirmatory-v3-freeze`

Frozen source commit: `41bf58cfcc89732014a3d1c20a533edb2bd7339b`

Result SHA-256: `d0165cc9bdc3d961a103b3c0f5c51155747ae638ca68becad59e1059fc0e8ca4`

Independent audit: `results/g11_m7_result_audit_v1_2026-07-22.json`

## 1. Decision

The frozen V3 headline gate **failed**. It must not be reported as a passed untouched
confirmatory experiment.

The reason is narrow but non-negotiable: one Windows checkpoint replacement raised a
`PermissionError`. V3 serialized every exception into the same `failures` array, so
the predeclared `no_unexpected_failures` gate is false. The valid temporary checkpoint
was inspected, backed up, and manually promoted before resuming. The completed method
has a full reconstructed seed ledger and the final artifact passes the independent
integrity audit, but this intervention means the run is not an uninterrupted frozen
confirmation.

At the same time, the result is valuable scientific evidence. All predeclared
performance sub-gates other than the execution-failure gate passed, the full 640-cell
matrix completed, and the independently reconstructed statistics equal the serialized
statistics. This is a **strict protocol failure with strong recovered performance
evidence**, not a numerical-method falsification.

## 2. Frozen gate results

| Gate | Threshold | Observed | Decision |
|---|---:|---:|---|
| Protocol complete | 640 cells | 640/640 | pass |
| DCS target attainment | at least 90% | 91.40625% | pass |
| Matched target cells | at least 3 | 254 | pass |
| Matched geometric work ratio | greater than 1.25 | 16.5177 | pass |
| Independent seed clusters | at least 10 | 20 | pass |
| One-sided 95% cluster lower bound | greater than 1 | 14.6992 | pass |
| No unexpected execution failures | zero | one `PermissionError` | **fail** |
| Frozen headline | all gates | false | **fail** |

The raw defensive estimator attained the requested empirical RMSE in 40.625% of cells.
DCS attained it in 91.40625%. Eleven completed cells support only a resource-censored
raw work-ratio lower bound; those cells are not included in the 16.5177 matched-RMSE
headline. The censored-cell geometric lower bound is 66.5820 and remains diagnostic.

## 3. Integrity and recovery audit

The independent auditor does not call the runner's summary function. It reconstructs:

- the 640-cell regime/task/probability/replicate matrix;
- every target flag, standard error, confidence interval, work ratio, and frozen gate;
- all 1,280 method seed ledgers from pilot and final batch metadata;
- the aggregate seed-evidence hash;
- the frozen config bytes, Git tag, core-source blobs, and calibration inputs; and
- cluster-level uncertainty and reference-consistency diagnostics.

The audit reports:

- `integrity_passed = true`;
- zero integrity failures;
- one recovered operational incident and zero unresolved incidents;
- `scientific_gate_passed = false`, preserving the frozen decision; and
- all performance sub-gates passed when the recovered operational incident is shown
  separately as a post-hoc diagnostic.

Two methods resumed from checkpoints. Their operation-work ledgers and sampler wall
times survive. V3 did not persist process CPU and orchestration time at every checkpoint,
so CPU time spent after the last successfully published checkpoint and before an
external interruption cannot be fully reconstructed. Operation work, not process CPU,
is the frozen performance metric.

## 4. What the aggregate headline hides

The 16.52x matched-cell aggregate is not a universal advantage. Performance weakens
in the regime labelled low-H and at the rarest probabilities.

The three M7 regimes are **not** a one-factor H experiment: `(H, eta, rho)` equals
`(0.07, 1.3, -0.7)`, `(0.12, 1.1, -0.6)`, and `(0.30, 0.8, -0.4)`. Consequently the
table below is descriptive by regime and cannot identify H, eta, or rho as the causal
driver. Parameter-separated one-factor-at-a-time experiments are mandatory in V4.

| Slice | DCS attainment | Matched cells | Matched geometric raw/DCS work | Main issue |
|---|---:|---:|---:|---|
| low-H-labelled combined regime | 78.125% | 18/160 | 29.21x | unstable rare-cell allocation and many DCS misses |
| primary combined regime | 93.333% | 65/240 | 17.63x | `1e-6` excursion/barrier tails |
| high-H-labelled combined regime | 98.333% | 171/240 | 15.18x | comparatively stable |
| terminal | 95.0% | 90/240 | 21.35x | low-H `1e-6` allocation tail |
| barrier | 87.083% | 81/240 | 12.94x | worst task-level attainment |
| excursion | 92.5% | 83/160 | 15.87x | primary-H `1e-6` instability |
| probability `1e-3` | 98.125% | 117/160 | 17.60x | stable |
| probability `1e-6` | 76.25% | 31/160 | 9.53x | dominant failure slice |

The most important 20-replicate groups are:

| Group | DCS misses | Matched cells | Matched raw/DCS work |
|---|---:|---:|---:|
| low-H barrier `1e-6` | 15 | 1 | **0.015x** |
| low-H terminal `1e-6` | 10 | 0 | n/a |
| primary-H excursion `1e-6` | 8 | 2 | **0.522x** |
| low-H barrier `1e-5` | 7 | 0 | n/a |
| primary-H barrier `1e-6` | 3 | 3 | **0.752x** |

Therefore DCS-MLMC can be substantially worse than its raw comparator in some of the
hardest matched cells. A paper must not claim uniform dominance from the aggregate.

## 5. Allocation-tail evidence

Across all DCS allocations, the median per-level final count is 2,966, the 99th
percentile is about 910,127, and the maximum is 31,874,693. In the low-H regime the
median is 24,836, the 99th percentile is about 2.26 million, and the maximum remains
31.87 million. Raw's largest uncapped allocation is 690,388,637, which explains why a
finite resource-censoring rule is necessary.

The frozen safety-factor allocation controls its design variance, but it does not
guarantee that a noisy pilot estimate will attain the empirical final-sample target.
The observed 55 DCS misses and the extreme count tail make allocation uncertainty a
paper-critical issue rather than an implementation detail.

## 6. Reference consistency

Against independent calibration/validation estimates, DCS has median absolute
standardized discrepancy 0.687, mean signed discrepancy 0.038 standard deviations,
and 93.75% of 640 comparisons inside an independent 95% interval. The largest absolute
standardized discrepancy is 3.30. This supports absence of a broad estimator bias
signal, but it is not a formal multiple-testing pass and it does not validate a
continuous-time estimand.

Raw has wider discrepancies because many cells do not attain their requested RMSE:
its median absolute standardized discrepancy is 0.863 and 88.91% lie inside the same
diagnostic interval.

## 7. Result-driven theory decision

The most appropriate V4 theory is a **margin-localized rough-Volterra threshold
stability and finite-level crossover theory**.

It has two linked parts:

1. On a good event where active affine slopes are bounded below, bound the fine/coarse
   scalar-threshold difference by coefficient errors plus an explicit mesh-enrichment
   or order-statistic defect. Retain the bad-event probability instead of assuming a
   global denominator lower bound that is false near time zero.
2. Convert this bound into defensive raw and DCS second-moment bounds, then combine
   observed single-level and correction variance/cost profiles in a non-asymptotic
   crossover rule. The production estimator chooses DCS single-level IS when the
   multilevel construction is not predicted to be cheaper.

This theory is selected because the weakness is localized to the low-H-labelled
combined regime, barrier active-index instability, and the rarest targets. M7 alone
cannot attribute the regime effect to H because eta and rho were changed with it. A
generic common-likelihood theorem would not address the observed failure and is
already prior art. A universal claim that DCS always improves MLMC is contradicted by
the result.

## 8. Journal-readiness verdict

The project remains a PhD-level working-paper core, not yet a top-journal-ready
submission.

Positive evidence:

- exact finite-grid estimator identities and defensive bounds;
- a large, predeclared 20-cluster experiment;
- strong aggregate work reduction with a large cluster lower confidence bound;
- reproducible provenance and independently reconstructed seed evidence; and
- transparent retention of failed hypotheses and resource-censored baselines.

Blocking issues:

1. the frozen V3 headline failed and required manual checkpoint recovery;
2. the strongest DCS single-level IS and task-tuned CEM baselines are not yet in the
   confirmatory matrix;
3. the low-H-labelled combined regime and `1e-6` results reject uniform efficiency
   claims;
4. the rough-Volterra threshold-rate assumption is not yet derived from model-level
   coefficient, margin, and mesh-enrichment controls;
5. the current result is a fixed 128-step estimand, not a continuous-monitoring
   theorem; and
6. independent-environment reproduction remains open.

In addition, the M7 regime design confounds H, eta, and rho, so it cannot support a
parameter-specific sensitivity claim.

V4 must be qualified before a new untouched seed namespace is frozen. The recovered
V3 data may guide theory and qualification, but it may not be relabeled as the final
confirmation.

## 9. Allowed and prohibited claims

Allowed:

- “In a recovered 640-cell frozen experiment, all scientific performance sub-gates
  passed, while the strict headline failed because of one recorded checkpoint I/O
  incident.”
- “Among 254 matched-RMSE cells, the raw/DCS operation-work ratio had geometric mean
  16.52 and a one-sided cluster lower bound of 14.70.”
- “Performance was not uniform; low-H and `1e-6` slices were materially weaker.”

Prohibited:

- “M7 V3 passed.”
- “DCS-MLMC uniformly dominates raw MLMC or single-level IS.”
- “The 16.52x ratio applies to all cells.”
- “The current experiment proves a continuous-time rough-Bergomi complexity rate.”
- “Common-likelihood MLIS or conditional smoothing is new by itself.”
