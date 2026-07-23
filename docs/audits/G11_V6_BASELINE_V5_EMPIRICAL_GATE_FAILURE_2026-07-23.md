# G11 V6 Baseline V5 Per-Record Empirical-Gate Failure

Date: 2026-07-23

Decision: stop and preserve V5 at the first observed per-record empirical-RMSE
miss; do not increase the safety factor from the observed outcome; replace the
all-record decision rule with the aggregate accuracy rule already prespecified in
the V6 research plan

## 1. Preserved failure

V5 ran from clean commit
`fb4dd6a6fe49db7a9c004be28e807b606b8feaf9`. GitHub CI passed on Python 3.10
and 3.11 before execution. The process was stopped with 198 durable records and
zero stderr bytes. Its external checkpoint root is:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_formal\commit_fb4dd6a\qualification\baseline_v5`

The first detected failure was:

| field | value |
|---|---:|
| cell | `h005-terminal-p1e-04` |
| cluster | 16 |
| method | `pure_cem` |
| fitted-control convergence | pass |
| resource censoring | false |
| final count | 5,108 |
| allocation-pilot variance | `4.08626869178339e-7` |
| final single-path variance | `1.42678513410377e-5` |
| final/pilot variance ratio | `34.9165765` |
| empirical sampling variance | `2.79323636277167e-9` |
| requested sampling-variance target | `4.0e-10` |
| empirical/target ratio | `6.9830909` |
| reference z-score | `0.9461` |

The record is not evidence of estimator bias: its reference z-score is below one.
It is evidence that one 4,096-path plug-in pilot, even with a fivefold multiplier,
does not upper-bound a heavy-tailed pure-CEM variance realization.

## 2. Why the safety factor is not retuned

Changing the factor from 5 to 35 or 40 after seeing this record would be
outcome-dependent allocation tuning. It would not provide a finite-sample guarantee
for the next rarer likelihood contribution and would contaminate a later efficiency
comparison. V6 therefore leaves the pure-CEM and defensive-CEM allocation rules
unchanged.

The 198 preserved records also show why the observation must not be generalized into
a new constant. Excluding zero-variance crude pilots, the pure-CEM final/pilot ratio
had median `0.931`, 95th percentile `1.773`, and maximum `34.917`. The isolated
maximum is a tail event, not a stable calibration coefficient.

## 3. The actual protocol error

The baseline runner promoted

`all_empirical_targets_attained = all(record empirical targets)`

into the baseline qualification decision. This requires 100% success over 1,296
random final-variance diagnostics. It is not a finite-sample coverage statement, and
its probability of failure grows mechanically with the number of records.

More importantly, it contradicts the already written V6 plan. Section A3 of
`G11_V6_DUAL_TRACK_PHD_IMPLEMENTATION_PLAN_2026-07-23.md` prespecifies accuracy at
the method-by-cell level:

1. the target-attainment rate must pass a one-sided exact lower-confidence gate; and
2. a cluster bootstrap RMSE upper bound, including reference uncertainty, must pass
   the frozen engineering tolerance.

Those co-gates are already implemented in `g11_v6_power_analysis.py` and
`g11_v6_confirmatory.py`. They retain every failed record and prohibit
complete-case deletion.

## 4. V6 correction

V6 makes no estimator, CEM-training, likelihood, pilot, final-count, or safety-factor
change. It:

- preserves `all_empirical_targets_attained` as a serialized diagnostic;
- excludes that random all-record diagnostic from operational baseline
  qualification;
- serializes the exact operational gate list and matrix contract;
- defers scientific accuracy to
  `g11-v6-power-analysis-qualification-v1`;
- requires the independent auditor to reconstruct the matrix, every summary gate,
  the operational/diagnostic split, and the resulting qualification decision; and
- uses a new protocol ID and seed namespace.

Thus a per-record miss is neither hidden nor repaired. It remains in the aggregate
attainment and RMSE analyses. Qualification or later confirmation still fails if
either prespecified aggregate accuracy co-gate fails.

## 5. Claim consequence

V5 is a failed, inadmissible formal attempt and remains part of the audit trail.
V6 may become an operationally qualified baseline source, but it cannot establish
accuracy or superiority by itself. Those claims require the separately frozen
aggregate accuracy analysis, paired V6-policy results, power/resource acceptance,
untouched confirmation, and independent reproduction.
