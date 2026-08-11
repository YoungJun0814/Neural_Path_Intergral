# G11 V16 operator V1 measurement audit

Date: 2026-08-11

## Decision

The V1 development artifact is retained as positive evidence that a learned
initializer can lie in the attraction basin of the corrected regular-regime mode.
It is **not** evidence of end-to-end operator speedup.

All six held-out tasks reached the same stationary action as a zero-start solve,
without natural fallback, and the separately measured one-start L-BFGS solves used
a median of `1/1.5833` times as many action evaluations as zero-start solves.

## Measurement defect found after execution

`correct_and_build_operator_transport` called the common mode-search routine.  That
routine always added a zero start and the V1 configuration also requested a random
start.  Consequently the actual correction executed three starts (zero, learned,
random), whereas the speed table compared only a separate learned-start solve with
a separate zero-start solve.  The table answers an initializer-quality question,
but not the actual pipeline-cost question.

No estimator bias or likelihood error follows from this defect.  It affects only
the amortized-cost interpretation.

## Corrective action

The mode-search contract now has an explicit `include_zero_start` flag and permits
zero random starts.  Operator confirmation must use exactly the supplied learned
start, then either:

1. certify the corrected positive-curvature stationary point and build the exact
   defensive proposal; or
2. fail closed to the natural proposal.

The confirmation artifact must count the correction attempts actually contained in
the certificate, compare against cold start, include multiple training seeds and a
shuffled-teacher control, and disclose teacher construction and neural training
costs separately.  V1 remains immutable at its recorded source commit.
