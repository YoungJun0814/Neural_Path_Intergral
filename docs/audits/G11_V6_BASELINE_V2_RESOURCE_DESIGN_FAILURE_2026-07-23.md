# G11 V6 Baseline V2 Resource-Design Failure

Date: 2026-07-23

Decision: stop V2 without completing it; preserve all checkpoints; replace it with
the separately versioned V3 protocol

## 1. What failed

`g11-v6-baseline-qualification-v2` used a one-look Hoeffding upper confidence bound
for the variance of the defensive-CEM contribution. The contribution has the exact
finite-grid form

`Y = 1_A dP/dQ`

and the proposal contains natural-law mass `delta=0.20`, so `0 <= Y <= B=5`.
With `n=4096`, familywise alpha `0.05`, one profile, and one predeclared look, the
second-moment radius alone is

`B^2 sqrt(log(2/(0.05/2))/(2n)) = 0.5782059325`.

This bound is valid but too loose for an achieved-relative-RMSE experiment. For the
`1e-3` target it implies about `14,455,149` final paths. For the `1e-4` target it
implies about `1,445,514,832` final paths and approximately
`1.295e12` operation-work units, far above the predeclared `5e10` cap.

Therefore V2 was mathematically guaranteed to become resource-censored on a
sentinel cell even in the best possible zero-observation pilot. Continuing it could
not produce a qualified artifact.

## 2. Preserved evidence

The stopped external run remains at:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_formal\commit_8fbddf0\qualification\baseline_v2`

At termination:

- 74 complete records were durably journaled;
- the in-progress record was
  `h005-terminal-p1e-03 / cluster 0 / defensive_cem`;
- it had executed `10,723,328` of `14,455,502` requested final samples; and
- its record-level checkpoint, seed ledger, work ledger, stdout, stderr, and PID
  receipt were retained.

No V2 record is admitted to V3. V3 uses a new protocol ID and seed namespace.

## 3. Why this is a protocol defect rather than a negative model result

The issue was detected from the deterministic allocation formula and the resource
cap, before reaching any `1e-4` baseline result. It does not show that defensive CEM
is inaccurate or inefficient. It shows that the chosen finite-sample variance
certificate is unusably conservative at rare-event scale.

The primary Route-A comparator remains pure CEM. V3 does not change the pure-CEM
algorithm, training rule, safety factor, target, or primary endpoint. The correction
only prevents the secondary defensive baseline from being artificially penalized by
an irrelevant `B^2` Hoeffding radius.

## 4. V3 mathematical correction

For nonnegative `Y <= B`,

`Var_Q(Y) <= E_Q[Y^2] <= B E_Q[Y] = B P(A)`.

The manifest was calibrated in the predeclared band

`P(A) <= 2 p_nominal`.

V3 therefore uses

`V_design = max(V_pilot, min(V_Hoeffding,UCB, B * 2 p_nominal))`.

This change is valid only with a certificate that the independent reference remains
inside the frozen rarity band. V3 requires, for every cell,

`p_reference + 4 SE_reference <= 2 p_nominal`

before any baseline execution. The numerical reference value is not used in the
allocation; it only certifies eligibility of the already frozen band. All 18
accepted references satisfy this condition. The largest observed ratio was about
`1.110`, below the predeclared multiplier `2.0`.

The resulting worst structural variance designs are:

- `0.10` for `p_nominal=1e-2`;
- `0.01` for `p_nominal=1e-3`; and
- `0.001` for `p_nominal=1e-4`.

At relative sampling RMSE `0.20`, the corresponding defensive final allocations are
at most approximately `2.5M` paths before minimum-count rounding, safely below the
operation-work cap.

## 5. Bias, leakage, and accuracy audit

The V3 correction does not recycle final observations or normalize weights:

- proposal training and allocation pilots remain independent of final samples;
- the final estimator remains the ordinary exact-likelihood sample mean;
- the rarity multiplier is fixed globally, not fitted per cell;
- the reference is used only for a pre-execution band certificate and post-execution
  accuracy scoring;
- integer allocation remains frozen before final sampling;
- empirical final sampling variance must still attain the requested target; and
- an independent JSON-only auditor recomputes the band certificate, Hoeffding upper
  bound, structural second-moment bound, selected design variance, allocation,
  likelihood-result statistics, work, and seed roles.

If the final empirical-RMSE gate fails, V3 fails. It may not adapt the allocation
after observing final samples.

## 6. Verification completed before the V3 full run

- focused baseline and independent-auditor tests pass;
- Ruff passes on all changed Python files;
- an actual terminal/barrier rBergomi smoke matrix completes without censoring;
- every smoke design and empirical target passes;
- every defensive design certificate passes; and
- the independent auditor accepts all records and rejects only the expected
  non-smoke qualification requirement.

Smoke CEM convergence is intentionally not a scientific pass because smoke training
uses only two iterations. The full V3 gate still requires every CEM fit to converge.

## 7. Publication interpretation

V2 cannot be cited as evidence for or against the proposed method. It is a preserved
protocol failure. V3 is admissible only after a clean committed full rerun, all
method and accuracy gates, the independent audit, and the later outcome-blind
confirmation. This correction should be disclosed in the reproducibility history.
