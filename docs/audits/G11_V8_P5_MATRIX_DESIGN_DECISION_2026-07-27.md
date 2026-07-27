# G11 V8 P5 Reference and Comparison-Matrix Design Decision

Date: 2026-07-27

Decision: **PASS FOR OUTCOME-BLIND DESIGN; CALIBRATION, REFERENCES, AND PERFORMANCE OPEN**

P5 freezes the development comparison design before any V8 performance result is
generated.  It is a protection against a common but serious research error: choosing
the rare-event thresholds, comparator set, or aggregation rule after observing an
estimator's results.

## Frozen primary matrix

The primary matrix has 24 cells:

\[
H \in \{0.05, 0.12, 0.20\},\qquad
\text{task} \in \{\text{terminal left tail},\text{ discrete lower barrier}\},
\qquad p \in \{10^{-2},10^{-3},10^{-4},10^{-5}\}.
\]

It fixes the model parameters \(S_0=100,T=1,\xi=0.04,\eta=1.5,\rho=-0.7\),
uses a declared finite-grid estimand, and requires one-factor-at-a-time robustness
checks.  The comparison matrix contains crude MC, antithetic MC, conditional
rBergomi, pure/defensive CEM, smoothing RQMC, large-deviation subspace IS, and an
exact-likelihood coupling-flow IS baseline under the P4 common framework.

## Anti-leakage rules

- Thresholds must be calibrated before reference evaluation, stored in a separate
  hash-bound artifact, and never retuned using reference or final samples.
- `dcs_reference` and `raw_crosscheck` are separate code paths from evaluated final
  methods.  They must agree within the frozen combined-z diagnostic bound.
- Threshold calibration, references, reference-final samples, baseline training,
  pilots, and final-method samples have six distinct seed namespaces.  The audit
  fails closed on any overlap.
- A reference standard error may use at most 10% of the final target; its uncertainty
  enters every accuracy calculation.
- Each trained baseline is retrained per task and cell.  Final performance claims
  remain prohibited until thresholds, independent references, and actual runs are
  completed.

## What this decision does and does not establish

The design audit passes only as a *protocol* result.  It establishes neither an
observed DCS advantage nor a top-journal claim.  It authorizes the next phase: P6
cluster-level statistical design and the implementation of threshold/reference
artifacts.  Actual thresholds, independent reference values, and performance
outcomes are deliberately absent.

## Verification

The fail-closed checker is
`experiments/g11_v8_p5_matrix_audit.py`; its canonical and corruption tests are in
`tests/test_g11_v8_p5_matrix_audit.py`.  It checks the locked 24-cell matrix,
upstream P4 framework hash, exact comparator roster, code-path separation, six-way
seed isolation, reference precision, mesh diagnostics, and the prohibition on
performance claims.
