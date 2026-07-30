# G11 V8 R2Q — Exact raw mixture-weight optimization protocol

Date: 2026-07-31

## Scope

This stage retains the three V3 development selections and targets only the
five unresolved cell-method requirements.  It does not alter any reference
precision, allocation, concentration, likelihood-normalization, or structural
gate.

## Raw second-moment identity

For fixed expert densities \(q_0,\ldots,q_{K-1}\), define

\[
q_w=\sum_k w_kq_k,\qquad q_b=\sum_k b_kq_k.
\]

The raw importance-sampling contribution under candidate proposal \(Q_w\) is
\(Y_w=1_Ap/q_w\).  Its uncentered second moment is

\[
E_{Q_w}[Y_w^2]
=\int 1_A {p^2\over q_w}\,dx
=E_{Q_b}\left[
1_A {p\over q_w}{p\over q_b}
\right].
\]

The last expression is evaluated on training-only samples from the frozen base
mixture \(Q_b\).  Both likelihood ratios use exact all-expert balance
likelihoods.  No self-normalization or selected-component likelihood is used.

For fixed positive densities, \(1/q_w\) is convex in \(w\), so the population
second moment is convex on the simplex.  The implementation optimizes a smooth
simplex parameterization with Adam and retains the best explicitly re-evaluated
iterate.  The optimized empirical objective is required to be no larger than
the base objective.

## Defensive constraints

- Natural expert weight is fixed at 0.08.
- Every nonnatural expert has weight at least 0.005.
- Weights are positive and sum exactly to one within numerical tolerance.
- Therefore \(q_w\ge0.08p\) and \(p/q_w\le12.5\).
- Expert schedules are inherited by exact ID from the audited V3 result.

Four independent 32,768-path training replicates are used for weight fitting.
The subsequent candidate validation uses a different namespace and 8 new
replicates of 32,768 paths with independent permutation seeds.

## DCS separation

The raw off-policy objective is not applied to DCS.  The DCS contribution
analytically marginalizes a Gaussian coordinate and changes nontrivially with
the proposal mixture.  Reusing the raw formula without a new derivation would
not be justified.

Instead, DCS uses a finite preregistered grid of positive amplitude mixtures of
audited CEM profiles.  Every candidate:

- begins with the natural expert;
- has positive weights summing to one;
- keeps a strictly negative price-driver direction; and
- passes the exact rank-one price-span check before evaluation.

## Independence and selection

- V3 outcomes are disclosed development information.
- Weight-training and V4 validation namespaces are new.
- Proposal innovations, mixture labels, and diagnostic permutations have
  separate seed roles.
- All derived seeds are checked globally for collision.
- Candidate selection occurs only within this development stage.
- Even a complete selection cannot authorize a formal pilot until the result
  and complete proposal manifest are independently audited and committed.

## Pre-execution verification

- Exact raw off-policy identity: verified algebraically and by unit test.
- Positive defensive simplex: verified by unit test.
- DCS/raw method separation: enforced in code and test.
- Independent permuted blocks: inherited with a new seed namespace.
- Strict JSON finiteness: recursively enforced.
- Configuration and prior artifacts: hash bound.
- Twelve relevant protocol/audit tests: passing.
- Lint and changed-file type checks: passing.

## Authorization

- Execute weight-training and V4 development validation: authorized
- Freeze or promote proposals before audit: forbidden
- Open formal pilot/final namespaces: forbidden
- Make performance or submission claims: forbidden
