# G11 V8 R2L — Method-specific production-scale proposal protocol

Date: 2026-07-31

## Purpose

The cell-tuned V4 reference pilot left 8 of 48 cell-method requirements
infeasible.  This protocol changes only the proposal-development layer.  It
does not change the rBergomi estimand, event definitions, fixed finest grid,
likelihood identity, reference precision target, allocation cap, or the final
method.

## Mathematical admissibility split

Let \(P\) be the natural innovation law and let

\[
Q=\sum_{k=0}^{K-1} w_k Q_k,\qquad w_k>0,\quad \sum_k w_k=1,
\]

be a frozen finite mixture of deterministic Gaussian shifts.  The raw
importance-sampling estimator is

\[
\widehat p_N={1\over N}\sum_{i=1}^N
1_A(X_i){dP\over dQ}(X_i),\qquad X_i\sim Q.
\]

It is unbiased for every such mixture when the exact balance-mixture
likelihood is evaluated.  No rank-one condition is present in this identity.
The first expert is fixed to \(Q_0=P\), hence

\[
Q\ge w_0P,\qquad 0\le {dP\over dQ}\le {1\over w_0}.
\]

The raw proposal therefore combines three independently trained CEM temporal
profiles, rather than amplitudes of only one temporal profile.  This gives
coverage of multiple rare-event path geometries while retaining an explicit
likelihood bound.

The implemented DCS estimator is different.  It analytically integrates the
Gaussian coordinate along a common price-control direction.  Its current
identity requires all expert price-driver shifts to lie in one strictly
one-signed rank-one span.  DCS proposals therefore use amplitude mixtures of
one trained profile at a time, and every candidate is rejected unless the
exact structural check passes.

## Frozen failed requirements

The redesign is bound by hash to the audited V4 allocation failure and targets
exactly its eight infeasible requirements:

- five `raw_crosscheck` requirements;
- three `dcs_reference` requirements;
- seven unique rBergomi task cells.

Adding, removing, or changing a target invalidates configuration loading.

## Independence contract

The following namespaces are new and immutable:

- training: `v8-r2-production-scale-proposal-training-v1`;
- held-out validation:
  `v8-r2-production-scale-proposal-validation-v1`.

Every seed key contains protocol, stage, candidate, cell, replicate, and
namespace.  The result builder rejects any derived-seed collision.  Neither
namespace may be reused for a formal reference pilot.

## Production-relevant validation

Each candidate is evaluated on 8 independent replicates of 32,768 paths, the
same per-replicate size used by the formal reference pilot.  Each replicate is
also divided into 8 non-overlapping blocks of 4,096 paths.

Allocation prediction uses the largest value among:

- all eight full-replicate sample variances; and
- all 64 within-replicate block sample variances.

The block maximum is deliberately conservative.  A rare large contribution
has more influence on a 4,096-path block variance than on the enclosing
32,768-path variance, so this gate is less likely to hide the tail behavior
that invalidated V4.

## Candidate gates

A candidate passes only if all applicable conditions hold:

1. paths, likelihoods, and contributions are finite and path states positive;
2. the observed likelihood respects the exact \(1/w_0\) defensive bound;
3. pooled likelihood normalization has absolute z-score at most 4;
4. six-times-safety-factor projected allocation is at most 50% of the fixed
   cap of 8,388,608;
5. maximum block variance divided by median block variance is at most 30;
6. maximum single-contribution share is at most 5% on a full replicate and
   25% on every block;
7. raw candidates have at least 1,024 nonzero contributions per replicate and
   64 per block; and
8. DCS candidates pass the exact rank-one price-span check.

These are falsification gates, not proofs of a universally bounded relative
error.  Passing them authorizes an immutable manifest audit and a new fresh
pilot only; it does not authorize final execution or a performance claim.

## Pre-execution review

- Adaptedness: all proposal controls are deterministic functions of time.
- Likelihood: exact balance-mixture likelihood; no component-only likelihood.
- Self-normalization: forbidden and not used.
- Target leakage: event thresholds are frozen by the bound P5 manifest.
- Method mismatch: full-rank mixtures are evaluated only by the raw estimator.
- DCS structure: checked before simulation and again in the result.
- Seed leakage: training and held-out validation namespaces are distinct.
- Outcome peeking: the protocol is informed by prior burned pilots, disclosed,
  and frozen before its own namespace is executed.
- Claim discipline: all downstream authorizations remain false.

## Current authorization

- Execute this frozen development protocol: authorized
- Freeze selected proposals before an audit: forbidden
- Open another formal pilot: forbidden
- Execute final references: forbidden
- Make performance or submission claims: forbidden
