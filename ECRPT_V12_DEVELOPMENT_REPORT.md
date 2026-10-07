# ECRPT V12 Development Implementation and Audit Report

Date: 2026-08-11
Topic: **Exact Conditional Residual Path-Space Transport for Rare Events in Gaussian
Volterra Models**
Decision: software correctness branch complete; performance/qualification branch closed

## 1. What was implemented

### P0: tail-safe inference

- mergeable Chan/Welford sufficient statistics;
- bounded one-sided second-moment and variance certificate;
- IID/RQMC-specific minimum inferential-unit rules;
- resource censoring and an independent final certificate;
- explicit rejection of the zero-pilot/zero-standard-error pathology.

Primary module: `src/path_integral/tail_safe_allocation.py`.

### P1: exact residual transport

- Gaussian residual hyperplane projection and sampling;
- defensive translated-Gaussian mixtures with exact `p_R/q_R`;
- stable nonnegative and signed ordinary contributions;
- weighted CE from target or exact importance samples;
- canonical frozen proposal hashes;
- analytic half-space oracle tests.

Primary module: `src/path_integral/residual_transport.py`.

### P2: finite-grid rBergomi ECRPT

- positive price-block direction family and training-only selection;
- exact terminal/barrier scalar thresholds after residual conditioning;
- paired raw/ECRPT evaluation on the same frozen residual proposal;
- full latent/path reconstruction and likelihood-bound diagnostics;
- one-pass, adaptive, and annealed residual CE variants;
- training-inclusive work ledger, frozen micro-study protocol, and result auditor.

Primary modules: `src/path_integral/rbergomi_residual_transport.py`,
`src/path_integral/ecrpt_protocol.py`, and
`src/path_integral/ecrpt_microstudy_audit.py`.

### P3: task-conditioned correctness infrastructure

- deterministic task-feature network trained from teacher proposals;
- mixture-label canonicalization;
- positive price direction, orthogonal means, and defensive weights by construction;
- exact frozen proposal emission whose likelihood is independent of generator accuracy.

Primary module: `src/path_integral/amortized_residual_transport.py`.

### P4: nonlinear and multilevel extensions

- exact Householder residual coordinates;
- exact defensive triangular affine-coupling flow;
- common-coordinate fine/coarse rBergomi thresholds;
- stable signed CDF difference;
- `|G|` proposal training with sign restored in the correction;
- level-zero and adjacent correction adapter for the existing MLMC engine.

Primary modules: `src/path_integral/residual_coupling_transport.py` and
`src/path_integral/rbergomi_residual_mlmc.py`.

## 2. Mathematical audit

The implemented estimator is

\[
\widehat p_n=\frac1n\sum_{i=1}^n
g(R_i)\frac{p_R(R_i)}{q_R(R_i)},\qquad R_i\sim q_R.
\]

The following are exact for the declared finite grid:

1. `A=e^T X` is integrated only when `e` has zero volatility/local entries and
   strictly positive independent-price entries.
2. All likelihoods are densities on the Gaussian hyperplane `e^perp`, not singular
   ambient Lebesgue densities.
3. The natural component of weight `delta` gives `p_R/q_R <= 1/delta`.
4. Training-weight normalization is confined to cross entropy; final estimates are
   ordinary, non-self-normalized means.
5. Adaptive round weights are exactly `g^beta p_R/q_k`; the final estimate always
   returns to `beta=1` and uses fresh data.
6. Signed ML corrections train against `|G|` and retain sign in the estimator.
7. Task-conditioned and nonlinear-flow approximation error affects variance only;
   emitted/sampled proposal densities remain exact.

No theorem currently proves that weighted forward-KL training decreases chi-square
divergence or variance.  That distinction is both documented and enforced by the
held-out performance gate.

## 3. Development experiments

The frozen matrix contains three terminal left-tail cells and four independent
training/evaluation clusters per cell.  The primary comparators are exact conditional
rBergomi, smoothed RQMC, and V10R1 full-latent CEM-DCS.  The target relative RMSE is
20% and total work amortizes one-time training over `K=100` queries.

| Namespace | Candidate | Exact / paired / likelihood | Best-primary over ECRPT geometric ratio | One-sided lower | Cells favoring ECRPT | P2 pass |
|---|---|---:|---:|---:|---:|---:|
| V1 | one-pass CE | pass / pass / pass | 0.5281 | 0.1765 | 1/3 | no |
| V2 | 3-round adaptive exact CE | pass / pass / pass | 0.1625 | 0.0487 | 1/3 | no |
| V3 | annealed CE | pass / pass / pass | 0.0140 | 0.0035 | 0/3 | no |

For every namespace, candidate accuracy, primary-comparator accuracy, the finite-sample
paired-variance gate, and the no-censoring requirement fail.  Every candidate record
(`12/12`) and every primary-comparator record (`36/36`) is resource-censored by the
conservative tail-safe forecast at the laptop cap.  These facts make a positive work
comparison inadmissible even before considering the unfavorable descriptive ratios.

Independent audits pass for all namespaces:

- V1 result SHA-256: `ef923673e2f8b941d2899e32a1e0db25b2830cbbf6987c3e49c37aa3aad18d23`;
- V2 result SHA-256: `b630afc666ae1e667b9f5386b550fb6b0cca53ce5fc7707bc13830881675f41a`;
- V3 result SHA-256: `6e9b3e8987c15fc7b52487e6b5bfa08a1bb385d010ffe7c340dd56053c74eea0`.

Each audit independently reconstructs the seed ledger, proposal hash, sufficient-
statistic moments, record roster, aggregate gate, and decision.

## 4. Interpretation of the negative result

This is not evidence of estimator bias.  Exactness diagnostics and paired mean
identities pass.  It is evidence that the current translated-Gaussian residual family
and CE training budgets do not reliably cover the rare local-volatility regions in
384 dimensions.  At probabilities near `10^-4` and `10^-5`, one-pass weights collapse;
adaptive and annealed variants can follow isolated training paths and incur more work
without consistent held-out coverage.

The empirical raw/ECRPT variance ratio can be below one when the finite raw sample
contains no rare hits.  That does not contradict Rao--Blackwell: the population
conditional variance inequality still holds.  It means the finite sample cannot
resolve the raw variance, so the fail-closed gate correctly refuses the mechanism
claim.

## 5. Objective publication assessment

The implementation and finite-dimensional theory are research-grade infrastructure,
but the current evidence is not a top-journal result and does not authorize
qualification.  Exact conditionalization by itself is established methodology; a
strong paper still needs both:

1. a nontrivial theorem, such as a Gaussian-Volterra transport stability/rate result
   or strict efficiency condition for a structured residual family; and
2. uncensored, accurate, training-inclusive evidence against strong exact conditional,
   RQMC, CEM-DCS, and flow/SMC comparators.

## 6. Next scientifically justified implementation

Priority 1 is not another dense Gaussian-mixture sweep.  It is a linear-cost,
Volterra-structured residual transport (low-rank/causal conditioner) trained with a
bridge method that controls weight ESS, followed by a fresh three-cell falsification.
The dense exact coupling flow implemented here is the density-correct oracle for that
work, but must be replaced by an `O(d r)` or causal architecture before a fair
training-inclusive benchmark.

Priority 2 is an uncensored evidence design.  It needs larger independent
randomization/path budgets, sharper but still valid bounded confidence sequences, and
an external compute resource.  A plug-in zero variance may not be used to reduce this
cost.

Priority 3 is the paper theorem.  Until a strict-efficiency, stability, or rough-
Volterra rate statement is proved and externally reviewed, the honest manuscript
position is an exact framework plus falsification study, not a top-journal performance
claim.
