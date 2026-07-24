# G11 V6 post-hoc accuracy-multiplicity sensitivity

Date: 2026-07-24  
Analysis source: `4c9adc68a17bce269d2c76a30cc11a5807f1da94`  
Protocol: `g11-v6-accuracy-multiplicity-sensitivity-v1`  
Decision: both completed environments pass the post-hoc sensitivity

## 1. Scope and non-retroactivity

This analysis asks whether the 36 method-by-cell accuracy groups from each
completed V6 experiment remain inside their engineering tolerances after a
conservative nominal family-wise adjustment.  It does **not** alter the primary
work-ratio estimand, rerun either estimator, discard a record, or change the
original V2 decision.

The config and analyzer were committed before these sensitivity results were
computed.  They were not, however, part of the original confirmation freeze.
Therefore:

> this is a frozen post-confirmation robustness analysis, not a
> preregistered simultaneous confirmatory analysis.

No manuscript may use this artifact to make a retrospective preregistration
claim.

## 2. Claim family and procedure

The family contains:

- 18 cells;
- two methods, pure CEM and the V6 policy; and
- two accuracy gates per method-by-cell group: target attainment and RMSE.

This gives 72 claims.  With family-wise alpha \(0.05\), the Bonferroni
per-claim alpha is

\[
\alpha_\mathrm{claim}=\frac{0.05}{72}
=0.0006944444,
\]

so every one-sided bound is evaluated at nominal confidence
\(0.9993055556\).  Target-attainment lower bounds use exact
Clopper--Pearson inversion.  RMSE upper bounds use the already declared
nonparametric cluster bootstrap with 20,000 resamples and a fixed seed.

The analyzer fails closed on:

- an unexpected schema or nested config field;
- duplicate cell-cluster records;
- a baseline/policy key mismatch;
- an incomplete cell-by-cluster Cartesian product;
- any cell, method, gate, or cluster-count mismatch; and
- a dirty analysis source tree.

## 3. Results

| Quantity | Windows confirmation V2 | Linux reproduction V3 |
|---|---:|---:|
| cells / clusters / claims | 18 / 64 / 72 | 18 / 64 / 72 |
| minimum simultaneous attainment lower | 0.830951 | 0.830951 |
| required attainment rate | 0.600000 | 0.600000 |
| maximum RMSE-upper/tolerance ratio | 0.697440 | 0.631699 |
| all attainment gates | pass | pass |
| all RMSE gates | pass | pass |
| complete family and clean provenance | pass | pass |
| overall post-hoc sensitivity | **pass** | **pass** |

The Windows worst RMSE ratio is the pure-CEM
`h012-barrier-p1e-04` cell.  The Linux worst ratio is the pure-CEM
`h005-barrier-p1e-04` cell.  Both retain substantial margin below one.
The minimum attainment bound comes from 62 successes in 64 clusters for the
pure-CEM `h005-barrier-p1e-04` cell.

## 4. Artifact identity

Config SHA-256:

`7a45a08e408a0a14bdb7a10a866fdd2044a01390a0de1cba51730cb1f240e2b9`

| Artifact | SHA-256 |
|---|---|
| Windows sensitivity JSON | `92ada6b30d597a949f3a3fae10c77b05953bc8b9299143a3f7c1ee8deef98d57` |
| Linux sensitivity JSON | `c4e19cd1457a732c69c4e03d9e8c40a40887655c0822fe36ac922f93cfec5e50` |
| Windows baseline input | `d96aed84237a057853f0bd9ef7821d69897ffa3de2578ace287cdeaa185c68d1` |
| Windows policy input | `8f5d9f8a94ef70a8ad02ecb84ed9e965321c5e336024b63d56a28ae4cc14ccbc` |
| Linux baseline input | `84a6e42981668cfeae2860df6f4a744f008032167197c4304c9dbec155c544d3` |
| Linux policy input | `d9971e4154098f582035daf6b7e1d44b1c549123ce98ea9cf1dcdf80a33ef16f` |

The JSON artifacts live outside the Git worktree under:

`NPI_formal/commit_4c9adc6/accuracy_multiplicity_sensitivity_v1/`.

## 5. Statistical limitation

Bonferroni makes no independence assumption and the attainment intervals have
finite-sample binomial coverage.  The RMSE component remains a percentile
bootstrap engineering bound, not an exact finite-sample theorem.  At the
0.9993056 quantile, 20,000 resamples provide about fourteen expected draws in
the upper tail; this is adequate for the observed large-margin sensitivity
check but not for claiming exact simultaneous RMSE coverage.

Consequently the correct conclusion is:

> the accuracy result is robust to a severe post-hoc nominal family-wise
> adjustment in both disjoint-seed environments.

The incorrect conclusion would be:

> the original V2 accuracy family had preregistered exact family-wise error
> control.

V7 must place the simultaneous procedure in its pre-outcome freeze and should
either predeclare bootstrap Monte Carlo-error handling or replace the RMSE gate
with a validated finite-sample alternative if an exact coverage claim is
required.
