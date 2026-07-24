# G11 V7 mechanism development V2 decision

Date: 2026-07-24  
Estimator execution source: `f42369516aa439f203f4fe27577f50cad2f15d06`  
Analysis source: `e3bb257`  
Decision: development gate passes; qualification is authorized but not yet run

## 1. Question

V7 tests whether DCS itself, rather than a router or proposal-training
amortization, reduces achieved-RMSE work under:

- the identical frozen defensive proposal;
- the identical 128-step estimand;
- the identical 10% relative sampling-RMSE target;
- the identical final-sample floor of 512;
- independent production planning/final seeds; and
- a raw comparator that does not execute DCS arithmetic.

The paired mechanism probe is kept separate from production work.

## 2. V1 disposition

V1 completed but failed its maximum sample-correlation diagnostic.  The raw
maximum correlation was ill-scaled in heavy-tailed rare cells and had no
sampling-error normalization.  V1 remains failed and is not pooled with V2.
V2 uses a new protocol and seed namespace and tests covariance divided by its
centered-product standard error.  See
`G11_V7_MECHANISM_PROBE_V1_FAILURE_2026-07-24.md`.

## 3. V2 paired-probe result

The V2 probe used 18 cells, 8 clusters, and 4,096 common-path raw/DCS pairs per
cell-cluster:

| Quantity | Result |
|---|---:|
| records | 144/144 |
| unique seed values | 288 |
| geometric raw/DCS variance ratio | 3.5081 |
| one-sided 95% lower ratio | 3.3937 |
| p-value against ratio one | \(1.35\times10^{-11}\) |
| maximum absolute residual-mean z | 2.9785 |
| maximum absolute orthogonality-covariance z | 2.8277 |
| all V2 development gates | pass |

The unstudentized maximum sample correlation remains 0.6362.  Its coexistence
with a maximum covariance z of 2.8277 confirms that correlation normalization
is unstable in the rarest cells; it is retained only as a diagnostic.

Paired-probe artifact SHA-256:

`4fcfe68f8d1f97b2a47492643090021068a410f601b0b65f7a41de554993468b`

## 4. Fixed production result

The production study completed 144 DCS and 144 raw records.  Every run is
complete, uncensored, planning-certified, independently audited, and
empirically inside its per-record target diagnostic.

| Method | Final samples | Planning work | Final work | Total work | Final wall seconds |
|---|---:|---:|---:|---:|---:|
| fixed DCS-SLIS | 501,268 | 528,482,304 | 449,136,128 | 1,065,698,816 | 23.29 |
| fixed raw defensive | 1,748,265 | 528,482,304 | 1,566,445,440 | 2,183,008,128 | 60.75 |

No record in either method hits the common floor.

The independent V6-compatible JSON audit and resource supplement both pass.

| Artifact | SHA-256 |
|---|---|
| fixed-estimator result | `d212776cd6fdd1e882fbf8f5f482f00e771d9d5933cccd28fbc40b51e9901049` |
| independent audit | `2dea36f0fc8d8347c55591d47c24e9e8d70ecb1229758136cb117a61d78e5fc1` |
| resource supplement | `d5bc55930d47246ccb360f24d1930289e8d300d7365b6e185bcfdd4c1be26df1` |

## 5. Joint paired effects

Ratios are raw divided by DCS.  Cells are equally weighted inside each cluster,
then a one-sided 95% t lower bound is computed across 8 clusters.

| Metric | Geometric ratio | One-sided lower |
|---|---:|---:|
| empirical single-path variance | 3.6373 | 3.4051 |
| final work | 3.5897 | 3.3780 |
| planning + final work | 1.9765 | 1.8984 |
| training-inclusive total work | 1.8933 | 1.8216 |
| isolated final wall time | 2.5920 | 2.4394 |

All ten joint integrity/mechanism gates pass, including exact input identities,
disjoint probe/production seeds, same execution source, and zero floor
occupancy.

Joint analysis SHA-256:

`9142605b0dd95dc140dff5f8a2eda8de9fe21db0d7a184228b2256c2b24192fe`

## 6. Interpretation

This closes the main mechanistic ambiguity in V6 at development scale:

> under one identical proposal and a nonbinding common floor, exact
> control-span marginalization reduces both variance and achieved-RMSE
> production work.

It also shows that the raw-only comparator remains slower in wall time even
after removing hidden DCS arithmetic.

This is not yet a qualification or confirmation claim:

- only 8 clusters were used;
- development thresholds were tuned before qualification, not before the
  entire project;
- the V2 probe artifact predates the new sufficient-statistic independent
  auditor field; and
- simultaneous accuracy inference has not yet been run on formal V7 seeds.

## 7. Qualification decision

Qualification is authorized with a minimum of 24 clusters.  Development power
calculations require fewer than 24 for all three primary practical effects, but
24 is retained for:

- cell-level accuracy inference;
- task/rarity secondary diagnostics;
- robustness against optimistic development variance; and
- a nontrivial independent-audit matrix.

Qualification thresholds are frozen before its outcomes:

- paired-probe variance lower ratio at least 2.0;
- production variance lower ratio at least 2.0;
- final-work lower ratio at least 1.5;
- no more than 10% floor occupancy per method;
- residual-mean and covariance z maxima at most 4.5; and
- a predeclared 72-claim Bonferroni nominal family-wise accuracy analysis.

No qualification threshold, cluster count, or cell may be changed after the
new qualification seeds are drawn.
