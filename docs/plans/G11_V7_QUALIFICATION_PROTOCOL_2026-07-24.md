# G11 V7 qualification protocol

Date: 2026-07-24  
Status: configs fixed before qualification outcomes  
Role: mechanism qualification, not untouched confirmation

## Frozen inputs

- 18 qualification-manifest cells;
- fixed 128-step rBergomi estimand;
- verified V6 task-conditioned proposal source;
- weights `(0.15, 0.35, 0.50)`;
- fixed raw defensive and fixed DCS-SLIS;
- 24 independent clusters;
- 10% requested relative sampling RMSE;
- common final floor 512;
- 8 planning replicates of 512 paths;
- independent mechanism, planning, and final seed namespaces.

## Frozen primary thresholds

| Endpoint | Requirement |
|---|---:|
| common-path raw/DCS variance lower ratio | at least 2.0 |
| production raw/DCS variance lower ratio | at least 2.0 |
| production raw/DCS final-work lower ratio | at least 1.5 |
| floor occupancy per method | at most 10% |
| maximum absolute residual-mean z | at most 4.5 |
| maximum absolute covariance z | at most 4.5 |

All ratio bounds are one-sided 95% cluster-level t bounds with equal cell
weights inside each cluster.

## Accuracy family

The formal family contains 72 claims:

- 18 cells;
- two methods; and
- attainment plus RMSE gates.

Family-wise alpha is 0.05 with Bonferroni per-claim alpha
\(0.05/72\).  Attainment uses exact Clopper--Pearson inversion.  RMSE uses a
50,000-repetition fixed-seed percentile bootstrap.  This gives about 34.7
expected bootstrap draws in each adjusted upper tail.

The attainment coverage is finite-sample exact.  The bootstrap RMSE coverage
is a predeclared nominal engineering analysis, not an exact finite-sample
theorem.

## Required artifacts

1. outcome-locked freeze receipt;
2. paired mechanism-probe result;
3. independent mechanism-probe JSON audit;
4. fixed-estimator result;
5. independent fixed-estimator JSON audit;
6. resource supplement;
7. joint mechanism/work analysis;
8. simultaneous accuracy analysis; and
9. a written pass/fail decision preserving every deviation.

## Stop rules

- Any duplicate seed, hash drift, dirty formal source, missing cell-cluster
  record, censoring, nonfinite moment, or failed independent audit stops the
  qualification.
- A mechanism failure prohibits promoting V7 to confirmation.
- Accuracy pass with efficiency fail is a negative practical result.
- Efficiency pass with accuracy fail is not publishable superiority.
- Qualification success authorizes a separately frozen confirmation; it is
  not itself untouched confirmation.
