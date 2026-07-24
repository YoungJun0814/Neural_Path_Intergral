# G11 V7 confirmation protocol

Date: 2026-07-24  
Status: implementation ready; freeze receipt required before outcomes  
Role: untouched mechanism and achieved-RMSE confirmation

## 1. Confirmation question

The confirmation repeats the V7 qualification comparison under new protocol IDs
and therefore new deterministic seed namespaces:

> Under an unchanged finite-grid estimand and unchanged proposal, does DCS retain a
> prespecified practical variance and final-work advantage over raw defensive
> importance sampling while both methods satisfy simultaneous accuracy requirements?

No routing claim is tested.  The two methods are fixed DCS-SLIS and fixed raw
defensive importance sampling.

## 2. Locked design

The following are unchanged from qualification:

- 18 cells and the same cell definitions;
- 128-step finest finite grid;
- rBergomi simulator and FFT engine;
- task-conditioned zero/half/full deterministic proposal bank;
- mixture weights `(0.15, 0.35, 0.50)`;
- 4,096 samples per paired mechanism cell-cluster;
- 10% requested relative sampling RMSE;
- eight planning replicates of 512 paths;
- common final floor 512;
- exact balance likelihood and ordinary sample means;
- operation-work accounting; and
- all mechanism and accuracy thresholds.

Only the following changes are allowed:

1. 24 qualification clusters become 64 confirmation clusters;
2. proposal-training amortization changes mechanically from 432 to 1,152
   cell-cluster queries;
3. protocol IDs change, creating disjoint simulation seed namespaces; and
4. the accuracy bootstrap seed changes.

`g11_v7_freeze_confirmation.py` compares the parsed qualification and confirmation
configs and rejects any other change.

## 3. Frozen primary gates

| Endpoint | Confirmation requirement |
|---|---:|
| Common-path raw/DCS variance lower ratio | at least 2.0 |
| Production raw/DCS variance lower ratio | at least 2.0 |
| Production raw/DCS final-work lower ratio | at least 1.5 |
| Final-floor occupancy per method | at most 10% |
| Maximum absolute residual-mean z | at most 4.5 |
| Maximum absolute covariance-product z | at most 4.5 |

Ratios use one-sided 95% cluster-level t bounds.  Each cluster first averages
log-ratios equally over all 18 cells.  The 64 cluster summaries are the independent
units of inference.

## 4. Simultaneous accuracy family

The family remains 72 claims:

- 18 cells;
- two methods; and
- attainment plus RMSE.

Family-wise alpha is 0.05.  Attainment uses exact one-sided Clopper--Pearson
inversion at Bonferroni-adjusted confidence.  RMSE uses a predeclared fixed-seed
50,000-repetition percentile bootstrap at the same adjusted level.

The method-cell co-gate requires:

- exact attainment lower bound at least 0.60; and
- RMSE upper bound no larger than 1.25 times the requested nominal sampling error,
  combined in quadrature with reference uncertainty.

Clopper--Pearson coverage is exact.  The RMSE bootstrap is nominal and must not be
described as a finite-sample theorem.

## 5. Execution order

1. Commit the protocol, configs, freeze code, audit code, and tests.
2. Confirm a clean source tree and passing CI.
3. Generate the freeze receipt from the audited V7 qualification.
4. Run the paired mechanism probe.
5. Run its independent JSON-only audit.
6. Run fixed DCS and raw achieved-RMSE estimators with durable checkpoints.
7. Run the independent fixed-estimator audit and resource supplement.
8. Generate joint mechanism/work and simultaneous-accuracy analyses.
9. Run the independent aggregate confirmation audit.
10. Write the decision without changing any gate.

No formal artifact may be regenerated after its outcome is inspected except to
correct a documented implementation defect.  Such a defect requires a new version,
new seed namespace, and explicit failure record for the superseded run.

## 6. Stop conditions

The confirmation fails closed on any:

- dirty execution source;
- config, manifest, reference, or proposal-source hash drift;
- duplicate or overlapping mechanism/planning/final seed;
- missing or duplicated cell-cluster-method record;
- nonfinite moment;
- resource censoring;
- independent audit failure;
- mechanism gate failure;
- aggregate accuracy failure; or
- unexplained deviation from the freeze receipt.

Accuracy pass with efficiency failure is a negative practical result.  Efficiency
pass with accuracy failure is not evidence of a usable estimator.

## 7. Interpretation boundary

A pass confirms the DCS mechanism for the frozen 18-cell, finite-grid, reused-proposal
regime.  It does not prove:

- uniform superiority over all rough-volatility parameters;
- superiority over an optimally task-tuned new proposal family;
- continuous-monitoring unbiasedness;
- a universal MLMC rate; or
- a neural-architecture contribution.

A cross-platform V7 reproduction remains a separate requirement after confirmation.
