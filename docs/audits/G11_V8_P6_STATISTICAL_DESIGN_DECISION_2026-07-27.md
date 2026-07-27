# G11 V8 P6 Statistical-Design Decision

Date: 2026-07-27

Decision: **PASS FOR P7 FALSIFICATION; NO EMPIRICAL POWER OR PERFORMANCE CLAIM**

P6 freezes the inference unit, primary endpoint family, multiplicity control,
accuracy co-gates, seed separation, failure handling, and planning-only cluster
counts before V8 development outcomes are generated.

## Primary inference

The inferential observation is one independent seed cluster, never an individual
Monte Carlo path.  Within a cluster, log-ratios from all 24 P5 cells are averaged
with equal cell weight.  The primary efficiency family has five one-sided,
Bonferroni-adjusted lower-ratio claims: raw/DCS probe variance, raw/DCS production
variance, raw/DCS final sampling work, pure-CEM/DCS total work, and smoothing-RQMC/
DCS total work.  Its familywise alpha is 0.025, hence each lower interval uses
alpha 0.005.

The separate accuracy family covers four methods, 24 cells, and two method-cell
co-gates: exact attainment and a bootstrap RMSE upper bound.  It has familywise
alpha 0.025 over 192 claims.  Attainment uses a one-sided Clopper--Pearson lower
bound of at least 0.80.  The RMSE interval is explicitly described as a *nominal*
simultaneous bootstrap upper bound; it is not called exact.

## Cluster and power policy

P8 is fixed at 32 new clusters and a future P10 confirmation at 48 new clusters.
The embedded normal calculations are only pre-outcome planning scenarios: they use
declared alternatives and cluster log-standard-deviation assumptions, with the
one-sided ratio boundary subtracted before calculating the required count.  They do
not estimate an effect and cannot be used as empirical evidence.  P8 may authorize
or block a later P9 freeze, but may not retune P10's count or endpoint family.

## Non-negotiable failure rules

- Resource-censored or incomplete records remain in the matrix and fail the primary
  claim; deletion is prohibited.
- Failed retries are charged to total work.
- P7, P8, P9, P10, bootstrap, and P11 have six distinct namespaces, all disjoint
  from P5 calibration, reference, training, pilot, and final namespaces.
- The design binds the P0 claim contract and P5 matrix hashes.  A later edit fails
  the audit rather than silently changing the experiment.

The implemented checker is
`experiments/g11_v8_p6_statistical_design_audit.py`; it verifies 46 independent
conditions and emits a non-overwriting JSON receipt.  Passing this phase authorizes
only P7 falsification-first development.
