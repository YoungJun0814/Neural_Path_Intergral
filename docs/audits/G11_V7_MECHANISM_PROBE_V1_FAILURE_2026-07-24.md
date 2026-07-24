# G11 V7 mechanism probe V1 failure

Date: 2026-07-24  
Execution source: `d998da7aa23a6589ce1f912cb531154ea0bf3fa2`  
Protocol: `g11-v7-mechanism-probe-development-v1`  
Decision: failed as a complete development protocol; retired

## 1. Completed matrix

V1 completed:

- 18 fixed-grid cells;
- 8 clusters;
- 4,096 common-path raw/DCS pairs per cell-cluster;
- 144 records;
- 288 unique seed values;
- 144 mechanism-probe work records;
- no nonfinite contribution or variance; and
- a clean source tree.

Artifact SHA-256:

`ced49808fbd91f02480ce8e0d181e9b9f93811782ebe2719223246bef7215315`

Config SHA-256:

`4aed39d67bd01f932aaaffb3d8ab332380c7ae822c648e652b6a626a3ff61665`

## 2. Gate result

| Gate | Result |
|---|---:|
| complete matrix | pass |
| finite positive variances | pass |
| numerical variance decomposition | pass |
| aggregate variance-ratio lower threshold | pass |
| residual-mean diagnostic | pass |
| maximum absolute correlation diagnostic | **fail** |

The paired raw/DCS variance ratio had geometric mean 3.1645 and one-sided 95%
lower ratio 2.9699.  The maximum absolute residual-mean z-score was 2.3699.
These are development observations only because the complete protocol failed.

The failing statistic was the maximum absolute sample correlation between DCS
and raw-minus-DCS residual, 0.8503 against the V1 threshold 0.12.

## 3. Failure analysis

The largest correlations occur mainly in \(10^{-4}\) cells.  In these cells the
sample variances can be dominated by a small number of large importance
contributions.  Dividing the covariance by two small and noisy standard
deviations makes the correlation coefficient highly unstable.

Across all 48 records in each nominal rarity group, the mean correlations were:

| nominal probability | mean correlation | maximum absolute correlation |
|---:|---:|---:|
| \(10^{-2}\) | 0.0045 | 0.4576 |
| \(10^{-3}\) | -0.0452 | 0.5028 |
| \(10^{-4}\) | -0.0744 | 0.8503 |

V1's raw maximum-correlation gate had neither a sampling-standard-error
normalization nor a multiple-record interpretation.  It therefore confounded
orthogonality with instability of the correlation denominator.  The gate is
not suitable for the declared rare-event family.

This does not convert the failed gate into a pass.  It also does not prove the
population covariance is zero.  The correct actions are to preserve V1 as
failed and design a new diagnostic protocol.

## 4. V2 correction

V2 adds the sample standard error of the covariance using the centered product
influence values

\[
(D_i-\bar D)\{(Y_i-D_i)-\overline{Y-D}\}.
\]

It tests the absolute covariance z-score rather than a raw correlation.  The
4.5 development threshold is still a falsification diagnostic, not an exact
family-wise theorem.  A future formal protocol requires a separately
predeclared simultaneous procedure.

V2 uses:

- schema `npi.g11.v7-mechanism-probe.config.v2`;
- protocol `g11-v7-mechanism-probe-development-v2`;
- a new seed namespace;
- the unchanged 18-cell, 8-cluster, 4,096-pair matrix;
- the unchanged variance-ratio and residual-mean thresholds; and
- no reuse or deletion of V1 records.

V2 must be committed before execution and rerun in full.  V1 cannot be pooled
with V2 for a headline effect.
