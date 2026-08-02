# G11 V9 Terminal Development Decision

Date: 2026-08-02

Decision: **development result is technically valid but the frozen continuation
gate failed; qualification is not authorized.**

## Integrity

The V3 matrix completed all 12 terminal cells, 4 independent clusters, 48 paired
raw/DCS records, and 144 external-comparator records.  The independent audit
reconstructed all 672 fresh seeds, bindings, bank costs, target-work values,
cluster-level confidence bounds, aggregates, and claim locks without failure.

Exactness and internal consistency passed:

- maximum exactness error: `4.5475e-13` (`<=1e-10`);
- likelihood-normalization pass fraction: `1.0` (`>=0.95`);
- DCS maximum combined reference z: `2.5452` (`<=4`);
- paired raw-minus-DCS maximum z: `2.6916` (`<=4`); and
- every external method used an exact likelihood and ordinary mean.

## Positive mechanism result

At the frozen repeated-query count K=100, DCS required less total work than the raw
estimator using the same mixture in every Hurst group:

| H | Geometric raw/DCS work ratio | One-sided 95% lower bound |
|---:|---:|---:|
| 0.05 | 2.0319 | 1.5726 |
| 0.12 | 1.5020 | 1.1015 |
| 0.20 | 1.5773 | 1.4212 |

This supports the narrow statement that exact conditional integration materially
improves the frozen mixture estimator after its additional CDF cost is charged.

## Why qualification is blocked

The best-primary comparator gate failed in every Hurst group:

| H | Geometric best-primary/DCS work ratio | Lower bound | Required |
|---:|---:|---:|---:|
| 0.05 | 0.4014 | 0.2516 | geo >=0.80, lower >=0.67 |
| 0.12 | 0.4226 | 0.2679 | geo >=0.80, lower >=0.67 |
| 0.20 | 0.2978 | 0.2682 | geo >=0.80, lower >=0.67 |

A ratio below one favors the comparator.  Defensive CEM was the best-primary
method in 37/48 paired positions; target-law conditional MC was best in 11/48.
At K=100 DCS did beat conditional MC geometrically for each H (ratios 2.87, 3.28,
1.51) and smoothing RQMC (17.38, 39.54, 44.43), but it did not consistently beat
defensive CEM (0.523, 0.580, 1.141).

The correctness/resource gate also failed:

- maximum external combined reference z: `9.0984 > 4`;
- conditional MC maximum z: 9.0984;
- smoothing RQMC maximum z: 4.9251;
- defensive CEM maximum z: 4.2278; and
- 22 resource-censored records: 16 smoothing RQMC and 6 conditional MC.

The most extreme best-primary ratio at H=0.20, p-design=1e-5 is influenced by a
resource-censored and inaccurate conditional baseline.  It is not promoted as a
performance fact.  The frozen global correctness gate already prevents it from
authorizing a group.

## Binding stop action

No Hurst group passed the complete rule.  Therefore:

- no qualification config or result may be created inside V9;
- no regime-conditional empirical claim is authorized;
- no broad performance, top-journal, or submission claim is authorized; and
- thresholds, comparator roster, accuracy boundary, or selected cells may not be
  changed and then presented as V9 qualification.

The scientifically useful V9 result is a validated mechanism improvement paired
with a falsification of practical competitiveness against the strongest baseline.
A future V10 may conditionally smooth a full-dimensional CEM proposal, but it must
be a new model and new protocol rather than a V9 repair.
