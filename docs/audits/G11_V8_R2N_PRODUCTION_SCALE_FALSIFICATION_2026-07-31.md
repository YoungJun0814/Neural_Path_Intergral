# G11 V8 R2N — Production-scale proposal falsification

Date: 2026-07-31

## Result

The V2 method-specific proposal protocol failed: 0 of 8 required
cell-method proposals passed every preregistered gate.  No proposal is
promoted and all downstream authorizations remain false.

The execution was technically complete:

- 21 independent CEM fits;
- 24 proposal candidates;
- 405 unique derived seeds;
- strict finite JSON output;
- clean source commit `67f85c4997574bda60be57f3ffb9f1218e6b1bef`.

## Positive findings

Several candidates met the conservative projected-allocation margin before
the concentration and raw block-coverage gates:

| Cell-method requirement | Best projected-to-cap ratio |
|---|---:|
| H=0.05, terminal, 1e-4, raw | 0.464 |
| H=0.12, barrier, 1e-3, DCS | 0.422 |
| H=0.12, terminal, 1e-5, DCS | 0.277 |
| H=0.20, barrier, 1e-4, raw | 0.491 |
| H=0.20, terminal, 1e-5, raw | 0.422 |

Likelihood normalization z-scores were within the frozen bound and observed
likelihoods respected the natural-component upper bound.

## Block-order defect

The audit found that the within-replicate blocks were not exchangeable mixture
blocks.  `simulate_rbergomi_mixture` draws randomized component counts but
simulates and concatenates samples in expert order.  Consecutive stored blocks
therefore contain different proposal components.

For example, one raw candidate had about 5,800 nonzero contributions per full
replicate, while the first stored block repeatedly had only 6–18 and later
blocks had more than 1,500.  Across every raw candidate, the ratio between the
largest and smallest mean block counts exceeded 10.  The independent audit
therefore confirms a storage-order confound.

The full-replicate estimates and variances remain valid: row order does not
change their mean or variance.  What is invalid is interpreting consecutive
component-grouped slices as i.i.d. mixture block diagnostics.

## Corrective action

The next protocol must use a separately seeded uniform permutation of path
indices before forming diagnostic blocks.  This permutation:

- changes no estimator value or full-replicate variance;
- is independent of path values and labels;
- converts the grouped storage order into exchangeable diagnostic partitions;
- has its own seed-ledger record; and
- is frozen before the new validation namespace is opened.

V2 training and validation namespaces are burned.  The next protocol must bind
the V2 result and passing audit by file hash and use new namespaces.

## Authorization

- Methodology correction development: authorized
- V2 candidate promotion: forbidden
- Proposal manifest construction: forbidden
- New formal pilot: forbidden
- Final execution, performance claims, or submission: forbidden
