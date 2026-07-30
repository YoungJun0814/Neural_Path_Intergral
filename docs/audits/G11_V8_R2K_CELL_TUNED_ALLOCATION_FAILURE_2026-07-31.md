# G11 V8 R2K — Cell-tuned reference allocation failure

Date: 2026-07-31

## Decision

The fresh `v8-r2-reference-cell-tuned-pilot-v1` pilot does **not**
authorize final reference execution.  The pilot namespace is burned and the
unopened final namespace remains closed.

## Reconstructed evidence

- Execution config:
  `configs/g11_v8/p5_sharded_reference_execution_v4.yaml`
- Proposal parent SHA-256:
  `40aa6c5e4662955fcf0079678c50198a57a39242c3cf7a543c6227cf918489a6`
- Source commit:
  `acc92de0b193778575197e86085fb610df60d2f8`
- Complete pilot roster: 384 of 384 unique shards
- Allocation manifest canonical SHA-256:
  `138c3e5d54173d21bd451e961792a67f2dc59a8b5ea197ce9fbd610e6d70cad2`
- Feasible requirements: 40 of 48
- Infeasible requirements: 8 of 48
- Total requested final samples: 231,515,723
- Largest requested-to-cap ratio: 5.5913439989

The package and failure receipt are independently audited in
`results/g11_v8_p5_reference_allocation_failure_audit_v4_2026-07-31.json`.
The audit reconstructs the allocation from the immutable shard contents,
checks the exact cell-method matrix, bindings, schemas, unique roster, and
fail-closed decisions.

## Interpretation

The failure is not evidence of estimator bias or an invalid likelihood
identity.  Every proposal remains a positive finite mixture with a natural
component, and all production estimates use the exact balance-mixture
likelihood.  The failure is statistical and computational: development
validation with 8 replicates of 8,192 paths did not expose several rare,
high-impact likelihood-weighted contributions that appeared in the fresh
production-size pilot.

The dominant failure modes are:

1. raw estimators whose single-profile amplitude mixtures do not cover enough
   distinct rare-event path shapes;
2. DCS estimators whose rank-one-compatible proposal remains vulnerable to a
   small number of high-variance cells; and
3. using a development gate that was materially smaller than the formal
   pilot when selecting proposals.

## Required redesign

The next protocol separates two mathematical cases.

- `raw_crosscheck` may use an arbitrary deterministic finite mixture.  Its
  exact estimator only requires the balance likelihood `p/q`; it does not
  require a rank-one price-control span.  Multiple independently fitted CEM
  time profiles may therefore be mixed to cover distinct path geometries.
- `dcs_reference` continues to require the rank-one price-control span used by
  the implemented conditional-smoothing identity.  It will use independently
  trained cell-specific profiles and rank-one amplitude mixtures only.

The redesign must be validated on held-out seeds at a production-relevant
sample scale before another fresh pilot is authorized.  Training seeds,
development-validation seeds, and the next formal-pilot namespace must remain
pairwise disjoint.

## Authorization state

- New proposal development: authorized
- Reuse of this pilot namespace: forbidden
- Final reference execution: not authorized
- Reference-complete or performance claims: not authorized
- Submission: not authorized
