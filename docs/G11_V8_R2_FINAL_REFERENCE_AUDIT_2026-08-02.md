# G11 V8 R2 final independent-reference audit

Date: 2026-08-02  
Protocol: `g11-v8-p5-sharded-reference-resource-cap-v1`

## Decision

The frozen finest-grid independent-reference gate passed.

- `reference_complete=true`
- `downstream_frozen_method_evaluation_authorized=true`
- `performance_claim_authorized=false`
- `submission_authorized=false`

The last two fields remain false deliberately. This phase validates the
reference values used to score methods; it does not itself establish that the
proposed method is faster or better than a baseline.

## Execution provenance

- source commit: `3623e4d8f035ad906ea5e97ba21a8922615b5932`
- environment SHA-256:
  `df5f8f38185e79deb715e247acd4b0efdc095bc08c5ea4c3d87212c99db30fcc`
- allocation SHA-256:
  `e8846afb0f7e08aa4a52cb115c73061aa79efda9e0ee9786c4e3a38045e548a5`
- authorization canonical SHA-256:
  `91dc43fb35203dd58eb6ec15b064a9346904e923b685c6c41643a59caaae10d8`
- partition rule:
  `sorted_shard_id_global_index_modulo_partition_count`

Partition 0 executed 20,744 of 20,744 shards. Partition 1 executed 20,743
of 20,743 shards. Both had `skipped=0`, emitted complete receipts, and wrote
zero stderr bytes.

## Statistical scope

The execution used 169,832,664 paths across 41,487 immutable shards:

- DCS primary reference: 124,896,513 paths
- independent raw cross-check: 44,936,151 paths
- 24 frozen model/task/probability cells
- 48 method-cell estimates
- 24 independent method-agreement tests

No self-normalization was used. Each estimate is the ordinary mean of its
importance-weighted contribution, and likelihood normalization was audited
separately.

## Gate results

All gates passed:

- precision: 48/48
- likelihood normalization: 48/48
- DCS/raw agreement: 24/24
- complete method-cell matrix: yes
- exact aggregate recomputation from raw shards: yes
- final source and environment authorization: yes

The most conservative margins were:

- largest `standard_error / target_standard_error`: `0.535652`
- largest relative standard error: `0.022523`
- largest absolute likelihood-normalization z-score: `2.834086`
- largest absolute DCS/raw agreement z-score: `1.508534`
- predeclared agreement limit: `4.0`

The worst agreement cell was
`h0.20-discrete_lower_barrier-p1e-05`; its two independent estimates differed
by `1.87096e-7`, or `1.5085` combined standard errors. This is evidence of
agreement, not a reason to average the two streams: DCS remains the declared
primary reference and raw remains a cross-check.

## Underflow falsification and correction

The first local final run failed deterministically when a finite canonical
log-price exponentiated to float64 zero. That execution was stopped, preserved,
and invalidated. The correction did not floor prices or drop paths: it made the
already simulated log-price the canonical state and evaluated all positive
threshold events and DCS affine reconstruction in log space.

The exact failed identity was added as a regression test, the full repository
suite passed (`818 passed`), and the second final execution started from an
empty directory under a new source-bound authorization. Details are in
`docs/G11_V8_R2_LOG_SPOT_REMEDIATION_AUDIT_2026-08-02.md`.

## Evidence files

- `results/g11_v8_p5_reference_external_benchmark_v3_2026-08-02.json`
- `results/g11_v8_p5_reference_final_authorization_v2_2026-08-02.json`
- `results/g11_v8_p5_reference_final_aggregate_v1_2026-08-02.json`
- `results/g11_v8_p5_reference_final_aggregate_audit_v1_2026-08-02.json`
- `results/g11_v8_p5_reference_final_execution_receipt_v1_2026-08-02.json`

The aggregate contains the SHA-256 roster of all 41,487 shard payloads. The raw
shards remain local and reproducible from the source, frozen manifest, and seed
contract; they are intentionally not committed as tens of thousands of Git
objects.

## What this unlocks

The reference bottleneck is closed. The next valid paper step is a fresh,
training-inclusive evaluation of the frozen proposed method and baselines
against these references, with disjoint final seeds, confidence intervals,
failure accounting, and total-work reporting. Only that downstream audit may
authorize a performance claim.
