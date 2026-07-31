# G11 V8 R2 external final-reference execution runbook

## Status and scope

The 384-shard V6 formal pilot is complete. Its frozen allocation requests
169,832,664 paths across 48 method/cell entries. One raw-crosscheck entry
exceeded the original operational cap by 2.75%. The cap amendment changes no
pilot statistic, proposal, target standard error, or requested sample count. It
only raises the raw operational cap to the smallest power of two above the
already-frozen request.

The amended allocation is statistically complete but **not authorized on the
current laptop**. A 4,096-path final-chunk benchmark gives a conservative local
forecast of 31,626.94 seconds, above the frozen 14,400-second laptop limit.

## Immutable identities

- Execution config: `configs/g11_v8/p5_sharded_reference_execution_v6.yaml`
- Formal pilot source commit: `2c2190388402b5c6323ea7bcc3d6ca9cf7593661`
- Pilot namespace: `v8-r2-reference-resource-cap-pilot-v1`
- Final namespace: `v8-r2-reference-resource-cap-final-v1`
- Expected allocation SHA-256:
  `e8846afb0f7e08aa4a52cb115c73061aa79efda9e0ee9786c4e3a38045e548a5`
- Expected paths: `169832664`
- Expected final chunks: `41487`

The pilot source identity remains embedded in the allocation. Final execution
uses a separate authorization that binds the external source commit and external
runtime environment. This separation is mandatory because a Linux external
worker cannot truthfully reuse the Windows pilot environment hash.

## Pre-launch gates

1. Use CPU and float64 only.
2. Benchmark the actual external environment with 4,096 paths for both methods
   on all three representative cells.
3. Recompute the forecast with safety factor 2.0 and the total partition count.
4. Require predicted wall time at most 21,600 seconds and predicted peak memory
   below 60% of available RAM.
5. Record a clean committed checkout, exact implementation hashes, the full
   external environment, and unique benchmark seed keys.
6. Do not open the final namespace before all preceding gates pass.

## Reconstruct and verify the exact allocation

From the latest repository state, reconstruct the large manifest outside the
repository:

```powershell
python -m experiments.g11_v8_p5_reference_allocation_amendment write-manifest `
  --config configs/g11_v8/p5_sharded_reference_execution_v6.yaml `
  --package results/g11_v8_p5_reference_pilot_package_v6_2026-07-31.json `
  --failure-receipt results/g11_v8_p5_reference_allocation_failure_v6_2026-07-31.json `
  --amendment-receipt results/g11_v8_p5_reference_allocation_amendment_receipt_v1_2026-07-31.json `
  --output C:/npi-evidence/g11_v8_p5_reference_allocation_amended_v1.json
```

Verify that the printed digest equals the immutable allocation digest above.
The generated file must remain outside the Git worktree.

## Benchmark and authorize the external topology

Every worker must use the same clean commit and identical container image. For
two 32-vCPU pods, use four total partitions (two 16-thread workers per pod) and
a genuinely shared output volume. Set available memory to the aggregate memory
assigned to those workers.

```powershell
git status --porcelain
python -m experiments.g11_v8_p5_reference_external benchmark `
  --config configs/g11_v8/p5_sharded_reference_execution_v6.yaml `
  --package results/g11_v8_p5_reference_pilot_package_v6_2026-07-31.json `
  --failure-receipt results/g11_v8_p5_reference_allocation_failure_v6_2026-07-31.json `
  --amendment-receipt results/g11_v8_p5_reference_allocation_amendment_receipt_v1_2026-07-31.json `
  --partition-count 4 `
  --topology-logical-cpus 64 `
  --topology-node-count 2 `
  --parallel-efficiency 0.70 `
  --available-memory-bytes 240000000000 `
  --maximum-wall-seconds 21600 `
  --output C:/npi-evidence/g11_v8_p5_external_benchmark_v1.json

python -m experiments.g11_v8_p5_reference_external authorize `
  --config configs/g11_v8/p5_sharded_reference_execution_v6.yaml `
  --benchmark C:/npi-evidence/g11_v8_p5_external_benchmark_v1.json `
  --output C:/npi-evidence/g11_v8_p5_external_authorization_v1.json
```

Authorization fails closed unless the measured forecast passes. It binds the
allocation hash, final source commit, all numerical implementation hashes,
external environment hash, partition rule, and partition count.

## Run disjoint workers

Launch exactly one process for each partition index `0, 1, 2, 3`. Example for
partition 0:

```powershell
python -m experiments.g11_v8_p5_reference_external worker `
  --config configs/g11_v8/p5_sharded_reference_execution_v6.yaml `
  --allocation-manifest C:/npi-evidence/g11_v8_p5_reference_allocation_amended_v1.json `
  --authorization C:/npi-evidence/g11_v8_p5_external_authorization_v1.json `
  --output-directory C:/npi-evidence/g11_v8_p5_reference_resource_cap_final_v1 `
  --partition-index 0
```

The global sorted-shard modulo rule gives complete, disjoint, balanced
partitions. Re-running the same partition resumes only its immutable completed
shards; different partitions never target the same shard.

## Aggregate and independently audit

```powershell
python -m experiments.g11_v8_p5_reference_external aggregate `
  --config configs/g11_v8/p5_sharded_reference_execution_v6.yaml `
  --allocation-manifest C:/npi-evidence/g11_v8_p5_reference_allocation_amended_v1.json `
  --authorization C:/npi-evidence/g11_v8_p5_external_authorization_v1.json `
  --final-directory C:/npi-evidence/g11_v8_p5_reference_resource_cap_final_v1 `
  --output C:/npi-evidence/g11_v8_p5_reference_aggregate_v1.json

python -m experiments.g11_v8_p5_reference_external audit `
  --config configs/g11_v8/p5_sharded_reference_execution_v6.yaml `
  --allocation-manifest C:/npi-evidence/g11_v8_p5_reference_allocation_amended_v1.json `
  --authorization C:/npi-evidence/g11_v8_p5_external_authorization_v1.json `
  --final-directory C:/npi-evidence/g11_v8_p5_reference_resource_cap_final_v1 `
  --aggregate C:/npi-evidence/g11_v8_p5_reference_aggregate_v1.json `
  --output C:/npi-evidence/g11_v8_p5_reference_result_audit_v1.json
```

Acceptance requires all 48 precision targets, all likelihood-normalization
checks, and all 24 independent-method agreement checks to pass. A failure must
remain a reported failure; raw crosscheck may not replace the DCS primary
reference, and self-normalization remains prohibited.
