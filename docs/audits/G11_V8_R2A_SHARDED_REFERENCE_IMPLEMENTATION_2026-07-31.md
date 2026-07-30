# G11 V8 R2a sharded-reference implementation decision

Date: 2026-07-31  
Scope: production wiring and lifecycle falsification  
Performance evidence: not authorized

## Decision

The R1 immutable shard lifecycle is now connected to the actual rough-Bergomi DCS
and raw-reference calculation paths. A fresh V2 threshold binding preserves the
clean 24-cell threshold manifest while binding a new protocol, allocation schema,
CPU/float64 backend, fixed-finest-grid estimand, and disjoint benchmark, pilot, and
final namespaces.

R2a passes only the implementation gate. The clean-source representative benchmark,
formal pilot, allocation freeze, final computation, and 48 method-cell reference
matrix remain separate subsequent gates.

## Implemented execution path

1. `g11_v8_p5_reference_benchmark.py` runs the three predeclared representative
   cells and produces conservative local/external forecasts.
2. `g11_v8_p5_reference_pilot.py` executes or resumes the exact 384-shard pilot
   roster.
3. `g11_v8_p5_reference_freeze_allocation.py` requires the complete pilot set and
   freezes all 48 final counts without inspecting a final outcome.
4. `g11_v8_p5_reference_shard.py` accepts only chunk identities and counts present
   in that allocation manifest.
5. `g11_v8_p5_reference_aggregate.py` requires the exact final shard set.
6. `g11_v8_p5_reference_result_audit.py` independently reloads every shard and
   recomputes the aggregate.

## Error review

- Actual RNG derivation includes the namespace as well as protocol, method, stage,
  cell, replicate, and stream. Benchmark paths therefore cannot collide with
  formal pilot paths.
- Pilot shards bind the V2 threshold binding hash. Final shards bind the allocation
  manifest hash.
- A formal run refuses a dirty worktree. Shard outputs are intended to live under
  ignored `tmp/` storage until the complete aggregate is frozen, so restart files
  do not falsely alter source provenance.
- The final runner cannot accept an arbitrary sample count.
- Source commit, runtime environment, dtype, device, estimand, threshold, and config
  must remain identical from pilots through final aggregation.
- The development protocol remains V1-failure-informed. Even a passing development
  reference cannot be relabelled as untouched qualification evidence.

## Verification

The synthetic lifecycle test constructs the full 24-cell, two-method, eight-pilot
roster and verifies:

- pilot resume without overwrite;
- a 48-entry allocation freeze;
- final-chunk resume;
- a complete 48 method-cell / 24 agreement aggregate; and
- exact independent recomputation by the result auditor.

The test uses analytic sufficient statistics, not simulated performance outcomes.
