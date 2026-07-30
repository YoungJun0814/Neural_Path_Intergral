# G11 V8 R1 reference-infrastructure decision

Date: 2026-07-31  
Decision class: software/statistical mechanism gate  
Empirical performance evidence: none

Frozen contract SHA-256:
`c8878f823ef7214fff8c864edd6cf48d85d754cea1a5d367b481f56542b39f38`  
Executable audit receipt SHA-256:
`cb6404b1a4491cb132804f672c285b742425cb903e11e337a0687af5dbf47b2f`

## Decision

R1 is implemented as an immutable, sharded, restart-safe reference lifecycle.
Its executable audit passes every declared mechanism and corruption check. This
means that the infrastructure can be used to begin R2
development benchmarking. It does **not** mean that the 24-cell reference matrix is
complete.

## Correctness controls implemented

- Pilot and final namespaces are different.
- Final counts are frozen solely from a complete, independent pilot set.
- Counts above the resource cap fail before final execution; there is no silent
  truncation.
- Every shard binds its protocol, cell, method, stage, count, configuration,
  threshold, parent, source, environment, and seed-key hashes.
- Completed artifacts are installed atomically and cannot be overwritten.
- Aggregation requires the exact frozen shard set and uses Chan/Welford sufficient
  statistics.
- Pilot and final digest forgery, seed reuse, dirty-source records, nonfinite
  statistics, invalid paths, and manifest mutation fail closed.
- Restart equivalence is tested by interrupting a multi-shard synthetic run and
  comparing its canonical aggregate with an uninterrupted run.
- DCS/raw agreement, likelihood normalization, standard-error, and resource gates
  propagate to the terminal result.

## Theoretical review

The final estimator remains unbiased because its integer sample count is measurable
with respect to independent pilot data and the final stream is not used for stopping
or selection. Chunking changes storage and execution order only. It does not change
the ordinary mean or sample-variance definition.

The independent-method \(z\)-score is valid only because R1 mandates disjoint seed
families. A future common-random-number cross-check must instead include the
covariance of the paired difference.

## Remaining limitations

R1 does not yet produce a qualified rBergomi reference. The next phase must:

1. create a new V2 threshold binding for a fresh protocol and namespace;
2. connect the sharded lifecycle to the actual DCS and raw simulation paths;
3. benchmark representative cells and reject an infeasible local launch;
4. execute fresh pilot and final namespaces on adequate hardware; and
5. independently audit all 48 method-cell aggregates.

GPU execution remains unauthorized until backend-specific numerical and random
stream validation is completed.
