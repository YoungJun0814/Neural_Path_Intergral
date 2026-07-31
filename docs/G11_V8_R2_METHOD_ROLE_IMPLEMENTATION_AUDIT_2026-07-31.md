# G11 V8 R2 method-role reference implementation audit

## Executive decision

The reference-design and external-execution implementation is technically
complete through the external hardware gate. The final reference itself is not
complete, and no performance or submission claim is authorized.

The current laptop must not launch the final workload. The measured,
safety-factored forecast is 28,474.67 seconds versus the frozen 14,400-second
limit. External execution requires a fresh benchmark and authorization in the
actual external environment.

## Frozen statistical target

- Estimand: fixed finest time grid; no continuous-time claim.
- Cells: 24 fixed task/H/probability cells.
- Methods: DCS primary reference and independent raw crosscheck.
- DCS relative standard-error target: 2%.
- Raw relative standard-error target: 5%.
- Cross-method agreement: combined absolute z-score at most 4.
- Likelihood-normalization absolute z-score: at most 4.
- Estimator: ordinary arithmetic mean only.
- Self-normalization: prohibited.
- Allocation: maximum variance over 8 independent pilot replicates, multiplied
  by safety factor 6.

These contracts are method-specific but not cell-outcome adaptive after the V6
pilot was opened.

## Formal evidence completed

### V5

- Formal pilot: 384/384 immutable shards.
- Resource gate: 47/48 entries passed.
- Failure: `h0.12-discrete_lower_barrier-p1e-05 / dcs_reference`.
- Required/cap ratio: 1.7788.
- Failure package and independent reconstruction audit: passed.

### V6

The V6 design changed no proposal, target, estimand, safety factor, or pilot
size. The DCS cap was derived by a frozen rule: twice the prior failed
requirement, rounded upward to the next power of two.

- Formal pilot: 384/384 new-namespace shards.
- Resource gate: 47/48 entries passed.
- New failure:
  `h0.05-discrete_lower_barrier-p1e-05 / raw_crosscheck`.
- Required samples: 17,238,220.
- Original raw cap: 16,777,216.
- Excess: 2.75%.
- Failure package and independent reconstruction audit: passed.

## Exact-count cap amendment

Repeating pilots until one happened to fit a cap would condition the design on
a favorable variance realization. That path was rejected.

Instead, the V6 requested counts were kept exactly unchanged. The raw
operational cap was raised to 33,554,432, the smallest power of two not below
the already-frozen raw request. The amendment changes:

- no pilot statistic;
- no requested sample count;
- no proposal or mixture weight;
- no target standard error;
- no final namespace outcome.

The independently reconstructed amended allocation has:

- allocation SHA-256:
  `e8846afb0f7e08aa4a52cb115c73061aa79efda9e0ee9786c4e3a38045e548a5`;
- 169,832,664 total paths;
- 41,487 immutable final chunks;
- 48/48 statistically resource-feasible entries.

The amendment audit passes, but hardware execution remains closed.

## Theoretical review

### Unbiasedness and likelihood correctness

Both reference methods use ordinary sample means of exact mixture-likelihood
weighted contributions. Every defensive mixture contains a positive natural
component, providing support coverage and a finite pointwise upper bound on the
mixture likelihood ratio. The raw crosscheck is not forced into the DCS
rank-one price-control span; imposing that restriction would destroy part of
its intended methodological independence.

No self-normalized importance-sampling estimator is used. Consequently, the
implementation does not introduce ratio-estimator bias into the reference.

### Adaptedness and fixed-grid scope

The proposal controls are deterministic time-piecewise controls. They are
adapted. All claims and thresholds are tied to the fixed finest discretization.
The code and documents do not infer a continuous-time rare-event probability
from these results.

### Pilot and final independence

Pilot and final namespaces are distinct, and seed derivation includes protocol,
namespace, stage, method, cell, and shard index. Allocation verifies that pilot
seed keys are unique. Final aggregation rejects any final key that duplicates a
pilot or another final key.

### Pilot/final environment separation

The original runner incorrectly required the external final environment hash to
equal the Windows pilot environment hash. This has been corrected by a separate
final authorization that binds:

- the exact allocation hash;
- the clean final-execution source commit;
- hashes of every numerical implementation file;
- the full final runtime environment and its hash;
- the measured external benchmark;
- the worker topology and partition count.

Final shards use the authorized final source/environment. The aggregate records
both pilot and final source/environment identities and audits them separately.

### Parallel execution correctness

The external runner globally sorts final shard IDs and assigns global index
modulo partition count. Tests verify complete coverage, pairwise disjointness,
and a maximum partition-size difference of one. The authorization also requires
`partition_count × torch_threads_per_worker <= topology_logical_cpus`, preventing
an overcommitted topology from being presented as a valid speed forecast.

## Technical verification

- Full test suite: 816 passed.
- Focused reference regression suite: 59 passed.
- External execution/allocation focused suite: 29 passed.
- Focused Ruff checks: passed.
- Focused mypy checks: passed.
- `git diff --check`: passed.

A repository-wide Ruff invocation reports 49 pre-existing issues in legacy
top-level visualization/notebook utility scripts. None is in the frozen
reference runtime or new external execution path. They do not invalidate this
reference evidence, but should be cleaned before a repository-wide Ruff gate is
made mandatory.

## Hardware decision

The current laptop benchmark passes allocation, roster, seed, CPU-topology, and
memory gates but fails wall time:

- predicted wall time: 28,474.67 seconds (7.91 hours);
- allowed wall time: 14,400 seconds;
- final execution authorization: false.

The preferred external topology is either one 64-logical-CPU environment with
four 16-thread workers or two identical 32-logical-CPU pods with two workers
each and a genuinely shared output volume. The actual topology must still pass
the 4,096-path benchmark; this document does not pre-authorize it.

## Remaining terminal sequence

1. Provision/connect the external CPU topology and shared storage.
2. Reconstruct the allocation outside the Git worktree and verify its digest.
3. Run the external benchmark in the actual container.
4. Generate final authorization only if every benchmark gate passes.
5. Run all disjoint worker partitions and resume until 41,487/41,487 shards are
   complete.
6. Aggregate all shards.
7. Require all 48 precision targets, 48 normalization checks, and 24
   cross-method agreement checks to pass.
8. Independently recompute the aggregate and publish either the pass or the
   exact falsification result.

Until steps 1-8 complete, the R2 reference is not finished and the downstream
performance experiment, top-journal claim, and submission decision remain
closed.
