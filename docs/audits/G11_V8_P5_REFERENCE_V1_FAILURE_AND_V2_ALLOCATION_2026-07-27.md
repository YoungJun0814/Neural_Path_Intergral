# G11 V8 P5 Reference V1 Failure and V2 Allocation Decision

Date: 2026-07-27

Decision: **V1 REFERENCE PRECISION GATE FAILED; V2 RE-EXECUTION AUTHORIZED; PERFORMANCE OPEN**

The clean-source V1 reference execution generated the complete 24-cell matrix at
commit `760b3dc40c1cc5b2542a041dad035c4271444546`. It used a 4-replicate,
4,096-sample pilot, the median pilot variance, and a 1.10 allocation safety factor.
Its source worktree was clean and there was no resource censoring.

The run passed likelihood-normalization checks and the independent DCS/raw agreement
check in every cell. It failed the fixed reference-precision gate: 37 of its 48
method--cell estimates exceed the required absolute standard error
`0.10 * 0.20 * nominal_probability`. The largest excess was 3.7326 times the target
for raw importance sampling in the H=0.05 discrete-barrier, 1e-5 cell. The compact,
hash-pinned failure receipt is
[`g11_v8_p5_independent_reference_v1_failure_receipt_2026-07-27.json`](../../results/g11_v8_p5_independent_reference_v1_failure_receipt_2026-07-27.json).

This is a statistical-design failure, not evidence for or against DCS performance.
The raw crosscheck has the largest variance in several rare-event cells, as expected;
the agreement test says only that the two independently seeded estimators are
consistent at their achieved uncertainty.

## Why V2 is methodologically different

It would be invalid to inspect a final stream, add samples until its estimated standard
error happens to pass, and then treat the resulting mean as an ordinary fixed-size
estimate. Such outcome-dependent stopping can bias a sample mean. V2 therefore keeps
pilot samples and final samples separate and fixes every final count **before** its
final stream is drawn.

V2 changes only the allocation rule and its seed namespace:

| Item | V1 | V2 |
|---|---:|---:|
| Pilot replicates | 4 | 8 |
| Pilot samples/replicate | 4,096 | 32,768 |
| Pilot variance used | median | maximum |
| Allocation safety factor | 1.10 | 6.0 |
| Final cap | 2,097,152 | 8,388,608 |
| Reference namespace | `p5-reference` | `p5-reference-v2` |

The maximum pilot variance prevents a low median from concealing one high-variance
rare-event pilot. The safety factor and cap are versioned before V2 output is
inspected. Because V1's observed precision failure motivated this change, V2 declares
`outcome_data_used: true` and is development-only. The V1 artifact remains a negative
result; it is not replaced or relabelled.

V2 must still pass the same complete-matrix, two-method precision, no-censoring,
normalization, independent-agreement, and seed-separation gates. Its only permitted
use is to test the revised allocation. A later clean, fresh-seed, outcome-blind freeze
must repeat the validated rule before any P5 performance comparison can be authorized.

## V2 laptop interruption

The first V2 execution started from clean commit
`ea2ffa2546ecda4646c9b27c2dbebbe66274edfd`. It was stopped after at least 24,812.5
seconds of wall time (about 6.9 hours) and 76,767.7 observed CPU seconds (about 21.3
CPU hours). The program writes its output atomically only at terminal completion, so
there is no result file, no partial estimate, and no gate outcome to interpret.

The generated V2 stream is nevertheless treated as **burned**. It must not be restarted
with the same `p5-reference-v2` namespace: a future execution needs a new protocol ID
and namespace, durable checkpointing, and separate per-cell/shard result audits before
any fragments can be aggregated. The machine-readable interruption receipt is
[`g11_v8_p5_independent_reference_v2_interruption_receipt_2026-07-27.json`](../../results/g11_v8_p5_independent_reference_v2_interruption_receipt_2026-07-27.json).

This is a hardware/resource limitation of the current laptop execution, not a
numerical pass, numerical failure, or model-performance result. The next execution
must be planned for an external multi-core CPU or a separately GPU-validated runner;
the present runner is explicitly CPU-only and must not be represented as GPU-validated.
