# G11 V8 R2c reference-allocation failure decision

Date: 2026-07-31  
Gate: pilot-frozen reference allocation  
Decision: fail; final execution prohibited

Pilot package file SHA-256:
`cb6230b3e449304ebd4e6b3eab24a26c17d8d89ca463a70bc14567593a67ca66`  
Allocation-failure receipt file SHA-256:
`a66ba3d788c7c97f10a9e571aaee5a0a3a84eedb6d9fe6453a1555fb0d7c0d03`  
Independent audit file SHA-256:
`981e52f204e8633931fec9b9bd578d42af8610fdbfd620ada13711fe861bcba8`  
Canonical allocation SHA-256:
`63c23ddf2f9a323614789af0afc4bfc6b0823d94791ed941de5e6505316f6ce7`

## Outcome

All 384 formal pilot shards completed:

- 24 threshold-bound cells;
- independent DCS and raw methods;
- eight replicates per method-cell;
- 32,768 paths per replicate;
- 12,582,912 total paths; and
- one clean source/config/threshold/environment state.

The allocation is **not** resource feasible. Only 37 of 48 method-cell entries fit
the frozen cap of 8,388,608 paths. Eleven entries exceed it. The exact requested
total is 1,448,253,332 final paths, so no final namespace was opened.

## Largest failures

| Cell | Method | Frozen requested paths | Multiple of cap |
|---|---|---:|---:|
| \(H=.12\), barrier, \(10^{-4}\) | raw | 519,649,238 | 61.95 |
| \(H=.20\), barrier, \(10^{-5}\) | raw | 214,806,592 | 25.61 |
| \(H=.05\), barrier, \(10^{-4}\) | raw | 143,730,962 | 17.13 |
| \(H=.05\), barrier, \(10^{-5}\) | raw | 133,509,519 | 15.92 |
| \(H=.05\), barrier, \(10^{-4}\) | DCS | 132,376,616 | 15.78 |
| \(H=.05\), barrier, \(10^{-5}\) | DCS | 122,285,379 | 14.58 |

The maximum-replicate-variance rule and factor-six allocation safety margin were
frozen before final sampling. They cannot be weakened after seeing these pilots and
then reused under the same namespace.

## Interpretation

This is not a software failure. The shard roster, hashes, seed separation,
normalization data, allocation formula, and cap gate all audit correctly. It is a
model/proposal failure: the current reference proposal leaves extremely heavy
ordinary contributions in rare barrier cells. DCS reduces some cells but does not
remove this barrier-specific tail.

Increasing the cap is not an acceptable primary fix. The requested 1.45 billion
paths would be costly and would preserve the underlying heavy-tail weakness.

## Required next design

The development pilot namespace is burned. The final namespace is still unopened.
A new development protocol must improve the barrier reference proposal before using
fresh pilots. Candidate changes must preserve:

- exact ordinary likelihood weighting;
- outcome-independent fixed final allocation;
- independent cross-check streams;
- no self-normalization;
- full training and failed-attempt cost accounting; and
- a later untouched qualification namespace.

The leading next action is a task-tuned, barrier-aware defensive mixture reference
with an independently trained proposal, followed by a small falsification study on
the six failed barrier/terminal extremes before reopening a 24-cell pilot matrix.
