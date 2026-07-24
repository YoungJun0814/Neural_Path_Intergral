# G11 V6 confirmation V1 abort audit

Date: 2026-07-24  
Decision: V1 retired; no V1 record may enter a scientific analysis

## Incident

The first untouched confirmation attempt used source commit `3a95c63` and
separate baseline/policy processes.  The policy stopped fail-closed after 98
durable records.  The baseline was manually stopped after 278 durable records
once the shared protocol was known to require a source correction.

The failing policy seed produced a valid but rare screening realization in
which the exact binomial interval excluded the frozen nominal-probability
point.  `_crude_work_interval` used:

- the exact screening interval for its lower and upper uncertainty bounds; and
- the nominal probability for its V4 point-work design.

It then passed those three values to a strict object requiring
`lower <= point <= upper`.  For example, at 99% confidence, two hits in 256
trials give a probability lower endpoint about `4.05e-4`, which exceeds the
`1e-4` nominal point.  This is possible under any finite screening design and
was not a floating-point roundoff issue.

## Correction

The corrected interval is the conservative envelope of:

1. the exact screening-compatible variance range; and
2. the predeclared nominal point variance.

The point design itself is unchanged.  The correction only expands a bound
when the two legitimate pilot inputs disagree, so the strict interval
invariant is always satisfied without clipping an estimator, deleting a
cluster, or consulting a final outcome.

## Contamination control

No V1 estimate, variance, work ratio, or aggregate scientific result was
inspected.  Only process status, durable record counts, the last completed
record identity, and the exception traceback were used for diagnosis.

Nevertheless, all V1 records are retired.  The freeze tool now versions every
protocol ID.  Confirmation V2 therefore receives a new seed namespace and
must restart both baseline and policy from zero under a clean corrective
commit.  Mixing numerically unaffected V1 records with V2 records is
prohibited.
