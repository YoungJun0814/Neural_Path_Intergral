# G11 V9 Reference V1 Resource Decision

Date: 2026-08-02

Decision: **V1 reference namespace closed for insufficient precision; candidate
benchmark remains unopened; a fresh-seed reference-only resource escalation is
statistically admissible.**

## Integrity result

The independent audit passed every reconstruction check.  The 12 cell means,
standard errors, randomization seeds, work counts, artifact hashes, and decision
locks were reproduced exactly.  The result was generated from a clean source tree.
There is no evidence of an implementation or likelihood error.

## Precision result

The frozen maximum relative standard error was 0.15.  Eight cells passed and four
failed:

| Cell | Estimate | Relative SE | Gate |
|---|---:|---:|---|
| H=0.05, p-design=1e-5 | 3.056856e-6 | 0.403 | fail |
| H=0.12, p-design=1e-5 | 1.8334495e-5 | 0.570 | fail |
| H=0.20, p-design=1e-4 | 1.2591962e-4 | 0.163 | fail |
| H=0.20, p-design=1e-5 | 1.6521342e-6 | 0.437 | fail |

The other eight cells satisfy the boundary.  Reference V1 therefore does not
authorize the proposal bank or any performance benchmark.

## Admissible amendment

No candidate, raw mechanism, or external-comparator performance outcome has been
generated.  V1 contains reference precision information only.  Consequently a
new reference-only namespace may use V1 solely as an allocation pilot without
inflating a subsequent performance test, provided that:

1. V1 samples are not pooled into the final V2 reference;
2. V2 uses disjoint proposal and RQMC seeds;
3. the 0.15 precision gate, estimands, thresholds, and cells are unchanged;
4. allocation is a deterministic precommitted function of V1 relative SE;
5. the V1 failure and V2 amendment remain visible; and
6. proposal training and performance evaluation remain blocked until V2 passes.

The allocation rule is

\[
R_{2,j}=2^{\lceil\log_2 \max(64,
\lceil 2R_1(\widehat{se}_{rel,j}/0.15)^2\rceil)\rceil},
\]

where \(R_1=32\) and 2 is a variance safety factor.  This changes computation only;
it does not weaken the scientific gate.
