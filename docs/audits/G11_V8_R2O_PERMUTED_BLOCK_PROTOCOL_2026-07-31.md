# G11 V8 R2O — Exchangeable diagnostic-block protocol

Date: 2026-07-31

## Correction

V3 corrects the V2 storage-order confound.  For every candidate and replicate,
it derives a third seed independent of proposal innovations and mixture-label
sampling.  After all pathwise contributions are computed, it applies the
corresponding uniform random permutation to contribution indices and only then
splits them into eight equal diagnostic blocks.

The permutation is used for diagnostics only.  It does not change:

- the full-replicate estimator;
- the exact balance likelihood;
- the full-replicate sample variance;
- event labels or simulated paths;
- DCS marginalization; or
- the target estimand.

Conditional on the simulated component counts and paths, a value-independent
uniform permutation removes the expert-concatenation storage order.  The
resulting blocks are exchangeable random partitions of the complete mixture
sample.  They are not claimed to be additional independent replicates; their
maximum variance is only a conservative within-replicate stress diagnostic.

## Seed contract

Each candidate replicate now has three distinct seed roles:

1. proposal Brownian innovations;
2. mixture labels; and
3. diagnostic-block permutation.

Protocol, role, candidate, cell, replicate, and V3 namespace are included in
every key.  All derived seeds are globally checked for uniqueness.

## Candidate extension

The V2 outcome is disclosed and used only for development.  V3 retains every
previous family and adds:

- one refined full-rank raw family with five amplitudes for each of the three
  independently fitted temporal profiles; and
- two additional rank-one DCS amplitude families only for the unresolved
  H=0.05 terminal 1e-5 cell.

All added raw candidates retain the 8% natural component and exact likelihood
upper bound 12.5.  All added DCS candidates remain scalar amplitude mixtures
of one temporal profile and must pass the exact rank-one structural check.

## Unchanged falsification gates

No V2 gate was relaxed after observing its outcome.  V3 keeps the same:

- 8 replicates × 32,768 paths;
- 8 blocks per replicate;
- allocation safety factor 6;
- projected-to-cap threshold 0.50;
- likelihood-normalization absolute z threshold 4;
- variance and concentration thresholds;
- raw nonzero-coverage thresholds; and
- downstream fail-closed decisions.

## Authorization

- Execute V3 development validation: authorized
- Reuse V1 or V2 seeds: forbidden
- Promote a candidate before independent V3 audit: forbidden
- Open a formal pilot or final run: forbidden
- Make performance or submission claims: forbidden
