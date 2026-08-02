# G11 V9 Proposal Bank V2 Decision

Date: 2026-08-02

Decision: **reference-gated terminal proposal bank passed independent audit;
development benchmark may proceed.**

## Bound facts

- 12 fixed terminal cells;
- 3 independent CEM training replicates per cell;
- 36 unique training seeds;
- 585,728 charged training paths;
- 143 charged optimizer updates;
- 459,210,752 algorithmic work units;
- 14.21 measured wall seconds and 225.84 cumulative process CPU seconds;
- zero failed restarts;
- defensive five-component rank-one mixture weights
  `(0.10, 0.15, 0.35, 0.25, 0.15)`; and
- bank SHA-256
  `ab57ae649c869ddb69bfef4f26ae751c224c13501e377e12addf8f960d7f61e5`.

The auditor reconstructed every schedule scale, seed, entry cost, total cost, bank
hash, reference-gate binding, and claim lock.  This authorizes only the frozen V9
development benchmark.  Qualification, performance, top-journal, and submission
claims remain locked.
