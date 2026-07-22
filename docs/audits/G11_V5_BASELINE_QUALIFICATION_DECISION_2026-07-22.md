# G11 V5 Baseline Qualification Decision

Date: 2026-07-22

Decision: **the frozen fresh-training baseline protocol passed.**

This decision qualifies the crude MC, pure CEM-SLIS, and defensive CEM-mixture
implementations used as V5 comparators. It does not state that the proposed hybrid
estimator beats them.

## 1. Frozen execution identity

- protocol: `g11-v5-baseline-qualification-v1`
- source commit: `3a093f5666218054cf2973d42eff928607fcfa28`
- source worktree: clean detached worktree
- config SHA-256: `1f0d7cb3b7ebd7e6786e80214221f011768d12c2ce4889590bf1f642b91f873c`
- seed-ledger SHA-256: `26aff43974cea3285902d9d9283f9ff93e621133ac8220ea3160f284a2c3d4ff`
- result SHA-256: `f6fbd539121a92eafbba1abf41f834364467e61ec6e3e41bd5a4aee547029740`
- design: 3 H regimes x 2 tasks x 20 independently trained clusters = 120 records
- smoke: false

## 2. Gate results

All serialized gates passed:

- complete 120-record cluster matrix;
- 120/120 CEM fits converged;
- every estimator and standard error is finite;
- 120 distinct training seeds and no within-cluster train/evaluation seed overlap;
- no defensive likelihood-bound violation;
- exact binomial interval handling for any zero-hit crude pilot; and
- frozen config, clean source, and non-smoke readiness.

No crude evaluation happened to have zero hits in this moderate-rarity baseline
matrix; the zero-hit fail-closed behavior remains unit-tested.

## 3. Training-inclusive efficiency diagnostics

For context only, the table reports the median clusterwise ratio
`(variance x total work)_method / (variance x work)_crude`. CEM training work is
included in the numerator.

| model/task | pure CEM median | defensive CEM median |
|---|---:|---:|
| H=0.05 terminal | 0.723 | 0.773 |
| H=0.05 barrier | 0.840 | 0.855 |
| H=0.12 terminal | 0.629 | 0.692 |
| H=0.12 barrier | 0.781 | 0.809 |
| H=0.30 terminal | 0.555 | 0.640 |
| H=0.30 barrier | 0.710 | 0.767 |

Pure CEM is somewhat cheaper in this matrix, but it has an unbounded likelihood.
Its maximum observed excess kurtosis was about 41.05, versus 4.01 for the defensive
mixture. These are diagnostics, not a tail theorem. The final comparison must retain
both baselines and must not omit CEM training work.

## 4. Claim boundary

This qualification establishes that the baselines are serious, independently
trained, auditable comparators. It does not establish:

- superiority of Hybrid DCS-MGI over CEM or crude MC;
- performance at rarities outside this baseline matrix;
- the V5 achieved-RMSE result; or
- any rough-Bergomi convergence theorem.

