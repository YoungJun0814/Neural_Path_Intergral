# G11 V5 Achieved-RMSE Qualification V2 Decision

Date: 2026-07-22

Decision: **numerical execution qualification passed; full G6 efficiency gate failed.**

The matched 128-step experiment is statistically accurate and internally auditable,
but it is not competitive with matched crude MC or CEM after selection and training
work are charged. It also does not study rare events. Consequently V5 is not ready
for untouched confirmation or a top-journal performance claim.

## 1. Frozen execution identity

- protocol: `g11-v5-achieved-rmse-qualification-128-v2`
- source commit: `53e6bb18917d1330c85afac9266acc8485b78d2a`
- clean detached worktree: true
- config SHA-256: `e6fa0f114edf5e7e76b1a062280df81d3b20bb629fa1e94c9b3262eb29e0f301`
- result SHA-256: `a0b940d589df3791b3c7d47723ff05a6646a4c5769e491a7cb65cf9b4e159b5a`
- independent result-audit SHA-256: `98ea6a24ac6ac2ea4e07f7183e9998326eb97e5f0461eb8fc930e959ab55472c`
- efficiency-audit source commit: `fd709ce4d89e2fb313b004308de24f09cbcdd932`
- efficiency-audit SHA-256: `d634819d4438b9dae46e174e553392fd411ab1569888794fa8a731fcf5316b85`

## 2. Numerical execution result

The 3 H x 2 task x 20 cluster matrix contains 120 complete runs and no resource
censoring. Every method selection and integer allocation was frozen before final
seeds, and the independent auditor reproduced selection, allocation, telescoping,
variance, work, seed-ledger, aggregate, and gate arithmetic with zero failures.

| cell | relative RMSE | empirical target attained | combined 95% coverage | bounded coverage |
|---|---:|---:|---:|---:|
| H=0.05 terminal | 0.0313 | 1.00 | 0.95 | 1.00 |
| H=0.05 barrier | 0.0295 | 1.00 | 1.00 | 1.00 |
| H=0.12 terminal | 0.0340 | 1.00 | 0.95 | 1.00 |
| H=0.12 barrier | 0.0295 | 1.00 | 0.95 | 1.00 |
| H=0.30 terminal | 0.0407 | 1.00 | 0.85 | 1.00 |
| H=0.30 barrier | 0.0390 | 1.00 | 0.95 | 1.00 |

All ten internal execution/accuracy gates pass. This validates the achieved-RMSE
engine for these finite-grid cells; it does not validate the paper's efficiency
claim.

## 3. Training-inclusive efficiency result

All 120 runs selected `start_4`, the finest-grid DCS-SLIS endpoint. No run used an
earlier MLMC start. Median selection work consumed 90.4%--91.9% of total work because
all nine profiles were sampled through the final 4,096-path look.

The ratio below is `projected baseline work / observed Hybrid work` at the same 10%
sampling-RMSE target. Values below one favor the baseline. CEM training is included.

| cell | event probability | crude ratio | pure CEM ratio | defensive CEM ratio |
|---|---:|---:|---:|---:|
| H=0.05 terminal | 0.255 | 0.019 | 0.545 | 0.545 |
| H=0.05 barrier | 0.280 | 0.017 | 0.552 | 0.553 |
| H=0.12 terminal | 0.250 | 0.019 | 0.542 | 0.543 |
| H=0.12 barrier | 0.279 | 0.017 | 0.552 | 0.552 |
| H=0.30 terminal | 0.256 | 0.019 | 0.545 | 0.545 |
| H=0.30 barrier | 0.281 | 0.017 | 0.552 | 0.552 |

Thus Hybrid is roughly 52--59 times more expensive than crude and about 1.8 times
more expensive than CEM in this moderate-probability matrix. The deterministic audit
fails all three baseline-efficiency gates.

## 4. Scientific interpretation

This matrix cannot support a rare-event claim: every reference probability is about
25%--28%, while the declared rare-event gate is at most 5%. Ordinary MC is expected
to be strong here. The result is still useful because it identifies a practical
routing rule: for moderate events, do not pay a nine-profile hybrid-selection cost.

The V2 result also shows that the present selector is statistically safe but too
conservative for this cost structure. Its simultaneous bounded intervals never
eliminate an MLMC candidate before the largest look, after which it selects DCS-SLIS
anyway.

## 5. Required V3 redesign before any final freeze

1. Calibrate fixed-128-grid terminal and barrier cells at genuinely rare
   probabilities, including at least `1e-2`, `1e-3`, and a feasible `1e-4` sentinel.
2. Keep one-factor-at-a-time H experiments and build fresh matched references for
   every new threshold.
3. Add an explicit moderate-event router to crude/DCS-SLIS so Hybrid selection is
   invoked only where projected savings can repay profiling work.
4. Redesign profiling with a predeclared work budget, cheap staged profiles, and a
   qualification gate on selection-work fraction. Do not tune the gate on V2 seeds.
5. Execute actual achieved-target crude, pure-CEM, defensive-CEM, DCS-SLIS, and
   Hybrid runs; use projection only for planning.
6. Recompute paired power and total resource forecasts from V3 data, not V4 profiles
   or the current moderate-event matrix.
7. Preserve G2--G3 as conditional until inverse-slope, coefficient, active-time, and
   mesh-enrichment bounds are proved.

Until these actions pass under a new protocol and seed namespace, the allowed claim
is implementation correctness and finite-grid accuracy, not practical or
top-journal-level efficiency superiority.
