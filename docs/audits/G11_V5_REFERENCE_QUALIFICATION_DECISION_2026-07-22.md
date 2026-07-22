# G11 V5 Reference Qualification Decision

Date: 2026-07-22

Decision: **the frozen finite-grid reference protocol passed.**

## 1. Frozen execution identity

- protocol: `g11-v5-reference-qualification-v1`
- source commit: `438e9119e5285579b487e930c7f4c2902b05ac5b`
- source worktree: clean detached worktree
- config SHA-256: `86517dacc96742dadac38295dda2024041f16ccbaaa46509a0aad95cfb7493e3`
- seed-ledger SHA-256: `8c73dd813556e2875d7a39249638548f10026e008198c2686132e0f8c4a61a16`
- result SHA-256: `a173ed5b10927c66c5717137418ea120b9a45383d00e3dcca4af6f020b85e49e`
- matrix: 4 models x 2 tasks = 8 cells and 16 method summaries
- resource-censored methods: 0
- smoke: false

## 2. Primary DCS references

| cell | estimate | standard error | samples |
|---|---:|---:|---:|
| H=0.05 terminal | 0.253331 | 0.002340 | 16,384 |
| H=0.05 barrier | 0.253593 | 0.002117 | 23,854 |
| H=0.12 terminal | 0.241363 | 0.002236 | 16,384 |
| H=0.12 barrier | 0.258461 | 0.002185 | 22,978 |
| H=0.30 terminal | 0.251629 | 0.002217 | 16,384 |
| H=0.30 barrier | 0.262278 | 0.002132 | 22,605 |
| eta=0 terminal | 0.282342 | 0.002105 | 16,384 |
| eta=0 barrier | 0.265907 | 0.002052 | 16,384 |

Every primary reference meets its predeclared standard-error contract.

## 3. Independent checks

- Every DCS reference was compared with a separately seeded raw estimator.
- The largest combined z-score was 3.182, below the predeclared limit 4.0.
- The eta-zero terminal reference agrees with the closed-form Black--Scholes digital
  probability (z approximately 1.150).
- The eta-zero discrete-barrier reference agrees with deterministic killed-density
  quadrature (z approximately 0.949).

The maximum raw-vs-DCS z-score is not negligible and must remain visible in the
artifact. Passing the prespecified cross-check does not make the reference exact;
downstream comparisons must propagate the recorded reference standard error.

## 4. Gate and claim boundary

The complete matrix, finite summaries, SE contracts, independent cross-checks,
eta-zero oracles, frozen config, clean source, and non-smoke gates all pass. G5 is
therefore closed for the declared finite grid.

This result does not provide a continuous-time reference, prove a rough-Bergomi
rate, or demonstrate model superiority. It must not be silently reused for a
different grid, task definition, or parameter regime.
