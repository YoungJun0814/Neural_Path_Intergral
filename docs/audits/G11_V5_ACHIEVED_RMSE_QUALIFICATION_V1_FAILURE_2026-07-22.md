# G11 V5 Achieved-RMSE Qualification V1 Failure Decision

Date: 2026-07-22

Decision: **V1 failed and must not be relabelled or rerun under the same protocol.**

The failure is informative: it exposed an estimand-grid mismatch before untouched
confirmation. It is not valid evidence that the hybrid method fails on a correctly
matched 128-step barrier estimand.

## 1. Frozen execution identity

- protocol: `g11-v5-achieved-rmse-qualification-v1`
- source commit: `1034a231b1b37a464f3f3c96fc6cac0b0a8972d9`
- source worktree: clean detached worktree
- config SHA-256: `12a4e135d1efc81eb23005fb191b57ad3a4c5b9a13286de7ff0b4d2950470770`
- result SHA-256: `99f822b12f8015f0938e7767d65b8804533d74eaed839208e562722f53f44d35`
- audit SHA-256: `b30b44c40317a04c99c666a6a4e2cffd8ac77285dfaaa72277c1ecebaa4492e0`
- matrix: 3 H regimes x 2 tasks x 20 clusters = 120 complete records
- resource censoring: 0
- independent result audit: pass, 0 reconstruction failures

## 2. Gate result

Eight implementation and execution gates passed, including completeness, design
targets, seed separation, unique preparation hashes, and absence of censoring. Two
accuracy gates failed:

- across-cluster relative RMSE;
- minimum combined asymptotic coverage.

| cell | relative RMSE vs supplied reference | combined 95% coverage |
|---|---:|---:|
| H=0.05 terminal | 0.0293 | 0.95 |
| H=0.12 terminal | 0.0264 | 1.00 |
| H=0.30 terminal | 0.0247 | 1.00 |
| H=0.05 barrier | 0.1321 | 0.10 |
| H=0.12 barrier | 0.0877 | 0.35 |
| H=0.30 barrier | 0.0862 | 0.45 |

Every cell met its estimator-reported empirical sampling-variance target. The
barrier coverage failure is therefore systematic relative to the supplied
reference, not a shortage of final Monte Carlo samples.

## 3. Root cause

The G6 finest grid is `8 x 2^4 = 128` monitoring steps. The qualified G5 baseline
and reference artifacts used 32 steps. Terminal probabilities are comparatively
insensitive here, but a discrete barrier hit probability changes when 96 additional
monitoring times are introduced. Binding the 32-step barrier probability to a
128-step estimator was an estimand error.

The implementation correctly refused to pass the mismatched experiment. The
independent auditor reproduced all arithmetic and gates, so the failure cannot be
attributed to the production summarizer.

## 4. Mandatory corrective action

1. Preserve V1 as failed qualification evidence.
2. Qualify fresh CEM baselines on the exact 128-step primary grid.
3. Generate 128-step DCS references with raw cross-checks and eta-zero oracles.
4. Freeze a new G6 v2 config with exact per-cell 128-step reference values and new
   canonical input hashes.
5. Use a new protocol/seed namespace; do not reuse V1 qualification seeds.
6. Independently audit V2 whether it passes or fails.
