# G11 V5 Matched 128-Step G5 Qualification Decision

Date: 2026-07-22

Decision: **the matched 128-step baseline and reference protocols passed.**

These v2 artifacts replace the 32-step G5 artifacts for the 128-step primary G6
estimand. The earlier artifacts remain valid only within their own 32-step scope.

## 1. Common execution identity

- source commit: `54b9effe91ed908b45c169f1af0cc3a3e0d93edf`
- clean detached worktree: true
- smoke: false
- grid: 128 steps

### Baseline artifact

- protocol: `g11-v5-baseline-qualification-128-v2`
- records: 3 H regimes x 2 tasks x 20 clusters = 120
- result raw SHA-256: `35d1ad8394de443e90097fabcc5cd71ea1a3058b3866c113f3dfc5d5a7215d1b`
- result canonical-JSON SHA-256: `b065694cbd871f1aa88a09041a6dfe2ce77ecc2fb9bb1c3397a28f420855c115`
- formal decision: pass

All 120 CEM fits converged. The cluster matrix is complete, training seeds are
unique, train/evaluation roles are disjoint, all estimators are finite, and the
defensive likelihood bound has zero violation.

### Reference artifact

- protocol: `g11-v5-reference-qualification-128-v2`
- cells/methods: 8/16
- result raw SHA-256: `a2c7546074febdbe5d50fb2dd079d09be1028767fd30f0a4aee5a735988ef635`
- result canonical-JSON SHA-256: `7b4e0de6f75c27f9d0be60e24bb6b44a79b717d46f9dcc0588563a96541893b4`
- resource-censored methods: 0
- largest DCS-vs-raw combined z-score: 1.924
- formal decision: pass

## 2. Matched primary references

| cell | DCS reference | standard error | samples |
|---|---:|---:|---:|
| H=0.05 terminal | 0.255201 | 0.002376 | 16,384 |
| H=0.05 barrier | 0.280357 | 0.002518 | 18,447 |
| H=0.12 terminal | 0.250180 | 0.002319 | 16,384 |
| H=0.12 barrier | 0.278966 | 0.002527 | 18,483 |
| H=0.30 terminal | 0.256149 | 0.002261 | 16,384 |
| H=0.30 barrier | 0.281175 | 0.002486 | 17,821 |

The 128-step barrier references are materially larger than their 32-step
counterparts because the event is monitored at more times. This confirms the V1
root-cause diagnosis and prohibits cross-grid reference reuse.

## 3. Training-inclusive baseline diagnostics

Median `(variance x total work)` ratios relative to crude MC are:

| cell | pure CEM | defensive CEM |
|---|---:|---:|
| H=0.05 terminal | 0.720 | 0.774 |
| H=0.05 barrier | 0.886 | 0.902 |
| H=0.12 terminal | 0.622 | 0.698 |
| H=0.12 barrier | 0.837 | 0.851 |
| H=0.30 terminal | 0.552 | 0.637 |
| H=0.30 barrier | 0.744 | 0.803 |

Pure CEM's largest observed excess kurtosis is about 97.01, compared with 3.43 for
the defensive mixture. This diagnostic supports retaining both the strong pure-CEM
comparator and the bounded defensive comparator; it is not a tail theorem.

## 4. Claim boundary

G5 is now closed for the exact 128-step primary matrix. This does not close G2--G3,
does not prove Hybrid DCS-MGI superiority, and does not rescue failed G6 V1. A new
G6 v2 config must bind these exact references and canonical artifact hashes while
using a new qualification seed namespace.
