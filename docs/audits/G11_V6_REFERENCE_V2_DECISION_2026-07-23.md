# G11 V6 Reference V2 Decision

Date: 2026-07-23

Protocol: `g11-v6-reference-qualification-v2`

## Decision

The task-conditioned V2 reference **passes the complete 18-cell qualification
contract**. It is accepted as the independent finite-grid reference artifact for V6
qualification.

This decision does not claim that the V6 routed policy outperforms its baselines.
It establishes only that every frozen cell now has a sufficiently precise,
independently cross-checked probability reference.

## Frozen identities

- source commit: `cc6995852c09585b2e3107f55760096fe7b9ab20`;
- qualification manifest:
  `4e6c8715dfeca6cfc6d7be9252d47a8ed0894393302bbc74393c69857fe92d32`;
- proposal-training artifact:
  `b74770898cc1c4896adb6a65062595d6decadbc1948ad11cc00095f61c2ebc01`;
- reference artifact:
  `786173bd1316d882f8abe049abbd2f12fc05c681864c8367b61c5cd612f73db7`.

The accepted artifact is stored outside the repository at:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_formal\commit_cc69958\reference_v2_full\g11_v6_reference_qualification_v2_full.json`

## Aggregate audit

| Check | Result |
|---|---:|
| Frozen cells | 18 |
| Independent methods | 36 |
| Canonical seeds | 978 |
| Complete reference matrix | pass |
| DCS SE contracts | pass |
| Raw cross-check SE contracts | pass |
| Resource censoring | 0 |
| Independent-method agreement | pass in all cells |
| Likelihood normalization | pass in all cells |
| Maximum requested final samples | 458,389 |
| Maximum achieved/target SE ratio | 0.824 |
| Maximum absolute agreement z | 1.767 |
| Maximum absolute normalization z | 2.536 |
| DCS estimates inside calibration bands | 18/18 |

## Why V2 is admissible after V1 failed

V1 failed before a full reference artifact or any policy qualification result
existed. The replacement did not change:

- any event threshold or finite-grid estimand;
- the target standard error;
- the pilot/final split;
- the 20-million sample cap;
- the independent raw cross-check;
- the combined-z agreement gate; or
- the likelihood-normalization gate.

V2 changed only the defensive sampling proposal. It reused the task-conditioned
zero/half/full bank learned on prespecified development cells before qualification,
verified the raw training artifact and ledgers, retained natural mixture mass 0.15,
and used a new protocol/seed namespace.

On the V1 stopping cell `h005-terminal-p1e-04`, requested DCS final samples fell
from 40,440,463 to 188,550 without relaxing precision. This is an approximately
214.5-fold reduction in planned work and explains why the full V2 matrix became
feasible.

## Remaining restriction

This artifact supports comparisons against a common independent reference at the
fixed 128-step estimand. It is not evidence for a continuous-time probability and
does not by itself establish training-inclusive superiority of the routed model.
