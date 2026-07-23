# G11 V6 Reference V1 Resource-Failure Audit

Date: 2026-07-23

Protocol: `g11-v6-reference-qualification-v1`
Source commit: `d64194559ee27a37af67c03200515b16be6367dd`

## Decision

The first full 18-cell reference attempt was stopped by its predeclared resource
gate. It is **not** a qualified reference artifact and it is **not** evidence that
the V6 estimator is inaccurate.

The decisive checkpoint was:

- cell: `h005-terminal-p1e-04`;
- method: `dcs_reference`;
- independently planned final sample request: `40,440,463`;
- predeclared maximum final samples: `20,000,000`; and
- resource-censoring ratio: approximately `2.022`.

Once planning fixed a request above the cap, the aggregate gate
`no_reference_resource_censoring` could no longer pass. The process was therefore
terminated after 32,768 capped final samples rather than spending the remaining
19,967,232 samples on an already disqualified protocol.

## Preserved evidence

The attempt was not deleted or relabeled. Its durable method checkpoints remain
outside the repository under:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_formal\commit_d641945\reference\checkpoints`

Before the failing cell, the following fixed allocations completed:

| Cell | Method | Requested and completed final samples |
|---|---|---:|
| `h005-terminal-p1e-02` | DCS reference | 1,053,850 |
| `h005-terminal-p1e-02` | raw cross-check | 264,810 |
| `h005-terminal-p1e-03` | DCS reference | 8,500,918 |
| `h005-terminal-p1e-03` | raw cross-check | 1,958,148 |

Those partial outcomes may be used to audit the failed protocol, but they are not
eligible as final cell references and must not be mixed into the replacement run.

## Cause classification

The failure is attributed to **reference proposal inefficiency**. V1 used one fixed,
task-agnostic proposal schedule. Its requested work increased sharply as rarity
increased. The finite-grid estimand, exact likelihood orientation, DCS identity,
planning/final seed separation, target standard error, and maximum sample cap were
not found to be incorrect.

## Corrective action

The replacement reference protocol must:

1. keep the same target-standard-error and sampling contracts;
2. keep a positive natural mixture component and its deterministic likelihood bound;
3. use the already frozen, reference-free, task-conditioned zero/half/full proposal
   bank derived before qualification outcomes existed;
4. verify that bank against the raw training artifact and both strict ledgers;
5. use a new protocol and seed namespace;
6. run a `p=1e-4` resource pilot before another full 18-cell attempt; and
7. preserve this V1 failure and never resume it as if it belonged to the new
   protocol.

The correction changes only the sampling proposal used to measure the same
probabilities. It does not relax an accuracy gate or alter an event threshold.
