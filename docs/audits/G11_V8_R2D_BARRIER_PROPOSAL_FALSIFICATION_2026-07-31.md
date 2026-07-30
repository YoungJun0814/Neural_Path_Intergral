# G11 V8 R2d barrier-proposal falsification

Date: 2026-07-31  
Gate: barrier-aware reference-proposal development  
Decision: fail; no existing candidate may be promoted

## Outcome

Four frozen, exact-likelihood defensive rank-one mixtures were evaluated on the six
cells responsible for the reference-allocation failure. Each candidate used four
independent replicates of 8,192 paths per cell. The likelihood-normalization and
raw-event-coverage checks passed, but no single candidate satisfied the frozen
8,388,608-path allocation cap for every method-cell pair.

This is a valid falsification rather than a numerical crash:

- all 192 derived seeds are unique and reconstruct exactly;
- all candidate schedules and weights match the frozen configuration;
- projected sample counts reconstruct from the maximum replicate variance, the
  factor-six safety margin, and the frozen standard-error target;
- the source tree was clean at execution; and
- the result keeps every final and performance-claim gate closed.

## Why cell-specific reuse is still insufficient

Choosing the best existing candidate separately for each cell and estimation method
does not solve the bottleneck:

| Cell | Method | Best existing candidate | Best cap ratio |
|---|---|---|---:|
| \(H=.20\), barrier, \(10^{-5}\) | raw | mid-loaded rank-one | 1.185 |
| \(H=.05\), barrier, \(10^{-5}\) | DCS | mid-loaded rank-one | 1.527 |

All other method-cell minima fit the cap. These two strict failures rule out both a
single global candidate and post-hoc method-specific reuse of this candidate roster.

## Theoretical interpretation

Ordinary importance weights remain exact: this experiment does not identify an
estimator bias. It identifies inadequate overlap and heavy-tailed contributions.
The rank-one DCS construction additionally requires all experts inside one proposal
to be scalar multiples of a common, strictly one-signed price-driver schedule.
Mixing arbitrary time profiles inside one DCS proposal would violate its analytic
marginalization assumptions and is therefore prohibited.

The next admissible design is a separately trained time profile for each difficult
cell, followed by a defensive amplitude mixture containing the natural component
and positive scalar multiples of that one profile. Training and candidate selection
remain development-only; a selected proposal must be frozen before a fresh pilot,
and eventual qualification must use untouched seeds and namespaces.

The result is not evidence for a performance claim, journal-level superiority, or a
completed reference table.
