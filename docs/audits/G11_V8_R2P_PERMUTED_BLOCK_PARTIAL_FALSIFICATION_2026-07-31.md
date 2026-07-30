# G11 V8 R2P — Permuted-block partial falsification

Date: 2026-07-31

## Result

V3 selected 3 of 8 required proposals.  Because the roster is incomplete,
the overall protocol correctly reports failure and no proposal is promoted to
a reference manifest.

Selected development proposals:

| Requirement | Candidate | Projected-to-cap ratio |
|---|---|---:|
| H=0.12, terminal, 1e-5, DCS | train-1 / rank-one dense | 0.456 |
| H=0.20, barrier, 1e-4, raw | full-rank broad | 0.352 |
| H=0.20, terminal, 1e-5, raw | full-rank refined | 0.431 |

Unresolved best ratios:

| Requirement | Best ratio | Main remaining gate |
|---|---:|---|
| H=0.05, terminal, 1e-4, raw | 0.512 | allocation margin |
| H=0.12, barrier, 1e-3, DCS | 0.557 | allocation margin |
| H=0.05, terminal, 1e-5, DCS | 0.687 | allocation and block concentration |
| H=0.12, barrier, 1e-5, raw | 1.234 | allocation |
| H=0.05, terminal, 1e-5, raw | 1.459 | allocation and concentration |

## Technical audit

The independent V3 audit passed all artifact-integrity checks:

- 35 unique candidates;
- 861 unique training, simulation-label, and permutation seeds;
- exact recomputation of every candidate pass flag;
- exact recomputation of the three minimum-ratio selections;
- strict finite JSON;
- clean source commit
  `3c0d245ad744531f6a10aeb85666fae0f7ad6cd5`; and
- all downstream decisions fail closed.

## Block correction verification

The V2 component-order signature disappeared.  Across raw candidates, the
largest-to-smallest mean event count across the eight stored diagnostic
positions is now close to one and below two.  For representative candidates
it is approximately 1.04–1.05, rather than greater than ten.

This supports that the independent permutation removed the storage-order
confound.  It does not prove light tails; the remaining allocation failures
are treated as genuine proposal-variance failures.

## Next mathematical action

The raw mixtures currently distribute nonnatural mass uniformly across
experts.  That choice is exact but not variance-optimal.  For fixed expert
densities \(q_k\), the raw second moment is

\[
M(w)=E_P\left[{1_Ap\over \sum_k w_kq_k}\right],
\]

which is convex in positive mixture weights \(w\).  A new development stage
may therefore learn weights under a positive floor and fixed natural mass,
using off-policy exact-mixture identities on training-only samples, followed
by fully independent V4 validation.

DCS weights cannot be optimized with the same raw identity without a separate
derivation, because the conditional marginalization depends on the proposal
mixture.  DCS will retain rank-one schedules and use a preregistered finite
weight/amplitude grid on training-only data, then independent validation.

## Authorization

- Develop fixed-expert raw mixture-weight optimization: authorized
- Develop rank-one DCS candidate grid: authorized
- Promote V3 partial selections: forbidden until a complete manifest exists
- Open a formal pilot or final run: forbidden
- Make performance or submission claims: forbidden
