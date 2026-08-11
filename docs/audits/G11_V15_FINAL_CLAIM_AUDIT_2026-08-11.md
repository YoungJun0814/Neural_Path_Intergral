# G11 V15 final claim audit

Date: 2026-08-11

Decision: **NO-GO for top-journal submission; GO for continued theorem-first
development and working-paper circulation.**

## 1. Completed and checked

- Current primary-source novelty ledger and explicit closest-work boundary.
- Continuous/finite-grid probability and scaling contract.
- Stable conditional digital, put, and call oracles.
- Orthonormal Cameron–Martin Galerkin basis and exact finite-noise action.
- Independent L-BFGS and trust-region Newton mode solvers.
- Exact defensive multimode finite-rank Gaussian sampling and density.
- Dense-covariance, moment, normalization, and likelihood-bound oracles.
- T15-1 through T15-4 proof chains and theorem tests.
- Streaming adjacent-grid mesh diagnostic and pilot-frozen MLMC allocation.
- Structure-preserving neural initializer, deterministic correction, and fallback.
- Five-comparator frozen protocol with training-inclusive work.
- One-cell development and disjoint-seed three-cell qualification.
- Small-noise diagnostic through probability about `1.34e-6`.
- Hash-bound results and offline fail-closed audit.

## 2. Numerical result audit

Qualification integrity and numerical gates pass.  Across its three cells:

- candidate/reference accuracy z: `0.994`, `1.260`, `2.220`;
- likelihood-normalization z: at most `1.002`;
- defensive-bound violation: `0` in all cells;
- strongest-primary/V15 work ratio at 100 queries: `1.430`, `4.689`, `3.964`.

Small-noise v2 has disjoint primary/reference streams from one frozen exact proposal.
At epsilon `0.125`, relative variance is `0.502` and the observed exponent ratio is
`0.985`.  This is strong mechanism evidence, not an asymptotic proof.

Mesh v1 failed its bias budget because the correction was sampling-noise dominated.
Streaming v2 increased the sample count and finest grid without deleting v1; v2
passes the numerical bias budget.  Its weak fitted rate and low correction SNR leave
T15-6 and T15-7 open.

## 3. Claim matrix

| Claim | Status | Authorized wording |
|---|---|---|
| finite-grid conditional formula | proved/tested | exact for the declared BLP grid |
| ordinary-IS unbiasedness | proved | exact for a frozen proposal |
| defensive likelihood bound | proved/tested | `dP/dQ <= 1/delta` |
| Rao–Blackwell ordering | proved | conditional estimator has no larger population variance under assumptions |
| qualification efficiency | observed | 1.43–4.69× on the frozen three-cell 100-query matrix |
| small-noise efficiency | open | empirical exponent trend only |
| continuous-time mesh rate | open | empirical bias diagnostic only |
| neural amortization advantage | open | structure and correction validated; cost advantage not established |
| top-journal readiness | false | working-paper core only |

## 4. Prohibited statements

- “The method is logarithmically efficient” or “asymptotically optimal.”
- “The method has bounded relative error” as a theorem.
- “The implementation exactly simulates continuous-time rBergomi.”
- “The neural operator is the source of the current qualification gain.”
- “The method dominates CEM/RQMC generally.”
- “The paper is ready for Mathematical Finance or Finance and Stochastics.”

## 5. Blocking conditions

1. T15-5 is open, hence G5 and the top-journal gate are false.
2. T15-6/T15-7 are open; the mesh experiment is not rate-identifying.
3. Qualification covers only three moderate-to-small probability cells.
4. The deepest small-noise reference is not an unrelated proposal family.
5. The neural operator lacks a real teacher-bank/OOD cost study.
6. No independent person/hardware reproduction or external proof review exists.

These are research blockers, not software defects, and cannot be closed by adding
more unit tests or rewriting the conclusion.
