# G11 V5 Freeze-Readiness Audit

Date: 2026-07-22

Decision: **not ready to freeze**

This is a fail-closed decision. The V5 implementation path is operational and its
development smoke tests pass, but theory gates G2--G3 and formal qualification gates
G4--G6 are not complete. No current result may be called untouched confirmatory
evidence.

## 1. Completed implementation controls

- Exact rBergomi DCS evaluations expose the affine intercept and slope used by the
  estimator.
- Terminal and barrier threshold diagnostics reconstruct the estimator and separate
  common-grid error from fine-only monitoring enrichment.
- Initially hit barriers retain an extended-real `+inf` threshold and are excluded
  only from finite-threshold rate fits.
- Finite-look Hoeffding variance intervals divide alpha across profiles, two
  moments, and all declared looks.
- The selector rejects undeclared optional looks and does not turn a zero-observed
  correction into zero uncertainty.
- Selected candidates and integer allocations are frozen before final seeds exist.
- Resource infeasibility is serialized as censoring before final sampling.
- Hybrid checkpoints bind preparation, counts, moments, work, and seeds; resumed
  Gaussian-oracle executions are bitwise identical to uninterrupted executions.
- Crude MC zero-hit pilots receive exact binomial uncertainty.
- Every CEM baseline cluster retrains and evaluates on disjoint seeds.
- Independent result-audit code reconstructs selection, allocation, estimates,
  variances, work, and gates without importing the production summarizer.

## 2. Development evidence obtained

| Artifact class | Development result | Formal status |
|---|---|---|
| threshold diagnostics | 3 H regimes, terminal/barrier exactness pass | smoke only |
| selector oracle | coverage 1.0, invalid elimination 0/240, median regret 1.0, p90 about 1.2012 | smoke only |
| fresh CEM baselines | 12/12 smoke fits converged; all safety gates pass | smoke only |
| references | independent methods and eta-zero oracles agree | target SE intentionally misses under smoke cap |
| end-to-end hybrid | two cells complete, design targets pass | one cluster and relaxed smoke RMSE |
| independent audit | end-to-end smoke artifact passes | not a formal-result audit |
| power forecast | 20 clusters exceeds conservative required count 13 at hardest RMSE | based on V4 profile data, not V5 achieved-RMSE data |

Generated smoke JSON files are development products. They were produced from a
dirty worktree and must not be inputs to a frozen confirmation.

## 3. Blocking items

1. Prove the terminal inverse-slope/coefficient assumptions or explicitly demote the
   model-level terminal rate to a conditional theorem.
2. Prove or demote the barrier early-active and mesh-enrichment moment rate.
3. Run the full 500-repetition selector qualification from a clean commit.
4. Run full fresh-training baseline qualification.
5. Generate every reference with the declared SE contract; smoke references are
   resource-censored and fail this gate by design.
6. Run V5 achieved-RMSE qualification and update power/resource forecasts from V5,
   not V4 profiles.
7. Perform the final literature cutoff search and external novelty challenge.
8. Freeze source/config/input hashes and register a never-used final seed namespace.
9. Execute 20 clusters per primary cell from a clean detached worktree.
10. Reproduce the frozen artifact on Linux and run the independent result audit.

## 4. Claim ceiling before these blockers clear

Allowed now:

> We implemented and oracle-tested an exact finite-grid defensive control-span
> marginalization estimator with uncertainty-aware hybrid level selection.

Not allowed now:

- a proved rough-Bergomi terminal or barrier complexity rate;
- a continuous-monitoring barrier claim;
- superiority over CEM or MLMC;
- “top-journal ready” or “submission complete”; or
- performance claims based on the smoke artifacts.

## 5. Next executable gate

Run the full selector qualification first because it is laptop-feasible and does not
depend on the unresolved rough-volatility theorem. In parallel, run the non-final
threshold diagnostic configuration at full paths/replicates to decide whether the
G2--G3 assumptions are empirically falsified. Reference generation and 20-cluster
achieved-RMSE qualification should wait until those two checks finish.
