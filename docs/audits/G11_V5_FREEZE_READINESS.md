# G11 V5 Freeze-Readiness Audit

Date: 2026-07-22

Decision: **not ready to freeze; G4 selector qualification passed**

This is a fail-closed decision. The V5 implementation path is operational and its
development smoke tests pass and formal selector gate G4 has passed. Theory gates
G2--G3 and formal qualification gates G5--G6 are not complete. No current result
may be called untouched confirmatory evidence.

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
| threshold diagnostics | clean non-smoke 3-H run; exactness passes and rates remain descriptive | full development evidence, not G2--G3 proof |
| selector oracle | frozen 4,000-record run: coverage 0.99875, invalid elimination 0, median regret 1.0, p90 about 1.2012 | formal G4 pass |
| fresh CEM baselines | frozen 128-step 120-record v2 run; 120/120 fits and all gates pass | formal matched-primary pass |
| references | frozen 128-step 8-cell v2 run; all SE, cross-check, and oracle gates pass | formal matched-primary pass |
| end-to-end hybrid | matched 128-step V2: 120/120 complete, accuracy gates pass | internal qualification pass; full efficiency fail |
| independent audit | V2 arithmetic, aggregates, work, and seed ledger pass | formal result-audit pass |
| efficiency audit | selection 90%+ of work; crude and CEM cheaper in all six cells | G6 performance fail |
| power forecast | 20 clusters exceeds conservative required count 13 at hardest RMSE | based on V4 profile data, not V5 achieved-RMSE data |

The selector qualification was produced from clean source commit `a49c51a` using a
frozen configuration and is formal G4 evidence. Generated smoke JSON files remain
development products; they were produced from a dirty worktree and must not be
inputs to a frozen confirmation.

## 3. Blocking items

1. Prove the terminal inverse-slope/coefficient assumptions or explicitly demote the
   model-level terminal rate to a conditional theorem.
2. Prove or demote the barrier early-active and mesh-enrichment moment rate.
3. Redesign V3 around genuinely rare cells and a bounded profiling-work fraction;
   V2 accuracy passed but full efficiency failed.
4. Run actual matched achieved-target baselines and update power/resource forecasts
   from V3, not V4 profiles.
5. Perform the final literature cutoff search and external novelty challenge.
6. Freeze source/config/input hashes and register a never-used final seed namespace.
7. Execute the untouched primary matrix from a clean detached worktree.
8. Reproduce the frozen artifact on Linux and run the independent result audit.

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

The full non-final threshold diagnostic did not immediately falsify the G2--G3
mechanism, but it did not prove it. Matched G5 and G6 numerical accuracy now pass,
but G6 efficiency fails because the cells are not rare and profiling dominates.
Freeze nothing further until a predeclared rare-event V3 redesign passes actual
matched-baseline and selection-work gates.
