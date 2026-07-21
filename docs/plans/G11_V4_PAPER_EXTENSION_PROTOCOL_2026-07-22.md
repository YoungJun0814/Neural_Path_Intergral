# G11 V4 Paper-Extension Qualification Protocol

Date: 2026-07-22

Frozen config: `configs/g11_v4_crossover_qualification.yaml`

Required tag: `g11-v4-crossover-qualification-v1-freeze`

## 1. Purpose and claim class

This protocol is a **post-M7 qualification study**, not a new untouched confirmation.
M7 results were inspected before this design was finalized. Its purpose is to remove
three blockers exposed by M7:

1. absence of a finest-grid DCS single-level comparator;
2. absence of a task-tuned CEM single-level IS comparator; and
3. confounding of H, eta, and rho in the earlier regime matrix.

Results may determine the final confirmatory design. They may not be combined with M7
and relabelled as one predeclared experiment.

## 2. Estimand and event scope

Every selected cell estimates the event probability on the same declared 128-step
finite grid. There is no continuous-monitoring claim.

The common one-factor matrix uses terminal and discrete-barrier events. The earlier
absolute stress-level excursion event failed non-degeneracy in several OAT calibration
regimes. Those failures are preserved. Excursion is excluded from the common OAT
matrix rather than retuned after seeing results.

The selected target probabilities are `1e-4` and `1e-6`. The rho `-0.85` barrier
`1e-6` cell is excluded because its independent calibration validation estimate failed
the frozen probability band. All other selected cells passed their calibration
probability and precision gates.

## 3. One-factor-at-a-time regime design

Base parameters are `(H, eta, rho)=(0.12, 1.1, -0.6)`. Exactly one of these parameters
changes in each OAT regime:

| Regime | H | eta | rho | Changed factor |
|---|---:|---:|---:|---|
| base | 0.12 | 1.1 | -0.6 | none |
| oat_h007 | 0.07 | 1.1 | -0.6 | H |
| oat_h030 | 0.30 | 1.1 | -0.6 | H |
| oat_eta080 | 0.12 | 0.8 | -0.6 | eta |
| oat_eta140 | 0.12 | 1.4 | -0.6 | eta |
| oat_rho_m030 | 0.12 | 1.1 | -0.3 | rho |
| oat_rho_m085 | 0.12 | 1.1 | -0.85 | rho |

This design supports descriptive one-factor sensitivity. It is not a global parameter
uniformity theorem.

## 4. Estimators

For each selected cell and seed replicate, the protocol profiles:

1. **Crude single-level MC** on the finest grid under the natural law.
2. **Task-tuned CEM SLIS** on the four base cells. CEM training data and evaluation
   seeds are disjoint; training work is included in total work.
3. **DCS SLIS** on every grid, using the same defensive proposal as DCS-MLMC.
4. **DCS-MLMC** corrections on levels 1 through 4.

The CEM comparison is intentionally limited to base cells in this qualification to
bound laptop cost. A final paper confirmation must either tune it on every selected
cell with separately frozen training seeds or justify an amortized shared controller.

## 5. Sampling and seed separation

- profile paths per level and method: 8,192;
- independent seed replicates: 5;
- hierarchy: 8, 16, 32, 64, 128 steps;
- engine and evidence dtype: FFT and `torch.float64`;
- seed roles: `training` and `profile` are distinct;
- proposal and mixture-label streams are distinct; and
- a deterministic seed ledger is stored for every cell.

The cell-level progress artifact is atomically replaced using V4 retry logic. A
completed cell is not rerun after resume.

Text input hashes are computed after CRLF-to-LF normalization, matching the repository
text policy and preventing platform-specific false mismatches.

## 6. Multi-RMSE crossover

For relative RMSE targets 10%, 20%, and 30%, define sampling variance

`epsilon^2 = (p_target * relative_RMSE)^2`.

For each possible DCS start level, compute the continuous optimal-allocation
coefficient

`K_l0 = (sqrt(V_l0 C_l0) + sum_(l>l0) sqrt(V_Delta,l C_Delta,l))^2`.

The total-work diagnostic is

`W_l0 = P_profile + K_l0/epsilon^2`.

The finest start is DCS-SLIS, so it is always a candidate. The selected DCS
construction is then compared with crude single-level and, where enabled, CEM SLIS.
CEM training work and discarded profile work are included. No zero-event pilot is
allowed to appear as a zero-cost winner; an empirical zero variance makes that
single-level candidate unavailable.

The operation proxy is `paths * steps * log2(steps)`. Sampler wall time is also
recorded. The proxy is consistent with M7 but does not model every constant-factor
analytic cost; a final hardware study must report both work and wall time.

## 7. Qualification gates

The frozen qualification passes only if:

1. all 27 selected cells complete;
2. at least five seed replicates are present;
3. at least three RMSE targets are evaluated;
4. base, H, eta, and rho categories are all represented; and
5. at least 95% of full-hierarchy DCS estimates lie within four combined standard
   errors of the independent calibration reference.

There is deliberately no gate requiring MLMC to win. The experiment is designed to
choose the estimator construction and may falsify broad MLMC superiority.

## 8. Required interpretation

Allowed:

- count how often SLIS or an earlier MLMC start minimizes predicted total work;
- show how the choice changes with RMSE and one parameter at a time;
- report training-inclusive comparisons on the four CEM-enabled base cells; and
- use the result to freeze a larger final protocol.

Prohibited:

- call this an untouched confirmation;
- infer a continuous-time result;
- treat a fitted work coefficient as an achieved RMSE experiment;
- claim CEM was beaten outside the four cells where it was trained and evaluated; or
- attribute the earlier M7 combined-regime effect to H alone.

## 9. Final-confirmation requirements after qualification

A submission-level successor needs:

1. at least 10--20 independent clusters in the selected OAT matrix;
2. actual achieved-RMSE runs at more than one RMSE, not coefficients alone;
3. task-tuned SLIS or a frozen amortized proposal on every headline cell;
4. a clean independent seed namespace created only after this qualification is
   audited;
5. an independent machine/environment reproduction; and
6. either a continuous-time bias budget or an explicit finite-grid title and claim.
