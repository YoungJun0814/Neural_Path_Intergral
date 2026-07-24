# G11 V8 Top-Journal Implementation Plan

Date: 2026-07-25

Status: implementation started; no V8 outcome protocol is frozen

Primary objective: convert the confirmed V7 finite-grid DCS mechanism into a
top-journal candidate through a new model-level theorem, strong external
comparators, outcome-blind confirmation, and independent physical-environment
reproduction.

## 0. Executive decision

V8 keeps the confirmed V7 estimator core. It does not add quantum terminology and
does not call the current method a neural architecture. The recommended paper is:

> **Defensive Conditional Path Integration for Rare Events under Rough Volatility:
> Exactness, Complexity, and Amortized Efficiency**

The paper is promoted only if three distinct claims pass:

1. **Exactness:** the estimator targets the declared finite-grid probability with an
   exact balance-mixture likelihood and ordinary sample mean.
2. **Mechanism:** DCS is the proposal-conditional expectation of the raw contribution
   and has a quantitatively nontrivial Rao--Blackwell variance advantage.
3. **External competitiveness:** DCS retains a training-inclusive achieved-RMSE work
   advantage against predeclared strong external comparators.

V7 already establishes Claims 1 and 2 on the frozen 18-cell matrix. V8 must add a
new theory result and Claim 3 without changing or relabelling V7 outcomes.

## 1. Non-negotiable scientific boundaries

- The current primary estimand is a probability on a declared 128-step grid.
- A discrete barrier is not a continuously monitored barrier.
- The proposal conditional law of the integrated coordinate is generally not a
  standard Gaussian under a mixture proposal.
- The DCS identity must be derived through the exact target-over-mixture likelihood
  cancellation.
- The final estimator is an ordinary, non-self-normalized mean.
- Training, tuning, failed retries, profiling, planning, and final work are charged.
- Pilot, training, reference, mechanism-probe, qualification, and confirmation seeds
  are disjoint.
- Cells or paths are not treated as independent inferential replicates; independent
  seed clusters are the inference units.
- A correction-variance rate is not called an end-to-end complexity theorem unless
  weak-bias and cost exponents are also available.
- An arbitrary full-path normalizing flow may not be combined with DCS unless the
  conditional integral and exact density remain tractable.

## 2. Phase and commit policy

Each phase has one commit at most, created only after the phase implementation,
theory audit, technical audit, tests, lint, and type checks pass. Intermediate
commits are prohibited. A failed phase is left uncommitted until fixed or explicitly
closed by a failure decision.

| Phase | Purpose | End-of-phase commit condition |
|---|---|---|
| P0 | Claim, estimand, comparator, and prohibited-claim contract | machine audit and corruption tests pass |
| P1 | Reproducible novelty and closest-work audit | search ledger, primary-source matrix, and audit pass |
| P2 | Exact finite-grid and strict-improvement theorem stack | proof ledger and symbolic/oracle tests pass |
| P3 | Barrier mesh, weak-bias, and complexity route | theorem proved or scope formally downgraded |
| P4 | Strong-baseline common execution framework | all baseline likelihood and cost oracles pass |
| P5 | Reference and one-factor experimental matrix | independent reference and matrix audits pass |
| P6 | Statistical design and power planner | primary family, thresholds, and cluster count fixed |
| P7 | Falsification-first development | failure analysis complete; no confirmatory claim |
| P8 | Independent-seed qualification | all qualification gates and audits pass |
| P9 | Outcome-blind confirmation freeze | clean source, image, configs, and hashes bound |
| P10 | Formal confirmation | complete uncensored matrix and independent audits pass |
| P11 | Independent physical reproduction | new machine, new seeds, same source and effect gates pass |
| P12 | External proof and code review | all findings resolved or disclosed |
| P13 | Manuscript and artifact package | claim-to-evidence audit passes |
| P14 | Venue-specific submission audit | journal scope, format, and disclosure checks pass |

## 3. P0 -- claim and estimand contract

### Deliverables

- this implementation plan;
- `docs/theory/G11_V8_CLAIM_AND_ESTIMAND_CONTRACT.md`;
- `configs/g11_v8/top_journal_claim_contract_v1.yaml`;
- `experiments/g11_v8_claim_contract_audit.py`; and
- corruption-focused tests.

### Fixed comparator roles

| Role | Comparator | Reason |
|---|---|---|
| Mechanism | fixed raw defensive IS | identical proposal isolates DCS |
| Adaptive work | task-tuned pure CEM | strongest current repeated-query opponent |
| Closest published method | numerical-smoothing RQMC | closest conditional-smoothing computation line |
| Required secondary | crude/antithetic MC | prevents moderate-event overclaim |
| Required secondary | defensive CEM | separates safety from proposal quality |
| Required secondary | large-deviation adaptive IS | strong rare-event proposal family |
| Required secondary | exact-likelihood flow IS | contemporary flexible proposal family |

The headline may not select a comparator after outcomes. Superiority against more
than one primary external comparator requires simultaneous one-sided intervals.

### P0 pass gate

- at most three paper contributions;
- explicit finite-grid target;
- continuous monitoring set to false;
- terminal and discrete barrier are primary;
- occupation events are excluded from the primary theorem;
- all comparator roles are present;
- training-inclusive cost is required;
- forbidden claims are machine-readable; and
- one end-of-phase commit is declared.

## 4. P1 -- reproducible novelty audit

### Closest-work families

The primary-source audit must cover:

1. rough Bergomi conditional Monte Carlo and variance reduction;
2. numerical smoothing with QMC, ASGQ, and MLMC;
3. multilevel importance sampling for discontinuous observables;
4. balance-heuristic multiple importance sampling;
5. large-deviation and state-dependent adaptive IS;
6. CEM rare-event estimators;
7. exact-likelihood normalizing-flow rare-event estimators;
8. rough-volatility weak approximation and MLMC; and
9. learning-enhanced control variates under rough volatility.

Every record contains the query, interface, access date, primary URL/DOI, version,
model, event, proposal, density contract, conditional integration, theorem, work
accounting, code availability, overlap, and non-overlap.

### Required closest baselines

- McCrickerd and Pakkanen, conditional rough Bergomi Monte Carlo:
  <https://arxiv.org/abs/1708.02563>
- Bayer, Ben Hammouda, and Tempone, numerical smoothing with QMC:
  <https://arxiv.org/abs/2111.01874>
- Bayer, Ben Hammouda, and Tempone, numerical smoothing with MLMC:
  <https://arxiv.org/abs/2003.05708>
- Tong and Stadler, large-deviation adaptive IS:
  <https://arxiv.org/abs/2209.06278>
- Gao et al., flow-assisted rare-event IS:
  <https://arxiv.org/abs/2310.19167>
- Giles, MLMC complexity theorem:
  <https://doi.org/10.1287/opre.1070.0496>

### P1 falsification rule

If prior work contains the material mathematical mechanism, V8 may not claim that
conditional smoothing itself is new. The contribution must be narrowed to the
defensive residual-mixture identity, rough-path specialization, a new model-level
rate or strict-efficiency theorem, and confirmatory total-work evidence. If that
boundary is still incremental after expert review, the top-journal route stops.

## 5. P2 -- finite-grid theorem stack

### T1. Defensive mixture exactness

For

\[
Q=\delta P+(1-\delta)\widetilde Q,\qquad \delta>0,
\]

prove absolute continuity, the exact balance likelihood, the bound
\(dP/dQ\leq 1/\delta\), unbiasedness, and square integrability.

### T2. Proposal-conditional DCS identity

For \(X=UZ+R\), prove

\[
Y_{\mathrm{DCS}}(R)=E_Q[Y_{\mathrm{raw}}\mid R].
\]

No proof may replace \(Q(Z\mid R)\) by a standard Gaussian without deriving the
mixture cancellation.

### T3. Exact variance decomposition

Prove

\[
\operatorname{Var}(Y_{\mathrm{raw}})
-\operatorname{Var}(Y_{\mathrm{DCS}})
=E[\operatorname{Var}(Y_{\mathrm{raw}}\mid R)].
\]

### T4. Strict improvement

Identify nondegeneracy assumptions under which the right-hand side is strictly
positive. A quantitative or rare-event-asymptotic lower bound is the preferred new
theoretical contribution. A renamed classical Rao--Blackwell inequality is not
sufficient for the strongest journal tier.

### T5. Finite-grid scalar threshold

Prove measurability, tie handling, zero-slope handling, and exact scalar-threshold
representations for terminal and discrete-barrier events.

## 6. P3 -- barrier, weak bias, and complexity

Write the fine/coarse threshold defect as

\[
A_h-A_{2h}
=D_{\mathrm{coefficient}}
+D_{\mathrm{active-time}}
+D_{\mathrm{mesh}}.
\]

The mesh term includes barrier crossings visible only on the fine grid and may not
be omitted.

The desired chain is:

\[
E|A_h-A_{2h}|^2\leq C h^{2r},
\]

\[
\operatorname{Var}(Y_h^{\mathrm{DCS}}-Y_{2h}^{\mathrm{DCS}})
=O(h^{2r}),
\]

\[
|E[Y_h]-E[Y]|\leq C h^\alpha.
\]

Only after proving or citing the weak-bias exponent \(\alpha\), correction-variance
exponent \(\beta\), and sample-cost exponent \(\gamma\) may the standard MLMC
complexity regimes be stated.

### P3 decision

- full model-level proof: theory-led top-journal route remains open;
- conditional rate with explicit assumptions: computational journal route;
- empirical slopes only: no asymptotic complexity claim;
- barrier proof fails: terminal primary theorem, barrier finite-grid experiment only.

## 7. P4 -- strong-baseline framework

Every baseline implements:

```text
train(task, budget, seed) -> frozen proposal
plan(task, proposal, pilot_seed) -> integer allocation
estimate(task, proposal, final_seed) -> estimate, variance, work
audit(artifact) -> pass/fail
```

### Baselines

- crude and antithetic MC;
- published-style conditional rough Bergomi estimator;
- fresh task-tuned pure CEM;
- fresh task-tuned defensive CEM;
- numerical-smoothing RQMC;
- large-deviation subspace adaptive IS; and
- exact-likelihood flow-assisted IS.

### Flow constraint

An optional Residual-Flow DCS extension is permitted only if

\[
q_\phi(z,r)=q_\phi(r)q(z\mid r)
\]

has an exact density and analytically or rigorously numerically tractable
conditional event integral. An arbitrary flow that entangles \(Z\) and \(R\) is a
baseline, not a valid DCS extension.

### Cost ledger

The ledger charges training samples, optimizer steps, hyperparameter trials, failed
restarts, screening, planning, final sampling, likelihoods, CDF/quadrature calls,
wall time, CPU/GPU time, and peak memory.

Report:

1. algorithmic work units;
2. wall time on declared standardized hardware; and
3. actual compute cost or energy-normalized cost for heterogeneous CPU/GPU methods.

## 8. P5 -- reference and matrix

### Primary matrix

- \(H\in\{0.05,0.12,0.20\}\);
- terminal and discrete barrier;
- nominal probabilities \(10^{-2},10^{-3},10^{-4},10^{-5}\);
- one primary maturity and fixed primary \(\eta,\rho\); and
- 24 primary cells.

### Robustness matrix

Change one factor at a time:

- \(H\);
- volatility of volatility \(\eta\);
- correlation \(\rho\);
- maturity \(T\); and
- barrier monitoring frequency.

### Mesh matrix

Use 32, 64, 128, 256, and 512 steps. Measure coefficient, active-time, fine-only
crossing, threshold, weak-bias, correction-variance, and cost diagnostics.

### References

Reference seeds and execution code paths are independent of final methods. Use at
least two cross-check methods where feasible. Reference standard error must be no
larger than 10--20% of the final target standard error and must enter the reported
accuracy calculation.

## 9. P6 -- statistical design

### Primary endpoints

\[
R_{\mathrm{var}}
=\frac{\operatorname{Var}(Y_{\mathrm{raw}})}
{\operatorname{Var}(Y_{\mathrm{DCS}})}
\]

and

\[
R_{\mathrm{work},b}
=\frac{W_b}{W_{\mathrm{DCS}}}
\]

for each predeclared primary baseline \(b\).

Cell log-ratios are equally averaged within each independent seed cluster. Cluster
averages are the inference units.

### Provisional practical gates

These thresholds are fixed now but must be bound again in the outcome-blind P9
receipt:

- probe-variance lower ratio greater than 2.0;
- production-variance lower ratio greater than 2.0;
- final-work lower ratio greater than 1.5 versus fixed raw;
- training-inclusive lower ratio greater than 1.20 versus each primary external
  comparator;
- exact attainment lower bound at least 0.80;
- simultaneous RMSE-upper/tolerance ratio at most 1;
- no resource censoring;
- floor-binding fraction at most 5%; and
- zero duplicate or cross-phase seed intersections.

If the paper claims superiority over multiple baselines, simultaneous intervals are
required. The weakest comparator may not be selected after outcomes.

## 10. P7--P11 -- evidence ladder

### P7 development

Use development-only seeds to falsify likelihood normalization, mean identity,
conditional integration, flow Jacobians, proposal coverage, CEM collapse, weight
tails, seed separation, checkpoint identity, and cost accounting. Development
outcomes are never promoted to confirmation.

### P8 qualification

Use 24--32 new clusters. Estimate confirmation power and resources. All accuracy,
mechanism, comparator, cost, and provenance gates must pass before P9.

### P9 freeze

Bind clean source commit, container image, configs, references, proposals, baseline
commits, hyperparameters, primary claim family, thresholds, bootstrap seeds, seed
namespaces, work caps, expected records, and failure rules. Any later bug requires a
new source, freeze version, and seed namespace.

### P10 confirmation

Execute the complete matrix with no record deletion. Run independent record,
resource, accuracy, effect, seed, and aggregate audits. A capped method is retained
as censored according to the frozen failure rule.

### P11 independent physical reproduction

Use another physical CPU/GPU host, a clean clone, immutable container, and disjoint
seeds. Algorithmic work is primary; wall time is hardware-sensitive and secondary.

## 11. P12--P14 -- review, manuscript, and submission

### External reviews

The mathematical review covers likelihood cancellation, conditional laws,
measurability, inverse slopes, active times, mesh enrichment, weak bias, and MLMC
exponents. The code review covers estimators, couplings, seeds, checkpoints,
references, baselines, and cost ledgers.

### Manuscript structure

1. problem and financial relevance;
2. exact novelty boundary;
3. finite-grid rough-volatility target;
4. defensive path proposal;
5. DCS identity and strict-improvement theorem;
6. mesh, bias, and complexity analysis;
7. algorithms and total-work accounting;
8. frozen experiment;
9. strong comparators and ablations;
10. negative results and limitations;
11. reproduction statement; and
12. proof and artifact appendices.

### Venue decision

- Full model-level theory plus external superiority: strongest mathematical-finance
  venue route.
- Conditional theory plus significant computational improvement: rigorous
  computational financial-mathematics route.
- No strong-baseline advantage or no surviving novelty: stop the top-journal claim
  and publish a narrower finite-grid or reproducibility result.

## 12. Phase-level error audit checklist

At every phase end answer:

1. Did any claim silently change its estimand?
2. Was a proposal conditional law replaced by a target-law Gaussian?
3. Was any normalization, clipping, or record deletion introduced?
4. Were training, tuning, failures, and retries charged?
5. Did any pilot, reference, or final seed overlap?
6. Was a development choice tuned on confirmation outcomes?
7. Was a conditional theorem described as unconditional?
8. Was a finite-grid result described as continuous time?
9. Were cells or paths used as pseudoreplicates?
10. Was wall time compared across heterogeneous hardware without qualification?
11. Did a flow destroy the DCS conditional integral?
12. Does every table value trace to a hashed artifact?

Any unresolved `yes` blocks the phase commit.

## 13. Estimated schedule

| Work | Estimate |
|---|---:|
| P0--P1 claim and novelty boundary | 3--4 weeks |
| P2--P3 theory | 2--4 months |
| P4 baseline framework | 2--3 months |
| P5--P6 matrix, references, statistics | 1--2 months |
| P7 development | about 1 month |
| P8 qualification | 2--4 weeks |
| P9 freeze | 1 week |
| P10 confirmation | 1--4 compute weeks |
| P11 physical reproduction | 2--3 weeks |
| P12 external review | about 1 month |
| P13--P14 manuscript and submission | 1--2 months |

The realistic program is six to ten months. No schedule is allowed to weaken a
scientific gate.
