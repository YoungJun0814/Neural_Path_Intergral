# G11 V5 Selector Qualification Decision

Date: 2026-07-22

Decision: **G4 passed for the frozen finite-look oracle protocol.**

This decision is deliberately narrow. It qualifies the implementation of the
uncertainty-aware selector under the eight bounded oracle cases declared before
execution. It does not establish a rough-Bergomi convergence rate, superiority over
CEM or MLMC, or readiness for an untouched confirmatory study.

## 1. Frozen execution identity

- protocol: `g11-v5-selector-qualification-v1`
- schema: `npi.g11.v5-selector-qualification.v1`
- source commit: `a49c51ac1ac7c817afcf07a411a0bff7df7223f7`
- source worktree: clean detached worktree
- config SHA-256: `c90bbc62813323b8a69e04f622096d1bc256c95fe480b1115a648a1e74a08c88`
- seed-ledger SHA-256: `f9b92e7736164b2a944b09caff2c1cebad8991a2d328b2b0b6da4661c70a6f5c`
- result SHA-256: `0903f6698c7d67b49490e703ee2a37fb2849352bf5da24c027410dc02241cfbc`
- execution size: 8 cases x 500 repetitions = 4,000 records
- smoke flag: `false`

The formal-readiness checks recorded in the artifact are all true: frozen config,
clean source, and non-smoke execution.

## 2. Predeclared gate results

| Gate | Observed result | Decision |
|---|---:|---|
| simultaneous coverage at least declared 95% | 0.99875 | pass |
| invalid optimal-candidate elimination on the coverage event | 0 | pass |
| median work regret at most 1.10 | 1.00000 | pass |
| 90th-percentile work regret at most 1.25 | 1.20120 | pass |
| early-stop fraction at least 0.75 | 1.00000 | pass |

The artifact-level `qualification_passed` field is true, and every serialized gate
is true.

## 3. What this result establishes

Within the frozen bounded-oracle family, the implementation:

1. maintains the declared simultaneous variance-interval coverage;
2. does not eliminate the oracle-optimal candidate when the simultaneous coverage
   event holds;
3. satisfies the declared median and tail work-regret limits; and
4. terminates at a predeclared finite look in every repetition.

These observations supplement the deterministic selector theorem: the theorem
gives conditional safety on the interval-coverage event, while this experiment
checks the implemented finite-sample interval and selection chain on adversarial
oracle cases including near ties, misleading early observations, zero observed
corrections, and preprocessing-cost reversal.

## 4. Claim boundary and remaining gates

G4 is closed, but V5 remains **not ready to freeze**. The following obligations are
unaffected by this qualification:

- G2: terminal inverse-slope and coefficient-rate proof, or explicit demotion to a
  conditional theorem;
- G3: barrier early-active and fine-only mesh-enrichment moment rate, or explicit
  demotion;
- G5: full fresh-training baselines and references meeting their declared standard
  error contracts;
- G6: full V5 achieved-RMSE qualification and an updated V5-based power/resource
  forecast;
- G7--G10: novelty cutoff, frozen untouched confirmation, Linux reproduction, and
  manuscript/result audit.

Therefore the correct current statement is:

> The frozen oracle qualification supports the finite-look robust selector's
> implementation and predeclared finite-sample gates; model-level rough-volatility
> and comparative-performance claims remain open.
