# G11 V7 mechanism-first implementation plan

Date: 2026-07-24  
Status: development protocol implemented; development outcomes not yet observed  
Target: isolate the DCS mathematical mechanism before rebuilding any router

## 1. Decision

V6 confirmed a real 2.32x training-inclusive repeated-query saving against
fresh pure CEM and reproduced it on disjoint Linux seeds.  It did not establish
the intended Hybrid mechanism:

- every route selected DCS-SLIS;
- final and non-training work were worse for the policy;
- most final allocations hit unequal method-specific floors; and
- the gain came from amortizing a shared proposal bank.

V7 therefore starts with the narrow causal question:

> under one identical defensive proposal, does exact control-span
> marginalization reduce variance enough to reduce achieved-RMSE work after
> its own arithmetic cost is exposed?

The router is not expanded until this question passes.  Adding more routes to
an unidentified mechanism would make attribution harder, not stronger.

## 2. V7 model

The V7 core is:

> an amortized defensive path-integral importance sampler whose event
> coordinate is integrated exactly and whose raw/DCS choice is evaluated
> under a mechanism-identifiable work contract.

The estimator remains exact for the declared 128-step rBergomi probability.
The proposal is the V6 task-conditioned zero/half/full control bank:

- weights `(0.15, 0.35, 0.50)`;
- deterministic terminal and barrier schedules;
- exact balance-mixture likelihood;
- natural defensive component;
- no self-normalization, clipping, or pilot reuse.

V7 adds two explicitly separated paths:

1. **paired oracle probe:** one simulated batch returns raw and DCS
   contributions on identical paths;
2. **production comparison:** raw-only and DCS production runs have separate
   planning/final seeds and isolated wall-time ledgers.

The oracle probe proves nothing from its fitted moments; it tests executable
implications of the already proved finite-grid conditional identity.

## 3. Fairness corrections

### 3.1 Raw-only comparator

The old raw sampler called the full DCS evaluator and discarded the
marginalized output.  Its path values were correct, but its wall time included
unnecessary DCS arithmetic.  V7 computes raw values directly as

\[
1_A\exp(\log L)
\]

or the exact adjacent-event difference times the same likelihood.

Oracle tests require agreement with the generic DCS evaluator within
`3e-16` absolute error under identical seeds and monkeypatch the DCS evaluators
to prove the raw path does not call them.

### 3.2 Identical proposal

Both methods use the same frozen proposal weights, controls, training source,
simulator, grid, references, and task conventions.  Only conditional
marginalization is switched.

### 3.3 Identical target and floor

Development uses:

- requested relative sampling RMSE: `0.10`;
- minimum final samples: `512` for both methods;
- planning: 8 independent replicates of 512 paths;
- allocation safety factor: `6.0`;
- 18 cells and 8 seed clusters.

The prior 24-cluster V6 design variances project:

| Quantity | V7 development projection |
|---|---:|
| DCS records at floor | 0% |
| raw records at floor | 0% |
| geometric raw/DCS final sample ratio | about 3.45 |
| DCS final samples at 24-cluster scale | about 1.73 million |
| raw final samples at 24-cluster scale | about 5.53 million |

These are resource projections only.  They are not V7 results and cannot enter
the paper as evidence.

### 3.4 Work definitions

V7 reports all of the following:

- single-path empirical variance ratio;
- final sampling work ratio;
- non-training work ratio: planning plus final;
- training-inclusive work ratio;
- isolated final wall-time ratio; and
- floor occupancy by method.

The paired oracle-probe cost is separately labelled `mechanism_probe`.  It is
not added to either production estimator and cannot create an artificial
speedup.

## 4. Development protocols fixed before results

### 4.1 Paired probe

Config:
`configs/g11_v7/mechanism_probe_development_v2.yaml`

- 18 cells;
- 8 clusters;
- 4,096 paired contributions per cell-cluster;
- disjoint `g11-v7-mechanism-probe-*` seeds;
- one-sided 95% cluster-level inference;
- development lower raw/DCS variance-ratio threshold: 1.5;
- maximum absolute per-record residual z diagnostic: 4.5;
- maximum absolute DCS/residual covariance z diagnostic: 4.5.

V1 used a raw maximum-correlation gate and failed.  That statistic was
ill-scaled in the rarest heavy-tailed cells, so V1 remains retired and V2 uses a
new protocol and seed namespace.  See
`G11_V7_MECHANISM_PROBE_V1_FAILURE_2026-07-24.md`.

The two z thresholds are development falsification thresholds.  A future formal family
must use a predeclared simultaneous procedure rather than copying these maxima
without power analysis.

### 4.2 Fixed production estimators

Config:
`configs/g11_v7/fixed_estimators_development_v1.yaml`

- fixed DCS-SLIS and fixed raw defensive IS;
- same 18 cells and 8 clusters;
- 10% requested relative RMSE;
- common floor 512;
- independent planning and final seed roles;
- verified proposal-training source;
- exact work and checkpoint ledgers.

The file deliberately reuses the already audited V6 secondary-runner schema.
The protocol ID and seed namespace are V7.  A formal V7 freeze will receive a
native V7 receipt and independent audit before qualification.

### 4.3 Joint analysis

Config:
`configs/g11_v7/mechanism_analysis_development_v2.yaml`

Development passes only if:

1. probe and production matrices are complete;
2. manifest, reference, proposal source, and execution source agree;
3. probe and production seed values are disjoint;
4. paired-probe variance lower ratio is at least 1.5;
5. production empirical variance lower ratio is at least 1.5;
6. production final-work lower ratio is at least 1.2; and
7. no more than 10% of either method's records hit the common floor.

All ratios use equal cell weights inside each cluster, followed by a one-sided
t interval across clusters.  Eight clusters are for development and power
planning, not a publication-level final inference.

## 5. Qualification design after development

Development has only three legitimate outcomes.

### Outcome D1: mechanism and work pass

Freeze a new qualification protocol before new samples:

- cluster count selected by paired power analysis with a minimum of 24;
- preferably 64 clusters for comparison with V6;
- predeclared simultaneous accuracy family;
- raw/DCS mechanism, final-work, and wall-time endpoints;
- task- and rarity-stratified secondary effects;
- independent JSON-only audit; and
- Linux reproduction with a disjoint seed namespace.

Only after D1 qualification may a raw/DCS router be introduced.

### Outcome D2: variance passes, work fails

Profile the incremental conditional-integration cost and optimize the DCS
kernel without changing the estimator.  Repeat a new development protocol.
Do not claim practical superiority from variance alone.

### Outcome D3: variance fails

Check exactness, proposal alignment, and cell strata.  If the identity is
correct but the practical ratio is small, report the negative result and do
not spend resources on routing.  Consider a predeclared higher-rank control
span only as a new method with a new theorem and exact multivariate integral.

## 6. Router re-entry gate

A router returns only if qualification establishes heterogeneous crossovers:

- some frozen strata favor raw;
- some favor rank-one DCS;
- and, if retained, some favor multilevel Hybrid after profiling cost.

If DCS dominates every declared stratum, the honest model is a fixed DCS
estimator, not a router.  If raw dominates after arithmetic cost, DCS is a
theoretical variance reduction without a practical case.

## 7. Route B theory work

The finite-grid Rao--Blackwell identity is complete.  The remaining
top-journal mathematical obligations are:

1. independent line-by-line review of the terminal coefficient coupling;
2. direct construction and measurability of \(A_h,B_h\);
3. verified weak-bias passage to the continuous terminal model;
4. exact statement of constants and compact parameter domains; and
5. a separate barrier active-index/mesh-enrichment theorem, or explicit
   exclusion of barriers from the theory headline.

Empirical slope regressions remain diagnostics and cannot close these items.

## 8. Stronger comparator stage

After mechanism qualification, compare against:

- fresh task-tuned pure CEM;
- amortized fixed proposal without DCS;
- adaptive importance sampling with all adaptation work charged;
- a credible flow/transport proposal if exact likelihood and training cost are
  available; and
- crude MC only where the frozen rarity band makes it computationally legal.

Comparator inclusion is decided before outcome inspection.  A neural or flow
baseline without exact density evaluation is not admissible for the unbiased
headline.

## 9. Error audit checklist

Every V7 artifact must fail closed on:

- duplicate or incomplete cell-cluster keys;
- unknown config fields;
- proposal, manifest, reference, or source hash drift;
- reused probe/planning/final seeds;
- final count below the common floor;
- nonfinite likelihood, contribution, variance, or ratio;
- self-normalization or clipping;
- dirty source in formal phases;
- resource censoring;
- missing work categories; and
- outcome-dependent threshold or cluster-count changes.

## 10. Journal-level gate

V7 mechanism qualification plus the existing V6 Windows/Linux amortization
result would support a strong specialized computational paper.  A top
mathematical-finance or numerical-analysis submission additionally requires:

- a verified model-level terminal theorem;
- a clear novelty boundary against conditional Monte Carlo, SLIS/MLIS, and
  hierarchical importance sampling;
- stronger adaptive/transport comparators;
- formal simultaneous accuracy inference;
- cross-platform reproduction of the final V7 protocol; and
- no claim that the classical Rao--Blackwell inequality itself is novel.

The intended top-level contribution is the combination:

> exact rough-path control-span marginalization + model-specific analysis +
> mechanism-identified achieved-RMSE efficiency.

Any one component alone is below the intended journal tier.
