# G11 V6 confirmation V2 decision

Date: 2026-07-24  
Execution source: `f1249c9d7d74502efbd4ec4fcdb529c1e9e307bd`  
Protocol: `g11-v6-confirmatory-v2`  
Decision: Windows confirmation and independent Linux reproduction pass;
top-journal mechanism claim does not yet pass

## 1. Frozen design

The confirmation used:

- 18 fixed-grid rough-Bergomi cells;
- 64 independent seed clusters;
- 1,152 pure-CEM records and 1,152 V6 policy records;
- the qualification-frozen task-conditioned proposal bank;
- 20% requested relative sampling RMSE;
- no self-normalization, clipping, record deletion, or resource censoring;
- exact training-inclusive work;
- equal weighting of cells inside each cluster;
- a one-sided 95% paired efficiency analysis;
- a predeclared practical geometric work ratio of 1.20; and
- per-method/per-cell attainment and bootstrap-RMSE co-gates.

The baseline, policy, audit, manifest, reference, power, and confirmatory config
hashes match the V2 freeze receipt.  Both result artifacts record a clean source
tree and the same committed source.

## 2. Integrity result

| Check | Result |
|---|---:|
| baseline records | 1,152/1,152 |
| policy records | 1,152/1,152 |
| complete paired records | 1,152/1,152 |
| baseline independent audits | 1,152/1,152 pass |
| policy independent audits | 1,152/1,152 pass |
| resource censoring | 0 |
| policy unresolved routes | 0 |
| policy final-seed separation failures | 0 |
| baseline resource supplement | pass |
| policy resource supplement | pass |

Five pure-CEM records missed their individual empirical-RMSE diagnostic and no
policy record did.  This does not invalidate the predeclared decision: random
per-record misses were explicitly deferred to the method-by-cell attainment and
bootstrap-RMSE co-gates.  All 36 method-by-cell accuracy groups pass.  The
smallest one-sided exact attainment lower bound is about 0.9049 against the
0.60 requirement, and the largest bootstrap-RMSE-upper/tolerance ratio is about
0.5512.

## 3. Primary confirmation result

For each cluster, log pure-CEM/policy work ratios were averaged equally across
the 18 cells.  Across 64 clusters:

| Quantity | Value |
|---|---:|
| geometric mean pure-CEM/policy total-work ratio | 2.3169 |
| one-sided 95% lower ratio | 2.3004 |
| predeclared practical ratio | 1.2000 |
| mean log ratio | 0.84024 |
| standard error of mean log ratio | 0.00430 |
| one-sided p-value against no saving | \(1.07\times10^{-89}\) |

The lower bound exceeds both 1.0 and the actual predeclared 1.20 practical
threshold.  The corrected confirmatory analyzer therefore passes every
scientific and formal gate.

The concluding software validation passes CI-scope Ruff, mypy over 81 source
files, and the full `523/523` pytest suite after adding the hardware-freeze and
post-hoc multiplicity-sensitivity checks.

## 4. Independent Linux reproduction

The reproduction used a clean in-container clone of execution source
`f1249c9`, PyTorch 2.9.1 CPU, Python 3.10, and Linux
`6.6.87.2-microsoft-standard-WSL2`.  Protocol V3 changed every protocol ID, so
the Windows V2 and Linux V3 seed values and seed-key identities are disjoint.
The reproduction matrix again contains 1,152 baseline and 1,152 policy records.
Both Linux-side frozen independent audits pass 1,152/1,152 records, and both
resource supplements pass.

| Quantity | Windows V2 | Linux V3 |
|---|---:|---:|
| geometric mean pure-CEM/policy work ratio | 2.3169 | 2.3186 |
| one-sided 95% lower ratio | 2.3004 | 2.3080 |
| mean log ratio | 0.84024 | 0.84097 |
| standard error | 0.00430 | 0.00275 |
| all 36 accuracy groups pass | yes | yes |

The frozen cross-environment audit reports:

- canonical seed count: 25,569;
- reproduction seed count: 25,459;
- disjoint seed streams: pass;
- different operating system: pass;
- same execution source commit: pass;
- same manifest, reference, power, and estimand: pass;
- both scientific gates: pass; and
- effect-difference z-score: 0.142 against the frozen maximum of 3.0.

`hardware_reproduction_passed=true`.  The Linux confirmation analyzer itself
runs from the later clean analysis commit containing the practical-threshold
correction, while both numerical execution artifacts use the identical
`f1249c9` estimator source.  The hardware audit checks this execution-source
identity explicitly.

## 5. Mechanistic decomposition

The headline total-work result is real, but its cause must be stated exactly.

| Work component | pure CEM | V6 policy |
|---|---:|---:|
| proposal training | 16,882,073,600 | 88,080,384 |
| allocation pilot | 4,227,858,432 | 2,113,929,216 |
| screening | 0 | 300,711,936 |
| final sampling | 5,313,779,072 | 8,537,311,104 |
| total | 26,423,711,104 | 11,040,032,640 |

Paired geometric diagnostics give:

- pure-CEM/DCS single-path variance: 2.1113;
- pure-CEM/policy final-work ratio: 0.5225;
- pure-CEM/policy non-training-work ratio: 0.8004; and
- pure-CEM/policy total-work ratio: 2.3169.

Thus DCS has lower final single-path variance than freshly trained pure CEM, but
the frozen minimum sample floors reverse that gain in final work.  The policy
uses more final and more total non-training work.  Its confirmed total-work
advantage comes primarily from amortizing one fixed proposal bank instead of
running a new CEM fit for every cell-cluster query.

This is a useful repeated-query result.  It is not evidence that the router or
Hybrid selector produced the speedup:

- 1,152/1,152 routes select DCS-SLIS;
- 1,152/1,152 route reasons are `minimum Hybrid profiling work exceeds the
  frozen cap`; and
- the selector contributes no selected multilevel route.

The honest current model name for the confirmed experiment is therefore:

> amortized task-conditioned defensive control-span Gaussian marginalization
> for repeated fixed-grid rare-event queries.

Calling the confirmed object an empirically successful Hybrid router would be
incorrect.

## 6. Theory-diagnostic result

The frozen terminal-theory diagnostic at source `66e0d4a` passes:

- exact fine/coarse direction geometry;
- pathwise reconstruction and decomposition;
- finite analytic terminal inverse-slope moment bounds;
- finite empirical inverse moments;
- complete terminal rate contracts for every declared \(r<H\); and
- no implementation failure.

All three terminal cells have stable fitted rate windows.  Their point slopes
exceed the conservative declared threshold and DCS second-moment exponents.
However, the lower bootstrap slope bound is below the declared DCS exponent for
the \(H=0.12\) terminal cell and is marginally below it for \(H=0.30\).  This
does not refute an analytic upper-rate theorem, but it prevents presenting the
finite-sample fits as proof.

The \(H=0.30\) barrier rate is unidentified by the common stability rule.
Barriers remain excluded from the terminal coefficient theorem.  The
coefficient and weak-rate arguments remain proof candidates requiring
independent mathematical review.

## 7. Deviations and corrections

### V1 abort

Confirmation V1 stopped fail-closed after a valid rare screening realization
excluded the frozen nominal point from an exact binomial interval.  Its 98
policy and 278 baseline records were retired.  V2 used new protocol IDs and a
disjoint seed namespace.  See
`G11_V6_CONFIRMATION_V1_ABORT_2026-07-24.md`.

### Analyst-blinding disclosure

During V2 monitoring, a compact one-line progress JSON was accidentally printed
when only its beginning was requested.  This exposed several policy-only record
values after the V2 protocol and source had already been frozen.  No paired
baseline comparison or aggregate result was inspected, and no config, seed
count, threshold, estimator, or gate was changed afterward.

The run remains a frozen, complete confirmation, but the paper must not claim
perfect analyst blinding.  It may accurately claim pre-outcome protocol freeze
and no post-freeze adaptation.

### Practical-threshold analyzer correction

The first V2 analysis implementation checked whether the one-sided work-ratio
lower bound exceeded 1.0 even though the frozen config declared 1.20.  After
the result, the analyzer was corrected to require the unchanged 1.20 threshold
and a regression test was added.  The observed lower bound is 2.3004, so the
decision is unchanged.  This correction and its timing must remain in the
reproducibility history.

## 8. Journal-level decision

This result is strong enough for a PhD-level computational chapter and supports
a serious specialized-journal engineering claim:

> for a predeclared repeated-query workload over the 18 fixed-grid cells, the
> amortized V6 estimator reduces training-inclusive work relative to fitting
> pure CEM separately for every query while maintaining the frozen aggregate
> accuracy criteria.

It is not yet enough for a top mathematical-finance or numerical-analysis
journal headline because:

1. the router and Hybrid selector collapse to one method;
2. the DCS mechanism is masked by unequal final-sample floors;
3. excluding proposal training reverses the work advantage;
4. the terminal coefficient/weak-rate proof has not received independent
   mathematical review;
5. the barrier rate theorem is open; and
6. the confirmatory accuracy family lacks a predeclared simultaneous
   multiplicity correction.

## 9. Required next study

The next experiment must be a new, explicitly exploratory-to-qualification V7
mechanism study, not a reinterpretation of V2:

1. compare raw SLIS and DCS-SLIS under the identical frozen proposal;
2. use identical minimum final-sample floors and tighter RMSE targets so the
   floor does not mask variance reduction;
3. report training-free and training-inclusive work separately;
4. predeclare one-off and repeated-query amortization regimes and their
   break-even query counts;
5. include stronger fixed-proposal, CEM, adaptive-IS, and flow/transport
   comparators where feasible;
6. use simultaneous accuracy inference across the headline family;
7. retain the completed Windows/Linux disjoint-seed reproduction in the
   submission archive; and
8. obtain an independent proof audit before claiming continuous-theory novelty.

Until those items are complete, the defensible classification is:

> confirmed practical amortization result; promising but not yet top-journal
> mechanism/theory result.
