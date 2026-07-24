# G11 V7 mechanism qualification V1 decision

Date: 2026-07-24  
Decision: **PASS — authorizes a separately frozen confirmation**  
Role: mechanism qualification, not untouched confirmation

## 1. Question and locked comparison

V6 established a repeated-query, training-inclusive advantage over pure CEM, but
every routed policy chose DCS-SLIS.  It therefore did not isolate whether the gain
came from the DCS conditional integration itself or only from amortizing an existing
proposal bank.

V7 asks the narrower causal question:

> With the same target cells, deterministic proposal bank, mixture weights, final
> sample floor, requested relative RMSE, and independent seed design, does replacing
> a raw defensive observation by its DCS conditional expectation reduce variance
> and achieved-RMSE work?

The comparison is fixed DCS-SLIS versus fixed raw defensive importance sampling.
The raw implementation has a separately tested fast path and does not execute DCS
arithmetic and discard its result.

## 2. Frozen design

- 18 finite-grid rBergomi cells;
- terminal and discrete-barrier events at nominal probabilities from
  \(10^{-3}\) through \(10^{-5}\);
- 24 independent clusters per cell;
- 4,096 common-path paired samples per cell-cluster in the mechanism probe;
- 10% requested relative sampling RMSE in the production estimator;
- common final-sample floor of 512;
- eight planning replicates of 512 paths;
- exact balance-mixture likelihood, no self-normalization;
- disjoint mechanism-probe, planning, and final seed namespaces; and
- 72 simultaneous accuracy claims with family-wise alpha 0.05.

The outcome-locked freeze receipt has SHA-256
`24942e124a585c5744afa1c398b516e988b9d326188bc11e7d463d0fa6c313ac`.
It was created from a clean source tree at commit `62da8b9` before qualification
outcomes were generated.

## 3. Primary results

All prespecified mechanism, work, floor, provenance, and aggregate-accuracy gates
passed.

| Endpoint, raw divided by DCS | Point ratio | One-sided 95% lower ratio | Frozen minimum |
|---|---:|---:|---:|
| Common-path probe variance | 3.3977 | 3.3071 | 2.0 |
| Production execution variance | 3.5376 | 3.4359 | 2.0 |
| Final sampling work | 3.4529 | 3.3287 | 1.5 |
| Non-training total work | 1.9313 | 1.8817 | diagnostic |
| Training-inclusive total work | 1.9033 | 1.8551 | diagnostic |
| Isolated final wall time | 2.5088 | 2.4172 | diagnostic |

Neither method hit the common final-sample floor.  The fixed-estimator result
contains 864 complete records and no censoring.  DCS used 1,610,573 final samples
versus 5,165,808 for raw, and 3.117 billion versus 6.302 billion
training-inclusive work units.

The paired Rao--Blackwell falsification diagnostics also passed:

- maximum absolute residual-mean z: 2.7156, below 4.5;
- maximum absolute covariance-product z: 3.6505, below 4.5; and
- 432/432 independently recomputed paired records passed.

These z statistics are diagnostics for violations of the conditional-expectation
identity.  They are not a proof of independence or a replacement for the analytic
Rao--Blackwell identity.

## 4. Accuracy co-gates and visible deviations

The simultaneous accuracy analysis passed all 72 predeclared claims:

- Bonferroni simultaneous confidence level per claim:
  \(1-0.05/72=0.9993055556\);
- exact Clopper--Pearson attainment bounds;
- fixed-seed 50,000-repetition percentile bootstrap for RMSE;
- minimum attainment lower bound: 0.6633, above 0.60; and
- maximum RMSE-upper/tolerance ratio: 0.7546, below 1.

Three DCS method-cell groups attained the per-record target in 23/24 clusters:
`h012-barrier-p1e-03`, `h012-barrier-p1e-04`, and
`h005-barrier-p1e-04`.  Raw attained it in 24/24 clusters in every group.
This is not hidden or relabelled.  The per-record Boolean was predeclared as a
diagnostic; the decision rule was the method-cell exact-attainment plus bootstrap
RMSE co-gate.  Those aggregate gates passed.

The attainment margin, 0.6633 versus 0.60, is materially narrower than the
mechanism-ratio margins.  Confirmation therefore requires more clusters and new
seeds; qualification must not be presented as the final accuracy claim.

## 5. Independent audit and artifact ledger

The aggregate independent auditor:

- imports neither production mechanism analyzer nor production accuracy analyzer;
- verifies freeze and configuration hashes;
- verifies source commit and clean-worktree claims;
- checks all expected records and seed disjointness;
- recomputes the five cluster-level work/variance effects;
- recomputes all 36 method-cell accuracy groups, including the full 50,000
  bootstrap repetitions; and
- fails closed on any mismatch.

It passed with zero failures at audit-source commit `7036c90`.  The aggregate audit
SHA-256 is
`d8bbd68cd2d2e3efb0385af0add98119def35289315892ca3010fc0718092034`.

| Artifact | SHA-256 |
|---|---|
| Paired probe | `e532913cdafc6bc9a0bf37f278f81891f75f6cf8023f3a7f39791df82b24666c` |
| Independent paired-probe audit | `66802f7459b80fbc034d97d7f55e6a89a2dd67204ba2eb5f6d424c04d47bec63` |
| Fixed estimators | `69fd5619d00177e4dd15a6ad4ee64f9a1f6314b66f05e9989539113ba20a98e6` |
| Independent fixed-estimator audit | `01e1d6a03c8fcb61088ef57e7cd157e0beab32f54fc0f58d5034df8f22302011` |
| Resource supplement | `7072141da568e53914e4fad00e2d0aaf1d03ce99ea4e82d398b1308fff7da5ce` |
| Joint mechanism/work analysis | `89db2f045a7736ce2cd75b53919928f4761bd46e6c66dfcaf15b78cf6655d1ae` |
| Simultaneous accuracy analysis | `080d36c0cb6ebf1e3d5083fc6c3b76c55f61837a0cf2a15e628ab28a0ee5be96` |

## 6. What is established

Within the frozen finite-grid, task-conditioned proposal regime, the evidence now
supports a mechanism-level statement:

> DCS is not merely benefiting from a cheaper reused proposal.  Against a raw
> estimator using the same proposal, it removes conditional Gaussian noise,
> reduces variance by about 3.5 times, reduces final sampling work by about
> 3.45 times, and remains faster in isolated final wall time.

This closes the principal mechanism-identification gap left by V6 qualification.
It does not establish uniform superiority over every importance-sampling proposal,
every rough-volatility parameter, or continuously monitored events.

## 7. Theory and measurement boundaries

1. The Rao--Blackwell identity is exact for the implemented finite-dimensional
   Gaussian shift family and exactly represented scalar-threshold tasks.
2. Cluster-level t bounds have only 24 independent units, although the observed
   margins are large.
3. Clopper--Pearson coverage is finite-sample exact.  The percentile bootstrap RMSE
   bound is a predeclared nominal engineering procedure, not an exact theorem.
4. The work proxy deliberately counts simulated path operations but does not assign
   every floating-point instruction inside DCS.  The 2.51 times isolated wall-time
   ratio supplies a separate practical check.
5. References and the deterministic proposal bank were inherited from frozen V6
   artifacts.  This is valid for qualification, but new outcome seeds are mandatory
   for confirmation.
6. The estimand is the declared 128-step finite-grid probability.  No
   continuous-monitoring unbiasedness claim follows.

## 8. Publication-level decision

The current package is stronger than a typical implementation-only project and is
a credible PhD-level computational core.  It is not yet a top-journal submission
because the following remain:

1. a separately frozen, new-seed confirmation with at least 64 clusters;
2. a V7 cross-platform reproduction of that confirmation;
3. external proof review of the theorem ledger;
4. a model-level barrier/mesh or explicitly bounded finite-grid theory section;
5. strong adaptive/flow and task-tuned CEM comparators under the same
   training-inclusive contract; and
6. a refreshed, reproducible novelty/literature audit.

The next authorized action is to freeze the 64-cluster confirmation without changing
the estimator, thresholds, cell matrix, or proposal after seeing its outcomes.
