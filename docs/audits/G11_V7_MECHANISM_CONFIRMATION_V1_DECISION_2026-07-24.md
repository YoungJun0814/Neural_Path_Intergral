# G11 V7 mechanism confirmation V1 decision

Date: 2026-07-24  
Decision: **PASS**  
Role: untouched, new-seed confirmation of the fixed-proposal DCS mechanism

## 1. Confirmatory claim

The V7 confirmation tests the following prespecified statement:

> On the frozen 18-cell, 128-step rBergomi matrix, and under the same deterministic
> defensive proposal, DCS conditional Gaussian integration retains a practically
> significant variance and achieved-RMSE work advantage over raw defensive
> importance sampling while both methods satisfy simultaneous accuracy gates.

This is a fixed DCS versus fixed raw comparison.  It is not a routing result and it
does not claim that the proposal bank is globally optimal.

## 2. Outcome lock and execution identity

The valid freeze V2 receipt:

- binds clean source commit `80a5bc9`;
- plans 64 clusters;
- permits no qualification-to-confirmation changes except cluster count,
  mechanically updated training amortization, protocol/seed namespaces, and
  bootstrap seed;
- binds the confirmation manifest, reference, proposal source, and four config
  hashes; and
- has SHA-256
  `d6382db7d5e72bbc32e4f68b0349de01a5b5b72dda9bec7e561bcfaf457bb518`.

Both Python 3.10 and 3.11 GitHub Actions jobs passed before formal outcomes were
generated.

## 3. Matrix and statistical unit

- 18 finite-grid cells;
- terminal and discrete-barrier tasks;
- roughness \(H\in\{0.05,0.12,0.20\}\);
- nominal probabilities from \(10^{-2}\) through \(10^{-5}\);
- 64 independent clusters;
- 1,152 paired mechanism records;
- 2,304 fixed-estimator records;
- 4,096 common-path samples per paired cell-cluster;
- 10% requested relative sampling RMSE;
- eight 512-sample planning replicates; and
- common final floor 512.

For the effect ratios, cell log-ratios are first equally averaged within each
cluster.  The 64 cluster averages are the independent inference units.  Samples or
cells are not falsely treated as independent replicates.

## 4. Confirmatory mechanism and efficiency results

Every frozen mechanism, integrity, floor, work, and provenance gate passed.

| Raw divided by DCS | Geometric ratio | One-sided 95% lower ratio | Frozen minimum |
|---|---:|---:|---:|
| Common-path probe variance | 3.4415 | 3.3953 | 2.0 |
| Production execution variance | 3.5151 | 3.4548 | 2.0 |
| Final sampling work | 3.3606 | 3.2975 | 1.5 |
| Non-training work | 1.8933 | 1.8674 | diagnostic |
| Training-inclusive total work | 1.8830 | 1.8573 | diagnostic |
| Isolated final wall time | 2.3553 | 2.3059 | diagnostic |

The common floor was used by neither method.  Thus the primary final-work ratio is
not created by both methods being pinned to an arbitrary minimum.

The paired falsification diagnostics passed:

- maximum absolute residual-mean z: 3.3909;
- maximum absolute covariance-product z: 3.5929; and
- frozen maximum for both: 4.5.

The independent paired auditor recomputed and passed 1,152/1,152 records.

## 5. Absolute computational totals

| Quantity | Fixed DCS | Fixed raw |
|---|---:|---:|
| Cell-cluster records | 1,152 | 1,152 |
| Final samples | 4,551,948 | 13,778,839 |
| Final work units | 4,078,545,408 | 12,345,839,744 |
| Training-inclusive work units | 8,394,484,224 | 16,661,778,560 |
| Isolated final wall seconds | 263.11 | 590.49 |

Ratios of these grand totals are descriptive.  The inferential headline ratios in
Section 4 use the prespecified cluster-balanced geometric analysis and should not be
replaced by ratios of totals.

## 6. Simultaneous accuracy

The formal family contains 72 claims:

- 18 cells;
- two methods; and
- an attainment claim plus an RMSE claim per method-cell.

The family-wise alpha is 0.05.  Attainment uses one-sided exact
Clopper--Pearson inversion.  RMSE uses a fixed-seed 50,000-repetition percentile
bootstrap at the Bonferroni-adjusted confidence level.

All 72 claims passed:

- minimum exact attainment lower bound: 0.8053, above 0.60;
- maximum bootstrap RMSE-upper/tolerance ratio: 0.7780, below 1.

The visible individual-target misses were:

| Method | Cell | Attained clusters |
|---|---|---:|
| DCS | `h005-barrier-p1e-04` | 61/64 |
| DCS | `h012-barrier-p1e-03` | 63/64 |
| DCS | `h012-terminal-p1e-03` | 63/64 |
| Raw | `h012-barrier-p1e-04` | 63/64 |

Across all records this is 5/1,152 individual misses for DCS and 1/1,152 for raw.
These are not discarded.  The pre-outcome decision rule was the method-cell
attainment-plus-RMSE co-gate, not an all-record Boolean AND.

## 7. Independent aggregate audit

The final auditor does not import either production mechanism analyzer or production
accuracy analyzer.  It independently:

- verifies freeze and config hashes;
- verifies source commit and clean-worktree identities;
- checks exact record counts;
- checks 2,304 probe seeds and 48,200 fixed-estimator seeds for duplicates and
  cross-family overlap;
- verifies every artifact link;
- recomputes five cluster-level effects and both floor counts;
- recomputes all 36 method-cell accuracy groups; and
- repeats all 50,000 bootstrap resamples with the frozen seed convention.

It passed with zero failures.  Its SHA-256 is
`06f5b68342b291c72458eb13ccd0db2ac911d98e75210452acc9c1553e5ae141`.

A separate post-confirmation phase-seed auditor also binds the qualification and
confirmation probe/fixed artifacts and checks all six pairwise intersections.  It
passed with counts 864, 18,058, 2,304, and 48,200 and zero intersection in every
pair.  Its SHA-256 is
`a535cf2d962f8628d78a76577a9f4a0f5f3ff91c2e2e2ad885f87473e115ed1b`.
It is an additional provenance audit, not a new outcome gate.

## 8. Artifact ledger

| Artifact | SHA-256 |
|---|---|
| Valid freeze V2 | `d6382db7d5e72bbc32e4f68b0349de01a5b5b72dda9bec7e561bcfaf457bb518` |
| Paired mechanism probe | `1d82e8b78763ad8bdeec99fb69ed1517054c5f800a0bdecb9b7c389b5bc1a619` |
| Independent paired-probe audit | `0b80da0c7210f2f43d9a266b819a366cb910043714b3c1e8256f7e32c7b2ab56` |
| Fixed estimators | `aeae0c890a0a863b7f240f06732dfb856e5b0c5c91e1c38b94e4c0d2b3ca9cc8` |
| Independent fixed-estimator audit | `9857f7d3e5d0d0676187ab6368de7e388614cb7372b60c7f3c82f02bb7f91b0d` |
| Resource supplement | `f6b1f79fba01056cd4f4f53bfecdd0d653b10941a039ec3d1c9ac90e94570c08` |
| Joint mechanism/work analysis | `c4dd42f85976eeb03d1fb73aa949886d3e4c56d3c473e4342c3ad05703af8997` |
| Simultaneous accuracy | `8b5a40422a36a6e32da17cc3d87a80bc483b2f84b01d364170e465ec501853ca` |
| Independent aggregate audit | `06f5b68342b291c72458eb13ccd0db2ac911d98e75210452acc9c1553e5ae141` |
| Cross-phase seed audit | `a535cf2d962f8628d78a76577a9f4a0f5f3ff91c2e2e2ad885f87473e115ed1b` |

## 9. Deviations and operational incidents

1. Freeze V1 at commit `1574964` was superseded before any outcome because the
   Python 3.10 CI job exposed a last-bit equality test issue.  The paired raw
   diagnostic was changed to use the exact production raw likelihood expression,
   and its test was strengthened to bitwise equality.  The supersession is recorded
   separately.
2. During execution monitoring, a text search counted nested audit-related keys and
   temporarily overstated progress.  The count was corrected using the exact byte
   token `b'"audit":'`.  This monitor read neither changed the process nor entered
   any estimator or decision.
3. The first fixed-estimator audit CLI call supplied the estimator config where the
   independent-audit config was required.  It failed before creating an audit
   artifact.  The unchanged source artifact was then audited with the frozen
   V6-compatible independent-audit config and passed.

None of these events changed a result, cell, threshold, sample allocation, seed,
proposal, or valid freeze V2.

## 10. What the result establishes

The confirmation supports the following finite-grid statement:

> For the frozen task-conditioned defensive proposal and 18-cell rBergomi regime,
> DCS is the exact conditional expectation of the raw contribution along the
> integrated Gaussian coordinate and produces a repeatable, practically large
> reduction in variance, achieved-RMSE work, and wall time.

Together with V6, there are now two separated results:

1. V6: reusing the proposal bank is advantageous relative to retraining pure CEM
   for each query; and
2. V7: even after holding that proposal fixed, DCS itself supplies a roughly
   3.5-fold variance reduction.

This is stronger than attributing the entire advantage to amortization.

## 11. Theoretical and technical boundary audit

### Exact within current scope

- exact balance-mixture likelihood;
- positive natural-component defensive bound;
- ordinary, non-self-normalized sample mean;
- finite-dimensional DCS conditional-expectation identity;
- Rao--Blackwell variance non-increase;
- exact finite-grid scalar-threshold representation for the implemented tasks;
- independent planning and final seeds; and
- bitwise identity of production raw and paired-probe raw code paths.

### Statistical qualifications

- the cluster-level t interval is a finite-64-cluster parametric inference, not an
  exact distribution-free interval;
- Clopper--Pearson attainment coverage is exact;
- the percentile bootstrap RMSE upper bound is nominal, not a finite-sample theorem;
  and
- the 4.5-z orthogonality checks are falsification diagnostics, not proofs.

### Not established

- a continuously monitored barrier estimand;
- a universal rough-Bergomi convergence or complexity theorem;
- uniform superiority over all parameter values or event definitions;
- superiority over every task-tuned adaptive IS, flow, or transport proposal;
- a successful hybrid router; or
- a neural-architecture contribution.

## 12. Publication-level conclusion and next gates

V7 is now a confirmed, mechanism-identified computational result rather than only a
development or qualification result.  Together with the exactness contract,
independent audits, V6 cross-platform evidence, and training-inclusive accounting,
it is a credible PhD-level paper core and a realistic specialized-journal
submission candidate.

It is still premature to call it top-journal complete.  The highest-value remaining
work is:

1. disjoint-seed V7 cross-platform reproduction;
2. matched task-tuned CEM, adaptive IS, and a strong flow/transport comparator;
3. external proof review;
4. a barrier mesh/weak-bias theorem or an explicitly bounded finite-grid scope;
5. refreshed reproducible novelty and literature audit; and
6. paper assembly with one primary claim and all negative V5/V6 findings preserved.

The confirmed result should be presented as a **mechanism-identified amortized DCS
path-measure estimator**, not as a quantum path integral or a neural model.
