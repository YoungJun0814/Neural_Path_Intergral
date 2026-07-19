# G11 Implementation and Error Audit

Date: 2026-07-19
Scope: M0--M6 development evidence
Decision: implementation core passes; M7 confirmatory freeze is not yet authorized

## 1. Outcome first

The finite-grid DCS-MGI-MLMC estimator is now implemented from the generic Gaussian
identity through an actual rBergomi MLMC sampler. The exactness, seed, checkpoint,
rate, rarity-calibration, and development gates described below pass. The current
evidence supports continuing toward a paper.

It does **not** yet support a top-journal performance claim. No confirmatory frozen
CPU run, independent GPU/environment reproduction, continuous-time bias theorem, or
multi-regime rare-efficiency study has been completed. The correct next state is M7
preparation, not manuscript submission.

## 2. Mathematical audit

| Claim | Status | Scope and qualification |
|---|---|---|
| T11-1 Gaussian-mixture marginalization | pass | finite-dimensional identity-covariance Gaussian shifts; ordinary expectation |
| T11-2 Rao--Blackwell identity under the mixture proposal | pass | proposal conditional density is derived; it is not replaced by the target conditional law |
| T11-3 defensive full and residual likelihood bound | pass | requires a positive-weight zero-mean component; pathwise bound is `1/delta` |
| T11-4 MLMC telescoping | pass | exact for a declared finest finite grid and normalized fine-to-coarse Gaussian map |
| T11-5 scalar threshold | pass | terminal, discrete barrier, and hit-plus-occupation tasks with strictly positive price direction |
| T11-6 rate upper bound | pass as an upper bound | `O(h^(2r))` follows from an `L2` threshold assumption; no exact exponent equality is claimed |
| T11-7 complexity corollary | conditional only | requires bias, correction variance, and cost exponents; it is not yet a rough-Bergomi uniform theorem |

The complete argument and proof-level cautions are in
[`G11_THEOREMS.md`](../theory/G11_THEOREMS.md) and
[`G11_PROOF_AUDIT.md`](../theory/G11_PROOF_AUDIT.md).

### Important interpretation boundary

The estimator targets `p_L`, the event probability on the fixed finest grid. It is
not an unbiased estimator of a continuously monitored barrier probability. The
empirical rate study concerns adjacent finite-grid corrections. A separate weak-bias
result is required before any continuous-time complexity statement.

## 3. Technical audit

### 3.1 Exactness and regression

- The generic Gaussian oracle passed 500 randomized factorization cases covering
  dimensions 2, 3, 7, and 32; mixture sizes 1, 2, and 5; and ranks 0, 1, and 2.
- All analytic mean, likelihood normalization, defensive-bound, reconstruction,
  invariance, and Rao--Blackwell gates passed.
- The generic rBergomi adapter is pathwise equal to the frozen G10 rank-one output
  within the declared float64 tolerances.
- Terminal, barrier, and hit-plus-occupation hard events equal their scalar-threshold
  representations pathwise.
- Reference and FFT simulator branches both satisfy the adapter contract.
- The old stable Gaussian functions now delegate to one audited implementation;
  G10 public APIs remain compatibility wrappers.

### 3.2 MLMC engine

- Fixed-finest-grid and continuous-target declarations are distinct types, preventing
  silent relabeling.
- Pilot and final samples use disjoint, hash-derived seed roles and explicit proposal
  and label streams.
- Integer allocation recomputes its achieved design variance.
- Checkpoint/resume is bitwise identical to uninterrupted deterministic execution in
  the analytic oracle.
- The Gaussian 95% interval audit used 1,000 repetitions and fell inside the
  predeclared 93--97% coverage gate.
- Rare raw corrections can have almost all-zero pilots. The engine therefore supports
  discarded pilot extension to a declared minimum nonzero count and a declared cap.
- Long development matrices write atomic cell-level progress and resume only when the
  config byte hash and run mode match.

### 3.3 Rate evidence

The full M4 study used three roughness regimes (`H=0.07, 0.12, 0.30`), three tasks,
12 independent seed clusters, 8,192 paths per level, and six adjacent levels. All
9 cells passed the common-window, bootstrap-CV, exactness, positive-threshold-rate,
and paired `beta_DCS` versus `2r` equivalence gates.

Observed second-moment exponent ranges were:

| H | threshold `L2` exponent | raw correction exponent | DCS correction exponent |
|---:|---:|---:|---:|
| 0.07 | 0.081--0.088 | 0.015--0.022 | 0.070--0.072 |
| 0.12 | 0.207--0.239 | 0.089--0.093 | 0.201--0.231 |
| 0.30 | 0.591--0.749 | 0.273--0.331 | 0.584--0.753 |

These results are compatible with the conditional theorem. They are development
evidence, not an untouched confirmatory estimate of a universal exponent.

### 3.4 Rare-event alignment

The first MLMC development tasks had probabilities near 0.25--0.28. They were useful
for engine validation but were not aligned with the research objective. A disjoint
calibration/validation protocol was therefore added.

For the primary `H=0.12` regime, terminal, barrier, and non-degenerate excursion tasks
were calibrated at `10^-3`, `10^-4`, `10^-5`, and `10^-6` on a fixed 128-step grid.
All 12 validation estimates were within the predeclared `[0.5p, 2p]` band, and the
largest relative standard error was about 3.3%. The excursion occupation constraint
removed 6.6--25.9% of the corresponding barrier mass, so it is not a relabeled
barrier-only task.

### 3.5 Rare MLMC development

The resource-controlled rare matrix had 12 paired cells: two defensive weights,
terminal/barrier/excursion tasks at `10^-3` and `10^-6`, and 20% relative RMSE.

- DCS reached the empirical sampling-variance target in 11/12 cells (91.7%), passing
  the M6 proposed-method attainment gate.
- Raw defensive MLMC reached it in only 3/12 cells (25%). Its pilot/final heavy-tail
  behavior makes several `10^-6` cells infeasible at the tested resource scale.
- In the three cells where both methods reached matched RMSE, the geometric
  operation-work ratio was 43.85x in favor of DCS.
- Across all allocated cells, including raw cells that missed the target, the
  geometric allocated-work ratio was 4.29x. This number is diagnostic only and must
  not be presented as a matched-RMSE speedup.
- Measured sampler wall time across saved cells was about 2,323 seconds. Because the
  run resumed across process limits, startup/orchestration time is not a single-run
  wall-clock benchmark. Operation work is the primary development comparison.

## 4. Errors found and corrected during implementation

1. **Duplicated Gaussian tail code:** consolidated into `stable_gaussian.py` with
   independent SciPy extreme-tail tests.
2. **Natural-only smoothing rejection:** the original price-span builder required a
   nonzero proposal shift. Natural smoothing now uses a fixed positive event direction
   and retains likelihood one.
3. **Missing aggregate seed hash:** cell-level MLMC hashes existed but the development
   artifact lacked a top-level manifest hash. A canonical aggregate hash was added.
4. **Degenerate excursion event:** the first rare configuration made occupation nearly
   automatic. Stress and occupation thresholds were strengthened and a direct
   exclusion-fraction gate was added.
5. **Rare raw pilot underestimation:** a small pilot could see no correction events and
   allocate far too few final paths. Independent pilot extension and explicit resource
   caps were added. Raw resource failures remain visible rather than being silently
   called successful estimates.
6. **Loss of long-run partial results:** the first large rare matrix hit the process
   time limit before writing output. Atomic per-cell progress and config-hash-checked
   resume were implemented and then exercised successfully.
7. **Ambiguous attainment summary:** averaging raw and DCS attainment incorrectly made
   the proposed-method M6 gate depend on baseline infeasibility. DCS attainment and
   raw feasibility are now reported separately; speedup uses only matched-target cells.

## 5. Remaining blockers before M7 freeze

1. Freeze the cross-roughness task scope correctly: high-H passed all rarity gates,
   while low-H `10^-6` excursion removed only 0.73% of matched barrier mass and failed
   the predeclared 1% non-degeneracy gate. Low-H terminal/barrier may remain, but its
   excursion cell must be excluded by protocol rather than silently relabeled.
2. Decide and freeze a raw-baseline resource-censoring rule before seeing confirmatory
   results. A failed raw target cannot be assigned a fabricated point speedup.
3. Commit the implementation, put the exact source commit into the frozen config, and
   create a completely untouched confirmatory seed namespace.
4. Record actual controller calibration cost if a learned or CEM-generated control is
   used. The current rare proposal is a fixed declared three-component schedule.
5. Run the frozen CPU study and then a second environment/GPU reproduction.
6. Add a continuous-time bias study or keep every headline explicitly finite-grid.
7. Obtain an independent mathematical review of T11-6/T11-7 and a final primary-source
   novelty review. The 2025 hierarchical occupation-time MLIS paper already covers
   common likelihood, preprocessing-inclusive work, and occupation-time MLIS; those
   are not novel claims here.

## 6. Audit commands and artifacts

- `python -m pytest -q`: final rerun passed 316/316 tests after the rare-pilot,
  checkpoint/resume, and attainment-summary edits.
- G11-targeted Ruff checks passed. Repository-wide Ruff still reports 49 pre-existing
  issues in legacy root scripts (`fix_viz.py`, `generate_concept_plots.py`, and
  `update_maxiter.py`); those unrelated files were not modified.
- The strict artifact audit validates config byte hashes, strict JSON, seed hashes,
  provenance, categorized work, input hashes, and boolean gates. Its latest result is
  `results/g11_artifact_audit_v1_2026-07-19.json`.

## 7. Decision

Continue. The estimator and rate mechanism are technically credible and now aligned
with genuinely rare events. Do not freeze or submit yet. The next legitimate action is
to close the M7 blockers above, then run an untouched confirmatory protocol.
