# G11 V4 Crossover Qualification Decision

Date: 2026-07-22

Protocol: `g11-v4-crossover-qualification-v1`

Frozen source: `b97f64cd4ed95ef5f550cba1f94148f0f0cf70d2`

Decision class: post-M7 qualification, not untouched confirmation

## 1. Decision

The frozen V4 qualification **passes every predeclared qualification gate**. The
artifact contains all 27 cells and 135 seed-runs, represents base, H, eta, and rho
one-factor regimes, evaluates three relative-RMSE targets, and places all 135
full-hierarchy DCS estimates within four combined standard errors of the independent
calibration references. The largest absolute reference discrepancy is 2.466 combined
standard errors.

This is a positive design decision, not a paper-level performance confirmation. The
profiles predict continuous-allocation work; they do not execute an allocation that
demonstrates the requested RMSE. CEM is present only in the four base cells, the
estimand is a fixed 128-step grid, and the artifact was produced on one laptop
environment.

## 2. Frozen-result integrity

| Item | Result |
|---|---:|
| Frozen source clean | yes |
| Result SHA-256 | `eedc353d325e5f7396c0112e16847867e8eaee370ea7fe1657a1f6aa2c5f1ba7` |
| Cells / independent seed-runs | 27 / 135 |
| Profile paths per method and level | 8,192 |
| Relative-RMSE targets | 10%, 20%, 30% |
| Reference comparisons within 4 SE | 135 / 135 |
| Median / maximum absolute reference z | 0.636 / 2.466 |
| Serialized qualification gate | pass |
| Independent reconstruction | pass |

The independent audit reconstructs the expected cell matrix, every deterministic
seed-ledger hash, profile standard error, telescoping estimate and standard error,
continuous-allocation coefficient, preprocessing-inclusive total work, optimal start
level, final comparator decision, summary, and gate. It also validates the frozen tag,
input manifests, normalized input hashes, finite-grid claim, and `torch.float64`
evidence dtype.

## 3. Which construction won

For each seed-run, start level 4 means finest-grid DCS single-level importance
sampling (DCS-SLIS); start level 0 means the full five-level DCS-MLMC hierarchy.

| Optimal DCS start | Meaning | Count / 135 | Fraction |
|---:|---|---:|---:|
| 0 | full MLMC | 13 | 9.63% |
| 1 | truncated MLMC | 4 | 2.96% |
| 2 | truncated MLMC | 4 | 2.96% |
| 3 | two-level construction | 24 | 17.78% |
| 4 | DCS-SLIS | 90 | 66.67% |

The same start counts occur at 10%, 20%, and 30% relative RMSE because the DCS
profiling cost is shared by all starts and the online coefficient ordering is
unchanged by a common variance target. This invariance is a consequence of this
qualification design, not a theorem that crossover can never depend on RMSE.

Relative to selecting DCS-SLIS unconditionally, the geometric total-work reduction
from the adaptive hybrid is modest: DCS-SLIS costs 1.060x, 1.026x, and 1.014x the
hybrid at 10%, 20%, and 30% relative RMSE, respectively. Thus a claim that MLMC is
uniformly necessary is false, but discarding MLMC is also wrong.

The correct V4 model is therefore:

> **Margin-aware Hybrid DCS-MGI:** profile DCS single levels and adjacent corrections,
> include the finest single level as a legal endpoint, and choose the
> preprocessing-inclusive minimum-work start level before the final frozen run.

## 4. Parameter-separated finding

At the 20% relative-RMSE diagnostic, the one-factor H=0.30 regime accounts for all 13
full-MLMC selections and three of the four start-1 selections. Its 20 seed-runs select
starts `{0: 13, 1: 4, 2: 2, 4: 1}`. Every other OAT regime selects primarily start 4,
with occasional start 2 or 3.

This supports the mechanistic hypothesis that smoother paths can make deeper
multilevel correction worthwhile. It does not establish a causal or uniform H
theorem: only two H perturbations were tested, finite-profile uncertainty remains,
and event type and rarity interact with the choice. Importantly, unlike M7, eta and
rho were held fixed when H changed.

## 5. Strong-baseline interpretation

The CEM proposal converged in two iterations for the `1e-4` base tasks and three
iterations for the `1e-6` base tasks. Its training work is included. At 20% relative
RMSE, the geometric CEM-total-work / best-DCS-total-work ratios are:

| Base task | CEM / best DCS |
|---|---:|
| terminal `1e-4` | 1.472x |
| terminal `1e-6` | 1.894x |
| barrier `1e-4` | 1.470x |
| barrier `1e-6` | 1.813x |

These four comparisons favor the hybrid DCS construction. They do not authorize a
claim against CEM elsewhere in the OAT matrix. The crude finest-grid estimator is
available in only 41 of 135 seed-runs because an 8,192-path profile often observes no
event. Such a zero empirical variance is treated as unavailable, never as zero-cost
superiority. DCS is the selected family in all 135 runs at all three RMSE targets,
subject to the profile-prediction limitation.

## 6. Theory chosen after M7 and checked against V4

The selected theory is **margin-localized scalar-threshold stability plus a
finite-level SLIS/MLMC crossover rule**.

It is the best fit to the evidence because it does not assume a global positive
denominator or uniform MLMC superiority. On a good event with active slopes at least
`kappa`, the threshold error is bounded by numerator error, denominator error, and an
explicit mesh/rank-enrichment defect. The complement probability remains in the
moment bound. Defensive likelihood control then makes the DCS second-moment term
quadratic in the good-event threshold defect, while the raw term is linear. Finally,
the finite-level rule compares every possible MLMC start with DCS-SLIS including
preprocessing work.

The deterministic ratio, aggregation, moment, and crossover statements are proved,
implemented, and tested. The following remain conditional proof obligations:

- rBergomi coefficient-error and small-active-slope rates;
- barrier mesh-enrichment rates uniform enough for the claimed rare-event range;
- occupation rank-change rates; and
- continuous-monitoring weak bias.

V4's H=0.30 crossover is consistent with the finite-level rule. It is not empirical
proof of the conditional asymptotic rate theorem.

## 7. Technical and theoretical audit verdict

No known exactness error was found in the implemented V4 scope:

- likelihoods remain ordinary exact balance-mixture likelihoods;
- estimates are not self-normalized;
- CEM training and evaluation seeds are separated;
- fine/coarse correction terms use declared coupled construction;
- profile and training work are included exactly once in the crossover comparison;
- a zero-hit baseline cannot win with zero reported cost;
- the OAT matrix changes one parameter at a time;
- finite-grid and qualification labels are serialized and audited; and
- resume output is protected by retrying durable atomic replacement.

The principal risk is no longer a discovered algebraic or implementation error. It
is external validity: finite profiles, only five replicates, a single machine, partial
CEM coverage, and no achieved-RMSE allocation or continuous-time bias budget.

## 8. Publication decision and next freeze

The result raises the project from an undifferentiated DCS-MLMC prototype to a
falsifiable adaptive estimator design. It is a strong PhD-level working-paper core,
but it is not yet sufficient for a top journal.

The next untouched protocol must be frozen only after this report and audit are
committed. It should:

1. use the hybrid start-level chooser trained only on separate pilot seeds;
2. execute actual allocations at two or more RMSE targets;
3. use 10--20 independent seed clusters per headline cell;
4. train or freeze a strong proposal baseline for every headline cell;
5. retain a fixed-resource fallback when crude MC has zero pilot hits;
6. report both operation work and hardware wall time;
7. reproduce the principal table on an independent machine; and
8. either add a continuous-time weak-bias budget or state finite-grid scope in the
   title, theorem, abstract, and experiments.

Until those conditions are satisfied, the allowed headline is that V4 qualification
supports an adaptive DCS-SLIS/MLMC construction and falsifies uniform preference for
either endpoint. It is not an achieved-RMSE speedup theorem or a final confirmatory
performance claim.
