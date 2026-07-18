# G9 MGVS Frozen Falsification Report

Date: 2026-07-18
Decision: **primary end-to-end superiority hypothesis rejected**

## 1. What was frozen

The confirmatory protocol was fixed before the final run and used:

- 12 core and 6 stress rBergomi/task regimes;
- separately calibrated four-segment deterministic CEM controls;
- calibration-selected, strictly positive exponential time directions;
- levels `N = 32, 64, 128, 256, 512`;
- 50,000 independent reference paths per regime;
- 2,000 paths for each of 10 independent validation seeds;
- CEM-calibration, direction-calibration, reference, and validation simulation seeds
  mutually disjoint;
- paired bootstrap intervals and per-path online-work timing;
- exact target-over-proposal likelihoods without self-normalization.

The final result is stored in
`results/g9_mgvs_frozen_v3_2026-07-18.json`. Individual regime failures were not
removed from the aggregate.

An earlier v2 artifact was invalidated before interpretation because its numerical
seed audit found overlap between direction calibration and reference generation.
No method, gate, or regime was changed: v3 only moved reference, validation, and
bootstrap seeds into disjoint `50xxx`, `60xxx`, and `70xxx` namespaces. The invalid
v2 artifact is not retained as research evidence.

## 2. Aggregate result

| Metric | Frozen result | Required | Decision |
|---|---:|---:|---|
| Raw-MLMC / smoothed-MLMC work | 1.926x | >1.5x | pass |
| Raw / smoothed single-level work | 1.101x | >2.0x | fail |
| Fixed-suite, seed-clustered one-sided 95% lower bound | 1.025x | >1.0x | pass |
| Core regimes with geometric single-level improvement | 7/12 | >=80% | fail |
| Exactness gates | all | all | pass |
| Stress exactness gates | all | all | pass |

The aggregate protocol therefore failed. For the fixed 12-regime suite, the small
positive single-level gain is distinguishable from no gain under seed-clustered
inference, but it is far below the predeclared practical-effect threshold. A
regime-heterogeneity sensitivity interval has lower bound `0.990x`; consequently no
population-wide superiority claim is supported.

The first generated summary used all 120 regime-by-seed cells as though independent,
giving a `1.044x` lower bound. A post-run audit identified that pseudoreplication. The
stored v3 summary and executable aggregation now cluster the fixed-suite interval by
the 10 independent validation seeds (`1.025x`) and retain `1.044x` only as an
explicitly superseded diagnostic. This correction does not use new simulations and
does not change the already-failed headline decision.

The reported work ratios are online variance-times-cost coefficients. CEM calibration
time is recorded in the calibration artifacts but was not added to either method, and
the direction-search runtime was not recorded. Because raw CEM and MGVS share the same
frozen CEM control while direction search is an additional nonnegative MGVS cost, the
`1.101x` online ratio is an upper bound on the corresponding finite-budget,
training-inclusive ratio. Thus including calibration cannot rescue the `2x` failure,
but an exact training-inclusive speedup must not be quoted from this experiment.

## 3. Regime-level diagnosis

No regime passed every strict gate because all 18 missed the `2x` single-level
total-work requirement. Other failures were:

- likelihood normalization diagnostic: 18/18 regimes;
- at least 8/10 improving single-level seeds: 14/18 regimes;
- pooled Rao--Blackwell finite-sample upper bound: 9/18 regimes;
- correction-work threshold: 3/18 regimes.

The likelihood failures do not invalidate the analytic Girsanov identity. They show
that strong single-Gaussian CEM proposals have likelihood tails that are inadequately
resolved by the declared finite validation budget. This undermines reliable practical
confidence intervals even when conditional-mean consistency passes.

## 4. What did work

The failed headline does not by itself negate the exact finite-grid identities. The
following independently checked component evidence remains valid:

1. Event/threshold and likelihood reconstruction errors remained below the exactness
   threshold in every core and stress run.
2. Reference consistency and coverage were stable.
3. MGVS improved raw-MLMC versus smoothed-MLMC work by 1.926x geometrically.
4. Smoothed correction variance had a positive fitted level slope in 18/18 regimes.
5. The mean/median fitted smoothed correction exponent was `0.308/0.286`, versus
   `0.035/-0.009` raw; smoothing improved the exponent in 16/18 regimes.
6. FFT BLP replay error remained about `1e-13`, with a previously frozen 1024-step
   reference-loop speedup of 108.4x.

These are useful component results, but they do not establish an end-to-end superior
rare-event pricer.

## 5. Why the primary hypothesis failed

### 5.1 One integrated coordinate is not enough

MGVS removes the conditional variance associated with one positive direction in the
independent price Brownian path. Substantial residual randomness remains in the
`N-1` dimensional orthogonal price subspace and in the entire augmented volatility
driver. Calibration-selected time weights improved some regimes but did not change
this dimensional limitation.

### 5.2 Exact smoothing has non-negligible finite-grid cost

The occupation event requires a pathwise order statistic, while likelihood
decomposition and affine thresholds require additional path tensors. The FFT engine
makes the underlying finest-grid simulation so fast that this post-processing cost
offsets much of the variance reduction. A faster implementation may improve the
1.10x ratio, but cannot by itself create the missing conditional variance reduction.

### 5.3 The correction rate is better but still too low

The smoothed exponent improved on average, but `beta` was typically below the measured
FFT cost exponent. The present evidence does not support a canonical MLMC complexity
claim for the hit-plus-occupation target.

### 5.4 The single CEM proposal is fragile in the tails

Large likelihood-normalization z-scores show that a single aggressive mean shift is
not a robust proposal across the full regime set. A defensive mixture would help tail
coverage, but its mixture likelihood depends nonlinearly on the integrated coordinate;
the current closed-form `Phi(a+b)` formula would no longer apply directly.

## 6. Claim decisions

The following claims are prohibited:

- “MGVS is at least twice as efficient as frozen CEM in practice.”
- “MGVS robustly improves total work across rough-volatility regimes.”
- “The hit-plus-occupation MLMC correction has a proven favorable complexity rate.”
- “The frozen G9 experiment passed.”

The following narrower claims remain supported:

- exact finite-grid monotone Gaussian smoothing under deterministic time controls;
- population Rao--Blackwell variance nonincrease;
- exact adjacent BLP correction smoothing;
- pathwise-equivalent FFT BLP convolution;
- empirically improved MLMC correction work and variance slope on the frozen suite.

## 7. Publication decision

MGVS in its current one-coordinate form should **not** be submitted as the main model
for a top journal. A narrower computational note could be possible after a rate theorem
and independent reproduction, but that would not meet the project's original
end-to-end practical objective.

Per the falsification-first plan, the optional neural conditional-mean control variate
must not be attached merely to rescue a failed headline. The current M5 program should
not claim the predeclared success path.

## 8. Scientifically defensible next models

Ranked by alignment with the observed failure:

1. **Defensive-mixture subspace conditional integration.** Use a natural/CEM defensive
   mixture for likelihood-tail stability and integrate a low-dimensional positive
   Gaussian subspace numerically. This directly addresses both the likelihood and
   one-coordinate limitations, but exact unbiased quadrature/error control must be
   developed.
2. **Multi-coordinate monotone Gaussian polytope smoothing.** Integrate two to four
   positive price directions and compute the resulting Gaussian half-space/order-
   statistic probability. This targets the residual price-subspace variance, with
   computational geometry as the main technical risk.
3. **Correction-only MGVS paper.** Retain the current method as an MLMC correction
   regularizer and pursue `beta`/complexity theorems. This has lower practical ambition
   but the strongest evidence already available.
4. **Production optimization only.** Replace sorted top-k with selection algorithms and
   remove audit-only replay work. This is worthwhile engineering but is unlikely to
   close a predeclared 2x gap by itself.

Selecting option 1 or 2 is a new research scope and must be explicitly authorized;
neither should be silently presented as the already-frozen MGVS method.
