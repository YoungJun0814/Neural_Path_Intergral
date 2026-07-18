# G10 Defensive Control-Span Marginalization: Final Audit

Date: 2026-07-19
Decision: **mathematical mechanism validated; primary practical-superiority hypothesis rejected**

## 1. Implemented model

G10 implements Defensive Control-Span Marginalized Gaussian Integration (DCS-MGI)
for the declared kappa=1 BLP rBergomi finite grid.

The proposal is a frozen randomized mixture of a natural expert and a deterministic
time-piecewise CEM expert. The complete price-driver mean-shift span is integrated
analytically. Only the volatility-driver marginal mixture likelihood remains:

`L_bar(Y) = p_Y(Y) / [alpha p_Y(Y) + (1-alpha) q_Y(Y)]`.

For the rank-one downside controls used here, the price shift is strictly one-signed.
The finite-grid hit-plus-occupation event is therefore a scalar Gaussian threshold,
and the estimator is exactly `L_bar Phi(a)`.

The adjacent-grid estimator uses one fine target coordinate for both grids and is
exactly `L_bar [Phi(a_f)-Phi(a_c)]`.

## 2. Theoretical and technical verdict

The implemented finite-grid estimator is internally consistent under its declared
assumptions:

1. all experts are deterministic time-only controls;
2. target, not proposal, price coordinates define the integrated span;
3. all price shifts replay inside the declared rank-one span;
4. full component and balance-mixture densities reconstruct pathwise;
5. hard events replay from the Gaussian threshold pathwise;
6. raw and marginalized estimates are ordinary, non-self-normalized means;
7. the natural component gives the pathwise bound `L_bar <= 1/alpha`;
8. fine and coarse corrections share one likelihood and one integrated coordinate.

The maximum frozen exactness error was `8.27e-14`; the maximum defensive-bound
violation was exactly zero. These checks support the finite-grid identities. They do
not prove continuous-time barrier exactness or a general rough-Volterra theorem.

## 3. Frozen protocol

The final untouched run used:

- 12 core and 6 stress regimes;
- natural weights selected on separate calibration paths and then frozen;
- 10 independent validation seeds with 2,000 paths per seed;
- single grid `N=512`;
- exact adjacent grids `N=32,64,128,256,512`;
- disjoint calibration, validation, single-label, and correction-label seed spaces;
- G9's independent 50,000-path references for target consistency;
- fixed-suite inference clustered by validation seed;
- online variance-times-cost, excluding calibration cost.

The result is stored in
`results/g10_control_span_frozen_v1_2026-07-19.json`.

An initial calibration/frozen artifact was invalidated before final reporting because
the integrity test found a cross-regime numerical seed collision between calibration
Brownian streams and calibration label streams. Calibration was rerun with labels in
the `40M` namespace, and the final untouched validation used `90M`, `100M`, and `110M`
namespaces. The invalid preliminary result was overwritten and is not research evidence.

## 4. Frozen result

| Metric | Result | Required | Decision |
|---|---:|---:|---|
| Raw defensive / DCS-MGI single work | 1.335x | >2.0x | fail |
| Fixed-suite one-sided 95% lower bound | 1.258x | >1.0x | pass |
| Regime-heterogeneity sensitivity lower bound | 1.199x | >1.0x | pass |
| Core regimes with geometric improvement | 12/12 | >=80% | pass |
| Core likelihood pass fraction | 12/12 | >=90% | pass |
| Raw / marginalized correction work | 2.395x | >1.5x | pass |
| Core and stress exactness | 18/18 | all | pass |
| Reference consistency | 18/18 | all | pass |

The primary headline failed solely because the predeclared `2x` single-level effect
size was not reached. Every core regime improved, so this is not a negative result in
the sense of estimator usefulness; it is a rejection of the stronger practical claim.

Seventeen regimes missed the `2x` single-level threshold; one stress regime passed all
strict gates. Seven regimes had a per-regime one-sided work lower bound at or below one,
and one regime missed its correction-work gate. All pooled correction mean checks and
all pathwise identities passed.

## 5. What G10 fixed relative to G9

G9 had likelihood-normalization failures in all 18 frozen regimes and improved only
7/12 core regimes. G10 achieved:

- bounded outer likelihood with zero pathwise bound violations;
- likelihood-normalization pass in 12/12 core regimes;
- geometric improvement in 12/12 core regimes;
- single-level work improvement from G9's `1.101x` to `1.335x`;
- correction work improvement from G9's `1.926x` to `2.395x`.

Thus control-span marginalization is a genuine structural improvement over an
arbitrary one-coordinate direction. It is not sufficient for the declared `2x`
end-to-end objective.

## 6. Why production optimization cannot rescue the headline alone

Across the frozen core seed cells, the geometric raw/marginalized variance-only gain was
approximately `1.92x`, while marginalized/raw online cost was approximately `1.44x`.
Even a hypothetical zero-overhead implementation would not reach the predeclared
`2x` geometric variance threshold on these frozen paths. Selection algorithms,
kernel fusion, and removal of audit-only replay are useful engineering, but cannot
create the missing conditional variance reduction.

The reported `1.335x` is online-only. G10-specific alpha calibration is an additional
nonnegative cost, so a finite-budget training-inclusive ratio cannot be larger without
an explicit amortization model. No training-inclusive speedup is claimed.

## 7. Rank-two extension result

A mathematically valid rank-two extension was implemented to test whether one extra
positive event direction closes the gap. It uses an oblique positive basis and an
unbiased randomized stratified-normal inner rule. Deterministic quadrature bias is not
introduced.

On four representative development regimes, the best selected rank-two variants had:

- geometric raw/rank-two work ratio `0.916x`;
- geometric rank-one/rank-two work ratio `0.634x`;
- improved-regime fraction `0/4`;
- all exactness and paired-mean admissibility checks passing.

The additional conditional variance reduction was smaller than the inner integration
cost. Per the V10 stop rule, rank two is stopped and must not be run on confirmatory
seeds merely to search for a favorable result.

## 8. Claim decisions

Supported:

- exact finite-grid control-span marginalization for deterministic mixtures;
- exact cancellation of the complete price-driver likelihood span;
- pathwise defensive marginal-likelihood bound;
- exact adjacent BLP correction marginalization;
- robust frozen likelihood behavior and broad positive single-level improvement;
- greater than `2x` frozen correction-work improvement.

Prohibited:

- “DCS-MGI is at least twice as efficient end to end”;
- “the current estimator is training-inclusive superior”;
- “rank-two marginalization improves practical work”;
- “a continuous-time or general rough-Volterra complexity theorem is proved”;
- “the current result is ready for a top mathematical-finance journal.”

## 9. Publication decision

The current work is stronger than G9 and can support a serious numerical-method
manuscript after formal proofs and independent reproduction. It is not yet a top-tier
main-model paper because the primary practical gate failed and the novelty currently
rests on one finite-grid rBergomi specialization.

The most defensible next paper is a **correction-focused DCS-MGI paper**:

1. prove the control-span marginalization and defensive-bound theorems on a general
   Gaussian Volterra discretization;
2. prove conditions for adjacent correction variance decay;
3. separate online and amortized calibration complexity;
4. reproduce the frozen experiment independently on CPU and GPU;
5. add barrier-only and alternative path-functional tasks;
6. compare against published numerical-smoothing and modern MIS baselines.

If those theorems establish canonical MLMC complexity and the frozen correction gain
survives independent reproduction, SIAM Journal on Financial Mathematics or a strong
computational-finance journal becomes realistic. A top mathematical-finance journal
would additionally require a broader rough-Volterra theorem and a substantially
stronger conceptual contribution.

## 10. Final research decision

Keep rank-one DCS-MGI as the production-safe estimator and as a correction-level
research contribution. Stop rank two. Do not claim the failed `2x` main-model
hypothesis. Any attempt to condition on volatility-driver randomness, solve a PPDE,
or introduce a learned transport is a new research scope and must receive a new
theory/falsification contract rather than being appended to G10 after inspection.
