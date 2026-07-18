# Control-Span Marginalized Gaussian Integration Plan V10

Date: 2026-07-18
Status: executed; primary single-level hypothesis rejected, correction hypothesis passed

Final decision: see `G10_CONTROL_SPAN_FALSIFICATION_REPORT_2026-07-19.md`.

## 1. Decision

G9 established exact one-coordinate Gaussian smoothing but rejected its primary
end-to-end superiority hypothesis. V10 will not add an unconstrained neural module to
that failed estimator. It will test a new mathematical estimator:

**Defensive Control-Span Marginalized Gaussian Integration (DCS-MGI).**

The first implementation is the exact rank-one case. It analytically integrates the
entire deterministic price-driver control span of a natural/CEM defensive mixture.
This is materially different from G9, where the smoothing direction was selected for
variance and generally left a nonzero orthogonal price likelihood.

The rank-two or higher event subspace is conditional on rank-one success and is not a
rescue mechanism that may be silently attached after validation.

## 2. Problems inherited from G9

The frozen G9 result found:

1. raw/smoothed online single-level work ratio `1.101x`, below the `2x` gate;
2. only `7/12` core regimes improved geometrically;
3. likelihood-normalization diagnostics failed in `18/18` regimes;
4. one integrated coordinate left `N-1` price dimensions and all volatility-driver
   randomness;
5. direction-search cost was not included, so the online result is an upper bound on
   finite-budget training-inclusive performance.

V10 directly targets items 3 and part of 4. It must not claim in advance that this is
sufficient for item 1.

## 3. Frozen mathematical convention

Work on the declared kappa=1 BLP rBergomi finite grid. Let the standardized target
price-driver normal be `X in R^N`. For deterministic expert `j`, write its standardized
price shift as

`m_j = sqrt(h) (u_{j,2,0}, ..., u_{j,2,N-1})`.

Let `Y` denote the target first-driver path together with its induced augmented BLP
local variables. The proposal is the exact randomized mixture

`Q_mix = sum_j pi_j Q_j`, with `pi_j > 0` and `sum_j pi_j = 1`.

The production defensive mixture contains a natural expert `u_0 = 0` of mass
`alpha = pi_0` and a frozen deterministic time-only CEM expert. Self-normalization is
prohibited.

## 4. T10-1: price-control-span factorization

Assume all price shifts lie in a deterministic subspace `S` and let `U` have
orthonormal columns spanning `S`. Decompose

`X = U Z + R`, where `Z = U^T X` and `R = (I-UU^T)X`.

Under the target law, `Z` is standard Gaussian and independent of `(Y,R)`. Because
`m_j in S`, the residual price shift is zero for every expert. Therefore the marginal
proposal density of `(Y,R)` is

`q_bar(y,r) = p_R(r) sum_j pi_j q_{j,Y}(y)`.

The exact marginal target-over-proposal likelihood is

`L_bar(y) = p_Y(y) / sum_j pi_j q_{j,Y}(y)`.

All price-driver likelihood factors disappear after marginalizing `Z`; no mixture
likelihood remains inside the Gaussian event integral.

## 5. T10-2: exact marginalized estimator

For the declared finite-grid event `A`, define

`g(Y,R) = P_P(A | Y,R)`.

Then

`E_Qmix[L_bar(Y) g(Y,R)] = P_P(A)`.

This identity follows by integrating the full balance-mixture estimator over `Z`.
It is exact at the population level and does not require self-normalization, learned
surrogates, or approximate quadrature in the rank-one implementation.

For the natural/CEM mixture, the natural price shift is zero and the CEM price shift
is rank one. If the CEM price schedule is strictly negative, choose

`q = -m_CEM / ||m_CEM||`, so every `q_i > 0`.

Conditional log spots are then

`log S_k(Z) = A_k + B_k Z`, with `B_k > 0`,

and the hit-plus-occupation event is exactly `Z <= a(Y,R)`. Hence

`g(Y,R) = Phi(a(Y,R))`

and the V10 contribution is `L_bar Phi(a)`.

## 6. T10-3: defensive likelihood bound

If the natural component has mixture mass `alpha > 0`, then

`q_bar / p_bar = alpha + sum_{j>0} pi_j q_{j,Y}/p_Y >= alpha`.

Consequently

`0 < L_bar <= 1/alpha` pathwise.

This is a structural tail guarantee absent from the single aggressive CEM proposal.
It does not imply that the event estimator has low variance; `Phi(a)` can still be
rare and the volatility-driver mixture can still be poorly located.

## 7. T10-4: Rao--Blackwell relation

Let `H = 1_A L_mix` be the ordinary full defensive-mixture contribution. The V10
contribution equals `E_Qmix[H | Y,R]`. Therefore

`Var(V10) <= Var(H)`

and both means are identical. Equality can occur, so wall-clock superiority is not a
theorem.

The valid baseline for this theorem is the raw estimator under the same defensive
mixture, weights, paths, finite grid, and frozen controls. A comparison against raw
single CEM is a practical benchmark, not the Rao--Blackwell theorem baseline.

## 8. T10-5: adjacent-grid correction

On an exact BLP fine/coarse coupling, use the same fine-grid target price coordinate
and the same fine control-span direction for both levels. The fine and coarse events
are `Z <= a_f` and `Z <= a_c`, so the exact marginalized correction is

`L_bar [Phi(a_f) - Phi(a_c)]`.

The likelihood is the fine path's exact marginal mixture likelihood. Independent
fine and coarse likelihoods are prohibited because they break telescoping.

## 9. Non-negotiable implementation checks

1. Every expert must declare deterministic time-only control.
2. All price shifts must lie in the declared span to numerical tolerance.
3. Rank-one production requires a strictly one-signed nonzero price shift so the
   oriented direction is strictly positive.
4. Target Brownian increments, not proposal innovations, define the integrated
   coordinate and residual.
5. The all-expert full likelihood must replay from target coordinates.
6. Reconstructed full component and mixture log densities must agree pathwise.
7. The hard event must equal the scalar-threshold event pathwise.
8. Natural-component weight must be strictly positive for the `1/alpha` claim.
9. Raw and marginalized estimators remain unnormalized sample means.
10. Only finite-grid exactness is claimed.

## 10. Implementation milestones

### M0: theorem oracle

- implement rank-one span extraction and rejection tests;
- implement outer marginal mixture likelihood;
- implement full-density reconstruction diagnostic;
- test a low-dimensional Gaussian oracle against direct quadrature/large Monte Carlo;
- test the `1/alpha` likelihood bound.

Pass: pathwise replay below `1e-11`, hard-event replay exact, paired mean difference
within three standard errors, and no normalization failure in the oracle budget.

### M1: rBergomi single-grid estimator

- add FFT-capable deterministic mixture simulation;
- evaluate raw full-mixture and exact DCS-MGI contributions on identical paths;
- expose variance, kurtosis, likelihood, and timing diagnostics;
- reject mixed-sign or non-collinear price controls.

### M2: exact adjacent-grid estimator

- add mixture sampling to the exact FFT BLP adjacent coupling;
- use one fine coordinate for both grids;
- verify correction mean consistency and telescoping against single-grid references.

### M3: development-only selection

On development seeds only, select `alpha` from a fixed set such as
`{0.05, 0.10, 0.20, 0.50}`. Selection objective is online work with hard constraints:

- outer likelihood-normalization `|z| <= 4`;
- maximum marginal likelihood no greater than `1/alpha + 1e-12`;
- exactness error `<=1e-11`;
- no self-normalization;
- minimum natural-component realized count declared in advance.

### M4: untouched multi-regime falsification

Use disjoint calibration, reference, validation, and bootstrap seed namespaces.
Cluster fixed-suite inference by validation seed. Preserve regime-heterogeneity
sensitivity and individual failures.

Primary gates:

1. geometric raw-defensive/DCS-MGI online work ratio `>2x`;
2. fixed-suite one-sided 95% lower bound `>1x`;
3. geometric improvement in at least `80%` of core regimes;
4. all exactness and target consistency gates pass;
5. likelihood normalization passes in at least `90%` of core regimes;
6. correction work ratio `>1.5x` if the adjacent estimator reaches M2.

Training-inclusive reporting must separately add CEM and mixture-weight calibration,
and must include any V10-only selection cost.

## 11. Stop rules

Stop rank-one V10 as a main model if any occurs:

1. the control-span theorem fails a pathwise oracle;
2. the defensive bound is violated beyond floating-point tolerance;
3. development online work ratio is below `1.25x` after implementation profiling;
4. likelihood diagnostics remain unstable despite the natural component;
5. frozen M4 misses either the `2x` or `80%` primary gate.

Only after M1 exactness passes may rank-two geometry be considered. Only after rank-one
fails specifically from residual conditional variance, while likelihood stability and
work are adequate, may rank-two be authorized.

## 12. Rank-two extension boundary

Rank two adds an event-oriented direction to the price-control direction. Orthogonal
bases cannot both be strictly positive, so the scalar threshold formula does not
extend by assertion. A valid rank-two solver must use either:

1. a rigorously transformed correlated positive basis and one-dimensional conditional
   integration; or
2. an unbiased randomized inner estimator with its variance and cost included.

Deterministic quadrature without an a priori or a posteriori certified error bound is
not an exact estimator and may not be labeled bias-free.

## 13. Novelty and literature boundary

Conditional/numerical smoothing of discontinuous MLMC payoffs already exists; see
[Bayer, Ben Hammouda, and Tempone](https://arxiv.org/abs/2003.05708). Smoothing plus
sparse-grid/QMC methods also exists; see
[Bayer, Ben Hammouda, and Tempone](https://arxiv.org/abs/2111.01874). Multiple
importance sampling and balance-mixture denominators are established methods; see the
[multiple-importance-sampling review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8871238/).

The defensible candidate contribution is the exact elimination of the complete
price-control span from a defensive mixture in an augmented rough-Volterra simulator,
its bounded marginal likelihood, and the exact adjacent-grid hit-plus-occupation
specialization. Novelty must still be confirmed by a systematic expert literature
review before submission.

## 14. Publication threshold

Rank-one implementation alone is not a top-journal paper. A serious submission needs:

1. T10-1--T10-5 formal proofs with assumptions and measurable-space details;
2. frozen multi-regime results against raw CEM, raw defensive mixture, G9 MGVS, and a
   modern numerical-smoothing baseline;
3. probabilities spanning at least `1e-4` to `1e-6` with reliable references;
4. exact online and training-inclusive cost accounting;
5. CPU/GPU profiling and independent reproduction;
6. either a rough-Volterra class theorem or a validated rank-two extension showing
   that the method is more than a single rBergomi specialization.

The target-journal claim remains conditional on these results. No top-journal
probability is asserted before M4 evidence and an external novelty review.
