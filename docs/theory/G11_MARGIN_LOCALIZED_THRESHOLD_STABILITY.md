# G11 V4 Margin-Localized Threshold Stability and Crossover Theory

Date: 2026-07-22

Status: deterministic lemmas and conditional moment theorem implemented and tested;
model-level rough-Volterra rate remains an explicit proof obligation

## 1. Why this is the V4 theory

M7 V3 did not fail because the Gaussian marginalization identity was wrong. Its main
scientific weakness was concentrated in the low-H-labelled combined regime, barrier
events, and probability `1e-6`. DCS attained the target in 91.41% overall but only
78.13% in that combined regime, 87.08% for barriers, and 76.25% at `1e-6`. Some hard
matched cells favored raw rather than DCS-MLMC. Because M7 changed H, eta, and rho
together, it cannot attribute the regime effect to roughness alone.

The missing theorem is therefore not another generic likelihood identity. It is a
threshold-stability result that remains valid when early-time affine slopes approach
zero, active barrier indices change, or fine grids introduce new candidate times. The
result must also state when multilevel sampling is actually cheaper than the same DCS
estimator used at the finest level.

## 2. Setup

Conditional on the residual Gaussian coordinates `R`, write the adjacent events as

`F_f = 1{Z <= A_f(R)}` and `F_c = 1{Z <= A_c(R)}`,

where `Z` is standard normal under the target law. Let the proposal be a defensive
mixture with natural-component weight `delta > 0`, so both the full and residual
target-over-proposal likelihoods are bounded by `1/delta`.

For a monitored affine log-price candidate, write

`r_(ell,i) = n_(ell,i) / b_(ell,i)` with `b_(ell,i) > 0`.

For a downside barrier, `n_(ell,i)=log(B)-a_(ell,i)` and the threshold is a maximum
over monitoring indices. Terminal events have one candidate. Occupation events use an
order statistic and may change the required rank when the grid changes.

## 3. Lemma V4-1: ratio stability on a margin event

Fix `kappa > 0` and define the good event

`G_kappa = {min_i b_(f,i) >= kappa and min_i b_(c,i) >= kappa}`.

Let

- `D_n = sup_i |n_(f,i)-n_(c,i)|`,
- `D_b = sup_i |b_(f,i)-b_(c,i)|`, and
- `M_c = sup_i |n_(c,i)|`.

Then, pathwise on `G_kappa`,

`sup_i |r_(f,i)-r_(c,i)| <= D_n/kappa + M_c D_b/kappa^2`.

### Proof

For every common index,

`n_f/b_f - n_c/b_c = (n_f-n_c)/b_f + n_c(b_c-b_f)/(b_f b_c)`.

Take absolute values, use both denominator lower bounds, and then take the supremum.
No statement is made on `G_kappa^c`. QED.

This is deliberately a localized statement. A global deterministic lower bound on
all post-initial rBergomi price slopes is not assumed.

## 4. Lemma V4-2: common-grid aggregation and mesh enrichment

Let `A_f^o` be an intermediate threshold obtained by evaluating fine coefficients
with the coarse decision rule. Then

`|A_f-A_c| <= |A_f-A_f^o| + |A_f^o-A_c|`.

The first term is the **mesh, rank, or active-index enrichment defect** `D_mesh`. The
second is a common-grid coefficient error.

For maxima, minima, and a fixed-rank order statistic on arrays of equal size, the
aggregation map is 1-Lipschitz in the sup norm. Hence on `G_kappa`,

`|A_f-A_c| <= B_kappa`,

where

`B_kappa = D_n/kappa + M_c D_b/kappa^2 + D_mesh`.

For a maximum, an exact pathwise choice is

`D_mesh = max_i r_(f,i) - max_(i in embedded coarse indices) r_(f,i) >= 0`.

The analogous minimum defect reverses the extrema. For occupation thresholds whose
rank changes with the grid, `D_mesh` must include the rank-change defect. It is not
valid to silently apply a fixed-cardinality order-statistic lemma.

### Proof

Insert `A_f^o` and apply the triangle inequality. The common aggregation error is at
most the largest common candidate error. For maxima and minima, the displayed
enrichment defect is nonnegative and gives the exact remaining one-sided difference;
the reverse difference is already controlled by the common error. QED.

## 5. Theorem V4-3: defensive good/bad-event moment bounds

Let

`G(R) = Phi(A_f(R))-Phi(A_c(R))`.

Assume only that `|A_f-A_c| <= B_kappa` on `G_kappa`. Let
`phi_max=1/sqrt(2*pi)`. Then the DCS correction satisfies

`E_Q[H_DCS^2]`
`<= delta^-1 [phi_max^2 E_(P_R)(B_kappa^2 1_G) + P_R(G^c)]`.

The raw correction satisfies

`E_Q[H_raw^2]`
`<= delta^-1 [phi_max E_(P_R)(B_kappa 1_G) + P_R(G^c)]`.

### Proof for DCS

For the residual likelihood `Lbar`,

`E_Q[(Lbar G)^2] = E_(P_R)[Lbar G^2] <= delta^-1 E_(P_R)[G^2]`.

On the good event, normal-CDF Lipschitz continuity gives
`|G| <= phi_max B_kappa`. On the bad event, `|G| <= 1`. Split the expectation.
QED.

### Proof for raw

For `Delta=F_f-F_c`,

`E_Q[(L Delta)^2] = E_P[L Delta^2] <= delta^-1 E_P[Delta^2]`.

Conditional on `R`, `E[Delta^2|R]=|Phi(A_f)-Phi(A_c)|`. Use the good-event Lipschitz
bound and the unit bad-event bound. QED.

## 6. Conditional rate corollary

If for a hierarchy scale `h` and some `r>0`,

- `E[B_kappa^2 1_G] = O(h^(2r))`, and
- `P(G_kappa^c) = O(h^(2r))`,

then the DCS second moment is `O(h^(2r))`.

If additionally `E[B_kappa 1_G]=O(h^r)`, the raw second moment is `O(h^r)`; the
`O(h^(2r))` bad-event term is smaller.

This is an upper-bound result, not an equality of exponents. A model theorem must
derive the coefficient, mesh, and margin estimates rather than fit a regression and
call it a proof.

## 7. Task-specific consequences

### 7.1 Terminal

There is one candidate and no mesh-enrichment defect. The proof obligation reduces to
the terminal intercept error, terminal slope error, numerator moments, and a lower-tail
or inverse-moment bound for the terminal slope. This is the first model-level theorem
to attempt.

### 7.2 Discrete barrier

The maximum is Lipschitz, but fine grids add monitoring times and early slopes may be
small. A correct proof needs either:

- an active-time localization showing that very early candidates cannot maximize the
  threshold except on a controlled bad event; or
- a direct bound on the mesh-enrichment defect and the probability of small active
  slopes.

M7's 87.08% barrier attainment and 15/20 DCS misses in the low-H `1e-6` barrier group
make this the highest-priority theorem after terminal.

### 7.3 Hit plus occupation

The hit threshold is a maximum, while the occupation threshold is a grid-dependent
order statistic. The minimum combining the two is Lipschitz, but rank change and
monitoring-grid enrichment must remain explicit. Until those defects are controlled,
the occupation rate is conditional or empirical only.

## 8. Theorem V4-4: finite-level crossover

For independent level terms with variances `V_l` and per-sample costs `C_l`, the
continuous optimal-allocation online-work coefficient is

`K = (sum_l sqrt(V_l C_l))^2`.

For every possible start level `l0`, compare

`K_(l0) = (sqrt(V_(l0) C_(l0)) + sum_(l=l0+1)^L sqrt(V_Delta,l C_Delta,l))^2`

with the finest-level single estimator `K_L=V_L C_L`. Selecting the smallest value,
including `l0=L`, prevents MLMC from being used where it is intrinsically more
expensive.

For requested sampling variance `epsilon^2` and declared preprocessing work `P_l0`,
compare total work

`W_l0(epsilon) = P_l0 + K_l0/epsilon^2`.

Training, tuning, pilot, and setup work must be included exactly once. This finite-level
criterion is standard allocation theory and is not claimed as a novelty by itself.
The contribution candidate is its use with the margin-localized rough-Volterra
threshold analysis and exact defensive control-span marginalization.

## 9. Implementation traceability

| Mathematical object | Code | Test obligation |
|---|---|---|
| ratio good-event bound | `ratio_candidate_stability` | randomized/pathwise inequality |
| max/min enrichment split | `aggregate_threshold_stability` | exact fine/common/coarse examples |
| explicit order/mesh defect | `combine_common_and_mesh_defect` | nonnegative finite validation |
| defensive moment theorem | `defensive_moment_upper_bounds` | linear raw and quadratic DCS scaling |
| online crossover | `evaluate_multilevel_crossover` | MLMC and SLIS selection cases |
| preprocessing-inclusive crossover | `evaluate_total_work_crossover` | training cost reverses online choice |

## 10. Claim boundary

Implemented and proved:

- deterministic ratio stability on a denominator-margin good event;
- exact common-grid plus mesh-enrichment decomposition for max/min thresholds;
- defensive raw and DCS second-moment bounds retaining the bad-event probability;
- finite-level online and preprocessing-inclusive crossover calculations.

Not yet proved:

- a complete rBergomi coefficient and active-slope probability rate;
- a barrier mesh-enrichment rate uniform in rare threshold;
- an occupation rank-change rate;
- a continuous-monitoring weak-bias rate; or
- uniform DCS-MLMC superiority.
