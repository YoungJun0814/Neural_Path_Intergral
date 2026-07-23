# G11 V6 Terminal Slope Negative-Moment Theorem

Date: 2026-07-23

Status: the inverse-slope bound is proved for the implemented target-law terminal
slope and positive deterministic rank-one direction. The separate terminal coefficient
document is a conservative proof candidate, not yet an independently reviewed journal
theorem; every barrier rate remains open.

## 1. Object covered by the theorem

On an `n`-step grid with `Delta t=T/n`, the implemented scalar marginalization uses a
strictly positive Euclidean-unit direction `u^(n)=(u_0,...,u_(n-1))` in the independent
price driver. Conditional on the volatility driver, the terminal log spot has affine
slope

`B_n = sqrt(1-rho^2) sqrt(Delta t) sum_i u_i sqrt(V_i)`.

Under the target rBergomi law used by the simulator,

`V_i = xi exp(eta Y_i - eta^2 Var(Y_i)/2)`,

where the vector `Y=(Y_i)` is centered Gaussian. Define the grid-scaled direction
mass

`c_n = sqrt(Delta t) sum_i u_i`.

The theorem requires `c_n >= c_* > 0` over the hierarchy for which a uniform claim is
made. Positivity and unit normalization alone are not enough: a direction concentrated
on a fixed number of coordinates can have `c_n -> 0`.

## 2. Discrete BLP variance lemma

For every implemented grid point `t_i <= T`,

`Var(Y_i) <= t_i^(2H) <= T^(2H)`.

Proof. The most recent singular-kernel cell is integrated exactly. On every earlier
cell, the BLP scheme replaces the Volterra kernel by its cell average. If `I` is one
such cell, Jensen gives

`|I| (average_I K)^2 <= integral_I K(s)^2 ds`.

The cells are driven by independent Brownian increments, so their variances add.
Including the exact local cell and the `sqrt(2H)` normalization bounds the discrete
variance by the exact kernel integral

`2H integral_0^t (t-s)^(2H-1) ds = t^(2H)`.

This lemma covers both the reference and FFT implementations because they use the
same local covariance and historical cell-average coefficients.

## 3. Theorem: all inverse terminal-slope moments are finite

For every `q>0`,

`E_P[B_n^(-q)]`

is bounded above by

`[sqrt(1-rho^2) sqrt(xi) c_n]^(-q)`

times

`exp(eta^2 T^(2H) (q/4 + q^2/8))`.

Consequently, if `inf_n c_n >= c_*>0`, the same expression with `c_*` is a finite
uniform-in-grid bound.

### Proof

Let `A_n=sum_i u_i` and `a_i=u_i/A_n`. The `a_i` are positive and sum to one. Write

`X_i = eta Y_i/2 - eta^2 Var(Y_i)/4`,

so `sqrt(V_i)=sqrt(xi) exp(X_i)`. Convexity of the exponential gives

`sum_i a_i exp(X_i) >= exp(sum_i a_i X_i)`.

Therefore

`B_n >= sqrt(1-rho^2) sqrt(xi) c_n exp(sum_i a_i X_i)`.

After raising to power `-q` and taking expectations, only a Gaussian exponential
moment remains. Put `G_n=sum_i a_i Y_i`. Since the weights are positive and sum to
one, Cauchy--Schwarz for Gaussian covariances and the variance lemma imply

`Var(G_n) <= sum_(i,j) a_i a_j sqrt(Var(Y_i)Var(Y_j)) <= T^(2H)`.

Also `sum_i a_i Var(Y_i) <= T^(2H)`. The centered Gaussian MGF now yields

`E exp(-q eta G_n/2)`

`= exp(q^2 eta^2 Var(G_n)/8)`

and the deterministic variance correction contributes at most

`exp(q eta^2 T^(2H)/4)`.

Multiplying the factors proves the stated bound. QED.

## 4. Defensive-proposal corollary

Let `L=dP/dQ` be the exact defensive-mixture likelihood and let the zero-shift mass be
`delta>0`. Since `0<L<=1/delta`,

`E_Q[L^2 B_n^(-q)] <= delta^(-1) E_Q[L B_n^(-q)]`

`= delta^(-1) E_P[B_n^(-q)]`.

Thus the same target-law inverse-slope bound supplies the likelihood-weighted moment
needed in localized second-moment arguments. This step would be invalid for a pure
non-defensive Gaussian shift because no deterministic likelihood bound exists.

## 5. Verification for the declared V6 direction families

The theory-diagnostic proposal schedules have price-driver magnitudes proportional to
the same two-segment vector: zero for the defensive expert, `(2,1)` for the middle
expert, and `(4,2)` for the strongest expert, up to a common sign.
`rank_one_price_control_span` orients the nonzero anchor positively and normalizes it.

For two equal, grid-aligned segments with positive magnitudes `a` and `b`,

`c_n = sqrt(T/2) (a+b)/sqrt(a^2+b^2)`.

This quantity is independent of `n` and strictly positive. For `(a,b)=(4,2)`,

`c_n = sqrt(T/2) 6/sqrt(20)`.

The CEM-anchored computational policy uses separate positive price magnitudes after
orientation: approximately `(0.508085,0.442047)` for terminal and
`(0.358043,0.070209)` for barrier. Its half/full experts are proportional, so the same
formula applies. The corresponding masses are approximately `0.997593 sqrt(T)` and
`0.829958 sqrt(T)`, respectively.

Every declared hierarchy starts from eight steps and doubles the grid, so the
midpoint segment boundary remains aligned at every level. The direction-mass premise
is therefore discharged for both the diagnostic and CEM-anchored computational
families, not merely observed in a plot.

## 6. Code contract

The function `terminal_slope_inverse_moment_bound` in
`src/path_integral/rbergomi_theory_diagnostics.py` implements the displayed bound and
rejects nonpositive, nonunit, grid-inconsistent, or lower-mass-violating directions.
The constant-volatility test (`eta=0`) is an equality oracle. A separate two-resolution
test verifies the grid invariance of the current piecewise direction mass.

The multilevel diagnostic artifact records both empirical inverse moments and this
analytic upper bound. The empirical values remain diagnostics; the proof above is
what discharges the negative-moment obligation.

## 7. What remains conditional

The separate `G11_V6_TERMINAL_COEFFICIENT_RATE_THEOREM.md` gives a proof candidate
for the fine/coarse terminal coefficient, threshold, weak-bias, and cost obligations
at every rate `r<H`. After independent mathematical review, the resulting FFT-MLMC
complexity would be conservatively
`O(epsilon^(-1/r) log(epsilon^-1))`, not `O(epsilon^-2)`.

For discrete barriers the active time can occur close to zero and fine-only monitoring
points add a mesh-enrichment term. The terminal result must not be transferred to the
barrier case without the separate obligations in the V6 plan.
