# G11 V6 Terminal Coefficient, Threshold, and Complexity Theorem

Date: 2026-07-23

Status: self-contained proof candidate at every exponent strictly below `H` for the
terminal strict-lognormal model under the assumptions below. It has passed the
implementation-level consistency audit, but it is **not yet a journal-ready theorem**:
the continuous/discrete coupling lemma and coefficient decomposition still require an
independent mathematical review. The endpoint exponent, sharper slope-only rate, and
every barrier claim remain outside the theorem.

## 1. Why this theorem is deliberately conservative

The executable BLP approximation has a rough Volterra error at scale `h^H`. The
terminal intercept contains a stochastic integral of that error, so the common
coefficient rate cannot in general be better than `H`. Empirically, the terminal
slope often converges near `H+1/2`, but that sharper slope-only behavior is not used
in this theorem. The proved common rate is any

`r < H`.

Writing `r=H-epsilon` with a declared `epsilon in (0,H)` prevents an endpoint or
uniformity claim from being smuggled in through a fitted regression.

This rate is consistent with the BLP kernel-approximation analysis in
[Bennedsen, Lunde, and Pakkanen](https://arxiv.org/abs/1507.03004) and with the known
rough-volatility strong-rate bottleneck discussed by
[Bayer, Hall, and Tempone](https://arxiv.org/abs/2009.01219). Those papers do not,
by themselves, prove the conditional-threshold statement below; the remaining
steps are supplied here.

## 2. Exact object and assumptions

Let `h=T/n`, with adjacent levels `h` and `2h`. The code uses the `kappa=1` BLP
scheme: the singular most-recent kernel cell is integrated exactly and every older
kernel cell is replaced by its cell average. Fine and coarse paths use the exact
joint Gaussian coupling implemented in `rbergomi_coupling.py`.

The theorem assumes:

1. `H` lies in a compact interval `[H_min,H_max]` contained in `(0,1/2)`.
2. `T`, `xi`, and `eta` lie in fixed compact positive/bounded sets.
3. `|rho| <= 1-rho_margin` for a positive `rho_margin`.
4. Proposal controls are deterministic, uniformly bounded, grid-aligned,
   piecewise-constant functions with a bounded Cameron--Martin norm.
5. The price-driver direction is positive, Euclidean-unit normalized, obtained
   from one fixed grid-aligned piecewise function, and has
   `sqrt(h) sum_i u_i >= c_direction > 0`.
6. The defensive natural component has fixed mass `delta>0`.
7. Variance is evaluated as the genuine lognormal value. There is no fixed
   volatility floor. A nonrepresentable floating-point draw is an explicit failed
   numerical run, not a silently modified model.
8. All mathematical statements are in real arithmetic. The implementation theorem
   applies to runs that pass the strict finite/positive numerical checks.

The constants can be uniform on the declared compact domain. To avoid a separate
endpoint-uniform kernel analysis, the public rate is stated for each fixed
`r=H-epsilon`.

## 3. BLP Volterra approximation lemma

Let

`Y_t = sqrt(2H) integral_0^t (t-s)^(H-1/2) dW_s`

and let `Y_t^h` be the implemented BLP approximation at a grid point. On the newest
cell the two coincide. On every older cell, `Y^h` is the `L2` projection of the
kernel onto constants. Ito isometry gives

`E|Y_t-Y_t^h|^2`

equal to the summed squared `L2` kernel projection error. Scaling
`s=t-hx` on the cells adjacent to the singularity and applying the mean-value bound
away from it yield

`sup_(t in grid) E|Y_t-Y_t^h|^2 <= C h^(2H)`.

For every finite `p`, the difference is Gaussian, so Gaussian moment equivalence
gives

`sup_t ||Y_t-Y_t^h||_Lp <= C_p h^H`.

The variance correction also satisfies

`0 <= Var(Y_t)-Var(Y_t^h) = E|Y_t-Y_t^h|^2 <= C h^(2H)`,

because the cell-average construction is an orthogonal projection. The adjacent
coupling can be realized with the same Brownian motion and its exact local kernel
integrals; therefore the triangle inequality gives

`sup_t ||Y_t^h-Y_t^(2h)||_Lp <= C_p h^H`.

This lemma describes the exact kernel and coupling used by the code. An independently
normalized coarse direction or a coarse local integral formed by merely summing the
two fine local integrals would not satisfy this contract.

## 4. Lognormal volatility transformation lemma

Define

`sqrt(V_t^h) = sqrt(xi) exp(eta Y_t^h/2 - eta^2 Var(Y_t^h)/4)`.

The Gaussian vectors have exponential moments of every finite order uniformly on
the compact parameter domain. The mean-value identity for the exponential,
Hölder's inequality, the Volterra lemma, and the `O(h^(2H))` variance-correction
error imply, for every finite `p`,

`sup_t ||sqrt(V_t^h)-sqrt(V_t^(2h))||_Lp <= C_p h^H`,

and the same rate holds for `V` itself. Bounded deterministic mean shifts only
multiply the Gaussian exponential-moment constants by a uniform finite factor.

The old implementation used `max(V,1e-10)`. That operation is Lipschitz and would
preserve a coefficient rate for a *floored* model, but it creates a nonvanishing
model bias relative to standard rBergomi. The implementation now rejects
underflow/overflow instead of silently applying that floor.

## 5. Terminal coefficient theorem candidate

For one fine direction `u^h`, write both adjacent terminal log prices on the same
standard-normal coordinate `Z_h`:

`log S_T^h = A_h + B_h Z_h`,

`log S_T^(2h) = A_(2h) + B_(2h) Z_h`.

The coarse coefficient uses pair sums

`u_(2h,k) = u_(h,2k)+u_(h,2k+1)`;

it is not renormalized.

### Theorem 1 (proof candidate)

For every finite `p>=2` and every `r<H`, there is a finite domain-dependent constant
such that

`||A_h-A_(2h)||_Lp + ||B_h-B_(2h)||_Lp <= C_(p,r) h^r`.

### Proof

The slope is a weighted sum of `sqrt(V)` with weights
`sqrt(h)u_i`. Positivity and the grid-scaled `L1` mass bound control the sum of the
absolute weights uniformly. The lognormal transformation lemma and Minkowski's
inequality give an `O(h^H)` bound for the volatility-approximation part. Replacing a
left-grid volatility value by its continuous value over one interval has the same
rate from the `H`-Hölder `Lp` increments of the Volterra process. Thus the slope
bound holds at every `r<H`.

For the log-price difference, split the error into:

- the integrated variance drift;
- the volatility-driver stochastic integral;
- the independent price-driver stochastic integral;
- the deterministic proposal mean shift; and
- the BLP variance-correction term.

Minkowski handles the drift terms. Burkholder--Davis--Gundy, followed by the
lognormal transformation lemma, handles both stochastic-integral terms and gives
`O(h^r)`. The deterministic control term is bounded by the same volatility
difference times the uniform control norm.

Finally,

`A_h-A_(2h) = (log S_T^h-log S_T^(2h))-(B_h-B_(2h))Z_h`.

Gaussian moments of `Z_h` are finite, and Hölder's inequality with a higher
coefficient moment gives the claimed intercept bound. QED.

The proof does not use the empirically faster slope-only rate.

## 6. Threshold and DCS second-moment theorem

For terminal threshold `K`, define

`a_h = (log K-A_h)/B_h`.

The previously proved terminal inverse-slope theorem supplies every negative moment
of `B_h` uniformly in the grid. The intercept has every finite moment. The
deterministic ratio-localization inequality, Hölder's inequality, and Theorem 1
therefore give, for every `r<H`,

`||a_h-a_(2h)||_L2 <= C_r h^r`.

Under a defensive mixture, the residual likelihood is bounded by `1/delta`.
Since the standard-normal CDF is globally Lipschitz,

`E_Q[(Lbar(Phi(a_h)-Phi(a_(2h))))^2] <= C_r h^(2r)`.

Thus the terminal DCS correction has

`beta = 2r`

for every `r<H`. This is a second-moment upper bound. It is not a statement that the
fitted exponent must equal `2H`, and it is not a barrier theorem.

## 7. Weak bias and FFT-MLMC complexity

Couple the finite approximation to the continuous terminal model and repeat the same
conditional-threshold argument. It gives

`|P(S_T^h <= K)-P(S_T <= K)| <= C_r h^r`

for every `r<H`. Hence one may take

- weak-bias exponent `alpha=r`;
- correction-variance exponent `beta=2r`; and
- polynomial cost exponent `gamma=1`, with the actual FFT factor
  `C_h=O(h^-1 log(h^-1))`.

Because `H<1/2`, `beta<gamma`. The standard MLMC allocation sum is dominated by the
finest level. With `h_L^r` proportional to `epsilon`, the resulting conservative
complexity is

`O(epsilon^(-1/r) log(epsilon^-1))`.

The code reports the polynomial exponent `1/r` and separately flags the FFT log
factor. In particular, this theorem does **not** support an
`O(epsilon^-2)` continuous-time claim for rough `H`. For `H=0.05`, `0.12`, and
`0.30`, the conservative exponents are intrinsically severe. This is a mathematically
useful negative limitation and explains why Route A is framed around a fixed
finest-grid practical estimator.

The generic MLMC dependency is the usual bias/variance/cost theorem; the numerical
smoothing literature, including
[Bayer, Ben Hammouda, and Tempone](https://arxiv.org/abs/2003.05708), demonstrates
why smoothing can restore rates for regular SDE discretizations, but it does not
remove the `H` bottleneck proved here for this rough-volatility coefficient.

## 8. Barrier exclusion

No statement above applies automatically to a discrete or continuous barrier. A
barrier threshold is a maximum over time and introduces:

- a possibly early active time where the slope is small;
- a changing active index;
- fine-only monitoring points; and
- continuous-monitoring bias if that target is claimed.

The existing barrier diagnostics measure these terms. Until separate analytic bounds
are proved, barrier experiments remain exact finite-grid Route A evidence only.

## 9. Theorem-to-code contract

| Mathematical item | Executable contract |
|---|---|
| genuine lognormal variance | `strict_lognormal_variance`; no positive clamp |
| adjacent BLP marginals | `rbergomi_coupling.py` and FFT/reference equality tests |
| common scalar coordinate | fine direction plus coarse pair sums |
| inverse terminal slope | `terminal_slope_inverse_moment_bound` |
| conservative exponents | `terminal_rate_contract(H, epsilon_margin)` |
| raw coefficient moments | `coefficient_moment_diagnostics` |
| exact threshold localization | `evaluate_rbergomi_threshold_coupling` |
| barrier exclusion | `TerminalRateContract.barrier_included == False` |

Passing empirical diagnostics proves none of these assumptions; it only fails to
falsify them. Conversely, a strict-lognormal overflow/underflow, coupling identity
failure, direction-mass failure, or rate-incompatible diagnostic forces a failed run
or a theorem-scope reduction. A journal claim also requires a separate line-by-line
proof audit by a stochastic-analysis specialist; the executable contract intentionally
reports `journal_claim_ready=false` until that review is recorded.
