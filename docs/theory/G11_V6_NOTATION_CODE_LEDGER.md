# G11 V6 Notation-to-Code Ledger

Date: 2026-07-23

Status: finite-grid implementation mapping verified; conservative terminal
coefficient, threshold, weak-bias, and complexity rates have a self-contained proof
candidate for the strict-lognormal model. Independent mathematical review and
barrier rates remain open.

## 1. Purpose

This ledger prevents a theorem about one discretization, Gaussian coordinate, or
direction normalization from being attached to different executable code. Numerical
diagnostics may falsify a proposed assumption but cannot prove an asymptotic theorem.

## 2. Fixed target and parameter domain

- Primary estimand: `p_L = E_P[F_L]` on the declared finest grid.
- Primary V6 grid: `L` corresponding to 128 time steps.
- Primary tasks: terminal left tail and discretely monitored lower barrier.
- The scalar smoothing theorem requires `H in (0,1/2)` and `|rho|<1`.
- Any theorem with uniform constants must declare compact parameter subsets bounded
  away from `H=0`, `H=1/2`, and `|rho|=1`.
- No continuous-monitoring or continuous-time exactness follows from this ledger.

## 3. Gaussian inputs and coupling

| Mathematical object | Executable object | Contract |
|---|---|---|
| fine standardized Brownian normals | proposal increments divided by `sqrt(fine_dt)` | two drivers, finite `float64` |
| adjacent coarse increment | sum of paired fine Brownian increments | exact shared probability space |
| Volterra path | BLP/FFT convolution in `rbergomi_fft.py` | same innovations used by coupled paths |
| proposal component | deterministic time control in `rbergomi_mixture.py` | final path cannot change the control |
| mixture likelihood | `mixture_log_likelihood` | exact balance likelihood, not selected-component weight |
| defensive component | expert index zero | runtime replay must be identically zero |
| defensive mass | `weights[0]` | bound `L <= 1/weights[0]` |

Relevant code:

- `src/path_integral/rbergomi_fft.py`;
- `src/path_integral/rbergomi_multilevel.py`;
- `src/path_integral/rbergomi_mixture.py`;
- `src/path_integral/rbergomi_mlmc_sampler.py`; and
- `src/path_integral/rbergomi_hybrid.py`.

## 4. Common scalar coordinate

Let the standardized independent price-driver vector be `X` and let `u_L` be the
positive unit direction. The implementation computes

`Z_L = <X,u_L>` and `R_L = X-u_L Z_L`.

The executable map is `orthogonal_gaussian_residual` called by
`affine_rbergomi_log_spot` in `rbergomi_smoothing.py`.

Required identities:

- `||u_L||_2 = 1`;
- every entry of `u_L` is strictly positive;
- `<R_L,u_L> = 0` to floating-point tolerance; and
- the reconstructed path equals `intercept + slope * Z_L`.

For an adjacent coarse grid the code does **not** renormalize a new coarse direction.
It uses the pair sums

`u_(ell-1,k)^(embedded) = u_(ell,2k) + u_(ell,2k+1)`.

This is the weight multiplying `sqrt(fine_dt)` in the coarse affine slope. A proof
using an independently unit-normalized coarse vector would be about a different
coupling.

Executable check: `direction_regularity_diagnostics` in
`rbergomi_theory_diagnostics.py`.

## 5. Affine log-spot coefficients

Conditional on volatility-driver inputs and the orthogonal price residual, the code
represents every monitored log spot as

`log S_(t_k) = A_(ell,k) + B_(ell,k) Z_L`.

The increment and cumulative slopes are exactly

`Delta B_(ell,k)`
`= sqrt(1-rho^2) * sqrt(V_(ell,k)) * sqrt(fine_dt) * w_(ell,k)`,

`B_(ell,k) = sum_(j<=k) Delta B_(ell,j)`.

Here `w` is a fine direction entry on the fine path and a pair sum on the adjacent
coarse path. The volatility factor is independent of the integrated price coordinate.

Executable object: `AffineRBergomiLogSpot` from `affine_rbergomi_log_spot`.

Verified finite-grid facts:

- `B_(ell,0)=0`;
- post-initial slopes are strictly positive for the declared positive direction and
  `|rho|<1`;
- path reconstruction is checked; and
- fine and coarse arrays use their actual coupled volatility paths.

Proved model-level fact:

- for the declared positive direction family with grid-scaled mass bounded below,
  every terminal inverse-slope moment has an explicit mesh-uniform target-law bound;
  see `G11_V6_TERMINAL_SLOPE_THEOREM.md`.

Candidate asymptotic facts:

- for every `r<H`, adjacent terminal intercept and slope differences are
  `O(h^r)` in every finite `L^p` under the declared compact parameter and
  deterministic-control domain; and
- the constants are uniform on that declared compact domain.

The proof candidate and its exact limitations are in
`G11_V6_TERMINAL_COEFFICIENT_RATE_THEOREM.md`; the independent obligation audit is
in `G11_V6_TERMINAL_COEFFICIENT_PROOF_AUDIT_2026-07-23.md`.

Executable diagnostics: `slope_lower_tail_diagnostics` and
`coefficient_moment_diagnostics`. Their output is empirical evidence only.

## 6. Proposal likelihood and cancellation

For deterministic Gaussian shifts `m_j`, the full likelihood is

`L(x) = [sum_j pi_j exp(m_j^T x-||m_j||^2/2)]^-1`.

After decomposing each mean into the integrated span and its residual component, the
residual likelihood is

`Lbar(r) = [sum_j pi_j exp(c_j^T r-||c_j||^2/2)]^-1`.

The conditional law of `Z_L` under the proposal is generally not standard normal.
The DCS identity follows because `L` cancels the proposal density before integrating
the target Gaussian coordinate. The implementation must never replace this
cancellation with a proposal-conditional standard-normal assumption.

Relevant code and oracles:

- `src/path_integral/gaussian_span_marginalization.py`;
- `src/path_integral/rbergomi_smoothing.py`;
- `tests/test_gaussian_span_marginalization.py`; and
- `tests/test_rbergomi_dcs_mlmc.py`.

## 7. Terminal threshold

For a terminal lower-tail level `K`, the conditional threshold is

`a_ell = (log K-A_(ell,T))/B_(ell,T)`.

The marginalized contribution is `Lbar Phi(a_ell)`. For an adjacent correction it is
`Lbar[Phi(a_ell)-Phi(a_(ell-1))]` on the common coordinate.

The deterministic localized ratio inequality is implemented by
`evaluate_rbergomi_threshold_coupling` and `threshold_stability.py`. It separates:

- numerator error `|A_ell-A_(ell-1)|`;
- denominator error `|B_ell-B_(ell-1)|`;
- a coarse numerator envelope; and
- the bad event where a required slope is below `kappa`.

Pointwise positivity of `B` does not discharge the bad-event term. Route B must prove
an appropriate lower-tail or inverse-moment result before an unconditional terminal
rBergomi rate is claimed.

## 8. Discrete barrier threshold

For a lower barrier `K`, the conditional threshold is the maximum over monitored
times:

`a_ell = max_k (log K-A_(ell,k))/B_(ell,k)`.

Adjacent fine/coarse error has two distinct pieces:

1. error among monitoring times shared by both grids; and
2. the nonnegative enrichment defect from fine-only monitoring times.

The exact decomposition is returned by `evaluate_rbergomi_threshold_coupling`.
`barrier_obligation_diagnostics` separately reports active-time and enrichment
quantities. The terminal theorem cannot absorb the second term.

## 9. Selection and final-sample filtration

Let `G` contain proposal training, rarity screening, routing, selector profiles, and
allocation pilots. The route, proposal, selected start, and integer allocation are
`G`-measurable. Final sample streams are independent of `G`.

Conditional on `G`:

- every direct MC/IS final mean uses its exact likelihood;
- every Hybrid start telescopes to the same `p_L`; and
- therefore the selected final estimate has conditional mean `p_L`.

Pilot samples may not be inserted into the reported final mean. Reference values may
score accuracy but may not enter the router or candidate choice.

Executable safeguards:

- `rarity_router.py` has no reference-probability input;
- `prepare_hybrid_run` and `prepare_single_term_run` allocate no final seed;
- `execute_hybrid_run` derives final seeds only after the preparation hash is frozen;
- strict checkpoint parsing prevents a resumed run from changing the allocation; and
- `seed_ledger.py` rejects duplicate canonical stream identities.

## 10. Bounded and unbounded methods

Defensive mixtures with natural mass `delta_0` satisfy an absolute likelihood bound
and can use the registered bounded intervals. Pure single-shift CEM has an unbounded
Gaussian likelihood ratio. Its final mean remains unbiased after independent proposal
training, and it has finite moments for a finite deterministic shift, but it must not
be assigned a fictitious Hoeffding bound.

`SingleTermDesign(absolute_bound=None)` enforces this distinction. The common runner
then returns an asymptotic interval and `bounded_confidence_interval=None`.

## 11. Claim ledger after the V6 foundation changes

| Claim | Status |
|---|---|
| strict finite-grid cell/protocol identity | implemented and tested |
| actual zero-component defensive mass exposed | implemented and tested |
| random rarity routing preserves final unbiasedness | proved by conditioning; oracle-tested |
| profiling work cap is predeclared and immutable | implemented and tested |
| pure CEM is not given a false bounded interval | implemented and tested |
| direction/pair-sum convention diagnostics | implemented and tested |
| empirical slope/coefficient/barrier diagnostics | implemented and tested; not proofs |
| uniform terminal slope inverse moments | proved for the declared positive direction family |
| terminal rBergomi coefficient rates | proof candidate for every exponent `r<H`; independent review pending |
| terminal DCS correction rate | candidate second moment `O(h^(2r))` for every `r<H` |
| terminal continuous-time bias/complexity | candidate bias `O(h^r)` and FFT-MLMC `O(epsilon^(-1/r) log epsilon^-1)` |
| discrete-barrier model rate | conditional/open |

## 12. Immediate proof sequence

1. discharge proof-audit obligations O1--O5 with explicit filtrations, indices, and
   compact-domain constants;
2. obtain and record an independent stochastic-analysis review;
3. commit and rerun the strict-lognormal diagnostics from a clean source state;
4. verify that coefficient and correction diagnostics do not contradict the
   conservative terminal rates;
5. keep the unfavorable rough-regime complexity conclusion rather than replacing it
   with a fitted `O(epsilon^-2)` claim; and
6. treat barrier active-time and enrichment terms as a separate future theorem.
