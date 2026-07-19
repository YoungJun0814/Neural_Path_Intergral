# G11 Theorem Contract: Defensive Gaussian Control-Span Marginalization in MLMC

Date: 2026-07-19
Status: proof-complete for the finite-dimensional statements T11-1--T11-6
Scope: deterministic Gaussian-shift mixtures and finite-grid functionals

## 1. Purpose and claim boundary

This note supplies the mathematical contract for the V11 implementation. It proves
the finite-dimensional identities that the code is allowed to use. It does not prove:

- a continuous-time barrier or occupation-time result;
- a rough-Bergomi threshold convergence rate;
- novelty relative to every paper in the literature;
- an asymptotic wall-clock speedup;
- validity for path-dependent or random controls.

The results below are deliberately stated on Euclidean Gaussian space. A discretized
Girsanov change of measure with a deterministic control is an instance of this setup
after Brownian increments are standardized.

## 2. Setup

Fix a level and suppress the level index. Let `d >= 1` and let `P` be the standard
Gaussian law on `R^d`, with Lebesgue density

`p(x) = phi_d(x)`.

For `j = 0,...,J-1`, fix a deterministic mean `m_j in R^d`, a weight `pi_j > 0`, and
`sum_j pi_j = 1`. Let

`Q_j = N(m_j, I_d)`,
`Q = sum_j pi_j Q_j`.

The component and mixture density ratios with respect to `P` are

`D_j(x) = dQ_j/dP(x)`
`       = exp(m_j^T x - ||m_j||^2/2)`,

`D(x) = dQ/dP(x) = sum_j pi_j D_j(x)`.

The exact target-over-proposal balance likelihood is `L(x) = 1/D(x)`.

Let `U in R^(d x k)` have orthonormal columns, where `0 <= k <= d`. Choose an
orthonormal complement `V in R^(d x (d-k))`, so that `[U,V]` is orthogonal. Define

`Z = U^T X`, `R = V^T X`,

and decompose each mean as

`b_j = U^T m_j`, `c_j = V^T m_j`.

Under `P`, `Z` and `R` are independent standard Gaussians in their respective
dimensions. Empty zero-dimensional factors are interpreted in the usual way.

The residual component ratios and their mixture are

`Dbar_j(r) = exp(c_j^T r - ||c_j||^2/2)`,
`Dbar(r) = sum_j pi_j Dbar_j(r)`,
`Lbar(r) = 1/Dbar(r)`.

For a measurable functional `F : R^d -> R`, define the target-conditional function

`G(r) = integral F(Uz + Vr) phi_k(z) dz`,

whenever this integral is finite. In the application `F` is bounded, so all
integrability conditions below hold automatically when a defensive component is
present.

## 3. Lemma: orthogonal Gaussian factorization

For `x = Uz + Vr`, the target density in orthogonal coordinates is

`p_(Z,R)(z,r) = phi_k(z) phi_(d-k)(r)`.

The `j`th proposal density is

`q_(j,Z,R)(z,r)`
`  = phi_k(z-b_j) phi_(d-k)(r-c_j)`
`  = p_(Z,R)(z,r)`
`    * exp(b_j^T z - ||b_j||^2/2)`
`    * exp(c_j^T r - ||c_j||^2/2)`.

Integrating over `z` gives

`q_R(r) = phi_(d-k)(r) Dbar(r)`.

### Proof

Orthogonality has unit Jacobian and preserves the Euclidean norm. The standard normal
density therefore factors into independent `Z` and `R` densities. The usual Gaussian
shift identity gives the displayed component factorization. Since

`integral phi_k(z) exp(b_j^T z - ||b_j||^2/2) dz = 1`,

integrating the mixture over `z` removes every `b_j` factor and yields the residual
mixture density. QED.

## 4. T11-1: exact Gaussian-mixture marginalization

### Theorem

If `F` is `P`-integrable, then

`E_Q[Lbar(R) G(R)] = E_P[F(X)]`.

### Proof

Using the residual proposal density from the lemma,

`E_Q[Lbar(R) G(R)]`
` = integral q_R(r) Lbar(r) G(r) dr`
` = integral phi_(d-k)(r) Dbar(r) Dbar(r)^(-1) G(r) dr`
` = integral phi_(d-k)(r)`
`     [integral F(Uz+Vr) phi_k(z) dz] dr`
` = E_P[F(X)]`,

where Fubini/Tonelli is justified by integrability; bounded rare-event functionals are
immediate. QED.

### Important interpretation

`G` uses the **target** conditional Gaussian law. It is nevertheless evaluated on a
residual sampled under `Q`. The correction factor `Lbar` changes the residual marginal
law back to the target. The theorem does not assert that `Z | R` under `Q` is standard
Gaussian; in general it is a residual-dependent Gaussian mixture and is not standard.

## 5. T11-2: Rao--Blackwell identity under the proposal

### Theorem

Assume `LF` is `Q`-integrable. Define `H(X) = L(X)F(X)`. Then, `Q`-almost surely,

`E_Q[H(X) | R] = Lbar(R) G(R)`.

If `H` is square-integrable, then

`Var_Q(Lbar(R)G(R)) <= Var_Q(H(X))`.

### Proof

For any residual point with `q_R(r)>0`, the conditional density under `Q` is
`q(z,r)/q_R(r)`. Therefore

`E_Q[H | R=r]`
` = [1/q_R(r)] integral q(z,r) [p(z,r)/q(z,r)] F(Uz+Vr) dz`
` = [phi_(d-k)(r)/q_R(r)]`
`   integral phi_k(z) F(Uz+Vr) dz`
` = Lbar(r) G(r)`.

The likelihood cancellation occurs before conditioning and is valid even though the
proposal conditional distribution is a mixture. The variance inequality is the law
of total variance. QED.

### Baseline boundary

The variance comparison is against the raw balance-mixture estimator under the same
`Q`, weights, controls, paths, and functional. It says nothing about wall-clock cost
and does not directly compare against a different single-CEM proposal.

## 6. T11-3: defensive likelihood bounds

### Theorem

Suppose one component is the target: `m_0=0`, with `pi_0=delta>0`. Then, pathwise,

`0 < L(X) <= 1/delta`,
`0 < Lbar(R) <= 1/delta`.

Consequently, every bounded `F` gives square-integrable raw and marginalized
contributions.

### Proof

Because `D_0(x)=1`,

`D(x) = delta + sum_(j>0) pi_j D_j(x) >= delta`.

Likewise `c_0=0`, `Dbar_0(r)=1`, and `Dbar(r)>=delta`. Taking positive reciprocals
gives both bounds. For bounded `F`, `|LF|` is bounded by `||F||_infinity/delta`; the
conditional expectation is bounded by `||F||_infinity`, so the marginalized
contribution has the same type of bound. QED.

## 7. T11-4: exact multilevel telescoping with level-specific proposals

### Setup

For levels `l=0,...,L`, let `X_l ~ P_l=N(0,I_(d_l))`. For `l>=1`, let
`C_l : R^(d_l) -> R^(d_(l-1))` satisfy `C_l C_l^T=I`. The target functionals are
`F_l`, and

`Delta_0(x_0)=F_0(x_0)`,
`Delta_l(x_l)=F_l(x_l)-F_(l-1)(C_l x_l)`.

At each level choose any frozen defensive deterministic-shift mixture `Q_l`, any
orthonormal span `U_l`, and the resulting residual likelihood `Lbar_l`. Let

`G_l(R_l) = E_(P_l)[Delta_l(X_l) | R_l]`.

Use independent sample sets between levels. Within one level correction, the fine and
coarse terms are both functions of the same `X_l` and use the same `Lbar_l`.

### Theorem

For positive integers `N_l`, define

`p_hat_L = sum_(l=0)^L (1/N_l) sum_(i=1)^(N_l)`
`          Lbar_l(R_l^(i)) G_l(R_l^(i))`.

Then

`E[p_hat_L] = E_(P_L)[F_L(X_L)]`.

### Proof

T11-1 at each level gives

`E_Ql[Lbar_l G_l] = E_Pl[Delta_l]`.

Because `C_l X_l` is standard Gaussian when `C_l C_l^T=I`,

`E_Pl[F_(l-1)(C_l X_l)] = E_(P_(l-1))[F_(l-1)(X_(l-1))]`.

Therefore the expected sum telescopes:

`E[P_hat_L]`
` = E[F_0] + sum_(l=1)^L (E[F_l]-E[F_(l-1)])`
` = E[F_L]`.

Different proposals and spans across levels do not enter this algebra. QED.

### Why separate fine/coarse likelihoods are wrong here

The declared correction is one target expectation under the fine Gaussian law:

`E_Pl[F_l(X_l)-F_(l-1)(C_lX_l)]`.

Multiplying its two terms by independently generated or independently normalized
likelihoods defines a different random variable and generally destroys the desired
coupling variance. It may remain unbiased if constructed with two separate valid
estimators, but it is not the common-sample MLMC correction in this theorem and does
not inherit its conditional formula or variance comparison.

## 8. T11-5: scalar-threshold specialization

### Theorem

Let the integrated span be rank one and write `Z~N(0,1)`. Suppose, conditional on
`R`, two Boolean functionals can be written

`F_f = 1{Z <= A_f(R)}`,
`F_c = 1{Z <= A_c(R)}`,

where the thresholds are measurable extended-real functions. Then

`E_P[F_f-F_c | R] = Phi(A_f(R))-Phi(A_c(R))`.

For finite thresholds this is an ordinary difference. Extended-real cases use
`Phi(-infinity)=0`, `Phi(+infinity)=1` and must be evaluated without subtracting
indeterminate threshold values.

### Proof

Conditional on `R`, only the scalar standard normal `Z` is random. Integrating each
indicator gives its normal CDF. Linearity of conditional expectation gives the signed
difference. QED.

### Finite-grid downside threshold

Suppose for monitoring indices `n=1,...,N`,

`log S_n(z) = A_n + B_n z`, with every `B_n>0`.

For a downside barrier `B_hit`, the event `min_n S_n <= B_hit` is equivalent to

`z <= max_n [(log B_hit-A_n)/B_n]`.

For a stress level `B_occ` and required count `K`, the event that at least `K`
right endpoints satisfy `S_n<=B_occ` is equivalent to

`z <= kth_largest_n [(log B_occ-A_n)/B_n]`,

where `kth_largest` means the `K`th largest value. Requiring both events gives the
minimum of the two thresholds.

In the rBergomi application, after conditioning on the volatility driver and the
price residual, a strictly positive price direction `q` gives

`B_n = sqrt(1-rho^2) sum_(i<n) sqrt(v_i h) q_i > 0`

provided `|rho|<1`, `v_i>0`, and all `q_i>0`. This proves the finite-grid threshold
identity under those conditions. It does not prove a continuously monitored event.

## 9. T11-6: correction second-moment upper bounds

### Assumptions

Let `A_f` and `A_c` be finite almost surely. Let `delta` be a common lower bound on
the natural-component weight. Suppose

`||A_f-A_c||_(L2(P_R)) <= C_A h^r`,

for `C_A<infinity`, `h>0`, and `r>0`. Let

`G(R)=Phi(A_f(R))-Phi(A_c(R))`.

### Theorem: marginalized correction

The marginalized contribution `Hbar=Lbar G` satisfies

`Var_Q(Hbar) <= E_Q[Hbar^2]`
`              <= (C_A^2/(2*pi*delta)) h^(2r)`.

### Proof

Using `q_R Lbar^2 = p_R Lbar` and the defensive bound,

`E_Q[Hbar^2] = E_(P_R)[Lbar(R) G(R)^2]`
`             <= delta^(-1) E_(P_R)[G(R)^2]`.

The normal CDF is globally Lipschitz with constant
`phi_max=1/sqrt(2*pi)`, so

`|G| <= phi_max |A_f-A_c|`.

Squaring and applying the assumed `L2` rate proves the result. QED.

### Theorem: raw correction upper bound

Let

`Delta=1{Z<=A_f(R)}-1{Z<=A_c(R)}`,
`H=L(X)Delta`.

Then

`E_Q[H^2] <= [C_A/(sqrt(2*pi)*delta)] h^r`.

### Proof

The full defensive bound gives

`E_Q[H^2] = E_P[L(X) Delta^2] <= delta^(-1) E_P[Delta^2]`.

Conditional on `R`, `Delta^2` is the indicator that `Z` lies between the two
thresholds. Hence

`E_P[Delta^2 | R] = |Phi(A_f)-Phi(A_c)|`
`                  <= phi_max |A_f-A_c|`.

Cauchy--Schwarz gives

`E|A_f-A_c| <= ||A_f-A_c||_2 <= C_A h^r`.

Combining the inequalities proves the result. QED.

### Claim boundary

The theorems give `O(h^(2r))` and `O(h^r)` upper bounds. They do not establish matching
lower bounds. Therefore “the rate exactly doubles” requires additional non-degeneracy
conditions. Empirical regression cannot replace those conditions.

## 10. Conditional T11-7 complexity corollary

Assume a continuous target `p` exists and, for constants `a,beta,gamma>0`,

`|E[F_L]-p| <= c_1 h_L^a`,
`Var(Hbar_l) <= c_2 h_l^beta`,
`Cost(Hbar_l) <= c_3 h_l^(-gamma)`,

with independent level estimators and the compatibility condition

`a >= 0.5 min(beta,gamma)`.

The classical MLMC allocation then gives the standard complexity cases:

- `beta>gamma`: `O(epsilon^-2)`;
- `beta=gamma`: `O(epsilon^-2 log(epsilon)^2)`;
- `beta<gamma`: `O(epsilon^(-2-(gamma-beta)/a))`.

Under T11-6, `beta=2r` is an available upper-bound exponent if the threshold-rate
assumption is proved for the model and hierarchy. An FFT cost of `N log N` carries an
additional logarithmic factor and must not be silently replaced by a pure `gamma=1`
cost in a theorem.

For a fixed finest-grid estimand `E[F_L]`, no weak-bias assertion is needed. The MLMC
estimator remains exactly unbiased for that finite-grid target, and only its sampling
work at fixed `L` is reported.

## 11. Discretized Girsanov mapping

For step size `h`, let proposal Brownian innovations satisfy

`Delta W^Q_n ~ N(0,h)`.

Under a deterministic control schedule `u_n`, the target-coordinate increment stored
on a proposal sample is

`Delta W^P_n = Delta W^Q_n + u_n h`.

The standardized target coordinate

`X_n = Delta W^P_n/sqrt(h)`

has distribution `N(sqrt(h)u_n,1)` under `Q`. Thus the finite-dimensional mean is

`m_n=sqrt(h)u_n`,

and the component log density ratio is

`log(dQ/dP) = sum_n [u_n Delta W^P_n - 0.5 u_n^2 h]`.

This matches `m^T X-||m||^2/2`. The implementation must use this convention
consistently. Replacing `sqrt(h)u` by `hu` in standardized coordinates is wrong.

## 12. Measurability and integrability checklist

The production application satisfies the theorem assumptions when:

1. model/task parameters and deterministic schedules are frozen;
2. simulation maps and finite-grid task functions are Borel measurable;
3. mixture weights are strictly positive;
4. the natural component has a fixed positive minimum weight;
5. all simulated contributions are finite;
6. the scalar threshold construction passes its pathwise equivalence test.

Because rare-event indicators are bounded and defensive likelihoods are pathwise
bounded, first and second moments exist. If the natural component is removed, the
identities T11-1 and T11-2 may still hold under integrability, but the defensive moment
guarantees and every theorem depending on `delta` no longer follow.

## 13. Theorem-to-code traceability

| Mathematical object | Planned code object | Required test |
|---|---|---|
| `m_j,pi_j` | `GaussianMixtureShiftSpec` | validation and label permutation |
| `U,V` | `OrthonormalControlSpan` | orthogonality and rotation invariance |
| `D_j,D` | full component/mixture log ratios | direct Gaussian density oracle |
| `Dbar_j,Dbar` | residual component/mixture log ratios | marginal integration oracle |
| `Lbar G` | marginalized contribution | analytic mean and Rao--Blackwell check |
| `C_l` | normalized dyadic aggregation | covariance identity and telescoping |
| `Phi(A_f)-Phi(A_c)` | stable signed CDF difference | extreme-tail high-precision oracle |
| defensive bound | maximum bound violation | randomized pathwise tests |
| `O(h^(2r))` inequality | rate diagnostics | finite-level inequality and regression |

## 14. Final mathematical verdict

T11-1--T11-6 are valid under the assumptions stated here. The core implementation is
therefore authorized to proceed to Gaussian oracle testing.

Still open and not authorized as a claim:

1. a rough-Bergomi `L2` threshold rate;
2. a matching lower bound for raw or marginalized corrections;
3. a continuous-time hit-plus-occupation bias rate;
4. an asymptotic complexity improvement in the intended rough regime;
5. a novelty claim before completion of the literature matrix.
