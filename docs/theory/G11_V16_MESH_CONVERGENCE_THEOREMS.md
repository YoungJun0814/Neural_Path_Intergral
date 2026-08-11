# G11 V16 BLP mesh and transport convergence

Date: 2026-08-11

Status: T16-6 through T16-8 proved for constant forward variance, a uniform BLP
grid, the Riemann--Liouville kernel with `0<H<1/2`, and a fixed terminal event.
No explicit weak rate and no simultaneous noise/mesh complexity rate are claimed.

## 1. One Brownian driver, two BLP coordinates

Put `alpha=H-1/2` and let `Delta=T/N`.  On cell `I_i=(t_i,t_(i+1)]`,
BLP whitens the two observables

```text
int_(I_i) 1 dW_s,
int_(I_i) (t_(i+1)-s)^alpha dW_s.
```

If `G_Delta` is their exact `2 x 2` Gram matrix and `L_Delta` its Cholesky
factor, the two functions

```text
(e_(i,1),e_(i,2))^T = L_Delta^(-1)
  (1,(t_(i+1)-s)^alpha)^T 1_(I_i)(s)
```

are orthonormal in `L2(I_i)`.  Functions from different cells are orthogonal.
The BLP standard-normal vector is therefore an orthonormal coordinate system for

```text
V_N = direct_sum_i span{1_(I_i),
        (t_(i+1)-s)^alpha 1_(I_i)(s)} subset L2(0,T).
```

This proves the exact isometry

```text
|z|^2 = ||sum_(i,a) z_(i,a)e_(i,a)||_L2^2.
```

In particular, BLP does not introduce a second continuum Brownian driver.  The code
in `blp_cameron_martin_embedding.py` constructs these functions and independently
checks their Gram matrix by adaptive quadrature.

## 2. Consistency of the deterministic BLP skeleton

Let `K` be the Riemann--Liouville Volterra operator and extend the left-grid BLP
skeleton piecewise constantly in output time.  Denote its deterministic operator by
`K_N:V_N -> L2(0,T)`.  On the most recent cell BLP integrates the singular kernel
exactly; on each older cell it uses the cell average, which is the `L2` projection on
constants.

### Lemma 16-6a

```text
||K_N P_(V_N) - K||_HS -> 0.
```

For the Riemann--Liouville kernel this follows by splitting the Volterra triangle
into the most recent diagonal strip and its complement.  The strip's squared kernel
mass is `O(Delta^(2H))`.  Away from the diagonal, cell averages converge in `L2`.
Simple functions are dense on the triangle, so the two limits combine to
Hilbert--Schmidt convergence.  The same argument covers the piecewise-constant
output-time extension.

Consequently, if `u_N` is bounded in `L2` and `u_N` converges weakly to `u`, then

```text
K_N u_N -> K u strongly in L2.
```

Indeed the operator error is uniform on bounded sets and the Volterra operator is
compact.  On every bounded Cameron--Martin ball, Cauchy--Schwarz also gives a uniform
bound on `K_Nu_N`; hence exponentiation is locally Lipschitz on the relevant range.

Define the continuum skeleton quantities

```text
sigma_u(t) = sqrt(xi) exp(eta (Ku)(t)/2),
A(u) = rho int_0^T sigma_u(t)u(t)dt,
I(u) = int_0^T sigma_u(t)^2dt.
```

Their BLP left-grid analogues satisfy

```text
A_N(u_N) -> A(u),   I_N(u_N) -> I(u).
```

The first limit is a weak/strong product limit; the second is strong convergence plus
Riemann consistency.  Strict positivity of `xi` keeps `I` away from zero on bounded
sets.

## 3. T16-6: Gamma convergence of the rate action

Embed the full-grid BLP action into `L2(0,T)` by setting it to infinity outside
`V_N`:

```text
J_N(u)=0.5||u||_2^2
 + ((A_N(u)-k)_+)^2/[2(1-rho^2)I_N(u)],  u in V_N.
```

Then `J_N` Gamma-converges to the T16-4 action `J` in the weak `L2` topology.

- **liminf:** weak lower semicontinuity controls the energy, while Lemma 16-6a
  supplies convergence of `A_N` and `I_N`.
- **recovery:** the piecewise-constant projection of any `u` belongs to `V_N`,
  converges strongly to `u`, and makes every action term converge.
- **equicoercivity:** `J_N(u)>=||u||_2^2/2`.

Therefore

```text
min J_N -> min J.
```

Every sequence of full-grid minimizers has a weakly convergent subsequence whose
limit is a continuum minimizer.  Since the conditional costs also converge, equality
of the minimum values forces convergence of the norms; the same subsequence therefore
converges strongly.  If the continuum minimizer is unique, the full sequence
converges strongly.

This theorem concerns the **full BLP Cameron--Martin space** or a genuinely dense
continuum Galerkin family.  A fixed channel-separated DCT basis is not covered.  A
fixed finite number of continuum cosine modes converges only to the corresponding
restricted action.  To recover `min J`, its mode count must tend to infinity.

### Bridge-corrector corollary

On each cell there is one unit BLP direction orthogonal to the embedded constant
drift.  It has zero cell Brownian increment.  Give a fixed number of these local
bridge directions smooth DCT envelopes across cells.  Each resulting unit vector
converges weakly to zero: its cell averages vanish and its oscillation scale is
`Delta`.  Compactness of `K` then gives a vanishing skeleton effect.

A bounded hybrid control can therefore use these modes as finite-grid correction
directions without changing the continuum action.  Their energy remains explicit.
For any strongly convergent sequence of hybrid minimizers supplied by T16-6, the
bridge coefficient norm must tend to zero; otherwise orthogonality would leave a
strict positive energy defect.  This justifies the implemented
`mesh_compatible_hybrid` basis as a practical discretization correction, but the
coefficient-decay diagnostic must still be reported rather than presumed at a coarse
grid.

## 4. T16-7: fixed-noise terminal probability convergence

Fix `epsilon>0`.  The BLP Gaussian skeleton converges to the continuous Volterra
process, and the left-point price stochastic integral converges in `L2` by Ito
isometry and the lognormal moment bounds.  Thus the integrated variance and correlated
return converge in probability.  After the independent price driver is integrated
out, the terminal conditional payoff is a bounded continuous Gaussian CDF of these
two quantities.  Dominated convergence gives

```text
p_(epsilon,N) -> p_epsilon.
```

This is a convergence theorem, not an explicit rate.  Published weak-rate results for
rough-volatility schemes impose payoff/scheme structure that is not silently assumed
for this nonlinear conditional digital.  T15-6 is therefore closed only at the
qualitative convergence level required to identify the finite-grid target.

## 5. T16-8: mesh consistency of the safety operator

Let `(phi_m)` be the continuum cosine basis and

```text
C phi_m = lambda_m phi_m,
lambda_m = scale (1+m)^(-p), p>1.
```

On grid `N`, the implementation embeds the first `N` piecewise-constant cosine
drifts exactly into BLP coordinates.  The remaining orthogonal within-cell bridge
space receives eigenvalue `scale N^(-q)`, `q>1`.  Fixed cosine projections converge
strongly, the main spectral tail is summable, and the bridge trace is
`scale N^(1-q)`.  Hence the embedded covariance perturbations satisfy

```text
||C_N-C||_trace -> 0.
```

For every fixed `epsilon>0`, this also gives convergence of the associated equivalent
Gaussian covariance perturbations and their Fredholm determinants.  Every frozen
grid remains full rank, while the spurious bridge mass disappears in the continuum.

The V16-v4 two-channel DCT safety result is not evidence for this theorem.  Its
fixed-grid likelihood and estimates remain valid, but its mesh interpretation is
quarantined by `results/g11_v16_trace_safety_v4_status_2026-08-11.json`.  V16-v5 is
the first implementation matching T16-8.

## 6. What remains open

- a quantitative weak-error rate for the exact conditional digital used here;
- a joint limit specifying how `N(epsilon)` must grow while preserving the T16-5
  second-moment exponent;
- an end-to-end complexity theorem including optimization, mixture construction and
  evaluation;
- Hessian eigenspace convergence without an isolated minimizer and a uniform spectral
  gap;
- non-constant forward variance curves outside a separately checked extension.

These are not needed for the sequential statements `N -> infinity` at fixed noise and
then the ideal continuous `epsilon -> 0` theorem, but they are required for a fully
mesh-uniform computational-complexity claim.

## 7. Primary-source boundary

The BLP/hybrid discretization and general weak-rate theory are prior art.  The closest
rate boundaries used to avoid overclaiming are Bayer--Fukasawa--Nakahara,
DOI `10.1137/22M1482871`, and Gassiat, DOI `10.1137/22M1485760`.  V16 claims neither
of their rates for the present conditional digital without an assumption match.
