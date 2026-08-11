# G11 V16 trace-class safety transport

Date: 2026-08-11

Status: exact-measure construction proved; continuous-time logarithmic-efficiency
transfer conditional on the uniform Laplace lemma stated below.

## 1. Why uniform covariance inflation cannot be the continuum answer

On every fixed `d`-dimensional grid, `N(0,I/epsilon)` is equivalent to `N(0,I)` and
gives the T16-2 exponent. In infinitely many Wiener coordinates, multiplying every
covariance eigenvalue by the same non-unit constant produces a mutually singular
Gaussian measure. An infinite-dimensional exact likelihood therefore does not exist
for uniform inflation.

The fix is not to ignore the problem, but to replace the identity inflation by a
strictly positive trace-class covariance on the Cameron--Martin space.

## 2. Construction

Let `(E,H,P)` be an abstract Wiener space and let `C:H->H` be positive,
self-adjoint, injective and trace class. Write

```text
C e_j = lambda_j e_j,
lambda_j > 0,
sum_j lambda_j < infinity.
```

Let `U~N_H(0,C)` be independent of the reference Wiener path and define `Q_epsilon`
as the law obtained by adding the random Cameron--Martin drift
`U/sqrt(epsilon)`. Conditionally on `U=u`, this is an ordinary Cameron--Martin shift.
Marginally it is a centered Gaussian covariance perturbation with relative covariance

```text
I + C/epsilon.
```

Because `C/epsilon` is Hilbert--Schmidt and `I+C/epsilon` is boundedly invertible for
every fixed `epsilon>0`, the Feldman--Hajek conditions give `Q_epsilon equivalent P`.
Thus this construction avoids the singularity of uniform infinite-dimensional
inflation.

## 3. Exact density

Let `X_j` be the Paley--Wiener coordinate associated with `e_j`. The exact density is

```text
log(dQ_epsilon/dP)
 = -0.5 sum_j log(1+lambda_j/epsilon)
   +0.5 sum_j [lambda_j/(epsilon+lambda_j)] X_j^2.
```

For fixed `epsilon`, both sums are well defined: trace class controls the Fredholm
determinant, and

```text
sum_j lambda_j/(epsilon+lambda_j)
 <= epsilon^(-1) sum_j lambda_j < infinity.
```

The implemented fixed-grid component is the exact truncation of this formula. It uses
a full DCT basis on both BLP local channels and a spectrum

```text
lambda_m = scale (1+m)^(-p), p>1.
```

All finite-grid eigenvalues are strictly positive, so fixed-grid T16-2 still applies.
The summability condition is explicit rather than inferred from a finite array.

## 4. Determinant scale lemma

Trace class gives

```text
lim epsilon sum_j log(1+lambda_j/epsilon) = 0.
```

Indeed each summand multiplied by `epsilon` tends to zero and is bounded by
`lambda_j` through `log(1+x)<=x`; dominated convergence applies. Hence the exact
normalizing determinant has no cost at large-deviation speed `1/epsilon`.

For each fixed `h in H`,

```text
sum_j [lambda_j/(epsilon+lambda_j)] <h,e_j>^2 -> ||h||_H^2
```

by dominated convergence. Pointwise, the likelihood therefore recovers the complete
Cameron--Martin energy even though the covariance perturbation is trace class.

## 5. T16-5 proof boundary

The preceding statements prove exact equivalence and the pointwise exponent. To
promote them to a continuous-time second-moment theorem one still needs the following
uniform Laplace lemma.

> For the conditional Volterra payoff and the compact Volterra skeleton map, the
> second-moment integral under `Q_epsilon` has no escaping sequence in eigen-directions
> where `lambda_j << epsilon`, and its Laplace upper bound equals
> `-2 inf_h J_k(h)`.

The lower bound follows from full support of `N_H(0,C)` around every fixed
Cameron--Martin minimizer. The hard part is the uniform upper bound: pointwise
convergence of the weighted energy is insufficient by itself because the relevant
coordinate index may depend on `epsilon`.

Accordingly:

- exact continuous-time equivalence and density: **proved**;
- fixed-grid logarithmic efficiency for every truncation: **proved**;
- mesh-uniform/continuous logarithmic efficiency: **conditional**, not yet claimed.

## 6. Counterexample guard

If any `lambda_j=0`, a dominating control with a component in that direction need not
be covered at the safety scale. If the spectrum is not Hilbert--Schmidt, Gaussian
equivalence can fail. If it is not trace class, the simple Fredholm determinant formula
and determinant-scale proof above are not available. V16 therefore requires strictly
positive eigenvalues and `p>1` for its implemented polynomial spectrum.

## 7. Primary-source boundary

The Gaussian equivalence/density criterion is classical prior art; V16 does not claim
it. Relevant primary records include Rozanov, *On the Density of One Gaussian Measure
with Respect to Another*, DOI `10.1137/1107006`, and the infinite-dimensional
Gaussian approximation framework of Pinski et al., DOI `10.1137/140962802`.

