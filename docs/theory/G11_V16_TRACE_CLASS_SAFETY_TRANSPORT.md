# G11 V16 trace-class safety transport

Date: 2026-08-11

Status: exact-measure construction and continuous-time logarithmic efficiency proved
under the T16-4 terminal-left-tail assumptions.  A simultaneous `epsilon -> 0`,
`N -> infinity` complexity theorem is not claimed.

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

The implemented fixed-grid component is the exact truncation of this formula, but it
does **not** assign a separate continuum DCT spectrum to each BLP local channel.  The
two local coordinates are orthonormal observables of the same Brownian driver.  The
mesh-compatible construction therefore uses

```text
main directions: exact BLP images of N piecewise-constant cosine drifts,
lambda_m = scale (1+m)^(-p), p>1;

bridge complement: the N orthogonal within-cell directions,
lambda_bridge,N = scale N^(-q), q>1.
```

All `2N` finite-grid eigenvalues are strictly positive, so fixed-grid T16-2 applies.
The main spectra converge to a positive trace-class continuum operator and the total
bridge trace is `scale N^(1-q) -> 0`.  The earlier V16-v4 experiment, which put an
independent low-frequency DCT basis on both local channels, remains valid fixed-grid
evidence but is explicitly quarantined as invalid mesh evidence.

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

## 5. T16-5: continuous second-moment theorem

Let `Psi_epsilon` be the exact conditional terminal-left-tail probability after the
independent price Brownian motion has been integrated out, and put

```text
p_epsilon = E_P[Psi_epsilon],
J_* = inf_h {0.5 ||h||_H^2 + C_k(h)}.
```

Here `C_k` is the conditional Gaussian tail cost in T16-4.  Assume its local-uniform
Mills-ratio limit on Cameron--Martin sublevels and the compact/weak continuity of the
Volterra skeleton used in that theorem.  For the ordinary IS estimator

```text
Y_epsilon = Psi_epsilon dP/dQ_epsilon,
```

the exact change-of-measure identity gives

```text
E_Qepsilon[Y_epsilon^2]
 = E_P[Psi_epsilon^2 dP/dQ_epsilon].
```

Write `w_j(epsilon)=lambda_j/(epsilon+lambda_j)`.  Apart from the Fredholm
determinant, whose `epsilon log` vanishes by Section 4,

```text
dP/dQ_epsilon
 = exp(-0.5 sum_j w_j(epsilon) X_j^2).
```

For every fixed `m`, discard all negative quadratic terms after coordinate `m`.  This
is an **upper** bound, not a pointwise passage through the infinite sum.  Applying the
Gaussian LDP/Laplace upper bound with fixed `m` yields

```text
limsup epsilon log E_Qepsilon[Y_epsilon^2]
 <= - inf_h [
      0.5||h||_H^2
      + 0.5 sum_(j<=m) <h,e_j>_H^2
      + 2 C_k(h)
    ].
```

The objectives increase with `m`.  Their minimizers stay in a common weakly compact
Cameron--Martin ball because the first energy term is never removed.  Along a weakly
convergent minimizing subsequence, compactness of the Volterra map makes `C_k`
continuous, while every fixed coordinate converges.  Taking `m -> infinity` therefore
gives

```text
limsup epsilon log E_Qepsilon[Y_epsilon^2]
 <= - inf_h [||h||_H^2 + 2 C_k(h)]
 = -2 J_*.
```

This finite-coordinate argument is the missing no-escape lemma.  It does not require
the non-uniform quadratic forms themselves to be equicoercive; the reference Gaussian
rate supplies the common coercivity.  Finally Jensen's inequality gives

```text
E_Qepsilon[Y_epsilon^2] >= p_epsilon^2,
```

and T16-4 gives the reverse exponent.  Hence

```text
lim epsilon log E_Qepsilon[Y_epsilon^2] = -2 J_*.
```

Any exact mixture with a fixed positive mass on `Q_epsilon` inherits the same exponent
from `q_mix >= alpha q_epsilon`.  Thus continuous-time logarithmic efficiency is
proved for the ideal trace-class safety component and for such mixtures.  This is not
a bounded-relative-error theorem and does not imply uniform finite-cost performance.

### Assumptions that cannot be dropped silently

- `C` is fixed, positive, injective and trace class; every `lambda_j` is positive.
- The event and Volterra family are exactly those covered by T16-4.
- The conditional tail exponent is locally uniform on rate sublevels.
- `C_k` is weakly continuous on bounded Cameron--Martin sets, which follows here from
  compactness of the Volterra operator, positivity of integrated variance, and the
  weak/strong product limit in `A(h)`.
- The final estimator uses the ordinary balance-mixture likelihood.

The theorem is continuous-time.  It does not by itself authorize a joint
mesh/noise-limit or end-to-end work-complexity claim.

## 6. Counterexample guard

If any `lambda_j=0`, the finite-coordinate exhaustion cannot recover that direction
from the safety likelihood. If the spectrum is not Hilbert--Schmidt, Gaussian
equivalence can fail. If it is not trace class, the simple Fredholm determinant formula
and determinant-scale proof above are not available. V16 therefore requires strictly
positive eigenvalues and `p>1` for its continuum polynomial spectrum.  On a BLP grid,
`q>1` is additionally required to make the total artificial bridge trace vanish.

## 7. Primary-source boundary

The Gaussian equivalence/density criterion is classical prior art; V16 does not claim
it. Relevant primary records include Rozanov, *On the Density of One Gaussian Measure
with Respect to Another*, DOI `10.1137/1107006`, and the infinite-dimensional
Gaussian approximation framework of Pinski et al., DOI `10.1137/140962802`.
