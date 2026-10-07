# G11 V16 joint mesh/noise logarithmic efficiency

Date: 2026-08-11

Status: T16-9 proved under the same constant-`xi`, fixed terminal-left-tail and
Riemann--Liouville assumptions as T16-4--T16-8.  This is an exponent theorem, not an
end-to-end work-complexity theorem.

## 1. Joint family

Let `epsilon -> 0` and let the uniform BLP grid size `N(epsilon) -> infinity` at an
otherwise arbitrary rate.  On grid `N`, use the T16-8 covariance perturbation

```text
C_N = sum_(m<N) lambda_m phi_(m,N) tensor phi_(m,N)
      + scale N^(-q) P_(bridge,N),
lambda_m = scale (1+m)^(-p),  p>1, q>1.
```

The safety law is the exact finite-grid Gaussian

```text
Q_(epsilon,N) = N(0, I_(2N) + C_N/epsilon).
```

Every eigenvalue is positive at every frozen grid.  No pseudo-density or
self-normalization is used.

## 2. Exponential approximation of the reference family

Let `delta_N` be the Hilbert--Schmidt error of the BLP Volterra operator.  The proof
of Lemma 16-6a gives `delta_N -> 0`; for the Riemann--Liouville kernel the diagonal
strip already has squared mass `O(Delta^(2H))`.

Couple the BLP and continuous families with the same Brownian path.  The scaled
Volterra error is centered Gaussian with variance bounded by

```text
epsilon delta_N^2.
```

Gaussian concentration therefore gives, for every fixed `a>0`,

```text
limsup_(epsilon->0) epsilon log
 P(||sqrt(epsilon)(Y_N-Y)|| > a)
 <= -c a^2/delta_N^2 -> -infinity.
```

On a bounded scaled-Volterra localization set, the exponential volatility map is
Lipschitz.  The finite-variation price error vanishes uniformly and the adapted
martingale error has quadratic variation `O(epsilon delta_N^2)` plus the left-grid
time approximation.  The exponential martingale inequality gives the same
superexponential bound.  Gaussian localization is then removed exactly as in the
Wick-correction argument of T16-4.

Thus the BLP terminal family is an exponentially good approximation of the continuous
family at speed `1/epsilon` for **every** schedule `N(epsilon)->infinity`.  It inherits
the T16-4 LDP and

```text
lim epsilon log p_(epsilon,N(epsilon)) = -J_*.
```

## 3. Uniform determinant control

For the main spectrum,

```text
epsilon sum_(m<N) log(1+lambda_m/epsilon)
 <= epsilon sum_(m>=0) log(1+lambda_m/epsilon) -> 0
```

by the T16-5 trace-class lemma.  For the bridge complement,

```text
epsilon N log(1+scale N^(-q)/epsilon)
 <= scale N^(1-q) -> 0.
```

Hence the exact finite-grid likelihood normalizer is negligible at speed
`1/epsilon` uniformly along the joint sequence.

## 4. T16-9 second-moment proof

Write the ordinary safety estimator as

```text
Y_(epsilon,N) = Psi_(epsilon,N) dP_N/dQ_(epsilon,N).
```

Fix `m`.  Once `N>=m`, retain only the first `m` negative likelihood quadratics and
discard the rest to upper-bound the second moment.  The embedded cosine directions
`phi_(j,N)` converge strongly to `phi_j`.  Combining this fact with the exponential
approximation in Section 2 and the fixed-`m` Laplace upper bound gives

```text
limsup epsilon log E_Q[Y_(epsilon,N)^2]
 <= -inf_h [0.5||h||^2
             +0.5 sum_(j<=m)<h,phi_j>^2
             +2C_k(h)].
```

Now let `m -> infinity`.  The unchanged reference energy makes the objectives
equicoercive in the weak Cameron--Martin topology, so the T16-5 finite-coordinate
exhaustion applies without a dimension-dependent escape.  Therefore

```text
limsup epsilon log E_Q[Y_(epsilon,N)^2] <= -2J_*.
```

Jensen and the joint probability exponent supply the reverse inequality.  Hence

```text
lim epsilon log E_Q[Y_(epsilon,N(epsilon))^2] = -2J_*
```

for every `N(epsilon)->infinity`.  Any exact mixture carrying fixed positive mass on
this safety law inherits the result from the balance-density domination inequality.
The shifted components may include the hybrid bridge correctors; they cannot worsen
the exponent because the proof uses only the safety component.

## 5. Boundaries

T16-9 proves a joint asymptotic **exponent**.  It does not prove:

- bounded relative error;
- an explicit finite-`epsilon` variance constant;
- the least-cost choice of `N(epsilon)`;
- a weak-bias rate for a requested absolute or relative tolerance;
- optimizer/training complexity or neural amortization gain;
- exact simulation of the continuous-time process.

Those quantities require quantitative constants and constitute T15-8.  In
particular, an arbitrary slowly growing `N(epsilon)` is enough for the exponent but
may be useless for finite-accuracy computation.
