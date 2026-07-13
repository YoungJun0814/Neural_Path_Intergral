# PI2 Review: Soft Heston Desirability and Two-Driver Oracle

Date: 2026-07-13<br>
Status: requested oracle items 1–2 complete; PI2 objective benchmark pending

## Outcome

The project now has a deterministic soft Heston conditional desirability,
analytic state gradients independently cross-checked by Richardson finite
differences, and the corresponding two-coordinate Föllmer/Doob drift. The
implementation is suitable as a numerical oracle for the next neural
PI/PICE controller stage.

This is not yet a practical online feedback controller. Evaluating Fourier
quadrature per simulated state would be too expensive; the next controller
stage must use oracle samples, a precomputed interpolation grid, or supervised
distillation rather than calling this scalar reference on every path and step.

## Soft conditional desirability

For

$$
g_{\tau,K}(S_T)=\operatorname{sigmoid}
\left(\frac{K-S_T}{\tau K}\right),
$$

the oracle evaluates

$$
h_\tau(t,S,v)=E[g_{\tau,K}(S_T)\mid S_t=S,v_t=v].
$$

If \(Y\) is standard logistic, integration by parts gives

$$
h_\tau(t,S,v)
=E_Y\left[F_{S_T\mid S_t=S,v_t=v}(K+\tau K Y)\right].
$$

This turns the smooth payoff expectation into a deterministic mixture of
conditional Heston CDF values. Gauss–Legendre nodes are generated with SciPy
`roots_legendre`, avoiding a native Torch/MKL conflict observed with NumPy's
eigenvalue-based `leggauss` implementation on Windows.

The CDF mixture uses one vector-valued Gil–Pelaez integration rather than one
Fourier integration per logistic node. A vectorized CDF test agrees with the
existing scalar Heston inversion to approximately machine precision.

## Independent gradient verification

The affine Heston characteristic function is differentiated analytically:

$$
\partial_x\phi=iu\phi,
\qquad
\partial_v\phi=D(u)\phi.
$$

Differentiating the Gil–Pelaez integral provides the primary
\(\partial_x\log h\) and \(\partial_v\log h\). An independent check evaluates
second-order differences at step \(\delta\) and \(\delta/2\), followed by
Richardson extrapolation. Variance switches to a second-order forward scheme
when a central stencil would cross \(v=0\).

At the representative state

```text
S=100, v=0.04, remaining T=0.5, K=85, tau=0.05
kappa=1.8, theta=0.04, xi=0.45, rho=-0.65, r=0.03
```

the default-order result is approximately

```text
h                    = 0.14301959
d_log_h / d_log_S    = -7.50989
d_log_h / d_v        =  9.85637
u1                   = -2.07861
u2                   =  0.67412
```

For the order-48 diagnostic run, analytic-versus-Richardson discrepancies were
about `2.6e-9` for log spot and `1.4e-8` for variance. Logistic quadrature order
96 versus 128 also converges within the registered gradient tolerances.

## Independent Monte Carlo check

A 50,000-path, `dt=1/512` full-truncation Heston run produced

```text
Fourier/logistic oracle = 0.14301959
Monte Carlo mean        = 0.14295652
Monte Carlo SE          = 0.00119469
difference              = -0.053 standard errors
```

The permanent regression test uses 30,000 paths with a four-standard-error
plus discretization allowance.

## Near-boundary and near-maturity safeguards

Small variance and short remaining maturity make the conditional distribution
sharp, so a fixed Fourier cutoff can create Gibbs-type CDF nonmonotonicity.
Observed examples included:

- `v=1e-5`: cutoff 180 produced a monotonicity violation around `4.0e-5`;
  cutoff 500 removed it;
- remaining maturity `0.01`: cutoff 180 failed while an increased cutoff
  passed;
- remaining maturity `0.001`: approximately 900 was required in the diagnostic.

The oracle therefore checks CDF monotonicity and increases the cutoff from 180
geometrically up to 1440. It raises an error if the maximum still fails. It does
not apply an undocumented isotonic repair or silently floor the desirability.
As in the scalar Heston reference, roundoff-scale raw CDF values outside
`[0,1]` are range-clipped; analytic tail derivatives are then set to zero.

Soft desirability below the configured reliability threshold also raises an
error instead of returning a meaningless log gradient.

## Theoretical error audit

No unresolved error was found in requested items 1–2:

1. The logistic-mixture identity has the correct sign: for deterministic
   terminal spot it reduces to `sigmoid((K-S)/(tau*K))`.
2. The gradient variable is \(x=\log S\), not spot \(S\).
3. The variance coefficient is the affine Heston \(D(u)\), so
   \(\partial_v\phi=D\phi\).
4. The oracle is exactly \(\sigma^\top\nabla\log h\) in the same independent
   Brownian basis as the two-driver simulator.
5. The correlation term appears in \(u_1\), while \(u_2\) uses
   \(\sqrt{1-\rho^2}\); no correlated Brownian density is counted twice.
6. The soft continuous oracle is not identified with a hard conditional law or
   an exact finite-grid zero-variance proposal.
7. CDF nonmonotonicity and unreliable soft probabilities are surfaced as
   diagnostics rather than hidden by a monotone repair or desirability floor.

## Remaining PI2 work

- define and validate a near-maturity control clipping/distillation policy;
- build a trainable two-driver controller that consumes oracle samples;
- implement PI free-energy, feedback PICE, and \(J_2\) refinement;
- compare oracle, learned two-driver, one-driver, affine, and CEM proposals;
- complete time-step refinement and sealed multi-seed evaluation.
