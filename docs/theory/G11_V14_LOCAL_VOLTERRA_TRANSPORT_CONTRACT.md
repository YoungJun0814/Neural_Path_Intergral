# G11 V14 Exact Conditional Local-Volterra Transport Contract

Date: 2026-08-11

## Finite-grid law

The BLP rBergomi simulator uses `2N` independent standard Gaussian local innovations
`Z` to construct the Volterra variance path and an independent `N`-dimensional price
driver. For a terminal threshold event, conditional on `Z`, terminal log-price is a
scalar Gaussian. Therefore

\[
g(Z)=P(S_T\le K\mid Z)=\Phi\!\left(
\frac{\log K-m_T(Z)}{s_T(Z)}\right)
\]

is an exact conditional probability under the implemented finite-grid law. This
integrates the complete independent price driver, not one fixed price direction.

## Local transport

Let `p` be the standard Gaussian density on `R^(2N)`. For training-only powers
`beta_j`, adaptive SMC targets densities proportional to `g(z)^beta_j p(z)`. V14
retains only their empirical means `mu_j` and freezes the defensive translation
mixture

\[
q(z)=\delta p(z)+(1-\delta)\sum_j\omega_j p(z-\mu_j).
\]

The exact balance likelihood is

\[
\frac{q(z)}{p(z)}=\delta+(1-\delta)\sum_j\omega_j
\exp(z^\top\mu_j-\|\mu_j\|^2/2),
\qquad \frac{p}{q}\le\frac1\delta.
\]

Fresh IID draws from frozen `q` produce the ordinary estimator

\[
\widehat P=\frac1M\sum_i g(Z_i)\frac{p(Z_i)}{q(Z_i)}.
\]

It is unbiased conditional on the frozen proposal. SMC normalized weights are never
used in this final estimator and SMC particles are not inferential units.

## Variance identity

Sampling an auxiliary standard normal `A` gives the paired raw contribution
`L(Z) 1{A<=a(Z)}`. The exact Rao--Blackwell gap is

\[
E_q[L(Z)^2 g(Z)(1-g(Z))]\ge0.
\]

## Claim boundary

- Exactness is finite-grid and terminal-event specific.
- SMC mean estimation has no asserted mesh-uniform convergence rate.
- Development efficiency is empirical and training-inclusive.
- Distribution-free bounded-range certificates are reported separately; a censored
  certificate does not invalidate unbiasedness but prohibits that specific finite-
  sample guarantee.
- Barrier and occupation events require a different conditional integral.
