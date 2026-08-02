# G11 V8 Moment-Localized Population Strictness Theorem

Date: 2026-08-02

Status: proof-complete for a finite-dimensional standard-Gaussian target and a
finite defensive Gaussian location mixture. The executable certificate and an
independent quadrature oracle are implemented. This is not a continuous-time rough
volatility rate, a universal relative-efficiency result, or an externally established
novelty claim.

## 1. Setting

Let the target input be (X\sim N(0,I_d)). Fix a deterministic Euclidean-unit
direction (u\in\mathbb R^d), and write

\[
X=Zu+R,\qquad Z\sim N(0,1),\qquad R\sim N(0,I_{d-1}),\qquad Z\perp R.
\]

The proposal is the finite location mixture

\[
Q=\sum_{j=0}^{J-1}w_j N(\mu_j,I_d),
\qquad w_j>0,\qquad \sum_jw_j=1.
\]

At least one component is exactly natural: (mu_j=0). Its total weight is the
defensive mass (delta>0). Decompose every component shift as

\[
b_j=u^\top\mu_j,
\qquad v_j=\mu_j-b_ju,
\qquad v_j\perp u.
\]

Conditional on (R=r), the event is a scalar threshold

\[
E=\{Z\le a(r)\}.
\]

The threshold may be an arbitrary measurable finite-grid terminal or discrete-barrier
map. It need not be linear in (r).

## 2. Exact residual density bound

The proposal-to-target residual density ratio is

\[
D(r)=\frac{dQ_R}{dP_R}(r)
=\sum_j w_j\exp\left(v_j^\top r-\frac12\lVert v_j\rVert^2\right).
\]

On the residual ball (lVert r\rVert\le R_0), Cauchy--Schwarz gives the explicit
upper bound

\[
D(r)\le M(R_0)
:=\sum_jw_j\exp\left(lVert v_j\rVert R_0-rac12\lVert v_j\rVert^2\right).
\]

This expression exposes the complete proposal geometry and all component weights.
In particular, an exactly natural component contributes its defensive mass
(delta) to (M(R_0)). No replacement of (M) by (1/\delta) is made: the latter
is an upper bound for the full likelihood (dP/dQ), not the residual ratio (D),
and would have the wrong direction for this step.

## 3. Moment localization

Assume a target-law threshold moment bound

\[
E_P[|a(R)|^p]\le K_p
\]

for some (p>0). For (A>0), Markov's inequality yields

\[
P_R(|a(R)|>A)\le K_p/A^p.
\]

Because (lVert R\rVert^2\sim\chi^2_{d-1}), the intersection

\[
\mathcal B=\{\lVert R\rVert\le R_0,\ |a(R)|\le A\}
\]

has target residual probability at least

\[
\eta(A,R_0)
=\left[F_{\chi^2_{d-1}}(R_0^2)-K_p/A^p\right]_+.
\]

For (d=1), there is no residual random coordinate and the first term is defined as
one.

## 4. Quantitative population theorem

Let (B_\parallel=\max_j|b_j|). If (eta(A,R_0)>0), then

\[
\boxed{
\operatorname{Var}_Q(Y_{\mathrm{raw}})
-\operatorname{Var}_Q(Y_{\mathrm{DCS}})
\ge
\frac{\eta(A,R_0)}{M(R_0)}
\Phi(-A)^2\Phi(-A-B_\parallel)>0.}
\]

### Proof

The proposal-conditional pointwise theorem already proved in
`G11_V8_FINITE_GRID_STRICTNESS_THEOREMS.md` states

\[
\operatorname{Var}_Q(Y_{\mathrm{raw}}\mid R=r)
\ge D(r)^{-2}\Phi(a(r))^2\frac{1-s(r)}{s(r)},
\]

where (s(r)=Q(Z\le a(r)\mid R=r)). Integrating against
(dQ_R=D(r)dP_R) produces one, not two, inverse factors of (D):

\[
\operatorname{Var}_Q(Y_{\mathrm{raw}})
-\operatorname{Var}_Q(Y_{\mathrm{DCS}})
\ge
E_{P_R}\left[D(R)^{-1}\Phi(a(R))^2\frac{1-s(R)}{s(R)}\right].
\]

On (mathcal B), (D^{-1}\ge M(R_0)^{-1}),
(Phi(a)\ge\Phi(-A)), and

\[
1-s=\sum_j\alpha_j(R)\Phi(b_j-a)
\ge\Phi(-A-B_\parallel).
\]

Also (s\le1). Restricting the expectation to (mathcal B) and applying the
moment-localization probability bound proves the result.

## 5. Rarity and model dependence

The theorem intentionally does not provide a rarity-independent factor. Rarer tasks
typically require a larger (A) to keep (eta>0), while the Gaussian factors
(Phi(-A)^2\Phi(-A-B_\parallel)) then shrink rapidly. Larger residual controls also
increase (M(R_0)). This behavior is consistent with the heterogeneous Stage B
variance ratios and rules out using finite-grid strictness as evidence for a uniform
factor-two claim.

For a terminal affine log-price

\[
\log S_T^h=A_h+B_hZ,
\qquad a_h=(\log K-A_h)/B_h,
\]

an explicit (K_p) can be obtained from moments of (A_h) and inverse moments of
(B_h). The existing terminal inverse-slope theorem gives, for (q>0),

\[
E[B_h^{-q}]
\le
[\sqrt{1-\rho^2}\sqrt{\xi}\,c_*]^{-q}
\exp\left(\eta_v^2T^{2H}(q/4+q^2/8)\right),
\]

where (eta_v) denotes vol-of-vol and (c_*) is the grid-scaled direction-mass
lower bound. If (E|A_h|^{2p}\le C_{A,2p}), then

\[
K_p
\le2^{p-1}\left(
|\log K|^p C_B(p)
+C_{A,2p}^{1/2}C_B(2p)^{1/2}
\right).
\]

Thus the certificate can expose (H,T,\rho,\xi,eta_v), the event level (K),
direction geometry, defensive weight, and proposal controls once a valid intercept
moment constant is supplied. The current repository has not derived a sharp uniform
closed form for (C_{A,2p}); therefore it must not advertise a model-level rarity
rate from this corollary.

## 6. Executable contract

`moment_localized_population_certificate` in
`src/path_integral/gaussian_mixture_strictness.py`:

- requires CPU `float64` tensors;
- requires positive normalized mixture weights;
- requires an exactly unit integration direction;
- requires an exactly zero, positive-weight defensive component;
- evaluates the chi-square localization probability;
- evaluates (M(R_0)) with `logsumexp`;
- evaluates the Gaussian tail factors with `log_ndtr`; and
- returns `strict_under_theorem=false` and log bound (-\infty) when the supplied
  moment localization cannot prove positive mass.

The tests include an independent SciPy quadrature of the integrated pointwise bound.
They test the derivation, not novelty or continuous-time validity.

## 7. Claim boundary

This theorem authorizes a finite-dimensional, explicit, rarity-dependent absolute
variance-gap certificate. It does not authorize:

- a lower bound on the raw-to-DCS variance ratio;
- bounded relative error as rarity tends to infinity;
- a universal material improvement factor;
- a nonlinear rBergomi indicator weak rate;
- a barrier monitoring-mesh rate;
- an MLMC complexity theorem; or
- a top-journal novelty statement before subscription-database and external expert
  review.
