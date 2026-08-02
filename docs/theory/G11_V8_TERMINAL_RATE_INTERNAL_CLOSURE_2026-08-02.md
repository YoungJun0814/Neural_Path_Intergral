# G11 V8 Terminal rBergomi Rate: Internal Closure Record

Date: 2026-08-02

Status: the previously compressed obligations O1--O5 are expanded below into a
coherent proof for each fixed exponent (r<H), under the stated direction and compact
parameter assumptions. This is an internal proof record, not the independent
stochastic-analysis review required for a submission. Discrete and continuous
barriers remain excluded.

## 1. Exact probability space and grid objects

Let ((W^1,W^2)) be independent Brownian motions on one filtered probability space.
For fixed (H\in(0,1/2)), define

\[
Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s^1,
\qquad
V_t=\xi\exp\left(\eta Y_t-\frac12\eta^2t^{2H}\right).
\]

The continuous log price is

\[
X_T=\log S_0-\frac12\int_0^T V_tdt
+\rho\int_0^T\sqrt{V_t}\,dW_t^1
+\sqrt{1-\rho^2}\int_0^T\sqrt{V_t}\,dW_t^2.
\]

On a uniform grid (t_i=ih), the implemented BLP approximation uses the exact
kernel stochastic integral on ((t_{i-1},t_i]) and the cell-average kernel on every
older interval. Write this process as (Y_i^h) and its exact discrete variance as
(q_i^h). Then

\[
V_i^h=\xi\exp\left(\eta Y_i^h-\frac12\eta^2q_i^h\right)
\]

and the implemented left-point log scheme is

\[
X_T^h=\log S_0+\sum_{i=0}^{n-1}
\left[-\frac12V_i^hh
+\sqrt{V_i^h}\left(\rho\Delta_iW^1+sqrt{1-\rho^2}\Delta_iW^2\right)\right].
\]

The adjacent (h,2h) implementation constructs the exact two BLP marginals on
this common Brownian space. In particular, its coarse recent-cell kernel integral is
sampled jointly with, and is not formed by summing, the two fine recent-cell
integrals. This distinction is required for the stated coarse marginal.

## 2. O1: common conditioning coordinate

Let (psi\in L^2([0,T])) be positive, unit normalized, piecewise constant on a
fixed partition whose breakpoints occur on every grid in the declared hierarchy.
Define

\[
Z=\int_0^T\psi(t)dW_t^2,
\qquad
u_i^h=h^{-1/2}\int_{t_i}^{t_{i+1}}\psi(t)dt.
\]

Because (psi) is constant on each grid cell,
(sum_i(u_i^h)^2=1), and

\[
\Delta_iW^2=\sqrt h\,u_i^hZ+\Delta_iW^{2,\perp},
\]

where the residual Gaussian vector is independent of (Z). On an adjacent coarse
cell, the coefficient of the same (Z) is

\[
\sqrt h(u_{2k}^h+u_{2k+1}^h),
\]

which is exactly the pair-sum convention used by the code. No independent coarse
renormalization is permitted.

Let

\[
\mathcal G=\sigma(W^1, W^{2,\perp}).
\]

Then (Z\sim N(0,1)) is independent of (mathcal G), and

\[
X_T^h=A_h+B_hZ,
\quad
B_h=\sqrt{1-\rho^2}\sum_i\sqrt{V_i^h}\int_{t_i}^{t_{i+1}}\psi(t)dt>0,
\]

with (A_h) measurable with respect to (mathcal G). Deterministic Gaussian
proposal shifts change component means but not this target-law isonormal
decomposition; the mixture label is added to the proposal conditioning sigma-field.

## 3. O2: implementation-specific Volterra error

For fixed (t_i), let (K_i(s)=\sqrt{2H}(t_i-s)^{H-1/2}). The BLP kernel equals
(K_i) on the most recent cell and its (L^2)-projection onto constants on every
older cell. Ito isometry therefore gives

\[
E|Y_{t_i}-Y_i^h|^2
=\sum_{I\subset[0,t_i-h]}\int_I|K_i(s)-\bar K_{i,I}|^2ds.
\]

On a historical cell, the one-dimensional Poincare inequality and
(K_i'(s)=O((t_i-s)^{H-3/2})) give

\[
\int_I|K_i-\bar K_{i,I}|^2
\le Ch^2\int_I(t_i-s)^{2H-3}ds.
\]

Summing cells at distances (kh), (k\ge1), yields

\[
\sup_iE|Y_{t_i}-Y_i^h|^2
\le Ch^{2H}\sum_{k\ge1}k^{2H-3}
\le C_Hh^{2H}.
\]

The series is finite for (H<1/2). Gaussian moment equivalence gives the same
(h^H) rate in every finite (L^p). Coupling the fine and coarse schemes to the
same (Y) and using the triangle inequality gives

\[
\sup_k\lVert Y_{t_{2k}}^h-Y_{t_{2k}}^{2h}\rVert_p\le C_{p,H}h^H.
\]

Constants are uniform on a compact Hurst interval bounded away from zero and one
half after replacing the public endpoint by any fixed (r<H).

## 4. O3: lognormal and price-integral transfer

The centered Gaussian arrays (Y^h,Y^{2h},Y) have uniformly bounded variances on
([0,T]), hence uniformly finite exponential moments of every fixed order. The
inequality

\[
|e^x-e^y|\le|x-y|(e^x+e^y)
\]

together with Holder's inequality transfers the Volterra rate to

\[
\sup_i\lVert\sqrt{V_i^h}-\sqrt{V_{t_i}}\rVert_p
+\sup_i\lVert V_i^h-V_{t_i}\rVert_p\le C_ph^r,
\qquad r<H.
\]

The same argument and the (L^p) increments of (Y) give

\[
\lVert\sqrt{V_t}-\sqrt{V_s}\rVert_p
+\lVert V_t-V_s\rVert_p\le C_p|t-s|^r.
\]

For the integrated-variance drift, Minkowski gives an (O(h^r)) Riemann-sum and
BLP error. For each stochastic integral, BDG gives

\[
\left\lVert\int_0^T
(\sqrt{V_{\lfloor t/h\rfloor h}^h}-\sqrt{V_t})dW_t^k
\right\rVert_p
\le C_p
\left(\int_0^T
\lVert\sqrt{V_{\lfloor t/h\rfloor h}^h}-\sqrt{V_t}\rVert_p^2dt
\right)^{1/2}
\le C_ph^r.
\]

Consequently

\[
\lVert X_T^h-X_T\rVert_p
+\lVert X_T^h-X_T^{2h}\rVert_p\le C_ph^r.
\]

This statement is for the strict lognormal variance. A positive numerical floor
would define a different model and is not part of the proof.

## 5. O4: affine coefficient rate

Define the continuous slope

\[
B=\sqrt{1-\rho^2}\int_0^T\sqrt{V_t}\psi(t)dt
\]

and (A=X_T-BZ). Positivity of (psi) and the uniform grid-scaled (L^1) direction
mass imply (B_h,B>0). The volatility-transfer bound and
(lVert\psi\rVert_{L^1}<\infty) yield

\[
\lVert B_h-B\rVert_p+\lVert B_h-B_{2h}\rVert_p\le C_ph^r.
\]

Since (Z) is independent of every slope and has moments of all orders,

\[
\lVert A_h-A\rVert_p
\le\lVert X_T^h-X_T\rVert_p
+\lVert(B_h-B)Z\rVert_p
\le C_ph^r,
\]

and likewise for adjacent levels. This directly defines (A_h) from the
code-compatible projection and discharges its measurability requirement.

## 6. O5: inverse slope, threshold, weak bias, and correction rate

The already proved weighted-Jensen bound supplies every inverse moment of (B_h),
uniformly over the declared hierarchy. The same argument applies to (B). For

\[
a_h=(\log K-A_h)/B_h,
\qquad a=(\log K-A)/B,
\]

the deterministic ratio identity, Holder's inequality, coefficient moments, and
inverse-slope moments give

\[
\lVert a_h-a\rVert_2
+\lVert a_h-a_{2h}\rVert_2\le C_rh^r,
\qquad r<H.
\]

Conditioning on (mathcal G) and using the global Lipschitz constant of (Phi)
gives the continuous terminal-event weak error

\[
|P(S_T^h\le K)-P(S_T\le K)|
\le E|\Phi(a_h)-\Phi(a)|
\le C_rh^r.
\]

Under a defensive proposal with mass (delta>0), the exact likelihood satisfies
(L\le1/\delta). Hence

\[
E_Q\left[\bar L^2
(\Phi(a_h)-\Phi(a_{2h}))^2\right]
\le\delta^{-1}E_P[(\Phi(a_h)-\Phi(a_{2h}))^2]
\le C_rh^{2r}.
\]

Thus the internally supported terminal exponents are

\[
\alpha=r,\qquad\beta=2r,\qquad r<H.
\]

With FFT path cost (O(h^{-1}\log h^{-1})), the standard MLMC allocation algebra
then gives the conservative terminal complexity

\[
O(\varepsilon^{-1/r}\log\varepsilon^{-1}),
\]

not canonical (O(\varepsilon^{-2})) at small (H).

## 7. Why barriers remain outside the theorem

For a discrete barrier the scalar threshold is a maximum over time. Adjacent grids
introduce a changing active index and fine-only monitoring points. The terminal
coefficient proof controls neither the density of the discrete minimum near the
barrier nor the probability of a crossing between monitoring grids. A continuous
barrier additionally requires a monitoring-bias theorem. None is inferred from the
terminal proof.

## 8. Literature and claim correction

This internal proof cannot be described as the first nonlinear rough-volatility weak
rate. Friz--Salkeld--Wagenhofer already prove weak rates for a rough-volatility class
including rough Bergomi and establish optimality for their smooth-test setting.
Gassiat and Bayer--Fukasawa--Nakahara also provide close weak-rate boundaries. The
potentially distinguishing object is narrower: a rare terminal indicator after exact
proposal-conditional integration inside a defensive Gaussian balance mixture.

## 9. Final internal verdict

O1--O5 are internally closed at every fixed (r<H) under the declared assumptions.
O6, independent mathematical review, is not discharged by this repository. Therefore:

- terminal theory may be presented as a proof candidate with a complete internal
  derivation;
- barrier theory remains finite-grid only;
- empirical slopes are not proof;
- Stage B's failed broad gate is unaffected; and
- no submission-ready or top-journal claim is authorized without independent review
  and a final subscription-database novelty challenge.
