# Mathematical Specification

Status: implementation-aligned specification for the publication rebuild<br>
Last updated: 2026-07-13

## 1. Scope and measure convention

Let \(\mathbb M\) denote the financial target measure and \(\mathbb Q_\phi\)
a controlled simulation proposal. The likelihood in every estimator is

$$
L_\phi=\frac{d\mathbb M}{d\mathbb Q_\phi}.
$$

For risk-neutral pricing, \(\mathbb M\) is risk neutral. For a physical stress
probability, it is the calibrated physical measure. A physical drift must not
be compared with a risk-neutral reference price.

For a nonnegative path functional \(G\),

$$
\widehat\mu_N=\frac1N\sum_{i=1}^N G(X^{(i)})L_\phi^{(i)},
\qquad X^{(i)}\sim\mathbb Q_\phi.
$$

All reported estimates use frozen controls. Training paths are not evaluation
paths.

## 2. Heston target dynamics and analytic reference

Under \(\mathbb M\),

$$
\begin{aligned}
dS_t &= \mu S_t\,dt+\sqrt{v_t}S_t\,dW_t^S,\\
dv_t &= \kappa(\theta-v_t)\,dt+\xi\sqrt{v_t}\,dW_t^v,\\
d\langle W^S,W^v\rangle_t&=\rho\,dt.
\end{aligned}
$$

For risk-neutral experiments, \(\mu=r-q\). The semi-analytic reference uses
the stable little-Heston-trap characteristic function. Calls use the standard
\(P_1/P_2\) Fourier representation. The terminal CDF uses Gil–Pelaez
inversion, and rare thresholds are numerical inverses of that CDF rather than
hand-selected strikes.

## 3. Heston discretization

Let \(n=\lceil T/\Delta t_{\max}\rceil\) and \(h=T/n\), so the last grid point
is exactly \(T\). With \(v_k^+=\max(v_k,0)\), full-truncation Euler is

$$
v_{k+1}=v_k+\kappa(\theta-v_k^+)h
          +\xi\sqrt{v_k^+}\,\Delta W_k^v.
$$

The raw variance state is retained for the next recursion; the public path
reports its nonnegative effective value. Spot uses conditional log-Euler,

$$
S_{k+1}=S_k\exp\left[
  \left(\mu-\tfrac12v_k^+\right)h
  +\sqrt{v_k^+}\,\Delta W_k^S
\right].
$$

It is positive without a post-step floor and avoids artificial far-left-tail
mass. The Fourier result is a continuous-time reference, so time-step
refinement is still required to quantify variance-discretization bias.

## 4. Brownian controlled proposal

The Markov benchmark controls the spot Brownian basis:

$$
dW_t^{S,\mathbb M}=dW_t^{S,\mathbb Q}+u_\phi(t,X_t)\,dt.
$$

Because the variance Brownian is correlated with it, the proposal dynamics are

$$
\begin{aligned}
dS_t &= [\mu+\sqrt{v_t}u_\phi(t,X_t)]S_t\,dt
       +\sqrt{v_t}S_t\,dW_t^{S,\mathbb Q},\\
dv_t &= [\kappa(\theta-v_t)+\rho\xi\sqrt{v_t}u_\phi(t,X_t)]dt
       +\xi\sqrt{v_t}\,dW_t^{v,\mathbb Q}.
\end{aligned}
$$

Omitting \(\rho\xi\sqrt v\,u\) changes the path law without changing the
declared likelihood and creates bias. The exact likelihood for the discretized
Brownian shift is

$$
\log L_\phi
=-\sum_{k=0}^{n-1}u_{\phi,k}\Delta W_k^{S,\mathbb Q}
 -\frac12\sum_{k=0}^{n-1}u_{\phi,k}^2h.
$$

It is accumulated in log space. Required diagnostics include
\(E_{\mathbb Q}[L]\), finiteness, weight ESS, maximum normalized weight,
top-1% weight share, contribution ESS, and top-1% contribution share.

## 5. Constant-control cross-entropy method

For constant \(u\),

$$
\log L_u=-uW_T^{\mathbb Q}-\tfrac12u^2T,
\qquad
\frac{W_T^{\mathbb M}}{T}
=-\frac{\log L_u}{uT}+\frac12u.
$$

For score \(H(X)\), adaptive level \(\gamma\), and elite event
\(A_\gamma=\{H(X)\ge\gamma\}\), the weighted maximum-likelihood projection is

$$
u_{j+1}
=\frac{E_{\mathbb Q_{u_j}}[
  1_{A_\gamma}L_{u_j}W_T^{\mathbb M}/T
]}{E_{\mathbb Q_{u_j}}[1_{A_\gamma}L_{u_j}]}.
$$

Each elite label, likelihood, and sufficient statistic comes from the same
trajectory. The invalid historical helper that paired elite labels with
independently resimulated states is not used.

## 6. Markov neural controller

The G2 feedback baseline is

$$
u_\phi(t,S,v)=u_{\max}\tanh f_\phi(z_t),
$$

where

$$
z_t=\left(
\log(S/S_0),\ \log(S/K),\ \log(v/\theta),\ t/T
\right).
$$

It is initialized exactly at the validation-selected CEM constant. The output
bound is a numerical safeguard and an integrability restriction. Running
averages are deliberately excluded: this is the Markov baseline.

## 7. Hard-event second-moment gradient

For \(A=\{S_T\le K\}\),

$$
J(\phi)=E_{\mathbb Q_\phi}[1_A L_\phi^2]
=\int_A \frac{p(x)^2}{q_\phi(x)}\,dx.
$$

A naive reparameterized derivative of the hard indicator ignores its moving
boundary. The implementation instead uses

$$
\nabla_\phi J
=-E_{\mathbb Q_\phi}\left[
1_A L_\phi^2\nabla_\phi\log q_\phi(X)
\right].
$$

For the controlled Brownian path, at a fixed sampled trajectory,

$$
\nabla_\phi\log q_\phi(X)
=\sum_k \nabla_\phi u_\phi(t_k,X_k)\Delta W_k^{S,\mathbb Q}.
$$

The simulator exposes the same-path \(\mathbb Q\)-Brownian increments. States,
increments, labels, and squared contributions are detached; autograd
differentiates only \(u_\phi\). Scaled-second-moment and log-second-moment use
this score gradient. A Gaussian tail test checks it against a closed-form
derivative.

The entropy-stress objective uses a smooth terminal reward and an estimated
\(\mathrm{KL}(\mathbb Q\Vert\mathbb M)\) penalty. It shapes a proposal; it is
not a variance objective or a probability estimator. For almost-everywhere
smooth payoffs such as vanilla calls and puts, ordinary reparameterized
pathwise differentiation remains available. It must not be silently applied
to a hard indicator.

## 8. Rough Bergomi target

The rough Bergomi model is

$$
V_t=\xi_0\exp\left(
\eta Y_t-\tfrac12\eta^2\operatorname{Var}(Y_t)
\right),
\quad
Y_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}dW_s^1,
$$

and

$$
dS_t=S_t\sqrt{V_t}\left(
\rho\,dW_t^1+\sqrt{1-\rho^2}\,dW_t^2
\right).
$$

The BLP \(\kappa=1\) hybrid scheme treats the singular nearest cell exactly
and earlier cells with deterministic power-kernel weights. The discrete Wick
compensator uses the variance of the implemented discrete convolution, giving
\(E[V_{t_k}]=\xi_0\) at the grid. Other \(\kappa\) values are rejected.

## 9. Jumps

For Bates/SVJJ, \(N_k\sim\operatorname{Poisson}(\lambda h)\). Conditional on
\(N_k=n\), the summed log-price jump is Gaussian with mean \(nm_J\) and
variance \(ns_J^2\). The summed SVJJ variance jump is Gamma with shape \(n\)
and rate equal to the reciprocal mean jump size. A Brownian-only control leaves
this independent law unchanged, so no jump likelihood is added. A future
jump-control path must add its likelihood before being enabled for claims.

## 10. Statistical and cost reporting

For independent contributions \(Y_i=G_iL_i\),

$$
\widehat{\operatorname{se}}(\widehat\mu_N)=\frac{s_Y}{\sqrt N}.
$$

Repeated reports include mean, empirical standard deviation, mean reported
standard error, relative bias, relative RMSE, bias z-score, and CI coverage.

With measured cost per path \(c\), online work-normalized VRF is

$$
\mathrm{VRF}_{\mathrm{work}}
=\frac{\sigma_{\mathrm{MC}}^2c_{\mathrm{MC}}}
       {\sigma_{\mathrm{IS}}^2c_{\mathrm{IS}}}.
$$

End-to-end cost adds training time amortized over held-out evaluation paths.
Event-frequency amplification is never reported as computational speedup.

## 11. Current exclusions

- Terminal events are supported.
- Barriers are discretely monitored unless a tested bridge correction is used.
- Multi-asset claims are outside the current scope.
- Jump-control claims are excluded until a compound likelihood exists.
- The current benchmark is risk-neutral Heston. Rough-volatility claims begin
  only after G2 is passed.
