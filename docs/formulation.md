# Unified Mathematical Formulation

This document is the **single source of truth** for the stochastic dynamics,
measure changes, and loss functions used throughout the codebase. README, paper
drafts, and code comments must match what is written here.

---

## 1. Base (P-measure) dynamics

The underlying market is modeled by a two-factor Heston-type SDE on a
probability space $(\Omega,\mathcal F,\mathbb P)$:

$$
\begin{aligned}
dS_t &= \mu(S_t,v_t,t)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{\mathbb P,S}, \\
dv_t &= \kappa\!\left(\theta - v_t\right)dt + \xi\sqrt{v_t}\,dW_t^{\mathbb P,v}, \\
d\langle W^{\mathbb P,S}, W^{\mathbb P,v}\rangle_t &= \rho\,dt, \qquad \rho\in(-1,1).
\end{aligned}
$$

The noise is decomposed as

$$
dW_t^{\mathbb P,v}=\rho\,dW_t^{\mathbb P,S}+\sqrt{1-\rho^2}\,dW_t^{\mathbb P,\perp},\qquad W^{\mathbb P,S}\perp W^{\mathbb P,\perp}.
$$

When jumps are enabled (`model_type ∈ {bates, svjj}`), the price SDE picks up
an additional compensated jump term:

$$
dS_t^{\text{jump}} = S_{t^-}\!\left[(e^{J_t}-1)\,dN_t - \lambda\kappa_J\,dt\right],
\qquad J_t\sim\mathcal N(m_J,s_J^2),\qquad N_t\sim\mathrm{Poisson}(\lambda t),
$$

with $\kappa_J = e^{m_J+\tfrac12 s_J^2}-1$. For SVJJ the variance process also
receives exponential upward jumps at the same Poisson times.

### Discretization (code convention)

We use the **Full-Truncation Euler** scheme (Lord–Koekkoek–van Dijk 2010),
which is the standard admissible Euler scheme for Heston:

$$
\begin{aligned}
v_{t+\Delta t} &= v_t + \kappa\!\left(\theta - v_t^+\right)\Delta t + \xi\sqrt{v_t^+}\,\Delta W_t^v,\\
S_{t+\Delta t} &= S_t + \mu\,S_t\,\Delta t + \sqrt{v_t^+}\,S_t\,\Delta W_t^S,
\end{aligned}
\qquad v_t^+:=\max(v_t,0).
$$

Both drift and diffusion coefficients use $v_t^+$ consistently (this was not the
case prior to Phase 1). For rBergomi we use the **hybrid scheme** of
Bennedsen–Lunde–Pakkanen (2017); see `src/physics_engine.py::RBergomiSimulator`.

### Jumps

Jump counts per step are drawn from $\mathrm{Poisson}(\lambda \Delta t)$
(not $\mathrm{Bernoulli}(\lambda \Delta t)$ as before), supporting multiple
jumps in the same interval.

---

## 2. Girsanov change of measure

### 2.1 Control and measure $\mathbb Q$

We introduce a bounded measurable control
$u:[0,T]\times\mathbb R_{>0}^2\to\mathbb R$ and define a new measure $\mathbb Q$
via the Radon–Nikodym density

$$
\frac{d\mathbb P}{d\mathbb Q}
=\mathcal E_T(u)
:=\exp\!\left(-\int_0^T u_t\,dW_t^{\mathbb Q,S} - \tfrac12\int_0^T u_t^{\,2}\,dt\right),
$$

where $W_t^{\mathbb Q,S}$ is a standard $\mathbb Q$-Brownian motion such that

$$
dW_t^{\mathbb P,S} = dW_t^{\mathbb Q,S} + u_t\,dt.
$$

We apply the control **only to the price driver** $W^{\mathbb P,S}$. The
orthogonal component $W^{\mathbb P,\perp}$ is not shifted.

### 2.2 Dynamics under $\mathbb Q$

Substituting the shift and rewriting:

$$
\begin{aligned}
dS_t &= \bigl(\mu + \sqrt{v_t}\,u_t\bigr)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{\mathbb Q,S},\\[2pt]
dv_t &= \bigl[\kappa(\theta-v_t) + \rho\,\xi\sqrt{v_t}\,u_t\bigr]\,dt
        + \xi\sqrt{v_t}\,\bigl(\rho\,dW_t^{\mathbb Q,S} + \sqrt{1-\rho^2}\,dW_t^{\mathbb Q,\perp}\bigr).
\end{aligned}
$$

The **v-process drift picks up $\rho\xi\sqrt{v_t}u_t$ under $\mathbb Q$** whenever
$\rho\neq 0$. This correction was omitted in pre-Phase-1 code and is the main
source of bias that Phase 1.2 fixes.

### 2.3 Importance-sampling estimator

For any integrable functional $F(S_{[0,T]},v_{[0,T]})$,

$$
\mathbb E^{\mathbb P}[F]
= \mathbb E^{\mathbb Q}\!\left[F\cdot\mathcal E_T(u)\right]
= \mathbb E^{\mathbb Q}\!\left[F\cdot\exp\!\bigl(-\textstyle\int u\,dW^{\mathbb Q,S}-\tfrac12\int u^2\,dt\bigr)\right].
$$

When we simulate under $\mathbb Q$ (i.e. the controlled dynamics in 2.2), the
log-weight accumulated along a path is

$$
\log\mathcal E_T^{(i)} = -\sum_k u_{t_k}^{(i)}\,z_{1,k}^{(i)}\sqrt{\Delta t} - \tfrac12\sum_k \bigl(u_{t_k}^{(i)}\bigr)^2\Delta t,
$$

with $z_{1,k}^{(i)}\sim\mathcal N(0,1)$ the **$\mathbb Q$-Brownian** increments
for asset $S$ used during the simulation (this is what the code samples).

---

## 3. Training objectives

### 3.1 Variance-minimizing importance sampler (price or prob estimator)

Given target functional $g$,

$$
\mathcal L_{\text{VM}}(u) = \mathbb E^{\mathbb Q}\!\left[\bigl(g(S_T)\cdot\mathcal E_T(u)\bigr)^2\right].
$$

This is the classical IS objective (Asmussen & Glynn 2007). Minimizing
$\mathcal L_{\text{VM}}$ also minimizes $\mathrm{Var}^{\mathbb Q}[g\mathcal E_T]$
because the mean is $\mathbb E^{\mathbb P}[g]$ which is independent of $u$.

### 3.2 KL-regularized crash generation

$$
\mathcal L_{\text{KL}}(u)
= -\mathbb E^{\mathbb Q}[\,\log\mathbf 1_{S_T<K}\,]
+ \lambda\cdot\mathrm{KL}(\mathbb Q\,\|\,\mathbb P),
\qquad
\mathrm{KL}(\mathbb Q\|\mathbb P)=\mathbb E^{\mathbb Q}\!\left[\tfrac12\!\int u^2\,dt\right].
$$

The indicator is replaced by a smooth surrogate (softplus hinge) for
gradients. $\lambda$ trades off between "more crashes" and "less deviation from
$\mathbb P$".

### 3.3 Cross-entropy method (CEM)

Pilot sampling + IS weight update per Rubinstein–Kroese (2004). Included as
benchmark only.

### 3.4 Distribution matching for the base NeuralSDE

For training the P-dynamics (before any control), we match empirical
moments or a proper divergence to the real return series:

$$
\mathcal L_{\text{MM}} = \sum_{k=1}^{4} w_k\bigl(m_k^{\text{model}} - m_k^{\text{data}}\bigr)^2,
\qquad \mathcal L_{\text{MMD}} = \|\mu_{\text{model}}-\mu_{\text{data}}\|_{\mathcal H_k}^2.
$$

The moment targets are **computed from the training data**, not hard-coded.

---

## 4. Unbiasedness check (Phase 2.2)

For any fixed $u$, a Monte-Carlo estimate of $\mathbb E^{\mathbb P}[F]$ obtained
from $N$ controlled paths should converge to the un-controlled MC estimate
(from $M$ natural paths) up to statistical noise:

$$
\widehat I_{\text{IS}} - \widehat I_{\text{MC}}
\in
\bigl[-3\sqrt{\hat s_{\text{IS}}^2/N + \hat s_{\text{MC}}^2/M},\;
+3\sqrt{\hat s_{\text{IS}}^2/N + \hat s_{\text{MC}}^2/M}\bigr]
$$

with probability 99.7% under CLT. `tests/test_girsanov_unbiased.py` encodes
exactly this check and must pass before any efficiency claim is advertised.

---

## 5. Variance Reduction Factor (VRF)

The published efficiency gain is the **work-normalized variance reduction**:

$$
\mathrm{VRF} := \frac{\mathrm{Var}_{\mathbb P}[F]/\mathrm{cost}_{\mathbb P}}
                    {\mathrm{Var}_{\mathbb Q}[F\mathcal E_T]/\mathrm{cost}_{\mathbb Q}}.
$$

Effective sample size is reported alongside:

$$
\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}, \qquad w_i = \mathcal E_T^{(i)}.
$$

Both metrics replace the previous "crash-rate ratio" quote.

---

## 6. Notation cheat sheet

| Symbol | Meaning |
|---|---|
| $S_t, v_t$ | Asset price, variance |
| $\mu, \kappa, \theta, \xi, \rho$ | Heston params |
| $\lambda, m_J, s_J$ | Jump intensity, log-jump mean/std |
| $u_t$ | Control (drift shift on $\sqrt v\,dW^S$) |
| $\mathcal E_T$ | Doleans exponential / Radon–Nikodym density |
| $H$ | Hurst exponent (rBergomi) |
| $\eta, \xi$ | rBergomi vol-of-vol and forward-variance |
