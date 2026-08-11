# G11 V15 probability-space and scaling contract

Date: 2026-08-11

Status: P1 implementation contract. Continuous-time identities are mathematical
targets; implemented exactness remains relative to the declared finite-grid law.

## 1. Continuous Gaussian-Volterra model

Let `W` and `B` be independent standard Brownian motions on a filtered probability
space satisfying the usual conditions. Let

```text
Y_t = int_0^t K(t,s) dW_s,
V_t = xi_0(t) exp(eta Y_t - 0.5 eta^2 Var(Y_t)),
dS_t/S_t = sqrt(V_t) [rho dW_t + sqrt(1-rho^2) dB_t],
```

where `rho in (-1,1)`, `xi_0` is strictly positive and the stochastic integrals are
Itô integrals. The fractional rBergomi kernel is a principal specialization, not the
definition of the full class.

For `X_T=log S_T`, conditioning on `sigma(W_s: s<=T)` gives

```text
m_T(W) = log S_0 + rho int_0^T sqrt(V_t) dW_t
         - 0.5 int_0^T V_t dt,
c_T(W) = (1-rho^2) int_0^T V_t dt,
X_T | W ~ Normal(m_T(W), c_T(W)).
```

For a left terminal threshold `K_S>0`,

```text
g(W) = P(S_T <= K_S | W)
     = Phi((log K_S - m_T(W))/sqrt(c_T(W))).
```

This is a conditional-law identity. It is not an exact simulation algorithm for the
continuous Volterra stochastic integrals.

## 2. Zero-variance conditional target

Let `P_W` be the law of `W`, `p_K=E[g(W)]`, and assume `0<p_K<infinity`. Define

```text
dPi*_K/dP_W = g/p_K.
```

For any proposal `Q` dominating `Pi*_K`, the ordinary-IS contribution is
`Y=g dP_W/dQ`. Direct calculation gives

```text
Var_Q(Y)/p_K^2 = chi-square(Pi*_K || Q).
```

The divergence direction is part of the contract. Reversing it is an error.

## 3. Small-noise family

The primary asymptotic track is a declared small-noise family, not an unproved use of
the published rBergomi small-time LDP. For `epsilon in (0,1]`, set

```text
Y_t^epsilon = sqrt(epsilon) int_0^t K(t,s) dW_s,
V_t^epsilon = xi_0(t)
              exp(eta Y_t^epsilon
                  - 0.5 eta^2 epsilon Var(Y_t)),
dS_t^epsilon/S_t^epsilon
  = sqrt(epsilon V_t^epsilon)
    [rho dW_t + sqrt(1-rho^2) dB_t].
```

At `epsilon=1` this is the original model. As `epsilon` decreases, both volatility
fluctuations and price diffusion vanish consistently, including the Itô correction.
For fixed `K_S<S_0`, the left-tail event is rare as `epsilon -> 0`.

Conditional on `W`,

```text
m_T^epsilon(W)
  = log S_0
    + sqrt(epsilon) rho int sqrt(V_t^epsilon) dW_t
    - 0.5 epsilon int V_t^epsilon dt,
c_T^epsilon(W)
  = epsilon (1-rho^2) int V_t^epsilon dt.
```

## 4. Contracted small-noise action

Let `u,v in L2[0,T]` be controls for `sqrt(epsilon)W` and
`sqrt(epsilon)B`. Define

```text
V_t^u = xi_0(t) exp(eta (K u)(t)),
A(u) = rho int_0^T sqrt(V_t^u) u_t dt,
I(u) = int_0^T V_t^u dt,
k = log(K_S/S_0) < 0.
```

The leading controlled terminal log return is

```text
x(u,v) = A(u)
         + sqrt(1-rho^2) int_0^T sqrt(V_t^u) v_t dt.
```

For fixed `u`, minimizing `0.5||v||_2^2` subject to `x(u,v)<=k` gives

```text
C_k(u) = ((A(u)-k)_+)^2 / (2(1-rho^2) I(u)).
```

Thus the contracted candidate rate action is

```text
J_k(u) = 0.5||u||_2^2 + C_k(u).
```

This reduction follows from a Hilbert-space projection of the independent price
control. A full continuous-time LDP/Laplace-principle proof still requires
exponential tightness, continuity/localization for the exponential Volterra map and
boundary regularity. The corresponding fixed-BLP-grid statement is proved in
`G11_V16_FIXED_GRID_SMALL_NOISE_THEOREMS.md`; the continuum formula remains a
theorem target.

The finite-epsilon conditional action used by the optimizer is

```text
J_epsilon(h)
  = 0.5||h||^2 - epsilon log g_epsilon(h/sqrt(epsilon)).
```

The expected limit is `J_k(h)`. Convergence is not assumed; it must be proved or
falsified numerically and analytically.

## 5. Published small-time LDP boundary

Jacquier--Pakkanen--Stone use `alpha in (-1/2,0)`, `beta=2alpha+1=2H`, and a
specific rescaled process `X_t^epsilon=epsilon^beta X_{epsilon t}` with speed
`epsilon^(-beta)`. Their paper explicitly treats a rescaled process and notes further
conditions for financial applications. V15 does not infer fixed-strike small-time IS
efficiency from that result.

Any future small-time theorem must independently align event scaling, topology, rate
function and martingale assumptions. T16-4 now matches the declared small-noise family
to Gulisashvili's Theorem 13 by setting that paper's scaling exponent (not the roughness
parameter) to `1/2` and proving exponential equivalence of the deterministic Wick
correction. This authorizes the constant-`xi`, fixed-terminal-left-tail probability
exponent only; it does not authorize a small-time or continuous-proposal-efficiency
claim.

## 6. Finite-grid implementation law

The primary implementation uses the existing BLP FFT rBergomi law. At `N` steps it
is the image of `2N` independent standard Gaussian local-cell variables and `N`
independent price variables. V15 conditions on all `2N` local variables and integrates
the `N` price variables for terminal claims.

For small-noise evaluation at a local standard-normal coordinate `z`:

1. simulate the BLP local law at `sqrt(epsilon) z`;
2. multiply the finite-grid Wick compensator by `epsilon` and recover the integrated
   variance `I_N`;
3. replace the simulator's `-0.5 I_N` drift by `-0.5 epsilon I_N`;
4. use conditional variance `epsilon(1-rho^2)I_N`.

At `epsilon=1` this must reproduce V14 pathwise. The BLP auxiliary local coordinate
does not automatically identify with a continuum Cameron--Martin control. That mesh
identification is T15-7, not an implementation assumption.

V15 small-noise development results produced before the Wick-compensator correction
are invalid for asymptotic evidence and are explicitly quarantined by the V16 result
status manifest.

## 7. Excluded cases

- `rho=+/-1`: conditional variance degenerates;
- nonpositive forward variance;
- continuous barriers and occupation events;
- self-normalized importance sampling;
- data-dependent proposal updates using final inference samples;
- unrestricted infinite-dimensional covariance changes;
- fixed-strike small-time claims without a matching theorem.

## 8. P1 decision

The probability space and small-noise regime are internally consistent and authorize
P2 finite-grid oracle implementation. Continuous-time LDP efficiency, mesh-uniform
rates and novelty remain open. Qualification remains locked by the P0 external-review
requirement and later theorem gates.
