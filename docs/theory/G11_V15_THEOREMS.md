# G11 V15 theorem ledger and proof boundary

Date: 2026-08-11

This document distinguishes exact finite-dimensional statements from continuum and
rare-event asymptotic claims.  Numerical evidence cannot change a theorem status.
The machine-readable source of status is
`configs/g11_v15/theorem_ledger_v1.yaml`.

## 1. Setting

Let `W` and `B` be independent Brownian motions and let `V(W)` be positive,
progressively measurable, with `int_0^T V_t dt` finite almost surely.  For
`|rho| < 1`, define

```text
log S_T = log S_0 + rho int_0^T sqrt(V_t) dW_t
          + sqrt(1-rho^2) int_0^T sqrt(V_t) dB_t
          - 0.5 int_0^T V_t dt.
```

The implemented finite-grid reference is the BLP exact-covariance construction
declared in `G11_V15_PROBABILITY_AND_SCALING_CONTRACT.md`.  A statement about that
grid is not silently promoted to a statement about the continuous model.

## 2. T15-1 — conditional representation

**Status: proved under the assumptions above; finite-grid implementation
oracle-tested.**

Conditionally on `sigma(W_s: s <= T)`, the independent Ito integral with respect to
`B` is centered Gaussian with variance `int V dt`.  Consequently

```text
m(W) = log S_0 + rho int sqrt(V) dW - 0.5 int V dt,
c(W) = (1-rho^2) int V dt,
P(S_T <= K | W) = Phi((log K - m(W))/sqrt(c(W))).
```

The put and call formulas follow by integrating the lognormal conditional law.  At
`rho = +/-1`, `c=0` and the formula must be treated as a degenerate limit; V15
excludes those endpoints.  For the explicit small-noise family, every stochastic
price term and compensator has the epsilon scaling declared in the probability
contract.

## 3. T15-2 — exact likelihood, unbiasedness, and Rao–Blackwell ordering

**Status: proved for every frozen finite-dimensional proposal; the conditional
identity itself also holds in continuous time whenever the likelihood is
well-defined and square-integrable.**

Let `P_W` be the driver reference law, `Q_W` a frozen proposal with `P_W << Q_W`,
and `L=dP_W/dQ_W`.  If `g(W)=P(S_T<=K|W)`, then

```text
E_Q[g(W)L(W)] = E_P[g(W)] = P(S_T<=K).
```

If the independent price driver is also sampled, the hard contribution is
`1_{S_T<=K}L(W)`.  Under `Q`, conditioning this contribution on `W` gives
`g(W)L(W)`.  The law of total variance therefore proves

```text
Var_Q(gL) <= Var_Q(1_event L).
```

Training and proposal selection must finish before the evaluation seed is drawn.
The theorem does not authorize reuse of evaluation outcomes to alter the proposal.

For a finite-rank component with covariance
`Sigma=I+U diag(lambda-1) U^T`, orthonormal `U`, and every `lambda_i>0`, the exact
component ratio is

```text
log(q_j/p)(x)
 = -0.5 sum_i log(lambda_i)
   -0.5 (x-mu)^T Sigma^{-1}(x-mu) + 0.5 x^T x,
Sigma^{-1}=I+U diag(1/lambda_i-1)U^T.
```

The balance-mixture denominator is the log-sum-exp over every component, not merely
the sampled component.

## 4. T15-3 — zero-variance target and divergence direction

**Status: proved.**

For `P_K=E_P[g]>0`, define `dPi*/dP=g/P_K`.  Then the ordinary conditional IS
relative variance satisfies the identity

```text
Var_Q(gL)/P_K^2 = chi_square(Pi* || Q).
```

The direction is `Pi* || Q`; reversing it is incorrect.  The identity is a
population characterization, not a directly normalized training loss, because
`P_K` is unknown in the target problem.

## 5. T15-4 — defensive robustness

**Status: proved; deliberately non-asymptotic and rarity-dependent.**

For

```text
Q = delta P + (1-delta) R,  0 < delta < 1,
```

we have `q >= delta p`, hence `L <= 1/delta` pathwise.  For `0<=g<=1`,

```text
E_Q[(gL)^2] = E_P[g^2 L] <= E_P[g^2]/delta <= P_K/delta,
Var_Q(gL)/P_K^2 <= 1/(delta P_K)-1.
```

This protects exactness and moments against missed modes, neural error, or clipped
curvature.  It does **not** by itself prove rare-event efficiency because the upper
bound deteriorates as `P_K` tends to zero.

## 6. T15-5 — small-noise rare-event efficiency

**Status: open.  Gate G5 is not passed.**

The candidate reduced action is

```text
J_k(u) = 0.5 ||u||_H^2
         + ((A(u)-k)_+)^2 / (2(1-rho^2) I(u)),
A(u)=rho int sqrt(V^u) u dt,
I(u)=int V^u dt,
V^u=xi exp(eta K_H u).
```

A journal theorem still requires:

1. an LDP for the exact declared small-noise family in a topology where all maps are
   measurable and sufficiently continuous, with localization for stochastic
   integrals;
2. exponential equivalence or a direct proof for the chosen discretization;
3. compactness/existence and coverage of every dominating minimizer;
4. a second-moment upper bound for the multimode Laplace proposal;
5. a counterexample analysis showing what happens when a mode is omitted.

The published rough-Bergomi small-time scaling is not substituted for this proof.
Until these obligations are discharged, the phrases “logarithmically efficient,”
“bounded relative error,” and “asymptotically optimal” are prohibited.

## 7. T15-6 to T15-9

| ID | Subject | Status | Required closure |
|---|---|---|---|
| T15-6 | finite-grid weak convergence/rate | open | model-specific bias theorem with explicit `H` domain |
| T15-7 | transport mesh consistency | open | set/energy convergence of minimizers and subspaces |
| T15-8 | end-to-end complexity | open | combine bias, sampling variance, solve, and inference work |
| T15-9 | operator-assisted certification | conditional | deterministic correction and fallback are exact; useful cost/quality bound is open |

Empirical mesh studies may falsify candidate rates, choose numerical resolution, and
quantify observed cost.  They cannot mark T15-6 or T15-7 as proved.

## 8. Gate conclusion

T15-1 through T15-4 have complete proof chains and executable finite-dimensional
oracles.  T15-5 is open, so the current result is not yet authorized to use a
`Mathematical Finance` or `Finance and Stochastics` asymptotic-efficiency headline.
Implementation may continue because the remaining numerical and operator work is
valuable under the narrower “exact finite-grid conditional transport” claim.
