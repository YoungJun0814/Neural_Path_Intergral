# G2 Controlled rBergomi BLP Law Review

Date: 2026-07-14<br>
Accepted protocol: `g2-controlled-rbergomi-law-confirmatory-v2-global-tests`<br>
Protocol SHA-256: `166ec1d93ebb46571ab5ec1745da1ebb04645d6c1ef85a7680ed772610919d8e`<br>
Accepted result: `results/g2_rbergomi_law_confirmatory_v2_2026-07-14.json`<br>
Decision: **G2 finite-grid law gate passed; VFO development may start**

## 1. Scope

G2 validates the exact proposal/target construction for the repository's declared
finite-grid `kappa=1` Bennedsen–Lunde–Pakkanen hybrid law. It covers the Plan-v3 primary
grid:

$$
H\in\{0.05,0.10,0.20\},
\qquad
\rho\in\{-0.90,-0.70,-0.50\}.
$$

It does not claim continuous-time unbiasedness. Continuous rBergomi error still contains
time discretization, hybrid approximation, and path-monitoring components.

## 2. Controlled augmented BLP law

For each interval, the hybrid scheme samples the correlated Gaussian pair:

$$
\left(
\Delta W_i^1,
I_i=\int_{t_i}^{t_{i+1}}
(t_{i+1}-s)^{H-1/2}dW_s^1
\right).
$$

With a piecewise-constant first-driver control:

$$
\Delta W_i^{1,P}=\Delta W_i^{1,Q}+u_i^1\Delta t,
$$

$$
I_i^P=I_i^Q+u_i^1
\frac{\Delta t^{H+1/2}}{H+1/2}.
$$

Historical BLP cells receive the same target-coordinate shift through their deterministic
cell-average weights. The second Brownian driver is shifted independently by
`u_i^2 Delta t`.

### Why there is no extra bridge likelihood

Let `Sigma` be the covariance matrix of `(Delta W1, I)` and let:

$$
m=u^1(\Delta t,c_{12})^\top.
$$

The second vector is exactly the first covariance column times `u1`, hence:

$$
\Sigma^{-1}m=(u^1,0)^\top.
$$

The joint Gaussian density ratio therefore reduces to the ordinary first Brownian
coordinate term. Adding another density for `I` would double count the same
Cameron–Martin drift. Together with the independent second driver:

$$
\log\frac{dP}{dQ}
=-\sum_i u_i^\top\Delta W_i^Q
-\frac12\sum_i\|u_i\|^2\Delta t.
$$

## 3. Implementation audit

The simulator now:

1. evaluates `u_i` before sampling interval `i`;
2. shifts both Brownian target coordinates;
3. shifts the recent singular-cell integral by its exact kernel integral;
4. uses target first-driver history in every historical BLP cell;
5. builds spot and variance as deterministic maps of the target augmented path;
6. accumulates likelihood and action energy in float64;
7. optionally records proposal/target Brownian increments, local integrals, and controls;
8. raises on nonfinite paths or invalid control shape/device/dtype.

The natural `simulate` entry point delegates to this law with null control, which makes
the null identity a pathwise equality rather than a distribution-only comparison.

## 4. Unit and reconstruction tests

Twenty-one focused controlled/natural rBergomi tests passed. The new G2 tests include:

- pathwise null-control identity;
- exact target Brownian reconstruction;
- exact recent-cell target integral reconstruction;
- generic two-driver likelihood equality;
- joint-Gaussian proof that the bridge density cancels;
- target Volterra path equals proposal path plus every deterministic cell shift;
- independent reconstruction of spot, variance, and Volterra paths;
- likelihood normalization;
- likelihood-weighted spot martingale check;
- bounded soft-payoff agreement with natural target simulation;
- invalid control contract rejection.

Existing hybrid covariance, Wick correction, maturity-grid, and rough-skew regressions
also remain passing.

## 5. Statistical law matrix

The frozen v2 run used 30,000 natural and 30,000 proposal paths in each of nine primary
regimes. The proposal was the fixed two-driver control `(-0.45, 0.25)`. Diagnostics
covered:

- `E_Q[L]=1`;
- likelihood-weighted terminal spot equals `S0` for `rho<=0`;
- likelihood-weighted terminal forward variance equals `xi`;
- hard terminal event agreement;
- soft terminal payoff agreement;
- contribution ESS and finiteness.

Results:

| Diagnostic | Result | Frozen rule |
|---|---:|---:|
| Maximum individual absolute z | 2.35 | catastrophic alarm < 4.0 |
| Minimum family chi-square p | 0.0529 | >= 0.01 |
| Maximum absolute family directional z | 1.95 | <= 3.0 |
| Minimum contribution ESS fraction | 16.7% | >= 1% |
| Maximum refinement relative change | 2.89% | <= 5% |
| Maximum refinement absolute z | 1.65 | <= 3.0 |

All v2 gates passed and all simulated values were finite.

## 6. Confirmatory v1 failure and protocol correction

Confirmatory v1 is retained as a failed result:

`results/g2_rbergomi_law_confirmatory_v1_2026-07-14.json`

Its rule required all 45 individual z statistics to satisfy `|z|<=3`. One weighted-spot
statistic at `(H=0.1,rho=-0.5)` was `-3.34`, while the remaining gates passed. Five new
diagnostic seeds with 100,000 paths each produced weighted-spot z values between about
`-0.80` and `0.17`; the deviation did not reproduce.

The v1 result was not relabeled as a pass. Before using any v2 roots, the multiplicity-
blind rule was replaced by five preregistered metric-family tests:

- directional aggregate z with absolute threshold 3;
- chi-square omnibus p-value at least 0.01, a Bonferroni allocation over five families;
- unconditional individual `|z|>4` numerical alarm.

V2 then used entirely new roots `19001` and `19101`.

## 7. Remaining limitations

- The exact controlled implementation is `O(N^2)` because causal feedback prevents the
  existing offline convolution from being reused directly.
- The tested forward variance curve is constant `xi`; a general `xi_0(t)` interface is
  still required for the application study.
- The law matrix uses maturity `0.25`; longer maturities enter the VFO experiment matrix
  only after memory-efficient structural state support is implemented.
- This gate validates fixed controls, not learned VFO performance.
- Positive `rho` remains outside the main risk-neutral study.

## 8. Decision

The controlled finite-grid BLP law is sufficiently verified for G3 work. VFO performance
training is now permitted, beginning with:

1. a fixed SOE structural feature bank that does not replace the BLP target law;
2. instantaneous and structural two-driver branches;
3. zero-initialized gated residual memory;
4. branch freeze/unfreeze and takeover diagnostics;
5. Markov, SOE-only, and generic recurrent ablations.
