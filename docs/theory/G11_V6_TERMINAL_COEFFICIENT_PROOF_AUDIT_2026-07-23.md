# G11 V6 Terminal Coefficient Proof Audit

Date: 2026-07-23

Scope: terminal rBergomi DCS only; barriers are explicitly excluded

## 1. Verdict

The current terminal argument is a mathematically coherent **proof candidate**, not
yet a journal-ready theorem. No contradiction was found in the inverse-slope,
ratio-localization, defensive-likelihood, or MLMC exponent algebra. The remaining
risk is concentrated in the coefficient-rate step: the proof document states the
right decomposition and rate mechanism, but a journal submission still needs a
line-by-line continuous/discrete coupling lemma with all sigma-fields, indices,
uniform constants, and proposal shifts written explicitly.

Accordingly:

- Route B has not failed and does not need to be abandoned;
- the executable contract may report conservative candidate exponents;
- it must report `journal_claim_ready=false`;
- empirical slopes may only falsify, never prove, the theorem; and
- no terminal rate may be transferred to a barrier event.

## 2. Audited mathematical object

For the strict, unfloored rBergomi target law,

`V_t = xi exp(eta Y_t - eta^2 Var(Y_t)/2)`,

the independent price-driver vector is decomposed along a positive unit direction
`u^h`. Conditional on the volatility driver, mixture label, and the price residual
orthogonal to `u^h`,

`log(S_T^h) = A_h + B_h Z`,

where `Z` is one standard normal coordinate. The adjacent coarse grid uses pair
sums of the fine direction, not a separately normalized direction. This preserves
the same `Z` and is the convention implemented in the coupling and smoothing code.

The proposed theorem chain is

`BLP Volterra rate -> lognormal-volatility rate -> (A,B) rate -> threshold rate`

`-> DCS second-moment rate -> weak bias + cost -> MLMC complexity`.

## 3. Obligation-by-obligation audit

| Obligation | Status | Audit finding |
|---|---|---|
| Exact BLP grid law | pass | Direct and FFT paths share the local Gaussian covariance and historical cell-average coefficients. |
| Exact adjacent marginals | pass at code-contract level | The coarse recent-cell integral is not formed by naively summing fine local integrals; the joint Gaussian construction supplies the required marginal. |
| Standard rBergomi target | pass after correction | The fixed `1e-10` variance floor was removed. Non-normal-range values now fail the numerical run instead of silently changing the model. |
| Positive common direction | pass | The direction is positive, unit normalized, grid aligned, and coarse weights are fine pair sums. |
| Uniform inverse slope moments | pass | Weighted Jensen plus the Gaussian MGF yields the displayed finite bound when the grid-scaled L1 mass is bounded below. |
| Defensive likelihood moment | pass | A natural component of mass `delta>0` gives `dP/dQ <= 1/delta`; no self-normalization is used. |
| Coefficient rate | proof candidate | The `O(h^r)`, `r<H`, mechanism is plausible and consistent with the known strong rough-volatility bottleneck, but the current proof is still compressed at the stochastic-integral and common-filtration steps. |
| Threshold ratio rate | pass conditional on coefficient rate | The deterministic ratio identity plus inverse-slope moments is valid. |
| DCS second moment | pass conditional on threshold rate | Global Lipschitz continuity of `Phi` and the defensive likelihood bound give exponent `2r`. |
| Continuous weak bias | proof candidate | Requires the continuous price/Volterra coupling and limiting direction to be stated as a separate lemma. |
| FFT-MLMC algebra | pass conditional on rates | With `alpha=r`, `beta=2r`, and polynomial `gamma=1`, the exponent is `1/r`; the FFT logarithm remains explicit. |
| Barrier extension | not proved | Active-time localization and fine-only monitoring-point enrichment are separate terms. |

## 4. Exact remaining proof obligations

### O1. Common probability space and filtration

Define one Brownian pair `(W1,W2)` and construct the continuous model, fine BLP
model, and coarse BLP model on it. State which variables enter the conditional
sigma-field and prove that the common coordinate `Z` remains independent standard
normal after conditioning. Include deterministic proposal shifts and mixture labels
without changing the target-law coefficient identity.

### O2. Implementation-specific Volterra estimate

For the exact `kappa=1` cell-average coefficients used in
`physics_engine.py` and `rbergomi_fft.py`, prove

`sup_i ||Y(t_i)-Y_i^h||_Lp <= C_p h^H`

and the adjacent version. The proof must explicitly separate the exact singular cell
from historical projection cells. If uniformity over a compact Hurst interval is
claimed, the constant and the admissible exponent margin must be uniform too.

### O3. Lognormal and price-integral transfer

Write a lemma giving uniform finite moments of `V^h`, `sqrt(V^h)`, and their
fine/coarse differences. Then apply BDG to the two price stochastic integrals and a
deterministic inequality to the integrated-variance drift. State the exact left-point
index convention used by the simulator.

### O4. Affine coefficient decomposition

Define `A_h` and `B_h` directly from the code-level residual projection, not only
through `log(S_T^h)-B_h Z`. Verify measurability with respect to the conditional
sigma-field and show that the fine/coarse pair uses the identical `Z`.

### O5. Continuous terminal weak bias

Define a positive unit direction in `L2([0,T])` whose cell discretizations are the
implemented directions. Prove convergence of both affine coefficients to their
continuous counterparts before applying ratio localization. Without this lemma, the
fixed-grid correction theorem does not by itself imply continuous-time weak bias.

### O6. External review record

Before a journal theorem claim, obtain an independent stochastic-analysis review
that records either:

- every obligation O1--O5 as discharged; or
- the exact assumptions under which the statement must remain conditional.

## 5. Conservative candidate exponents

With the configured margin `epsilon=0.01`, `r=H-epsilon`:

| H | r | candidate beta=2r | candidate FFT-MLMC polynomial exponent 1/r |
|---:|---:|---:|---:|
| 0.05 | 0.04 | 0.08 | 25.000 |
| 0.12 | 0.11 | 0.22 | 9.091 |
| 0.30 | 0.29 | 0.58 | 3.448 |

These are upper-bound contracts, not fitted equalities. They are deliberately
unfavorable at very small H and do not support a canonical `O(epsilon^-2)` claim.

## 6. Full provisional falsification result

The strict-lognormal full diagnostic used:

- 3 Hurst values;
- terminal and finite-grid barrier tasks;
- 6 adjacent levels;
- 12 independent replicates per level; and
- 8,192 paths per replicate.

It produced 432 complete records. Every implementation gate passed:

- direction geometry;
- pathwise decomposition exactness;
- finite empirical inverse moments;
- finite analytic inverse-moment bounds;
- valid terminal rate contracts; and
- no failed diagnostic record.

The run is not formal because the worktree was dirty. Its current artifact is:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_P0_artifacts\g11_v6_theory_diagnostics_full_strict_provisional_v2_2026-07-23.json`.

The terminal fitted DCS-variance exponents were approximately `0.1513`, `0.2225`,
and `0.6121` for `H=0.05`, `0.12`, and `0.30`, respectively. They do not falsify the
conservative candidate exponents above. The `H=0.30` barrier rate had no common
stable raw/DCS window; this reinforces, rather than relaxes, the barrier exclusion.

## 7. Allowed and prohibited claims

Allowed now:

- “A proof candidate and executable theorem contract were developed.”
- “The full provisional diagnostics did not falsify the terminal candidate.”
- “The inverse terminal-slope moment bound is analytic.”
- “Barrier rates remain open.”

Prohibited now:

- “The terminal theorem is independently established.”
- “The empirical slope proves the asymptotic rate.”
- “DCS restores canonical MLMC complexity in rough Bergomi.”
- “The terminal theorem also covers barriers.”
- “The dirty provisional artifact is confirmatory evidence.”
