# G4 exact path-integral mixture: implementation and falsification review

Date: 2026-07-15

Status: **exact-law implementation passed; neural-mixture core claim failed**

## 1. Executive decision

Plan v4 tested whether mode-specialized causal neural controllers combined through an
exact marginal path-mixture likelihood could outperform a single feedback controller
and practical rare-event baselines in two-tail rough-volatility estimation.

The Gaussian multimodal oracle and every finite-grid likelihood/replay correctness gate
passed. The final rBergomi rarity redesign also beat natural MC online. However, a
proper two-driver constant CEM mixture was materially more efficient than the neural
mixture on every development seed. The neural architecture claim is therefore stopped,
and a 20-seed neural confirmatory run is not performed.

## 2. Theoretical corrections made during implementation

### 2.1 Component-wise likelihood is not generally biased

If expert label `K` is drawn from the declared prior weight and retained, then

$$
\widehat p_{component}=M^{-1}\sum_m1_E(X_m)\frac{dP}{dQ_{K_m}}(X_m)
$$

is unbiased. It is different from the marginal balance-mixture estimator

$$
\widehat p_{balance}=M^{-1}\sum_m1_E(X_m)
\left[\sum_k\alpha_k\frac{dQ_k}{dP}(X_m)\right]^{-1}.
$$

The reason to use all-expert replay is variance stability, not exclusive unbiasedness.
Plan v3 and Plan v4 were corrected accordingly. In the final rBergomi experiment, the
component-weight estimator had roughly 35 times the variance of the neural balance
estimator and displayed severe finite-sample instability.

### 2.2 Mixture is not a larger universal path-law class

A mixture of absolutely continuous adapted Brownian drift laws has a posterior-weighted
effective adapted drift. Mixture value is therefore computational mode decomposition,
not a strict expressivity theorem over general Föllmer drifts. No such overclaim is
retained.

### 2.3 Null-mixture pathwise identity scope

`K=1` is required to match the existing controlled simulator pathwise. For `K>1`,
expert-wise batch partition changes RNG tensor ordering, so the correct requirement is
law/density/moment equality rather than bitwise path identity.

## 3. Gaussian G4-M0 oracle

Configuration: `configs/g4_gaussian_mixture_oracle.yaml`
Result: `results/g4_gaussian_mixture_oracle_development_2026-07-15.json`

For `T=1` and `E={|W_T|>=3}`:

- analytic probability: `0.0026997960632601866`
- best single constant drift: `8.61e-9`, numerically zero
- single second moment: `0.002699796063260186`
- optimal symmetric mixture absolute drift: `3.1548497436`
- mixture second moment: `3.1587692132e-5`
- second-moment improvement: `85.47x`
- analytic versus primitive log-density max error: `3.55e-15`
- one-million-path balance-estimator bias z: `-0.472`
- mixture likelihood normalization z: `0.512`

The component estimator remained theoretically unbiased but became practically
heavy-tailed at the aggressive optimum. This reproduced the expected multiple-IS
failure mechanism and justified balance weighting.

## 4. G4-M1 rBergomi exact-law implementation

Implemented components:

- lean stateless two-driver feedback controller;
- exact randomized expert sampling;
- target-coordinate replay of every required expert;
- stable `logsumexp` marginal density;
- component-wise and balance-mixture likelihoods;
- selected-control reuse optimization;
- mode-specific PI and PICE;
- off-policy weight-only J2;
- two-driver constant CEM baseline.

Correctness evidence:

- `K=1` simulator and mixture paths agree pathwise;
- replayed constant-control density agrees with the analytic density within `3e-14`;
- expert-label permutation leaves the marginal density invariant;
- both component and marginal likelihoods normalize;
- selected expert controls are exactly reused from the generated path;
- final maximum selected replay error: `1.07e-14`;
- all new focused tests passed before experiments.

The selected-control reuse optimization avoids replaying the generating expert. It
reduced neural-mixture inference cost without changing the marginal density.

## 5. Development sequence

### 5.1 Initial approximately 1% union event

At thresholds `55/131`, neural balance mixture reduced raw variance but its initial
all-expert replay cost made it worse than natural MC. Reusing selected controls improved
the cost, but the final natural-relative work ratio remained `0.926`, below the `1.10`
gate.

### 5.2 One allowed rarity redesign

The redesign used separate calibration seed `42001`, with 200,000 paths:

- empirical 0.1% quantile: `41.77649`;
- empirical 99.9% quantile: `139.34694`;
- frozen hard thresholds: `42/139`;
- new train and validation seeds;
- five validation seeds, 50,000 paths per method/seed.

Before adding the strong CEM baseline, the neural mixture achieved:

- raw variance `5.63e-4` versus natural `2.04e-3`;
- online work-VRF `1.19` versus natural;
- online work-VRF `1.97` versus single neural feedback;
- improvement versus single feedback on 5/5 seeds;
- 49.3%/50.7% left/right contribution balance;
- learned mixture weights `0.342/0.658`;
- break-even `19` queries versus the inferior single feedback;
- break-even about `205` queries versus natural MC.

This was an online development success but not yet a strong-baseline result.

## 6. Strong-baseline result

Result: `results/g4_rbergomi_mixture_rarity_redesign_v2_with_cem_2026-07-15.json`

| Method | Estimate | Single-path variance | Cost/path | Online work |
|---|---:|---:|---:|---:|
| Natural MC | 0.002040 | 2.036e-3 | 1.118e-5 | 2.269e-8 |
| Single neural feedback | 0.002144 | 2.170e-3 | 1.849e-5 | 4.022e-8 |
| Neural exact mixture | 0.002147 | 5.628e-4 | 3.396e-5 | 1.913e-8 |
| Constant CEM exact mixture | 0.002152 | 4.026e-4 | 2.402e-5 | 9.655e-9 |

The CEM mixture is better for both raw variance and cost. Seed-level comparison gives a
geometric neural/CEM disadvantage of approximately `2.12x`, with CEM winning all five
seeds. CEM training took about `1.55 s`; neural mixture training took about `20.37 s`.
The neural mixture has no break-even against CEM.

The left CEM expert reached the original convergence criterion in eight iterations.
The right expert reached the hard target once but not twice within eight iterations.
A separate 16-iteration audit using the same training seed converged in 12 iterations to
control `(0.16577, 3.06747)`. Thus the earlier convergence flag is a budget issue, not a
weak-baseline justification. The development configs now allow 12 iterations.

## 7. Technical interpretation

The neural mixture solved the multimodal coverage problem: both tails contributed and
balance weighting removed the component estimator's extreme variance. Its failure is
computational and approximation-related:

1. the terminal strangle is well described by two global drift directions;
2. state feedback adds limited variance reduction beyond mode selection;
3. neural inference and alternate-expert replay cost more than the residual gain;
4. training from zero is inferior to a likelihood-weighted CEM projection for this task.

The result does not imply that feedback is useless for all path-dependent tasks. It
does show that neural feedback is not justified for this declared terminal task and
cannot be used as the paper's central claim.

## 8. Publication and continuation decision

The exact mixture engine, measure audit, Gaussian restricted-family result, and negative
neural comparison are valid research assets. They are not sufficient for a renowned
journal because mixture IS and CEM are established and the proposed neural architecture
loses to the correct strong baseline.

The next falsifiable hypothesis, if pursued, must start from the winning CEM law rather
than replace it. A defensible candidate is a **CEM-anchored zero-initialized residual
feedback mixture**. It must reduce exactly to CEM at initialization and beat CEM after
including residual inference cost. If that single residual pilot fails, neural mixture
development should stop entirely and the project should pivot to theorem/application
work around exact controlled rBergomi rather than another architecture search.
