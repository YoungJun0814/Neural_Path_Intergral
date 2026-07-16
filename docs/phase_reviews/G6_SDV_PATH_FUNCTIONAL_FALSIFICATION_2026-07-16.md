# G6 SDV path-functional implementation and falsification report

Date: 2026-07-16

Active plan: `PATH_INTEGRAL_RESEARCH_PLAN_V6.md`

Decision: **G0 passed; G3 failed; G4 was not opened**

## 1. Executive decision

The selected first-ranked direction was implemented through its pre-registered
development stop gate. The result is technically valid but scientifically
negative:

- the Gaussian conditional-moment oracle passed;
- the finite-grid barrier-hit plus occupation event engine passed;
- four-segment two-driver CEM decisively beat constant CEM;
- SDV preserved exact likelihoods and unbiased probability estimates;
- SDV did **not** reduce variance relative to the four-segment CEM anchor and
  was more expensive online.

Consequently, the present SDV model is not a prestigious-journal submission
result. Running the untouched 20-seed confirmatory experiment after this
development failure would be statistically and procedurally unjustified.

## 2. What was implemented

### 2.1 Gaussian path-functional oracle (M0)

`src/path_integral/gaussian_excursion_oracle.py` implements a one-dimensional
Gaussian random walk augmented by:

- a barrier-hit flag;
- a right-endpoint stress-occupation count;
- backward dynamic programming for the conditional event probability;
- the conditional Brownian-moment projection

  \[
  u_i=\frac{E[G\,\Delta W_i\mid\mathcal F_i]}
  {\Delta t\,E[G\mid\mathcal F_i]};
  \]

- bounded adaptive simulation with the exact discrete Gaussian likelihood.

The implementation uses event-aligned Gaussian transition bins rather than
Gauss--Hermite nodes. This is an intentional numerical deviation from the
planning document: placing both discontinuity thresholds on bin edges avoids
quadrature instability at the hard indicators. It remains an explicitly
labelled finite-state spatial approximation; it is not presented as an exact
continuous-space solution.

### 2.2 Path-functional engine (M1)

`src/path_integral/path_functionals.py` fixes the primary event as

\[
A=\left\{\min_{1\le i\le N}S_i\le30,
\quad \Delta t\sum_{i=1}^N1_{\{S_i\le60\}}\ge0.30\right\}.
\]

The implementation includes the hard event, a bounded soft training payoff,
a CEM level score whose nonnegative set equals the hard event, and causal
prefix states. Occupation is counted at right endpoints only.

### 2.3 Strong time-piecewise baseline (M2/G0)

`src/training/rbergomi_piecewise_cem.py` fits equal-duration two-driver
Gaussian mean shifts. For segment `j`, the likelihood-weighted MLE is

\[
\widehat u_j=
\frac{\sum_n\widetilde w_n\sum_{i\in I_j}\Delta W^P_{n,i}}
{|I_j|\Delta t}.
\]

The hard threshold was calibrated once under the natural law to an observed
probability of `0.0036333` and then frozen. It was not redesigned after any
performance result.

### 2.4 Spectral Doob--Volterra controller (M3/G3)

`src/path_integral/controllers/sdv.py` implements:

- causal SOE modes updated only after the current control is evaluated;
- time, log spot, log variance, exact BLP Volterra state, running minimum,
  occupation, hit flag, and SOE features;
- a positive bounded desirability head;
- a bounded residual control around the frozen four-segment CEM anchor;
- zero residual-head initialization;
- exact state reset and causal target-path replay.

The SOE approximation is a controller feature only. The simulator still uses
the validated BLP rBergomi grid law, so SOE kernel error cannot bias the target
probability estimator.

`src/training/sdv.py` implements exact defensive-mixture data generation and
the two finite-grid regressions:

\[
\mathcal L_h=E_{Q_{beh}}
\left[L_{beh}\sum_i(\widehat h_i-G_{soft})^2\right],
\]

\[
\mathcal L_m=E_{Q_{beh}}\left[L_{beh}\sum_i
\left\|\sqrt{\Delta t}\widehat h_i\widehat u_i
-G_{soft}\frac{\Delta W_i^P}{\sqrt{\Delta t}}\right\|^2\right].
\]

Self-normalization is used only to estimate the training regression risk. Every
reported hard-event probability uses the ordinary, non-self-normalized
importance-sampling contribution.

## 3. Theoretical and technical audit

### 3.1 Conditions that passed

1. **Adaptedness.** Control at step `i` sees only the state through `t_i`.
   Suffix perturbations do not change earlier SDV outputs.
2. **Exact component likelihood.** For proposal increment `dW^Q` and bounded
   pre-increment control `u_i`, the implementation uses

   \[
   \log\frac{dP}{dQ}
   =-\sum_i u_i\cdot\Delta W_i^Q
   -\frac12\sum_i\|u_i\|^2\Delta t.
   \]

3. **Exact randomized-mixture likelihood.** All experts are replayed on the
   same canonical target coordinates before the marginal log-sum-exp density
   is evaluated.
4. **Target-law separation.** Neither SOE modes nor the desirability network
   replaces the rBergomi BLP simulator.
5. **Event convention.** Hard evaluation, soft training payoff, CEM score, and
   causal occupation state share one right-endpoint convention.
6. **No final self-normalization.** Final estimates are sample means of
   `1_A * dP/dQ`.
7. **Matched work.** Controller inference time is included in online work;
   training time is included in break-even.

Maximum full-run likelihood/replay discrepancy was
`5.33e-15`. Initial SDV and anchor controls differed by at most `1.11e-16`.

### 3.2 Claims that are deliberately not made

- no continuous-time hard-event Doob theorem;
- no global optimality theorem for the neural control;
- no claim that the SOE lift changes or exactly represents the target law;
- no learned variance bound;
- no quantum advantage;
- no novelty claim for “spectral plus Doob” by itself.

### 3.3 Remaining limitations, not correctness bugs

- The M0 reference has spatial-grid error, explicitly measured by refinement.
- The rBergomi result is for the declared finite monitoring grid, not a
  continuously monitored event.
- CEM and SDV optimize finite-sample training risks; neither is guaranteed to
  find a global variance optimum.
- A very strong deterministic four-segment anchor leaves little residual
  variance for the neural feedback model to remove.

## 4. Frozen results

### 4.1 M0 Gaussian oracle: pass

Source: `results/m0_gaussian_excursion_frozen_2026-07-16.json`

| Metric | Result |
|---|---:|
| Reference probability, 101 bins | 0.0344848 |
| Reference probability, 201 bins | 0.0335430 |
| Reference probability, 401 bins | 0.0333203 |
| Oracle/reference z-score | -0.237 |
| Second-moment VRF vs selected constant tilt | 3.612 |
| Decision | PASS |

### 4.2 G0 piecewise CEM: pass

Source: `results/g0_piecewise_cem_frozen_2026-07-16.json`

| Metric | Result | Required |
|---|---:|---:|
| Geometric raw VRF vs constant CEM | 2.909 | > 1.25 |
| Geometric work VRF | 2.845 | > 1.10 |
| Improving seeds | 5/5 | >= 4/5 |
| Probability difference z vs natural | 0.723 | abs(z) <= 3 |
| Incremental break-even | 2 queries | <= 50 |
| Decision | PASS | PASS |

The frozen four-segment anchor is

```text
[(3.2870549, -0.6817602),
 (2.8734573, -0.7801504),
 (2.1761513, -0.8173875),
 (0.8197942, -0.3603176)]
```

### 4.3 G3 SDV: fail

Authoritative source:
`results/g3_sdv_frozen_fair_timing_2026-07-16.json`

| Metric | Result | Required |
|---|---:|---:|
| Geometric raw VRF vs piecewise CEM | 0.9978 | > 1.35 |
| Geometric work VRF | 0.4512 | > 1.25 |
| Improving seeds | 0/5 | >= 4/5 |
| Probability difference z vs natural | 0.933 | abs(z) <= 3 |
| Exact replay/likelihood | 5.33e-15 | <= 2e-12 |
| Incremental break-even | none | <= 50 |
| Decision | FAIL | PASS required |

The earlier `g3_sdv_frozen_2026-07-16.json` result included augmented-path
recording in one SDV timing repeat and is not the authoritative work metric.
The corrected run gives every method the same non-recording timing condition.
Its raw variance conclusion is also independently negative.

## 5. Why G3 failed

The evidence separates correctness from usefulness:

- probability estimates remained aligned with natural Monte Carlo;
- likelihood and replay identities passed to floating-point precision;
- SDV's final single-path variance was `0.00022606`, versus `0.00022555` for
  piecewise CEM;
- SDV cost per path was `0.0001914 s`, versus `0.0000866 s` for piecewise CEM.

Thus the failure is not a missing likelihood term or future leakage. The
conditional-moment regression changed the already strong anchor too little to
improve variance, while two per-step MLP evaluations and SOE updates more than
doubled online cost. The stochastic training loss also did not show reliable
out-of-sample alignment with hard-event second moment.

## 6. Publication assessment

This completed result is not yet a high-impact journal paper:

- the main neural hypothesis was falsified at its development gate;
- the positive contribution is currently a rigorous benchmark and negative
  result rather than a new superior sampler;
- submitting only the present performance claim would overstate novelty and
  practical value.

The reusable scientific assets are still meaningful: an exact-law rBergomi
path-functional benchmark, a strong time-dependent CEM baseline, a finite-grid
conditional-moment oracle, stateful exact mixture replay, and a falsifiable
training/evaluation protocol.

## 7. Authorized next research decision

Under Plan v6, the following actions are prohibited after G3 failure:

- lowering G3 thresholds;
- changing the rare event because SDV lost;
- using the untouched 20 confirmatory seeds for tuning;
- presenting controlled event frequency as efficiency.

The next defensible project is a new, separately pre-registered plan. The most
promising choices are:

1. theorem-first finite-grid Volterra importance sampling with a provable
   approximation/error decomposition;
2. non-neural or linear-in-features backward conditional-moment regression,
   which may retain path feedback without per-step MLP cost;
3. a vectorized/compiled causal operator only after demonstrating a raw
   variance gain large enough to overcome measured inference cost.

The remaining archived ideas, including multilevel and quantum directions,
remain in `RESEARCH_DIRECTION_BACKLOG_POST_G5_2026-07-16.md` and were not used
to rescue this failed gate.

## 8. Reproduction commands

```bash
python -m experiments.m0_gaussian_excursion \
  --output results/m0_gaussian_excursion_frozen_2026-07-16.json

python -m experiments.g0_piecewise_cem \
  --output results/g0_piecewise_cem_frozen_2026-07-16.json

python -m experiments.g3_sdv \
  --output results/g3_sdv_frozen_fair_timing_2026-07-16.json \
  --checkpoint results/checkpoints/g3_sdv_frozen_2026-07-16.pt
```
