# G2 Markov Benchmark — Smoke Execution Report

Date: 2026-07-13<br>
Protocol: `g2-heston-terminal-left-tail-v1`<br>
Protocol SHA-256: `56d90523c69bea30751ff32c13809965aaf76d5d38680376021e32ce72353d6e`<br>
Result: `results/g2_heston_score_gradient_smoke_2026-07-13.json`
Refinement: `results/heston_tail_refinement_smoke_2026-07-13.json`

## Gate decision

**G2 is not yet passed.** The analytic reference, estimator, CEM, frozen split,
and neural path are operational. The evidence is non-publication-grade because
the smoke mode uses three evaluation seeds, 10,000 paths per seed, five neural
epochs, and reduced training/validation budgets.

The neural controllers also do not yet beat the CEM constant control on
work-normalized efficiency. A neural-superiority claim is therefore prohibited.

## Completed components

- Heston characteristic-function call/put pricer.
- Gil–Pelaez terminal CDF and numerical left-tail quantiles.
- Monte Carlo/Fourier cross-check and positive conditional log-Euler spot.
- Valid same-trajectory likelihood-weighted constant-control CEM.
- Frozen, disjoint train/validation/evaluation seeds.
- Log-domain likelihood and estimator-contribution diagnostics.
- Repeated bias z-score, relative RMSE, and CI coverage reports.
- Markov feedback controller initialized exactly at selected CEM control.
- Scaled-second-moment, log-second-moment, and entropy-stress ablations.
- Likelihood-ratio gradient for hard indicators, checked on a Gaussian tail.
- Online and training-inclusive work-normalized VRF.
- Versioned neural checkpoints with deterministic state SHA-256 verification.
- A four-grid Heston tail-refinement smoke run at \(10^{-4}\).

## Smoke configuration

- Heston: \(S_0=100\), \(v_0=\theta=0.04\), \(\kappa=1.5\),
  \(\xi=0.30\), \(\rho=-0.70\), \(T=1\), \(r=q=0\).
- Time step: \(1/256\).
- Target probabilities: \(10^{-4},10^{-5},10^{-6}\).
- Thresholds are inverted from the continuous-time Heston CDF.
- Evaluation: 3 seeds × 10,000 paths.
- Neural training: 5 epochs, 1,500 paths per batch.

## Results

These values are diagnostics, not paper results. VRF greater than one favors IS
over naive MC at measured cost.

| Probability | Method | Relative RMSE | Bias z | CI coverage | Online VRF | End-to-end VRF |
|---:|---|---:|---:|---:|---:|---:|
| 1e-4 | CEM constant | 0.0463 | -0.262 | 1.000 | 634.8 | 150.5 |
| 1e-4 | Neural scaled second moment | 0.0511 | -0.311 | 1.000 | 268.3 | 35.7 |
| 1e-4 | Neural log second moment | 0.0519 | -0.375 | 1.000 | 237.2 | 46.7 |
| 1e-4 | Neural entropy stress | 0.0330 | -0.899 | 1.000 | 319.0 | 53.3 |
| 1e-5 | CEM constant | 0.0956 | -0.609 | 0.667 | 2917.0 | 625.2 |
| 1e-5 | Neural scaled second moment | 0.0715 | -0.475 | 0.667 | 810.5 | 157.8 |
| 1e-5 | Neural log second moment | 0.0741 | -0.439 | 0.667 | 894.3 | 155.7 |
| 1e-5 | Neural entropy stress | 0.0744 | -0.965 | 0.667 | 1149.7 | 217.9 |
| 1e-6 | CEM constant | 0.0839 | -0.597 | 0.667 | 12488.0 | 2679.2 |
| 1e-6 | Neural scaled second moment | 0.0747 | -2.266 | 1.000 | 6935.1 | 1280.9 |
| 1e-6 | Neural log second moment | 0.0631 | -2.078 | 1.000 | 6515.4 | 1161.8 |
| 1e-6 | Neural entropy stress | 0.1063 | -0.374 | 1.000 | 6090.7 | 1089.5 |

Naive MC saw no events at \(10^{-5}\) or \(10^{-6}\). Its empirical zero
variance is not used in VRF; analytic Bernoulli variance \(p(1-p)\) is used.

## Time-step refinement smoke

| Steps | Effective dt | Mean estimate | Relative bias | Bias z | Relative RMSE |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.015625 | 9.5910e-5 | -0.0409 | -1.723 | 0.0529 |
| 128 | 0.0078125 | 1.0025e-4 | 0.0025 | 1.146 | 0.0039 |
| 256 | 0.00390625 | 9.9004e-5 | -0.0100 | -0.374 | 0.0389 |
| 512 | 0.001953125 | 9.9677e-5 | -0.0032 | -0.102 | 0.0449 |

The coarsest grid has the largest observed bias. Three seeds are insufficient
to estimate a convergence slope, and the non-monotone RMSE is sampling noise
rather than evidence that 128 steps is optimal.

## Interpretation

1. CEM and neural estimates stay near the analytic target in smoke mode.
2. Three seeds cannot establish coverage; 0.667 and 1.000 mean only two or
   three covered intervals.
3. Feedback sometimes lowers relative RMSE, but inference and training cost
   leave it less work-efficient than CEM.
4. Five epochs cannot establish objective equivalence or convergence.
5. Strong tilt can make weight-only ESS and \(E_Q[L]\) noisy while event
   contributions remain usable. Both diagnostic families must be reported.

## Required before G2 pass

- Run the frozen full protocol: 20 evaluation seeds and 50,000 paths per seed.
- Extend time-step refinement to every rare threshold with the full seed budget.
- Tune only on train/validation seeds and never after inspecting full evaluation.
- Beat CEM under equal measured work, or narrow the contribution claim.
- Preregister acceptable bias-z and coverage criteria before the full run.
