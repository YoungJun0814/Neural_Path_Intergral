# G5 CEM-anchored residual: implementation and stop review

Date: 2026-07-15

Status: **exact implementation passed; neural refinement hypothesis failed**

## 1. Executive decision

G5 tested one final, tightly bounded neural hypothesis after the Plan-v4 neural mixture
lost to a strong two-driver constant-CEM mixture. Instead of relearning a proposal from
zero, each neural expert was initialized as an exact copy of its winning CEM expert and
was allowed to learn only a small state-feedback residual.

The controlled law, marginal mixture likelihood, replay checks, tail coverage, and
probability-consistency gate all passed. The learned residual nevertheless increased
mean single-path variance by `18.5%`, increased inference cost per path by `63.8%`, and
produced only `0.523x` geometric work efficiency relative to the CEM base. CEM won the
work comparison on all five validation seeds. There is no incremental break-even.

The predeclared stop rule is therefore active. Further neural architecture search on
this terminal two-tail task is prohibited.

## 2. Implemented model

For mixture expert `k` and grid time `i`, the proposal control is

$$
u_{k,i}^{CAR}=\operatorname{clip}\left(
u_k^{CEM}+b_{res}\tanh f_{k,\theta}(\mathcal F_i),
-b_{global},b_{global}\right).
$$

The last neural layer is zero-initialized. At initialization,

$$
f_{k,\theta_0}=0,\qquad u_{k,i}^{CAR}=u_k^{CEM}
$$

for every state and path. The frozen anchors were:

- left expert: `(3.8983303289, -1.6949394681)`;
- right expert: `(0.1657657137, 3.0674658171)`;
- mixture weights: `(0.5, 0.5)`;
- residual bounds: `(2, 2)` and global bounds `(6, 6)`.

The residual is a width-8 causal state controller using normalized time, log spot,
log variance, the current Volterra state, distances to both thresholds, and a mode
one-hot feature. Each mode received 20 soft path-integral and 20 behavior-snapshot PICE
updates. No hard-event J2, mixture-weight tuning, architecture tuning, or post-smoke
configuration change was allowed.

## 3. Exact finite-grid estimator

The simulator uses bounded adapted controls on the two independent target Brownian
coordinates. For a realized target-coordinate increment sequence, each expert density
is replayed as

$$
\log\frac{dQ_k}{dP}
=\sum_i u_{k,i}\cdot\Delta W_i^P
-\frac12\sum_i\lVert u_{k,i}\rVert^2\Delta t.
$$

With randomized label prior `alpha_k`, the balance-mixture estimator uses

$$
\log\frac{dQ_{mix}}{dP}
=\operatorname{logsumexp}_k\left(
\log\alpha_k+\log\frac{dQ_k}{dP}\right),
$$

and the hard-event contribution is

$$
Y=1_A\exp\left(-\log\frac{dQ_{mix}}{dP}\right).
$$

Thus the finite-grid probability estimate remains unbiased under the declared mixture
law. The neural residual changes proposal variance and cost, not the target quantity.
This statement is limited to the implemented finite grid and bounded adapted controls;
it is not a continuous-time discretization theorem.

## 4. Correctness audit

Before training, exact-reduction tests established:

- arbitrary-state residual outputs are exactly zero at initialization;
- constant CEM and anchored control outputs are bitwise equal;
- with the same Brownian draws, spot, variance, and selected likelihood paths are
  bitwise equal;
- component and marginal mixture log densities are bitwise equal;
- gradients reach residual parameters;
- output controls obey residual and global bounds;
- non-finite Volterra state values are rejected.

After training:

- exact-initialization audit: passed;
- maximum selected-expert replay error: `1.0658e-14` (`<=1e-10` required);
- left/right contribution shares: `47.0% / 53.0%` (`>=10%` each required);
- residual versus natural-MC difference: `z=0.951` (`|z|<=3` required);
- all reported run statistics were finite.

During review, the cross-seed z-score aggregation was corrected from the mean of
per-seed standard errors to

$$
SE(\bar p)=\frac{\sqrt{\sum_s SE_s^2}}{5}.
$$

A unit test now fixes this formula. The corrected result did not change the efficiency
decision.

## 5. Frozen development protocol

- rBergomi: `H=0.10`, `rho=-0.70`, `eta=1.5`, `xi=0.04`, `T=0.5`, `dt=1/64`;
- hard terminal event: `S_T<=42` or `S_T>=139`;
- soft training thresholds: `48/135`, scale `5`;
- training batch: 1,500 paths per update;
- validation: seeds `42601`--`42605`, 50,000 paths per seed;
- timing: median of three evaluations;
- density: all-expert marginal balance likelihood;
- primary endpoint: seed-level `log(work_CEM/work_residual)`;
- work proxy: single-path variance times measured seconds per path.

This is a development experiment (`frozen: false`), not a sealed publication-grade
confirmatory experiment. Because it failed decisively, consuming 20+ new confirmation
seeds would not be scientifically justified.

## 6. Aggregate result

| Method | Estimate | Combined SE | Single-path variance | Cost/path | Online work |
|---|---:|---:|---:|---:|---:|
| Natural MC | 0.002200 | 9.371e-5 | 2.195e-3 | 1.098e-5 | 2.412e-8 |
| CEM exact mixture | 0.002231 | 4.929e-5 | 6.075e-4 | 2.300e-5 | 1.382e-8 |
| CEM-anchored residual | 0.002303 | 5.367e-5 | 7.200e-4 | 3.769e-5 | 2.750e-8 |

Relative to CEM:

- residual/CEM raw variance: `1.185`;
- residual/CEM inference cost: `1.638`;
- residual/CEM mean-work ratio: `1.990`;
- geometric work-VRF `work_CEM/work_residual`: `0.523`;
- contribution ESS fraction: `0.00923` residual versus `0.01015` CEM;
- incremental neural training: `16.54 s`;
- break-even repeated queries: none.

## 7. Seed-level work audit

| Seed | CEM variance | Residual variance | CEM work | Residual work | CEM/residual work |
|---:|---:|---:|---:|---:|---:|
| 42601 | 1.151e-3 | 1.383e-3 | 2.577e-8 | 5.588e-8 | 0.461 |
| 42602 | 3.169e-4 | 3.110e-4 | 7.259e-9 | 1.192e-8 | 0.609 |
| 42603 | 8.163e-4 | 9.591e-4 | 1.814e-8 | 3.527e-8 | 0.514 |
| 42604 | 4.471e-4 | 5.696e-4 | 1.070e-8 | 2.049e-8 | 0.522 |
| 42605 | 3.057e-4 | 3.767e-4 | 7.207e-9 | 1.394e-8 | 0.517 |

One seed showed a small raw-variance improvement, but the residual lost the
training-excluded online-work comparison on every seed. This makes the failure robust
to the difference between arithmetic and geometric aggregation.

## 8. Gate decision

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Exact CEM initialization | true | true | pass |
| Replay error | `<=1e-10` | `1.07e-14` | pass |
| Both modes | `>=10%` | `47.0% / 53.0%` | pass |
| Probability consistency | `|z|<=3` | `0.951` | pass |
| Raw variance | residual `<` CEM | `1.185x` CEM | fail |
| Geometric work-VRF | `>1.10` | `0.523` | fail |
| Seed direction | `>=4/5` | `0/5` | fail |
| Break-even | `<=25` queries | none | fail |

Overall: **fail and stop**.

## 9. Interpretation and publication boundary

The negative result has a clear technical explanation. The constant two-driver CEM
anchors already place mass effectively in each terminal tail. The learned feedback did
not discover enough remaining state dependence to compensate for either its higher
variance or its neural inference cost. Exact initialization protected the starting law,
but the PI/PICE updates were not monotone in hard-event second moment; warm-starting
therefore does not guarantee improvement.

What is retained:

- exact controlled rBergomi finite-grid simulation;
- exact multimodal balance-mixture likelihood and replay;
- Gaussian mixture oracle and tests;
- a practical two-driver CEM mixture that beats natural MC online for this task;
- reproducible evidence that the tested neural refinements are unnecessary here.

What is rejected:

- a positive claim that VFO, neural mixture, or anchored neural residual beats strong
  constant controls on this terminal event;
- post-hoc width, seed, threshold, or objective searches;
- a top-journal neural-model submission based on the present terminal-task result.

A defensible next paper must change the scientific question, not merely the network.
The two acceptable routes are (1) a theorem-led analysis of exact multimodal tilting for
finite-grid Volterra systems, or (2) a separately preregistered path-dependent
application where constant terminal tilts are structurally insufficient. Either route
must retain CEM, exact marginal density, training-inclusive work, and sealed multi-seed
evaluation as non-negotiable baselines.

## 10. Reproduction

```bash
python -m experiments.g5_cem_anchored_residual \
  --config configs/g5_cem_anchored_residual.yaml \
  --output results/g5_cem_anchored_residual_development_2026-07-15.json

python -m pytest tests/test_cem_anchored_residual.py \
  tests/test_path_mixture_training.py tests/test_rbergomi_mixture.py -q
```

Primary artifact:
`results/g5_cem_anchored_residual_development_2026-07-15.json`.
