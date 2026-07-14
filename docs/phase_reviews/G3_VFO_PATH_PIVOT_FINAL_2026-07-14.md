# G3 VFO path-dependent pivot: final development review

Date: 2026-07-14
Status: **FAILED — memory-superiority claim retired**

## 1. Decision

The one task/representation redesign permitted by Plan v3 was executed after the
terminal-event matched ablation failed. The redesigned task is a discretely monitored
down-barrier event under a strongly rough, high-leverage rBergomi regime. Every
controller receives the same causal running-minimum state, the same initialization,
the same number and order of PI/PICE/J2 updates, and the same validation seeds.

The structural controller is statistically and practically indistinguishable from the
training-matched instantaneous controller. Its observed online work ratio is
`1.00068`, only a `0.068%` difference and far below both timing resolution and the
prospective lower-CI threshold of `1.10`. The full structural-plus-residual VFO is
less efficient, with an online work ratio of `0.85334` versus the instantaneous
controller. Therefore G3 does not pass, the allowed pivot is exhausted, and no
memory-superiority or amortized-VFO claim may be advanced from these experiments.

## 2. Frozen development setup

- Config: `configs/g3_vfo_path_pivot.yaml`
- Protocol ID: `g3-vfo-path-dependent-pivot-development`
- Protocol SHA-256:
  `4a0f0845c5a2bd0b961ecb56906dd4488cf787e3d47599d42ec0c6a3531dad40`
- Result: `results/g3_vfo_path_pivot_development_2026-07-14.json`
- Model: `H=0.05`, `rho=-0.90`, `eta=1.5`, `xi=0.04`, `S0=100`,
  `T=0.5`, `dt=1/64`
- Event: `min_i S_i <= 75`, monitored only on the declared simulation grid
- Soft curriculum target: barrier `80`, scale `5`
- Validation: three development seeds, 20,000 paths per method and seed

This is a development result, not a sealed confirmatory result. A confirmatory run is
not justified after the prospective effect-size gate failed.

## 3. Matched result

| Method | Estimate | Combined SE | Single-path variance | Seconds/path | Variance × cost | Work ratio vs instant |
|---|---:|---:|---:|---:|---:|---:|
| Natural MC | 0.075600 | 0.001079 | 0.069885 | 9.342e-6 | 6.529e-7 | 0.8836 |
| Instant-only, matched | 0.078274 | 0.000556 | 0.018579 | 3.105e-5 | 5.769e-7 | 1.0000 |
| Structural-only, matched | 0.078294 | 0.000552 | 0.018252 | 3.158e-5 | 5.765e-7 | 1.0007 |
| Full VFO | 0.078419 | 0.000597 | 0.021417 | 3.156e-5 | 6.760e-7 | 0.8533 |

The instantaneous controller substantially reduces raw variance, but controller
inference consumes most of that gain. It is not materially better than natural MC on
the development timing proxy. The structural branch reduces variance by only `1.76%`
relative to instant-only while adding enough inference work to erase that reduction.
The full VFO increases variance by `15.28%` relative to instant-only and is also more
expensive.

The maximum residual energy fraction is `0.000304`; no takeover alarm occurs. This is
numerically safe but scientifically negative: the residual branch does not learn a
material correction. The structural gate is nonzero, yet its estimator-level effect
is negligible. The SOE approximation itself is accurate (`1.09e-4` relative L2 error,
`5.64e-4` maximum relative error), so kernel fitting error does not explain the lack of
gain.

## 4. Mathematical and technical audit

### 4.1 Adaptedness

At step `i`, the controller is evaluated from `S_i`, `v_i`, the Volterra state, the
running minimum through `i`, and memory constructed only from increments before `i`.
The new increment is sampled only after the control is evaluated. The target driver
increment is observed by the SOE bank after that control call. Suffix-invariance and
state-reset tests exercise this ordering.

### 4.2 Measure direction and likelihood

The simulator samples independent proposal increments `Delta W^Q` and reconstructs

```text
Delta W^P = Delta W^Q + u_i Delta t.
```

The returned likelihood is

```text
log(dP/dQ) = -sum_i u_i · Delta W_i^Q
              - 0.5 sum_i |u_i|^2 Delta t.
```

The recent singular-cell integral receives the deterministic mean shift induced by the
first Brownian control. Its conditional Brownian-bridge residual is unchanged, so no
extra bridge-density factor is required. G2 independently passed reconstruction,
normalization, bounded-payoff, martingale, and refinement tests for this finite-grid
law.

### 4.3 Objective coordinates

- Soft PI minimizes the Boue--Dupuis control functional for a positive smooth
  approximation of the lower-tail event.
- PICE replays the candidate controller on a fixed target-coordinate path and uses the
  behavior proposal only through exact `dP/dQ_behavior` weights.
- Hard J2 first samples without an autograd graph, then applies the score-function
  gradient on a fixed path. This avoids differentiating through the random simulator
  and avoids retaining a stale recurrent graph.
- No self-normalized estimator is reported as the final probability estimator.

### 4.4 Event semantics

The barrier event is the minimum over simulated grid points. It is not a
continuous-monitoring barrier probability. No Brownian-bridge crossing correction is
implemented or claimed.

### 4.5 Statistical limitations

The development runner currently reports an independent-SE difference z-score even
though common seeds are used. It does not retain pathwise paired differences, and its
wall-clock cost uses one timed evaluation rather than the ten-repeat protocol required
for a sealed result. Consequently the observed `1.00068` structural ratio must be
classified as timing noise, not as an improvement. These limitations do not reverse
the decision: the full VFO is materially worse and the structural effect is roughly
two orders of magnitude below the predeclared continuation threshold.

The controlled estimates differ from natural MC by approximately `2.2` independent-SE
units, below the development alarm of `3`. This does not establish bias. A future
confirmatory runner would need pathwise paired differences and independent high-budget
reference estimates, but G3's effect-size failure makes that extra run unwarranted.

## 5. Consequences for the research program

1. The exact controlled rBergomi BLP implementation remains a valid technical asset.
2. The staged PI/PICE/J2 controller implementation remains a valid experimental
   platform and negative-control result.
3. VFO is retained only as an ablation/diagnostic model. It is not a supported paper
   contribution.
4. G4 amortized VFO, G7 sealed VFO benchmarking, and G8 VFO application do not start
   under Plan v3 because their prerequisite failed.
5. MR and DVDN do not start: no grid-transfer failure, control instability, gradient
   conflict, or desirability-calibration need has been established.
6. CAPT or a path-law mixture would require a new, separately falsifiable hypothesis
   demonstrating a drift-expressivity limitation. The present result alone does not
   establish that limitation.

## 6. Publication assessment at this gate

The current package is not ready for a renowned theory journal or a strong
computational-finance journal. It has an exact finite-grid controlled-law
implementation and unusually transparent negative evidence, but it lacks both a
nontrivial quantitative theorem and a robust work-normalized advantage over strong
baselines. Continuing to amortization or sealed benchmarking without a new scientific
hypothesis would increase compute without addressing either missing requirement.

The honest Plan-v3 endpoint is therefore a stopped core hypothesis, not a successful
flagship model. Any continuation toward publication should begin with a new plan whose
primary contribution does not assume VFO memory superiority.
