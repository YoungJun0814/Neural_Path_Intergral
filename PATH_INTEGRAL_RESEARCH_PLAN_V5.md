# CEM-Anchored Residual Path-Integral Mixture: 실행계획 v5

Status: **executed; neural residual hypothesis falsified and stopped**

Date: 2026-07-15

## 0. Motivation

Plan v4의 exact neural mixture는 natural MC를 이겼지만 two-driver constant CEM
mixture보다 약 2.12배 비효율적이었다. 따라서 새 neural controller를 다시 zero부터
학습하는 방식은 중단한다. v5는 winning CEM law를 exact submodel로 포함하는 한 번의
residual pilot만 허용한다.

$$
u_{k,i}^{CAR}=\operatorname{clip}
\left(
u_k^{CEM}+b_{res}\tanh f_{k,\theta}(\mathcal F_i),
-b_{global},b_{global}
\right).
$$

마지막 layer를 zero initialize하여

$$
f_{k,\theta_0}=0,
\qquad
u_{k,i}^{CAR}=u_k^{CEM}
$$

를 경로별로 만족시킨다.

## 1. Claim boundary

주장 후보:

- strong CEM proposal을 exact initialization으로 보존하는 feedback refinement;
- rough-volatility two-driver prefix에서 state-dependent correction이 필요한지에 대한
  estimator-level 검정;
- residual inference cost를 포함한 quantitative go/no-go.

주장하지 않음:

- residual control 또는 warm-start 자체의 신규성;
- neural model이 constant IS보다 일반적으로 우월함;
- CEM base가 global optimum임;
- terminal task 결과가 path-dependent task로 자동 일반화됨.

## 2. Frozen base

Plan v4 audit에서 얻은 CEM controls를 고정한다.

- left: `(3.8983303289, -1.6949394681)`
- right: `(0.1657657137, 3.0674658171)`
- mixture weights: `(0.5, 0.5)`
- hard terminal thresholds: `(42, 139)`
- soft curriculum thresholds: `(48, 135)`
- rBergomi: `H=.10`, `rho=-.70`, `eta=1.5`, `xi=.04`, `T=.5`, `dt=1/64`

Right base는 별도 CEM convergence audit의 12-iteration control이다. Base 선정 seed와
residual train/validation seed를 분리한다.

## 3. Correctness gates

학습 전:

- state별 residual output exactly zero;
- anchored control equals constant base exactly;
- same RNG에서 base와 anchored spot/variance/likelihood pathwise identical;
- mixture component/marginal density pathwise identical;
- suffix perturbation cannot alter earlier control;
- residual bound와 global bound 준수.

학습 후:

- selected replay error `<=1e-10`;
- both tail contribution share `>=10%`;
- reported probability difference from natural reference `|z|<=3`.

## 4. Training

각 mode는 CEM base에서 독립적으로 학습한다.

1. soft PI residual refinement;
2. behavior snapshot을 사용하는 mode PICE;
3. mixture weight와 CEM base는 고정;
4. hard J2 joint expert refinement는 이 pilot에서 금지.

Residual hidden width는 `8`, residual bound는 driver별 `2`, global bound는 `6`으로
제한한다. CEM base가 이미 strong하므로 큰 network나 추가 memory branch를 사용하지
않는다.

## 5. Matched-work protocol

- 새로운 train seed와 validation seed 5개;
- seed당 50,000 validation paths;
- timing 3회 median;
- base CEM과 residual mixture가 동일 weights와 all-expert balance likelihood 사용;
- primary endpoint: seed-level
  `log(work_CEM/work_residual)`;
- raw variance, inference cost, work, mode share, likelihood ESS 보고;
- CEM 학습비용은 공통비용으로 상쇄하고 residual 추가 학습비용의 break-even 보고.

## 6. Single stop gate

모두 만족해야 한다.

- residual raw variance가 base CEM보다 작음;
- geometric work-VRF versus CEM `>1.10`;
- 개선 방향이 최소 4/5 seed에서 일치;
- incremental training break-even `<=25` repeated queries;
- correctness gates 전부 통과.

한 항목이라도 실패하면:

- residual/neural mixture core claim 폐기;
- 추가 width, task, threshold 또는 objective redesign 금지;
- exact controlled rBergomi, Gaussian theorem, CEM mixture 결과만 보존;
- 후속 작업은 architecture search가 아니라 theorem 또는 reproducible application으로
  전환.

## 7. Publication implication

이 pilot 통과는 논문 투고 충분조건이 아니다. 통과 후에도 path-dependent task, strong
mixture/niching baseline, 20+ sealed seeds, theorem이 필요하다. 실패하면 현재 neural
결과를 저명 저널의 positive model claim으로 사용할 수 없다.

## 8. Execution endpoint (2026-07-15)

The one permitted pilot was executed without changing the model, task, training seeds,
or validation seeds after inspection of the smoke output. The development result is:

- exact CEM initialization: passed;
- maximum selected-expert replay error: `1.07e-14`, passed;
- probability difference versus natural MC: `z=0.951`, passed;
- minimum tail contribution share: `47.0%`, passed;
- mean raw variance: `7.200e-4` residual versus `6.075e-4` CEM, failed;
- geometric work-VRF versus CEM: `0.523`, required `>1.10`, failed;
- work improvement: `0/5` validation seeds, required `>=4/5`, failed;
- incremental break-even: none, failed.

Therefore the stop rule is active. No further width, objective, threshold, task, or
architecture search is permitted for this neural residual claim. The retained practical
model is the exact two-driver constant-CEM balance mixture. Any new research phase must
be a separately justified theorem project or a genuinely path-dependent application,
not post-hoc repair of this terminal-event pilot.

Full review:
`docs/phase_reviews/G5_CEM_ANCHORED_RESIDUAL_2026-07-15.md`.
