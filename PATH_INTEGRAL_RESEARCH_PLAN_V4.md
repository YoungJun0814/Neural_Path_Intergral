# Exact Path-Integral Mixture for Multimodal Rare Events: 실행계획 v4

> **Execution endpoint (2026-07-15): NEURAL CORE STOPPED AT G4-M2.** Gaussian
> oracle and exact rBergomi mixture-law gates passed. A rarity redesign produced an
> online gain versus natural MC, but a two-driver constant CEM mixture was about 2.12x
> more work-efficient than the neural mixture across 5/5 development seeds. Neural
> confirmatory execution is therefore not authorized. See
> `docs/phase_reviews/G4_EXACT_MIXTURE_FALSIFICATION_2026-07-15.md`.

Status: active falsification plan
Date: 2026-07-15
Predecessor: `PATH_INTEGRAL_RESEARCH_PLAN_V3.md` stopped at G3

## 0. 연구 결정

Plan v3의 VFO memory-superiority 가설은 terminal task와 허용된 한 번의
path-dependent redesign에서 모두 실패했다. v4는 VFO를 확장하지 않는다. 새 core
가설은 다음으로 제한한다.

> 다중모드 rare-event target에서, mode-specialized causal path-integral controller의
> exact finite-grid mixture가 단일 feedback controller보다 학습 안정성과
> work-normalized estimator efficiency를 개선하는가?

Mixture importance sampling, multiple importance sampling, PICE, Föllmer drift 또는
neural drift 자체를 발명했다고 주장하지 않는다. 논문 기여 후보는 rough-volatility
two-driver law, exact all-expert path replay, mode curriculum, training-inclusive work를
결합한 결과로 한정한다.

## 1. Novelty boundary

직접 경계는 최소 다음을 포함한다.

1. PICE와 adaptive path-integral importance sampling:
   <https://arxiv.org/abs/1505.01874>
2. 금융 option pricing을 위한 neural importance sampling과 universal approximation:
   <https://link.springer.com/article/10.1007/s00780-024-00549-x>
3. Generalized multiple importance sampling:
   <https://projecteuclid.org/journals/statistical-science/volume-34/issue-1/Generalized-Multiple-Importance-Sampling/10.1214/18-STS668.full>
4. Multimodal rare-event mixture/niching의 2026 직접 경계:
   <https://arxiv.org/abs/2604.06417>
5. Deep transport importance sampling:
   <https://epubs.siam.org/doi/10.1137/23M1546981>

따라서 다음 표현은 금지한다.

- first mixture importance sampler
- mixture가 Föllmer drift보다 더 넓은 절대연속 law class를 표현한다
- randomized expert label을 보존한 component-wise estimator와 marginal-mixture
  balance estimator를 같은 것으로 취급한다
- finite-grid exactness를 continuous-time exactness로 확장한다
- multimodal toy 결과만으로 rough-volatility 실무 우위를 주장한다

## 2. Canonical finite-grid law

`i=0,...,N-1`과 두 독립 Brownian driver에서 target increment는

$$
\Delta W_i^P\sim N(0,\Delta t_i I_2).
$$

Expert `k`의 adapted control은 target prefix의 함수다.

$$
u_{k,i}=\pi_k(\mathcal F_i^P,\zeta).
$$

Expert density ratio는 target coordinate에서

$$
r_k(x)=\frac{dQ_k}{dP}(x)
=\exp\left(
\sum_i u_{k,i}(x)^\top\Delta W_i^P
-\frac12\sum_i\|u_{k,i}(x)\|^2\Delta t_i
\right).
$$

전역 mixture weight `alpha_k>0`, `sum_k alpha_k=1`에 대해

$$
r_{mix}(x)=\frac{dQ_{mix}}{dP}(x)=\sum_k\alpha_k r_k(x),
\qquad
L_{mix}(x)=\frac{dP}{dQ_{mix}}(x)=r_{mix}(x)^{-1}.
$$

Sampling은 먼저 latent expert `K~Categorical(alpha)`를 뽑고 그 expert의 controlled
law에서 path를 생성한다. Primary estimation은 생성 expert뿐 아니라 **모든 expert를
동일 target path에서 causal replay**하여 `r_mix`를 계산한다.

$$
\widehat p=\frac1M\sum_{m=1}^M
1_E(X^{(m)})L_{mix}(X^{(m)}),
\qquad X^{(m)}\sim Q_{mix}.
$$

이 balance estimator는 declared finite-grid law에서 unbiased다. Label을 보존하고
`L_K=dP/dQ_K`를 사용하는

$$
\widehat p_{component}=\frac1M\sum_m1_E(X^{(m)})L_{K_m}(X^{(m)})
$$

도 `K`가 사전 weight로 무작위 선택되면 unbiased다. 따라서 component weight를
“편향된 잘못된 estimator”라고 부르지 않는다. 다만 marginal mixture weight는 모든
proposal 정보를 사용하는 deterministic-mixture/balance weighting이며, mode 반대편의
폭발적 component weight를 완화할 수 있다. 두 estimator의 variance와 all-expert replay
cost를 모두 보고한다.

## 3. 중요한 이론적 제한

Mixture는 일반적인 adapted drift보다 근본적으로 더 큰 law class가 아니다. Expert의
prefix density process를 `Z_{k,i}`라 하면 mixture 자체의 effective drift는 형식적으로

$$
u_{mix,i}=\sum_k\omega_{k,i}u_{k,i},
\qquad
\omega_{k,i}=\frac{\alpha_k Z_{k,i}}
{\sum_j\alpha_j Z_{j,i}}.
$$

즉 mixture의 가치는 universal expressivity가 아니라 다음의 계산적 귀납편향에 있다.

- separated mode마다 쉬운 expert objective 제공
- mode collapse 진단 가능
- expert initialization과 hard-event refinement 분리
- bounded small controller 여러 개와 큰 단일 controller 사이의 work trade-off

이 사실 때문에 v4의 이론 주장은 “strict law expressivity”가 아니라 제한된 controller
family와 estimator work에 대한 quantitative statement여야 한다.

## 4. Gaussian two-tail oracle

### 4.1 Event

한 driver, horizon `T`에서

$$
E_a=\{|W_T|\ge a\},
\qquad p_a=2\Phi(-a/\sqrt T).
$$

### 4.2 Single constant drift

Constant drift `m`의 second moment는

$$
J_{single}(m)
=E_P\left[1_{E_a}\exp(-mW_T+\tfrac12m^2T)\right].
$$

대칭성으로 이 함수는 even이다. 단일 constant drift는 한 tail을 선호하면 반대 tail의
weight를 악화시킨다. Numerical oracle는 global grid와 scalar optimization으로 `m=0`
minimum을 확인해야 한다. 증명 문서에서는 convexity를 별도 확인하며 수치 결과를
증명으로 대체하지 않는다.

### 4.3 Symmetric two-expert mixture

Expert drift가 `+m,-m`, weight가 `1/2`이면

$$
r_{mix}(W_T)=e^{-m^2T/2}\cosh(mW_T),
$$

$$
J_{mix}(m)=E_P\left[
1_{E_a}\frac{e^{m^2T/2}}{\cosh(mW_T)}
\right].
$$

`m=0` 근방 expansion은

$$
J_{mix}(m)
=p_a+\frac{m^2}{2}E_P[1_{E_a}(T-W_T^2)]+O(m^4).
$$

`a>0`에서 tail conditional second moment가 `T`보다 크므로 충분히 작은 nonzero `m`은
second moment를 감소시킨다. 이 restricted-family oracle가 실패하면 rBergomi mixture
구현을 시작하지 않는다.

### 4.4 Oracle gates

- analytic probability reported bias `|z|<=3`
- `E_Qmix[L_mix]=1` reported `|z|<=3`
- analytic density와 replay density max error `<=1e-10` in float64
- best mixture second moment가 best single constant drift보다 최소 20% 작음
- component-wise estimator와 marginal-mixture estimator가 모두 analytic mean과
  일치하고, mixture balance estimator의 second moment가 더 작은지 별도 보고

## 5. Lean rBergomi expert

VFO의 instant-only ablation도 inactive structural/GRU branch를 계산했으므로 production
baseline으로 사용하지 않는다. v4는 state 없는 lean controller를 별도 구현한다.

Input:

$$
(t/T,\log(S_t/S_0),\log(v_t/\xi),Y_t,\zeta),
$$

여기서 `zeta`는 lower/upper threshold와 mode direction이다. 첫 prototype은 task별
expert로 제한하여 task amortization과 혼동하지 않는다.

Output은 independent Brownian basis의 bounded two-driver drift다.

$$
u_t=(u_t^{(1)},u_t^{(2)}).
$$

Lean baseline과 mixture expert는 동일 hidden width를 사용한다. Mixture는 모든 expert
forward replay 비용과 `logsumexp` 비용을 ledger에 포함한다.

## 6. Training curriculum

### M0 — Mode-specific soft PI

Left expert:

$$
V_L=\operatorname{softplus}((S_T-B_L)/\tau_L).
$$

Right expert:

$$
V_R=\operatorname{softplus}((B_R-S_T)/\tau_R).
$$

각 expert를 독립적으로 Boue--Dupuis functional로 초기화한다.

### M1 — PICE replay

각 mode의 behavior expert가 만든 target path에서 candidate를 causal replay한다. Soft
tilted weight에는 exact behavior likelihood를 사용한다. Self-normalized weight는 gradient
training에만 사용하고 final probability estimator에는 사용하지 않는다.

### M2 — Mixture weight J2

Behavior mixture `b`에서 얻은 target path에 대해 candidate mixture `r_theta`의 second
moment는

$$
J_2(\theta)
=E_{Q_b}\left[
1_E\frac{1}{r_b r_\theta}
\right].
$$

첫 implementation은 controller를 동결하고 mixture logits만 최적화한다. Weight floor를
두어 어느 mode도 제거되지 않게 하며, mode removal은 별도 ablation에서만 허용한다.

### M3 — Optional expert refinement

Weight-only M2가 통과한 뒤에만 expert parameter를 alternating J2로 미세조정한다. 한
번에 logits와 모든 expert를 joint update하지 않는다.

## 7. Primary development task

첫 rBergomi task는 terminal digital strangle이다.

$$
E=\{S_T\le B_L\}\cup\{S_T\ge B_R\}.
$$

이는 two-sided market stress, digital strangle, absolute-move risk에 대응하지만 실데이터
상품가격 claim은 하지 않는다. Threshold는 별도 pilot seed에서 각 tail rarity를 맞춘 뒤
model-selection 전에 고정한다.

후속 path task는 terminal gate 통과 후에만 down-barrier/up-barrier union으로 확장한다.
Continuous monitoring은 bridge correction이 없으면 주장하지 않는다.

## 8. Baselines

필수:

- natural Monte Carlo
- antithetic natural Monte Carlo
- best single constant drift
- CEM constant mixture
- lean single feedback trained on union potential
- left expert only and right expert only
- fixed-weight exact mixture
- learned-weight exact mixture

추가 비교는 applicability와 정확한 구현이 확인된 경우에만 사용한다.

- feedforward neural IS objective
- adaptive multilevel splitting for path-union event
- niching IS 또는 Gaussian-mixture CE

VFO는 실패한 historical ablation이며 primary baseline으로 재튜닝하지 않는다.

## 9. Statistical and work protocol

- train/selection/audit/sealed seed 분리
- development 최소 5 seeds, confirmatory 최소 20 seeds
- common random numbers를 사용하되 paired path contribution을 직접 저장/집계
- endpoint: seed-level paired log work-efficiency ratio
- timing: warm-up 후 최소 10회, median과 dispersion 보고
- raw variance와 variance × per-path total cost 모두 보고
- mixture density replay 비용을 제외한 speedup 금지
- training-inclusive break-even `M*` 보고
- component balance, posterior responsibility entropy, top weight share 보고
- threshold/seed를 본 뒤 수정하면 새 protocol ID와 hash 필요

## 10. Phase gates

### G4-M0 — Gaussian oracle

Section 4.4를 모두 통과해야 한다.

### G4-M1 — rBergomi mixture law

- `K=1`이 single proposal likelihood와 pathwise 동일
- `K=1` null expert가 natural entry point와 pathwise 동일
- `K>1` null experts가 natural law와 distribution/moment 수준에서 동일
- all-expert replay density와 analytic constant-control density 일치
- `E_Qmix[L_mix]=1`
- bounded payoff estimate가 natural reference와 `|z|<=3`
- component-wise estimator와 mixture balance estimator의 unbiasedness를 각각 확인
- all-expert replay cost를 포함한 balance weighting의 variance/work trade-off 보고
- expert label permutation invariance

### G4-M2 — Multimodal effect

Development 5 seeds에서:

- 두 mode 모두 contribution ESS에 유의미하게 기여
- mode collapse 없음
- mixture raw second moment가 best single feedback보다 작음
- mixture work-VRF geometric mean `>1.10`
- natural MC를 포함한 best online-work baseline 대비 work-VRF `>1.10`
- paired improvement direction이 최소 4/5 seed에서 일치

실패 시 task/threshold redesign은 한 번만 허용한다. 재실패하면 mixture core claim을
폐기한다.

### G4-M3 — Confirmatory

- config/code/checkpoint hash freeze
- 20+ unseen seeds
- paired 95% lower CI of work-VRF `>1.10`
- reported bias `|z|<=3`
- total training-inclusive break-even이 target query workload에서 성립

## 11. Stop rules

다음 중 하나면 해당 claim을 즉시 중단한다.

- mixture normalization 또는 all-expert replay correctness 실패
- 한 expert가 99% 이상 responsibility를 차지하고 두 번째 mode가 누락
- raw variance는 줄지만 total online work가 개선되지 않음
- 단일 lean feedback이 동일 budget에서 mixture와 동률 또는 우위
- 2026 direct literature와 차별점이 exact rBergomi application 외에 남지 않음
- theorem이 standard mixture identity 재서술에 머묾

## 12. Publication bar

저명 specialist journal을 고려하려면 모두 필요하다.

1. exact finite-grid mixture theorem과 continuous/discrete scope 분리
2. restricted-family 이상의 quantitative result 또는 asymptotic mode result
3. rough-volatility에서 strong baseline 대비 robust work advantage
4. multimodal failure mechanism과 responsibility diagnostic
5. 20+ sealed seeds 및 training-inclusive cost
6. 공개 재현 pipeline

Gaussian oracle만 성공하거나 rBergomi work gate가 실패하면 methodological note 이상으로
과대포장하지 않는다.

## 13. Immediate implementation order

1. exact mixture primitive와 Gaussian oracle
2. mixture law unit tests
3. lean rBergomi feedback expert
4. all-expert target replay와 mixture simulator
5. mode PI/PICE와 weight-only J2
6. development benchmark
7. 이론·기술 독립 감사
8. gate 통과 시에만 confirmatory freeze
