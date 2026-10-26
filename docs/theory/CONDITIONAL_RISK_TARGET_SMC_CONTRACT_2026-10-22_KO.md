# Conditional risk-target SMC 수학 계약

2026-10-22 America/New_York. 새 원리가 아니라 finite-grid 조건부기대값,
SMC sampler와 Gaussian-reference MH를 현재 estimand에 적용하는 계약이다.

## 1. 목표와 bridge

고정 q는373성분 identity-covariance Gaussian mixture다. 좌표61/63의 mean은 모든 성분에서0이다.
U=61은 적분하고 미사용63을 제외한 X∈R62만 탐색한다.

```
Dq(x)=q_X(x)/p_X(x), m2(x)=E_phi(U)[g(x,U)^2],
F(x)=m2(x)/Dq(x)^2,
gamma_beta(x)=q_X(x) F(x)^beta.
```

Z0=1, gamma1=p_X m2/Dq, Z1=M2(q)다.
gamma_beta/p_X=Dq F^beta이고 incremental log weight는 Δbeta logF다.
`q_X(m2/Dq)^beta`는 endpoint p_X m2이므로 이번 risk target이 아니다.
q의 natural mass0.1로 Dq≥0.1, 0≤m2≤1에서 F≤100이지만 logF는 양수일 수 있다.
기존의 nonpositive-log-payoff validator를 그대로 쓰지 않는다.

모든 potential은 양의 finite m2를 요구한다. 0/underflow/비유한값을 clip하여 성공으로 넘기지 않는다.

## 2. 세 가지 mutation

### A. 표준 Gaussian p_X-reference pCN

`x'=sqrt(1-s²)x+s*epsilon`은 p_X에 reversible하다.
MH log ratio는 `Delta logDq + beta*Delta logF`다.
beta*Delta logF만 쓰면 q-reference proposal이 아닌 pCN에서 틀린 target을 보존한다.

### B. 혼합성분 중심 pCN

q_X=Σ_l w_l phi(x-mu_l)라고 둔다. augmented target은
`gamma_beta(l,x)=w_l phi(x-mu_l) F(x)^beta`다.

1. l|x를 q의 exact component posterior `softmax(logw+mu_l·x-|mu_l|²/2)`로 Gibbs sampling.
2. `x'=mu_l+sqrt(1-s²)(x-mu_l)+s*epsilon`을 제안.
3. MH ratio는 `beta*Delta logF`다.

조건부 Gaussian-reference proposal가 phi(x-mu_l)에 reversible하고 F가 l에 독립이므로
위 Gibbs/MH composition은 원하는 augmented target을 보존하며 marginal도 gamma_beta다.
성분 선택을 payoff가 큰 방향으로 임의 선택하거나 proposal correction을 생략하지 않는다.

### C. q 독립 제안

정규화된 q_X에서 새 x'를 뽑으면 MH ratio는 `beta*Delta logF`다.
acceptance로 전역 이동을 기록하되, 이 accepted path를 독립 target 표본이라고 부르지 않는다.
성공한 global move가 있어도 initial ancestry를 reset하지 않는다.

## 3. SMC와 lineage

고정 temperatures와 resampling calendar를 사용한다.
정규화 weight를 stage 사이에 보존하고, resampling 직전 increment의 log mean을 logZ에 누적한다.
stratified resampling 뒤에만 weight를1/N로 바꾼다. final stage는 resampling하지 않는다.
각 stage의 mutation은 **final beta1도 포함**해 해당 target-invariant kernel로 수행한다.
mutation은 normalizer increment를 새로 만들지 않으며 weight/initial ancestry를 임의 reset하지 않는다.

정확한 potential/실수 연산/필요한 적분가능성/올바른 resampling 및 invariant transition 조건에서는
통상 SMC normalizer 해석을 적용할 수 있다. 그러나 finite-N normalized particle의 평균/ESS가
독립표본 추정량 또는 불편 empirical target 평균이 되는 것은 아니다.

실제 m2는 numerical quadrature다. 따라서 **구현이 true M2(q)의 exact unbiased SMC**라고 주장하지 않는다.
엄격한 개별 numerical gate를 적용해도 proxy 자체는 rigorous bias certificate가 아니다.

initial ancestor 개수, weighted ancestor ESS, largest ancestor mass, final nominal weight ESS,
고유 particle 수, mutation acceptance와 실제 jump를 분리한다.
낮은 ancestry는 보수적 gate이며 충분히 mixing한 chain의 이론적 불가능성 증명은 아니다.
반대로 고유 floating-point particle 수가 높거나 resampling 후 ESS=N이라고 독립성을 인정하지 않는다.

## 4. numerical contract

Volterra potential은 현재 동결된 LastVarianceResidualSlice와 analytic batch 적분을 재사용한다.
각 제안마다 m2 개별 상대 numerical gate≤1e-6, critical/envelope/Jensen gate를 요구한다.
이전 IID fitting에서의 aggregate-only tiny-contribution 예외는 MH target에는 적용하지 않는다.
한 제안이라도 unresolved이면 해당 island를 numerical failure로 보존하고 재시도하지 않는다.

q log density는 X62의 full normalized mixture로 계산한다.
물리법칙 입력은 X를full64에 넣고61/63을0으로 둬 profile을 만든다.
새 kernel은 scalar root/적분 tolerance나 기존 결과를 수정하지 않는다.

## 5. 문헌과 독창성 경계

SMC sampler의 연속 target/normalizer 틀:
[Del Moral, Doucet & Jasra (2006), JRSSB](https://doi.org/10.1111/j.1467-9868.2006.00553.x),
[저자 제공 논문](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).
Gaussian-reference 보존 pCN 설계:
[Cotter et al. (2013), Statistical Science](https://doi.org/10.1214/13-STS421),
[저자 원고](https://eprints.maths.manchester.ac.uk/2215/1/1202.0709v3.pdf).
이 문헌들이 현재 Volterra 코드의 성능이나 novelty를 입증하는 것은 아니다.
현재 bridge/혼합성분 조건부 ratio는 위에 직접 유도했고 알려진 답/상세균형 검사로 검증한다.
