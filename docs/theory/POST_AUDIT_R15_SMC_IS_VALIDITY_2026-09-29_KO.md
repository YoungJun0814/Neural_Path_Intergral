# R1.5 조건부 SMC·IS의 유한격자 정합성 점검

작성일: 2026-09-29

적용 대상: `N=32`의 유한차원 `2N` 표준정규 local coordinate와 bounded 조건부 digital payoff. 연속시간 rough volatility의 균일 오차 정리가 아니다.

SMC normalizer·mutation의 일반적 배경은 [Del Moral–Doucet–Jasra의 SMC sampler 논문](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2006.00553.x), Gaussian prior 보존 pCN의 배경은 [Cotter–Roberts–Stuart–White](https://arxiv.org/abs/1202.0709)를 따른다. 아래 정합성 논증은 **현재 코드의 고정 일정·가중치 운반·kernel 순서에 대한 적용**이며 새 일반 정리라고 주장하지 않는다.

## 1. 정확히 무엇을 추정하는가

독립 가격 driver를 적분한 뒤 local coordinate `Z∼p=φ_d`에서 `0≤g(Z)≤1`인 조건부 event 확률을 계산한다. `d=2N`, `μ=E_p[g]`. 두 local coordinate 묶음은 같은 volatility Brownian의 셀 내 관측을 표현한다. 서로 다른 두 물리 Brownian driver로 해석하지 않는다. `μ`는 명시된 left-endpoint rBergomi 유한격자 target이며, 연속시간 가격·시장 실측 tail 확률과 동일하다는 주장은 하지 않는다.

아래 pCN log-acceptance 구현의 직접적인 적용 범위는 이 실험의 비퇴화 조건부 Gaussian CDF처럼 **모든 유한 경로에서 `0<g<1`**인 경우다. 일반적인 hard indicator로 `g=0`을 허용하려면 `-∞-(-∞)`의 acceptance convention을 별도로 정의·테스트해야 한다. 현재 코드의 SMC 타당성을 임의의 0-payoff potential 전체로 확장해 주장하지 않는다.

## 2. 고정 일정 weighted SMC의 normalizer

`0=β₀<β₁<···<β_T=1`을 데이터/입자에 의존하지 않게 고정하고, `γ_t(f)=E_p[g^{β_t}f]`, `Z_t=γ_t(1)`, `π_t=γ_t/Z_t`로 둔다. 초기 `X₀^i∼p`, `W₀^i=1/M`, `\hat Z₀=1`이다. 단계 `t`에서 구현은

```text
a_i = W_(t-1)^i · g(X_(t-1)^i)^(β_t-β_(t-1))
c_t = Σ_i a_i
Zhat_t = Zhat_(t-1) · c_t
W_t^i = a_i/c_t
```

를 계산한다. 매 단계 resampling하지 않으므로 건너뛴 단계의 `W_t`를 균등으로 잘못 되돌리면 다른 추정기가 된다. 현재 구현은 resampling하는 4번째 단계에만 `W=1/M`으로 되돌린다. 마지막 단계는 resampling하지 않는다.

resampling `A_1,...,A_M`은 조건부로 `E[#{j:A_j=i}|W]=M W_i`이면 충분하다. 구현한 multinomial과 independent-offset stratified resampling은 이 조건을 만족한다. pCN 제안 `X'=√(1-s²)X+sξ`, `ξ∼p`는 `p`에 대해 가역이다. `min(1, exp(β_t(log g(X')-log g(X))))`를 쓰는 MH 전이는 `π_t∝p g^{β_t}`에 불변이다.

각 단계 resampling의 조건부 기대와 `π_t`-불변 mutation kernel을 순서대로 적용하면 임의 적분 가능한 `f`에 대해

```text
E[ Zhat_t · Σ_i W_t^i f(X_t^i) ] = γ_t(f)
```

가 귀납적으로 성립한다. `f=1`을 택하면 `E[Zhat_T]=μ`. 이는 **고정 격자·정확한 kernel와 arithmetic의 수학적 성질**이다. 유한 반복의 관측 RSE가 참 오차를 반드시 포착하거나, 조상 수가 많은 실행이 모든 rare mode를 발견했다는 뜻은 아니다. 데이터 적응적 resampling/온도 변경으로 일반화하는 정리는 본 단계에서 주장하지 않는다.

### 구현상 확인 사항

- 이론의 potential은 `g^β`이며, `(gp/q)^β`를 tempering한 CE weight와 혼동하지 않는다.
- mutation의 MH ratio에 Gaussian prior ratio를 중복 곱하지 않는다. pCN의 prior 가역성이 이미 상쇄한다.
- 매 단계 `logsumexp`로 normalizer increment를 계산하며, 모든 particle potential이 0인 수치 퇴화는 오류로 처리한다.
- `W`를 운반하는 부분 resampling의 상수-payoff 및 “resampling·mutation 없음 = 같은 Gaussian draw의 plain MC” oracle을 테스트했다.
- SMC SE의 표본 단위는 한 입자나 한 resampling 조상이 아니라 **독립 전체 SMC 반복의 normalizer 출력**이다.
- `final_weight_ess`, incremental ESS, unique-initial-ancestor는 서로 다른 진단이다. 높은 final-weight ESS가 genealogy 회복 또는 IID target sampling을 의미하지 않는다.

## 3. 학습된 exact-mixture IS

학습·선택으로 정해진 임의의 `q`가 최종 표본 전에 동결되고, `Y_i∼q`가 학습 데이터와 독립이라고 하자. 추정기는

```text
μhat = (1/n) Σ_i g(Y_i) p(Y_i)/q(Y_i).
```

조건부 기대 `E[μhat|q]=μ`. 따라서 학습 알고리즘이 랜덤하고 나쁜 `q`를 선택하더라도 support와 적분 가능성이 유지되면 불편성은 잃지 않는다. 그러나 큰 분산 때문에 한 번의 실행이 크게 빗나갈 수 있다. 방어적 `q=δp+(1-δ)q_learned`에서는 `q≥δp`이므로 `0≤g p/q≤1/δ`. 현재 `δ=.1`, 즉 이 유한차원 bounded payoff에서 기여 상한은 10이다. 이 절대 상한은 `μ`가 매우 작을 때 유용한 상대오차 상한을 주지 않는다.

혼합 proposal에서 성분 label을 먼저 뽑아도 분모는 선택된 성분 밀도가 아닌 **모든 성분을 합친 `q`**다. covariance regularization과 defensive mixture 질량은 sampling과 log-density 계산 양쪽에 동일하게 반영된다. SMC의 상관된 최종 particle은 final IS 표본으로 사용하지 않는다. SMC particle은 proposal fit에만 사용하고, final은 새 IID Gaussian mixture draw다. 정규화한 bank 가중치를 fit에 쓰는 것은 허용하지만, final을 `ΣwX/Σw`로 self-normalize하지 않는다.

## 4. KL/M₂ 및 경험 적합의 한계

독립 bank `Y∼G`에서 같은 defensive mean-shift family `q_θ`를 fit할 때

```text
M₂(q_θ) = E_qθ[(g p/q_θ)²]
        = E_G[g²p²/(G q_θ)].
```

따라서 log integrand `2 log g + log(p/G) - log(q_θ/p)`를 쓴 구현이 맞다. `log E_G[·]`를 Adam으로 최소화하는 것은 양수 목적의 순서를 보존하지만, **경험 log-M₂ 자체가 unbiased M₂ 추정치**는 아니다. 같은 낮은-ESS bank를 반복 사용해 얻은 in-sample loss 감소가 새 final 표본의 분산 개선을 보장하지 않는다. rank·basis·KL–M₂ 결과에는 별도 bank와 최종 표본이 반드시 필요하다.

## 5. 이번 검토가 증명하지 못하는 것

1. SMC pCN의 유한 단계 혼합 시간 또는 아직 보이지 않은 rare mode의 완전한 탐색.
2. 표본 기반 RSE의 uniform tail-robust coverage. 1% 표본이 67~77% 기여하는 설정에서는 드문 더 큰 기여가 아직 관측되지 않았을 수 있다.
3. empirical weighted clustering이 진짜 경로공간 mode 수를 식별한다는 정리.
4. 유한격자 `μ_N`와 연속시간 `μ`의 mesh bias bound 또는 `N→∞`에서의 uniform 효율.
5. 한 번의 frozen-workflow wall-time 비교에서의 기계 독립 속도 우위.
6. 새로운 Volterra 특화 second-moment 정리의 완성. 여기서의 SMC/IS 불편성과 defensive moment bound는 알려진 일반 원리를 이 target에 올바르게 적용한 것이다.

이 제약 때문에 이번 독립 교차검증 통과를 top-tier 논문 핵심 기여나 R3 confirmation으로 승격하지 않는다.
