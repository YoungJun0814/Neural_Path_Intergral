# G11 V16 현재 모델 구조와 성과

작성일: 2026-08-12  
현재 정책: `v16_hybrid_routing_v5`

## 1. 한 문장 설명

V16은 rough Bergomi의 전체 경로를 무작정 신경망으로 생성하지 않는다. 먼저
독립 가격 Brownian driver를 조건부 Gaussian CDF로 정확히 적분해 없앤 뒤,
남은 Gaussian Volterra 경로에만 rare-event transport를 적용하고, 최종적으로
항상 exact balance likelihood를 사용한 ordinary importance sampling으로 값을
추정한다.

## 2. 왜 이 구조가 필요한가

희귀 사건에서는 보통 Monte Carlo 표본 대부분이 사건을 보지 못한다. Proposal을
사건 쪽으로 이동하면 표본 효율은 좋아지지만, 이동한 분포의 정확한 밀도를 모르면
추정값이 편향된다. V16은 다음 두 원칙을 동시에 지킨다.

1. 사건에 가까운 경로를 더 자주 생성한다.
2. 자연 Gaussian 성분을 양의 질량으로 남기고 전체 mixture density를 정확히
   계산해 `dP/dQ`로 보정한다.

따라서 proposal 학습이 실패해도 likelihood가 틀어지지는 않는다. 성능은 나빠질
수 있지만 estimator의 유한격자 정확성은 유지된다.

## 3. 계산 흐름

```text
rBergomi finite grid (N=32)
        |
        v
독립 가격 driver를 조건부 Gaussian CDF로 정확히 적분
        |
        v
남은 2N차원 local Gaussian Volterra 경로
        |
        +-- residual SMC (V14)
        +-- full-target tempered SMC
        +-- 두 proposal의 exact balance mixture
        |
        v
구조 파라미터 기반 V5 route 선택
        |
        v
fresh Q 표본 + exact dP/dQ + ordinary sample mean
```

최종 추정량은 `Y = f(X) dP/dQ(X), X ~ Q`이며 `f(X)`는 독립 가격 driver를
적분한 조건부 terminal-event 확률이다. Self-normalization은 사용하지 않는다.

## 4. Proposal 구성요소

### 4.1 V14 conditional residual transport

조건부 사건확률 `f(x)`의 fractional powers를 potential로 사용하는 SMC에서 local
경로 중심을 학습한다. 자연 Gaussian과 이동 Gaussian들을 섞어 exact defensive
mixture를 만든다.

### 4.2 Tempered target transport

온도 0에서 1까지 SMC를 진행해 target 쪽 표본을 만들고, PCA/k-means 또는
replicate별 clustering으로 Gaussian mixture를 적합한다. 학습 입자는 최종
추정 표본으로 재사용하지 않는다.

### 4.3 V14/tempered balance hybrid

두 normalized proposal `Q_1`, `Q_2`를 `Q = alpha Q_1 + (1-alpha) Q_2`로
결합한다. 전체 component를 분모에 포함하므로 likelihood는 exact하며,
`M2(Q) <= M2(Q_j)/alpha_j`가 성립한다.

### 4.4 V5 fail-closed router

Router는 reference 확률이나 evaluation 결과를 읽지 않고 Hurst, vol-of-vol,
correlation, strike ratio만 사용한다.

- 확인된 rough K2: V14/tempered hybrid;
- 확인된 rough K0.5와 K1: tempered target route;
- 확인된 regular cell: CM action route;
- 반복적으로 불안정했던 joint extremes: exact V14 correctness fallback.

Fallback은 실패를 숨기는 장치가 아니다. 정확성은 유지하되 그 셀에서는 새로운
방법이 비교기보다 빠르다는 주장을 명시적으로 포기하는 장치다.

## 5. 확인된 성능

모든 비율은 training cost를 포함한 100-query work-normalized variance에서 가장
강한 accuracy-qualified comparator를 후보로 나눈 값이다. 1보다 크면 V16이 더
효율적이다.

| 셀 | V5 역할 | 비율 | robust RSE | 판정 |
|---|---:|---:|---:|---|
| rough H=0.05, K=1 canonical | dominance | 5.978 | 2.03% | 통과 |
| rough H=0.05, K=2 OOD | dominance | 1.280 | 2.16% | 통과 |
| rough H=0.05, K=0.5 OOD | dominance | 5.023 | 1.82% | 통과 |
| regular H=0.12, K=1 OOD | dominance | 1.815 | 0.67% | 통과 |
| rough + eta=2.0 | correctness fallback | 0.471 | 4.53% | 정확성만 통과 |
| rough + rho=-0.9 | correctness fallback | 0.605 | 4.21% | 정확성만 통과 |
| eta=2.0 + rho=-0.9 | correctness fallback | 0.639 | 0.71% | 정확성만 통과 |

모든 V5 OOD 셀에서 likelihood-bound violation은 0이고, 외부 SMC reference
accuracy와 likelihood-normalization gate를 통과했다. Canonical과 OOD 결과는
서로 다른 clean committed source와 fresh seed에서 생성됐다.

## 6. 실패에서 확인된 사실

rough/high-eta 셀에서는 global/replicate clustering, covariance multiscale,
V14/tempered hybrid, residual target-power 변경, replicate-centre 보존, independent
exact-risk selection, proposal-bank balance mixture를 모두 시험했지만 training-seed
전반의 우월성을 얻지 못했다.

특히 exact-risk selector의 식은 맞지만 simultaneous empirical-Bernstein 반경이
추정 위험보다 수만 배 커, 현재 표본수에서는 선택을 정당화하지 못했다. 최종 V5는
이 selector를 사용하지 않는다.

## 7. 이론적으로 보장되는 것

- declared finite grid에서 조건부 terminal representation;
- frozen proposal에 대한 ordinary-IS unbiasedness;
- positive natural mass에 의한 pointwise likelihood bound;
- exact balance-mixture second-moment inheritance;
- independent finite-bank selection의 조건부 unbiasedness;
- parameter-only fail-closed routing의 셀별 exactness;
- 명시된 가정 아래 small-noise exponent와 trace-class safety logarithmic
  efficiency.

## 8. 아직 보장되지 않는 것

- 모든 OOD 셀에서의 성능 우월성;
- bounded relative error 또는 uniform asymptotic optimality;
- continuous-time exact simulation;
- continuous target의 quantitative relative mesh-bias rate와 완전한 end-to-end
  complexity;
- barrier/multi-asset/jump task로의 자동 일반화;
- 외부 전문가가 확인한 submission-level novelty.

## 9. 현재 논문 수준

내부 코드·수학·통계 기준으로는 강한 박사급 working-paper core이며, named-cell
finite-grid empirical claim은 자동 audit를 통과했다. 그러나 top-journal 제출
승인은 아직 아니다. 남은 hard blockers는 독립 novelty reviewer 0/2, 독립
연구자/물리 하드웨어 재현 부재, continuous relative-bias/complexity gate, 그리고
fallback 셀의 비우월성이다.

현재 허용되는 표현은 “정확한 조건부 path-space transport와 명시적 fail-closed
fallback을 갖춘, 일부 deep-tail regime에서 확인된 training-inclusive 효율 개선”이다.
