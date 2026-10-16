# P2: weak error 문헌 적용 범위와 moment-matched 대안 검토

작성일: 2026-10-10. 기존 frozen BLP law와 결과를 수정하지 않는다.

## 1. 인접 격자 차이는 연속시간 bias 상계가 아니다

가정적으로 μ_N=μ∞+C N^(-α)이고 이 식이 관심 구간 전체에 정확하다면,
fine N 대비 coarse N/2 차이의 절댓값을 2^α-1로 나눠야 남은 fine bias가 된다.
α=.05/.1/.5일 때 배수는 약 28.35/13.93/2.414다. 이는 설명용 가정이며
우리 digital rare-event의 수렴률을 추정하거나 증명한 값이 아니다.

관측 small difference가 10%를 통과해도 continuum 10% certificate로 바꿀 수 없다.
여러 미세 격자의 관측 안정성, 적용 가능한 정리, tail 및 작은 μ에 대한 상대 상수를
구분해야 한다. Richardson extrapolation도 rate와 leading expansion 없이 쓰지 않는다.

## 2. 확인한 1차 문헌과 적용되지 않는 부분

[Gassiat, Weak error rates of numerical schemes for rough volatility](https://arxiv.org/html/2203.09298v2)의
§3은 RMS historical weights를 논의한다. 정리 3.1은 선형 integrand/매끄러운 bounded
test function 또는 별도의 cubic 조건에 한정된다. nonlinear lognormal volatility,
random integrated variance와 함께 쓰는 극단 conditional digital payoff에 그대로 적용되지 않는다.

[Friz–Salkeld–Wagenhofer, Weak error estimates for rough volatility models](https://arxiv.org/pdf/2212.01591)의
정리 1.1은 exact-grid Gaussian driver의 left-point scheme과 polynomial test functions
등 구체적인 조건을 가진다. BLP-approximated driver와 우리 digital probability에 바로
min(3H+.5,1)을 선언할 근거가 아니다. conditional CDF가 경로별 smooth여도 integrated
variance가 작을 때 그 미분의 균등 제어와 극단 tail 상대 오차는 별도 과제다.

## 3. 현재 커널의 구조적 한계: 오류와 구분

α=H-.5, step h, lag j>=2에서 기존 historical coefficient는
m_j=h^α [j^(α+1)-(j-1)^(α+1)]/(α+1)이다.
이는 cell의 kernel 평균으로, 원래 BLP finite law에 맞는 구현이며 버그가 아니다.
하지만 Jensen 부등식으로 h m_j² < ∫cell r^(2α)dr라 driver variance가
continuous t^(2H)보다 작다. 현재 compensator는 **그 finite variance**로 보정해
E[v_N(t)]=ξ를 유지한다. 평균 forward variance가 맞는다고 covariance와 tail이 같지는 않다.

대안 c_j=[h^(2H-1)(j^(2H)-(j-1)^(2H))/(2H)]^.5를 쓰면
2H[local-cell variance + h Σc_j²]=t^(2H)가 모든 monitoring time에 성립한다.
local cell의 exact Gaussian joint law는 그대로다. historical driver만 바뀌므로
이는 별도의 finite-grid target이다. 원래 q/bank/reference를 새 law로 인증하지 않는다.

## 4. 타당성의 직접 증명과 한계

unit cell에서 f(r)=r^α>0, m=∫f, r=(∫f²)^.5라 놓자.
mean-weight squared L2 error는 V=r²-m²이고 RMS-weight error는
2r(r-m)=2rV/(r+m)다. 따라서 V<=error_RMS<=2V.
각 cell에 h와 kernel scaling을 곱해 합하면 원래 mean-weight approximation의
L2 consistency가 유지되는 경우 RMS kernel도 consistency를 유지한다.
이는 **driver kernel**의 비교이지 price digital weak/relative error rate 증명이 아니다.

historical coefficients는 과거 Brownian increments만 사용하며 price에는 left
volatility를 사용한다. 따라서 adaptedness를 유지한다. 정규화된 joint Gaussian
innovation law와 exact conditional price integration도 유지한다. Fourier convolution은
같은 선형 연산의 구현이며 새 probability density approximation이 아니다.

exact marginal variance는 full covariance 일치나 극단 사건 정밀도 보장이 아니다.
mean weights보다 RMS weights가 모든 payoff에 우월하다는 주장도 하지 않는다.
문헌의 알려진 원리를 적용한 것이며 새로운 일반 수학 원리로 제시하지 않는다.

H=.05, T=1의 실제 커널에서 terminal driver variance의 상대 부족은 다음과 같다.
이는 simulation 없이 coefficient로 계산한 값이다.

| N | 원래 variance 부족 | RMS variance |
|---|---:|---:|
| 32 | 0.056916% | 1 (roundoff 이내) |
| 128 | 0.049619% | 1 |
| 512 | 0.043200% | 1 |
| 1024 | 0.040307% | 1 |
| 2048 | 0.037608% | 1 |
| 4096 | 0.035089% | 1 |

부족한 분산이 작다는 이유로 극단 tail 효과도 작다고 증명할 수는 없다. 반대로
이 표만으로 현재 10% 이상 price-mesh 차이의 원인이라고 결론 내릴 근거도 없다.
따라서 이 대안을 주 해결책으로 교체하지 않고, 기존 fine-grid 측정과 price
quadrature 문제를 우선한다. 새 core의 oracle 검증과 실제 rare-event 개선은 별개다.

## 5. 구현 범위와 다음 판단

별도 `MomentMatchedVolterraTerminal` 클래스로 구현한다. 기존 simulator/cache는
그대로 보존한다. 각 H의 exact variance, 원래 커널 불변성, Gaussian coupling,
constant-volatility identity, adaptedness, FFT/autograd directional finite difference,
singleton/batch 일치, L2 cell bound를 검사한다.

우선 기존 law의 fresh fine-grid 결과를 완료한다. 이후 RMS law 자체의 μ와
fine/coarse 차이를 새 seed의 paired SMC 및 다른 reference로 대조해야 한다.
두 law의 차이를 estimator bias로 혼동하지 않는다. 작은 driver variance mismatch
수치만으로 대규모 학습을 승인하지 않는다. 논문 주제와 gate를 변경하는 경우
기존 manifest를 덮어쓰지 말고 새 numerical-method protocol을 먼저 동결한다.
