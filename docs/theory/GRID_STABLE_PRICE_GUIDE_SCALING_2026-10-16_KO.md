# 가격 가이드의 grid scaling: 다음 구조 개선의 근거와 제한

2026-10-16. 현재 실행 중인 실험/target/가이드를 바꾸지 않는 이론 검토다.
기존 P3의 physical window/distributed guide 조건과 연결한다.

## 1. 한 white coordinate의 고정 에너지는 고정 가격 효과가 아니다

step h, constant volatility sqrtξ, correlated price increment
ρ sqrtξ sqrt(h) z_i에서 Gaussian mean shift z_i→z_i+b의 가격 drift 효과는
ρ sqrtξ b sqrt(h)이고 Cameron–Martin energy는 b²/2다.
같은 b·같은 energy라도 h→0에서 이 가격 효과는0으로 간다.

물리 길이 τ=mh의 window에서 m개 white coordinates를 각각 b/sqrt(m)만큼
이동시키면 energy는 여전히 b²/2이고 가격 효과는
ρ sqrtξ sqrt(h) m b/sqrt(m)=ρ sqrtξ b sqrtτ다.
고정 물리 window에서는 grid refinement에도 이 효과가 유지된다.

이는 constant-volatility/정해진 선형 price functional에 대한 정확한 계산이다.
비선형 lognormal Volterra volatility에서 rare-event 효율이 자동 개선된다는 정리가 아니다.
window drift가 future volatility도 바꾼다는 효과를 기존 simulator에 그대로 남겨야 한다.

## 2. 구조적 시사점

N32와 N4096에 같은 `price_amplitude`를 사용했다고 같은 물리적 가이드가 아니다.
all-monitor static reference는 각 finite-grid에서 정규화된 올바른 estimator이지만,
그 특정 one-cell price tilt가 refinement에서 같은 제어 효과를 가진다는 보장은 없다.
이는 현재 conditional variance/likelihood 구현의 버그가 아니라 guide 설계의 제한이다.

고정 물리 시간의 price window와 Volterra window/distributed feature를 비교할 근거가
있다. 먼저 기존 point guide와 같은 성분/energy 상한을 맞춘 작은 구조 대조를 설계한다.
final contribution을 보고 window를 고르면 그 자료는 training/selection으로만 사용한다.

## 3. 정확성 조건

Volterra feature u와 price-window v의 직교성을 자동 가정하지 않는다.
volatility feature가 price window 시작 이전의 Gaussian coordinates만 사용하도록
정의하면 disjoint support로 uᵀv=0을 확보할 수 있다. overlap이 있으면
Gram G=A Aᵀ와 제약 일관성을 확인하고 m=AᵀG^†d 및 실제 ||m||²/2를 사용한다.
단순히 a²+b²를 density에 넣으면 틀릴 수 있다.

Gaussian mean mixture density는 Σw exp(mᵀz-||m||²/2) 전체를 계산한다.
fixed-label deterministic means는 adapted deterministic drift로 해석할 수 있지만,
실현된 future path를 보고 shift를 고르는 것은 이 계약과 다르다.

## 4. 현재 단계에서 하지 않는 주장

현재 all-monitor FFT pilot의 결과를 보기 전에 window가 더 낫다고 판정하지 않는다.
새 window model의 구현·학습·성능 검증을 완료했다고 표시하지 않는다.
미세 grid reference의 독립 accuracy, 동일 target 비교, training-inclusive cost가 먼저다.
이는 알려진 Gaussian control scaling을 적용한 구조 검토이며 새로운 일반 수학 정리의
발견이나 최상위 저널 게재 가능성의 증명이 아니다.
