# Price window는 가격뿐 아니라 미래 rough driver도 바꾼다

2026-10-19. G0/G1 실제 pilot 결과를 보기 전의 추가 이론 검토다. 기존
[상수 변동성 scaling](GRID_STABLE_PRICE_GUIDE_SCALING_2026-10-16_KO.md) 및
[조건부 density consistency](PHYSICAL_WINDOW_DENSITY_CONSISTENCY_2026-10-16_KO.md)를 보완한다.
새 모델의 효율이나 독창성을 증명하는 문서가 아니다.

## 직접·간접 효과의 구분

Y_s=sqrt(2H)∫_0^s(s−r)^(H−1/2)dW_r에서, 정규화된 price 방향은 실제로
volatility와 상관된 W의 방향이다. 독립 price Brownian은 conditional CDF에서 이미 적분했다.
따라서 W의 [t,t+τ] 구간에 deterministic drift b/sqrt(τ)를 가하면

    ΔY_s = b sqrt(2H) / ((H+1/2)sqrt(τ))
            * [(s−t)_+^(H+1/2) − (s−t−τ)_+^(H+1/2)].

이는 kernel을 drift에 대해 직접 적분한 식이다. component의 downside sign은 b에 포함한다.
Y_t는 변하지 않지만 미래 Y_s는 변하므로, 가격 방향이 past feature와 직교한다는
사실을 미래 volatility 전체와 독립이라는 주장으로 확대하면 안 된다.

상수 volatility ξ일 때 직접적인 log-price forcing은 ρ sqrt(ξ) b sqrt(τ)다.
그러나 rough volatility의 간접 효과는 다음 별도 scaling을 갖는다.

    ΔY_(t+τ) = b sqrt(2H)/(H+1/2) * τ^H.

one-cell τ=h_N에서는 이 peak가 h_N^H로 줄어든다. H=.05에서 N32→4096으로
바꾸면 직접가격 효과의 비율은1/sqrt(128)≈.0884지만, 이 driver peak의 비율은
128^(-.05)≈.7846이다. 따라서 N4096에서 직접 가격 forcing이 작다는 이유만으로
G0의 price 방향이 무의미하다고 단정할 수 없다. rare-event contribution은 nonlinear
lognormal volatility와 과거 volatility forcing의 결합에도 의존한다.

## 느린 peak와 시간 적분 norm은 다르다

Cauchy–Schwarz와 Fubini로, t+τ<=T일 때

    ∫_t^T |ΔY_s|² ds
      <= b² ∫_t^(t+τ) (T−r)^(2H) dr
      <= b² τ (T−t)^(2H).

증명: 각 s에서 drift 구간 적분의 제곱을 τ times kernel² 적분으로 지배한 뒤,
2H∫_r^T(s−r)^(2H−1)ds=(T−r)^(2H)를 사용한다.
one-cell peak가 느리게 줄어도 integrated norm이0으로 가는 것과 모순되지 않는다.
이 absolute norm bound로 극소 확률 μ의 유용한 relative-error bound를 얻었다고
주장하지 않는다. 고정 physical τ이면 두 효과가 모두 grid refinement에서 유지된다.

## 현재 BLP 구현에 대한 정확한 점검

original historical kernel은 cell-average 가중치다. grid-aligned window의 동일한
first-white shift b/sqrt(m), m=τ/h를 넣으면 Brownian increment mean은 b*h/sqrt(τ),
local singular integral mean은 b*c12/sqrt(τ)다. historical cell-average와 exact local
integral을 합한 **driver mean shift**는 위 연속 kernel 적분과 grid points에서 일치한다.
이는 BLP의 전체 covariance/법칙이 연속 시간 법칙과 같다는 뜻이 아니며, 별도의
moment-matched RMS kernel로 교체하는 주장도 아니다.

독립 dense probing operator와 위 식을 대조하는 oracle, kernel 직접 quadrature,
integrated norm bound와 H=.05 scaling oracle를 추가한다. 실제 simulator/payoff는 수정하지 않는다.

## 해석·다음 ablation의 제한

G0→G1은 price window 외에도 volatility point/window feature와 시점 구성이 달라진다.
따라서 G1이 좋아져도 원인이 price window 하나라고 해석할 수 없다. 후속 원인 분리에는
G0의 같은31개 volatility point/같은 centres를 유지하고 price 방향만 바꾸는 matched
ablation 등이 필요하다. 현재 고정32block pilot에 결과를 본 뒤 arm을 추가하지 않는다.
G0/G1의 reference mean agreement, guide-free SMC 대조, 새 precision 확인은 여전히 별도다.
