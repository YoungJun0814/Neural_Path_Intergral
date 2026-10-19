# G0/G1 physical-window 가이드: geometry oracle만 먼저 구현

2026-10-16. 기존 통합V2의 P3(373component 상한, point/window/distributed feature) 및
[grid scaling 검토](../theory/GRID_STABLE_PRICE_GUIDE_SCALING_2026-10-16_KO.md)에 근거한다.
P2 해결을 선언하거나 실제 q 성능 비교/학습을 먼저 여는 계획이 아니다.

## 고정 geometry

N은32의 배수,32<=N<=4096. G0는 t=kT/32,k1..31의 point driver와 다음 price
white coordinate를 쓴다. G1은15point(t=kT/16,k1..15)+16window를 쓴다.
window 중심은 T/8..7T/8에 균등하게 배치하고 가까운 grid index로 반올림한다.
window volatility feature의 past 길이는T/8이다. G1의 **31개 price 방향 전부**
future 길이T/8로 분포시키며, 만기까지T/16만 남은 마지막 point-volatility template만
T/16로 자른다. 따라서 모든 price 방향이 양의 고정 물리 길이를 갖는다.
이는 표본 생성 전 이론 검토에서 확정한 수정이다. 초기 geometry-only prototype의
15개 한-cell price 방향을 유지하면 전체 G1의 strong grid consistency를 주장할 수 없다.
기존 실험 기록이나 실행 중인 resource-v2의 설정은 바꾸지 않는다.

volatility feature는 window의 discrete driver 평균(또는 point)이다. 실제 BLP linear
operator의 adjoint FFT로 그 white-coordinate row를 만든 뒤 normalize한다.
future support의 analytically-zero 항에 생기는 FFT roundoff는 크기를 검사하고
명시적으로0으로 둔다. payoff/target/importance contribution을 clipping하는 것이 아니다.
price 방향은 window 시작 이후의 W white coordinates에 같은 크기로 분포시키고 normalize한다.
disjoint supports로 두 방향의 inner product=0을 검증한다. 다른 window끼리 직교한다고
주장하지 않는다.

각 template에서 m=a*u+b*v, a2/4/6/8,b0/2/4, sign은 downside rho convention을
따른다. 31*4*3+natural=373components, natural mass.1. 에너지는 실제 ||m||²/2를 사용한다.
두 선형 제약의 minimum-energy mean이지 nonlinear rare-event optimum은 아니다.

기존 `GaussianMixtureShiftSpec`, full balance-mixture density, exact Gaussian mixture
sampler를 재사용한다. physical simulator/conditional payoff/원래 q는 수정하지 않는다.

## oracle와 범위

- N32 G0의 모든 mean과 full density를 기존 dense guide와 대조.
- G1 adjoint rows를 작은 grid의 독립 dense driver operator와 대조.
- causal support, unit norms, paired orthogonality와 실제 energy를 검사.
- point price effect∝sqrt(h), fixed-window effect∝sqrt(T/8)를 constant-volatility
  linear functional에서 검사. 비선형 payoff 효율 보장은 아니다.
- 공통 Brownian dyadic embedding에서 G0 한-cell price 방향의 squared distance는
  2-sqrt(2)로 줄지 않는 반면 G1의 고정 물리 window 방향은 수렴하는지 검사.
- 별도 generator의 exact label/noise reconstruction, natural support와 density 검사.
- N4096 준비 memory/finite geometry 확인. 실제 rare-event sampling 성과는 평가하지 않는다.

이 단계는 geometry와 software identity만 구현한다. 원래 G2 weights 학습, bank fit,
최종 μ/M2 비교, large-training은 미실행 상태다. 실제 reference pilot를 하려면
별도 count/seed/budget/비교 family를 결과 생성 전에 봉인한다.
