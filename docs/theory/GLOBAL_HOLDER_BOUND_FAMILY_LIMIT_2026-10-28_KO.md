# 현재 global-envelope/Hölder proof 계열의 구조적 한계

2026-10-28 America/New_York. 이것은 **계산한 상계 공식의 하한**에 관한 논증이다.
실제 discretization error의 하한, rBergomi의 불가능성, 모든 relative-bias proof의
불가능성 정리가 아니다. 기존 frozen 결과나 통과 기준을 바꾸지 않는다.

## 1. 왜 r 조정과 inverse-moment 상수 개선만 반복하지 않는가

[root-bound 유도](TAIL_WEIGHTED_ROOT_BOUND_AND_SPLIT_HMC_2026-10-28_KO.md)의
`Broot=b[σI sqrt(CI)+|ρ|σJ sqrt(CJ)]`에서 r>1,q=r/(r−1),
b=1−1/(2q)>.5다. [Jensen 보강](SHARP_JENSEN_INVERSE_MOMENT_SUPPLEMENT_2026-10-28_KO.md)을
적용해도 각 inverse-moment upper Br은 `(ξT)^(−r)` 이상이다.
이는 해당 explicit exponential formula에서 바로 따른다.
더 일반적으로 `E I^(−r)≥(E I)^(−r)=(ξT)^(−r)`이므로 유효한 inverse-moment
upper를 더 날카롭게 만들어도 이 Jensen lower 아래로 내려갈 수 없다.

CDF envelope `K0≥sup_s φ(s)^(2q)/Φ(s)`는 s0에서
`K0≥2φ(0)^(2q)`를 만족한다. 따라서 `K0^(1/q)≥1/(2π)`다.
현재 proof가 별도로 합한 CI의 첫 양의 항으로부터

`CI ≥ 1/(4πcξT)`, `c=1−ρ²`,

그리고 아래가 따른다.

**`Broot ≥ σI/(4sqrt(πcξT)) = sqrt(Cv Aupper_N)/(4sqrt(πcξ)) = L_N`.**

여기 Broot는 **확률 범위로 자르기 전** 미분·Hölder 공식의 값이다.
확률 거듭제곱 차이는1이하이므로 최종 bound를 min(1,Broot)로 줄일 수 있다.
이때 floor도 min(1,L_N)로 해석해야 한다. high-eta의 미절단 floor가1보다
크다는 것을 clipped bound의 하한이1보다 크다는 주장으로 바꾸지 않는다.

여기서 σI는 현재 proof에서 사용한 `sqrt(T Cv Aupper_N)`이며,
Aupper는 실제 driver MSE 자체가 아니라 현재의 보수적 driver-MSE 상계다.
따라서 이 결과는 이 공식을 그대로 유지하면서 inverse-moment 상수나 r만
조정하는 것의 한계를 말한다. ΔI의 더 날카로운 통제, derivative와 error의
상관을 보존하는 부등식, 다른 weak-error 구조는 이 계열을 벗어날 수 있다.

## 2. 상대10% band와의 관계

μ_N을 정확히 안다고 가정하는 낙관적인 상황에서도, 대칭 root budget B로
μ가[.9μ_N,1.1μ_N] 안에 있다고 하려면 최소한 upper endpoint에서

`B ≤ ((1.1)^b−1) μ_N^b ≤ .1 sqrt(μ_N)`

가 필요하다. 마지막 inequality는 .5<b<1,0<μ_N≤1 및 concavity로 성립한다.
이는 필요조건을 일부러 느슨하게 만든 것이다. 실제 relative certificate에는
lower endpoint/μ_N uncertainty/rounding도 추가로 다뤄야 한다.

현재 L_N이 이 느슨한 필요조건보다 크면, **이 공식 계열만으로** 그10% band를
얻을 수는 없다. 관측 μ_N plug-in을 쓰는 아래 숫자는 원래 확률을 인증하는
자료가 아니라, 정확히 그 값을 안다고 가정해도 현 bound가 얼마나 부족한지의 진단이다.
두 셀에서는 min(1,L_N)도 해당 필요조건보다 크므로 범위 절단으로 해결되지 않는다.

## 3. 해석과 다음 수학적 의무

고정 H=.05에서 현재 Aupper_N은N^(−.1), L_N은N^(−.05)다.
이를 실제 필요한 grid의 complexity나 실제 weak rate라고 부르지 않는다.
이 bound family의 극단적으로 느슨한 floor일 뿐이다.

같은 CDF envelope에 q/r만 더 많이 scan하거나, global inverse-moment bound만
더 최적화하는 계산은 이 바닥값을 없애지 못한다. 생산 실험과 구분해서 중단한다.
다음에는 다음 중 하나의 **실제 증명**이 필요하다.

1. tail-weighted error correlation/localization을 보존하고 complement remainder도
   제어하는 상대오차 논증. 표본에서 본 tail quantile은 확정 remainder가 아니다.
2. 현재 BLP와 nonlinear digital target의 weak cancellation을 이용하는
   유효한 finite-N 수치 상수/상계. 미지 상수의 asymptotic rate만으로 대체하지 않는다.
3. 다른 정당한 continuum estimator에서 unbiasedness와 finite variance/finite
   expected work를 함께 입증. random-level truncation/timeout은 별도 bias를 남긴다.

이 목록은 확보한 정리가 아니라 남은 증명 의무다. 어느 것도 얻지 못한 상태에서
P2를 완료 처리하거나 모델 차원·operator·GPU 증설로 대체하지 않는다.
