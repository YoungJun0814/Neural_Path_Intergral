# Conditional CDF의 절대 격자 오차 bound와 상대 tail 오차의 한계

2026-10-16. 자연 Gaussian law의 constant ξ, |ρ|<1, 유한 T에 대한 직접 검토다.
아래는 주어진 driver MSE bound를 payoff의 절대 오차로 옮기는 식이다. 현재 실험의
adjacent difference로 continuous driver MSE를 인증하거나 P2를 통과시키는 정리가 아니다.

## 1. 가정과 표기

Y_t와 left-predictable Y_N(t)가 같은 Brownian motion 위의 centered jointly Gaussian
driver이고 각각의 variance가 M 이하라 하자. 현재 BLP mean-weight scheme에서는
M=T^(2H)를 쓸 수 있으며, variance-matched 대안도 grid points에서 같은 상한을 가진다.
각자 자신의 finite/continuous variance로 보정한다:

v=ξ exp(ηY-.5η²VarY), σ=sqrt(v),
I=∫v dt, J=∫σ dW. N 버전도 같은 정의이며 stochastic integral은 adapted Itô다.
A_N=∫ E|Y_t-Y_N(t)|²dt에 대해 **별도로 유효한 상한**이 주어졌다고 가정한다.

## 2. Integrated variance의 inverse moments

Jensen과 Var(평균 Y)<=M에서

I >= ξT exp(η 평균Y-.5η²M),
E[I^(-r)] <= B_r=(ξT)^(-r) exp[.5η²M(r+r²)], r>0.

동일한 inequality는 discrete uniform-time sum I_N에도 성립한다.
따라서 random conditional variance가 작아질 수 있다는 이유만으로 inverse moments가
존재하지 않는 것은 아니다. 다만 이 bound의 상수는 η와 r에 따라 빠르게 커진다.

## 3. 두 Gaussian lognormal volatility의 강한 오차

e²=E|Y-Y_N|²에 대해 Gaussian mgf를 직접 계산하면

E|σ-σ_N|²=2ξ[1-exp(-η²e²/8)] <= ξη²e²/4.

adaptedness와 Itô isometry로 ||J-J_N||₂ <= sqrt(ξη² A_N/4).
또한 V=VarY,V_N=VarY_N,C=Cov(Y,Y_N)라 할 때

E|v-v_N|²=ξ²[exp(η²V)+exp(η²V_N)-2exp(η²C)] <= C_v e²,

C_v=ξ² exp(η²M)[η²+η⁴ M cosh(η²M/2)].

마지막 부등식은 e²=V+V_N-2C, |V-V_N|²<=4M e²,
cosh(x)-1<=x² cosh(|x|)/2, 1-exp(-x)<=x를 이용한다.
시간 적분의 Cauchy–Schwarz로 ||I-I_N||₂ <= sqrt(T C_v A_N).

## 4. CDF derivative를 통한 절대 오차

c=1-ρ², a=log(K/S0)+I/2-ρJ, F(a,I)=Φ(a/sqrt(cI)).
고정 a에서

|∂a F| <= 1/sqrt(2πcI),
|∂I F| <= 1/[2 sqrt(2πe) I].

(a,I)와(a_N,I_N) 사이 직선을 적분하고 minimum I를 사용한다.
E max(I^(-r),I_N^(-r))<=2B_r 및 Cauchy–Schwarz에서

|μ-μ_N| <= sqrt(A_N) × {
 sqrt(B_1/(πc)) [.5 sqrt(T C_v)+.5|ρ|η sqrtξ]
 + sqrt(B_2)/(2sqrt(πe)) sqrt(T C_v)
}.

η>=0 convention을 썼다(음수 η를 허용하면 해당 η 항에 절댓값을 쓴다).
이 식에는 μ가 작다는 이점이 들어 있지 않다. 따라서 극단 μ≈10^(-8)에서
이 절대 상한을 μ로 나누면 실용적 relative-bias certificate가 되기 어렵다.

## 5. 무엇이 증명되지 않았는가

A_N 자체의 유한-N 상한과 coupling이 먼저 필요하다. finite adjacent Monte Carlo
차이를 A_N이나 infinite-grid tail로 바꾸지 않는다. 알려진 strong rate를 써도
그것이 최적 weak rate라는 뜻은 아니다. 절대 bound가 너무 큰 것은 simulator bug나
실제 bias가 크다는 증명이 아니라 이 일반적 부등식이 느슨하다는 뜻일 수 있다.

이 검토는 Jensen/Gaussian mgf/Itô isometry/미분의 알려진 원리를 조합한 것이다.
새로운 일반 수렴 정리, top-journal novelty, 또는 현재 P2/대규모 학습 승인을 주장하지 않는다.
Gaussian moment 식, inverse-moment Jensen inequality와 CDF derivative bounds는
작은 독립 oracle로 검사한다. rare-event-aware하고 유효한 상대 오차 상계는 남은 연구 과제다.
