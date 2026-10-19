# 고정 물리 window의 밀도 일관성과 한-cell price 방향의 극한

2026-10-16. 조건부 수학 검토 초안. 새로운 정리의 독창성, 현재 rBergomi 실험의
오차율, 실제 효율 우위를 입증한 문서가 아니다. 기존 Gaussian continuity,
stable convergence, conditional Jensen을 현재 가이드 설계에 적용한다.

## 1. 비교하는 대상과 가정

공통 Brownian 공간의 isonormal process를 W, Hilbert 공간을 H=L²[0,T]로 쓴다.
BLP의 두 whitened coordinates/시간구간은 같은 Brownian 공간의 구간 적분 및
bridge representer로 실현해야 한다. 서로 다른 grid의 흰 벡터를 좌표별로 같은
난수라고 놓는 것은 이 coupling이 아니다.

유한하고 N에 따라 늘지 않는 J개 template, 고정 mixture weights w_iab>=0,
합=1, defensive mass 0<δ<1, 유한한 amplitudes a,b를 가정한다.
past volatility unit representer u_i,N는 H에서 u_i로 강수렴한다고 가정한다.
이것은 별도의 driver convergence 조건이다. geometry unit test만으로 증명되지 않는다.
또한 원래 조건부 payoff가 0<=g_N<=1이고 같은 공간에서 g_N→g in L¹,
μ=E[g]>0이라고 가정한다. 이전 absolute-bias 검토의 integrated driver MSE가
0으로 감을 보이는 것은 이 payoff 가정을 확보하는 한 경로이지, 여기서 완료한 작업은 아니다.

실제 중요도 밀도 및 second moment는

    D_N = q_N/p_N = δ+(1−δ) Σ w_iab exp(W(h_iab,N)−||h_iab,N||²/2),
    M₂,N = E_p[g_N² / D_N],   μ_N = E_p[g_N].

평균을 empirical normalization하지 않는다. D_N>=δ이므로 1/D_N<=1/δ다.
아래는 μ 추정량의 M₂이지, 다른 원래 q의 M₂를 바꾸는 규칙이 아니다.

## 2. 한-cell price 방향: 약수렴과 남는 중요도 잡음

서로 다른 고정 시점 t_i에 대해

    v_i,N = sign * 1_[t_i,t_i+h_N] / sqrt(h_N),  h_N=T/N.

각 norm은1이지만 v_i,N은 H에서0으로 **약수렴**할 뿐 강수렴하지 않는다.
dyadic 두 grid의 같은 시점 방향은 inner product=1/sqrt(2)이므로
||v_i,N−v_i,2N||²=2−sqrt(2)>0다. 따라서 bounded Cameron–Martin energy만으로
grid-consistent shift를 얻었다고 말할 수 없다.

X_i,N=W(v_i,N)는 큰 N에서 상호 독립 표준정규이고, 벡터 X_N은 σ(W)에 대해
독립 표준정규 Z로 mixing-stable convergence한다. 다음의 elementary Gaussian
논증으로 충분하다. 유한 Gaussian cylinder (W(f_1),…,W(f_k))와의 covariance는
<v_i,N,f_l>→0이므로 공동 Gaussian limit에서 독립이다. bounded σ(W)-measurable
변수는 cylinder 변수로 L¹ 근사할 수 있어 같은 결론이 확장된다.
이는 X_N 자체가 W의 함수라는 사실과 모순되지 않는다.

u_i,N의 강수렴과 payoff L¹ 수렴을 이용하면

    lim M₂,N = E_W,Z [g(W)² / D_*(W,Z)],
    D_* = δ+(1−δ) Σ w_iab exp(a U_i−a²/2) exp(b Z_i−b²/2),
    U_i=W(u_i).

현재 causal support에서는 <u_i,N,v_i,N>=0이어서 energy가 a²+b²다.
극한 공식은 reciprocal의 uniform bound 1/δ, payoff L¹ 근사, Gaussian
cylinder 근사를 통해 얻는다. u와 X를 무조건 서로 독립이라고 놓고 유한 N의
밀도를 계산하는 것은 잘못이며, 독립은 위 극한에서만 사용한다.

Z를 적분한 volatility-only density를

    D_vol(W) = E_Z[D_*|W]
             = δ+(1−δ) Σ w_iab exp(a U_i−a²/2)

라고 놓으면 conditional Jensen으로

    lim M₂,N >= E_W[g(W)² / D_vol(W)].

같은 volatility mixture에 비해 사라지는 one-cell price controls가 극한에서는
payoff 정보를 주지 못하고 역밀도에 잡음을 추가할 수 있다는 뜻이다.
g>0 a.s.이고 현재처럼 일부 양의 b와 양의 weight가 있으면 D_*가 Z에 대해
상수가 아니어서 strict inequality다. 이것은 이산 추정량의 unbiasedness 위반이 아니다.

적용 범위: 고정31template G0에 대한 조건부 결론이다. N과 함께 template 수가
늘어나는 all-monitor 49,141성분 실험에는 이 고정-J 명제를 그대로 적용할 수 없다.
현재 유한 N에서 관측한 실패의 유일한 원인, 혹은 G1의 성능 우위를 증명하지 않는다.

## 3. 고정 physical window: 강수렴이면 second moment도 연속

고정 양의 길이 τ_i를 갖는 normalized window v_i,N이 v_i로 강수렴하고
u_i,N도 강수렴하면 h_iab,N=a u_i,N+b v_i,N→h_iab다.
시점의 grid 반올림 오차가0으로 가고 window 길이가 양으로 유지되어야 한다.
G1의 마지막 point-volatility template은 만기에 의해 τ=T/16으로 잘리지만,
N에 관계없이 양수이므로 허용된다. 나머지 price window는 τ=T/8이다.

L(h)=exp(W(h)−||h||²/2)에 대해 Gaussian moment identity로 정확히

    E|L(h_N)−L(h)|²
      = exp(||h_N||²)+exp(||h||²)−2exp(<h_N,h>) → 0.

고정 유한 mixture이므로 D_N→D in L², 따라서 L¹에서도 수렴한다. 동시에

    |M₂,N−M₂| <= (2/δ) E|g_N−g| + (1/δ²) E|D_N−D| → 0.

증명: 분자를 바꾸는 항에는 |g_N²−g²|<=2|g_N−g|를, 분모를 바꾸는 항에는
|1/D_N−1/D|<=|D_N−D|/δ²를 쓴다. μ_N→μ>0이므로 M₂,N/μ_N²−1도 수렴한다.
이 식은 useful rare-event relative-error rate나 유한 계산량을 보장하지 않는다.
Gaussian exponential의 큰 상수 때문에 위 bound가 실무에서는 매우 느슨할 수 있다.
CI coverage/훈련 효율/고차 moment/최적성/continuum bias는 따로 검증해야 한다.

## 4. 구현상 의미와 반증 기준

- G1의 모든 price direction을 positive-duration window로 한다. 15개 one-cell을
  남긴 초기 geometry-only prototype은 이 명제의 가정을 만족하지 않아 표본 생성 전에 수정했다.
- volatility point/window rows는 실제 BLP operator의 adjoint로 만들고 과거 support를 확인한다.
  nonlinear payoff는 기존 simulator로 평가한다. driver convergence의 분석적 bound는 미완료다.
- 두 grid의 coefficient vector가 비슷하다는 잘못된 비교 대신 Brownian embedding을 쓴다.
  price 좌표의 정확한 dyadic embedding은 coarse coefficient를 두 자식에 1/sqrt(2)씩 분배한다.
  이는 price 방향 전용 검사이며 전체 BLP latent의 embedding 공식을 대체하지 않는다.
- fixed amplitudes/weights 조건을 사용한다. 향후 학습 weights가 grid별로 달라지면
  weights 자체의 수렴 조건을 추가해야 한다. 여기서는 학습을 구현하지 않았다.
- 실제 pilot에서 G1이 실패해도 이 연속성 명제와 모순되지 않는다. 효율과 일관성은 다르다.
  독립 reference agreement 및 새 표본 precision gate 없이 P2/대규모 학습을 열지 않는다.

## 5. oracle 및 문헌 경계

`tests/test_physical_window_guide.py`: dense/FFT identity, exact mixture,
energy/causality, physical-duration price scaling, Brownian price embedding.
`tests/test_physical_window_density_limit.py`: 별도의 2D Gaussian-quadrature toy에서
한-cell shift의 noise limit과 fixed window의 차이를 확인한다. rBergomi 성능 실험이 아니다.

Gaussian stable convergence의 일반 배경은
[Peccati–Taqqu, Stable convergence of multiple Wiener–Itô integrals](https://arxiv.org/abs/math/0604530)를
참조한다. 위 first-chaos 특수 논증과 현재 importance-mixture 결론을 그 논문이
직접 증명한다고 인용하지 않는다. BLP driver scheme의 MSE 이론은
[Bennedsen–Lunde–Pakkanen](https://arxiv.org/abs/1507.03004)의 적용 가정을 확인해야 한다.
[Cotter et al.](https://arxiv.org/abs/1202.0709)의 function-space 관점은 관련 배경이지만,
본 mixture IS의 novelty 또는 저널 게재 가능성을 보장하는 근거가 아니다.
