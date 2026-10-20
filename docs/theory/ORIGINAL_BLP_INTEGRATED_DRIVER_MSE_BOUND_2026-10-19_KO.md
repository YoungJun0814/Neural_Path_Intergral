# Original BLP left-driver의 명시적 integrated MSE 상한

2026-10-19. 이전 [조건부 CDF 절대 bias 검토](CONDITIONAL_CDF_ABSOLUTE_BIAS_BOUND_2026-10-16_KO.md)에
필요한 A_N=∫_0^T E|Y_s−Y_N(floor(s/h)h)|²ds의 한 보수적 상한을 제공한다.
동일 Brownian coupling,κ=1,exact local singular integral 및 historical **cell mean**
weights인 현재 original BLP에 한정한다. RMS moment-matched 대체 kernel에 적용하지 않는다.
새로운 weak-error rate/희귀 확률 상대 오차 보증/독창성 주장으로 해석하지 않는다.

## 1. 연속 driver의 increment

α=H−1/2,0<H<1/2, Y_t=sqrt(2H)∫_0^t(t−r)^α dW_r다.
δ=s−t>0에 대해 Itô isometry와 변수변환으로

    E|Y_s−Y_t|²
      = δ^(2H) [1+2H∫_0^(t/δ)((x+1)^α−x^α)²dx].

0<x<1에서는 차이의 절댓값<=x^α, x>=1에서는 mean-value theorem으로
절댓값<=|α|x^(α−1)이다. 따라서

    E|Y_s−Y_t|² <= C_inc δ^(2H),
    C_inc = 2 + H α²/(1−H).

이는 일부러 느슨한 닫힌 상수다. t=0에서도 적용된다.

## 2. grid point에서 original BLP projection error

마지막 local cell은 exact하므로 오차가 없다. 나머지 cell의 kernel은 평균으로
대체한다. Gaussian Itô integral에서 이 평균은 상수함수로의 L² projection이다.
단위 cell [l,l+1],l>=1에서 f(x)=x^α에 대해, 독립 Uniform X,X'를 쓰면

    ∫f²−(∫f)² = Var(f(X))
        = E|f(X)−f(X')|²/2
        <= sup|f'|² Var(X) = α² l^(2H−3)/12.

Brownian cell들 간의 독립성과 scale변환을 이용해

    e_k² = E|Y_(kh)−Y_N(kh)|²
        <= (H α²/6) h^(2H) Σ_(l=1)^(k−1) l^(2H−3)
        <= C_proj h^(2H),
    C_proj = (H α²/6) [1+1/(2−2H)].

마지막 부등식은 감소함수에 대한 series<=1+∫_1^∞ bound다. k=0,1의 projection
error는0이며 같은 상한으로 지배된다. 원래 cell-mean BLP에서는 projection
직교성으로 e_k²=Var(Y_(kh))−Var(Y_N(kh))이기도 하다. 다른 covariance coupling이나
다른 kernel의 variance deficit을 이 error와 동일시하면 안 된다.

## 3. 시간 적분

s∈[kh,(k+1)h)에서

    |Y_s−Y_N(kh)|² <= 2|Y_s−Y_(kh)|²+2|Y_(kh)−Y_N(kh)|².

expectation과 cell 적분을 합하면

    A_N <= T h^(2H) [2 C_inc/(2H+1) + 2 C_proj].

`src/path_integral/blp_driver_error_bound.py`는 이 식만 계산한다. empirical error,
fit된 weak rate, Monte Carlo estimator를 계산하는 코드가 아니다. 입력과 overflow 범위를 검사한다.
반환값은 float64 근사 평가이며 directed-rounding interval certificate는 아니다.
양의 해석적 값이0으로 underflow한 경우도 허용하지 않는다.

## 4. rare-event 해석의 한계

H=.05에서 h^(2H)=h^.1은 매우 느리게 작아진다. terminal grid-point variance deficit가
작더라도 각 cell 안에서의 rough driver 변화까지 작다는 뜻은 아니다. 위 A_N 상한은
grid projection error와 within-cell left-point error를 함께 지배한다.

이는 이전 조건부 CDF 절대 bias bound에 넣을 수 있는 유효한 가정상 입력이다.
하지만 상수는 느슨하고 μ가10^-8 정도면 유용한 relative-bias 보증이 되지 않을 수 있다.
상한이 작지 않다는 사실은 실제 bias가 크다는 lower bound도 아니다. 따라서 이 식으로
P2를 통과시키거나 observed adjacent difference를 continuum certificate로 바꾸지 않는다.

검증은 increment integral 직접 quadrature, original BLP variance/projection identity의
작은 grid 대조, grid scaling 및 N=1 exact MSE와의 비교다. 이것은 독립적인 전체
연속시간 rare-event oracle를 대신하지 않는다.

## 5. 앞선 consistency 가정과의 연결

고정 H,T,η,ξ 및 |ρ|<1에서 앞선 conditional-CDF bound의 inverse-moment/derivative
상수는 N과 무관하다. 두 driver는 centered jointly Gaussian이고, original BLP projection의
variance는 exact variance보다 크지 않아 공통 M=T^(2H)를 쓸 수 있다. 각 volatility에
자기 variance compensator를 쓰고 동일 W의 adapted left-point Itô 적분으로 가격을
구성한다는 가정 아래, 위 A_N 상한과 앞선 유도를 결합하면

    E|g_N−g| <= C sqrt(A_N) <= C' h^H,
    |μ_N−μ| <= E|g_N−g|.

이는 **유도된 보수적 strong-type absolute-error upper rate**다. 실제 weak asymptotic
rate가H라고 단정하거나 유한 N의 상대 오차에 맞춰 rate를 fit한 것이 아니다.
상수는 극소 μ에 비해 매우 클 수 있다.

고정 양의 시점의 point-volatility representer는 projection error와 시점 반올림의
increment bound로 L²[0,T]에서 수렴한다. 고정 길이의 averaged-volatility window도
같은 bound와 mean-square continuity에 의해 수렴하며, limit variance가 양수이므로
normalization을 유지할 수 있다. 양의 physical-duration price window는 Brownian
L² norm에서 수렴한다. 따라서 고정 finite dictionary/고정 amplitudes와 weights라는
범위에서는 앞선 density-consistency 문서의 representer 및 payoff 수렴 가정을
이 original BLP coupling으로 뒷받침할 수 있다.

이 결론은 학습 weights의 N별 수렴, task-uniform 성질, H→0의 균등 상수, 효율 우위,
희귀 확률 relative-error certificate를 주지 않는다. 한-cell price 방향에는 strong
representer convergence 가정이 성립하지 않아 같은 명제를 그대로 적용할 수 없다.
이 proof chain은 연구 초안이며 별도 수학 검토와 선행연구 novelty 검토가 필요하다.
