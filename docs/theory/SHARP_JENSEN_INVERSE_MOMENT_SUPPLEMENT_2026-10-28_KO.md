# 時間평균 Gaussian 분산을 사용하는 Jensen 상계 보강

2026-10-28 America/New_York. 기존 frozen HMC/root-bound protocol을 바꾸지 않는
별도 deterministic 수학 보강이다. 실행 결과를 보고 SMC seed/count/gate를 조정하지 않는다.
원래 μ/BLP coupling과 H=.05,η1.5/2 target을 유지한다.

## 1. 연속시간에서 최대 시점 분산을 쓸 필요가 없는 부분

`Y_t=sqrt(2H)∫₀ᵗ(t−s)^(H−1/2)dW_s`, `Ybar=T⁻¹∫₀ᵀY_t dt`라 하자.
stochastic Fubini 및 Itô isometry로

`Ybar = sqrt(2H)/(T(H+.5)) ∫₀ᵀ(T−s)^(H+.5)dW_s`,

`Var(Ybar) = H T^(2H)/[(H+.5)²(H+1)]`.

평균 point variance는 `T⁻¹∫ VarY_t dt=T^(2H)/(2H+1)`이다.
Jensen에서

`I≥ξT exp[ηYbar−.5η² T^(2H)/(2H+1)]`

이므로 모든 r>0에서

`E I^(−r) ≤ (ξT)^(−r) exp{.5η²[r T^(2H)/(2H+1)+r² Var(Ybar)]}`.

H=.05,T1일 때 Var(Ybar)는 약.15742다. 기존 proof의 최대분산1을 사용하는 것보다
상수는 작아진다. inverse moments의 quadratic-in-r 성질 자체는 사라지지 않는다.

## 2. Original BLP finite grid도 같은 방식으로 정확하게 계산

left driver k=0,...,N−1의 평균은 finite Gaussian linear combination이다.
local Cholesky를 L, historical coefficient를 b_l(kernel_l)이라 하면

`Var(Ybar_N) = 2H/N² * [Σ_(m=0)^(N−2)(L10+L00 Σ_(l=1)^m b_l)²+(N−1)L11²]`.

각 local coordinate의 계수를 먼저 합하고 독립 표준 Gaussian의 variance를 취한 식이다.
평균 point variance는 기존 kernel의 `N⁻¹ Σ_(k=0)^(N−1)Var(Y_N(kh))`다.
이를 위 Jensen 식에 넣으면 discrete inverse moment bound도 계산된다.
finite average variance가 continuous보다 항상 작다고 별도로 가정하지 않는다.
코드는 작은 grid에서 기존 dense driver row의 직접합과 비교한다.

## 3. Interpolation inverse moments의 factor2도 제거

`Iθ=(1−θ)I_N+θI`에서 x↦x^(−r)는 convex다. 따라서

`E Iθ^(−r) ≤ (1−θ)E I_N^(−r)+θE I^(−r) ≤ max(B_N,r, B_cont,r)`.

이것은 `E max(I_N^(−r),I^(−r))≤B_N,r+B_cont,r`보다 작다.
θ별 미분의 expectation을 먼저 제어한 뒤 θ적분을 하므로 사용할 수 있다.
pathwise minimum을 쓴 앞선 식을 근거 없이 단순 교체한 것이 아니다.

## 4. 무엇을 계산하고 어떻게 판정하는가

새 core `sharp_jensen_continuum_bound.py`에서 기존 absolute derivative bound와
probability-root bound 둘 다 이 inverse moments로 다시 유도·계산한다.
첫 absolute 계수는 `sqrt(B1/(2πc))`, 두 번째는 `sqrt(B2)/(2sqrt(2πe))`이며,
각각 `.5σI+|ρ|σJ`, `σI`를 곱한다. 여기의 Br은 위의 endpoint max다.
root bound는 기존 CI/CJ 식의 `(2B_s)^(1/r)`를 이 endpoint max의1/r power로 바꾼다.

고정 r{1.05,1.1,1.25,1.5,2,3,4,8},N{4096,8192,16384},두 셀의48조합을 전부 기록한다.
이전 root/absolute 상수와 비교하되 작은 relative remainder를 실제로 얻었는지 구분한다.
관측 mean은 certified grid interval이 아니며 float64는 directed interval이 아니다.
새 bound가 여전히[0,1]이면 practical relative certificate는 미해결로 남긴다.

이 계산의 source/config/문서를 별도 봉인한다. 기존 frozen source는 수정하지 않는다.
이는 사전 fixed deterministic refinement이며 새 통계 confirmation이나 P2 통과가 아니다.
