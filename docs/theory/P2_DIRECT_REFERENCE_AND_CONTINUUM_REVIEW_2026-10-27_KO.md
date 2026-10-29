# P2 직접 구현의 이론 검토와 연속시간 인증의 미해결 의무

2026-10-27 America/New_York. 이 문서는 새로운 일반 정리나 P2 통과 선언이 아니다.
실행 전 계약은 [P2 직접 실행계획](../plans/P2_DIRECT_CLOSURE_EXECUTION_2026-10-27_KO.md)이다.

## 1. 계산 대상과 Gaussian augmentation

현재 확률은 원래 BLP 유한격자에서 가격의 독립 Gaussian 난수를 적분한
`μ_N=E_p[g_N(Z)]`, `0<g_N≤1`이다. fine N4096의 local 좌표8192개와
coarse coupling용 추가 Gaussian2048개를 합쳐10240개 표준 Gaussian 좌표를 사용한다.
기존 adapter가 coarse N2048의 원래 marginal law를 정확한 선형 coupling으로 만든다.

독립 기준의 target은 `H=(g_f+g_c)/2`다. 다른 volatility kernel이나 threshold로
쉽게 만드는 것이 아니다. fine/coarse 각각의 확률은 같은 p에서의 기대값이다.

## 2. Cyclic pair MH의 정합성

M=dimension/2개의 pair를 row별 uniform shift s∈{0,...,M−1}로 회전시킨다.
변환 T_s는 orthogonal permutation, `T_s⁻¹=T_(M−s mod M)`이고,
표준 Gaussian p에 대해 `p(T_s z)=p(z)`, absolute Jacobian1이다.
역 shift도 같은 확률1/M이므로 proposal은 p-reversible이다.

target `π_β(dz)∝p(z)H(z)^β dz`에서 acceptance는

`min(1,exp(β(logH(T_s z)−logH(z))))`

이다. forward/reverse flux가 둘 다
`p(z) min(H(z)^β,H(T_s z)^β)/M`로 같아진다.
따라서 이 이동과 별도로 정합성이 확인된 gradient MH 이동은 각각 π_β-invariant이고,
그 합성도 invariant다. 합성 전체가 reversible이라고 추가 주장할 필요는 없다.

이는 **Volterra 물리 경로의 시간 평행이동 대칭이 아니다.** 물리 확률은 바뀌므로
반드시 H를 다시 계산해 MH 보정한다. augmentation과 미사용 좌표를 포함해도
공동 p가 표준 Gaussian이라는 유한차원 논증은 같다.

원시 난수 전체를 움직이는 정적 Gaussian-space MCMC이므로 adapted drift control이나
연속시간 Girsanov theorem을 주장하지 않는다. 미래 좌표도 섞는 이 proposal을
인과적인 SDE controller라고 부르면 잘못이다. 본래 P에 따른 가격 적분은 기존
predictable left-point law 그대로 평가한다.

permutation은 latent norm과 기존 값의 집합을 보존하므로 새 희귀 크기를 만들지는 못한다.
gradient Gaussian 이동을 함께 두는 이유다. 두 좌표를 함께 옮기는 선택도 표현력 제한이다.
좋은 acceptance가 독립 모드 탐색이나 분산 감소를 보장하지 않는다.

## 3. SMC normalizer와 endpoint 복원

고정 β schedule에서 `πβ∝pH^β`, 가중치 carry, unbiased stratified resampling,
각 β-invariant mutation을 사용한다. 최종 component 확률 추정은

`μhat_j = Zhat Σ_i W_i g_j(Z_i)/H(Z_i)`

이다. 전체 product normalizer Zhat를 빼고 normalized endpoint mean만 사용하면 안 된다.
component/umbrella ratio는2이하이고 다음 identity가 성립한다.

`(μhat_f+μhat_c)/2=Zhat`.

이 identity는 모든 whole에서 검사한다. 다만 identity가 맞는 것만으로 μhat의 tail coverage나
요구 정밀도가 확보되지는 않는다. SMC의 통계 단위는 particle이 아니라 독립 whole다.
새 방법은 learned model 개선이나 최적 제어의 신규 수학 원리라고 주장하지 않는다.

## 4. Signed adjacent correction의 경계

각 독립 whole에서 복원한 μhat_f,μhat_c로 `(μf−μc)/μf`를 평가하고,
같은 whole 내 covariance를 포함한 paired delta-method SE를 사용한다.
서로 독립 표본인 것처럼 두 SE를 더하지 않는다. 4개 cell/arm 비교를 조정한다.

작은 adjacent correction은 유한격자 간 비교다. 이것만으로 무한 telescoping tail이
작다는 사실이 따라오지 않는다. normal/delta interval은 소수 whole 및 미관측 tail에
취약할 수 있으므로 distribution-free certificate라고 부르지 않는다.

## 5. 기존 absolute continuum bound의 실행 가능한 연결

동일 Brownian coupling의 original BLP left-driver에 대해 기존 문서의
`A_N≤C_A N^(−2H)`와 `|μ−μ_N|≤C_CDF sqrt(A_N)`를 그대로 연결했다.
Gaussian mgf/Jensen/Itô isometry/CDF derivative의 가정은 기존 두 문서에 명시돼 있다.
새 module은 그 식의 log-domain 평가이며, 더 날카로운 정리를 새로 증명한 것이 아니다.

- [Conditional-CDF 절대 bias 유도](CONDITIONAL_CDF_ABSOLUTE_BIAS_BOUND_2026-10-16_KO.md)
- [Original BLP integrated-driver MSE 상한](ORIGINAL_BLP_INTEGRATED_DRIVER_MSE_BOUND_2026-10-19_KO.md)

N4096에서 unclipped absolute upper는 canonical 약68.83,high-eta 약5207.37이다.
확률의 범위로 자르면 둘 다1로, 현재 희귀 확률의 상대 bias를 인증하기에는 무의미하다.
N8192/16384에서도 clipped upper는1이다.

관측 μ의10%를 이 느슨한 식으로 맞추는 진단 N 규모는 약10^217.59/10^218.10이다.
**실제 모형에 그 정도 격자가 필요하다는 결과가 아니다.** 현재 부등식의 상수가
너무 느슨하다는 진단이며, 실제 bias의 lower bound나 현실적인 계산 자원 요구가 아니다.
관측 μ도 참μ의 certified lower bound가 아니므로 이 숫자는 상대 인증이 아니다.

## 6. 문헌 정리를 바로 대입할 수 없는 이유

[Friz–Salkeld–Wagenhofer, Annals of Applied Probability 2025](https://doi.org/10.1214/24-AAP2109)의
[원문 Theorem1.1](https://arxiv.org/pdf/2212.01591)은 polynomial test functions와
정확한 Volterra driver를 이용한 left-point scheme에 관한 weak-rate 결과다.
현재 BLP kernel의 digital downside probability에 명시적인 유한-N 상대오차 상수를
직접 제공하지 않는다. 기존 정리의 대상·scheme·상수 확인 없이 H=.05를 rate식에
대입하는 것은 충분한 증명이 아니다.

[Bonesini–Jacquier–Pannier v4(2026-05)](https://arxiv.org/abs/2304.03042)의 요약은
5회 미분 가능한 test function의 weak rate를 다룬다. 현재 target의 조건부 CDF가
latent coordinates에서 매끄럽다는 사실만으로 그 정리의 scalar test function 조건,
BLP driver 오차, 명시적 finite-N 상수가 모두 충족됐다고 결론내릴 수 없다.
이 논문을 이번에 현재 모형에 적용하는 완결 증명을 수행한 것은 아니다.

[Gassiat의 hybrid weak-error 연구](https://doi.org/10.1137/22M1485760)도 관련되지만,
초록이 명시하는 특수 integrand/test-function 조건을 무시해 현재 nonlinear digital target의
명시적 상대오차 certificate로 가져오지 않는다.

이는 가능한 다른 정리가 전혀 없다는 문헌 전체에 대한 불가능성 주장이 아니다.
검토한 정리만으로 현재 필요한 수치 인증이 확보되지 않았다는 판단이다.

## 7. 유한 mesh 관측만으로 remainder를 정할 수 없는 논리적 이유

어떤 유한한 mesh prefix `μ_(N1),...,μ_(NL)`가 주어졌다고 하자.
추가 구조를 주지 않으면 이 prefix를 똑같이 갖고도 서로 다른 limit a,b로 수렴하는
확률값 수열을 만들 수 있다. prefix 뒤에서 각각 a,b를 상수로 두면 된다.
두 수열은 각자의 limit에 대해 충분히 뒤에서는 오차0이므로, 상수와 시작지점이
명시되지 않은 어떤 asymptotic O(N^−α) 성질과도 양립한다.

따라서 유한 개 mesh 차이와 알려지지 않은 상수의 asymptotic rate만으로
`|μ−μ_NL|/μ`의 수치 상한을 결정할 수는 없다.
이 구성은 실제 rBergomi sequence가 임의로 바뀐다는 뜻이나 모형 자체의 불가능성
정리가 아니다. 현재 근거에서 빠진 가정이 무엇인지 보이는 논리적 반례다.

P2 전체 해결에는 현재 target/scheme에서 유효하고 계산 가능한 remainder bound,
또는 정당하게 검증된 다른 연속시간 인증 방법이 여전히 필요하다. 코드·반복 수·GPU를
추가하는 것만으로 이 수학적 의무가 자동으로 충족되지는 않는다.

## 8. Randomized multilevel debiasing도 조건 확인 없이 우회로가 되지 않는다

연속시간 추정량을 직접 unbiased하게 만들 수 있다면 finite-N bias를 별도로
인증하지 않는 다른 경로를 검토할 수 있다. 관련 출발점은
[Rhee–Glynn, Operations Research 2015](https://web.stanford.edu/~glynn/papers/2015/RheeG15.html)의
[원문](https://web.stanford.edu/~glynn/papers/2015/RheeG15.pdf)이다. 다만 그 정리의
현재 nonlinear Volterra digital target 적용을 이번에 입증한 것은 아니다.

단일 random level L에 대해 `P(L=l)=p_l>0`, coupled correction을 Δ_l,
`v_l=E[Δ_l²]`, 평균 작업량을 c_l이라고 하면, single-term correction의 second moment와
평균 작업량에는 각각 `Σ v_l/p_l`, `Σ p_l c_l`가 들어간다. Cauchy–Schwarz로

`(Σ sqrt(v_l c_l))² ≤ (Σ v_l/p_l)(Σ p_l c_l)`

이므로 왼쪽이 발산하면 이 single-term 구성에서 두 양을 모두 유한하게 만들 수 없다.
예를 들어 **실제 양쪽 비교 가능한 decay가** `v_l=Θ(2^(−βl))`,
`c_l=Θ(2^(γl))`이고 β≤γ라면 이 문제가 생긴다. 이것은 현재 모형의 β를
측정하거나 증명한 결과가 아니며, 다른 coupling/estimator의 불가능성 정리도 아니다.

현재 proof chain이 주는 것은 `E|g_N−g|=O(N^(−H))`의 보수적 상계다.
0≤g≤1에서 얻을 수 있는 second-moment upper bound만으로는 작업량 증가보다
충분히 빠른 correction decay가 증명되지 않는다. **느린 upper bound만 있다는 이유로
실제 variance가 반드시 느리게 감소한다고 결론내릴 수는 없다.** 반대로 좋은
유한-level fit만 보고 infinite-level finite variance/work가 확보됐다고 할 수도 없다.

따라서 randomized debiasing을 다음 생산 estimator로 넣으려면 최소한 다음이 필요하다.

1. 현재 original BLP/조건부 payoff에서 연속시간 limit과 telescoping의 정당성.
2. 모든 충분히 높은 level에 유효한 correction second-moment 통제와 작업량 통제.
3. 선택한 random-level distribution에서 variance·평균 작업량·메모리의 유한성.
4. 수준 truncation이나 timeout으로 남기는 bias를 숨기지 않는 실행 계약.
5. 새 독립 전체 estimator 반복의 precision 및 tail-contribution 진단.

원문의 높은 strong-order 조건을 H=.05의 driver에 그대로 대입하거나,
truncated randomization을 연속시간 unbiased라고 부르는 구현은 하지 않는다.
이 경로는 검토 가능한 연구 대안이지만 지금 P2가 해결됐다고 선언할 근거는 아니다.

## 9. 다음 proof가 제어해야 할 양: tail-weighted 미분 모멘트

기존 bound를 단순히 더 촘촘하게 계산하는 대신, 어느 항에서 희귀 사건 크기를
잃는지 명시한다. 아래는 동일 Brownian coupling에서의 미적분 항등식과
Cauchy–Schwarz에 의한 **조건부 연구 축약**이며, 새로운 작은 수치 상계는 아니다.

`ΔI=I−I_N`, `ΔJ=J−J_N`, `Iθ=I_N+θΔI`, `Jθ=J_N+θΔJ`,
`c=1−ρ²`, `aθ=log(K/S0)+Iθ/2−ρJθ`, `sθ=aθ/sqrt(cIθ)`라 놓는다.
I와I_N은 양수이므로 Iθ도 양수다. Φ의 pathwise chain rule에서

`g−g_N = ∫₀¹ φ(sθ) [Aθ ΔI − ρ ΔJ/sqrt(cIθ)] dθ`,

`Aθ = 1/(2sqrt(cIθ)) − sθ/(2Iθ)`

를 얻는다. 따라서 Ω×[0,1]의 product measure에서 Cauchy–Schwarz를 쓰면

`|μ−μ_N| ≤ sqrt(C_I(N)) ||ΔI||₂ + |ρ| sqrt(C_J(N)) ||ΔJ||₂`,

`C_I(N)=E∫₀¹ φ(sθ)² Aθ² dθ`,

`C_J(N)=E∫₀¹ φ(sθ)²/(cIθ) dθ`.

이 항등식/부등식은 양 endpoint의 inverse moments 및 기존 L² 오차 통제로
적분가능성을 정당화할 수 있다. 예를 들어 `Iθ⁻¹≤max(I⁻¹,I_N⁻¹)`이고
`sup_s φ(s)`, `sup_s |s|φ(s)`가 유한하다. θ interpolation은 random variable의
결정론적 보간일 뿐, 비적응 stochastic integral를 새로 정의한 것이 아니다.

기존 absolute bound는 φ(sθ), |sθ|φ(sθ)를 전역 최댓값으로 바꾸고 inverse moments를
적용한다. 그러면 tail-specific 작은 값이 사라진다. 위 식은 이를 보존하지만,
**C_I,C_J가 실제로 충분히 작다는 인증은 아직 없다.** 두 endpoint 중 하나가
연속시간이므로 유한격자 표본의 plug-in 평균으로 이것을 참 상계라고 할 수 없다.

이 방향을 사용하려면 다음을 추가로 증명해야 한다.

- rare region과 그 complement를 모두 덮는, 현재 threshold와 η에 유효한
  C_I/C_J의 계산 가능한 상계. 관측된 희귀 표본만 조건부로 남기면 안 된다.
- 확률의 certified lower bound 또는 그것과 동등한 정당한 상대오차 변환.
- 원래 BLP와 연속시간 coupling을 유지한 ΔI/ΔJ의 실제 충분한 상계.
- 수치 평가를 인증이라고 부르려면 필요한 rounding/enclosure 통제.

chain-rule 식은 독립 autograd/dense θ quadrature oracle로 점검한다. 이 검사는
유한한 임의 endpoint에서의 미분식을 확인할 뿐, C_I/C_J의 모형별 값이나
연속시간 relative-bias certificate를 증명하지 않는다. 원리 자체는 기본 미적분과
모멘트 부등식의 조합이며 논문의 신규 정리라고 주장하지 않는다.
