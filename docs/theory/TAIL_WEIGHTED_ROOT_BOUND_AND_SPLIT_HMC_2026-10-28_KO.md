# Tail-weighted probability-root bound와 Gaussian-split HMC 검토

2026-10-28 America/New_York. 기본 부등식의 새 조합을 현재 target에 전개한 연구
초안이며, 최상위 저널 신규 정리나 작은 상대오차 certificate를 이미 얻었다는 주장이 아니다.

## 1. HMC: 무엇이 보존되는가

유한 Gaussian latent z와 독립 Gaussian momentum p에 대해
`Hβ(z,p)=(||z||²+||p||²)/2−βlogg_N(z)`다. 힘 F(z)는 norm cap을 적용한
∇logg_N이며 target potential에는 cap을 적용하지 않는다.

한 step은 `p←p+εβF(z)/2`, `(z,p)←(cosε z+sinε p,cosε p−sinε z)`,
`p←p+εβF(z)/2`다. shear와 rotation은 Jacobian determinant1이며,
momentum flip R에 대해 symmetric composition S는 `R S R=S⁻¹`를 만족한다.
따라서 `T=R S^L`은 involution이고 absolute Jacobian1이다. 수치 proposal의
full MH ratio `Hβ(old)−Hβ(new)`로 확대공간 target을 보존한다.
deterministic norm cap은 force shear의 이 성질을 없애지 않지만 symplectic gradient
flow나 원래 Hamiltonian의 exact conservation이라고 주장할 수는 없다.

rotation은 Gaussian quadratic energy를 정확한 산술에서 보존한다. 각 kick δ의
quadratic energy 변화 `p·δ+||δ||²/2`만 합하면 큰 차원의 norm끼리 빼지 않고
같은 acceptance ratio를 계산할 수 있다. float64 roundoff는 별도 dense oracle로 점검한다.
constant potential에서는 force0이고 prior rotation acceptance는1이다.

HMC 원리는 기존 연구다. 참고:
[Beskos–Pinski–Sanz-Serna–Stuart 2011](https://doi.org/10.1016/j.spa.2011.06.003),
[공개 원문](https://wrap.warwick.ac.uk/id/eprint/38357/1/WRAP_Stuart_Hybrid.pdf),
[Neal 2011](https://sites.stat.columbia.edu/gelman/bayescomputation/Neal2011.pdf).
이 원문들의 infinite-dimensional 가정/효율 정리를 현재 rough payoff가 자동 만족한다고
주장하지 않는다. 여기서는 유한차원 detailed-balance/MH 논증만 사용한다.

SMC는 fixed β와 unbiased stratified resampling, invariant mutation, carried weights 및
full product normalizer를 유지한다. endpoint sample mean만으로 probability를 추정하지 않는다.
독립 whole가 추론 단위이며 momentum step이나 resampled particles가 독립 반복은 아니다.

## 2. 원래 continuum coupling의 가정

기존 [CDF 오차 문서](CONDITIONAL_CDF_ABSOLUTE_BIAS_BOUND_2026-10-16_KO.md)와
[BLP MSE 문서](ORIGINAL_BLP_INTEGRATED_DRIVER_MSE_BOUND_2026-10-19_KO.md)의
동일 Brownian coupling, centered jointly Gaussian drivers, 자기 variance compensator,
constant ξ>0,0<H<.5,η>0,|ρ|<1,T>0를 유지한다.

`A_N=∫E|Y_t−Y_N(t)|²dt`, `||ΔI||₂≤σI=sqrt(T C_v A_N)`,
`||ΔJ||₂≤σJ=η sqrt(ξ A_N)/2`를 사용한다. 기존 계산 가능한 A_N 상계가 입력이다.
`B_s=(ξT)^(−s) exp[η²T^(2H)(s+s²)/2]`로 양 endpoint의 inverse moments가 지배된다.

## 3. Tail-weighted 미분식에서 확률의 거듭제곱을 보존

Iθ=I_N+θΔI,Jθ=J_N+θΔJ,c=1−ρ²,
sθ=[log(K/S0)+Iθ/2−ρJθ]/sqrt(cIθ),gθ=Φ(sθ),μθ=E gθ다.
chain rule은

`d gθ/dθ = φ(sθ)[(1/(2sqrt(cIθ))−sθ/(2Iθ))ΔI−ρΔJ/sqrt(cIθ)]`.

Iθ>0, `Iθ^(−s)≤max(I^(−s),I_N^(−s))`이므로 `E Iθ^(−s)≤2B_s`다.
이는 보수적 bound이며 θ에 대해 uniform이다. inverse moments와 기존 L² 오차는
미분의 적분가능성을 제공하므로 μθ는 absolutely continuous이고 a.e. 미분할 수 있다.

r>1,q=r/(r−1),p=2q라 놓는다. `K0(p)≥sup_s φ(s)^p/Φ(s)`,
`K1(p)≥sup_s |s|^p φ(s)^p/Φ(s)`인 유한 상수를 아래처럼 explicit하게 만든다.
Hölder에서

`E[φ(sθ)²/Iθ] ≤ (2B_r)^(1/r) K0(p)^(1/q) μθ^(1/q)`,

`E[sθ²φ(sθ)²/Iθ²] ≤ (2B_(2r))^(1/r) K1(p)^(1/q) μθ^(1/q)`.

`(u−v)²≤2u²+2v²`를 적용하면

`C_I = (2B_r)^(1/r)K0^(1/q)/(2c)+(2B_(2r))^(1/r)K1^(1/q)/2`,

`C_J = (2B_r)^(1/r)K0^(1/q)/c`,

`|μθ'| ≤ D_N μθ^a`, `D_N=σI sqrt(C_I)+|ρ|σJ sqrt(C_J)`, `a=1/(2q)`.

μθ>0이고 θ의 compact interval에서 연속이므로 `b=1−a∈(.5,1)`에 대해

**`|μ^b−μ_N^b| ≤ b D_N`**.

이 식은 threshold-independent 전역 bound보다 확률의 power를 남기지만, 현재의
inverse moment 상수가 크면 여전히 무의미할 수 있다. small relative error가 결론은 아니다.

## 4. Gaussian CDF envelope의 명시적 유도

x≥0에서 a=1/(x+1)라 놓고 [x,x+a]만 적분하면

`Φ(−x)≥a φ(x+a)≥exp(−3/2) φ(x)/(x+1)`.

마지막 식은 xa≤1,a²/2≤1/2에서 따른다. x≥0에서 Φ(x)≥1/2다.
`M(m,k)=sup_(x≥0) x^m φ(x)^k`라 하면

`M(0,k)=(2π)^(−k/2)`,

`M(m,k)=(2π)^(−k/2)(m/(ke))^(m/2)` for m>0.

그러므로 p>1,m≥0에 대해 다음은 유효한 explicit upper다.

`sup_s |s|^m φ(s)^p/Φ(s) ≤ max(2M(m,p), exp(3/2)[M(m+1,p−1)+M(m,p−1)])`.

m0와mp를 넣어 K0,K1을 얻는다. 코드의 dense grid 검사는 이 증명의 대체물이 아니라
구현 오류 검출용 oracle다. 상수의 최적화나 새로운 Mills-ratio 정리를 주장하지 않는다.

## 5. 상대 certificate로 바꾸기 위해 여전히 필요한 것

μ_N의 **유효한 확정 또는 확률적 interval**[L,U]가 별도로 있으면, 이론상
`[max(0,L^b−bD_N)^(1/b),min(1,U^b+bD_N)^(1/b)]`로 원래 μ를 감쌀 수 있다.
L=0이거나 root budget이 크면 상대 error를 통제하지 못한다.
관측 mean±normal SE를 distribution-free certified interval이라고 하지 않는다.
float64 계산은 directed rounding enclosure가 아니므로 수치 인증이라는 표현도 제한한다.

이 부등식은 새롭게 얻은 유효한 proof step이지만 현재 실용적 P2 통과를 보장하지 않는다.
고정 r/N scan의 실제 결과, bound가 vacuous한지, 독립 기준 precision과의 구분을 모두 보고한다.
