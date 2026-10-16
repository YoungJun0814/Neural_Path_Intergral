# P1 recovery / P2 numerical-contract 재검토

작성일 2026-10-10. 기존 계획의 P1/P2 accuracy gate를 바꾸지 않는다.
본 문서는 구현 식·oracle·재현에 근거한 검토이며 모든 floating-point 입력에 대한
formal verification이나 실제 rare-event tail coverage 증명이 아니다.

## 1. 추정량과 reference 역할

p=N(0,I_2N), g=독립 가격 Brownian을 적분한 terminal downside conditional probability.
q는 고정된 사건 추정 proposal, r는 auxiliary/reference이다. Dq=q/p, Dr=r/p.
μ=E_r[g/Dr], M2(q)=E_r[g²/(Dq Dr)]. final estimator는 평범한 표본 평균이다.
SMC는 μ용 H=g 또는 risk용 H=δq*g²/Dq<=1의 정규화 상수를 추정한다.
후자만 마지막에 δq로 나눈다. auxiliary reference 개선은 q의 분산 개선과 다르다.

정적 shift, collective covariance, conditional-excursion, radial mixture의 density를
전부 합산한다. sample label의 단일 density나 empirical-normalization으로 대체하지 않는다.
defensive floor는 모든 공간에서 support를 보장하나 상대 정밀도나 mode discovery를 보장하지 않는다.

## 2. Collective covariance / 조건부 excursion

u_t는 과거 Volterra driver의 standardized linear projection, v_t는 다음 Brownian white
coordinate다. u_t·v_t=0 검사로 두 projection의 prior independence를 확보한다.
rank-two covariance inflation λ=25일 때 component/p는
λ^-1 exp[(1-1/λ)((u^Tz)²+(v^Tz)²)/2]. inverse λ는 rank2 determinant에서 나온다.
sampler는 두 orthogonal projection을 sqrtλ로 확대하며 여공간을 유지한다.

조건부 성분 p(.|u^Tz>=a)의 density는 indicator/Φ(-a)다.
joint 성분은 u/v 독립성이 있어 indicator/[Φ(-a)Φ(-b)]다.
sampling은 exact truncated-normal inverse CDF로 projection을 바꾸며 clipping하지 않는다.
중첩한 성분을 전부 합산한다. 원래4/5/6 조건부 mixture의 최대 기여 경로는 maxY=3.71754로
그 집중 영역 바깥이었음이 exact replay로 확인됐다. 이는 support가 0인 문제가 아니라
defensive natural component만 충분히 밀도를 공급하는 coverage-concentration 문제다.

## 3. Envelope repair와 발견된 기술 오류

envelope=.4*excursion(2/3/4/5/6)+.4*collective25+.2*(.1p+.9N(0,2I)).
각 component floor.1으로 전체 floor.1을 유지한다. radial logdensity inflation은
-d/2 log2+||z||²/4다. radial sample scale sqrt2는 반드시 float64로 만든다.

v1 구현의 scalar `torch.where(mask,sqrt2,1)`은 float32를 생성했다. density와 sampler의
아주 작은 분산 불일치도 exact-law 계약 위반이므로 실행을 중단하고 prefix를 무효 처리했다.
v2는 명시적 float64 scale을 사용하고 독립 PRNG reconstruction의 radial subset에 대해
bitwise equality regression을 추가했다. 실패 artifact와 interrupted reservation을 삭제하지 않았다.
같은 문제가 안 생긴다고 모든 future dtype/path 입력을 보장하지는 않는다.

## 4. SMC bridge와 MH ratio

Geometric natural base: Fβ=p H^β, incremental logpotential=Δβ logH.
Geometric proposal base: Fβ=r^(1-β)(pH)^β,
increment=Δβ(logH-logDr), Z0=1, Z1=E_pH.
prior-reversible pCN ratio는 βΔlogH+(1-β)ΔlogDr(proposal base)이고,
natural base에서는 βΔlogH다. r-independence ratio는 proposal base에서
βΔ(logH-logDr), natural base에서 βΔlogH-ΔlogDr다.

Arithmetic Fβ=(1-β)b+β pH/c, b=p 또는 r, c>0는 이전 pilot에서 동결한다.
increment=log(Fright/p)-log(Fleft/p), endpoint Z1=E_pH/c.
따라서 마지막에 c를 곱한다. c가 참값과 달라도 target 자체는 바뀌지 않는다.
pCN MH는 Δlog(Fβ/p), global-r MH는 여기에서 ΔlogDr를 뺀다.
geometric ratio를 arithmetic bridge에 재사용하면 오류다. 별도 함수 및 known endpoint
c=.1×/10× oracle로 확인한다.

고정 resampling schedule와 unbiased stratified resampling, invariant mutations 하의
product normalizer를 사용한다. 같은 whole SMC의 입자는 독립 inference unit이 아니다.
whole-run replication이 표준오차 단위다. adaptive β/현재 run의 c 재조정/실패 seed 교체는 없다.
알려진 SMC 원리의 적용이며 새 일반 정리로 주장하지 않는다.
문헌: [Del Moral–Doucet–Jasra](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).

## 5. Directional mutation

U=(u,v), U^TU=I에서 z'=z+U[(sqrt(1-s²)-1)U^Tz+sε]는 prior-reversible rank2 pCN이다.
현재 state와 독립으로 균등 선택한 direction mixture도 prior-reversible이다.
따라서 위 bridge의 pCN MH ratio를 그대로 적용한다. 현재 z를 보고 좋은 방향을 선택하면
proposal ratio를 새로 유도해야 하며 현재 증명/구현 범위 밖이다.
prior covariance/cross covariance 및 bounded multimodal oracle로 확인한다.

## 6. P2 exact adjacent coupling

fine first cell의 (W1,L1)을 유지하고 C1 residual을 독립 Gaussian으로 추가한다.
coarse W=W1+W2, L=C1+L2다. L1+L2가 아니다.
coarse whitening z_c=A z_f+Bε에서 AA^T+BB^T=I를 확인한다.
fine proposal r에서 z_f를 생성하고 ε는 원래 conditional prior로 생성하므로
joint likelihood는 p_f/r_f 하나다. g_c를 r_c density로 별도로 나누면 틀리다.
E_r,ε[(g_f-g_c)/Dr_f]=μ_f-μ_c, 각 marginal expectation도 정확한 grid target이다.
dense adapter와 O(paths*N) block adapter가 동일한 결과를 내는 회귀를 수행했다.

FFT excursion density는 exact BLP row geometry와 동일한 linear convolution을 계산한다.
kernel law나 model parameters를 바꾸지 않는다. dense monitoring operator와 density/sampler
equivalence를 확인했다. threshold boundary에는 floating-point 판정이 있으므로 모든
적대적인 near-boundary 입력에 대한 exact-real identity까지 주장하지 않는다.

## 7. Price/driver 분해의 의미와 금지되는 수정

F=fine payoff, C=coarse BLP payoff, M=fine volatility를 coarse left times에서 취한
coarse-price Euler diagnostic payoff라 놓으면 F-C=(F-M)+(M-C)다.
세 payoff는 같은 fine likelihood로 평가한다. stable signed-CDF arithmetic과
samplewise triangle identity 및 η≈0의 constant-volatility limit을 확인했다.
M은 원래 coarse law도 최종 생산 모델도 아니다. 분해는 선택한 순서에 의존한다.
두 correction은 상관되어 있어 SE를 독립이라고 합산하지 않는다.

관측 high-eta에서는 price-quadrature 변화가 driver 근사 변화보다 컸다.
그렇다고 미래 volatility를 Brownian increment 앞에 놓는 naive trapezoidal SDE 수정은 허용되지 않는다.
volatility stochastic integral은 adaptedness와 Itô convention을 유지해야 한다.
우선 안전한 개선은 더 촘촘한 nested grid 및 해당 새 finite-grid q/bank/reference의 별도 검증이다.
비-uniform grid/bridge quadrature를 도입하려면 Gaussian covariance·compensator·likelihood와
conditional price integration을 새로 유도해야 한다. N32 q나 bank를 N512 인증 자산으로 쓰지 않는다.

BLP driver의 mean-square consistency는 극단 tail의 상대 payoff accuracy를 자동 보장하지 않는다.
문헌: [Bennedsen–Lunde–Pakkanen hybrid scheme](https://arxiv.org/abs/1507.03004).

## 8. 통계적·연구적 범위

관측 RSE와 normal delta-method equivalence는 distribution-free confidence certificate가 아니다.
bootstrapping 8 whole-run로 미관측 rare tail을 보증할 수 없다. 실제 larger-count run이
이 한계를 드러냈으므로 최대 기여 비중·whole-block sensitivity·독립 mechanism을 함께 유지한다.
개발 중 반복된 설계 선택은 journal confirmation과 분리해야 한다.
pytest/lint/mypy/replay 성공은 estimator variance·continuum accuracy·새 모델의 우위를 뜻하지 않는다.

## 9. FFT envelope와 guide-free block-pCN 추가 검토

FFT envelope는 위2절의 standardized monitor 값을 FFT convolution으로 계산하고
동일한 excursion/collective/radial density를 합산한다. sampler는 corrected-f64 envelope를
그대로 호출한다. analytical row 생성은 O(N²) memory이고 scoring만 O(paths*N logN)다.
전체 방법을 memory-linear라고 주장하지 않는다. N16/32/64에서 dense geometry,
density, sample이 numerical tolerance 내 일치했다. 일부 component의 threshold
boundary 판정에 대한 모든 실수 입력의 bitwise identity는 보장하지 않는다.

guide-free block-pCN은 fixed coordinate projection P에 대해
z'=(I-P)z+sqrt(1-s²)Pz+sPε이다. p 아래 (z,z')의 교환 대칭 Gaussian law로
prior reversibility를 얻는다. 독립적으로 선택한 block mixture에도 같은 detailed
balance가 성립한다. 따라서 πβ∝pH^β MH acceptance는 min(1,H(z')^β/H(z)^β)다.
full kernel과 block kernel의 deterministic composition은 각 kernel의 invariance로
πβ를 보존한다. composition 전체의 reversibility가 필요하지 않다.
블록은 선택한 크기2/8/32/d의 aligned uniform block이고 현재 state/target과 독립이다.
현재 구현은 power-of-two d>=2로 제한한다. 일반 d, adaptive target-based block 선택,
gradient-selected directions에는 이 계약을 자동 적용하지 않는다.

natural-start envelope SMC와 이 guide-free kernel을 구분한다. 전자는 초기법칙만 p고
guided mutation을 쓴다. 후자는 원래 좌표의 prior kernel만 사용한다. 낮은 ancestry는
genealogical label의 단조 감소 진단이며, 높은 MH acceptance도 충분한 mixing 증명이 아니다.
