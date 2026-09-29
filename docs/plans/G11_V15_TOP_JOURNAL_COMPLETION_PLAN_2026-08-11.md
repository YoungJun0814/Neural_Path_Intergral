# G11 V15 최상위 저널 완성 계획

Date: 2026-08-11

Working title: **Exact Conditional Cameron--Martin Transport for Rare Events in
Gaussian Volterra Models**

Status: V14 독립-seed qualification 이후의 연구·구현 계획. 이 문서는 아직
V15 결과나 정리를 주장하지 않는다.

## 0. 결론부터

V14는 박사과정 working-paper core로서는 강하다. 그러나 현재 증거만으로
`Mathematical Finance`, `Finance and Stochastics`, 또는 확률·수치해석 분야의
최상위 전문 저널에 제출하는 것은 이르다.

현재 확보한 것은 다음과 같다.

- 128-step BLP rBergomi finite-grid law에서 terminal left-tail 확률을 정확한
  ordinary importance sampling으로 추정한다.
- 독립 price driver 전체를 조건부 적분하므로 최종 추정량은 hard indicator보다
  작거나 같은 분산을 갖는다.
- 자연분포 0.2와 학습된 평균이동 0.8의 balance mixture를 사용하여
  `p/q <= 5`를 보장한다.
- 세 개의 사전 고정 cell과 100-query 상각 조건에서 V14는 정확도-qualified
  최강 비교법 대비 development `2.3574x`, 독립 qualification `2.4159x`의
  training-inclusive work 개선을 보였다.
- 전체 테스트 956개와 독립 결과 감사가 통과했다.

현재 부족한 것은 다음 네 가지다.

1. **새 수학 원리:** SMC 평균이 왜 좋은 proposal인지에 대한 일반 정리나
   희귀도에 따른 효율 정리가 없다.
2. **연속시간 연결:** finite-grid exactness는 있으나 rBergomi 연속시간 확률에
   대한 bias/rate/complexity 정리가 없다.
3. **일반성:** 세 cell, terminal event, 한 parameter tuple, 한 grid에 한정된다.
4. **독창성 경계:** 조건부 로그정규 적분, numerical preintegration,
   low-dimensional CEM, large-deviation proposal은 각각 선행연구가 있다.

따라서 V15의 중심 기여는 단순한 신경망 추가가 아니다. 다음 명제를 논문의
중심으로 삼는다.

> Gaussian Volterra 금융모형에서 독립 price driver를 정확히 조건부 적분한
> 뒤, 조건부 rare-event density의 Cameron--Martin 지배경로와 유한랭크
> 곡률을 이용해 mesh-consistent defensive transport를 만들고, proposal
> 근사오차와 상대분산을 연결하며, 여러 금융 질의에서는 구조보존 neural
> operator로 이 transport를 상각한다.

신경망은 proposal을 예측하는 가속기일 뿐이다. 어떤 network 출력도 exact
balance likelihood로 보정되므로 추정량의 unbiasedness는 network 정확도에
의존하지 않는다.

## 1. 목표 저널과 제출 조건

### 1.1 최우선 목표: Mathematical Finance

이 경로는 다음 조건을 모두 만족할 때만 연다.

- 연속시간 Gaussian Volterra 조건부 표현과 finite-grid 수렴 정리가 완성된다.
- rare-event scaling에서 logarithmic 또는 bounded-relative-error 효율 정리가
  완성된다.
- rBergomi에서 위 일반 정리의 가정을 실제로 검증한 corollary가 있다.
- deep OTM digital/put 또는 tail-risk 계산에서 명확한 금융적 insight를 준다.
- 최신 novelty audit와 외부 수학 검토가 통과한다.

`Mathematical Finance`는 계산 논문도 받지만 수치 실험이 이론적 발전을
뒷받침해야 하며, routine application보다 방법론적 독창성과 금융적 통찰을
요구한다. V14 숫자만 늘리는 방식은 이 조건을 충족하지 못한다.

### 1.2 동급 또는 강한 대안

- **Finance and Stochastics:** 확률론·rough-volatility 정리가 주기여이고
  알고리즘은 corollary일 때.
- **SIAM Journal on Scientific Computing:** Volterra 이외의 Gaussian-process
  rare event까지 확장되고, 수치 알고리즘·복잡도·재현성이 주기여일 때.
- **SIAM Journal on Financial Mathematics:** 금융수학 정리와 계산 breakthrough는
  있으나 최상위 확률 정리 또는 광범위 scientific-computing 일반성이 부족할 때.

### 1.3 투고 경로를 낮춰야 하는 조건

다음 중 하나라도 발생하면 `Mathematical Finance`용 표현을 금지한다.

- 연속시간 또는 rare-event 효율 정리가 실패한다.
- V15가 정확도-qualified best baseline을 안정적으로 이기지 못한다.
- V15의 차별점이 conditional Monte Carlo + 기존 LDP/CEM의 단순 결합으로
  판명된다.
- rBergomi가 단지 예제이고 금융적으로 새로운 결론이 없다.

이 경우에도 정직한 finite-grid numerical paper는 SIFIN, SISC 또는 JUQ 경로로
발전시킬 수 있다.

## 2. 논문이 풀 문제

### 2.1 연속시간 rBergomi 조건부 표현

독립 Brownian motion `W`와 `B`에 대해 대표적인 rBergomi law를

```text
V_t = xi_0(t) exp(eta W_t^H - 0.5 eta^2 t^(2H)),
dS_t/S_t = sqrt(V_t) [rho dW_t + sqrt(1-rho^2) dB_t]
```

로 둔다. `W`가 생성하는 sigma-field에 조건부로 terminal log-price는 Gaussian이다.

```text
m_T(W) = log S_0 + rho int_0^T sqrt(V_t) dW_t
         - 0.5 int_0^T V_t dt,
v_T(W) = (1-rho^2) int_0^T V_t dt,
g_K(W) = Phi((log K - m_T(W)) / sqrt(v_T(W))).
```

따라서 terminal left-tail probability는 `P(S_T <= K) = E[g_K(W)]`이다.
이 항등식은 continuous model 수준의 조건부 항등식이지만, 실제 알고리즘은
`m_T`, `v_T`, Volterra convolution을 이산화하므로 continuous-time exact
simulation이라고 부르면 안 된다.

### 2.2 조건부 zero-variance target

Gaussian reference measure를 `P`라 하고 `P_K = E_P[g_K]`라 하면 조건부
integrand에 대한 zero-variance density는

```text
dPi*_K/dP = g_K / P_K
```

이다. 임의의 `Q`에서 ordinary IS를 사용하면

```text
relative variance = chi-square(Pi*_K || Q).
```

이는 proposal 학습 목표를 모호한 elite 평균이 아니라 `Pi*_K` 근사오차로
정의하게 해 준다. V15는 이 항등식과 computable upper/lower diagnostic을
proposal certification의 출발점으로 사용한다.

### 2.3 Cameron--Martin action과 지배경로

Volterra driver의 Cameron--Martin space를 `H_K`라고 하자. rare-event 또는
small-noise family의 action 후보는

```text
A_theta,epsilon(h)
  = 0.5 ||h||^2_HK - log g_theta,epsilon(h)
```

또는 해당 large-deviation scaling으로 정규화한 극한 action이다. V15는
`A`의 모든 relevant local/global minimizer를 찾고, 각 minimizer 주변의
유한랭크 곡률을 proposal에 사용한다.

고정 만기 `K -> 0`와 small-time scaling은 서로 다른 문제다. 기존 rBergomi
pathwise LDP가 직접 적용되는 scaling을 P1에서 정확히 고정하기 전에는 어느
쪽의 asymptotic efficiency도 주장하지 않는다.

## 3. V15 모델 구조

### 3.1 층 1: exact conditionalization

- volatility-correlated driver를 남기고 독립 price driver 전체를 적분한다.
- terminal digital은 Gaussian CDF, European put/call은 조건부 Black--Scholes
  적분으로 처리한다.
- barrier/occupation은 동일 공식이 아니므로 headline에서 제외한다. 별도의
  boundary-crossing 적분과 오차 정리가 생기기 전에는 terminal 공식을 재사용하지
  않는다.

### 3.2 층 2: multimode Cameron--Martin transport

각 task parameter `theta=(H, eta, rho, xi_0, T, K, payoff)`에 대해

- homotopy와 multistart로 지배경로 `h_1,...,h_J`를 찾는다.
- 서로 같은 mode는 Cameron--Martin norm과 path distance로 병합한다.
- symmetry breaking 또는 mode switching이 있으면 모든 지배 mode를 보존한다.

단일 평균만 쓰는 V14는 여러 rare-event mechanism이 있을 때 실패할 수 있다.
V15의 mode coverage audit는 이 오류를 직접 검출해야 한다.

### 3.3 층 3: finite-rank Laplace transport

최종 proposal은

```text
Q = delta P + (1-delta) sum_j alpha_j Q_j,
```

로 둔다. `Q_j`는 Cameron--Martin mean shift와 유한랭크 covariance update만
가지는 Gaussian measure다.

- `delta`는 사전 고정하고 `0 < delta < 1`을 강제한다.
- covariance update rank `r`은 mesh에 무관한 상한을 둔다.
- 모든 covariance eigenvalue를 `[lambda_min, lambda_max]`로 clipping한다.
- density는 log-sum-exp, Woodbury identity, determinant lemma로 정확히 계산한다.
- 자연성분 때문에 `dP/dQ <= 1/delta`다.

무한차원 Gaussian measure에서 임의의 full-covariance 변경은 원 measure와
서로 singular해질 수 있다. 따라서 full covariance neural output은 금지한다.
Cameron--Martin mean shift와 finite-rank, equivalence-preserving perturbation만
허용한다.

### 3.4 층 4: structure-preserving neural operator

operator는 `theta`, Volterra-kernel features, maturity grid를 입력받아 다음을
예측한다.

- mode별 Cameron--Martin basis coefficients;
- mode weights;
- 유한랭크 curvature directions와 bounded eigenvalues;
- 신뢰도 및 예상 KKT residual.

권장 구조는 mesh-node set encoder + kernel/basis encoder + fixed-slot mode
decoder다. 단순 MLP, DeepONet형 decoder, attention/set encoder를 동일 parameter
budget으로 비교하고 가장 작은 충분 모델을 채택한다. 이름 때문에 복잡한
architecture를 선택하지 않는다.

network 출력은 다음 deterministic corrector를 반드시 거친다.

1. Cameron--Martin basis projection;
2. covariance spectrum clipping;
3. 짧은 trust-region/Newton action correction;
4. KKT residual과 action-gap certification;
5. 실패 시 V14 또는 natural conditional fallback.

따라서 operator가 OOD에서 틀려도 estimator는 biased해지지 않는다. 오직
variance와 cost만 악화될 수 있다.

## 4. 정리·증명 ledger

각 정리는 `proved`, `conditional`, `falsified`, `open` 중 하나로만 관리한다.

### T15-1. Conditional representation

적절한 integrability와 `|rho|<1` 하에서 terminal digital/European payoff의
조건부 적분 공식을 증명한다. finite-grid 구현과 continuous expression을 분리한다.

통과 조건:

- measurability와 stochastic-integral convention을 명시한다.
- `rho -> +/-1` degeneracy를 가정 밖 또는 별도 limit로 처리한다.
- `v_T>0`를 보장하는 가정을 기록한다.

### T15-2. Exact likelihood and Rao--Blackwell ordering

frozen defensive mixture에서 ordinary IS unbiasedness, likelihood bound,
conditional-vs-hard variance identity를 증명한다. adaptive training randomness에는
조건부로 증명하고, training/evaluation 독립성을 명시한다.

### T15-3. Zero-variance target identity

`relative variance = chi-square(Pi* || Q)`를 증명하고, defensive mixture 및
Gaussian parameter 오차로부터 계산 가능한 nonasymptotic 상계를 유도한다.

중요 오류 방지:

- `chi-square(Q || Pi*)`와 방향을 바꾸지 않는다.
- unknown normalizer `P_K`가 필요한 양은 이론 diagnostic과 구현 가능 diagnostic을
  구분한다.
- self-normalized IS 공식을 ordinary IS 분산식에 섞지 않는다.

### T15-4. Proposal perturbation stability

bounded Cameron--Martin norm, bounded finite-rank spectrum, defensive mass 하에서
mode 위치·곡률·weight 오차가 second moment에 미치는 영향을 정량화한다.
이 정리가 operator correction tolerance를 결정한다.

### T15-5. Rare-event efficiency

P1에서 고정한 scaling에 대해 다음 중 적어도 하나를 완성해야 한다.

- logarithmic efficiency;
- bounded relative error;
- asymptotically optimal second-moment exponent.

모든 dominating minimizer를 proposal이 포함한다는 조건과 minimizer 누락 시
실패하는 반례를 함께 제시한다. 기존 rBergomi pathwise LDP를 사용할 경우 그
topology, scaling, contraction map의 연속성 가정을 그대로 검증한다.

### T15-6. Finite-grid convergence

`g_{K,N} -> g_K`의 `L1` 또는 `L2` 수렴과 probability bias bound를 증명한다.
가능하면

```text
|E[g_{K,N}] - E[g_K]| <= C N^(-alpha)
```

를 보이되, `alpha`는 증명 전에 고정하지 않는다. H가 0에 가까워질 때 상수가
폭발할 수 있으므로 `H in [H_min,H_max]`, `H_min>0` 같은 uniformity 범위를
명시한다.

### T15-7. Mesh consistency of the transport

Galerkin/basis 근사 `h_{j,N}`이 continuum minimizer set으로 수렴하고,
finite-rank directions가 refinement에서 일관됨을 보인다. minimizer가 유일하지
않으면 point convergence 대신 set/energy convergence를 사용한다.

### T15-8. End-to-end complexity

discretization bias, final sampling variance, mode solve/operator inference,
likelihood cost를 모두 포함하여 target RMSE `epsilon`의 total-work bound를
도출한다. 100-query 숫자 하나가 아니라 query count `M`에 따른 break-even
function을 제시한다.

### T15-9. Operator-assisted certification

network prediction 후 corrector가 정해진 KKT/action tolerance를 만족하면
T15-4의 variance bound가 적용됨을 증명한다. network의 universal approximation
정리는 필수가 아니다. 실제로 필요한 것은 certified proposal quality와
amortized cost다.

## 5. 구현 단계

## P0. Claim·문헌·재현성 동결

산출물:

- `docs/literature/G11_V15_PRIMARY_SOURCE_LEDGER.md`
- 검색 DB, query, 날짜, 포함/제외 이유를 담은 machine-readable ledger
- closest-work derivation 비교표
- V15 claim contract YAML

필수 대조군 문헌:

- rBergomi 원 논문과 pathwise LDP;
- conditional Monte Carlo/turbocharging rBergomi;
- numerical preintegration + QMC/MLMC;
- LDP + adaptive IS + likelihood-informed subspace;
- failure-informed dimension reduction CEM;
- nonasymptotic suboptimal IS bounds;
- RQMC + Gaussian IS;
- 최신 non-Markovian stochastic-control/neural rare-event work.

Gate G0:

- 두 명의 독립 검토자가 “conditionalization”, “LDP mode”, “finite-rank Gaussian
  mixture”, “neural amortization” 각각의 prior-art 경계를 승인한다.
- 조합만 새롭고 새 정리·새 금융 insight가 없으면 top-journal 경로를 중단한다.

## P1. Probability-space 및 asymptotic regime 고정

구현 전에 다음을 하나의 notation 문서에 고정한다.

- `W`, `B`, Volterra kernel, filtration, Itô convention;
- continuous law와 BLP/hybrid/FFT finite-grid law의 관계;
- terminal event/payoff class;
- rare-event parameter와 scaling;
- Cameron--Martin space와 basis;
- 허용 parameter domain과 제외 singular cases.

Gate G1:

- 수식에서 conditional mean/variance를 독립 구현 두 개와 symbolic derivation으로
  대조한다.
- `rho=0`, `eta=0`, Black--Scholes limit에서 known answer를 회복한다.
- 적용 가능한 rBergomi LDP scaling이 명확하지 않으면 T15-5 주장을 잠근다.

## P2. Continuum/discrete oracle 계층

구현 항목:

- terminal digital, put, call의 안정적인 log-domain conditional integrals;
- double precision과 high-precision oracle;
- shared Gaussian path로 `N,2N,4N` coupled evaluation;
- analytic Black--Scholes/constant-volatility oracle;
- gradient와 Hessian-vector product finite-difference oracle.

테스트:

- pathwise conditional formula 대 raw price-driver integration;
- extreme normal CDF/SF에서 underflow 없는지;
- left/right tail 부호와 strike convention;
- CPU/GPU, float64/high precision 교차검증;
- mesh coupling reconstruction.

Gate G2:

- pathwise identity tolerance `<=1e-10`(float64 feasible 영역).
- extreme-tail log probability는 high-precision oracle와 사전 고정 상대/절대 오차
  이내.
- 하나라도 실패하면 optimizer나 학습 단계로 진행하지 않는다.

## P3. Cameron--Martin action solver

구현 항목:

- basis-independent action interface;
- autodiff gradient, Hessian-vector product;
- trust-region Newton-CG와 L-BFGS 두 solver;
- rarity homotopy와 mesh continuation;
- deterministic multistart, mode deflation/merging;
- KKT residual, negative-curvature, action-gap report;
- 모든 seed와 solver trace 저장.

오류 방지:

- SMC sample mean을 variational minimizer로 부르지 않는다.
- local minimum 하나를 global/dominating mode라고 단정하지 않는다.
- optimization objective에 unknown probability normalizer를 넣지 않는다.
- discretized Euclidean norm과 Cameron--Martin norm의 grid scaling을 혼동하지
  않는다.

Gate G3:

- low-dimensional analytic examples에서 known minimizer 회복.
- 두 optimizer와 multistart가 같은 최소 action set을 재현.
- `N -> 2N`에서 action과 path가 사전 tolerance로 안정화.
- unresolved lower-action mode가 있으면 qualification 금지.

## P4. Exact multimode finite-rank proposal

구현 항목:

- natural + multimode Gaussian balance sampler;
- Cameron--Martin mean likelihood;
- finite-rank covariance sampling/log-density;
- stable log-sum-exp mixture;
- exact cost ledger;
- proposal serialization hash;
- conditional terminal evaluator와 ordinary IS statistics.

테스트:

- one-dimensional symbolic density ratio;
- zero shift/rank-zero가 natural law 회복;
- component label을 주변화한 balance denominator;
- `E_Q[dP/dQ]=1` normalization z-test;
- `dP/dQ <= 1/delta` pathwise test;
- covariance SPD, determinant, inverse를 dense oracle와 비교;
- sample moments와 declared Gaussian moments 대조;
- proposal freeze 이후 parameter mutation 금지.

Gate G4:

- exactness/audit suite 전부 통과.
- V14 세 frozen cell에서 정확도를 유지하면서 V14보다 나쁘지 않은 total work.
- 실패 시 covariance update를 제거하고 multimode mean-shift만 유지한다.

## P5. 이론 증명 1차 완료

T15-1부터 T15-5까지 proof draft를 코드와 동시에 완성한다.

필수 검토 방식:

- lemma마다 assumptions-used 표;
- finite-dimensional statement와 infinite-dimensional statement 분리;
- counterexample/failure case 포함;
- 외부 확률론/rough-volatility 연구자 line-by-line review;
- theorem oracle test가 가능한 식은 property-based test로 연결.

Gate G5:

- T15-1~T15-4가 완전 증명.
- T15-5가 완전 증명 또는 명시적으로 falsified.
- T15-5가 없으면 Mathematical Finance/Finance and Stochastics headline을 잠근다.

## P6. Mesh consistency와 ML/bias 제어

실험 grid:

- 최소 `N={32,64,128,256,512,1024}`;
- rough regime는 필요 시 2048 이상 reference;
- common-random-number adjacent coupling;
- H별 rate와 constant를 분리 추정.

구현 항목:

- mode continuation across meshes;
- conditional correction estimator;
- pilot-frozen MLMC allocation 또는 Richardson bias control;
- bias/variance/sampling error budget 분리;
- finest-grid reference와 continuous extrapolation uncertainty.

Gate G6:

- T15-6/T15-7 가정과 실측 rate가 모순되지 않는다.
- headline cell의 discretization bias가 전체 RMSE budget의 사전 고정 비율 이하.
- coarse/fine proposal mismatch를 무시한 공통 likelihood는 사용하지 않는다.

## P7. Structure-preserving neural operator

데이터 생성:

- parameter domain을 training/interpolation/OOD stress로 사전 분리;
- P3 certified modes만 label로 사용;
- mode permutation을 고려한 matching loss;
- action, KKT residual, path coefficient, curvature를 공동 기록.

학습 목표:

```text
L = path-set loss
  + lambda_A action-gap loss
  + lambda_KKT stationarity loss
  + lambda_curv curvature loss.
```

평가:

- inference만 사용;
- inference + fixed corrector;
- cold-start solver;
- V14 SMC;
- nearest-neighbor/warm-start;
- 작은 MLP.

Gate G7:

- corrected operator가 cold-start solver와 같은 certified action tolerance 달성.
- OOD 실패를 confidence/fallback이 검출.
- query-count break-even에서 training cost 포함 이득.
- operator가 이득이 없으면 논문 본문에서 제거하고 variational transport만 유지.

## P8. 최강 baseline matrix

모든 headline cell에 같은 estimand, mesh, RMSE, hardware accounting을 사용한다.

필수 baseline:

1. crude Monte Carlo;
2. natural exact conditional Monte Carlo;
3. conditional RQMC/preintegration;
4. published rBergomi conditional-MC/turbocharging specialization;
5. task-tuned pure/defensive CEM;
6. failure-informed dimension-reduced CEM;
7. LDP + adaptive IS / likelihood-informed subspace;
8. subset/SMC or AMS where estimator assumptions match;
9. V14 local Volterra transport;
10. V15 cold solver, V15 operator, V15 ablations.

비교 규칙:

- 저자 공개 코드가 있으면 version/commit을 고정한다.
- baseline을 독립적으로 재현한 smoke result를 먼저 통과시킨다.
- training, calibration, pilot, rejected samples, density cost를 전부 포함한다.
- inaccurate 또는 zero-hit run은 zero cost winner가 될 수 없다.
- 같은 proposal을 raw/conditional estimator에 쓸 때 paired variance identity를 보고한다.
- oracle parameter를 쓴 baseline은 별도 oracle upper-bound로 표시한다.

Gate G8:

- 각 record에 적어도 하나의 reference-accurate 강한 baseline.
- headline 결론이 약한 baseline 하나에 의존하지 않는다.

## P9. Development matrix

### Parameter coverage

모든 full factorial을 무작정 돌리지 않고, space-filling primary matrix와 별도
stress matrix를 동결한다.

- `H`: rough low/intermediate/high 영역, 최소 5개 값;
- `eta`: low/base/high;
- `rho`: strongly negative/base/weak/zero;
- maturity: short/base/long;
- rarity: 최소 `1e-3`부터 `1e-8`;
- grid: P6 mesh family;
- query count: `M={1,10,100,1000}`.

### Financial tasks

Headline:

- terminal left/right digital tail;
- deep OTM put/call price;
- terminal loss/VaR exceedance probability.

Secondary:

- mixed/multifactor lognormal Gaussian-Volterra models;
- strike surface reuse and calibration-like query batches.

Barrier와 occupation은 올바른 conditional boundary solver와 정리가 완성된 경우에만
secondary extension으로 추가한다.

### Statistics

- proposal-training cluster를 inferential unit으로 사용;
- development와 qualification seed namespace 완전 분리;
- paired log-work ratio와 cluster bootstrap 또는 명시된 t-bound;
- empirical variance, fourth moment, ESS, maximum weight, kurtosis 보고;
- bounded contribution에 대한 empirical-Bernstein/anytime-valid certificate는
  estimator claim과 별도 표시;
- 선택적 stopping을 쓸 경우 confidence sequence 또는 독립 holdout을 사용;
- threshold calibration sample은 performance sample과 분리.

Development Gate G9:

- 모든 candidate record가 reference accuracy gate 통과;
- accuracy-qualified best baseline 대비 geometric total-work ratio `>=2.0`;
- one-sided 95% lower ratio `>1.25`;
- headline cell의 최소 80%에서 V15 우위;
- 각 rarity band에서 catastrophic loss 없음;
- V14 대비 action/cost/variance ablation으로 개선 원인이 설명됨;
- 성능 결과와 무관하게 T15 theorem gate 통과.

`2x`는 저널 수준을 자동 보장하는 숫자가 아니다. 이는 과도한 복잡성에 비해
실용적 이득이 없을 때 중단하기 위한 내부 최소선이다.

## P10. Frozen qualification

development 결과를 본 뒤 architecture, gate, cell, budget을 바꾸지 않는다.

- config/result/theory hash binding;
- 새로운 seed와 threshold-calibration namespace;
- 최소 20개의 독립 training cluster 또는 power가 정한 수;
- 결과 JSON은 sufficient statistics와 fourth moment 저장;
- 독립 audit program이 summary, hashes, seeds, likelihood, decision 재계산;
- failure도 삭제하지 않고 artifact로 보존.

Qualification Gate G10:

- G9의 accuracy와 work gate를 독립 data에서 재통과;
- development/qualification effect size가 사전 tolerance 안에서 일관;
- 어떤 comparator exclusion도 결과를 본 뒤 추가되지 않음.

## P11. 외부 재현과 적대적 검토

필수:

- Linux + 다른 CPU/GPU + clean environment;
- 설치부터 표/그림 생성까지 한 command 또는 workflow;
- 외부 연구자가 선택한 hidden cells;
- 수학 reviewer 1명, numerical reviewer 1명, finance reviewer 1명;
- novelty red-team: 가장 가까운 세 논문과 식 단위 비교;
- reproducibility package에 environment lock, seeds, raw sufficient statistics,
  audit receipt 포함.

Gate G11:

- 외부 환경에서 정확도와 우위 방향 재현;
- proof blocker 0개;
- high-severity code/audit blocker 0개;
- novelty objection이 해결되거나 claim을 축소.

## P12. 논문 완성

권장 본문 구조:

1. 금융 문제와 기존 방법의 구조적 한계;
2. Gaussian Volterra conditional representation;
3. Cameron--Martin multimode transport;
4. exact defensive likelihood와 nonasymptotic bound;
5. rare-event 및 mesh-consistency theorem;
6. operator-assisted amortization;
7. frozen 실험과 금융적 해석;
8. 실패 영역과 claim boundary.

본문에서 반드시 분리할 표현:

- exact conditional law vs discretization-exact algorithm;
- unbiased estimator vs trained proposal consistency;
- finite-grid empirical gain vs continuous-time theorem;
- asymptotic efficiency vs finite-rarity performance;
- operator speedup vs mathematical exactness.

## 6. 구현 우선순위

### Priority 0: 문헌·정리 가능성

P0와 P1을 먼저 한다. 이 단계에서 차별성이 사라지거나 LDP scaling이 맞지 않으면
대규모 구현을 하지 않는다.

### Priority 1: 비신경 V15 core

P2~P5를 진행한다. 가장 먼저 구현할 실제 모델은 multimode Cameron--Martin
mean-shift이며, finite-rank covariance는 mean-only ablation이 안정화된 뒤 추가한다.

### Priority 2: mesh와 강한 baseline

P6와 P8을 수행한다. top-journal 수준을 가르는 것은 작은 network 성능보다
continuous target과 strongest comparator에 대한 방어력이다.

### Priority 3: neural operator

P7은 certified solver data가 쌓인 뒤 시작한다. operator가 없더라도 논문 core가
성립해야 한다.

### Priority 4: 대규모 development/qualification

P9~P11은 local smoke가 아니라 외부 compute를 사용한다. 노트북에서는 oracle,
solver, unit/property tests, 작은 matrix까지만 수행한다.

## 7. 실패 원인별 사전 대응

| 실패 | 진단 | 조치 | 허용 주장 |
|---|---|---|---|
| mode 하나가 rare shell을 놓침 | multistart action/action-gap | multimode mixture와 deflation | 단일-mode 주장 금지 |
| covariance가 불안정 | eigenvalue/determinant audit | mean-only 또는 rank 축소 | curvature 이득 주장 금지 |
| mesh마다 shift가 변함 | CM norm과 energy convergence | basis/continuation 교정 | continuous claim 잠금 |
| operator OOD 실패 | KKT/action certificate | corrector 또는 V14 fallback | unbiasedness 유지, speed claim 축소 |
| rarest band weight 폭발 | max weight/fourth moment | defensive mass 증가, mode 추가 | tail guarantee 잠금 |
| training cost가 큼 | query-count break-even | operator/continuation 또는 scope 축소 | M별 결과만 주장 |
| baseline이 더 강함 | matched total-work rerun | 원인 분석 후 falsification 기록 | superiority 주장 금지 |
| LDP theorem 불성립 | assumption/counterexample audit | nonasymptotic theorem으로 전환 | MF/F&S headline 잠금 |
| barrier 공식 오용 | pathwise event mismatch | terminal-only로 회귀 | barrier claim 금지 |

## 8. 비협상 정확성 규칙

1. final estimator는 ordinary IS이며 self-normalization을 사용하지 않는다.
2. adaptive proposal과 final samples는 독립이다.
3. mixture component label을 denominator에서 조건화하지 않는다.
4. 자연성분과 exact balance denominator를 유지한다.
5. operator 출력은 frozen 후 final sampling에서 변경하지 않는다.
6. 모든 Gaussian measure change는 Cameron--Martin/equivalence 조건을 만족한다.
7. continuous-time와 finite-grid notation, result, table을 분리한다.
8. threshold/reference calibration을 test data와 분리한다.
9. inaccurate baseline은 승자도 패자도 아닌 invalid comparison으로 표시한다.
10. 실패한 hypothesis와 censored run을 삭제하지 않는다.
11. 실제 wall time과 architecture-neutral work를 함께 보고한다.
12. theorem status가 `open`이면 abstract/conclusion에서 결과처럼 쓰지 않는다.

## 9. 완료 정의

최상위 저널 제출 준비 완료는 코드가 돌아간다는 뜻이 아니다. 다음이 모두
충족되어야 한다.

- T15-1~T15-8 중 투고 claim에 필요한 정리가 완전 증명됨;
- rBergomi corollary의 모든 가정 검증;
- novelty red-team과 최신 primary-source audit 통과;
- frozen development와 independent qualification 통과;
- strongest matched baselines와 비교 완료;
- mesh/parameter/rarity/query-count 일반화 완료;
- 외부 hardware와 hidden-cell 재현 통과;
- 공개 가능한 code/data/audit package 완성;
- 금융적 활용과 실패 영역이 숫자로 제시됨;
- 외부 전문가 proof review의 blocker가 0개.

그 전의 정확한 표현은 다음이다.

> V14는 독립 qualification을 통과한 박사급 finite-grid working-paper core이고,
> V15는 이를 top-journal 후보로 바꾸기 위한 이론·알고리즘 연구 프로그램이다.

## 10. 1차 문헌 출발점

1. Bayer, Friz, Gatheral, “Pricing under rough volatility,” Quantitative Finance,
   DOI: <https://doi.org/10.1080/14697688.2015.1099717>.
2. Jacquier, Pakkanen, Stone, “Pathwise large deviations for the Rough Bergomi
   model,” <https://arxiv.org/abs/1706.05291>.
3. McCrickerd, Pakkanen, “Turbocharging Monte Carlo pricing for the rough Bergomi
   model,” <https://arxiv.org/abs/1708.02563>.
4. Bennedsen, Lunde, Pakkanen, “Hybrid scheme for Brownian semistationary
   processes,” <https://doi.org/10.1007/s00780-017-0335-5>.
5. Bayer, Ben Hammouda, Tempone, “Multilevel Monte Carlo with Numerical Smoothing
   for Robust and Efficient Computation of Probabilities and Densities,”
   <https://doi.org/10.1137/22M1495718>.
6. Tong, Stadler, “Large Deviation Theory-based Adaptive Importance Sampling for
   Rare Events in High Dimensions,” <https://doi.org/10.1137/22M1524758>.
7. Uribe, Papaioannou, Marzouk, Straub, “Cross-Entropy-Based Importance Sampling
   with Failure-Informed Dimension Reduction,”
   <https://doi.org/10.1137/20M1344585>.
8. “Nonasymptotic Bounds for Suboptimal Importance Sampling,”
   <https://doi.org/10.1137/21M1427760>.
9. He, Zheng, Wang, “On the Error Rate of Importance Sampling with Randomized
   Quasi-Monte Carlo,” <https://doi.org/10.1137/22M1510121>.
10. Bourgey, De Marco, “Multilevel Monte Carlo simulation for VIX options in the
    rough Bergomi model,” <https://arxiv.org/abs/2105.05356>.

이 목록은 시작점이며 submission 시점의 완전한 literature review가 아니다.

## 11. 저장소 구현 지도와 첫 실행 순서

다음 파일명은 구현 중 충돌이 없으면 그대로 사용한다.

### P0--P1

- `docs/literature/G11_V15_PRIMARY_SOURCE_LEDGER.md`
- `docs/theory/G11_V15_PROBABILITY_AND_SCALING_CONTRACT.md`
- `configs/g11_v15/claim_contract_v1.yaml`
- `tests/test_v15_continuous_contract.py`

첫 마일스톤은 문헌 ledger와 probability-space contract가 동시에 통과한 뒤에만
커밋한다. 이 단계에서는 실험 성능 코드를 만들지 않는다.

### P2--P4

- `src/path_integral/volterra_conditional_payoffs.py`
- `src/path_integral/cameron_martin_basis.py`
- `src/path_integral/volterra_action.py`
- `src/path_integral/cameron_martin_modes.py`
- `src/path_integral/finite_rank_gaussian_transport.py`
- `src/path_integral/rbergomi_cm_transport.py`
- 대응 unit/property/oracle tests

구현 순서는 conditional oracle, action derivative, mean-only multimode proposal,
finite-rank covariance 순이다. covariance 구현이 mean-only 결과보다 먼저
성능 실험에 들어가면 안 된다.

### P5--P6

- `docs/theory/G11_V15_THEOREMS.md`
- `src/path_integral/rbergomi_cm_mesh.py`
- `src/path_integral/rbergomi_cm_mlmc.py`
- `experiments/g11_v15_mesh_development.py`
- `configs/g11_v15/mesh_development_v1.yaml`
- theorem assumption auditor와 result auditor

### P7

- `src/models/volterra_transport_operator.py`
- `src/training/volterra_transport_operator.py`
- `experiments/g11_v15_operator_development.py`
- `configs/g11_v15/operator_development_v1.yaml`
- OOD/fallback/certification tests

### P8--P10

- `src/path_integral/v15_baseline_protocol.py`
- `src/path_integral/v15_result_audit.py`
- `experiments/g11_v15_development.py`
- `experiments/g11_v15_qualification.py`
- `experiments/g11_v15_audit.py`
- frozen development/qualification YAML 및 hash-bound result artifacts

### 실행 시작 시 첫 12개 작업

1. 최신 primary-source 검색 ledger를 만들고 closest equations를 비교한다.
2. small-time과 small-noise 중 rBergomi corollary가 가능한 regime을 결정한다.
3. continuous conditional formula와 finite-grid convention을 고정한다.
4. Black--Scholes, `rho=0`, `eta=0` oracle tests를 먼저 작성한다.
5. Cameron--Martin basis와 grid-normalized norm을 구현한다.
6. conditional action, gradient, Hessian-vector product를 구현한다.
7. analytic toy action에서 solver를 검증한다.
8. rBergomi rarity homotopy와 multistart mode search를 구현한다.
9. V14 cell에서 mean-only multimode proposal을 V14와 paired 비교한다.
10. 위 단계가 통과한 뒤 finite-rank curvature를 추가한다.
11. T15-1~T15-5 proof draft와 counterexample audit를 끝낸다.
12. theorem gate가 통과한 뒤 mesh study와 대규모 baseline matrix를 연다.

각 마일스톤은 `theory -> oracle tests -> implementation -> independent audit ->
frozen experiment` 순서를 지킨다. 성능이 먼저 나온 뒤 그 결과에 맞춰 정리의
가정이나 gate를 바꾸는 것을 금지한다.
