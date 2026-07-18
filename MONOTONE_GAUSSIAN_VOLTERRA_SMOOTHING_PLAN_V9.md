# Monotone Gaussian Volterra Smoothing: 탑저널 목표 실행계획 v9

Date: 2026-07-17

Status: pre-implementation theory and falsification contract

Predecessors:

- `CONTROLLED_MULTILEVEL_VOLTERRA_PLAN_V7.md`
- `BOUNDARY_AWARE_VOLTERRA_BRANCHING_PLAN_V8.md`
- `docs/phase_reviews/G8_VOLTERRA_BRIDGE_BRANCHING_FALSIFICATION_2026-07-17.md`

## 0. 문서의 목적과 현실적 약속

이 문서는 현재 저장소를 유명 저널에 제출 가능한 연구로 발전시키기 위한 주 연구
가설, 수학적 계약, 구현 순서, 통계 protocol, 중단 기준을 고정한다. 어떤 계획도 게재를
보장할 수는 없다. 대신 다음을 보장하도록 설계한다.

1. 이미 실패한 G7/G8 구조를 더 복잡하게 만들어 결과를 포장하지 않는다.
2. finite-grid exactness, continuous-time statement, empirical rate를 구분한다.
3. 학습, tuning, 실패한 실험까지 포함한 총 계산량으로 비교한다.
4. validation 결과를 본 뒤 gate나 지표를 바꾸지 않는다.
5. 이론 명제가 증명되지 않으면 그 명제를 삭제하거나 논문의 범위를 낮춘다.

## 1. 현재 상태에서 반드시 해결해야 할 병목

G8의 conditional branching estimator는

\[
\operatorname{Var}(\bar Y_M)
=\operatorname{Var}(E[Y\mid C])
+\frac1M E[\operatorname{Var}(Y\mid C)]
\]

에서 두 번째 항만 줄였다. 실제 audit에서는 correction-level work가 약 `1.94x`
개선됐지만 end-to-end work ratio는 `0.299x`, correction-variance log-slope는
`-0.372`였다. 따라서 다음 단계는 branch 수를 늘리는 것이 아니라 hard indicator의
불연속성을 bias 없이 먼저 적분해야 한다.

V9의 주 연구 질문은 다음과 같다.

> rBergomi의 volatility-independent price Brownian direction 하나를 해석적으로
> 적분하여 hard hit-plus-occupation correction을 smooth하게 만들면, exact
> finite-grid telescoping을 유지하면서 positive multilevel variance decay와
> end-to-end 계산 우위를 회복할 수 있는가?

## 2. 제안 논문의 중심 주장

가제는 다음과 같다.

> **Bias-Free Gaussian Smoothing and Exact Multilevel Importance Sampling for
> Discontinuous Path Events under Rough Volatility**

주모델의 작업명은 `MGVS`로 고정한다.

> **MGVS: Monotone Gaussian Volterra Smoothing**

논문의 중심 기여 후보는 세 가지다.

1. rBergomi의 independent price driver에서 downside hit-plus-occupation event를
   스칼라 Gaussian threshold로 정확히 변환한다.
2. deterministic Girsanov control의 likelihood까지 함께 적분하여 correction을
   Gaussian CDF 차이로 계산한다.
3. exact adjacent BLP coupling 및 fast Volterra convolution과 결합하고,
   variance-rate/complexity/total-work를 이론과 실험으로 검증한다.

학습 conditional-mean control variate는 주기여가 아니다. MGVS가 독립적으로 gate를
통과한 뒤에만 secondary extension으로 고려한다.

## 3. 선행연구와 novelty 경계

| 알려진 결과 | V9에서의 처리 |
|---|---|
| BLP hybrid rough-Volterra simulation | 그대로 사용하며 신규성으로 주장하지 않음 |
| discontinuous payoff의 one-dimensional numerical smoothing | 일반 아이디어의 최초성을 주장하지 않음 |
| digital option path branching | G8 비교군이며 V9 신규성으로 주장하지 않음 |
| rough Bergomi VIX MLMC | rough-volatility MLMC 자체의 최초성을 주장하지 않음 |
| neural/learned control variates | optional extension이며 단독 신규성으로 주장하지 않음 |
| approximate/multifidelity control variates | unknown surrogate mean 처리에 대한 기준 이론으로 사용 |

필수 1차 문헌:

- BLP hybrid scheme: <https://arxiv.org/abs/1507.03004>
- numerical smoothing MLMC: <https://arxiv.org/abs/2003.05708>
- numerical smoothing/QMC: <https://arxiv.org/abs/2111.01874>
- multilevel path branching: <https://arxiv.org/abs/2209.03017>
- rough Bergomi VIX MLMC: <https://arxiv.org/abs/2105.05356>
- rough PPDE/weak rates, v4: <https://arxiv.org/abs/2304.03042>
- learned control variates: <https://arxiv.org/abs/1606.02261>
- neural control variates: <https://arxiv.org/abs/2006.01524>
- approximate control variates: <https://arxiv.org/abs/1811.04988>

논문에서 `first`, `novel`, `unique`를 사용하려면 arXiv 검색만으로는 부족하다.
MathSciNet, zbMATH, Web of Science/Scopus에서 다음 교집합을 별도로 systematic review한다.

1. rough Bergomi 또는 stochastic Volterra;
2. barrier/occupation/drawdown probability;
3. conditional Monte Carlo 또는 numerical smoothing;
4. importance sampling/Girsanov;
5. multilevel coupling.

현재 허용되는 novelty 표현은 다음뿐이다.

> an exact finite-grid specialization of monotone Gaussian smoothing to
> controlled rough-Volterra hit-and-occupation corrections, combined with an
> exact adjacent BLP coupling.

## 4. 확률모형과 허용 범위

Primary model은 다음 rBergomi discretization이다.

\[
X_{i+1}=X_i+(\mu-\tfrac12V_i)h
+\sqrt{V_i}\left(\rho\,\Delta W_i^{1,P}
+\rho_\perp\,\Delta W_i^{2,P}\right),
\qquad \rho_\perp=\sqrt{1-\rho^2}.
\]

`W1`은 Volterra variance를 구동하고 `W2`는 variance를 직접 구동하지 않는다.
Primary theorem과 실험은 다음 조건을 요구한다.

- `H in (0, 1/2)`;
- `|rho| < 1`;
- implemented variance is strictly positive, 현재 clamp 포함;
- control은 validation 전에 고정된 deterministic time-only two-driver schedule;
- smoothing direction은 deterministic, nonnegative, unit norm;
- payoff는 각 monitoring spot이 낮아질수록 event가 유지되는 downside-monotone event.

첫 논문에서 금지하는 확장은 다음과 같다.

- feedback/path-dependent control;
- evaluation path를 본 뒤 선택하는 smoothing direction;
- `rho=+-1`;
- 비단조 drawdown 또는 lookback-ratio payoff에 공식을 그대로 적용;
- continuous monitoring 결과를 discrete-grid 결과로 대체;
- self-normalized likelihood.

## 5. Monotone Gaussian coordinate

Level `ell`에 `N=N_ell`, `h=T/N`을 사용한다. Proposal second-driver increment를

\[
\Delta W_i^{2,Q}=\sqrt h\,\xi_i,
\qquad \xi\sim N(0,I_N)
\]

로 쓴다. Validation 전에 고정한

\[
q\in\mathbb R^N,
\qquad \lVert q\rVert_2=1,
\qquad q_i>0
\]

에 대해

\[
Z=q^\top\xi,
\qquad R=(I-qq^\top)\xi
\]

로 분해한다. 그러면 `Z ~ N(0,1)`, `Z`와 `R`은 독립이고 `q^T R=0`이다.
Smoothed sampler는 `R`만 샘플하고 `Z`는 적분한다.

Target increment는

\[
\Delta W_i^{2,P}=\sqrt h(R_i+q_iZ)+u_{i,2}h
\]

이다. `V_i`는 `Z`와 독립이므로 fine log spot은

\[
X_k^f(Z)=A_k^f+B_k^fZ,
\]

\[
B_{k+1}^f=B_k^f
+\rho_\perp\sqrt{V_k^f}\sqrt h\,q_k>0
\]

형태다. Coarse path도 동일 fine coordinates에서 구성하므로

\[
X_j^c(Z)=A_j^c+B_j^cZ,
\]

이며 `B_j^c>0`이다. Fine과 coarse에 서로 다른 `Z`를 사용하면 exact correction이
아니므로 금지한다.

## 6. Hard event의 정확한 threshold

Spot path가 `X_k(Z)=A_k+B_kZ`, `B_k>0`라 하자.

### 6.1 Hit threshold

초기 spot이 hit barrier보다 높으면

\[
a_{hit}=\max_{1\le k\le N}
\frac{\log B_{hit}-A_k}{B_k}.
\]

초기 spot이 이미 barrier 이하이면 `a_hit=+infinity`다.

### 6.2 Occupation threshold

현재 코드의 event는 right-endpoint occupation을 사용한다. 필요한 observation 수는

\[
r=\left\lceil\frac{\tau_{occ}-10^{-15}}{h}\right\rceil
\]

로 코드와 정확히 일치시킨다. 각 monitoring point에 대해

\[
t_k=\frac{\log B_{stress}-A_k}{B_k}
\]

를 계산한다. `1 <= r <= N`이면 `a_occ`는 `{t_k}`의 `r`번째 큰 값이다.
`r>N`이면 event는 불가능하므로 `a_occ=-infinity`다.

### 6.3 Combined event

\[
a=\min(a_{hit},a_{occ})
\]

이면 현재 hard event와 pathwise하게

\[
H(Z)=1_{\{Z\le a\}}
\]

가 성립한다. 이 등가는 random test point 비교뿐 아니라 adversarial threshold/tie
test로 검증한다.

## 7. Girsanov likelihood의 정확한 적분

Second-driver control의 standard-normal coefficient를

\[
w_i=\sqrt h\,u_{i,2},
\qquad b=q^\top w,
\qquad w_\perp=w-bq
\]

로 정의한다. First-driver likelihood와 orthogonal residual likelihood를 합쳐

\[
L_\perp=
\exp\left(
-\sum_i u_{i,1}\Delta W_i^{1,Q}
-w_\perp^\top R
-\frac12h\sum_i u_{i,1}^2
-\frac12\lVert w_\perp\rVert^2
\right)
\]

라 하면 전체 likelihood는

\[
L(Z)=L_\perp\exp(-bZ-\tfrac12b^2)
\]

이다. 따라서 부호를 포함한 핵심 identity는

\[
E_Q[1_{\{Z\le a\}}L(Z)\mid W^1,R]
=L_\perp\Phi(a+b)
\]

이다. `Phi(a-b)`가 아니라 `Phi(a+b)`다. 이 부호는 symbolic derivation과 numerical
quadrature unit test를 모두 통과해야 한다.

## 8. Smoothed multilevel estimator

Base level은

\[
\widetilde Y_0=L_{\perp,0}\Phi(a_0+b_0)
\]

로 정의한다. Adjacent correction은 하나의 fine innovation space에서

\[
\widetilde Y_\ell
=L_{\perp,\ell}
\left[\Phi(a_\ell^f+b_\ell)-\Phi(a_{\ell-1}^c+b_\ell)\right]
\]

로 정의한다.

### Proposition V9-1: finite-grid unbiasedness

각 level control과 `q`가 deterministic/frozen이면

\[
E_Q[\widetilde Y_0]=E_P[H_0],
\]

\[
E_Q[\widetilde Y_\ell]
=E_P[H_\ell-H_{\ell-1}],
\]

따라서

\[
\sum_{\ell=0}^{L}E_Q[\widetilde Y_\ell]=E_P[H_L].
\]

### Proposition V9-2: Rao--Blackwell dominance

Raw controlled correction을 `Y_ell`이라 하면

\[
\widetilde Y_\ell=E_Q[Y_\ell\mid\mathcal G_\ell],
\qquad
\mathcal G_\ell=\sigma(W^1,R),
\]

이므로

\[
\operatorname{Var}(\widetilde Y_\ell)
\le\operatorname{Var}(Y_\ell).
\]

이 명제는 동일 proposal/control/coupling에 대한 variance 비교다. 서로 다른 control의
표본분산만 비교해서 Rao--Blackwell 우위라고 주장하면 안 된다.

### Proposition V9-3: threshold-stability rate

다음은 사전 가정이 아니라 증명 목표다. 어떤 `p>2`, `alpha>0`에 대해

\[
E[|a_\ell^f-a_{\ell-1}^c|^p]=O(h_\ell^{p\alpha})
\]

이고 `L_perp`의 충분한 uniform moment가 존재하면 `Phi`의 Lipschitz 성질로

\[
\operatorname{Var}(\widetilde Y_\ell)
=O(h_\ell^{2\alpha})
\]

형태의 bound를 목표로 한다. Occupation order statistic 때문에 threshold-stability는
별도 boundary-density/nondegeneracy 가정이 필요할 수 있다. 증명하지 못하면 rate
theorem은 barrier-only case로 제한하고 occupation 결과는 empirical로 명시한다.

## 9. 수치적으로 반드시 해결할 오류원

### 9.1 Gaussian CDF difference cancellation

Deep level에서는 `a_f`와 `a_c`가 매우 가까워

\[
\Phi(x)-\Phi(y)
\]

직접 감산이 0으로 소실될 수 있다. 다음 stable primitive를 별도로 구현한다.

- 음의 tail: `log_ndtr`와 `expm1`;
- 양의 tail: survival-function representation;
- 중앙 구간: direct double-precision difference;
- signed order를 보존;
- `+-infinity`, equal arguments, subnormal 값을 명시적으로 처리.

Reference는 SciPy와 high-precision `mpmath`를 사용하고 `[-40,40]`, gap
`10^{-14}`부터 `10`까지 검증한다. Absolute error와 relative error를 tail magnitude에
맞게 별도 gate로 둔다.

### 9.2 Likelihood overflow

- log-likelihood는 float64로 누적;
- `L_perp * CDF difference`는 signed-log representation을 지원;
- sample mean/second moment는 pairwise 또는 compensated accumulation 사용;
- raw `E[L]=1` studentized gate를 단독 사용하지 않고 deterministic-control
  `log L` Gaussian moment를 함께 검증;
- defensive mixture는 theorem/stability ablation이며 자동 기본값이 아님.

### 9.3 Event semantics

- initial point의 hit 포함 여부;
- right-endpoint occupation;
- `1e-15` tolerance;
- `ceil` 경계;
- `r>N` 불가능 event;
- threshold equality의 `<=` 방향

을 기존 `DownsideExcursionTask.hard_event`와 bitwise 비교한다.

## 10. Fast Volterra convolution

현재 reference adjacent simulator는 history sum을 step별로 다시 계산하므로 실질적으로
quadratic scaling이 나타날 수 있다. Deterministic control에서는 모든 first-driver
target increments를 먼저 생성할 수 있으므로 historical Volterra term을 Toeplitz
convolution으로 계산한다.

Primary fast engine은 다음을 만족해야 한다.

1. recent-cell BLP singular integral은 기존 exact local Gaussian law 유지;
2. historical kernel sum만 zero-padded FFT convolution으로 계산;
3. fine/coarse는 동일 innovation에서 각각 정확한 BLP marginal 유지;
4. padding length는 linear convolution을 보장;
5. float64 reference와 float32 accelerator를 분리;
6. FFT normalization convention unit test;
7. batch/chunk 변경 시 path law와 결과가 변하지 않음.

Expected algorithmic cost 목표는 path당 `O(N log N)`이고 spot recursion은 `O(N)`이다.
Wall-clock exponent를 theorem으로 부르지 않고 operation complexity와 별도로 보고한다.

## 11. Smoothing direction q의 선택

Primary confirmatory method는 가장 단순한

\[
q_i=1/\sqrt N
\]

을 사용한다. 이 선택은 모든 cumulative slope를 양수로 만들고 tuning leakage가 없다.

Development-only 후보는 다음과 같다.

- deterministic time-decay softmax direction;
- expected volatility sensitivity 기반 direction;
- nonnegative control-aligned direction.

`q`는 model parameter와 level에는 의존할 수 있지만 evaluation Brownian path에는 의존할
수 없다. Development seed에서 선택한 뒤 config hash와 함께 freeze한다. Adaptive
`q(W1)`는 조건부로 이론화할 가능성이 있지만 첫 논문에서는 금지한다.

## 12. Optional honest conditional-mean control variate

MGVS가 G9-2와 G9-3을 통과한 뒤에도 residual variance가 충분히 크면 coarse feature
`C`로

\[
g_\theta(C)\approx E[\widetilde Y_\ell\mid C]
\]

를 학습한다. Primary estimator는 cross-fitting보다 검증이 단순한 honest split을 쓴다.
Training data, fine evaluation data, coarse-mean data는 서로 독립이다.

고정된 `g`와 coefficient `lambda`에 대해

\[
\widehat\mu_{n,m}
=\frac1n\sum_{i=1}^n
[\widetilde Y_i-\lambda g(C_i)]
+\frac\lambda m\sum_{j=1}^m g(\widetilde C_j)
\]

는 training data에 조건부로 비편향이다. Fine/coarse evaluation sample이 독립이면

\[
\operatorname{Var}(\widehat\mu\mid g)
=\frac{\operatorname{Var}(\widetilde Y-\lambda g)}n
+\frac{\lambda^2\operatorname{Var}(g)}m.
\]

고정된 `n,m`과 `Var(g)>0`에서 development-law optimal coefficient는

\[
\lambda^*_{n,m}
=\frac{\operatorname{Cov}(\widetilde Y,g)}
{\operatorname{Var}(g)(1+n/m)}.
\]

고정된 `lambda`에 대해

\[
A=\operatorname{Var}(\widetilde Y-\lambda g),
\qquad B=\lambda^2\operatorname{Var}(g)
\]

이고 fine/coarse cost가 `c_f,c_c`라면 fixed-budget continuous allocation은

\[
\frac nm=\sqrt{\frac{A c_c}{B c_f}}
\]

이다. `lambda`와 allocation은 서로 의존하므로 development covariance/cost로 joint
grid 또는 fixed-point optimization하고 전부 freeze한다. MLMC 전체에서는 각
fine-residual 및 coarse-surrogate component를 별도 `(variance,cost)` 항으로 보고
standard square-root allocation을 적용한다.

`lambda`와 `n:m`은 evaluation sample이 아니라 independent development data에서
선택하고 freeze한다. Unknown surrogate mean을 무시하거나 같은 evaluation data로
coefficient를 재추정한 뒤 ordinary IID standard error를 쓰는 것은 금지한다.

K-fold cross-fitting은 secondary ablation으로만 허용한다. Fold 간 shared training이
만드는 covariance를 무시한 naive standard error는 사용하지 않는다.

## 13. Theorem program

### 필수 정리

1. **T9-1 Monotone representation**: fine/coarse log spot의 affine-in-`Z` 표현.
2. **T9-2 Exact event threshold**: hit와 discrete occupation의 threshold 공식.
3. **T9-3 Controlled Gaussian integration**: `L_perp Phi(a+b)` identity.
4. **T9-4 Exact finite-grid telescoping**: level 합이 finest-grid target과 일치.
5. **T9-5 Rao--Blackwell inequality**: 동일 proposal에서 variance 비증가.
6. **T9-6 FFT marginal preservation**: fast convolution이 reference BLP law를 변경하지 않음.

### 탑저널을 위한 강화 정리

7. **T9-7 Threshold stability**: barrier case의 `L^p` rate.
8. **T9-8 Smoothed correction variance rate**: `beta`와 가정 명시.
9. **T9-9 Work complexity**: measured/algorithmic cost exponent `gamma`와
   `beta` 관계에 따른 MLMC complexity.
10. **T9-10 Occupation extension**: occupation-time boundary regularity 아래 rate 또는
    명시적인 반례/제한.
11. **T9-11 Optional ACV**: honest surrogate estimator의 unbiasedness와 optimal allocation.

`T9-1`부터 `T9-6`까지 증명하지 못하면 논문 프로젝트를 중단한다. `T9-7` 이후가
실패하면 Mathematical Finance 목표를 내리고 computational journal 범위로 제한한다.

## 14. 실험 대상

### 14.1 Oracle ladder

1. scalar Gaussian digital event;
2. Black--Scholes discrete barrier/occupation;
3. Heston을 variance-driver/independent-price-driver basis로 회전한 표현;
4. rBergomi BLP;
5. rBergomi controlled adjacent MLMC.

각 단계에서 이전 단계의 analytic 또는 high-precision reference를 통과해야 다음으로
이동한다.

### 14.2 Primary task ladder

- terminal digital;
- discrete down barrier;
- downside hit-plus-occupation;
- barrier-and-occupation with multiple monitoring horizons.

비단조 payoff는 primary scope 밖이다. 억지로 포함해 general-purpose method라고
주장하지 않는다.

### 14.3 Parameter regimes

Core grid는 최소 다음 축을 포함한다.

- `H`: `0.05, 0.10, 0.20`;
- `eta`: low/base/high;
- `rho`: `-0.3, -0.7, -0.9`;
- maturity: short/base/long;
- event probability: 대략 `1e-2`부터 `1e-6`;
- level: 최소 `N=16`부터 `1024`, FFT scaling은 더 큰 `N` 포함.

모든 조합을 무리하게 실행하지 않고, 사전에 고정한 core regime 12개와 stress regime
6개를 사용한다. 현재 G7/G8 task는 development 사례로만 사용하며 새 논문의 유일한
confirmatory task가 되어서는 안 된다.

## 15. 필수 비교군

동일 target, 동일 bias tolerance, 동일 hardware에서 다음을 비교한다.

1. crude Monte Carlo;
2. frozen time-piecewise CEM single-level;
3. G7 raw controlled MLMC;
4. G8 conditional branching;
5. generic one-dimensional numerical smoothing/root-quadrature baseline;
6. path branching 또는 합리적인 adaptation;
7. adaptive multilevel splitting/subset simulation 중 최소 하나;
8. 가능하면 randomized QMC plus smoothing.

Flow-based importance sampling은 구현과 tuning budget을 공정하게 맞출 수 있을 때만
추가한다. 약한 자체 구현을 SOTA라고 부르는 것은 금지한다.

## 16. Primary metrics

각 method/regime/seed에 다음을 기록한다.

- probability estimate와 confidence interval;
- reference difference z-score;
- single-sample variance, second moment, kurtosis;
- level별 mean/variance/bias proxy;
- variance slope `beta`와 95% interval;
- cost slope `gamma`와 95% interval;
- variance-times-cost 및 RMSE-times-work;
- measured wall-clock, algorithmic work proxy, peak memory;
- training/tuning/failure 비용;
- break-even query count;
- likelihood log-moment diagnostics;
- CDF cancellation/underflow counter;
- batch-size 및 hardware utilization.

`ESS`는 보조 진단일 뿐 primary efficiency metric으로 사용하지 않는다.

## 17. 통계 protocol

### 17.1 Split

- development seeds: architecture, `q`, control, sample allocation 선택;
- calibration seeds: gate threshold와 power calculation;
- confirmatory seeds: 최소 10개, 마지막까지 미사용;
- stress seeds: heavy-tail 및 numerical robustness.

G7/G8 validation seeds `9101--9105`는 이미 관찰했으므로 V9 confirmatory seed로 재사용하지
않는다.

### 17.2 비교 방식

- 가능한 method pair는 common random numbers로 paired 비교;
- seed별 log work ratio를 primary effect로 사용;
- geometric mean과 bootstrap/paired confidence interval 보고;
- slope는 사전 지정 level window에서 weighted regression;
- level window를 결과를 본 뒤 변경하면 exploratory로 라벨링;
- 여러 task/regime을 동시에 주장할 때 family-wise 또는 FDR 보정;
- outlier seed 삭제 금지. Numerical failure는 실패로 기록.

### 17.3 Reproducibility payload

각 frozen result에는 다음을 포함한다.

- config SHA-256;
- git commit;
- Python/PyTorch/SciPy 버전;
- CPU/GPU/OS/thread 설정;
- seed split;
- checkpoint hash;
- warm-up 횟수와 timing synchronization;
- raw per-seed/per-level metrics;
- `allow_nan=False` JSON.

## 18. Milestones와 stop gates

### M0 — symbolic and scalar oracle

구현:

- orthogonal Gaussian decomposition;
- threshold formulas;
- stable signed CDF difference;
- controlled truncated-normal identity.

#### G9-0

모두 통과해야 한다.

- direct event와 threshold event가 non-tie samples에서 bitwise 일치;
- high-precision CDF difference reference 통과;
- likelihood factorization maximum log error `<=1e-11`;
- `Phi(a+b)` quadrature relative/absolute tolerance 통과;
- `q^T R` maximum absolute error `<=1e-12` in float64;
- no-control/control oracle mean difference `|z|<=3`.

실패하면 rBergomi 구현으로 이동하지 않는다.

### M1 — exact rBergomi path reconstruction

구현:

- fixed innovations에서 `X=A+BZ` reconstruction;
- fine/coarse threshold;
- smoothed base/correction estimator.

#### G9-1

- random `Z` replay spot maximum relative error `<=1e-11`;
- direct hard correction conditional average와 analytic smoothing 일치;
- natural/controlled target agreement `|z|<=3`;
- 동일 control에서 smoothed/raw variance ratio의 paired 95% upper bound `<=1.05`;
- finite-grid telescoping finest direct estimate와 `|z|<=3`.

### M2 — FFT Volterra engine

#### G9-2A

- same innovations에서 reference path error `<=1e-11` float64;
- fine/coarse covariance와 BLP local law 통과;
- batch/chunk invariance;
- fitted cost exponent의 95% upper bound `<1.35` over frozen scaling window;
- `N>=1024`에서 reference loop 대비 의미 있는 speedup.

정확성 실패 시 FFT를 폐기하고 reference engine을 유지한다. 성능만 실패하면 논문에서
`O(N log N)` implementation claim을 제거한다.

### M3 — smoothing feasibility

Development와 untouched calibration regimes에서 평가한다.

#### G9-2B

- correction work ratio 대 raw G7 geometric mean `>1.5`;
- core regime 최소 80%에서 ratio `>1`;
- primary hit-plus-occupation variance slope point estimate `>0`;
- slope 95% lower bound `>0`인 core regime이 최소 2/3;
- deep-level kurtosis가 raw estimator보다 유의미하게 안정;
- mean consistency 전부 통과.

실패하면 learned CV, VFO, quantum layer를 추가하지 않고 MGVS 가설을 기각한다.

### M4 — frozen end-to-end comparison

#### G9-3

- single-level frozen CEM 대비 geometric total-work ratio `>2.0`;
- paired 95% interval lower bound `>1`;
- 개선 seed `>=8/10`;
- raw controlled MLMC와 G8 branching 모두 능가;
- training 없는 primary MGVS의 break-even `<=5` queries;
- probability `1e-4--1e-6`에서 안정적인 confidence coverage;
- 모든 exactness/likelihood gate 유지.

이 gate를 통과해야 computational top-journal manuscript를 시작한다.

### M5 — theorem and generalization

#### G9-4

- T9-1--T9-6 complete proof;
- barrier variance-rate theorem 또는 명확한 complexity theorem;
- occupation theorem 성공, 또는 occupation은 empirical extension으로 명시;
- Black--Scholes/Heston/rBergomi oracle ladder 통과;
- core/stress regimes에서 결과 방향 일관;
- systematic novelty review 완료.

### M6 — optional conditional-mean CV

MGVS가 이미 G9-3을 통과한 경우에만 진행한다.

#### G9-5

- honest estimator mean consistency;
- MGVS 단독 대비 incremental total-work ratio `>1.25`;
- training-inclusive break-even `<=20` queries;
- 최소 8/10 seed 개선;
- architecture ablation에서 단순 linear/MLP보다 복잡한 operator가 실제 우위.

실패하면 optional CV를 논문 주모델에서 제거한다.

## 19. 구현 파일 계획

### Core

- `src/path_integral/gaussian_smoothing.py`
- `src/path_integral/rbergomi_smoothing.py`
- `src/path_integral/rbergomi_fft.py`
- `src/evaluation/smoothed_multilevel.py`

### Optional training

- `src/training/conditional_mean_control_variate.py`

### Experiments/configs

- `experiments/m0_gaussian_smoothing_oracle.py`
- `experiments/g9_mgvs_development.py`
- `experiments/g9_mgvs_frozen.py`
- `configs/g9_mgvs_development.yaml`
- `configs/g9_mgvs_frozen.yaml`

### Tests

- `tests/test_gaussian_smoothing.py`
- `tests/test_rbergomi_smoothing.py`
- `tests/test_rbergomi_fft.py`
- `tests/test_smoothed_multilevel.py`
- `tests/test_conditional_mean_control_variate.py` if M6 starts.

모듈은 기존 G7/G8 reference implementation을 삭제하거나 덮어쓰지 않는다. 새로운 fast
engine은 동일 innovations replay test를 통해 reference와 대조한다.

## 20. 논문 figure/table 사전 설계

필수 figure:

1. Gaussian coordinate와 affine log-spot geometry;
2. raw indicator correction과 smoothed CDF correction 비교;
3. level별 variance/kurtosis decay;
4. RMSE versus total work;
5. parameter/rarity regime robustness;
6. FFT scaling과 memory;
7. 실패/성공 영역 phase diagram.

필수 table:

1. theorem assumptions와 실제 experiment coverage;
2. baseline별 exactness/variance/work;
3. training-inclusive break-even;
4. ablation: no smoothing, root-quadrature smoothing, analytic smoothing,
   branching, optional CV;
5. claim-to-evidence matrix.

## 21. 저널별 요구 수준과 투고 결정

### Primary: SIAM Journal on Financial Mathematics

Official scope:
<https://www.siam.org/publications/siam-journals/siam-journal-on-financial-mathematics/>

필요 조건:

- exact estimator 정리;
- significant, non-incremental computational improvement;
- rough-volatility 금융 문제에 대한 명확한 relevance;
- 여러 regime의 재현 가능한 검증.

G9-3과 G9-4를 모두 통과하면 현실적인 primary target이다.

### Alternative: SIAM Journal on Scientific Computing

Official scope:
<https://www.siam.org/publications/siam-journals/siam-journal-on-scientific-computing>

필요 조건:

- rBergomi를 넘어 two-driver stochastic-volatility/Volterra class로 표현;
- fast convolution과 numerical smoothing의 일반 계산 기여;
- 강한 effectiveness/reproducibility evidence.

### Stretch: Mathematical Finance / Finance and Stochastics

Mathematical Finance official scope:
<https://onlinelibrary.wiley.com/page/journal/14679965/homepage/productinformation.html>

필요 조건:

- threshold/variance rate 또는 continuous/discrete monitoring limit의 강한 정리;
- 금융적으로 중요한 sensitivity 또는 risk insight;
- 단순 computational trick을 넘어서는 methodological novelty.

정확한 finite-grid theorem과 실험 우위만 있고 rate theorem이 없으면 이 단계로 바로
제출하지 않는다.

### 투고 readiness matrix

| Exact theorem | Total-work gate | Rate theorem | 결정 |
|---|---|---|---|
| 실패 | 무관 | 무관 | 프로젝트 중단, 논문 주장 금지 |
| 통과 | 실패 | 통과/실패 | 검증된 component 또는 negative result로 보존; 탑저널 제출 금지 |
| 통과 | 통과 | 실패 | SIFIN/SISC computational scope 검토, Mathematical Finance 제외 |
| 통과 | 통과 | barrier만 통과 | barrier theorem + occupation empirical 논문으로 범위 제한 |
| 통과 | 통과 | occupation 포함 통과 | Mathematical Finance/Finance and Stochastics stretch 가능 |

## 22. 예상 일정

| 기간 | 목표 |
|---|---|
| 1--2주 | systematic literature matrix, symbolic proof draft |
| 3--4주 | M0 oracle와 stable CDF primitive |
| 5--7주 | M1 rBergomi smoothing과 exactness tests |
| 8--10주 | M2 FFT engine과 scaling audit |
| 11--13주 | M3 development/calibration experiments |
| 14--16주 | M4 frozen confirmatory experiment |
| 17--20주 | theorem completion, generalization, ablations |
| 21--24주 | manuscript, independent reproduction, internal review |

Gate 실패 시 일정은 다음 단계로 자동 이동하지 않고 해당 가설의 종료 문서 작성으로
전환한다.

## 23. 주요 위험과 사전 대응

| 위험 | 대응 |
|---|---|
| numerical smoothing 자체는 기존 연구와 중복 | rough-Volterra exact threshold, controlled likelihood, adjacent BLP 교집합만 신규성 후보로 제한 |
| occupation threshold rate 증명 실패 | barrier theorem + occupation empirical extension으로 claim 축소 |
| CDF 차이 cancellation | signed log-CDF primitive와 high-precision oracle |
| deterministic control만 허용되어 표현력 제한 | strong CEM baseline과 먼저 승부; feedback은 후속 논문 |
| FFT가 reference law를 바꿈 | same-innovation pathwise replay gate |
| wall-clock noise | operation proxy, fixed hardware, warm-up, paired seeds |
| 새 validation leakage | G7/G8 seeds 폐기, 새 seed hash freeze |
| generic neural CV가 논문을 흐림 | M6 optional gate 이전에는 구현하지 않음 |
| rare-event reference 부정확 | independent methods와 confidence-budgeted reference 사용 |
| continuous-time 과대주장 | discrete monitoring을 primary target으로 명시 |

## 24. 최종 논문 claim checklist

다음 질문에 모두 `yes`가 아니면 탑저널 제출본으로 부르지 않는다.

- [ ] Exact finite-grid estimator proof가 완결됐는가?
- [ ] `Phi(a+b)` sign과 event threshold가 독립 oracle로 검증됐는가?
- [ ] Fine/coarse BLP marginal이 reference와 정확히 일치하는가?
- [ ] Smoothed correction이 raw correction을 동일 proposal에서 능가하는가?
- [ ] Positive variance decay가 untouched regimes에서 재현되는가?
- [ ] Single CEM 대비 total work가 최소 `2x` 개선되는가?
- [ ] 최소 8/10 seed와 paired confidence interval이 개선을 지지하는가?
- [ ] Training/tuning/failed runs가 비용에 포함됐는가?
- [ ] Current literature와의 차별점이 교집합 수준으로 명확한가?
- [ ] Discrete/continuous claim이 분리됐는가?
- [ ] Code/config/result가 fresh environment에서 재현되는가?
- [ ] 부정적 ablation과 실패 regime도 공개했는가?

## 25. 최종 결정

V9의 우선순위는 다음과 같다.

1. **MGVS exact analytic smoothing**;
2. **fast exact BLP convolution**;
3. **variance-rate 및 total-work 검증**;
4. **필요할 때만 honest conditional-mean control variate**.

VFO, attention, higher dimension, quantum-inspired layer는 위 네 단계의 대체물이 아니다.
MGVS가 G9-3을 통과하지 못하면 해당 복잡성을 추가하지 않고 방향을 다시 선택한다.
