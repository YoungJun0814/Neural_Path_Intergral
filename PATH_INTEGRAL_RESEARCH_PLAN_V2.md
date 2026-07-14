# Neural Path-Integral Rare-Event Engine: 연구·구현·투고 계획 v2

> Historical detailed reference. Forward execution is governed by
> [PATH_INTEGRAL_RESEARCH_PLAN_V3.md](PATH_INTEGRAL_RESEARCH_PLAN_V3.md).

Status: active design draft<br>
Version: 2.0<br>
Date: 2026-07-13<br>
Repository: Neural_Path_Integral<br>
Relationship to prior plan: PUBLICATION_GRADE_RESEARCH_PLAN.md의 G0–G2 성과는 유지하되, 앞으로의 방법론 중심을 실제 path-integral stochastic control로 변경한다.

---

## 0. 문서의 목적과 최종 의사결정

이 문서는 다음 세 목표를 동시에 만족하는 연구를 완성하기 위한 기준 문서다.

1. 경로적분을 이름이나 비유가 아니라 실제 수학적·알고리즘적 핵심으로 사용한다.
2. 저명한 금융수학 또는 계산수학 저널에서 독립적인 방법론 기여로 평가받을 수준의 이론과 증거를 만든다.
3. 한 번 학습한 제어기를 여러 금융 희귀사건 질의에 재사용할 수 있는 실무형 Monte Carlo 엔진으로 만든다.

최종 권장 연구 질문은 다음과 같다.

> Brownian path-space의 path-integral/KL control로 정의되는 payoff-tilted optimal path measure를, rough-volatility의 causal memory state와 금융 task parameter에 조건화된 neural controller로 근사할 수 있는가? 그리고 이 근사가 정확한 likelihood를 유지하면서 강한 고전적·신경망 baseline보다 동일 총 계산비용당 더 작은 희귀사건 추정오차를 제공하는가?

권장 논문 제목은 다음과 같다.

> Neural Path-Integral Importance Sampling for Rare Events under Rough Volatility

이 제목은 desirability/free-energy representation, path action, optimal tilted path measure, path-integral policy update가 실제 논문과 코드에 포함될 때만 사용한다.

### 0.1 한 문장 기여 후보

> We formulate rare-event estimation under rough volatility as path-integral control of the Brownian drivers, approximate the resulting payoff-tilted Föllmer drift with a causal amortized memory controller, and quantify how control and memory approximation affect work-normalized importance-sampling error.

이 문장은 선행연구 검토와 정리 증명이 끝나기 전에는 초록의 확정 주장으로 사용하지 않는다.

### 0.2 연구의 중심과 보조 요소

| 구분 | 역할 |
|---|---|
| Path-integral/KL control | 논문의 수학적 중심 |
| Payoff-tilted path measure | 최적 proposal의 정의 |
| Girsanov likelihood | 정확한 estimator의 기반 |
| Neural controller | 최적 path law의 함수 근사기 |
| Rough-memory representation | 비마르코프 신규성 |
| Amortization | 실무 재사용성과 계산비용 신규성 |
| Heston | analytic/oracle correctness benchmark |
| rBergomi | 핵심 비마르코프 응용 |
| CEM/large deviations/FNN IS | 강한 비교 기준 |

### 0.3 명시적으로 하지 않을 주장

- 단순 Feynman–Kac 표현만으로 새로운 path-integral pricing 방법을 만들었다고 주장하지 않는다.
- PICE, CEM, Girsanov 또는 Boué–Dupuis 표현 자체를 신규 정리라고 주장하지 않는다.
- hard-event conditional law가 유한한 bounded drift로 정확히 구현된다고 주장하지 않는다.
- path-integral KL loss가 감소했다는 사실만으로 estimator variance가 감소했다고 주장하지 않는다.
- finite-step Gaussian mean-shift proposal이 임의의 discrete tilted path law를 정확히 표현한다고 주장하지 않는다.
- rBergomi의 auxiliary Markovian lift를 사용했다고 해서 continuous rough model의 bias가 제거됐다고 주장하지 않는다.
- risk-neutral pricing과 physical-measure stress probability를 같은 실험에서 혼합하지 않는다.
- Neural Path Integral이라는 명칭을 path-integral 유도 없이 사용하지 않는다.

---

## 1. 현재 연구 상태와 v2 전환 이유

### 1.1 이미 확보한 기반

현재 저장소에서 재사용할 기반은 다음과 같다.

- Heston full-truncation variance와 positive conditional log-Euler spot
- Heston characteristic function, terminal CDF, rare threshold
- correlated Heston Girsanov drift correction
- rBergomi BLP hybrid simulation과 discrete Wick normalization
- constant-control trajectory-likelihood CEM
- Markov affine/MLP feedback controller
- hard-indicator second-moment score gradient
- frozen control evaluation
- log-domain likelihood와 contribution diagnostics
- train/selection/audit/final-evaluation seed 분리
- online 및 end-to-end work-normalized efficiency framework

G2 Heston 결과는 compact affine과 CEM이 work 기준으로 거의 동률이라는 negative benchmark를 제공한다. 이 결과를 숨기거나 다시 tuning하여 superiority를 만들지 않는다.

### 1.2 기존 방향의 한계

일반적인 neural importance sampling은 이미 이론과 금융 수치실험을 포함해 출판되어 있다. Path-integral control과 cross-entropy adaptive importance sampling의 연결도 기존 PICE 연구에 존재한다. 따라서 다음 조합만으로는 신규성이 부족하다.

- neural drift + Girsanov
- path-dependent option + FNN
- path integral + CEM
- rough-volatility simulation + neural network

v2의 신규성 후보는 네 요소의 결합과 그 사이의 정량적 연결에 있다.

1. Brownian path-space의 payoff-tilted optimal law
2. rough memory를 사용하는 causal adapted control
3. 여러 금융 task에 재사용되는 amortized proposal
4. control/memory approximation과 Rényi-2 또는 relative variance의 관계

### 1.3 유지·전환·폐기

| 기존 요소 | v2 처리 |
|---|---|
| Heston CEM/affine benchmark | 유지, oracle 비교용 |
| direct hard-event second moment | 유지, 최종 refinement objective |
| entropy-stress objective | path-integral free-energy objective로 재정의 |
| rBergomi BLP simulator | 유지, controlled version 추가 |
| Markov MLP | state-only baseline으로 유지 |
| Neural Path Integral 명칭 | path-integral 모듈이 완성된 뒤 활성화 |
| jump extension | 본 논문 핵심에서 제외 |
| physical VaR/ES | 후속 연구로 분리 |

---

## 2. 학술적 위치와 신규성 경계

### 2.1 반드시 인정해야 하는 선행 결과

1. Boué–Dupuis variational representation은 Brownian functional의 free energy를 adapted drift control 문제로 표현한다.
2. Path-integral control은 linearly-solvable stochastic control에서 HJB를 선형화하고 Feynman–Kac expectation으로 control을 표현한다.
3. PICE는 path-integral optimal sampler를 parametrized feedback controller로 근사하기 위해 cross entropy를 사용한다.
4. Neural importance sampling for option pricing은 feedforward network, Cameron–Martin approximation, path-dependent option 및 barrier 사례를 이미 다룬다.
5. FBSDE, Koopman 및 large-deviation 기반 rare-event importance sampling이 강한 대안으로 존재한다.
6. rBergomi의 hybrid simulation과 sum-of-exponentials/Markovian approximation은 이미 확립된 연구 분야다.
7. 2026년 fully non-Markovian control과 off-model importance sampling 연구는 broad novelty claim을 더 좁힌다.

### 2.2 우리가 주장할 수 있는 차별점 후보

아래 문장은 related-work matrix가 완성된 뒤에만 확정한다.

- rough-volatility rare-event probability를 위한 path-integral feedback proposal
- BLP target simulator에 exact discrete Brownian likelihood를 결합한 joint two-driver control
- target dynamics를 lift로 대체하지 않고 auxiliary causal lift를 controller feature로만 사용하는 구조
- event/model parameter에 조건화된 amortized path-integral sampler
- path-integral control approximation error와 work-normalized estimator 성능의 정량적 연결
- Heston conditional transform를 이용한 oracle-to-neural validation

first, 최초, 유일 같은 표현은 systematic literature review가 끝날 때까지 금지한다.

### 2.3 저널별 필요한 기여

| 목표 | 필요한 수준 |
|---|---|
| Finance and Stochastics / Mathematical Finance | 비자명한 theorem, 명확한 금융 insight, 강한 rough-volatility 결과 |
| SIAM Journal on Financial Mathematics | 엄밀한 계산방법, significant improvement, reproducible evidence |
| Journal of Computational Physics | 금융을 넘어선 Volterra/SDE 일반성, complexity와 robustness |
| Quantitative Finance / Journal of Computational Finance | 강한 실무 benchmark와 재현성, 이론은 다소 약해도 가능 |

---

## 3. 측도, task 및 target functional

### 3.1 측도 convention

- \(\mathbb M\): 추정하려는 금융 target measure
- \(\mathbb Q_{\phi,\zeta}\): task \(\zeta\)에 대한 controlled proposal
- \(L_{\phi,\zeta}=d\mathbb M/d\mathbb Q_{\phi,\zeta}\)
- \(\zeta\): strike, barrier, maturity, event type 및 model parameter를 포함하는 task

모든 measure change와 divergence의 기본 공간은 state path만이 아니라 Brownian
drivers를 포함한 augmented canonical path space다. State \(X\)는 이 driver
path의 measurable map으로 본다. State map이 one-to-one이 아닐 수 있으므로
Brownian-path divergence와 state-law divergence를 무조건 동일시하지 않는다.
Estimator variance identity는 실제 likelihood가 정의되는 augmented sampling
space에서 사용한다.

본 논문의 main application은 risk-neutral pricing으로 고정한다.

$$
\mathbb M=\mathbb Q^{RN},
\qquad
\mu_t=r_t-q_t.
$$

Physical-measure stress probability는 drift, calibration, reference probability가 별도로 필요하므로 후속 extension으로 둔다.

### 3.2 추정 대상

Main target은 bounded nonnegative path functional이다.

$$
\mu_\zeta
=
\mathbb E_{\mathbb M}[G_\zeta(X_{[0,T]})].
$$

우선순위는 다음과 같다.

1. terminal left-tail digital event
2. down-crossing/barrier event
3. drawdown 또는 running-minimum event
4. deep-OTM bounded payoff

Unbounded vanilla payoff는 path-integral potential \(-\log G\)의 integrability와 zero region을 별도로 처리해야 하므로 main theorem의 첫 범위에서 제외한다.

### 3.3 hard event와 soft potential

Hard event \(A_\zeta\)에 대해

$$
G_\zeta=1_{A_\zeta}
$$

이면 \(-\log G_\zeta\)는 사건 밖에서 무한대다. 학습에는 strictly positive soft functional을 사용한다.

$$
g_{\tau,\zeta}(X)
=
\exp[-\Phi_{\tau,\zeta}(X)],
\qquad
0<g_{\tau,\zeta}\le 1.
$$

예를 들어 terminal left-tail에는

$$
g_{\tau,K}(S_T)
=
\operatorname{sigmoid}\left(\frac{K-S_T}{\tau K}\right)
$$

를 사용할 수 있다. Barrier/drawdown에는 signed path margin을 만든 뒤 같은 sigmoid를 적용한다.

중요한 원칙:

- soft functional은 proposal training용이다.
- 최종 확률 estimator에는 hard indicator를 사용한다.
- \(\tau\)는 train 또는 validation에서만 선택한다.
- final evaluation 결과를 본 뒤 \(\tau\)를 수정하지 않는다.

---

## 4. Path-integral control의 수학적 중심

### 4.1 payoff-tilted optimal path measure

Soft normalizing constant를

$$
Z_{\tau,\zeta}
=
\mathbb E_{\mathbb M}[g_{\tau,\zeta}(X)]
$$

라 두고 optimal tilted law를

$$
\frac{d\mathbb Q_{\tau,\zeta}^{\star}}{d\mathbb M}
=
\frac{g_{\tau,\zeta}(X)}{Z_{\tau,\zeta}}
$$

로 정의한다.

이 law에서 importance sampling contribution은 상수 \(Z_{\tau,\zeta}\)가 되므로 이상적인 zero-variance proposal이다.

Hard event의 경우

$$
\frac{d\mathbb Q_{A,\zeta}^{\star}}{d\mathbb M}
=
\frac{1_{A_\zeta}}{p_\zeta}
$$

는 event-conditioned law다. 이 law는 \(\mathbb M\)과 equivalent하지 않으며 사건 밖에 질량이 없다. 따라서 bounded Brownian drift proposal이 hard conditional law를 정확히 구현한다고 주장하면 안 된다.

### 4.2 Boué–Dupuis/path-integral variational problem

Brownian-driven target model과 admissible progressively measurable control \(u\)에 대해 formal target은

$$
-\log Z_{\tau,\zeta}
=
\inf_u
\mathbb E_{\mathbb Q_u}
\left[
\Phi_{\tau,\zeta}(X^u)
+\frac12\int_0^T\|u_t\|^2dt
\right].
$$

Control은 Brownian noise와 같은 channel로 들어가며 quadratic energy는 path-space relative entropy와 대응한다.

$$
\mathcal J_{\mathrm{PI}}(u;\tau,\zeta)
=
\mathbb E_{\mathbb Q_u}[\Phi_{\tau,\zeta}(X^u)]
+\mathrm{KL}(\mathbb Q_u\Vert\mathbb M).
$$

표현 가능한 law 범위에서

$$
\mathcal J_{\mathrm{PI}}(u;\tau,\zeta)+\log Z_{\tau,\zeta}
=
\mathrm{KL}(\mathbb Q_u\Vert\mathbb Q_{\tau,\zeta}^{\star}).
$$

따라서 PI objective는 forward KL을 최소화한다.

Markov 또는 finite-dimensional lifted state

$$
dZ_t=b(Z_t)dt+\sigma(Z_t)\left(dB_t+u_tdt\right)
$$

에서는 value \(V(t,z)\)가 formal HJB

$$
\partial_tV+\mathcal LV
-\frac12\left\|\sigma^\top\nabla V\right\|^2=0,
\qquad
V(T,z)=\Phi_{\tau,\zeta}(z)
$$

를 만족하고

$$
u^\star=-\sigma^\top\nabla V.
$$

Desirability transform

$$
\psi=e^{-V}
$$

를 적용하면

$$
\partial_t\psi+\mathcal L\psi=0,
\qquad
\psi(T,z)=e^{-\Phi_{\tau,\zeta}(z)},
$$

$$
u^\star=\sigma^\top\nabla\log\psi
$$

가 된다. 이 linearization과 Feynman–Kac expectation이 논문에서
path-integral control이라는 명칭을 정당화하는 Markov/lifted 표현이다.
Degenerate lifted diffusion에서는 classical solution을 자동 가정하지 않고
mild/viscosity 또는 path-space 표현을 우선한다.

이 linearization은 control이 noise와 동일한 column space로 들어가고 control
cost가 \(\frac12\|u\|^2\)인 matching condition에 의존한다. 이후 state drift에
독립 control channel이나 임의의 control penalty를 추가하면 같은 linear
desirability equation을 그대로 사용할 수 없다.

### 4.3 importance-sampling variance와 Rényi-2

임의 proposal \(\mathbb Q\)에서

$$
Y
=
g_{\tau,\zeta}(X)\frac{d\mathbb M}{d\mathbb Q}
$$

라 하면

$$
\frac{Y}{Z_{\tau,\zeta}}
=
\frac{d\mathbb Q_{\tau,\zeta}^{\star}}{d\mathbb Q}.
$$

따라서 정확히

$$
\frac{\operatorname{Var}_{\mathbb Q}(Y)}
{Z_{\tau,\zeta}^2}
=
\chi^2(\mathbb Q_{\tau,\zeta}^{\star}\Vert\mathbb Q),
$$

$$
\log\left(
1+\frac{\operatorname{Var}_{\mathbb Q}(Y)}{Z_{\tau,\zeta}^2}
\right)
=
D_2(\mathbb Q_{\tau,\zeta}^{\star}\Vert\mathbb Q).
$$

이 항등식이 path-law approximation과 estimator variance를 연결하는 중심이다.

### 4.4 반드시 구분해야 하는 divergence

- PI free-energy objective gap: \(\mathrm{KL}(\mathbb Q_u\Vert\mathbb Q^\star)\)
- PICE/cross-entropy projection: \(\mathrm{KL}(\mathbb Q^\star\Vert\mathbb Q_\phi)\)
- estimator relative variance: \(\chi^2(\mathbb Q^\star\Vert\mathbb Q_\phi)\)
- log relative second moment: \(D_2(\mathbb Q^\star\Vert\mathbb Q_\phi)\)

일반적으로 작은 forward KL은 작은 reverse chi-square를 보장하지 않는다. 따라서 PI loss만 보고 variance reduction을 주장하지 않는다.

권장 학습은 다음 세 단계를 결합한다.

1. PI free-energy 또는 soft potential로 proposal 초기화
2. PICE-style reverse-KL distribution matching
3. exact second-moment/Rényi-2 refinement

### 4.5 path action과 off-policy weight

Proposal \(\mathbb Q_u\)에서

$$
\log L_u
=
-\int_0^T u_t^\top dB_t^{\mathbb Q_u}
-\frac12\int_0^T\|u_t\|^2dt.
$$

Path action을

$$
\mathcal S_u(X)
=
\Phi_{\tau,\zeta}(X^u)
+\int_0^T u_t^\top dB_t^{\mathbb Q_u}
+\frac12\int_0^T\|u_t\|^2dt
$$

로 두면

$$
\omega_u(X)
=
g_{\tau,\zeta}(X^u)L_u
=
\exp[-\mathcal S_u(X)].
$$

이 \(\omega_u\)가 실제 path-integral trajectory weight다.

Normalized weight

$$
\bar\omega_i
=
\frac{\omega_i}{\sum_j\omega_j}
$$

는 PICE training에만 사용한다. Self-normalized estimator는 finite-sample bias가 있으므로 최종 확률·가격 보고에는 사용하지 않는다.

### 4.6 continuous theory와 discrete implementation의 차이

Continuous Brownian filtration에서는 positive tilted density가 martingale representation/Föllmer drift를 가질 수 있다. 그러나 finite-step simulator가 사용하는 left-adapted Gaussian mean shift는 임의의 discrete tilted transition law를 정확히 표현하지 못할 수 있다.

예를 들어 discrete Doob transition

$$
p^\star(x_{k+1}\mid x_k)
\propto
p(x_{k+1}\mid x_k)h_{k+1}(x_{k+1})
$$

는 일반적으로 Gaussian mean shift만으로 표현되지 않는다.

따라서 다음을 분리한다.

- continuous soft target의 Föllmer/Doob optimal drift
- time-discretized piecewise-constant control approximation
- fixed-grid proposal class 내부의 최적 second moment
- neural approximation error

Fixed grid에서 zero variance를 달성한다고 주장하지 않는다. Zero-variance law는 이론적 target이며, 구현 proposal은 근사다.

BLP recent-cell variable에는 \(\Delta B^1\)로 완전히 결정되지 않는 Brownian
bridge component가 포함된다. Piecewise-constant drift control은 이 component를
독립적으로 tilt하지 않는다. 이것도 fixed-grid proposal-class approximation의
일부이며, local integral에 별도 임의 control을 추가하려면 새로운 joint density와
continuous-time interpretation이 필요하다.

---

## 5. Heston path-integral oracle benchmark

### 5.1 독립 Brownian basis

로그 가격 \(X_t=\log S_t\)를 사용하고 독립 Brownian motions \(B^1,B^2\)로

$$
dX_t
=
\left(r-q-\frac12v_t\right)dt
+\sqrt{v_t}\,dB_t^{1,\mathbb M},
$$

$$
dv_t
=
\kappa(\theta-v_t)dt
+\xi\sqrt{v_t}
\left(
\rho\,dB_t^{1,\mathbb M}
+\sqrt{1-\rho^2}\,dB_t^{2,\mathbb M}
\right)
$$

로 쓴다.

두 Brownian driver를

$$
dB_t^{i,\mathbb M}
=
dB_t^{i,\mathbb Q}
+u_t^i dt,
\qquad i=1,2
$$

로 이동시키면

$$
\log L_u
=
-\sum_{i=1}^2\int_0^T u_t^i dB_t^{i,\mathbb Q}
-\frac12\sum_{i=1}^2\int_0^T|u_t^i|^2dt.
$$

현재 1차원 spot-basis control은 \(u^2=0\)인 제한된 baseline으로 해석한다.

### 5.2 soft desirability와 oracle drift

Soft terminal/path functional에 대해

$$
h_{\tau,\zeta}(t,x,v)
=
\mathbb E_{\mathbb M}
\left[
g_{\tau,\zeta}(X_{[0,T]})
\mid X_t=x,v_t=v
\right].
$$

Terminal Markov functional이면 Heston characteristic function/CDF를 이용해 \(h\)를 계산할 수 있다.

Continuous-time Doob/Föllmer drift 후보는

$$
u_1^\star
=
\sqrt v\,\partial_x\log h
+\rho\xi\sqrt v\,\partial_v\log h,
$$

$$
u_2^\star
=
\xi\sqrt{1-\rho^2}\sqrt v\,\partial_v\log h.
$$

이 formula는 diffusion matrix의 transpose와 \(\nabla\log h\)의 곱이다.

### 5.3 oracle의 사용 원칙

- hard terminal indicator 대신 soft \(h_\tau\)로 derivative 안정성을 확보한다.
- Fourier inversion, finite difference 또는 automatic differentiation의 수치오차를 교차검증한다.
- \(t\to T\), \(h\to0\), deep tail에서 drift blow-up 가능성을 점검한다.
- Feller condition이 깨지는 parameter에서는 \(v=0\) 경계의 classical
  smoothness를 자동 가정하지 않는다. Oracle derivative는 \(v>0\) 영역에서
  검증하고 boundary regularization/refinement를 별도 보고한다.
- bounded/clipped oracle는 exact oracle가 아니라 admissible approximation으로 표시한다.
- oracle와 learned controller를 동일 time grid와 Brownian basis에서 비교한다.
- oracle control의 continuous formula와 discrete estimator 사이에는 refinement study가 필요하다.

### 5.4 Heston이 답해야 할 질문

1. PI/PICE 학습이 analytic control 방향을 회복하는가?
2. full two-driver control이 one-driver control보다 필요한가?
3. PI initialization이 direct hard-event score training의 안정성을 개선하는가?
4. learned control이 CEM보다 못할 때 그 원인이 architecture, objective, 또는 discretization인가?
5. control \(L^2\) error와 observed log second moment 사이에 정량적 관계가 있는가?

---

## 6. rBergomi controlled path-space formulation

### 6.1 target model

Risk-neutral target measure에서

$$
Y_t
=
\sqrt{2H}
\int_0^t(t-s)^{H-\frac12}dB_s^{1,\mathbb M},
$$

$$
V_t
=
\xi_0(t)
\exp\left(
\eta Y_t-\frac12\eta^2t^{2H}
\right),
$$

$$
dX_t
=
\left(r_t-q_t-\frac12V_t\right)dt
+\sqrt{V_t}
\left(
\rho\,dB_t^{1,\mathbb M}
+\sqrt{1-\rho^2}\,dB_t^{2,\mathbb M}
\right).
$$

초기 구현은 flat \(\xi_0\)를 사용하되, 실무 application에서는 positive spline/PCA basis로 \(\xi_0(t)\)를 표현한다.

Risk-neutral pricing claim에는 discounted stock의 true-martingale 성질이
필요하다. Main parameter range는 우선 실무적으로 일반적인 \(\rho<0\) 영역으로
제한하고, parameter assumption 또는 알려진 martingale condition을 명시한다.
Positive-\(\rho\) regime은 strict-local-martingale 가능성을 별도 검토하기 전에는
pricing claim에 포함하지 않는다. 모든 refinement에서

$$
E_{\mathbb M}[e^{-(r-q)T}S_T]\approx S_0
$$

를 진단한다.

### 6.2 proposal measure와 memory correction

$$
dB_t^{i,\mathbb M}
=
dB_t^{i,\mathbb Q_u}+u_t^i dt.
$$

그러면

$$
Y_t^{\mathbb M}
=
\sqrt{2H}\int_0^t(t-s)^{H-\frac12}dB_s^{1,\mathbb Q_u}
+C_t^u,
$$

$$
C_t^u
=
\sqrt{2H}\int_0^t(t-s)^{H-\frac12}u_s^1ds.
$$

\(u_s^1\)이 feedback control이면 \(C_t^u\)는 random, causal, finite-variation path functional이다. Deterministic인 것은 kernel이지 correction 자체가 아니다.

Proposal log-price는

$$
dX_t^u
=
\left[
r_t-q_t-\frac12V_t^u
+\sqrt{V_t^u}
\left(
\rho u_t^1+\sqrt{1-\rho^2}u_t^2
\right)
\right]dt
$$

$$
\qquad
+\sqrt{V_t^u}
\left(
\rho\,dB_t^{1,\mathbb Q_u}
+\sqrt{1-\rho^2}\,dB_t^{2,\mathbb Q_u}
\right),
$$

$$
V_t^u
=
\xi_0(t)
\exp\left[
\eta\left(Y_t^{\mathbb Q_u}+C_t^u\right)
-\frac12\eta^2t^{2H}
\right].
$$

Likelihood는

$$
\log L_u
=
-\sum_{i=1}^2\int_0^T u_t^i dB_t^{i,\mathbb Q_u}
-\frac12\sum_{i=1}^2\int_0^T|u_t^i|^2dt.
$$

### 6.3 BLP discrete controlled correction

\(\alpha=H-\frac12\), grid size \(h\)라 하자. BLP \(\kappa=1\) hybrid scheme의 recent cell integral은

$$
I_i
=
\int_{t_{i-1}}^{t_i}(t_i-s)^\alpha dB_s^1.
$$

Left-point constant control \(u_{i-1}^1\)에 의한 exact local mean shift는

$$
\Delta I_i^u
=
u_{i-1}^1
\frac{h^{\alpha+1}}{\alpha+1}.
$$

Earlier cell의 average kernel weight를 \(\bar K_{i,j}\)라 하면

$$
C_{h,i}^u
=
\sqrt{2H}
\left[
u_{i-1}^1\frac{h^{\alpha+1}}{\alpha+1}
+\sum_{j=0}^{i-2}\bar K_{i,j}u_j^1h
\right].
$$

이 correction은 반드시 uncontrolled BLP에 사용한 동일 lower-triangular discrete operator로 계산한다. Continuous convolution을 별도로 근사해 섞으면 pathwise reconstruction과 likelihood가 불일치할 수 있다.

Recent-cell Gaussian pair \((\Delta B_i^1,I_i)\)의 mean은

$$
\left(
u_{i-1}^1h,
\quad
u_{i-1}^1\frac{h^{\alpha+1}}{\alpha+1}
\right)
$$

만큼 이동한다. 이 vector는 joint covariance의 첫 번째 column에 \(u_{i-1}^1\)를 곱한 것이므로 joint likelihood는 여전히

$$
-u_{i-1}^1\Delta B_i^{1,\mathbb Q}
-\frac12(u_{i-1}^1)^2h
$$

이다. 별도의 \(I_i\) likelihood 항을 중복 추가하면 안 된다.

### 6.4 target simulator와 memory feature의 분리

Main estimator는 BLP target simulator를 유지한다. Sum-of-exponentials lift는 우선 controller feature로만 사용한다.

이 설계의 장점:

- estimator target은 BLP discretized rBergomi로 고정된다.
- auxiliary lift 오차는 model bias가 아니라 proposal efficiency에만 영향을 준다.
- lift rank를 바꿔도 likelihood estimator의 기대값은 바뀌지 않는다.
- exact BLP와 lifted simulator의 차이를 논문에서 명확히 분리할 수 있다.

Auxiliary lift는

$$
Z_t^m
\approx
\int_0^t e^{-\lambda_m(t-s)}dB_s^{1,\mathbb M}
$$

로 정의하고 controlled simulation에서는 past reconstructed increment

$$
dB_s^{1,\mathbb M}
=
dB_s^{1,\mathbb Q}+u_s^1ds
$$

를 사용해 causal하게 갱신한다. 현재 step의 control을 계산할 때는 현재 increment 이전의 lift state만 사용한다.

Lifted model 자체를 simulator로 사용할 경우에는 kernel/model approximation bias가 생기므로 별도 ablation으로만 보고한다.

### 6.5 exact feedback convolution의 계산복잡도

Uncontrolled BLP noise path \(Y^{\mathbb Q}\)는 전체 Brownian increment를 미리
알기 때문에 matrix multiplication 또는 FFT로 일괄 계산할 수 있다. 반면
feedback \(u_k^1\)는 \(t_k\)의 state를 본 뒤 결정되므로 control correction
\(C_{h,i}^u\)의 전체 sequence를 simulation 시작 전에 알 수 없다.

따라서 exact controlled BLP의 naive causal 구현은 path당

$$
O(n^2)
$$

연산을 요구한다. 일반적인 one-shot FFT를 적용해 \(O(n\log n)\)이라고 바로
주장하면 안 된다.

구현 hierarchy:

1. \(O(n^2)\) exact causal reference: correctness와 small/medium grid용
2. block-online convolution: past block은 FFT, active block은 direct update
3. \(u^2\)-only feedback: volatility driver를 이동하지 않는 빠른 restricted
   proposal ablation
4. SOE lifted target simulator: \(O(mn)\) practical variant, 단 model bias를
   BLP refinement로 별도 인증

Main scientific claim에서 BLP target을 사용하려면 exact 또는 mathematically
equivalent online convolution이 필요하다. Approximate control correction을
BLP noise path에 섞고 BLP target에 unbiased하다고 주장하지 않는다.

Training은 lifted surrogate로 수행하고 frozen controller를 exact BLP에서
평가할 수 있지만, exact BLP evaluation 중에도 \(u^1\) feedback correction은
causal하게 계산되어야 한다.

---

## 7. Controller 설계

### 7.1 공통 출력

모든 main controller는 independent Brownian basis의 두 control을 출력한다.

$$
u_{\phi,\zeta}(t)
=
\begin{pmatrix}
u_{\phi,\zeta}^1(t)\\
u_{\phi,\zeta}^2(t)
\end{pmatrix}.
$$

Bounded output은

$$
u_{\phi,\zeta}
=
u_{\max}\tanh(f_\phi)
$$

로 구현한다.

Bound는 estimator bias를 만들지 않지만 attainable proposal class를 제한한다. 따라서 \(u_{\max}\) sensitivity를 수행하고 clipping loss를 보고한다.

### 7.2 비교 architecture

1. Constant two-driver control
2. Time-dependent open-loop spline control
3. Current-state Markov affine control
4. Current-state Markov MLP
5. Auxiliary SOE memory MLP
6. GRU/TCN causal memory comparator
7. Amortized SOE memory controller

Main method는 theory와 재현성이 좋은 SOE memory controller다. GRU/TCN은 flexible empirical comparator이며 main theorem의 대상으로 삼지 않는다.

### 7.3 입력 feature

State inputs:

- normalized time \(t/T\)
- log-moneyness \(\log(S_t/K)\)
- normalized variance 또는 \(\log(V_t/\xi_0(t))\)
- running minimum/barrier distance/drawdown state
- auxiliary memory states \(Z_t^{1:m}\)

Task inputs:

- maturity \(T\)
- strike/barrier in scale-invariant form
- event type embedding
- \(H,\eta,\rho\)
- forward variance curve coefficients
- rate/dividend summary if nonzero

원시 절대가격보다 dimensionless inputs를 사용한다.

### 7.4 causality 요구사항

- \(u_k\)는 \(t_k\)까지의 state와 increment만 사용한다.
- bidirectional RNN, full-path attention, future padding 정보는 금지한다.
- future Brownian increments를 바꿔도 과거 control이 바뀌지 않는 자동 테스트를 둔다.
- normalization 통계도 train set에서만 계산하고 future path summary를 사용하지 않는다.

### 7.5 amortization

Task distribution을

$$
\zeta\sim\Pi_{\mathrm{train}}
$$

에서 sampling하고

$$
u_{\phi}(t,\mathcal H_t;\zeta)
$$

를 학습한다.

서로 다른 maturity를 하나의 controller로 다루기 위해 unit interval로
정규화한다.

$$
s=\frac{t}{T},
\qquad
\widetilde B_s=\frac{B_{Ts}}{\sqrt T},
\qquad
\widetilde u_s=\sqrt T\,u_{Ts}.
$$

그러면

$$
\int_0^T\|u_t\|^2dt
=
\int_0^1\|\widetilde u_s\|^2ds.
$$

Network는 dimensionless \(\widetilde u\)를 출력하고 simulator adapter가
\(u_t=\widetilde u_{t/T}/\sqrt T\)로 변환하는 방식을 우선 검토한다. 이
normalization이 없는 상태에서 maturity별 control magnitude를 직접 비교하지
않는다.

Train/validation/test task를 분리한다.

- interpolation test: train 범위 내부의 unseen task
- local extrapolation test: 범위 경계 바로 밖
- out-of-domain stress test: claim 대상이 아닌 별도 진단

논문의 generalization claim은 interpolation에만 적용한다. Extrapolation은 탐색 결과로 표시한다.

Task sampling은 rarity bin, maturity, \(H\), event type별 stratified sampler를 사용해 쉬운 task가 gradient를 독점하지 않게 한다.

---

## 8. 학습 알고리즘

### 8.1 Stage A: classical warm start

- terminal event에는 CEM constant 또는 time-dependent spline 사용
- path-dependent event에는 adaptive-level CEM 사용
- target event fraction을 대략 1–10% 범위로 올림
- warm start는 train seed에서만 선택

### 8.2 Stage B: soft path-integral free-energy training

$$
\mathcal J_{\mathrm{PI}}(\phi)
=
\mathbb E_{\mathbb Q_\phi}
\left[
\Phi_{\tau,\zeta}(X^\phi)
+\frac12\int_0^T\|u_{\phi,t}\|^2dt
\right].
$$

가능한 gradient:

- smooth simulator reparameterization
- pathwise adjoint/backpropagation
- score/martingale estimator

Gradient estimator는 Gaussian analytic toy에서 finite difference와 비교한다. Hard indicator에는 이 objective를 직접 사용하지 않는다.

### 8.3 Stage C: PICE-style path-law matching

Current proposal \(\mathbb Q_{\bar\phi}\)에서 trajectory를 생성하고

$$
\omega_i
=
g_{\tau,\zeta}(X_i)L_{\bar\phi}(X_i)
$$

를 계산한다.

Target projection objective는

$$
\min_\phi
\mathrm{KL}
\left(
\mathbb Q_{\tau,\zeta}^{\star}
\Vert
\mathbb Q_\phi
\right).
$$

Score gradient는 current proposal sample로

$$
-\frac{
\sum_i\omega_i\nabla_\phi\log q_\phi(X_i)
}{
\sum_i\omega_i
}
$$

를 근사한다.

필수 규칙:

- \(X_i,\omega_i,\nabla\log q_\phi(X_i)\)는 같은 trajectory에서 계산
- behavior proposal과 candidate proposal을 metadata로 구분
- Stale off-policy data는 원래 behavior control과
  \(L_{\bar\phi}=d\mathbb M/d\mathbb Q_{\bar\phi}\)를 정확히 복원할 수 있을
  때만 사용한다. PICE target weight는 \(gL_{\bar\phi}\)이며 candidate
  likelihood를 다시 곱하지 않는다. Behavior density를 모르면 해당 sample은
  폐기한다.
- Behavior path에서 target-coordinate increment를
  \(\Delta B_k^{\mathbb M}=\Delta B_k^{\mathbb Q_{\bar\phi}}+
  u_{\bar\phi,k}h\)로 먼저 reconstruct한다. Candidate score는
  \(\Delta B_k^{\mathbb Q_\phi}=\Delta B_k^{\mathbb M}-u_{\phi,k}h\)를
  사용해야 한다. Candidate가 behavior와 다를 때 behavior Brownian
  increment를 그대로 candidate score에 넣으면 biased gradient가 된다.
- normalized weight는 training에만 사용
- Self-normalized PICE gradient는 finite sample에서 biased지만 consistent한
  optimization estimator다. ESS와 batch-size sensitivity를 보고하며, 이
  bias를 최종 확률 estimator에 전달하지 않는다.
- weight ESS가 최소치 아래면 temperature/level을 완화

### 8.4 Stage D: exact second-moment refinement

Hard final event에 대해

$$
\mathcal J_2(\phi)
=
\mathbb E_{\mathbb Q_\phi}
\left[
1_{A_\zeta}L_\phi^2
\right]
$$

또는 soft target에 대해

$$
\mathbb E_{\mathbb Q_\phi}
\left[
g_{\tau,\zeta}^2L_\phi^2
\right]
$$

를 최소화한다.

Hard indicator는 현재 검증된 score-function gradient를 2차원 Brownian basis로 확장한다.

Controller selection은 validation log second moment 및 measured work를 기준으로 한다. PI training loss가 더 낮아도 validation variance가 나쁘면 선택하지 않는다.

### 8.5 권장 curriculum

1. CEM warm start
2. 큰 \(\tau\)의 soft PI objective
3. adaptive ESS를 유지하며 \(\tau\) 감소
4. PICE projection
5. soft \(J_2\) refinement
6. hard \(J_2\) refinement
7. frozen validation
8. one-time independent audit
9. method freeze 후 sealed final evaluation

### 8.6 안정화

- log weights와 log-sum-exp
- float64 likelihood/action
- gradient clipping
- control energy penalty와 hard bound 분리
- per-task weight normalization
- contribution ESS threshold
- top-0.1%, top-1% contribution concentration
- adaptive batch size
- independent Brownian antithetic pairing은 likelihood와 함께 검증 후 사용

---

## 9. 이론 연구 프로그램

### 9.1 결과의 등급

| 결과 | 성격 | 신규성 주장 |
|---|---|---|
| P0: measure convention과 estimator | correctness | 없음 |
| P1: payoff-tilted law와 relative variance identity | foundation | 제한적 |
| P2: soft Föllmer/path-integral representation | foundation | 기존 정리의 적용 |
| P3: controlled BLP discrete reconstruction | 구현 정리 | 금융/수치 특화 |
| T1: control error에서 Rényi-2/variance bound | 핵심 후보 | 가능 |
| T2: memory approximation에서 control/variance bound | 핵심 stretch | 높음 |
| T3: amortized uniform task bound | stretch | 높음 |
| P4: soft-to-hard transfer 진단 | 보조 | 가능 |

### 9.2 Proposition P1: variance-divergence identity

Nonnegative \(G\)와

$$
\frac{d\mathbb Q^\star}{d\mathbb M}
=
\frac{G}{\mathbb E_{\mathbb M}[G]}
$$

에 대해

$$
\mathrm{CV}^2_{\mathbb Q}
\left(
G\frac{d\mathbb M}{d\mathbb Q}
\right)
=
\chi^2(\mathbb Q^\star\Vert\mathbb Q).
$$

이 결과는 main estimator, training objective, diagnostics의 공통 언어가 된다.

### 9.3 Proposition P2: soft tilted law의 Föllmer drift

\(g_{\tau,\zeta}>0\), \(E[g_{\tau,\zeta}]<\infty\)이고 density martingale이 적절한 integrability를 만족할 때

$$
Z_t
=
\frac{
E_{\mathbb M}[g_{\tau,\zeta}\mid\mathcal F_t]
}{
E_{\mathbb M}[g_{\tau,\zeta}]
}
$$

는

$$
Z_t
=
\mathcal E
\left(
\int_0^t u_s^{\star\top}dB_s^{\mathbb M}
\right)
$$

형태의 martingale representation을 가진다.

Under \(\mathbb Q^\star\),

$$
B_t^{\mathbb Q^\star}
=
B_t^{\mathbb M}
-\int_0^tu_s^\star ds
$$

가 Brownian motion이 된다.

이 결과는 standard foundation이며 그 자체를 main novelty로 주장하지 않는다.

### 9.4 Proposition P3: fixed-control discrete BLP correctness

Frozen left-adapted two-driver control에 대해 다음을 보인다.

1. controlled BLP state가 reconstructed target Brownian increments를 사용한 natural BLP state와 pathwise 일치
2. recent-cell singular integral mean correction이 exact
3. Wick variance correction은 noise covariance가 변하지 않으므로 동일
4. likelihood는 두 Brownian increments에 대한 Gaussian mean-shift density
5. frozen estimator는 chosen BLP discretization expectation에 unbiased

Continuous rBergomi expectation에 대한 unbiasedness는 주장하지 않는다.

### 9.5 Theorem T1 후보: control error와 relative variance

Soft target의 optimal law가 drift \(u^\star\)로 표현되고 proposal drift가 \(u\)라고 하자. Drift difference를

$$
\delta_t=u_t^\star-u_t
$$

라 한다. 두 drift는 같은 augmented canonical history에서 평가하며, state-law
상의 임의 coupling error와 혼동하지 않는다.

Assumption:

- \(\mathcal E(2\int\delta dB)\)가 true martingale
- pathwise 또는 exponential-moment 의미에서 \(\int_0^T\|\delta_t\|^2dt\)가 제어됨

강한 pathwise 조건

$$
\int_0^T\|\delta_t\|^2dt\le\varepsilon^2
$$

아래 목표 bound는

$$
1+\chi^2(\mathbb Q^\star\Vert\mathbb Q_u)
\le
\exp(\varepsilon^2),
$$

따라서

$$
\mathrm{CV}^2\le e^{\varepsilon^2}-1.
$$

Proof는 density ratio의 제곱을

$$
\mathcal E\left(2\int\delta dB\right)
\exp\left(\int\|\delta\|^2dt\right)
$$

로 분해하는 방식으로 검토한다.

주의:

- hard conditional law에는 bounded \(u^\star\)가 없을 수 있다.
- discrete left-point proposal class에서는 \(\mathbb Q^\star=\mathbb Q_{u^\star}\)가 성립하지 않을 수 있다.
- 따라서 theorem의 첫 버전은 continuous soft target 또는 representable reference class에 한정한다.
- fixed-grid 실험에는 proposal-class approximation term을 별도로 둔다.

### 9.6 Theorem T2 후보: memory approximation

Optimal history-dependent drift를 \(u^\star(t,\mathcal H_t;\zeta)\), \(m\)-factor memory representation을 사용한 최적 근사를 \(u_m^\star\)라 하자.

목표 decomposition:

$$
\|u^\star-u_{\phi,m}\|_{L^2}
\le
\varepsilon_{\mathrm{memory}}(m)
+\varepsilon_{\mathrm{network}}(\phi,m)
+\varepsilon_{\mathrm{task}}(\phi).
$$

T1과 결합해

$$
\log(1+\mathrm{CV}^2)
\lesssim
C
\left[
\varepsilon_{\mathrm{memory}}
+\varepsilon_{\mathrm{network}}
+\varepsilon_{\mathrm{task}}
\right]^2
$$

형태를 목표로 한다.

가장 어려운 부분은 kernel approximation error가 Föllmer drift error로 전달되는 stability다. 다음 fallback hierarchy를 둔다.

1. uniform pathwise theorem
2. exponential-moment theorem
3. integrated task-distribution bound
4. finite-grid proposition
5. empirical scaling law만 보고하고 상위 이론저널 claim 제거

### 9.7 Proposition P4: soft-to-hard discrepancy

Hard conditional law \(\mathbb Q_A^\star\)와 soft law \(\mathbb Q_\tau^\star\) 사이에는

$$
\frac{d\mathbb Q_A^\star}{d\mathbb Q_\tau^\star}
=
\frac{Z_\tau}{p}
\frac{1_A}{g_\tau}.
$$

따라서

$$
1+\chi^2(\mathbb Q_A^\star\Vert\mathbb Q_\tau^\star)
=
\frac{Z_\tau}{p^2}
E_{\mathbb M}
\left[
\frac{1_A}{g_\tau}
\right].
$$

이 identity를 이용해 \(\tau\) 선택과 soft-to-hard mismatch를 진단할 수 있다.

이 식만으로 learned proposal과 hard target 사이의 variance를 자동으로 bound하지는 않는다. Learned-proposal density ratio 조건 없이 divergence triangle inequality를 가정하지 않는다.

### 9.8 Theorem T3 후보: amortized uniformity

Compact task set \(\mathcal Z\)에서 \(u^\star(\cdot;\zeta)\)가 task에 대해 regular하고 conditional network가

$$
\sup_{\zeta\in\mathcal Z}
\int_0^T
\|u^\star_t(\zeta)-u_{\phi,t}(\zeta)\|^2dt
\le\varepsilon^2
$$

를 만족하면 uniform relative-variance bound를 도출한다.

이 결과는 강한 assumption을 필요로 하므로 empirical interpolation evidence와 명확히 분리한다.

### 9.9 이론 완료 조건

- 모든 theorem이 구현된 Brownian basis와 likelihood sign을 사용
- soft/hard 범위를 명시
- continuous/discrete 결과를 분리
- control class representability를 명시
- indicator/barrier regularity를 별도 처리
- theorem assumption을 numerical experiment에서 확인 가능한 proxy와 연결
- 기존 정리와 새로운 corollary/theorem을 문장 단위로 구분

---

## 10. 구현 구조

권장 신규 구조:

    src/
      path_integral/
        measures.py
        potentials.py
        action.py
        tilted_weights.py
        pice.py
        free_energy.py
        renyi_objective.py
      controls/
        base.py
        heston_oracle.py
        markov.py
        soe_memory.py
        recurrent.py
        amortized.py
      models/
        heston_controlled.py
        rbergomi_blp_controlled.py
        rbergomi_memory.py
      evaluation/
        likelihood.py
        divergence.py
        concentration.py
        work.py
        break_even.py
      theory_checks/
        gaussian_oracles.py
        heston_oracle_checks.py
        path_reconstruction.py
    experiments/
      pi0_gaussian/
      pi1_heston_oracle/
      pi2_training_objectives/
      pi3_rbergomi_controlled/
      pi4_memory/
      pi5_amortization/
      pi6_application/

기존 파일을 한 번에 대규모 이동하지 않는다. 신규 모듈을 추가하고 compatibility adapter를 둔 뒤, 테스트 통과 후 단계적으로 이동한다.

### 10.1 핵심 interface

Controlled simulator는 최소한 다음을 반환해야 한다.

- state paths 또는 streaming terminal/path statistics
- log likelihood \(d\mathbb M/d\mathbb Q\)
- proposal Brownian increments
- applied controls
- control energy
- event/payoff
- path action
- optional memory states

Control 호출 시점은 increment sampling 이전이어야 한다.

### 10.2 precision

- state simulation: float32 허용, reference run은 float64
- Brownian reconstruction: float64 test
- log likelihood/action: float64
- log-sum-exp: float64
- reported contribution: float64
- Fourier/CDF oracle: float64

Mixed precision은 likelihood agreement test 후에만 허용한다.

### 10.3 성능

- Uncontrolled BLP noise convolution과 feedback control convolution을 분리
- Feedback \(u^1\) correction은 one-shot FFT가 불가능하므로 exact
  \(O(n^2)\) reference와 block-online convolution을 별도 구현·검증
- Block-online implementation은 exact reference와 pathwise agreement 후 사용
- auxiliary SOE state update는 \(O(m)\) per step
- frozen controller inference 함수 제공
- training graph와 evaluation graph 분리
- wall-clock은 warm-up 후 repeated median과 distribution 보고
- GPU kernel launch overhead가 작은 affine/CEM 비교를 왜곡하지 않도록 batch size sweep

---

## 11. 테스트 계획

### 11.1 analytic unit tests

1. Gaussian exponential tilt \(g(W_T)=e^{aW_T}\)에서 optimal constant drift \(u^\star=a\)
2. Gaussian left-tail conditional \(h\)의 analytic drift sign
3. one-step relative variance와 chi-square identity
4. two-dimensional Gaussian likelihood sign과 normalization
5. PI action weight \(\exp(-\mathcal S)=gL\)
6. PICE weighted score가 constant Gaussian tilt를 회복
7. free-energy objective finite difference gradient
8. second-moment score gradient finite difference

### 11.2 Heston tests

- two-driver controlled path reconstruction
- \(u^2=0\)에서 기존 one-driver simulator와 agreement
- oracle \(h\)와 Fourier/CDF consistency
- \(\partial_x h,\partial_v h\) finite-difference convergence
- oracle drift near maturity diagnostics
- fixed-control unbiasedness across \(dt\)
- continuous reference와 discrete bias 분리

### 11.3 rBergomi pathwise tests

1. 동일 \(\mathbb Q\) Gaussian innovations와 frozen open-loop control 생성
2. \(\Delta B^{\mathbb M}=\Delta B^{\mathbb Q}+uh\) reconstruction
3. local \(I_i\) shift와 earlier-cell shift 적용
4. reconstructed natural BLP \(Y,V,S\)와 controlled BLP pathwise agreement
5. log likelihood를 joint Gaussian density ratio와 비교
6. zero control에서 기존 simulator와 bitwise 또는 tolerance agreement
7. random feedback correction이 causal하며 finite
8. future increment perturbation이 과거 control에 영향 없음
9. lift rank 변경이 zero-control target path에 영향 없음
10. \(E[V_{t_k}]=\xi_0(t_k)\) under target measure

### 11.4 statistical tests

- \(E_{\mathbb Q}[L]=1\)
- \(E_{\mathbb Q}[gL]=E_{\mathbb M}[g]\)
- hard event estimate unbiased for chosen discretization
- contribution ESS and tail share
- paired seed variance comparison
- time-step and lift-rank convergence
- final CI coverage

통계 테스트는 deterministic unit test처럼 매 commit 엄격히 돌리지 않는다. Release/nightly suite에서 fixed tolerance와 sample budget을 사용한다.

### 11.5 failure tests

- future-looking controller 거부
- NaN/Inf control 거부
- mismatched Brownian basis metadata 거부
- stale behavior likelihood 없이 off-policy data 사용 거부
- hard event를 ordinary pathwise gradient에 전달하면 오류
- jump control에 compound likelihood가 없으면 오류
- physical/risk-neutral measure metadata 혼용 거부
- final evaluation seed 재사용 거부

---

## 12. 실험 설계

### 12.1 Research questions

| ID | 질문 |
|---|---|
| RQ1 | Path-integral training이 analytic Gaussian/Heston oracle를 회복하는가? |
| RQ2 | PI initialization/PICE가 direct \(J_2\) 학습보다 안정적인가? |
| RQ3 | rough memory가 current-state controller보다 유효한가? |
| RQ4 | full two-driver control이 restricted one-driver보다 유효한가? |
| RQ5 | amortized controller가 held-out task에서 per-task 성능을 유지하는가? |
| RQ6 | training을 포함한 총 계산비용에서 break-even이 존재하는가? |
| RQ7 | control error proxy와 observed \(D_2\)/relative variance가 이론과 일치하는가? |

### 12.2 모델

| 모델 | 역할 |
|---|---|
| Brownian/Gaussian toy | exact theory |
| Black–Scholes terminal/barrier | analytic or high-accuracy sanity |
| Heston continuous reference | oracle/CDF benchmark |
| Heston Euler | implementation-aligned benchmark |
| rBergomi exact covariance small grid | BLP reference |
| rBergomi BLP | main target simulator |
| rBergomi lifted simulator | model-bias ablation |

### 12.3 사전 고정 rare-event regimes

Terminal probabilities:

$$
p\in\{10^{-4},10^{-5},10^{-6}\}.
$$

\(10^{-8}\)은 reference와 ESS가 안정된 경우의 stretch regime다.

Path-dependent regimes:

- down-crossing before \(T\)
- terminal left-tail and no-recovery condition
- running drawdown threshold

Event를 결과가 잘 나오는 방향으로 사후 변경하지 않는다. Pilot 후 세 regime를 preregister하고 sealed benchmark에 사용한다.

### 12.4 baseline

1. Crude Monte Carlo
2. Conditional Monte Carlo where available
3. Constant CEM
4. Time-dependent deterministic CEM
5. Large-deviation deterministic tilt
6. FNN/Cameron–Martin open-loop sampler
7. Current-state affine
8. Current-state MLP
9. Per-task PICE feedback
10. Per-task PI + \(J_2\)
11. Amortized Markov PI + \(J_2\)
12. Amortized memory PI + \(J_2\)
13. Heston oracle where available

Koopman/FBSDE baseline은 동일 문제에 안정적인 구현이 가능한 toy/Heston 범위에서 포함하고, rBergomi에서 억지로 부정확한 구현을 만들지 않는다.

### 12.5 metric

Correctness:

- estimate, bias, bias z-score
- confidence interval coverage
- \(E[L]\) normalization
- time-step/reference error

Statistical efficiency:

- single-path second moment
- relative variance
- log relative second moment
- contribution ESS
- top-0.1%, top-1% contribution share

Computational efficiency:

- cost per path
- online work-normalized VRF
- training time
- total time to target relative error
- multi-query break-even

Learning:

- train/validation PI loss
- reverse-KL proxy
- \(J_2\)
- control energy
- oracle control error
- task generalization gap

### 12.6 work-normalized comparison

Per-path contribution variance \(\sigma_a^2\), cost \(c_a\)인 method \(a\)에 대해

$$
\mathrm{Work}_a
=
\sigma_a^2c_a.
$$

Online VRF:

$$
\mathrm{VRF}_{work}
=
\frac{\sigma_b^2c_b}{\sigma_a^2c_a}.
$$

목표 relative standard error \(\epsilon\)에 필요한 경로 수는

$$
N_a(\epsilon)
\approx
\frac{\sigma_a^2}{\epsilon^2\mu^2}.
$$

\(M\)개 query의 total time은

$$
T_a(M,\epsilon)
=
T_{train,a}
+\sum_{j=1}^M
N_{a,j}(\epsilon)c_{a,j}.
$$

Break-even query count는

$$
M^\star
=
\min\{M:T_{\mathrm{amortized}}(M)<T_{\mathrm{baseline}}(M)\}.
$$

### 12.7 statistical protocol

- pilot/tuning seed와 final seed 분리
- 최소 20 independent final seed
- method 간 common random numbers를 사용한 paired comparison
- confidence interval은 seed-level paired log-work ratio에 적용
- multiple-comparison correction 또는 primary endpoint 사전 지정
- final controller와 config hash 저장
- final seed는 method freeze 전 접근 금지
- timing은 최소 10회 repeated run, warm-up 제외

### 12.8 초기 gate threshold

Pilot 완료 후 sealed protocol 전에 수치를 확정한다. 초기 제안:

- correctness: absolute reported bias z \(\le 3\)
- memory claim: 사전 지정 3 regime 중 최소 2개에서 paired 95% CI 기준 개선
- work claim: best non-memory baseline 대비 work VRF lower CI \(>1.10\)
- amortization: held-out task의 median log-efficiency가 per-task method의 80% 이상
- break-even: main task distribution에서 \(M^\star\le25\)
- concentration: contribution ESS와 top-share가 사전 안전기준 충족

Threshold를 audit 결과를 본 뒤 변경하면 해당 run은 exploratory로 강등한다.

---

## 13. 실무형 application

### 13.1 main use case

Risk-neutral calibrated rBergomi에서 반복되는 tail-sensitive pricing query:

- deep-OTM digital
- barrier hit probability
- down-and-in/down-and-out component
- drawdown-linked payoff

실무 엔진 입력:

- calibrated model snapshot
- \(S_0,r,q,\xi_0(\cdot),H,\eta,\rho\)
- event/payoff definition
- \(K,B,T\)
- target relative error 또는 compute budget

출력:

- price/probability estimate
- standard error와 confidence interval
- likelihood/contribution ESS
- concentration warning
- online and end-to-end cost
- baseline 대비 gain
- calibration snapshot hash

### 13.2 calibration

- main paper에서는 risk-neutral option surface calibration만 사용
- calibration loss와 rare-event MC error를 분리
- calibration parameter uncertainty는 estimator CI에 자동 포함되지 않음
- 여러 calibrated snapshot에 대한 robustness를 별도 보고
- \(\xi_0(t)\)는 positive spline/PCA coefficient로 저장
- data license가 공개를 금지하면 synthetic-reproducible snapshot을 함께 제공

### 13.3 product acceptance

실무적으로 쓸 수 있다고 주장하려면:

- batch query API
- deterministic seed/reproducible output
- target error stopping rule
- weight degeneracy warning
- fallback to baseline
- CPU/GPU benchmark
- model/config/checkpoint versioning
- failure-safe NaN/overflow handling

실시간 trading prediction이나 시장 crash prediction은 scope가 아니다.

---

## 14. 단계별 실행 로드맵

### Phase PI0 — 연구 명세와 오류 동결 (1–2주)

상태 (2026-07-13): **G-PI0 통과**. 구현 명세, objective 구분, 오류 금지선,
코드 대응표와 related-work boundary를
`docs/path_integral_mathematical_specification.md`에 동결했다.

작업:

- v2 measure/path-integral specification 작성
- related-work novelty matrix
- soft/hard target 정의
- independent Brownian basis 표준화
- theorem dependency graph
- experiment primary endpoints 초안

완료 조건 G-PI0:

- path-integral term이 코드 변수와 수식에 일대일 대응
- continuous/discrete claim 분리
- 신규성 문장이 선행연구와 충돌하지 않음

### Phase PI1 — Gaussian path-integral oracle (2–3주)

상태 (2026-07-13): **G-PI1 통과**. Gaussian exponential tilt, tail Doob drift,
PI action, constant PICE, 상대분산–Rényi-2 identity를 구현했고 전체 119개
test와 Ruff/Mypy를 통과했다. 이 gate는 rough-volatility 성능 주장의 근거가
아니며 PI2 진입 전 부호·측도·목적함수 검증 gate다.

작업:

- exponential tilt analytic toy
- Gaussian tail Doob drift
- PI action/weight
- PICE update
- \(J_2\) identity
- gradient tests

완료 조건 G-PI1:

- control sign, likelihood, action 모두 analytic result와 일치
- relative variance와 chi-square identity 재현
- PI/PICE/\(J_2\) objective의 차이 문서화

### Phase PI2 — Heston full two-driver oracle (4–7주)

상태 (2026-07-13): **진행 중**. independent-basis two-driver simulator,
float64 2D likelihood/energy, optional proposal/target Brownian 기록, one-driver
compatibility, pathwise reconstruction, soft conditional \(h\), analytic
Fourier/Richardson gradient와 2D oracle control까지 완료했다. Near-maturity
control clipping policy와 objective benchmark는 남아 있으므로 G-PI2는 아직
통과하지 않았다.

작업:

- two-driver Heston simulator
- soft conditional \(h\)
- oracle gradient
- CEM, affine, MLP, PI, PICE, \(J_2\) 비교
- time-step refinement

완료 조건 G-PI2:

- two-driver reconstruction/unbiasedness pass
- oracle direction recovery
- Heston negative/positive 결과를 tuning 없이 확정

### Phase PI3 — Controlled rBergomi BLP (7–11주)

작업:

- two-driver controlled BLP
- local singular-cell control correction
- same-operator memory correction
- pathwise reconstruction tests
- hard/soft fixed-control unbiasedness

완료 조건 G-PI3:

- reconstruction tolerance pass
- \(E[L]=1\)
- target estimate agreement
- all paths finite across preregistered parameter range

### Phase PI4 — Rough-memory path-integral control (11–16주)

작업:

- SOE auxiliary lift
- Markov vs memory controllers
- per-task PI/PICE/\(J_2\)
- barrier/drawdown task
- lift-rank ablation

완료 조건 G-PI4:

- 사전 지정 두 regime 이상에서 memory improvement
- future leakage test pass
- work-normalized result가 timing noise보다 큼

실패 시:

- memory representation 1회 재설계
- preregistered alternate path-dependent task 실행
- 그래도 실패하면 memory superiority claim 폐기

### Phase PI5 — Amortization (16–20주)

작업:

- conditional controller
- stratified task sampler
- held-out task split
- per-task fine-tuning comparator
- break-even curve

완료 조건 G-PI5:

- interpolation performance 유지
- \(M^\star\)가 실용 범위
- task leakage 없음

실패 시 per-task rough PI paper로 scope 축소한다.

### Phase PI6 — 이론 완성 (8–23주, 병행)

작업:

- P1–P4 formal proof
- T1 control-error bound
- T2 memory theorem 또는 fallback
- theorem-to-experiment mapping
- 외부 수학 검토

완료 조건 G-PI6:

- 적어도 하나의 비자명한 quantitative theorem
- standard result와 novel result 구분
- implementation convention과 proof 일치

### Phase PI7 — Sealed benchmark (21–26주)

작업:

- protocol hash
- final configs
- 20+ seeds
- full rarity/task/model matrix
- ablation
- CPU/GPU timing

완료 조건 G-PI7:

- primary endpoints pass
- negative result 포함
- 모든 main table 자동 재생성

### Phase PI8 — Calibrated practical application (25–29주)

작업:

- option-surface calibration snapshots
- multi-query batch
- target-error runtime
- failure/fallback demonstration

완료 조건 G-PI8:

- 실제 반복 query에서 break-even
- calibration와 MC uncertainty 분리
- 공개 가능한 reproduction snapshot

### Phase PI9 — 원고와 clean-room reproduction (29–34주)

작업:

- 논문 작성
- proof appendix
- artifact documentation
- independent clean environment
- journal-specific formatting

완료 조건 G-PI9:

- main tables/figures one-command regeneration
- no unsupported first/superiority claim
- target journal scope 적합성 재검토

---

## 15. 34주 주차별 산출물

| 주차 | 핵심 산출물 |
|---:|---|
| 1 | v2 math specification, novelty matrix |
| 2 | Gaussian tilt and action tests |
| 3 | PICE/\(J_2\) toy comparison |
| 4 | Heston independent basis |
| 5 | soft Heston desirability |
| 6 | oracle drift and gradients |
| 7 | Heston objective benchmark |
| 8 | controlled BLP design |
| 9 | local-cell correction implementation |
| 10 | pathwise reconstruction |
| 11 | rBergomi likelihood report |
| 12 | SOE memory state |
| 13 | memory controller prototype |
| 14 | terminal rough benchmark |
| 15 | barrier/drawdown benchmark |
| 16 | G-PI4 decision |
| 17 | task-conditioned controller |
| 18 | held-out task protocol |
| 19 | amortized benchmark |
| 20 | break-even and G-PI5 |
| 21 | T1 proof freeze |
| 22 | memory theorem/fallback |
| 23 | theory internal review |
| 24 | final protocol freeze |
| 25 | final compute batch 1 |
| 26 | final compute batch 2 |
| 27 | calibration snapshot |
| 28 | practical batch engine |
| 29 | application report |
| 30 | manuscript methods/theory |
| 31 | manuscript experiments |
| 32 | appendix/artifacts |
| 33 | clean-room reproduction |
| 34 | journal submission package |

---

## 16. 논문 구성

### Abstract

- rare-event Monte Carlo와 rough memory 문제
- path-integral optimal path law
- causal amortized controller
- quantitative theorem
- work-normalized main result

### 1. Introduction

- 실무상 반복 tail query
- rough volatility가 만드는 비마르코프 난점
- generic neural IS/PICE와의 차이
- 검증된 contribution만 3–4개

### 2. Related Work

- path-integral/linearly-solvable control
- adaptive importance sampling/PICE
- neural IS and FBSDE/Koopman
- rare-event large deviations
- rough-volatility simulation and lifts
- amortized stochastic control

### 3. Path-Space Formulation

- target/proposal measures
- tilted law
- PI free energy
- relative variance/Rényi identity
- soft/hard distinction

### 4. Controlled Rough-Volatility Dynamics

- Heston basis
- rBergomi BLP correction
- exact discrete likelihood
- auxiliary memory state

### 5. Neural Path-Integral Algorithm

- PI action
- PICE projection
- \(J_2\) refinement
- amortized controller
- curriculum

### 6. Theory

- Föllmer representation
- BLP correctness
- variance bound
- memory/amortized result

### 7. Verification

- Gaussian oracle
- Heston oracle
- convergence and likelihood tests

### 8. Rough-Volatility Experiments

- terminal/path-dependent tasks
- baselines
- memory ablation
- task generalization

### 9. Practical Application

- calibration snapshot
- batch queries
- break-even
- failure handling

### 10. Limitations

- soft training vs hard target
- proposal-class restriction
- BLP discretization
- calibration risk
- extrapolation limits

---

## 17. 필수 표와 그림

Tables:

1. Related-work novelty matrix
2. Measure/sign convention
3. Gaussian/Heston oracle validation
4. Controlled BLP correctness
5. Rough terminal benchmark
6. Barrier/drawdown benchmark
7. Memory ablation
8. Amortization held-out tasks
9. Online/end-to-end efficiency
10. Calibration application

Figures:

1. Base path law → tilted law → neural proposal diagram
2. Path action과 training objectives 관계
3. Heston oracle vs learned control field
4. rBergomi memory/control flow
5. lift rank vs log relative second moment
6. rarity vs work-normalized VRF
7. per-task vs amortized efficiency
8. break-even query curve
9. contribution concentration
10. control-error proxy vs \(D_2\)

---

## 18. 이론적·기술적 오류 검토 결과

아래 항목은 v1 또는 초기 아이디어에서 발생할 수 있는 오류를 v2에서 수정한 것이다.

| 위험/오류 | 올바른 처리 |
|---|---|
| Volterra control correction을 deterministic이라 표현 | kernel만 deterministic; feedback correction은 random causal finite variation |
| hard conditional law를 bounded drift로 정확히 생성 | soft positive law로 학습하고 hard estimator는 별도 평가 |
| PI KL 감소를 variance 감소와 동일시 | forward KL, reverse KL, chi-square를 분리 |
| finite grid mean-shift가 zero-variance law 표현 | 일반적으로 불가; proposal-class approximation 명시 |
| BLP recent-cell bridge mode까지 left-point drift가 자유롭게 제어한다고 가정 | bridge component는 proposal-class restriction으로 남김 |
| continuous oracle가 discrete simulator의 exact oracle | refinement가 필요한 approximation |
| auxiliary lift 사용 시 target bias가 자동 발생 | feature-only lift면 target bias 없음; lifted simulator일 때만 model bias |
| correlated Brownian 좌표에 독립 likelihood 사용 | independent basis에서 control; correlated basis면 covariance inverse 필요 |
| state path law와 Brownian augmented path law의 divergence를 동일시 | likelihood가 정의되는 augmented path space를 기본으로 사용 |
| noise-control matching 없이 linearly-solvable HJB를 사용 | Brownian shift와 quadratic energy 범위에서만 desirability linearization |
| BLP local integral에 별도 likelihood 추가 | local integral shift는 same Brownian shift의 결과, 중복 likelihood 금지 |
| continuous kernel correction과 discrete BLP 혼합 | 동일 BLP operator로 correction 계산 |
| feedback control convolution을 일반 FFT로 일괄 계산 가능하다고 가정 | \(u_k\)는 sequential하게 생성됨; exact online/block convolution 필요 |
| off-policy PICE에서 behavior Brownian을 candidate score에 그대로 사용 | target-coordinate path를 reconstruct한 뒤 candidate residual Brownian 계산 |
| stale PICE sample에 candidate likelihood를 중복 곱함 | target weight는 \(g\,d\mathbb M/d\mathbb Q_{behavior}\); candidate는 score에만 등장 |
| self-normalized weight로 최종 확률 보고 | training에만 사용; final은 unnormalized unbiased estimator |
| free-energy의 log sample mean을 unbiased estimate로 해석 | log 때문에 finite-sample bias; training/diagnostic 전용 |
| clipping이 estimator bias를 만든다고 해석 | exact likelihood면 estimator는 unbiased, proposal efficiency만 제한 |
| Girsanov가 time discretization bias를 제거 | measure-change와 discretization bias 분리 |
| Heston one-driver control을 full optimal control이라 표현 | full two-driver oracle와 restricted baseline 분리 |
| path-dependent encoder가 미래를 사용 | explicit causality test와 architecture restriction |
| risk-neutral calibration로 physical crash probability 주장 | 본 논문은 risk-neutral; physical extension 분리 |
| rBergomi stochastic exponential을 항상 true martingale로 가정 | main \(\rho<0\) 범위와 martingale condition 명시, discounted-spot diagnostic |
| calibration error를 MC confidence interval에 포함 | 별도 uncertainty layer로 보고 |
| barrier grid estimate를 continuous barrier probability로 표현 | bridge correction 없으면 discretely monitored라고 명시 |
| normalizing likelihood ESS만 보고 안정성 주장 | event contribution ESS와 concentration을 함께 사용 |

현재 검토에서 발견된 치명적인 수식 모순은 없다. 다만 T1–T3는 아직 증명된 결과가 아니라 증명 목표이며, assumption이 충족되지 않으면 theorem claim을 낮춰야 한다.

---

## 19. Risk register와 kill criteria

| 위험 | 조기 신호 | 대응 | 중단/피벗 |
|---|---|---|---|
| PI training이 CEM보다 불안정 | low ESS, exploding action | temperature/ASPIC-style smoothing | Heston oracle에서도 실패하면 algorithm novelty 중단 |
| PI loss는 감소하지만 \(J_2\) 악화 | divergence mismatch | PICE+\(J_2\) refinement | PI는 initializer로만 축소 |
| two-driver 비용이 gain 상쇄 | work VRF \(\le1\) | frozen compact policy | one-driver restriction을 실무 variant로 |
| exact controlled BLP가 너무 느림 | causal \(O(n^2)\) cost 지배 | block-online convolution, \(u^2\)-only ablation | practical target을 bias-certified SOE model로 분리 |
| rough memory 효과 없음 | Markov와 동률 | path-dependent task, representation 1회 변경 | memory superiority claim 폐기 |
| amortization gap 큼 | held-out task collapse | local fine-tune, mixture | per-task paper로 축소 |
| break-even이 너무 늦음 | \(M^\star>25\) | inference 최적화 | practical amortization claim 제거 |
| T2 proof 실패 | drift stability 확보 불가 | discrete/integrated fallback | 상위 이론저널 하향 |
| rBergomi reference 불충분 | refinement 불안정 | exact covariance small grid, independent code | final benchmark 연기 |
| real data 공개 불가 | license restriction | synthetic snapshot 공개 | empirical claim 축소 |

최상위 저널 no-go 조건:

- standard Girsanov/Boué–Dupuis 재서술 외에 quantitative theorem이 없음
- memory controller가 strong baseline을 일관되게 이기지 못함
- work-normalized 이점이 timing noise 수준
- amortization이 실질적인 query cost를 줄이지 못함
- continuous/discrete bias가 분리되지 않음

---

## 20. Definition of Done

### 수학

- [x] path-integral variational formulation 완전 명시
- [x] soft/hard law의 절대연속성 차이 명시
- [x] relative variance–chi-square identity proof
- [x] Heston two-driver oracle derivation
- [ ] controlled BLP discrete correction proof
- [ ] 적어도 하나의 quantitative variance theorem
- [x] continuous/discrete theorem 구분

### 구현

- [x] two-driver Heston
- [ ] two-driver controlled rBergomi BLP
- [x] path action
- [ ] PICE training
- [ ] PI free-energy training
- [ ] 2D hard \(J_2\) refinement
- [ ] SOE memory controller
- [ ] amortized controller
- [ ] practical batch API

### 검증

- [x] Gaussian oracle
- [x] Heston oracle
- [ ] BLP pathwise reconstruction
- [ ] likelihood normalization
- [ ] causality
- [ ] time-step convergence
- [ ] memory-rank ablation
- [ ] 20+ final seeds
- [ ] clean-room reproduction

### 논문 주장

- [x] novelty matrix
- [ ] no unsupported first claim
- [ ] strongest baseline
- [ ] negative result
- [ ] online/end-to-end 분리
- [ ] calibration/MC uncertainty 분리
- [ ] code/config/checkpoint hashes

---

## 21. 즉시 실행할 첫 6주

### Week 1

- [x] docs/path_integral_mathematical_specification.md 작성
- [x] independent Brownian basis naming 확정
- [x] soft potential interface
- [x] action/weight convention table
- [x] related-work matrix

### Week 2

- [x] Gaussian exponential tilt
- [x] Gaussian tail oracle
- [x] relative variance identity tests
- [x] PI action tests
- [x] PICE constant-control recovery

### Week 3

- [x] Heston two-driver simulator API
- [x] 2D likelihood
- [x] one-driver backward compatibility
- [x] path reconstruction tests

### Week 4

- [x] Heston conditional soft \(h\)
- [x] \(\partial_x,\partial_v\) validation
- [x] oracle control
- [ ] near-maturity clipping diagnostics

### Week 5

- [ ] PI free-energy trainer
- [ ] PICE feedback trainer
- [ ] objective comparison protocol
- [ ] validation selection

### Week 6

- [ ] Heston CEM/PI/PICE/\(J_2\)/oracle benchmark
- [ ] G-PI2 interim review
- [ ] controlled BLP implementation specification freeze
- [ ] theorem T1 proof attempt 시작

---

## 22. 우선 참고문헌과 모니터링 대상

Foundations:

1. Boué and Dupuis, A variational representation for certain functionals of Brownian motion.
2. Kappen and Ruiz, Adaptive Importance Sampling for Control and Inference, Journal of Statistical Physics, 2016.<br>
   https://link.springer.com/article/10.1007/s10955-016-1446-7
3. Thalmeier et al., Adaptive Smoothing for Path Integral Control, JMLR, 2020.<br>
   https://jmlr.csail.mit.edu/papers/v21/18-624.html

Neural and rare-event importance sampling:

4. Arandjelović, Rheinländer, and Shevchenko, Importance sampling for option pricing with feedforward neural networks, Finance and Stochastics, 2025.<br>
   https://link.springer.com/article/10.1007/s00780-024-00549-x
5. Hartmann et al., Adaptive importance sampling with forward-backward stochastic differential equations.<br>
   https://arxiv.org/abs/1802.04981
6. Zhang, Sahai, and Marzouk, A Koopman framework for rare event simulation in stochastic differential equations, Journal of Computational Physics, 2022.<br>
   https://www.sciencedirect.com/science/article/pii/S0021999122000870

Rough volatility:

7. Bennedsen, Lunde, and Pakkanen, Hybrid scheme for Brownian semistationary processes, Finance and Stochastics, 2017.<br>
   https://link.springer.com/article/10.1007/s00780-017-0335-5
8. Zhu, Loeper, Chen, and Langrené, Markovian approximation of the rough Bergomi model for Monte Carlo option pricing.<br>
   https://arxiv.org/abs/2007.02113
9. Bayer and Breneis, Markovian approximations of stochastic Volterra equations with the fractional kernel.<br>
   https://arxiv.org/abs/2108.05048
10. Horvath et al., Functional central limit theorems for rough volatility, Finance and Stochastics, 2024.<br>
    https://link.springer.com/article/10.1007/s00780-024-00533-5
11. Gassiat, On the martingale property in the rough Bergomi model.<br>
    https://arxiv.org/abs/1811.10935

Current novelty watch:

12. Adaptive Learning via Off-Model Training and Importance Sampling for Fully Non-Markovian Optimal Stochastic Control, 2026 preprint.<br>
    https://arxiv.org/abs/2604.13147
13. Path Integral Formulation of Option Pricing Beyond Black-Scholes: A Unified Framework for Jump-Diffusion and Rough Volatility, 2026 SSRN preprint.<br>
    https://ssrn.com/abstract=6764698

Preprints 12–13은 peer-reviewed foundation으로 간주하지 않지만 novelty claim을 작성할 때 반드시 비교한다.

---

## 23. 최종 연구 원칙

> 이 연구의 성공은 경로적분이라는 이름을 사용하는 데 있지 않다. Payoff-tilted path law를 실제로 정의하고, 그 law를 만드는 adapted control을 rough memory와 task parameter의 함수로 근사하며, 그 근사오차가 importance-sampling variance와 실제 계산비용에 어떤 영향을 주는지 이론과 재현 가능한 실험으로 보여주는 데 있다.
