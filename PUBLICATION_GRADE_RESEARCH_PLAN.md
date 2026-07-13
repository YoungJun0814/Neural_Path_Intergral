# Neural Path Integral / DriftNet: 저명 학술지 투고 수준 연구 완성 계획

> 문서 상태: 실행용 마스터 로드맵<br>
> 작성 기준일: 2026-07-13<br>
> 기준 저장소: `YoungJun0814/Neural_Path_Intergral`, `main` / `5b77b89`<br>
> 권장 연구 기간: 전업 7–9개월, 파트타임 10–14개월<br>
> 문서 목적: 아이디어 수준의 프로토타입을 수학적·수치적·실증적으로 방어 가능한 논문과 재현 가능한 연구 아티팩트로 전환한다.

---

## 0. 최종 의사결정 요약

### 0.1 권장 논문 주제

**Amortized Neural Importance Sampling for Rare Downside Events under Rough Volatility**

한국어 작업 제목:

> **비마르코프 거친 변동성 모형에서 희귀 하방사건 추정을 위한 상각형 신경 중요도 샘플링**

핵심 질문은 다음 하나로 고정한다.

> rBergomi와 같은 비마르코프 rough-volatility 모형에서, 만기·장벽·모형 파라미터가 달라질 때마다 다시 최적화하지 않고 재사용할 수 있는 memory-aware importance-sampling control을 학습할 수 있는가? 그리고 이 방법이 정확한 likelihood ratio를 유지하면서 기존 방법보다 동일 계산비용에서 더 낮은 상대오차를 제공하는가?

### 0.2 논문의 최소 기여 패키지

논문에 반드시 함께 들어가야 하는 기여는 아래 네 가지다.

1. **정확한 비마르코프 측도변환**
   - rBergomi의 Volterra driver를 제어할 때 변동성 경로 전체에 생기는 memory correction을 명시한다.
   - 두 독립 Brownian driver에 대한 다차원 Girsanov likelihood를 정확히 구현한다.

2. **Memory-aware amortized controller**
   - 현재 상태뿐 아니라 rough-volatility memory state를 입력으로 사용한다.
   - barrier $K$, maturity $T$, payoff/event type, model parameters를 conditioning하여 하나의 모델을 여러 문제에 재사용한다.

3. **이론적 보장**
   - 고정된 학습 후 proposal에 대한 이산시간 추정량의 무편향성 또는 명시적인 이산화 bias 분해를 증명한다.
   - time-step 및 Markovian-lift 근사의 consistency를 제시한다.
   - 최소한 finite-second-moment/stability 결과를, 가능하면 rare-event asymptotic efficiency 결과를 제공한다.

4. **엄격한 계산 실험**
   - 확률 (10^{-4})–(10^{-8}) 수준의 실제 희귀사건을 대상으로 한다.
   - 강한 classical 및 neural baseline과 동일 compute budget으로 비교한다.
   - 학습비용 포함/제외 결과, 독립 seed, confidence interval, bias diagnostics를 모두 보고한다.

이 네 요소 중 하나라도 빠지면 상위 저널용 논문이 아니라 좋은 구현 논문 또는 연구 노트 수준으로 내려갈 가능성이 높다.

### 0.3 범위 관리 원칙

#### 1차 논문의 핵심 범위

- Heston: 수학 및 구현 검증용 Markovian benchmark
- rBergomi: 핵심 연구 대상
- terminal downside, barrier/drawdown, deep-OTM payoff: 핵심 rare-event functional
- amortization across $K,T,H,\eta,\rho,\xi_0$: 핵심 신규성
- Brownian control: 필수

#### 조건부 확장

- Bates/SVJJ jump tilting: 20주차 전까지 핵심 결과가 안정적일 때만 포함
- multi-asset: 1차 논문에서 제외하는 것을 원칙으로 함
- 실데이터 stress-testing application: 핵심 계산 결과 이후 보조 섹션으로만 포함

#### 명시적 비목표

- “AI가 블랙스완을 예측한다”는 주장
- 역사적으로 관측되지 않은 구조적 사건을 생성한다는 주장
- 옵션가격과 physical crash probability를 하나의 측도에서 동시에 설명한다는 주장
- 단순 crash hit-rate 증가를 efficiency로 부르는 것
- 한 개 seed와 한 개 hand-picked state를 이용한 XAI 일반화

### 0.4 명명 정책

- 논문에서는 방법을 우선 **memory-aware/amortized neural importance sampling**으로 부른다.
- `DriftNet`은 구현체 또는 프로젝트명으로만 사용한다.
- `Neural Path Integral`은 linearly-solvable stochastic control, Feynman–Kac/path-measure 표현 등 실제 path-integral 유도를 논문의 핵심으로 제시할 때만 학술 용어로 사용한다.
- PDE residual이나 보존법칙을 직접 사용하지 않는다면 `Physics-Informed AI` 대신 `structure-informed stochastic control` 또는 `model-based neural control`을 사용한다.
- `Black Swan prediction` 대신 `rare downside-event estimation` 또는 `tail-risk simulation`을 사용한다.

---

## 1. 투고 성공의 정의

### 1.1 논문 수준별 목표

| 목표 수준 | 필요 조건 | 후보 저널군 |
|---|---|---|
| **A: 이론 중심 상위권** | 새로운 정리, 엄밀한 proof, rough/non-Markovian novelty, 강한 수치 결과 | *Finance and Stochastics*, *Mathematical Finance* |
| **B: 강한 계산금융 논문** | 정확한 방법론, consistency 분석, 광범위 benchmark, 재현성 | *SIAM Journal on Financial Mathematics*, *Quantitative Finance* |
| **C: 전문 계산 논문** | 구현 정확성, 실무적으로 유용한 benchmark, 충분한 validation | *Journal of Computational Finance* 등 |

처음부터 A 수준을 설계하되, 20–24주차의 이론 및 실험 결과에 따라 B 수준으로 현실적으로 조정한다. 저널 이름보다 “실제로 확보한 기여”를 먼저 확정한다.

### 1.2 제출 가능 상태의 정량적 정의

다음 조건을 모두 만족해야 `submission-ready`로 간주한다.

- 모든 P0/P1 correctness issue 해결
- CI 전 구간 통과: lint, type check, unit/integration tests
- 핵심 실험 독립 seed 최소 20개
- 모든 핵심 표에 95% confidence interval 또는 적절한 uncertainty measure 포함
- 목표 확률 (10^{-4},10^{-6}), 가능하면 (10^{-8})에서 결과 확보
- 적어도 6개의 비교 방법 포함
- 가장 강한 baseline 대비 통계적으로 유의한 work-normalized 개선
- 훈련비용을 포함한 break-even analysis 포함
- time-step convergence 및 memory approximation convergence 포함
- 학습/평가 sample 분리
- 논문 주장마다 코드, figure/table, theorem 중 하나 이상의 직접 증거 연결
- clean environment에서 한 명의 제3자가 핵심 표를 재생성 가능
- README의 홍보성 문구와 논문의 엄밀한 주장 완전 일치
- 원고, appendix, source code, configs, seeds, environment lockfile 공개 준비

### 1.3 주장–증거 매트릭스

연구 중 모든 핵심 주장을 아래 형식으로 추적한다.

| Claim ID | 주장 | 필요한 증거 | 실패 시 조치 |
|---|---|---|---|
| C1 | 추정량이 정확하다 | theorem + unbiasedness matrix + discretization study | claim을 discrete-model 기준으로 축소 |
| C2 | rough memory가 중요하다 | memory-aware vs Markov-state ablation | memory novelty 재설계 |
| C3 | amortization이 유효하다 | unseen $K,T,\theta$ test + per-instance control 비교 | amortized claim 제거 |
| C4 | 계산 효율이 높다 | correct work-normalized VRF + wall clock + CI | efficiency claim 제거 |
| C5 | 매우 희귀한 사건에 안정적이다 | (10^{-6}) 이하 relative error/coverage | 목표 rare-event 범위 축소 |
| C6 | 기존 방법보다 우수하다 | strong baselines + paired statistical tests | 적용 영역을 명시적으로 제한 |

---

## 2. 수학적 문제 설정

### 2.1 금융 측도와 샘플링 측도의 분리

현재 저장소의 $\mathbb P/\mathbb Q$ 표기는 physical/risk-neutral measure와 proposal measure를 혼동할 여지가 있다. 논문에서는 다음 표기를 사용한다.

- $\mathbb M$: 추정 대상 금융 측도
  - physical stress probability이면 $\mathbb M=\mathbb P$
  - derivative pricing이면 $\mathbb M=\mathbb Q^{RN}$
- $\mathbb Q_{\phi}$: neural control이 정의하는 proposal/sampling measure
- $L_T^{\phi}=d\mathbb M/d\mathbb Q_{\phi}$: importance weight

최종 추정 대상은

$$
I(\theta)=\mathbb E^{\mathbb M}_{\theta}\big[F(X_{[0,T]};\zeta)\big]
=\mathbb E^{\mathbb Q_{\phi}}_{\theta}\big[F(X^{\phi}_{[0,T]};\zeta)L_T^{\phi}\big]
$$

로 정의한다. 여기서 $\theta$는 모형 파라미터, $\zeta$는 barrier, maturity, payoff type 등의 task parameter다.

### 2.2 Heston 검증 모형

Heston은 novelty 대상이 아니라 정답을 확인할 수 있는 validation model이다.

$$
\begin{aligned}
dS_t/S_t &= \mu_tdt+\sqrt{v_t}\, dB_t,\\
dv_t &= \kappa(\theta-v_t)dt+\xi\sqrt{v_t}\, dW_t,\\
d\langle B,W\rangle_t &= \rho dt.
\end{aligned}
$$

필수 검증:

- characteristic-function 또는 고정밀 reference pricer와 vanilla price 비교
- exact/noncentral-$\chi^2$ variance sampling 또는 검증된 QE scheme과 비교
- time-step convergence
- Brownian basis별 control과 likelihood sign 검증
- constant, affine, state-dependent frozen control에 대한 unbiasedness

### 2.3 rBergomi 핵심 모형

정규화된 Volterra process를

$$
W_t^H=\sqrt{2H}\int_0^t(t-s)^{H-\frac12}dW_s^1,
\qquad \mathrm{Var}(W_t^H)=t^{2H}
$$

로 두고,

$$
V_t=\xi_0(t)\exp\left(\eta W_t^H-\frac12\eta^2t^{2H}\right)
$$

$$
\frac{dS_t}{S_t}=\mu_tdt+\sqrt{V_t}\left(\rho dW_t^1+\sqrt{1-\rho^2}\, dW_t^2\right)
$$

로 명시한다.

proposal measure에서 두 Brownian driver를 모두 이동시키는 경우

$$
dW_t^{i,\mathbb M}=dW_t^{i,\mathbb Q_\phi}+u_t^idt,\qquad i=1,2
$$

이며 likelihood는

$$
\log L_T^\phi
=-\sum_{i=1}^2\int_0^Tu_t^i dW_t^{i,\mathbb Q_\phi}
-\frac12\sum_{i=1}^2\int_0^T|u_t^i|^2dt.
$$

$W^1$ 이동은 단순한 현재 drift 수정이 아니라

$$
\sqrt{2H}\int_0^t(t-s)^{H-\frac12}u_s^1ds
$$

라는 deterministic convolution correction을 rough driver에 추가한다. 이 memory-aware measure change를 정확히 구현하고 활용하는 것이 논문의 핵심이다.

### 2.4 Jump extension

Jump extension은 핵심 결과가 안정된 뒤 포함한다. proposal compensator를

$$
\nu_{\mathbb Q_\phi}(dt,dz)=r_\phi(t,X_{t-},z)\nu_{\mathbb M}(dz)dt,
\qquad r_\phi>0
$$

로 변경하면 jump likelihood component는

$$
\log L_T^{jump}
=-\int_0^T\int\log r_\phi(t,X_{t-},z)N(dt,dz)
+\int_0^T\int(r_\phi-1)\nu_{\mathbb M}(dz)dt
$$

형태가 된다. 실제 구현 전 다음을 별도로 증명·검증한다.

- compensator 변경에 따른 drift correction
- intensity-only tilting과 mark-distribution tilting 구분
- Brownian 및 jump likelihood의 합성
- positivity/integrability 조건
- 다중 variance jump가 Gamma 합분포를 따르는지 검증

Jump extension이 1차 논문의 명료성을 해치면 별도 후속 논문으로 분리한다.

### 2.5 희귀사건과 payoff 정의

최소 세 종류를 사용한다.

1. Terminal event
   [
   F_1=\mathbf1{S_T/S_0\leq K}
   ]

2. Drawdown/barrier event
   [
   F_2=\mathbf1\left{\min_{t\leq T}S_t/S_0\leq B\right}
   ]

3. Deep-OTM payoff 또는 tail loss
   [
   F_3=(K-S_T)^+,\qquad
   F_4=(L_T-q)^+\mathbf1{L_T>q}.
   ]

학습에서는 smooth surrogate를 사용할 수 있지만 최종 평가는 항상 원래의 hard event/payoff로 수행한다.

---

## 3. 이론 연구 프로그램

### 3.1 필수 정리

#### Theorem T1: Frozen-control estimator correctness

학습 데이터와 독립적인 평가 batch에서 고정된 adapted control을 사용할 때,

$$
\hat I_N=\frac1N\sum_{i=1}^NF(X^{\phi,i})L_T^{\phi,i}
$$

가 선택한 **이산시간 base model**의 expectation에 대해 unbiased임을 보인다.

중요 구분:

- measure-change correctness
- time discretization bias
- Markovian-lift/kernel approximation bias
- Monte Carlo sampling error

를 하나의 “unbiased”라는 말로 합치지 않는다.

#### Theorem T2: Approximation error decomposition

가능한 목표 형태:

$$
|\mathbb E[F(X)]-\mathbb E[F(X^{h,m})]|
\leq C_F\left(\varepsilon_{time}(h)+\varepsilon_{memory}(m)\right),
$$

여기서 $h$는 time step, $m$은 Markovian lift factor 수 또는 kernel approximation rank다.

비매끄러운 indicator/barrier payoff에서는 일반 Lipschitz 결과를 그대로 사용할 수 없으므로 다음 중 하나가 필요하다.

- smooth approximation과 smoothing bias bound
- boundary regularity/anti-concentration 조건
- continuity correction 또는 Brownian-bridge correction
- 결과를 Lipschitz payoff로 먼저 증명하고 indicator는 별도 proposition으로 처리

#### Theorem T3: Likelihood stability

bounded control 또는 명시적인 exponential-integrability 조건 아래

- Novikov/Kazamaki 조건
- $E_{\mathbb Q_\phi}[L_T]=1$
- $E_{\mathbb Q_\phi}[(FL_T)^2]<\infty$

를 보인다. Neural network output bound는 단순한 training trick이 아니라 likelihood integrability를 위한 설계 조건으로 해석한다.

#### Theorem T4: Amortized approximation result

task/model parameter 집합 $\Theta\times\Zeta$이 compact하고 optimal control이 적절한 regularity를 가질 때, conditional network가 control family를 uniform 또는 integrated sense로 근사할 수 있음을 제시한다.

실현 가능한 목표는 다음 중 하나다.

- task distribution에 대한 integrated excess second-moment bound
- finite parameter grid에서 interpolation generalization bound
- universal approximation proposition + empirical uniformity 결과

과도한 theorem을 약속하지 말고 실제로 증명 가능한 가장 강한 형태를 선택한다.

### 3.2 상위 저널용 stretch theorem

다음 중 하나를 확보하면 A급 투고 가능성이 크게 높아진다.

1. small-noise 또는 increasing-threshold regime에서 logarithmic efficiency
2. approximate control error와 relative variance 사이의 정량적 bound
3. memory truncation/lift error가 importance-sampling variance에 미치는 영향 bound
4. amortized control의 uniform near-optimality
5. rough-volatility path space에서 HJB/Doob-transform 근사와 neural control의 연결

모든 stretch theorem을 동시에 추진하지 않는다. 12주차에 한 가지를 선택한다.

### 3.3 이론 검증 방법

- GBM 또는 OU에서 zero-variance/near-optimal control을 알고 있는 toy problem 사용
- Heston의 단순 payoff에서 PDE 또는 characteristic-function reference 사용
- theorem assumption을 코드 config와 직접 연결
- proof에 쓰인 discretization과 실제 code path를 동일하게 유지
- appendix에 notation table, measure convention, filtration/adaptedness, integrability assumptions 명시

---

## 4. 알고리즘 설계

### 4.1 Controller 구조

두 모델을 반드시 비교한다.

#### Markov-state baseline

$$
u_t=\pi_\phi(t,S_t,V_t;\zeta,\theta)
$$

현재 DriftNet 구조에 해당한다. rBergomi에서 정보가 불충분한 baseline으로 사용한다.

#### Memory-aware proposal

$$
u_t=\pi_\phi(t,S_t,V_t,M_t;\zeta,\theta)
$$

$M_t$는 다음 후보를 순서대로 평가한다.

1. **Markovian sum-of-exponentials lift** — 이론과 재현성이 가장 좋으므로 1순위
2. fixed-dimensional Volterra convolution state
3. GRU/TCN path encoder — flexible benchmark
4. neural controlled differential equation/signature — stretch option

최종 논문은 이론 가능한 lift controller를 main method로, GRU/TCN을 empirical comparator로 두는 것이 안전하다.

### 4.2 Amortization input

Controller input에는 다음을 포함한다.

- normalized time $t/T$
- log-moneyness 또는 barrier distance $\log(S_t/K)$
- normalized variance/forward variance state
- maturity $T$
- event/payoff embedding
- $H,\eta,\rho$ 및 forward variance curve representation
- path-dependent payoff의 running minimum/average/drawdown
- memory lift state

절대 가격 $S$을 그대로 입력하기보다 scale-invariant state를 우선한다.

### 4.3 학습 목적함수

비교할 핵심 loss:

1. Second-moment / variance loss
   [
   \mathcal L_{SM}=\mathbb E_{\mathbb Q_\phi}[(F L^\phi)^2]
   ]

2. Log-variance 또는 path-measure divergence 기반 loss

3. Relative-entropy control loss
   [
   \mathbb E[\text{terminal cost}]+\frac\lambda2\mathbb E\int|u_t|^2dt
   ]

4. Cross-entropy method — 올바르게 구현된 classical/neural baseline

핵심 규칙:

- hard indicator의 pathwise gradient만으로 학습하지 않는다.
- smooth terminal loss를 사용하더라도 최종 estimator에는 hard payoff를 사용한다.
- stochastic-integral 기반 KL estimator와 직접 energy estimator를 모두 비교한다.
- likelihood는 float64 및 log domain을 기본으로 한다.
- 동일 batch로 학습과 최종 성능을 평가하지 않는다.

### 4.4 Rare-event curriculum

목표 barrier를 한 번에 극단적으로 설정하지 않고 rarity continuation을 사용한다.

예시:

$$
p\approx10^{-2}\rightarrow10^{-3}\rightarrow10^{-4}\rightarrow10^{-6}\rightarrow10^{-8}.
$$

각 단계에서 이전 control을 warm start하며 다음을 모니터링한다.

- event hit rate under proposal
- relative error
- target-specific contribution ESS
- $E_{\mathbb Q}[L]$ deviation
- max/min log-weight
- gradient norm 및 control saturation

### 4.5 평가 시 sample splitting

1. training seed 및 paths
2. hyperparameter validation paths
3. final frozen-control evaluation paths
4. independent reference paths

를 구분한다. Final table 생성 후 tuning을 하면 해당 table은 다시 생성해야 한다.

---

## 5. 코드베이스 재설계

### 5.1 권장 구조

```text
src/
  models/
    heston.py
    rbergomi.py
    bates.py                  # optional
  discretization/
    heston_schemes.py
    volterra_hybrid.py
    markovian_lift.py
  measures/
    brownian_girsanov.py
    jump_likelihood.py        # optional
    diagnostics.py
  controls/
    markov_mlp.py
    lift_controller.py
    recurrent_controller.py
    amortized.py
  objectives/
    second_moment.py
    log_variance.py
    cross_entropy.py
  payoffs/
    terminal.py
    barrier.py
    drawdown.py
  evaluation/
    estimators.py
    efficiency.py
    confidence_intervals.py
    convergence.py
  data/
    market_data.py
    splits.py
experiments/
  00_correctness/
  01_heston/
  02_rbergomi/
  03_amortization/
  04_ablations/
  05_real_data/
configs/
  paper/
tests/
paper/
  main.tex
  appendix.tex
  references.bib
```

### 5.2 즉시 해결할 P0 오류

| ID | 오류 | 수정 및 검증 | 완료 조건 |
|---|---|---|---|
| P0-1 | rBergomi negative fractional power NaN | mask 전 valid lag만 계산 | 전체 path finite + exact covariance test |
| P0-2 | Volterra weight에 extra $dt$ factor | BLP scheme을 원 논문 식대로 재구현 | reference implementation과 path-law 비교 |
| P0-3 | $\sqrt{2H}$ normalization/compensator 불일치 | $\mathrm{Var}(W_t^H)=t^{2H}$ 보장 | empirical variance 3σ 이내 |
| P0-4 | `kappa_hybrid` 미사용 | 실제 $\kappa$-cell exact scheme 또는 옵션 제거 | config 변화가 algorithm에 반영 |
| P0-5 | VRF cost 방향 오류 | $\sigma^2c$ 기준으로 수정 | 느린 방법이 자동 보상되지 않음 |
| P0-6 | invalid CEM pairing | 동일 elite trajectory 또는 weighted likelihood fitting | analytic toy CEM recovery |
| P0-7 | README 126×/unbiased/XAI 과장 | 검증 전 모두 제거 또는 historical result로 표시 | README와 현재 table 일치 |

### 5.3 P1 오류

- Full-Truncation Euler 식과 저장 variance convention 통일
- positive price를 위한 log-Euler/QE 또는 검증된 scheme 채택
- $T/dt$ 비정수일 때 마지막 step 조정
- controlled Bates/SVJJ에서 jump 누락 해결 또는 API에서 지원 제거
- 다중 variance jump를 Gamma distribution으로 처리
- barrier monitoring의 discrete bias와 Brownian-bridge correction 검토
- log-weight overflow/underflow 방지
- weight-only ESS를 핵심 효율지표에서 제외
- P-measure stress와 risk-neutral pricing 코드 경로 분리
- notebook monkey patch 제거
- package entry point 및 type protocol 정리
- checkpoint schema versioning

### 5.4 테스트 계층

#### Unit tests

- likelihood sign 및 normalization
- correlated Brownian basis
- Volterra covariance
- Poisson/Gamma jumps
- payoff 및 barrier state
- VRF, relative error, confidence interval

#### Property-based tests

- zero control이면 proposal과 base path law 동일
- $E_{\mathbb Q}[L]\approx1$
- constant control analytic Gaussian case
- control bound 아래 finite likelihood
- $\rho=0,\pm1$ 경계 근처 동작

#### Statistical regression tests

- 다수 seed의 unbiasedness rejection rate
- time-step refinement slope
- rBergomi $E[V_t]=\xi_0(t)$
- Heston reference price
- rare-event estimator CI coverage

#### End-to-end tests

- 작은 config로 training → freeze → evaluation → table 생성
- clean CPU environment에서 10분 내 smoke paper pipeline
- full GPU benchmark는 nightly 또는 release workflow

---

## 6. 실험 설계

### 6.1 실험 질문

| RQ | 질문 |
|---|---|
| RQ1 | estimator와 likelihood가 정확한가? |
| RQ2 | memory-aware controller가 Markov-state controller보다 좋은가? |
| RQ3 | amortized model이 unseen task/model parameters에 일반화하는가? |
| RQ4 | 동일 총 계산비용에서 classical/neural baselines보다 효율적인가? |
| RQ5 | rarity가 $10^{-8}$로 증가해도 relative error가 안정적인가? |
| RQ6 | time-step 및 memory approximation에 결과가 강건한가? |
| RQ7 | training cost를 포함했을 때 몇 개 task 이후 break-even인가? |

### 6.2 모형 매트릭스

| Model | 역할 | 필수 여부 |
|---|---|---|
| Brownian motion / GBM | analytic zero-variance sanity | 필수 |
| OU/CIR | HJB/CEM sanity | 필수 |
| Heston | Markovian financial benchmark | 필수 |
| rBergomi exact-covariance small grid | reference | 필수 |
| rBergomi BLP hybrid | main scalable model | 필수 |
| rBergomi Markovian lift | main controller state | 필수 |
| Bates/SVJJ | jump robustness | 조건부 |

### 6.3 사건 난이도

각 모형에서 parameter sweep을 통해 base probability가 대략 다음이 되도록 barrier를 선택한다.

- warm-up: $10^{-2}$
- moderate rare: $10^{-3},10^{-4}$
- primary: $10^{-5},10^{-6}$
- stretch: $10^{-7},10^{-8}$

기존의 0.79% 결과는 warm-up에만 해당하며 headline result로 사용하지 않는다.

### 6.4 비교 방법

최소 비교군:

1. Naive Monte Carlo
2. Antithetic/QMC 가능 시 포함
3. Constant exponential tilting / Esscher control
4. Large-deviation 또는 most-likely-path control
5. Cross-Entropy Method
6. Adaptive Multilevel Splitting 또는 subset simulation
7. Per-instance neural control
8. Markov-state amortized neural control
9. Memory-aware amortized neural control — 제안 방법
10. 가능하면 기존 HJB/log-variance neural method 재현

비교 방법이 proposal과 다른 종류의 estimator를 사용하더라도 동일 error tolerance 또는 동일 wall-clock budget 기준으로 비교한다.

### 6.5 핵심 지표

#### 정확성

- point estimate
- absolute/relative bias against reference
- 95% confidence interval
- CI coverage
- $E_{\mathbb Q}[L]-1$

#### 통계 효율

- variance of single-path contribution $Y=FL$
- standard error
- relative error $SE/|\hat I|$
- coefficient of variation
- correct work-normalized VRF
- target-specific contribution concentration

#### 계산 효율

- training wall-clock
- evaluation wall-clock
- peak GPU/CPU memory
- paths/sec
- break-even task count

#### 안정성

- log-weight quantiles
- control saturation ratio
- failed/NaN runs
- between-seed variance
- parameter OOD degradation

### 6.6 올바른 work-normalized 비교

한 경로당 비용을 $c$, single-path contribution variance를 $\sigma^2$, 총 예산을 $B$라 하면

$$
\mathrm{Var}(\hat I\mid B)\approx\frac{\sigma^2c}{B}.
$$

따라서

$$
\mathrm{VRF}_{work}
=\frac{\sigma^2_{MC}c_{MC}}
{\sigma^2_{IS}c_{IS}}.
$$

두 가지 결과를 분리 보고한다.

1. **Online VRF**: 이미 학습된 control을 반복 사용하는 경우
2. **End-to-end VRF**: training amortization을 포함한 경우

Amortized method는 task 수 $n$에 따른 total cost curve를 제시한다.

### 6.7 통계 프로토콜

- 핵심 결과는 독립 training seed 최소 20개
- 각 frozen model마다 독립 evaluation batch
- 가능하면 common random numbers를 사용한 paired comparison
- 평균뿐 아니라 median, IQR, 95% CI 보고
- 여러 barrier/maturity 비교 시 family-wise interpretation 주의
- hyperparameter는 validation task에서 선택하고 test grid는 마지막에 한 번 공개
- failure run을 삭제하지 말고 실패율로 보고
- reference value가 불확실하면 서로 독립적인 두 고정밀 방법으로 교차검증

### 6.8 Ground truth 전략

- GBM/OU: analytic
- Heston vanilla: characteristic function 또는 검증된 high-accuracy pricer
- Heston barrier/path-dependent: 매우 미세한 grid + bridge correction + 대규모 QMC/reference IS
- rBergomi: exact covariance small-grid reference, BLP refinement, 독립 high-budget method
- $10^{-8}$ 사건: naive MC를 ground truth로 사용하지 않고 splitting/independent IS와 confidence interval 교차검증

“reference”에도 uncertainty가 있음을 표에 명시한다.

### 6.9 필수 ablation

1. current-state vs memory-aware
2. per-task vs amortized
3. $W^1$ control only vs $W^1,W^2$ joint control
4. second moment vs log-variance vs entropy loss
5. curriculum on/off
6. control bound sweep
7. Markovian lift rank $m$
8. time step $h$
9. training task-grid density
10. model parameter conditioning on/off
11. hard-event training surrogate 종류
12. training cost 포함/제외

---

## 7. 실데이터 및 금융 해석 계획

### 7.1 먼저 결정할 측도

#### 선택 A: Risk-neutral pricing paper — 권장 main application

- SPX/SPY option surface에 rBergomi calibration
- deep-OTM put, barrier, drawdown-linked payoff 가격 추정
- drift는 risk-free/carry 조건과 일치
- historical return calibration과 혼합하지 않음

장점: 수학적 문제와 estimator accuracy를 명확하게 정의할 수 있다.
단점: 2024년 neural option importance-sampling 논문과 직접 비교해야 한다.

#### 선택 B: Physical-measure stress probability

- return 및 realized-volatility time series에 physical model calibration
- rolling out-of-sample tail probability 및 VaR/ES validation
- crisis period를 사전에 정의

장점: 기존 DriftNet의 stress-testing 동기와 가깝다.
단점: model risk와 statistical estimation error가 IS 개선 효과와 섞인다.

**권장:** 1차 논문의 main theorem/experiments는 risk-neutral 또는 fully specified base measure 아래 계산방법론에 집중하고, physical stress application은 별도 보조 섹션이나 후속 논문으로 둔다.

### 7.2 데이터 거버넌스

- 데이터 vendor와 license 명시
- raw data는 공개 불가하면 download/preprocessing script와 schema 제공
- 날짜 기준 고정 snapshot
- survivorship/look-ahead 확인
- rates, dividends, forwards, option filtering 규칙 문서화
- train/validation/test 기간 사전 고정
- crisis 기간을 결과를 본 뒤 선택하지 않음

### 7.3 실증 검증

- calibration in-sample fit만으로 성공 판정하지 않음
- out-of-sample option surface 또는 tail-risk forecast 평가
- benchmark model: Heston, rBergomi, historical simulation, GARCH/EVT는 application에 따라 선택
- parameter uncertainty와 IS Monte Carlo error를 분리
- model misspecification stress test 포함

---

## 8. 재현성과 연구 운영

### 8.1 실행 환경

- Python minor version 고정
- PyTorch/CUDA 버전 고정
- `uv.lock`, `poetry.lock` 또는 fully pinned requirements 제공
- CPU reference와 GPU production config 분리
- deterministic mode의 한계 문서화
- Docker image digest 또는 reproducible container 제공

### 8.2 Experiment registry

각 run에 반드시 기록한다.

- git commit
- config 전체
- seed
- device/library versions
- start/end timestamp
- training/evaluation cost
- checkpoint checksum
- dataset snapshot ID
- metrics JSON/Parquet
- stdout/stderr log

### 8.3 Table/Figure 자동 생성

논문 표와 그림을 notebook에서 수동 복사하지 않는다.

```text
python -m experiments.run --config configs/paper/rq2.yaml
python -m experiments.aggregate --study rq2
python -m paper.make_tables --study rq2
python -m paper.make_figures --study rq2
```

형태의 one-command pipeline을 만든다. 모든 table cell은 raw run ID까지 추적 가능해야 한다.

### 8.4 공개 아티팩트

- 코드 release tag
- Zenodo/OSF DOI 가능 시 생성
- small reproducibility dataset
- pretrained amortized controller
- full config set
- minimal CPU demo
- full paper result reproduction instructions
- expected runtime 및 hardware 명시

---

## 9. 단계별 실행 로드맵

### Phase 0 — 주장 정리 및 연구 동결점 설정 (1주)

#### 작업

- README의 126×, “zero bias”, “statistically impossible”, 60% XAI headline 제거 또는 검증 전 상태로 변경
- 현재 결과를 `legacy_results/`로 분리
- 논문용 branch 및 issue board 생성
- measure notation과 primary use case 결정
- 이 문서의 scope를 연구 노트에 승인 기록

#### 완료 조건 G0

- 공개 문서에 검증되지 않은 정량 claim 없음
- P/Q/proposal notation 결정
- 1차 논문 범위와 제외 범위 확정

### Phase 1 — 수치 엔진 correctness (2–5주)

#### 작업

- Heston scheme 교체/정리
- rBergomi BLP 재구현
- exact-covariance reference 구현
- Brownian likelihood module 분리
- jump 코드는 비활성화하거나 정확히 수정
- full statistical test suite 구축

#### 산출물

- `docs/mathematical_specification.md`
- Heston/rBergomi validation report
- convergence plots
- CI green

#### 완료 조건 G1

- finite-path test 100% 통과
- $E[V_t]$, covariance, price convergence tolerance 충족
- $E_{\mathbb Q}[L]=1$ 다수 control/seed에서 통과
- Ruff/Mypy/Pytest 모두 통과

### Phase 2 — Markovian neural IS reference (6–8주)

#### 작업

- Heston에서 correct per-instance controller 구현
- second-moment, log-variance, entropy loss 비교
- 올바른 CEM 구현
- correct efficiency metrics 구현
- training/evaluation split

#### 완료 조건 G2

- analytic/toy problem에서 estimator correctness
- Heston $10^{-4}$–$10^{-6}$에서 constant/CEM 대비 경쟁력
- bias z-score 및 CI coverage 양호

### Phase 3 — Rough memory-aware control (9–13주)

#### 작업

- Markovian lift 구현
- joint Brownian control 구현
- memory-aware controller
- time/memory discretization ablation
- current-state controller와 비교

#### 완료 조건 G3

- rBergomi 전 구간 finite/stable
- memory-aware가 최소 두 rarity level과 여러 $\rho,H$에서 유의하게 개선
- improvement가 단순 parameter count 증가 때문이 아님을 ablation으로 확인

### Phase 4 — Amortization (14–17주)

#### 작업

- $K,T,\theta$ conditional controller
- task sampler 설계
- unseen interpolation/OOD test
- per-instance fine-tuning optional
- break-even analysis

#### 완료 조건 G4

- unseen in-domain task에서 per-instance controller 성능의 사전 정의 비율 이상 유지
- 여러 task를 평가할 때 training amortization 이점 확인
- OOD 실패 영역 명확히 기술

### Phase 5 — 이론 완성 (10–20주, 구현과 병행)

#### 작업

- T1–T3 proof 완성
- T4 또는 stretch theorem 하나 선택
- assumptions와 code constraints 일치 확인
- 외부 수학 연구자에게 proof review 요청

#### 완료 조건 G5

- 본문 theorem과 appendix proof 완성
- proof gap 목록 0개
- notation/code mismatch 0개
- theorem이 단순한 기존 Girsanov 재서술을 넘어서는 기여 포함

### Phase 6 — 대규모 benchmark (18–24주)

#### 작업

- 사전 등록된 experiment grid 동결
- 20+ seeds 실행
- $10^{-4}$–$10^{-8}$ sweep
- baselines 및 ablations
- wall-clock/memory 측정
- optional jump decision

#### 완료 조건 G6

- 핵심 table 3–5개 완성
- 모든 주요 결론에 uncertainty 포함
- 가장 강한 baseline 대비 우위 또는 명확한 적용영역 확보
- 결과가 약하면 claim 축소 또는 방법 재설계 후 다음 단계로 이동

### Phase 7 — 실데이터 application (22–26주, 조건부)

#### 작업

- data license 및 snapshot 확정
- calibration/validation split
- 한 개 명확한 application table
- model risk discussion

#### 완료 조건

- 계산방법 기여를 흐리지 않는 보조 evidence 확보
- 실데이터 결과가 불안정하면 appendix 또는 후속 연구로 이동

### Phase 8 — 원고 작성 및 내부 심사 (25–31주)

#### 작업

- 본문/appendix 작성
- figure/table freeze
- claim–evidence audit
- related-work novelty matrix
- 최소 두 명의 외부 reader review
- 공개 artifact dry run

#### 완료 조건 G7

- 제3자 clean-room reproduction 성공
- 모든 reviewer-style 질문에 문서화된 답변
- target journal 형식/길이/데이터 정책 충족
- submission checklist 100% 완료

### Phase 9 — 투고 및 대응 준비 (32주 이후)

- cover letter
- contribution summary 3문장
- suggested reviewers/conflicts 확인
- arXiv 및 code release timing 결정
- rebuttal용 추가 실험 budget 확보

---

## 10. 32주 주차별 일정

| 주차 | 핵심 목표 | 주 산출물 |
|---|---|---|
| 1 | claim 정리, scope/measure 확정 | revised README, research charter |
| 2 | test harness, Heston reference | analytic validation tests |
| 3 | rBergomi exact reference | covariance/mean tests |
| 4 | BLP hybrid 재구현 | convergence report |
| 5 | likelihood 및 CI green | correctness release v0.3 |
| 6 | Heston neural IS | per-instance baseline |
| 7 | log-variance/CEM | objective comparison |
| 8 | rare-event Heston matrix | G2 report |
| 9 | Markovian lift | lift validation |
| 10 | memory controller | first rough IS run |
| 11 | joint driver control | driver ablation |
| 12 | theorem stretch 선택 | theory memo |
| 13 | rough benchmark | G3 report |
| 14 | task conditioning | amortized prototype |
| 15 | task sampler/generalization | interpolation results |
| 16 | OOD 및 fine-tuning | OOD map |
| 17 | break-even | G4 report |
| 18 | experiment protocol freeze | preregistered grid |
| 19 | baseline full runs | raw registry |
| 20 | rarity sweep | $10^{-4}$–$10^{-8}$ results |
| 21 | ablations | ablation tables |
| 22 | discretization sensitivity | convergence tables |
| 23 | independent repeats | seed/CI results |
| 24 | benchmark freeze | G6 report |
| 25 | real-data/optional extension | application result |
| 26 | paper Methods/Theory | draft v0.4 |
| 27 | paper Experiments | draft v0.6 |
| 28 | Introduction/Related Work | draft v0.8 |
| 29 | appendix/repro package | release candidate |
| 30 | external review | review log |
| 31 | revision/clean-room run | submission candidate |
| 32 | journal-specific formatting | final submission |

---

## 11. 논문 구성안

### Abstract

- 문제: rough volatility에서 rare-event MC가 비효율적이고 최적 proposal이 non-Markovian
- 방법: memory-aware amortized neural control + exact path-space likelihood
- 이론: correctness/consistency/stability 결과
- 결과: rarity 범위, strongest baseline 대비 work-normalized improvement
- 제한: 적용 범위와 model assumptions

### 1. Introduction

- 희귀사건 계산 문제
- rough volatility가 추가하는 non-Markovian 난점
- 기존 neural IS의 한계: 주로 Markovian/per-instance
- 정확히 세거나 네 개의 contribution bullet

### 2. Related Work

- rare-event IS와 stochastic control
- HJB/Doob transform
- neural importance sampling
- rough-volatility simulation
- amortized optimization
- jump tilting은 포함 시 별도 subsection

### 3. Problem Formulation

- financial target measure vs proposal measure
- rBergomi dynamics
- event/payoff family
- efficiency definition

### 4. Method

- Volterra/Markovian memory representation
- controlled dynamics
- likelihood
- amortized controller
- training objective/curriculum

### 5. Theory

- estimator correctness
- approximation decomposition
- stability
- amortized 또는 efficiency theorem

### 6. Numerical Validation

- analytic toy
- Heston
- rBergomi scheme validation

### 7. Rare-Event Experiments

- baselines
- main performance
- amortization
- ablation/convergence

### 8. Application 또는 Extension

- real-data calibrated example 또는 jump extension 중 하나만 본문에 포함

### 9. Limitations

- model risk
- training cost
- extreme OOD
- theorem assumptions
- data/calibration limitations

### 10. Conclusion

- 검증된 기여만 요약

### Appendix

- proofs
- implementation details
- full parameter tables
- extra seeds/ablations
- reproducibility instructions

---

## 12. 예상 핵심 표와 그림

### 필수 표

1. Model/scheme correctness table
2. Heston rare-event benchmark
3. rBergomi rare-event benchmark
4. Amortized vs per-instance generalization
5. Work-normalized cost 및 break-even
6. Ablation table

### 필수 그림

1. Method diagram: base measure → memory controller → controlled simulator → likelihood estimator
2. Relative error vs rarity
3. Error vs wall-clock
4. Performance heatmap over $K,T,H,\rho$
5. Time-step/memory-rank convergence
6. Amortization break-even curve

샘플 crash path 그림은 설명용으로만 사용하고 성과 증거로 사용하지 않는다.

---

## 13. 리스크와 피벗 전략

| 리스크 | 조기 경고 | 대응/피벗 |
|---|---|---|
| memory controller가 효과 없음 | G3에서 Markov baseline과 차이 없음 | task를 path-dependent barrier로 강화하거나 state representation 재설계 |
| amortization 성능 저하 | unseen task에서 큰 variance | local fine-tuning/hypernetwork/mixture-of-experts 검토 |
| $10^{-8}$ 학습 불안정 | weight collapse, zero events | rarity continuation, splitting hybrid, log-variance loss |
| theorem이 약함 | T4 proof gap 지속 | B급 계산저널로 조정하고 consistency/experiment 강화 |
| training cost가 너무 큼 | break-even task 수 과도 | amortization을 핵심으로 최적화하거나 online claim 축소 |
| baseline이 더 우수함 | CEM/LD가 전 구간 우위 | neural이 유리한 high-dimensional/path-dependent 영역을 사전 규칙 아래 탐색 |
| real data가 모호함 | calibration instability | application을 appendix로 이동, method paper로 집중 |
| jump extension이 범위를 폭발시킴 | 20주차까지 correctness 미완료 | 후속 논문으로 분리 |
| reference value 불확실 | 방법 간 CI 불일치 | independent high-budget cross-validation 및 claim 보류 |

결과가 약할 때 실험을 선택적으로 숨기지 않는다. 적용영역을 좁히거나 연구 질문을 수정한다.

---

## 14. 첫 4주 상세 작업 목록

### Week 1

- [ ] `RESEARCH_CHARTER.md` 작성: target measure, payoff, novelty, non-goals
- [ ] README 과장 claim 제거
- [ ] legacy notebook 결과와 paper pipeline 분리
- [ ] issue label: `P0-correctness`, `theory`, `experiment`, `paper`
- [ ] branch protection 및 CI required 설정
- [ ] mathematical notation 확정
- [ ] reference bibliography 구축

### Week 2

- [ ] Heston simulator interface 재설계
- [ ] verified scheme/QE 또는 reference 구현
- [ ] characteristic-function price test
- [ ] constant/affine control likelihood test
- [ ] $E[L]=1$ property test
- [ ] correct VRF/relative-error module
- [ ] statistical CI helper

### Week 3

- [ ] exact covariance rBergomi small-grid simulator
- [ ] normalized Volterra covariance test
- [ ] $E[V_t]=\xi_0(t)$ test
- [ ] shared-driver correlation test
- [ ] BLP kernel weights 독립 reference 계산
- [ ] current rBergomi code 제거 또는 quarantine

### Week 4

- [ ] scalable BLP hybrid implementation
- [ ] time-step convergence
- [ ] implied-volatility/skew sanity against reference
- [ ] CPU/GPU consistency
- [ ] NaN/extreme parameter stress test
- [ ] Phase-1 correctness report 초안

첫 4주에는 새로운 neural architecture를 추가하지 않는다. 수치 엔진과 likelihood가 완전히 신뢰 가능한 상태가 된 이후 학습 연구를 재개한다.

---

## 15. 연구 의사결정 게이트

### Gate A — 5주차

**질문:** base simulators와 likelihood가 신뢰 가능한가?

- 아니오: 일정과 관계없이 Phase 1 유지
- 예: neural IS 개발 진행

### Gate B — 13주차

**질문:** rough memory가 통계적으로 의미 있는 개선을 제공하는가?

- 아니오: representation 또는 payoff 재설계
- 예: amortization 진행

### Gate C — 17주차

**질문:** amortization이 실제 계산 이점을 제공하는가?

- 아니오: per-instance rough-control paper로 scope 축소
- 예: amortized claim 유지

### Gate D — 20주차

**질문:** 상위 저널을 지지할 theorem이 있는가?

- 강한 theorem: A군 저널 준비
- consistency 수준: B군 저널 준비
- 이론 미완성: proof 보강 또는 계산저널로 조정

### Gate E — 24주차

**질문:** strongest baseline 대비 robust한 우위가 있는가?

- 예: 논문 freeze
- 특정 영역만 예: claim과 title을 그 영역으로 제한
- 아니오: submission 중단, 원인 분석 및 method pivot

---

## 16. Submission checklist

### 수학

- [ ] 모든 확률측도와 filtration 정의
- [ ] control adaptedness 명시
- [ ] Brownian/jump likelihood sign 교차검산
- [ ] integrability assumptions 명시
- [ ] theorem과 code discretization 일치
- [ ] indicator payoff 처리의 regularity 논의

### 실험

- [ ] baseline implementation 검증
- [ ] 독립 seed 20+
- [ ] confidence interval
- [ ] correct cost accounting
- [ ] training/evaluation 분리
- [ ] negative/failed results 기록
- [ ] convergence/ablation/OOD 포함

### 재현성

- [ ] clean clone 재현
- [ ] lockfile/container
- [ ] configs/seeds/checksums
- [ ] automatic tables/figures
- [ ] release tag 및 archive

### 원고

- [ ] contribution이 선행연구 대비 문장 단위로 명확
- [ ] 과장 표현 제거
- [ ] limitation section 충실
- [ ] abstract 수치와 main table 일치
- [ ] 모든 figure axis/error bar/units 확인
- [ ] notation consistency pass
- [ ] proof external review

---

## 17. 우선 참고문헌

아래 문헌을 related-work와 방법 설계의 시작점으로 삼고, 투고 직전 최신 문헌 검색을 다시 수행한다.

1. Bennedsen, Lunde, Pakkanen, “Hybrid scheme for Brownian semistationary processes,” *Finance and Stochastics* 21, 2017.<br>
   DOI: https://doi.org/10.1007/s00780-017-0335-5

2. Bayer, Friz, Gatheral, “Pricing under rough volatility,” *Quantitative Finance* 16(6), 2016.<br>
   DOI: https://doi.org/10.1080/14697688.2015.1099717

3. Nüsken, Richter, “Solving high-dimensional Hamilton–Jacobi–Bellman PDEs using neural networks: perspectives from controlled diffusions and path measures,” 2021.<br>
   Preprint: https://arxiv.org/abs/2005.05409

4. Arandjelović et al., “Importance sampling for option pricing with feedforward neural networks,” *Finance and Stochastics*, 2024.<br>
   DOI: https://doi.org/10.1007/s00780-024-00549-x

5. Zhang, Sahai, Marzouk, “A Koopman framework for rare event simulation in stochastic differential equations,” *Journal of Computational Physics*, 2022.<br>
   Article: https://www.sciencedirect.com/science/article/pii/S0021999122000870

6. Hult, Jain, Juneja, Nyquist, Vijayan, “A Deep Learning Approach for Rare Event Simulation in Diffusion Processes,” *Winter Simulation Conference*, 2024.<br>
   Paper: https://informs-sim.org/wsc24papers/con321.pdf

7. Lord, Koekkoek, van Dijk, “A comparison of biased simulation schemes for stochastic volatility models,” *Quantitative Finance* 10(2), 2010.<br>
   DOI: https://doi.org/10.1080/14697680802392496

8. Gao, “A Short Report on Importance Sampling for Rare Event Simulation in Diffusions,” 2025.<br>
   Preprint: https://arxiv.org/abs/2512.17766

9. Asmussen, Glynn, *Stochastic Simulation: Algorithms and Analysis*, Springer, 2007.

10. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2004.

---

## 18. 최종 원칙

이 연구의 성공 기준은 “제어된 분포에서 crash path를 많이 생성하는 것”이 아니다. 성공 기준은 다음 문장으로 요약된다.

> **정확한 measure change를 유지하면서, 비마르코프 rough-volatility의 memory를 이용한 재사용 가능한 proposal이, 실제 희귀사건 추정에서 기존 최강 방법보다 동일 총 계산비용당 더 작은 오차를 제공하고, 그 이유와 한계를 수학적으로 설명하는 것.**

이 문장을 theorem, code, experiments, paper가 동시에 지지할 때 저명 학술지 투고 수준에 도달한다.
