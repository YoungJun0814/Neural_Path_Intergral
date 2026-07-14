# 차세대 Neural Path-Integral 모델 후보군과 연구 선택 전략

Status: research decision document<br>
Version: 1.0<br>
Date: 2026-07-14<br>
Repository: Neural_Path_Integral<br>
Related plan: [PATH_INTEGRAL_RESEARCH_PLAN_V2.md](PATH_INTEGRAL_RESEARCH_PLAN_V2.md)

---

## 0. 문서의 목적

이 문서는 현재 연구를 단순한 기존 신경망의 rough-volatility 적용으로 끝내지 않고,
새로운 path-space 모델과 계산 원리를 갖는 연구로 발전시키기 위한 모델 후보군을
정의하고 우선순위를 결정한다.

구체적으로 다음 질문에 답한다.

1. 어떤 모델이 현재 코드와 수학적 기반에서 실제로 구현 가능한가?
2. 어떤 모델이 기존 PICE, neural importance sampling, Neural RDE, Markovian lift,
   normalizing flow 및 Schrödinger bridge 연구와 구별되는가?
3. 어떤 모델이 정확한 likelihood와 unbiased estimator를 유지할 수 있는가?
4. 어떤 모델이 rough-volatility의 비마르코프 기억을 본질적으로 활용하는가?
5. 어떤 모델이 논문 한 편을 넘어 범용 rare-event 계산 원리로 확장될 수 있는가?
6. 구현 실패와 과도한 연구 확장을 막기 위한 중단 기준은 무엇인가?

이 문서는 Plan v2를 즉시 대체하지 않는다. Plan v2의 controlled rBergomi, 정확한
likelihood, PICE, second-moment objective, memory 및 amortization 기반을 유지하면서,
그 위에 올릴 **새로운 주모델을 선택하는 설계 문서**다.

---

## 1. 최종 의사결정 요약

### 1.1 권고 순위

현재 코드, 이론적 위험, 선행연구 충돌, 실험 가능성과 장기 확장성을 종합하면 다음
순서가 가장 합리적이다.

| 순위 | 후보 | 강한 논문 성공 가능성 | 혁신 잠재력 | 구현 위험 | 권고 역할 |
|---:|---|---:|---:|---:|---|
| 1 | Volterra–Föllmer Operator (VFO) | 65–80% | 높음 | 중간 | 주모델 |
| 2 | Multiresolution Renormalized VPIC (MR-VPIC) | 50–65% | 매우 높음 | 중상 | flagship 확장 |
| 3 | Doob–Volterra Desirability Network (DVDN) | 45–60% | 매우 높음 | 높음 | 이론·안정성 보조모델 |
| 4 | Causal Path-Space Transport (CAPT) | 35–50% | 극히 높음 | 높음 | drift family 실패 시 확장 |
| 5 | Multimodal Schrödinger–Volterra Bridge (MSVB) | 30–45% | 극히 높음 | 매우 높음 | 다중 rare-path mode 확인 시 확장 |
| 6 | Universal Path-Measure Compiler (UPMC) | 10–25% | 가장 높음 | 극히 높음 | 장기 연구 프로그램 |

위 범위는 통계적으로 보정된 확률이 아니다. 현재 저장소를 출발점으로 한 명의 주
연구자가 충분한 GPU 자원과 12–24개월 이상의 연구 시간을 확보하고, 아래의 단계별
gate를 지킨다는 가정의 의사결정용 추정치다.

### 1.2 최종 권고 모델

가장 현실적인 주모델은 다음이다.

> **VFO: task- and kernel-conditioned Volterra–Föllmer operator**

가장 강한 최종 논문 모델은 다음 결합이다.

> **MR-VFO: Multiresolution Volterra–Föllmer Operator with optional
> desirability consistency**

구성 원칙은 다음과 같다.

1. VFO를 최소 주모델로 먼저 성공시킨다.
2. 서로 다른 time grid에서 control이 불일치하는 문제가 실제로 확인되면
   multiresolution consistency를 추가한다.
3. direct-control head가 불안정하거나 시간 일관성이 부족하면 DVDN의 positive
   desirability/martingale head를 추가한다.
4. drift-only proposal이 multi-modality 또는 heavy-tail 때문에 실패한다는 증거가
   있을 때만 CAPT를 개발한다.
5. 여러 seed가 서로 다른 rare-event mechanism에 수렴하는 증거가 있을 때만
   MSVB를 개발한다.

### 1.3 노벨상 수준의 목표에 대한 현실적 해석

한 개의 neural architecture가 발표 즉시 노벨상 수준의 연구가 되는 것은 현실적이지
않다. 경제학상 또는 광범위한 학문적 전환을 일으키려면 다음이 필요하다.

- 금융 한 분야를 넘어서는 새로운 계산 원리
- 기존 이론을 포괄하거나 재구성하는 정리
- 여러 분야에서 재현되는 실질적 우위
- 장기간의 후속연구와 채택
- 경제·과학적 의사결정에 대한 명확한 영향

본 연구에서 그 수준의 장기 아이디어는 단순한 rBergomi controller가 아니라 다음이다.

> **동역학, 기억 kernel과 사건 functional을 입력받아 정확한 likelihood를 가진
> causal rare-path sampler를 생성하는 범용 path-measure operator**

VFO는 이 장기 목표의 첫 번째 좁고 검증 가능한 구현이다. UPMC는 그 최종 확장이다.

---

## 2. 현재 연구 기반과 바꾸지 않을 원칙

### 2.1 이미 확보된 기반

현재 저장소에는 다음 기반이 있다.

- target/proposal measure convention
- Girsanov likelihood와 action
- Gaussian analytic oracle
- constant PICE projection
- independent Brownian basis의 two-driver Heston simulator
- Heston soft desirability oracle와 Fourier gradient
- rBergomi BLP/hybrid base simulator
- estimator 및 likelihood diagnostics
- Heston과 rBergomi correctness tests

따라서 차세대 모델의 병목은 더 큰 MLP를 만드는 것이 아니라 다음 네 가지다.

1. exact discrete controlled rBergomi path law
2. rough memory의 구조적 표현
3. payoff-tilted optimal path law의 안정적인 근사
4. control approximation과 estimator variance의 정량적 연결

### 2.2 모든 후보가 지켜야 할 비협상 조건

#### Adaptedness

시점 `t_i`의 control은 `t_i`까지의 filtration만 사용해야 한다.

$$
u_i \in \mathcal F_{t_i}.
$$

bidirectional attention, 전체 경로 normalization, 미래까지 포함한 Fourier/wavelet
coefficient와 terminal state를 controller input에 사용해서는 안 된다.

#### Exact likelihood on the declared sampling space

논문에서 사용하는 proposal에 대해 `dP/dQ`를 실제 구현과 동일한 좌표에서 계산해야
한다. target dynamics를 BLP discretization으로 선언한다면 unbiasedness claim은 우선
그 discrete target에 대해서만 한다.

#### Target simulator와 feature approximation의 분리

SOE 또는 Markovian lift를 memory feature로 사용하는 것과 target dynamics 자체를
lift로 대체하는 것을 구분한다. Main estimator가 BLP target을 유지하면 auxiliary
memory approximation이 곧 pricing bias가 되지는 않는다. 반대로 lifted target으로
가격을 계산하면 kernel approximation bias를 별도로 보고해야 한다.

#### Final estimator에서 self-normalization 금지

self-normalized weights는 PICE training에는 사용할 수 있지만 finite sample에서
biased하다. 최종 가격과 확률은 raw likelihood-weighted estimator로 보고한다.

#### Hard target과 soft training의 분리

soft potential로 학습하더라도 최종 평가는 원래 hard event/payoff에서 수행한다.
soft objective의 감소를 hard-event variance 감소로 간주하지 않는다.

#### Training-inclusive efficiency

실용성 주장은 inference cost만으로 평가하지 않는다.

$$
\text{total work}
=
\text{training work}
+
\text{sampling work}
+
\text{likelihood work}.
$$

---

## 3. 공통 수학적 기준

### 3.1 Target, proposal과 payoff-tilted law

target measure를 `P`, proposal을 `Q_θ`, bounded nonnegative payoff/event functional을
`G_ζ`라 하자.

$$
\mu_\zeta=E_P[G_\zeta].
$$

importance-sampling likelihood는 다음이다.

$$
L_\theta=\frac{dP}{dQ_\theta}.
$$

최종 estimator는 다음 형태를 유지한다.

$$
\widehat\mu_\zeta
=
\frac1n\sum_{m=1}^n
G_\zeta(X^{(m)})L_\theta^{(m)},
\qquad X^{(m)}\sim Q_\theta.
$$

payoff-tilted zero-variance law는 다음으로 정의한다.

$$
\frac{dQ_\zeta^\star}{dP}
=
\frac{G_\zeta}{\mu_\zeta}.
$$

`Q* << Q_θ`와 필요한 적분가능성이 성립하면 다음 identity를 얻는다.

$$
\frac{\operatorname{Var}_{Q_\theta}(G_\zeta L_\theta)}
{\mu_\zeta^2}
=
\chi^2(Q_\zeta^\star\Vert Q_\theta)
=
\exp\left(D_2(Q_\zeta^\star\Vert Q_\theta)\right)-1.
$$

이 identity 자체는 새로운 정리로 주장하지 않는다. 새로운 기여는 각 모델의 구조적
근사오차가 우변을 어떻게 제어하는지 밝히는 데 있다.

### 3.2 현재 Girsanov convention

저장소 convention을 유지하면:

$$
dW_t^P=dW_t^Q+u_tdt,
$$

$$
\log L
=
-\int_0^T u_t^\top dW_t^Q
-\frac12\int_0^T\|u_t\|^2dt.
$$

모든 새 모델과 theorem은 이 sign convention과 independent Brownian basis를 사용해야
한다. correlated price/volatility noise에 직접 control을 출력한 뒤 별도 식으로
변환하는 구현은 허용하지 않는다.

### 3.3 후보 평가 기준

| 기준 | 질문 | 중요도 |
|---|---|---:|
| 독창성 | 기존 요소의 이름 변경이 아니라 새로운 measure family/원리가 있는가? | 25% |
| 이론성 | estimator correctness와 효율을 정리로 연결할 수 있는가? | 20% |
| 구현 적합성 | 현재 코드와 점진적으로 통합 가능한가? | 20% |
| 실험 우위 | 강한 baseline을 이길 합리적 mechanism이 있는가? | 15% |
| 실용성 | 반복 질의와 총 계산비용에서 이점이 있는가? | 10% |
| 범용성 | 금융 밖의 non-Markovian rare-event 문제로 확장 가능한가? | 10% |

---

## 4. 선행연구 경계와 novelty 조건

다음 요소는 단독으로 신규 기여가 아니다.

1. Path-integral control과 adaptive importance sampling의 연결
2. PICE로 parameterized feedback controller를 학습하는 방법
3. neural drift를 사용한 option importance sampling
4. Neural RDE 또는 signature를 사용한 path-dependent control/PPDE 근사
5. rough kernel의 SOE/Markovian approximation
6. normalizing flow를 사용한 rare-event proposal
7. Schrödinger/Föllmer sampler
8. off-model data reweighting을 통한 반복 non-Markovian control
9. MLMC를 사용한 rough-volatility 가격 계산

따라서 아래 후보는 구성요소의 조합이 아니라 최소한 다음 중 두 가지 이상의 새로운
결합을 입증해야 한다.

- new causal path-measure family
- new structural Volterra memory decomposition
- new resolution-consistency principle
- new likelihood certificate or robustness guarantee
- new operator mapping model/payoff/kernel to control
- new approximation-error-to-variance theorem
- new multi-query work-complexity result

정식 논문에서 `first`, `unique`, `unprecedented`를 사용하기 전에는 별도의 systematic
literature review와 최신 preprint 검색을 다시 수행한다.

---

## 5. 후보 1 — Volterra–Föllmer Operator

### 5.1 이름과 연구 가설

> **VFO: Volterra–Föllmer Operator for Amortized Rare-Path Sampling**

핵심 가설은 다음이다.

> rough-volatility의 알려진 Volterra memory와 payoff가 유도하는 residual path
> memory를 분리한 causal operator가 generic Markov/RNN controller보다 적은
> approximation error로 payoff-tilted Föllmer drift를 근사하며, 이 개선이 동일 총
> 계산비용당 더 작은 relative variance로 이어진다.

### 5.2 Operator 정의

task와 model descriptor를 다음처럼 둔다.

$$
\zeta=
(H,\rho,\eta,\xi_0,T,K,B,\text{payoff type},\text{event parameters}).
$$

VFO는 다음 causal operator를 근사한다.

$$
\mathcal G_\theta:
(K_H,\zeta,\mathcal H_t)
\longmapsto
u_t^\theta\in\mathbb R^2.
$$

여기서 `K_H`는 rough kernel이며 `H_t`는 현재 시점까지의 Brownian/state history다.
한 strike의 optimal drift를 저장하는 network가 아니라 kernel, model과 payoff가 바뀔
때 대응하는 control law를 생성하는 operator를 목표로 한다.

### 5.3 Control 분해

$$
u_t^\theta
=
u_t^{\mathrm{inst}}
+u_t^{\mathrm{Volterra}}
+u_t^{\mathrm{res}}.
$$

#### Instantaneous branch

$$
u_t^{\mathrm{inst}}
=
f_{\theta,0}
(\tau_t,\log S_t,V_t,d_t^{\mathrm{event}},e_\zeta).
$$

`d_event`는 strike/barrier distance, running minimum/maximum, drawdown, realized
variance와 같은 causal event-progress feature다.

#### Structural Volterra branch

known kernel geometry를 반영하는 memory bank를 구성한다.

$$
M_t^{(k)}
=
\int_0^t K_{H,k}(t-s)dW_s.
$$

rBergomi에서 이 structural bank는 우선 rough variance를 구동하는 Brownian 방향에
적용한다. spot 전용 독립 방향의 과거가 payoff에 추가 정보를 주는 경우에는 그 정보는
residual/event branch에서 처리하고, 두 방향의 역할을 ablation으로 분리한다.

효율적 구현에서는 completely monotone SOE basis를 사용할 수 있다.

$$
K_H(r)\approx\sum_{k=1}^m w_k(H)e^{-\lambda_k(H)r},
\qquad w_k\ge0,\quad\lambda_k>0.
$$

각 state는 다음 recurrence를 갖는다.

$$
dM_t^{(k)}=-\lambda_k M_t^{(k)}dt+dW_t.
$$

중요한 제한은 다음이다.

- SOE는 우선 controller feature를 위한 auxiliary representation이다.
- main target simulator는 controlled BLP discretization을 유지한다.
- kernel basis를 학습하더라도 positivity와 stability constraint를 강제한다.
- exact BLP memory와 SOE memory의 차이는 별도 ablation으로 측정한다.

#### Payoff-tilt residual branch

known Volterra state만으로 설명되지 않는 path information을 학습한다.

$$
R_{i+1}
=
F_{\theta,R}
(R_i,\Delta W_i,M_i,S_i,V_i,d_i^{\mathrm{event}},e_\zeta).
$$

generic RNN이 전체 기억을 처음부터 학습하게 하지 않고 다음 분해를 강제한다.

$$
\text{history representation}
=
\underbrace{M_i}_{\text{known rough memory}}
+
\underbrace{R_i}_{\text{payoff-tilt residual}}.
$$

이 residual branch는 처음에는 작은 GRU/TCN/causal state-space cell로 구현한다. 새
기여는 cell 이름이 아니라 structural/residual 분해와 measure-level 결과다.

### 5.4 Task conditioning

task embedding `e_ζ`는 단순 concatenation, FiLM 또는 constrained hypernetwork로
구현할 수 있다. 첫 버전은 해석과 안정성을 위해 FiLM을 우선한다.

$$
h_i'=\gamma(e_\zeta)\odot h_i+\beta(e_\zeta).
$$

held-out test는 interpolation과 extrapolation을 구분한다.

- interpolation: 학습 범위 내부의 보지 않은 strike/maturity/parameter
- mild extrapolation: 인접한 `H`, rarity, maturity
- structural OOD: 보지 않은 payoff type 또는 kernel family

structural OOD 성능은 첫 논문의 필수 성공 기준으로 두지 않는다.

### 5.5 Two-driver head

VFO는 independent Brownian basis에서 두 control을 출력한다.

$$
u_i^\theta=
\begin{pmatrix}
u_i^{(1)}\\
u_i^{(2)}
\end{pmatrix}.
$$

초기에는 안정성을 위해 bounded head를 사용한다.

$$
u_i^{(j)}=u_{\max,j}\tanh(a_i^{(j)}).
$$

다만 bounded drift가 hard conditional law를 정확히 표현한다고 주장해서는 안 된다.
`u_max` sensitivity와 action distribution을 반드시 보고한다.

### 5.6 학습 전략

단일 복합 loss를 처음부터 최적화하지 않고 다음 curriculum을 사용한다.

#### Stage A — oracle alignment

Gaussian과 Heston soft oracle에서 control의 방향과 크기를 회복한다.

#### Stage B — path-integral initialization

soft potential로 rare region에 접근하는 초기 proposal을 학습한다.

#### Stage C — PICE path-law projection

$$
\min_\theta
\mathrm{KL}(Q_\zeta^\star\Vert Q_\theta).
$$

off-policy sample을 사용할 때 behavior path를 target coordinate로 복원한 뒤 candidate
residual Brownian을 계산한다. behavior likelihood를 candidate likelihood와 중복해서
곱하지 않는다.

#### Stage D — second-moment refinement

$$
J_2(\theta)=E_{Q_\theta}[(G_\zeta L_\theta)^2].
$$

PI/PICE objective가 감소해도 `J2`가 악화될 수 있으므로 최종 estimator objective를
직접 refinement한다.

#### Stage E — hard target evaluation

soft temperature를 anneal할 수 있지만 최종 checkpoint 선택 이후에는 원래 hard
payoff로 frozen evaluation한다.

### 5.7 이론 목표

#### T-VFO-1: discrete controlled-BLP unbiasedness

선언한 discrete target과 adapted control에 대해:

$$
E_{Q_\theta}[G L_\theta]=E_P[G].
$$

#### T-VFO-2: error decomposition

적절한 exponential-moment 조건 아래 다음 형태를 목표로 한다.

$$
D_2(Q^\star\Vert Q_\theta)
\le C
\left(
\varepsilon_{\mathrm{kernel}}
+\varepsilon_{\mathrm{memory}}
+\varepsilon_{\mathrm{task}}
+\varepsilon_{\mathrm{optimization}}
\right).
$$

위 식은 목표 형태이지 이미 증명된 부등식이 아니다. 실제 proof에서 squared error,
exponential moment 또는 `exp(C epsilon)-1` 형태가 필요하면 그 결과를 그대로 사용하며,
사전에 선형 bound를 가정하지 않는다.

초기 정리는 continuous hard event 전체보다 다음 중 하나로 범위를 제한한다.

- positive bounded soft payoff
- finite discrete grid
- bounded controls
- reference class 안에 존재하는 optimal drift
- 필요한 exponential moments가 명시된 경우

#### T-VFO-3: operator generalization 또는 fallback

uniform task theorem이 너무 강하면 empirical Lipschitz/generalization study로 낮춘다.
근거 없는 universal approximation 문장을 논문의 핵심 정리로 사용하지 않는다.

### 5.8 필수 ablation

| Variant | Instant | Structural memory | Residual memory | Task conditioning |
|---|---:|---:|---:|---:|
| Markov MLP | O | X | X | X |
| Generic GRU | O | X | O | X |
| SOE-only | O | O | X | X |
| VFO per-task | O | O | O | X |
| VFO amortized | O | O | O | O |

### 5.9 성공·중단 기준

VFO를 주모델로 유지하려면 다음을 만족해야 한다.

1. 사전 지정 rough regime 3개 중 2개 이상에서 Markov baseline보다 paired
   work-normalized efficiency가 개선된다.
2. generic GRU와 비교해 적어도 성능, parameter/sample efficiency 또는 grid transfer
   중 하나에서 명확한 이점이 있다.
3. held-out task median efficiency가 per-task controller의 80% 이상이다.
4. likelihood mean, contribution tail과 ESS diagnostics가 안정적이다.

memory effect가 없으면 residual branch를 한 번만 재설계한다. 이후에도 개선이 없으면
memory-superiority claim을 폐기하고 controlled-BLP 및 operator/objective paper로
범위를 축소한다.

---

## 6. 후보 2 — Multiresolution Renormalized VPIC

### 6.1 이름과 가설

> **MR-VPIC: Multiresolution Renormalized Volterra Path-Integral Controller**

핵심 가설은 다음이다.

> time-grid마다 별도의 controller를 학습하는 대신 Brownian filtration의 coarse/fine
> 관계와 rough kernel scaling을 보존하는 controller를 학습하면, resolution transfer와
> multilevel rare-event estimation을 동시에 개선할 수 있다.

### 6.2 해결하려는 근본 문제

`N=64`에서 학습한 controller가 `N=256`에서 동일한 continuous-time control을
표현한다는 보장은 없다. rough kernel은 최근 과거에서 singular하므로 fine-scale
innovation이 control과 payoff에 큰 영향을 줄 수 있다.

grid-specific 성능만 보고하면 모델이 path law가 아니라 discretization artifact를
학습했을 가능성을 배제할 수 없다.

### 6.3 Causal Brownian hierarchy

fine와 coarse increment를 다음처럼 coupling한다.

$$
\Delta W_i^{(\ell)}
=
\Delta W_{2i}^{(\ell+1)}
+
\Delta W_{2i+1}^{(\ell+1)}.
$$

각 level의 controller는 그 level의 현재 filtration만 사용한다. Haar 또는 wavelet
feature를 사용할 경우 support가 아직 끝나지 않은 coefficient에 미래 increment가
섞이지 않도록 online causal construction을 사용한다.

### 6.4 Scale-specific control

$$
u_t^{(L)}
=
u_t^{(0)}+
\sum_{\ell=1}^L \delta u_t^{(\ell)}.
$$

`u^(0)`는 long-horizon rare-event route를, `delta u^(ell)`은 rough fine-scale correction을
담당한다. 모든 scale head는 task/kernel embedding을 공유한다.

### 6.5 Renormalization consistency

fine controller를 coarse filtration에 projection했을 때 coarse controller와 일치하도록
한다.

$$
u^{(\ell)}
\approx
E[P_\ell u^{(\ell+1)}\mid\mathcal F^{(\ell)}].
$$

실용적 loss는 coupled sample을 사용해 다음처럼 구성한다.

$$
\mathcal L_{\mathrm{RG}}
=
\sum_{\ell=0}^{L-1}
\left\|
u^{(\ell)}-operatorname{stopgrad}
(\widehat P_\ell u^{(\ell+1)})
\right\|^2.
$$

projection target의 bias와 conditional-expectation approximation error를 문서화한다.
단순 interpolation consistency를 renormalization theorem이라고 부르지 않는다.

### 6.6 Multilevel importance sampling

각 level의 marginal proposal이 올바른 likelihood를 갖도록 coupling한 뒤:

$$
E_{P_L}[G_L]
=
E_{P_0}[G_0]
+
\sum_{\ell=1}^L
E[G_\ell-G_{\ell-1}].
$$

level correction estimator는 예를 들어:

$$
Y_\ell
=
G_\ell L_\ell-G_{\ell-1}L_{\ell-1}
$$

로 구성한다. fine/coarse path가 같은 joint random seed를 사용해도 각 marginal의
likelihood는 별도로 올바르게 계산해야 한다.

### 6.7 이론 목표

- grid refinement에 따른 control projection error bound
- kernel singularity와 scale truncation error의 관계
- controlled MLMC correction variance rate
- target tolerance `ε`에 대한 total complexity
- 가능하면 randomized-level debiasing을 통한 continuous-target 확장

마지막 항목은 stretch goal이다. 첫 논문에서는 finest declared discretization에 대한
정확한 estimator와 discretization convergence study로 충분할 수 있다.

### 6.8 혁신성과 위험

MLMC와 rough-volatility 결합 자체는 기존 연구다. 새 기여는 **하나의 learned causal
control law가 여러 resolution에서 measure-consistent하게 작동하고, multilevel
correction variance를 줄이는 구조와 정리**다.

주요 위험은 다음이다.

- discontinuous payoff로 인해 level variance rate가 나빠질 수 있음
- coarse/fine control 차이가 likelihood variance를 오히려 키울 수 있음
- causal wavelet 구현에서 look-ahead leakage가 발생할 수 있음
- 학습비용이 single-level 이득을 상쇄할 수 있음

VFO가 single-level에서 먼저 유효하다는 증거 없이 MR-VPIC를 시작하지 않는다.

---

## 7. 후보 3 — Doob–Volterra Desirability Network

### 7.1 이름과 가설

> **DVDN: Doob–Volterra Desirability Network**

핵심 가설은 다음이다.

> optimal control을 임의의 vector output으로 직접 학습하는 대신, positive
> desirability martingale과 그 martingale integrand를 함께 학습하면 시간 일관성과
> path-integral 구조가 개선되고 control approximation error가 감소한다.

### 7.2 Desirability process

positive payoff에 대해:

$$
\Psi_t=E_P[G\mid\mathcal F_t].
$$

martingale representation으로:

$$
d\Psi_t=Z_t^\top dW_t^P.
$$

일반적인 density-process convention에서 optimal Föllmer drift는 다음 구조를 갖는다.

$$
u_t^\star=\frac{Z_t}{\Psi_t}.
$$

실제 코드 sign은 저장소의 `dW^P=dW^Q+u dt` convention과 oracle test로 고정한다.

### 7.3 모델 구조

$$
(\log \Psi_i^\theta,Z_i^\theta)
=
\mathcal D_\theta
(t_i,S_i,V_i,M_i^{\mathrm{Volterra}},R_i,e_\zeta).
$$

positive constraint는 다음으로 보장한다.

$$
\Psi_i^\theta=\exp(f_\theta(\cdot)).
$$

control은:

$$
u_i^\theta
=
\frac{Z_i^\theta}{\Psi_i^\theta+\varepsilon_\Psi}.
$$

`epsilon_Psi`는 수치 안정화 parameter이며 결과가 이 값에 민감하지 않은지 검사한다.

### 7.4 Martingale consistency

discrete martingale residual은 다음이다.

$$
r_i
=
\Psi_{i+1}^\theta-\Psi_i^\theta-(Z_i^\theta)^\top\Delta W_i^P.
$$

학습 loss는:

$$
\mathcal L_{\mathrm{DVDN}}
=
\lambda_T\|\Psi_T^\theta-G\|^2
+\lambda_\Psi\sum_i\|r_i\|^2
+\lambda_u\log\widehat J_2
+\lambda_C\|u_i^{\mathrm{direct}}-Z_i/\Psi_i\|^2.
$$

마지막 항은 direct VFO head와 desirability-derived control을 함께 사용할 때만 추가한다.

### 7.5 가능한 장점

- 모든 시점의 conditional desirability를 학습
- arbitrary control output을 positive density process로 제약
- control의 시간 일관성 진단 가능
- event probability calibration과 IS control을 하나의 모델에서 분석 가능
- martingale residual과 estimator efficiency의 관계를 연구 가능

### 7.6 Hard indicator singularity

hard event에서는 `G`가 0일 수 있으므로 `log Psi_T`와 `Z/Psi`가 singular해질 수 있다.
따라서 첫 구현은 다음 순서를 따른다.

1. strictly positive soft payoff
2. lower floor가 있는 desirability
3. temperature continuation
4. 원래 hard payoff에서 raw IS evaluation

soft floor를 적용한 target과 원래 hard target을 같은 measure라고 주장하지 않는다.

### 7.7 이론 목표와 novelty 위험

목표 정리는 다음이다.

- martingale residual이 density-process error를 제어하는 조건
- desirability/control error에서 `D2` 또는 second moment로 가는 bound
- positive soft target에서 DVDN consistency

Deep BSDE, Doob transform, path-dependent PDE 및 Neural RDE 자체는 기존 연구다.
따라서 DVDN은 독립 주모델보다 VFO의 **path-measure consistency mechanism**으로
사용할 때 가장 방어 가능하다.

---

## 8. 후보 4 — Causal Path-Space Transport

### 8.1 이름과 가설

> **CAPT: Causal Adaptive Path Transport**

핵심 가설은 다음이다.

> fixed-covariance Gaussian drift proposal보다 causal invertible transport가
> multi-modal, skewed 또는 heavy-tailed payoff-tilted innovation law를 더 잘 표현해
> weight degeneracy를 줄일 수 있다.

### 8.2 Discrete causal triangular map

base innovation을:

$$
z_{1:N}\sim\mathcal N(0,I)
$$

로 두고 다음 lower-triangular transform을 학습한다.

$$
\widetilde z_i
=
T_{\theta,i}
(z_i;\widetilde z_{1:i-1},e_\zeta).
$$

`T_i`는 monotone spline 또는 constrained affine-spline transform이 될 수 있다.
Jacobian은 triangular이므로:

$$
\log|\det J_T|
=
\sum_i\log
\left|
\frac{\partial T_{\theta,i}}{\partial z_i}
\right|.
$$

proposal density는:

$$
q_\theta(\widetilde z)
=
\phi(T_\theta^{-1}(\widetilde z))
\left|\det J_{T^{-1}}(\widetilde z)\right|.
$$

따라서 declared discrete Gaussian innovation target `p`에 대해:

$$
\widehat\mu
=
G(X(\widetilde z))
\frac{p(\widetilde z)}{q_\theta(\widetilde z)}.
$$

### 8.3 Drift proposal보다 넓은 표현력

Girsanov drift head는 conditional Gaussian mean을 주로 이동시킨다. CAPT는 discrete
innovation의 conditional:

- scale
- skewness
- kurtosis
- tail shape
- multiple modes

까지 표현할 수 있다.

### 8.4 Defensive mixture

proposal misspecification을 완화하기 위해:

$$
q_{\theta,\epsilon}
=(1-\epsilon)q_\theta+\epsilon p,
\qquad \epsilon>0.
$$

그러면:

$$
q_{\theta,\epsilon}\ge\epsilon p,
\qquad
\frac{p}{q_{\theta,\epsilon}}\le\frac1\epsilon.
$$

bounded payoff이면 weighted contribution도 다음으로 제한된다.

$$
|G p/q_{\theta,\epsilon}|
\le
\|G\|_\infty/\epsilon.
$$

이는 단순 weight clipping과 달리 proposal 자체를 바꾸면서 exact mixture likelihood를
사용하므로 estimator bias를 만들지 않는다.

### 8.5 가장 중요한 이론적 제한

innovation scale을 바꾸는 discrete flow는 fixed grid에서 정확한 density ratio를 가질
수 있다. 그러나 continuous time에서 diffusion의 quadratic variation을 바꾸면 target
Wiener measure와 proposal이 singular해질 수 있다.

따라서 다음 주장을 엄격히 분리한다.

| 범위 | 가능한 주장 |
|---|---|
| fixed discrete grid | exact change-of-variables likelihood와 unbiasedness |
| drift-only continuous proposal | Girsanov absolute continuity 가능 |
| scale-changing continuous flow | 일반적으로 Girsanov로 정당화 불가능 |
| continuous adapted Wiener transport | 별도의 quasi-invariance 이론 필요 |

CAPT fixed-grid 결과를 continuous-time path-measure theorem으로 과장하면 치명적인
이론 오류가 된다.

### 8.6 추진 gate

다음 증거 중 둘 이상이 있을 때만 CAPT를 개발한다.

- VFO weight histogram이 안정적으로 multi-modal
- 여러 seed가 서로 다른 proposal mode에 수렴
- bounded drift cap을 늘려도 tail ESS가 개선되지 않음
- single Gaussian drift family의 conditional residual이 명확히 non-Gaussian
- defensive mixture를 사용해도 VFO가 rare regime에서 collapse

그 전에는 CAPT의 높은 표현력이 불필요한 복잡성일 가능성이 크다.

---

## 9. 후보 5 — Multimodal Schrödinger–Volterra Bridge

### 9.1 이름과 가설

> **MSVB: Multimodal Schrödinger–Volterra Bridge**

핵심 가설은 다음이다.

> 하나의 rare event가 여러 dominating path mechanism을 가질 때, 단일 control보다
> multiple entropic path bridges의 mixture가 payoff-tilted path law를 더 잘 덮고
> mode collapse를 방지한다.

### 9.2 Mixture path law

latent route `C`를 도입한다.

$$
Q_\theta
=
\sum_{k=1}^K\pi_k(\zeta)Q_{\theta,k}.
$$

각 expert는 다른 adapted two-driver control을 갖는다.

$$
dW_t^{Q_k}=dW_t^P-u_{\theta,k}(t,\mathcal H_t,\zeta)dt.
$$

canonical path `x`에서 mixture likelihood는 각 expert density를 모두 평가해:

$$
\frac{dP}{dQ_\theta}(x)
=
\left[
\sum_{k=1}^K
\pi_k(\zeta)
\frac{dQ_{\theta,k}}{dP}(x)
\right]^{-1}.
$$

샘플을 생성한 expert의 likelihood만 사용하는 것은 잘못된 mixture estimator가 될 수
있다. 모든 expert 아래에서 같은 canonical target path의 residual Brownian을 올바르게
복원해야 한다.

### 9.3 Schrödinger bridge 목적

reference rough path law `P`에 대해 softened endpoint/path constraint를 만족하는 최소
relative-entropy law를 찾는다.

$$
\min_Q \mathrm{KL}(Q\Vert P)
$$

subject to terminal 또는 path-event constraints.

각 expert가 서로 다른 entropic route를 담당하고 gating network가 task에 따라 mixture
weight를 결정한다.

### 9.4 Route discovery

처음부터 임의 expert를 여러 개 두면 label switching과 collapse가 발생한다. 다음
순서를 사용한다.

1. VFO 여러 seed에서 successful rare paths 수집
2. path summary 또는 action profile로 mechanism clustering
3. stable cluster가 있는지 독립 seed로 확인
4. 확인된 route 수만큼 expert 초기화
5. entropy/coverage regularizer와 exact mixture likelihood로 joint refinement

### 9.5 성공 조건과 위험

MSVB는 다음이 입증되어야 한다.

- 하나의 event에 복수의 재현 가능한 dominating route가 존재
- single VFO가 route 일부를 놓침
- mixture가 tail coverage와 ESS를 개선
- expert 수 증가에 따른 계산비용을 포함해도 우수
- 각 expert route에 금융적 또는 확률적 해석이 가능

Schrödinger/Föllmer bridge 자체는 신규성이 아니다. multi-route rough path mechanism과
정확한 mixture likelihood가 실제 필요하다는 증거가 없으면 이 후보는 rebranding으로
평가될 위험이 높다.

---

## 10. 후보 6 — Universal Path-Measure Compiler

### 10.1 장기 목표

> **UPMC: Universal Path-Measure Compiler**

UPMC는 다음 mapping을 목표로 한다.

$$
(\text{dynamics},\text{memory kernel},\text{event},\text{tolerance})
\longmapsto
(\text{causal sampler},\text{likelihood},\text{certificate}).
$$

사용자는 stochastic system과 rare event를 기술하고, compiler는 다음을 출력한다.

- executable causal proposal
- exact 또는 검증 가능한 likelihood evaluator
- estimator와 confidence interval
- support/absolute-continuity 진단
- OOD 및 weight-collapse 경고
- 목표 오차에 대한 sample/work allocation

### 10.2 필요한 구성요소

1. dynamics/kernel encoder
2. payoff/path-functional encoder
3. VFO-style causal operator
4. optional transport/bridge experts
5. symbolic or typed likelihood compiler
6. martingale and normalization tests
7. adaptive computation-budget allocator
8. failure certificate와 fallback sampler

### 10.3 왜 가장 혁신적이지만 성공 가능성은 낮은가

diffusion, jump process, point process와 hybrid system은 measure change가 서로 다르다.

- diffusion drift: Girsanov
- diffusion covariance change: continuous-time singularity 가능
- jump intensity change: point-process likelihood 필요
- support-changing transform: absolute continuity 검토 필요
- discretized black-box simulator: tractable density가 없을 수 있음

범용 compiler는 architecture 문제가 아니라 확률론, programming language, numerical
analysis와 scientific machine learning을 함께 다루는 장기 프로그램이다.

### 10.4 현실적인 확장 경로

$$
\text{rBergomi VFO}
\rightarrow
\text{Volterra-family VFO}
\rightarrow
\text{non-Markovian diffusion operator}
\rightarrow
\text{typed path-measure compiler}.
$$

첫 논문에서 UPMC를 구현하려고 하지 않는다. 대신 VFO API와 likelihood contract를
나중에 compiler로 일반화할 수 있도록 설계한다.

---

## 11. 후보군 비교와 선택 논리

### 11.1 표현력과 이론 위험

| 후보 | Proposal family | Continuous-time 해석 | Exact discrete likelihood | Multi-mode 표현 | 이론 난도 |
|---|---|---:|---:|---:|---:|
| VFO | adapted drift | 강함 | 가능 | 제한적 | 중간 |
| MR-VPIC | multiscale adapted drift | 강함 | 가능 | 제한적 | 중상 |
| DVDN | density-process-derived drift | 강함 | 가능 | 제한적 | 높음 |
| CAPT | causal nonlinear transport | 제한적 | 가능 | 강함 | 높음 |
| MSVB | mixture of adapted drifts | 강함 | 가능 | 강함 | 매우 높음 |
| UPMC | heterogeneous | 문제별 상이 | 문제별 상이 | 매우 강함 | 극히 높음 |

### 11.2 실무성과 연구성

| 후보 | 반복 질의 | Grid transfer | 실패 진단 | 계산비용 | 첫 논문 적합성 |
|---|---:|---:|---:|---:|---:|
| VFO | 강함 | 보통 | 강함 | 보통 | 가장 높음 |
| MR-VPIC | 강함 | 가장 강함 | 강함 | 높음 | 높음 |
| DVDN | 강함 | 보통 | 매우 강함 | 높음 | 중상 |
| CAPT | 강함 | 약함 | 강함 | 높음 | 중간 |
| MSVB | 강함 | 보통 | 중간 | 매우 높음 | 중하 |
| UPMC | 이론상 가장 강함 | 이론상 강함 | 목표상 가장 강함 | 극히 높음 | 낮음 |

### 11.3 왜 VFO가 1위인가

VFO는 완전히 안전한 선택이라서 1위가 아니다. 다음 균형이 가장 좋기 때문이다.

- 현재 Girsanov 및 two-driver 기반을 그대로 활용
- exact controlled rBergomi로 이동 가능
- SOE를 기존 모델 복제가 아닌 structural feature로 재해석
- residual memory와 task operator에서 새 모델을 정의 가능
- `D2`/relative variance theorem과 직접 연결 가능
- 실패 시에도 controlled-BLP, objective와 benchmark 자산이 남음

### 11.4 왜 가장 혁신적인 UPMC가 1위가 아닌가

혁신 잠재력과 성공확률은 다르다. UPMC는 범위가 넓어 핵심 theorem, 구현 correctness,
실험적 우위를 한 논문에서 동시에 확보하기 어렵다. 현재는 VFO를 성공시켜 UPMC의
첫 번째 typed module을 만드는 편이 장기적으로도 더 빠르다.

---

## 12. 권고 flagship — MR-VFO with Desirability Consistency

### 12.1 최종 모델 정의

권고 flagship은 다음 세 층으로 구성한다.

1. VFO structural/residual causal encoder
2. optional multiresolution consistency
3. optional desirability/martingale consistency

직접 control은:

$$
u_i^{\mathrm{VFO}}
=
\pi_\theta
(t_i,S_i,V_i,M_i,R_i,e_\zeta).
$$

desirability-derived control은:

$$
u_i^{\mathrm{DVDN}}
=
\frac{Z_i^\theta}{\Psi_i^\theta+\varepsilon_\Psi}.
$$

두 표현의 consistency는:

$$
\mathcal L_{\mathrm{control-consistency}}
=
\sum_i
\|u_i^{\mathrm{VFO}}-u_i^{\mathrm{DVDN}}\|^2.
$$

multiresolution consistency는:

$$
\mathcal L_{\mathrm{RG}}
=
\sum_\ell
\|u^{(\ell)}-\widehat P_\ell u^{(\ell+1)}\|^2.
$$

### 12.2 전체 loss 후보

$$
\mathcal L
=
\lambda_{\mathrm{PI}}\mathcal L_{\mathrm{PI}}
+\lambda_{\mathrm{PICE}}\mathcal L_{\mathrm{PICE}}
+\lambda_2\log\widehat J_2
+\lambda_\Psi\mathcal L_{\mathrm{mart}}
+\lambda_C\mathcal L_{\mathrm{control-consistency}}
+\lambda_{\mathrm{RG}}\mathcal L_{\mathrm{RG}}
+\lambda_A\mathcal R_{\mathrm{action}}.
$$

하지만 이 loss를 처음부터 한꺼번에 사용하지 않는다.

| 구현 단계 | 활성 loss |
|---|---|
| oracle 검증 | supervised oracle + action regularization |
| VFO 초기화 | PI |
| path-law matching | PI + PICE |
| estimator 최적화 | PICE + `J2` |
| DVDN 확장 | martingale + consistency + `J2` |
| MR 확장 | RG + coupled-level `J2` |

각 loss를 추가할 때 이전 gate 성능과 likelihood diagnostics가 악화되지 않는지
확인한다.

### 12.3 논문에서 주장할 수 있는 핵심 문장 후보

다음 문장은 모든 실험과 정리가 성공한 뒤에만 사용할 수 있다.

> We introduce a kernel- and task-conditioned Volterra–Föllmer operator that
> separates structural rough memory from payoff-induced residual memory,
> produces exact likelihood-corrected two-driver proposals for a discretized
> rough-volatility model, and links operator approximation to work-normalized
> rare-event estimation error.

MR 결과까지 성공하면 다음을 추가할 수 있다.

> A multiresolution consistency principle transfers the learned control across
> time grids and reduces controlled multilevel correction variance.

---

## 13. 단계별 연구개발 계획

### Phase M0 — Claim freeze와 후보 공통 contract

목표:

- common controller interface
- Brownian basis와 likelihood convention 고정
- task descriptor schema 고정
- future-information guard
- discrete-target claim 문구 고정

완료 조건:

- 모든 controller가 동일 simulator/evaluator를 사용
- proposal과 target coordinate reconstruction test 통과
- `E_Q[L]≈1`과 null-control pathwise identity 통과

### Phase M1 — Minimal VFO on Heston

목표:

- structural branch 없이 instantaneous + residual skeleton 구현
- two-driver oracle direction 회복
- PI/PICE/`J2` curriculum 검증

Heston에서 새로운 성능 주장을 만들지 않는다. 학습 pipeline correctness를 확인한 뒤
즉시 동결한다.

### Phase M2 — Exact controlled rBergomi

목표:

- controlled BLP path construction
- same-operator memory correction
- independent two-driver likelihood
- pathwise target reconstruction
- block-online feedback 가능성 확인

완료 조건:

- null-control base path exact match
- path reconstruction test
- likelihood normalization
- controlled/uncontrolled estimate CI agreement
- refinement convergence

### Phase M3 — VFO structural/residual prototype

구현 순서:

1. fixed SOE structural bank
2. residual state 없는 SOE-only controller
3. small residual state 추가
4. two-driver full head
5. per-task PI/PICE/`J2`
6. Markov/GRU/SOE ablation

이 단계에서 memory improvement가 없으면 residual branch를 한 번만 재설계한다.

### Phase M4 — Kernel/task operator

목표:

- `H`, `rho`, maturity, rarity와 payoff descriptor conditioning
- held-out interpolation test
- per-task vs amortized work comparison
- local fine-tuning 필요성 평가

### Phase M5 — 이론 T-VFO

실험과 병행한다.

- discrete unbiasedness proposition
- error decomposition lemma
- control error to `D2` bound
- memory/kernel approximation corollary
- theorem assumption diagnostics

### Phase M6 — Multiresolution pilot

작은 두 level에서 시작한다.

- coupled `N`/`2N` paths
- causal projection
- controller transfer
- correction variance
- likelihood tail

두 level pilot이 실패하면 full MR-VPIC를 중단한다.

### Phase M7 — DVDN pilot

positive soft payoff에서:

- desirability calibration
- martingale residual
- `Z/M` control
- direct VFO control과 alignment
- hard-event transfer

DVDN이 VFO의 estimator 성능 또는 안정성을 개선하지 못하면 논문 본모델에서 제거하고
진단 도구로만 남긴다.

### Phase M8 — CAPT/MSVB 선택 gate

VFO residual 분석 후 둘 중 필요한 것만 선택한다.

- non-Gaussian single-mode residual: CAPT
- distinct multiple path modes: MSVB
- 둘 다 없음: 확장하지 않음

### Phase M9 — Sealed benchmark와 calibrated application

모델 선택과 hyperparameter tuning을 종료한 후 frozen seed에서 수행한다.

주 application:

- deep-OTM digital tail
- barrier hit probability
- drawdown 또는 path-dependent loss
- repeated strike/maturity/barrier surface

vanilla option은 correctness 및 calibration 보조 실험으로 사용한다.

---

## 14. Falsification-first gate

| Gate | 질문 | 통과 기준 | 실패 시 조치 |
|---|---|---|---|
| G1 Oracle | 학습 pipeline이 알려진 optimal direction을 회복하는가? | Gaussian/Heston 방향·`J2` 개선 | objective/sign 수정 |
| G2 rBergomi law | controlled path와 likelihood가 정확한가? | pathwise·normalization·CI tests | 성능실험 금지 |
| G3 Memory | structural/residual memory가 필요한가? | 3 regime 중 2개 개선 | memory claim 축소 |
| G4 Operator | amortized held-out task가 유지되는가? | per-task median의 80% 이상 | per-task paper로 축소 |
| G5 Work | training 포함 break-even이 있는가? | 사전 지정 query 수 이하 | 실무 claim 제거 |
| G6 Resolution | grid transfer가 실제로 필요한가? | single-grid 성능 저하 확인 | MR 확장 생략 |
| G7 DVDN | martingale head가 안정성을 높이는가? | seed variance/ESS/`J2` 개선 | auxiliary 진단으로 축소 |
| G8 Expressivity | drift family가 부족한가? | multi-mode/non-Gaussian 증거 | CAPT/MSVB 생략 |

이 gate의 목적은 모든 후보를 끝까지 구현하는 것이 아니다. 가장 단순한 모델로 설명
가능한 결과에 불필요한 복잡성을 추가하지 않기 위함이다.

---

## 15. Benchmark와 통계 설계

### 15.1 Baseline

최소 baseline은 다음이다.

1. crude Monte Carlo
2. antithetic/conditional rough-Bergomi Monte Carlo
3. constant drift/CEM
4. Markov affine/MLP
5. direct `J2` neural importance sampling
6. generic GRU/TCN memory controller
7. fixed SOE/Markov-lift controller
8. PICE feedback
9. applicable한 경우 turbocharging 또는 강한 rough-volatility variance reduction
10. binary rare event에서 adaptive multilevel splitting 비교

모든 baseline이 같은 batch 크기나 같은 parameter 수를 가져야 하는 것은 아니다.
대신 다음 두 비교를 모두 제공한다.

- matched model capacity
- matched total wall-clock work

### 15.2 Regime grid

| 축 | 예시 범위 | 목적 |
|---|---|---|
| roughness `H` | low/mid/high rough regime | memory 효과 |
| correlation `rho` | main `rho<0` range | leverage와 martingale 안전성 |
| maturity | short/medium/long | history length |
| rarity | `1e-2` to 가능한 더 작은 확률 | rare-event scaling |
| payoff | terminal/path-dependent | memory necessity |
| resolution | `N`, `2N`, `4N` | grid transfer |

`rho>0`은 rBergomi martingale 문제를 별도 분석한 뒤 extension으로만 다룬다.

### 15.3 핵심 지표

- raw estimator mean, SE와 confidence interval
- relative variance
- variance-reduction factor
- effective sample size
- log-weight quantiles와 maximum contribution share
- `E_Q[L]`
- work-normalized variance
- training-inclusive break-even query count
- memory rank/kernel approximation error
- controller action energy
- seed-to-seed dispersion
- grid-transfer degradation
- theorem assumption proxy

### 15.4 통계 protocol

- train/selection/audit/sealed-evaluation seed 분리
- paired Brownian seeds로 baseline 비교
- 사전 지정 primary metric 하나 사용
- multiple regime 결과는 median만으로 숨기지 않고 전체 공개
- mean뿐 아니라 paired confidence interval 보고
- negative result와 gate failure도 기록
- evaluation seed를 본 뒤 tuning 금지

---

## 16. 구현 구조 제안

현재 코드와 충돌을 줄이기 위해 다음 구조를 권고한다.

```text
src/path_integral/
  controllers/
    protocol.py
    markov.py
    vfo.py
    desirability.py
    multiresolution.py
    causal_transport.py
    mixture_bridge.py
  memory/
    volterra_bank.py
    soe_bank.py
    residual_state.py
    causal_pyramid.py
  training/
    pi_trainer.py
    pice_trainer.py
    second_moment.py
    martingale_trainer.py
    multilevel_trainer.py
  rbergomi/
    controlled_blp.py
    reconstruction.py
    likelihood.py
  evaluation/
    estimator.py
    work_accounting.py
    diagnostics.py
```

### 16.1 Controller protocol

모든 controller는 최소한 다음 입력을 받는다.

```python
control = controller(
    time_state,
    market_state,
    memory_state,
    event_state,
    task_descriptor,
)
```

출력은 independent Brownian basis의 `[batch, 2]` control이다.

### 16.2 Causal-state contract

각 memory module은:

- `initialize(task, batch_size)`
- `observe(current_increment, current_state)`
- `features()`
- `detach_or_checkpoint()`

형태의 online interface를 갖는다. 전체 미래 tensor를 module에 전달하지 않는다.

### 16.3 Likelihood contract

각 proposal은 다음을 제공한다.

- proposal sample generation
- canonical target-coordinate path reconstruction
- `log_dP_dQ`
- null-control/base-law equivalence test
- support/absolute-continuity declaration

CAPT와 mixture proposal은 Girsanov proposal과 다른 likelihood engine을 사용하므로 type
수준에서 구분한다.

---

## 17. 기술적·이론적 위험 등록부

| 위험 | 영향 | 탐지 | 완화 |
|---|---|---|---|
| 미래 정보 leakage | 성능 전체 무효 | prefix invariance test | online state API |
| Girsanov sign 오류 | biased estimator | Gaussian/Heston oracle | convention 단일화 |
| behavior/candidate coordinate 혼동 | PICE 오류 | off-policy reconstruction test | canonical path reconstruction |
| soft/hard target 혼동 | 잘못된 성능 claim | separate evaluation | hard sealed metric |
| SOE target bias 은폐 | 가격 bias | BLP vs lift comparison | feature/target 분리 |
| action explosion | weight collapse | log-weight/action diagnostics | bounded curriculum/mixture |
| bounded control bias 오해 | 과장된 optimality | cap sensitivity | approximation claim 제한 |
| CAPT continuous singularity | 이론 무효 | quadratic-variation audit | fixed-grid claim |
| mixture likelihood 오계산 | biased estimator | exact small-mixture oracle | 모든 expert density 평가 |
| indicator desirability singularity | NaN/불안정 | soft-to-hard tests | positive curriculum |
| amortization OOD collapse | 실무성 상실 | held-out/OOD split | local fallback/failure flag |
| training cost 누락 | 허위 speedup | total work ledger | break-even 보고 |
| multiple testing/tuning | 과대평가 | sealed protocol audit | seed freeze |

---

## 18. 예상 논문 포지셔닝

### 18.1 VFO만 성공한 경우

주요 기여:

- exact controlled rBergomi discretization
- structural/residual Volterra controller
- task-conditioned path-integral IS
- approximation/variance proposition
- strong work-normalized benchmark

적합한 논문 성격은 mathematical/computational finance다.

### 18.2 MR-VFO까지 성공한 경우

추가 기여:

- resolution-consistent stochastic control
- controlled multilevel estimator
- grid-transfer 및 complexity result

finance를 넘어 scientific computing 또는 computational physics 독자층까지 확장할 수
있다.

### 18.3 DVDN theorem까지 성공한 경우

추가 기여:

- positive desirability martingale representation
- learned density process와 Föllmer drift consistency
- martingale residual에서 estimator efficiency로 가는 bound

이 경우 이론 금융·확률·stochastic control 쪽 포지셔닝이 강해진다.

### 18.4 CAPT/MSVB까지 필요한 경우

표현력 확장은 standalone novelty가 아니라 VFO가 실패한 구체적 path-law geometry를
해결한다는 narrative가 필요하다. 그렇지 않으면 architecture stacking으로 보일 위험이
크다.

---

## 19. 선행연구 대비 명시적 차별화 표

| 기존 분야 | 기존에 이미 있는 것 | 본 연구가 추가해야 하는 것 |
|---|---|---|
| PICE | parameterized feedback와 adaptive IS | rough-path operator와 variance theorem |
| Neural option IS | neural drift와 option benchmark | structural rough memory, task operator, exact BLP law |
| Neural RDE/PPDE | history encoding과 non-Markovian value approximation | likelihood-corrected rare-path proposal와 work metric |
| Markovian approximation | SOE/lift와 rough simulation | lift-as-feature, residual memory, estimator-error link |
| Off-model control | reweighting과 repeated recalibration | kernel/payoff-to-control operator와 rough rare-event study |
| Normalizing-flow rare events | flexible tractable proposal | causal Volterra path transport와 discrete/continuous audit |
| Schrödinger bridge | entropic path transport | evidence-driven multi-route rough path mixture |
| Rough-vol MLMC | multilevel pricing complexity | learned resolution-consistent control and likelihood coupling |

---

## 20. 최종 실행 결정

### 즉시 채택

- VFO를 차세대 main model로 채택한다.
- SOE memory controller는 최종 모델이 아니라 structural-memory baseline/branch로
  재정의한다.
- exact controlled rBergomi가 완성되기 전에는 새 architecture의 성능 claim을 만들지
  않는다.
- first theorem은 discrete positive target과 bounded/adapted control 범위에서 시작한다.

### 조건부 채택

- MR-VPIC: VFO의 grid dependence가 확인되면 채택
- DVDN: direct control의 시간 일관성/학습 안정성을 개선하면 채택
- CAPT: drift proposal의 non-Gaussian expressivity 한계가 확인되면 채택
- MSVB: 여러 stable rare-path mode가 확인되면 채택

### 첫 논문에서 보류

- UPMC 전체 구현
- diffusion covariance를 바꾸는 continuous-time flow claim
- jump model까지 한 번에 포함하는 범용 extension
- 모든 payoff와 kernel family에 대한 universal generalization claim

### 최종 한 문장

> 본 연구의 가장 성공 가능성이 높은 새로운 모델은 rough kernel의 알려진 구조와
> payoff가 만드는 residual path memory를 분리하고, model/payoff task에서 정확한
> two-driver Föllmer control로 가는 operator를 학습하는 VFO다. 가장 강한 확장은 이
> operator에 multiresolution consistency를 부여하는 MR-VFO이며, desirability,
> nonlinear transport와 bridge mixture는 관측된 실패 mechanism이 요구할 때만
> 추가한다.

---

## 21. 참고문헌과 novelty 감시 목록

아래 문헌은 모델 설계의 직접적인 novelty boundary다. 논문 작성 시 최신 버전과 정식
출판 여부를 다시 확인한다.

1. Kappen, H. J. and Ruiz, H. C., *Adaptive Importance Sampling for Control and
   Inference*, Journal of Statistical Physics, 2016.
   <https://link.springer.com/article/10.1007/s10955-016-1446-7>

2. *Importance sampling for option pricing with feedforward neural networks*,
   Finance and Stochastics, 2024.
   <https://link.springer.com/article/10.1007/s00780-024-00549-x>

3. *A Neural RDE approach for continuous-time non-Markovian stochastic control
   problems*, 2023.
   <https://arxiv.org/abs/2306.14258>

4. Fang, Ni and Wu, *A Neural RDE-based model for solving path-dependent PDEs*,
   2023.
   <https://arxiv.org/abs/2306.01123>

5. Bayer and Breneis, *Markovian approximations of stochastic Volterra
   equations with the fractional kernel*, 2021.
   <https://arxiv.org/abs/2108.05048>

6. *Markovian approximation of the rough Bergomi model for Monte Carlo option
   pricing*, 2020.
   <https://arxiv.org/abs/2007.02113>

7. Bolli and de Feo, *Optimal control of stochastic Volterra integral equations
   with completely monotone kernels and stochastic differential equations on
   Hilbert spaces with unbounded control and diffusion operators*, 2026.
   <https://arxiv.org/abs/2602.17578>

8. *Adaptive Learning via Off-Model Training and Importance Sampling for Fully
   Non-Markovian Optimal Stochastic Control*, 2026.
   <https://arxiv.org/abs/2604.13147>

9. McCrickerd and Pakkanen, *Turbocharging Monte Carlo pricing for the rough
   Bergomi model*, Quantitative Finance.
   <https://www.tandfonline.com/doi/abs/10.1080/14697688.2018.1459812>

10. Bourgey and De Marco, *Multilevel Monte Carlo simulation for VIX options in
    the rough Bergomi model*, 2021.
    <https://arxiv.org/abs/2105.05356>

11. Gassiat, *On the martingale property in the rough Bergomi model*, 2018.
    <https://arxiv.org/abs/1811.10935>

12. Ehre, Papaioannou and Straub, *Stein Variational Rare Event Simulation*,
    2023.
    <https://arxiv.org/abs/2308.04971>

13. *A Flow-Based Generative Model for Rare-Event Simulation*, 2023.
    <https://arxiv.org/abs/2305.07863>

14. *Conditioning Normalizing Flows for Rare Event Sampling*, 2022.
    <https://arxiv.org/abs/2207.14530>

15. *Accelerated Schrödinger–Föllmer samplers*, 2026.
    <https://arxiv.org/abs/2605.26800>

16. *Efficient and Unbiased Sampling from Boltzmann Distributions via
    Variance-Tuned Diffusion Models*, 2025.
    <https://arxiv.org/abs/2505.21005>

17. *Adaptive Multilevel Splitting: First Application to Rare-Event Derivative
    Pricing*, 2025.
    <https://arxiv.org/abs/2510.23461>

18. Backhoff-Veraguas et al., *Causal transport in discrete time and
    applications*, 2016.
    <https://arxiv.org/abs/1606.04062>

19. *Unified Rough Volatility Framework for Variance-Reduced Pricing and
    Sensitivity of Path-Dependent and Multi-Asset Derivatives*, working paper,
    2025.
    <https://papers.ssrn.com/sol3/Delivery.cfm/5912782.pdf?abstractid=5912782>

20. *Path Integral Formulation of Option Pricing Beyond Black-Scholes: Rough
    Volatility*, working paper, 2026.
    <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6764698>

---

## 22. 다음 문서 업데이트 조건

이 문서는 다음 사건 중 하나가 발생하면 version 1.1 이상으로 갱신한다.

1. controlled rBergomi correctness gate 통과
2. minimal VFO의 첫 Markov/GRU/SOE 비교 완료
3. T-VFO-2 proof가 성립하거나 반례가 발견됨
4. grid dependence 또는 multi-modal path evidence가 확인됨
5. novelty에 직접 충돌하는 새 논문이 발표됨

업데이트 시 후보 순위와 성공 가능성 범위를 다시 평가한다.
