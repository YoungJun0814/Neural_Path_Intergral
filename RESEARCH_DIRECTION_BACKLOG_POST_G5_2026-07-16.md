# G5 이후 연구 방향 백로그

Date: 2026-07-16

Status: reference-only backlog; selected direction is Plan v6

## 0. 결정

다음 주 연구는 **Spectral Doob--Volterra Path-Integral Sampler**로 고정한다.
이 문서는 선택하지 않은 아이디어를 폐기하지 않고, 별도의 가설과 gate를 갖춘 후속
연구 후보로 보존한다. 현재 terminal two-tail task에서 실패한 neural 구조를 이름만
바꾸어 재시도하는 근거로 사용해서는 안 된다.

선택 방향의 목표는 rough-volatility 환경의 진정한 path-dependent rare event에 대해
positive conditional desirability를 spectral 방식으로 근사하고, causal Doob control과
exact path likelihood를 결합하는 것이다.

## 1. 평가 기준

후보 순위는 다음 기준으로 정했다.

1. 새로운 수학적 claim을 명확히 정리로 만들 수 있는가;
2. 현재 exact controlled-rBergomi 및 mixture 엔진을 재사용할 수 있는가;
3. strong CEM/time-dependent CEM보다 유의미한 total-work 개선을 검증할 수 있는가;
4. 실무적인 path-dependent risk 또는 derivative 문제를 해결하는가;
5. 기존 Deep IS, neural PPDE, Schrödinger bridge, QAE 연구와 차별화 가능한가;
6. 실패 시에도 과대주장 없이 독립적인 결과를 보존할 수 있는가.

점수는 게재 확률이 아니라 현재 자산을 고려한 상대적 연구 잠재력이다.

| 순위 | 후보 | 잠재력 | 상태 |
|---:|---|---:|---|
| 1 | Spectral Doob--Volterra path-integral sampler | 9.2/10 | 선택; Plan v6 |
| 2 | Volterra importance sampling error/variance theorem | 8.4/10 | 이론 확장 후보 |
| 3 | Path-functional causal desirability sampler | 7.8/10 | 1위의 축소형 후보 |
| 4 | Tensor-network Volterra desirability | 7.0/10 | backend 후보 |
| 5 | Exact multimodal CEM plus multilevel refinement | 6.5/10 | classical fallback |
| 6 | Fault-tolerant quantum amplitude estimation | 5.2/10 | 별도 quantum 논문 |
| 7 | Dimension-weighted VFO/attention | 3.5/10 | 단독 claim 금지 |
| 8 | Variational quantum-circuit controller | 2.0/10 | 보류 |
| 9 | 현재 terminal task neural tuning | 1.0/10 | stop rule로 금지 |

## 2. 후보 2: theorem-first Volterra importance sampling

### 핵심 질문

Finite-grid Volterra system의 path functional에 대해 approximate Doob proposal과
mixture proposal의 오차가 estimator second moment에 어떻게 전달되는가?

### 가능한 정리

- bounded adapted control 아래 discrete Girsanov likelihood identity;
- randomized expert와 marginal balance likelihood의 unbiasedness;
- log-desirability approximation error에서 relative-variance bound로의 전달;
- SOE 또는 lifted-memory kernel error에서 control/density error로의 전달;
- time-grid와 memory-rank joint convergence;
- defensive mixture의 support 및 moment condition.

### 재개 조건

Plan v6의 Gaussian oracle 또는 rBergomi pilot에서 정리 가능한 안정적인 inequality가
확인될 때 독립 theorem paper로 분리한다. 단순히 알려진 Girsanov와 Feynman--Kac
결과를 다시 쓰는 것은 신규성이 아니다.

## 3. 후보 3: path-functional causal desirability sampler

Spectral decomposition을 사용하지 않고 causal network가

$$
h_i(X_{0:i})=E_P[G(X_{0:N})\mid\mathcal F_i]
$$

를 직접 근사하는 축소형 모델이다. Drawdown, occupation time, barrier hit/recovery,
integrated variance를 상태로 사용한다.

### 장점

- 구현과 falsification이 빠름;
- path dependence가 필요한 실무 문제를 직접 다룸;
- exact likelihood가 network approximation error를 보정함.

### 위험

과거 경로를 입력받는 neural importance sampling은 이미 일반적인 선행연구 영역이다.
따라서 Volterra-specific theorem, martingale consistency, multilevel consistency 또는
many-query amortization이 없으면 저명 저널 신규성이 부족하다.

### 재개 조건

Spectral branch가 계산상 실패하지만 conditional desirability 자체가 time-dependent
CEM보다 명확한 hard-J2 개선을 보일 때만 축소형으로 전환한다.

## 4. 후보 4: tensor-network desirability

Lifted Volterra dimension과 path-functional dimension의 desirability를 tensor train,
matrix-product state 또는 hierarchical low-rank representation으로 압축한다.

$$
h(t,Z_1,\ldots,Z_M,F)
\approx G_0(t)G_1(Z_1)\cdots G_M(Z_M)G_F(F).
$$

### 필요한 증거

- 동일 오차에서 MLP/operator보다 parameter와 inference cost가 작음;
- tensor rank 증가에 따른 error/work curve;
- positivity 또는 log-desirability parameterization;
- truncation error가 estimator variance와 연결됨;
- 단순 PCA, low-rank linear model, neural operator와 비교.

### 재개 조건

Plan v6에서 lifted-state dimension이 실제 병목이고, empirical singular spectrum이
빠르게 감소할 때 backend 후보로 추가한다. 처음부터 주모델로 사용하지 않는다.

## 5. 후보 5: exact multimodal CEM plus multilevel refinement

현재 practical winner인 two-driver CEM mixture를 다음 방향으로 확장한다.

- time-piecewise deterministic control;
- early/late event mode expert;
- adaptive but causal expert allocation;
- grid-coupled proposal과 MLMC;
- task-conditioned CEM parameter surrogate;
- mixture allocation의 second-moment optimization.

### 장점과 한계

구현 실패 위험이 낮고 exact likelihood 자산을 그대로 사용한다. 그러나 CEM과
multiple importance sampling 자체는 신규 방법이 아니므로, multilevel complexity
theorem 또는 path-mode allocation theorem 없이 성능 표만으로는 저명 저널 수준이
되기 어렵다.

### 재개 조건

Plan v6 G0에서 time-piecewise CEM이 모든 neural/spectral 후보를 이기면 정직한
classical 결과로 전환한다.

## 6. 후보 6: fault-tolerant quantum amplitude estimation

Classical path generator와 payoff oracle를 reversible circuit로 구현한 후 QAE로
expectation을 추정하는 별도 방향이다. 이상적인 oracle-query 기준으로 classical
Monte Carlo의 `O(epsilon^-2)`를 `O(epsilon^-1)`로 줄일 잠재력이 있다.

### 필수 항목

- Gaussian/path state preparation;
- reversible BLP convolution과 exponential arithmetic;
- barrier/drawdown accumulator;
- payoff rotation과 uncomputation;
- logical qubit, T-count, T-depth, error-budget resource estimate;
- state-preparation cost를 포함한 end-to-end classical comparison.

### 현재 보류 이유

이는 현재 classical model의 layer가 아니라 계산 backend이며, fault-tolerant resource
가정이 필요하다. State preparation 비용이 query advantage를 상쇄할 수 있고,
rBergomi memory 회로는 Heston보다 훨씬 복잡하다.

### 재개 조건

Plan v6 classical algorithm이 먼저 성공하고, payoff/path generator를 구조적으로
압축할 수 있을 때 별도 quantum-computing 논문으로 분리한다.

## 7. 후보 7: dimension-weighted VFO/attention

각 memory scale 또는 path feature에 fixed/conditional weight를 부여하는 구조다.
수학적으로 허용되지만 일반적인 linear projection 또는 attention만으로는 신규성이
없다.

물리적 kernel weight와 decision relevance weight를 반드시 분리해야 한다.

$$
Y_i\approx\sum_m a_mZ_i^{(m)},\qquad
u_i=u_i^{base}+\sum_mg_{i,m}B_mZ_i^{(m)}.
$$

- `a_m`: Volterra kernel 근사에서 고정;
- `g_{i,m}`: causal state-dependent relevance gate;
- total control 전체가 exact likelihood에 포함되어야 함.

### 재개 조건

독립 모델 claim으로는 재개하지 않는다. Plan v6 spectral ablation의 최소 비교군으로만
사용한다.

## 8. 후보 8: variational quantum-circuit controller

Classical network 대신 parameterized quantum circuit가 control을 출력하는 방향이다.

### 보류 이유

- classical path state amplitude encoding 비용;
- recurrent path evaluation의 circuit depth;
- noise와 barren plateau;
- exact classical likelihood와 연결되는 실질적 이점 부재;
- classical MLP/tensor model보다 나은 complexity 근거 부재.

Quantum hardware에서 계산했다는 사실만으로 scientific advantage가 되지 않는다.
새로운 state-preparation 또는 provable complexity result 없이는 진행하지 않는다.

## 9. 후보 9: 현재 terminal task의 추가 neural tuning

다음 결과로 이미 중단됐다.

- VFO matched-memory gate 실패;
- exact neural mixture가 strong CEM보다 약 2.12배 비효율;
- CEM-anchored residual geometric work-VRF `0.523`;
- residual work comparison 0/5 seeds.

Width, threshold, seed, objective 또는 hidden dimension을 바꾸어 같은 claim을 다시
시험하는 것은 사전 stop rule 위반이다. 새로운 path-functional scientific question이
정의되지 않는 한 재개하지 않는다.

## 10. Quantum-inspired claim boundary

우리 금융 확률경로는 실제 quantum amplitude가 아니다. Real-time Feynman weight
`exp(iS/hbar)`를 proposal probability에 넣지 않는다. Complex/negative weight는
Radon--Nikodym probability ratio가 아니며 sign problem을 만든다.

허용하는 연결은 다음뿐이다.

- imaginary-time/Feynman--Kac semigroup;
- positive desirability와 Doob transform;
- transfer-operator spectral decomposition;
- classical tensor-network compression;
- 향후 별도의 QAE backend.

따라서 Plan v6 논문에서는 실제 quantum advantage를 주장하지 않고
`spectral`, `Feynman--Kac`, `Doob`, `path-integral control`이라는 정확한 용어를
사용한다.

## 11. 백로그 운영 규칙

후보를 재개하려면 다음을 새 문서에 먼저 고정해야 한다.

1. 현재 선택 방향과 다른 scientific question;
2. primary endpoint와 strong baseline;
3. train/development/sealed seed split;
4. exactness와 adaptedness gate;
5. 단일 stop rule;
6. 기존 실패 결과를 뒤집기 위한 post-hoc tuning이 아님을 설명하는 novelty audit.

이 조건 없이 백로그 아이디어를 현재 Plan v6에 추가하지 않는다.
