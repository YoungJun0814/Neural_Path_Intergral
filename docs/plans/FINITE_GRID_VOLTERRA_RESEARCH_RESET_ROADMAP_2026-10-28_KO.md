# 연구 목표 재설정과 유한격자 Volterra 알고리즘 구현 로드맵

작성일: 2026-10-28, America/New_York

기준 저장소 HEAD: `59f280910e115e0ad873907dc5e7f68f2b80598e`
문서 상태: **사용자가 승인한 방향 전환의 구현 로드맵. 실행·성능 입증 보고서가 아니다.**

## 0. 결론과 사용법

주 목표를 다음으로 변경한다.

> **유한격자 Gaussian Volterra 모델에서, 경로의 기억·잔차 구조를 활용한 하나의 추정 알고리즘이 정확도를 유지하면서 전체 학습·선택·추정 비용을 줄이는지 입증한다. 그 개선을 만드는 구조와 실패 조건을 설명한다.**

운영 목표는 더 이상 “기존 P2 전체를 통과할 때까지 다음 기법을 추가한다”가 아니다.
새 단계는 기존 P0/P1/P2와 혼동하지 않도록 **FR0–FR7**로 부른다.

- 기존 P2의 독립 fine reference·연속시간 상대 bias 의무는 미해결로 보존한다.
- 기존 실패·원본 protocol·source archive·seed·통과 기준을 수정하지 않는다.
- 새 finite-grid 목표에서 얻은 성공을 과거 P2의 소급 통과로 쓰지 않는다.
- 기존 코드·음성 결과를 재사용하되, 이미 실패한 실험을 새 이름으로 반복하지 않는다.
- 모든 단계의 마지막은 `계속 / 후보 중단 / 불명확 / 범위 전환` 중 하나의 결정이다.
- 완벽한 무오류나 게재를 약속하지 않는다. 명시적 가정·독립 검증·반증으로 오류 가능성을 줄인다.

이 문서의 표본 수·예산·효과 크기는 **새 연구의 운영 기본안**이다. FR0에서 실행 manifest를
작성하고, reference-only feasibility와 throughput 결과를 반영하여 최초 후보 평가 **전에**
수치 계약을 봉인한다. 후보 성능을 본 뒤 유리하게 바꾸면 새 개발 protocol이며 확인 실험이 아니다.

## 1. 유지할 목표와 분리할 목표

| 구분 | 새 운영 방침 | 허용할 주장 |
|---|---|---|
| 주 연구 | 같은 finite-grid target에서 구조 기반 추정법의 정확도·총비용 개선 | 명시한 격자·task·예산·학습 반복 범위의 결과 |
| 주 이론 | 구조·표현 제한과 추정 위험의 정량적 연결 | 실제로 증명한 가정·유효 범위만 |
| 실무 연결 | bounded digital 및 정규화 put의 계산 workload | 합성 risk-neutral pricing 실험; 시장 검증과 구별 |
| 보조 연구 | missing-tail·잘못된 확신을 드러내는 진단 | 검증된 탐지·오탐·한계만 |
| 별도 고위험 연구 | 연속시간의 유용한 상대오차 정리 | 새 증명이 있을 때만; 주 알고리즘의 선행 의무로 두지 않음 |
| 현재 제외 | 새 시장 동역학, 양자 진폭, 대형 neural operator, foundation model | 효과의 직접 근거 전에는 추가하지 않음 |

경로적분 관점은 **경로에 대한 기대값과 경로공간의 측도 변경**으로 유지한다.
양자역학 요소나 신경망의 존재가 연구 기여의 필수조건은 아니다. 현재는 새 금융모형이 아니라
기존 금융모형을 계산하는 새 알고리즘을 만드는 연구다.

## 2. 현재 출발점: 무엇을 얻었고 무엇이 없는가

### 2.1 보존할 기반

1. 원래 BLP hybrid 유한격자 법칙과 조건부 digital/put/call evaluator.
2. 독립 가격 Brownian 좌표를 적분하는 `3N → 2N` 조건부 계산.
3. 정규화된 Gaussian mixture, full density, defensive natural component.
4. 학습·선택 후 proposal 동결 및 새 최종 표본의 ordinary IS.
5. 역할별 seed, 전체 학습 반복, 실제 비용, 실패·재생·감사 기록.

### 2.2 완료 사실과 미완료 사실

| 기존 작업 | 실제 결론 | 새 계획에서의 처리 |
|---|---|---|
| N32 K2 event/risk 보정, 셀당 5회 전체 학습 | 안정적인 최종 개선 미입증; bank 집중·압축 손실 미분리 | 같은 극단 셀에서 K2만 조금 바꾸는 반복 금지 |
| Gaussian bump·잔차 oracle | 좋은 사건 q와 나쁜 M2 평가 G가 공존 가능 | 정확식·테스트 재사용; 새 원리로 재포장 금지 |
| 정확 conditional replacement·57/61 부호 평균 | expectation 계약 확인; 실제 위험 비교 불안정 | 도구 재사용; 57이 주원인이라고 확정하지 않음 |
| 다중 시점 covariance | 일부 직접 precision 신호, 독립 M2는 불안정 | 기존 구현을 후보 부품으로만 유지 |
| 폭4 window·grouped density | 수치 정합성 확인, 1.2배 속도 정책 gate 미달 | event/risk 실험 0회; 성능 실패로 기록하지 않음 |
| N4096 frozen ensemble | 두 셀 약 0.9% 관측 RSE 기록 | 독립 탐색·전체 재학습의 성공과 구별 |
| 최신 guide-free HMC | control보다 whole RSE 악화 | 현재 고정 후보 중단; seed/count 증량 반복 금지 |
| global-envelope/Hölder continuum bound | 상수 개선에도 유용한 상대 인증 실패 | 같은 proof family의 상수 검색 중단 |

최신 HMC 개발 비교는 canonical control/HMC RSE `8.88% / 24.76%`, high-η
`5.15% / 20.95%`다. 방법·셀당 8whole 결과이며 보편적 HMC 열등성 정리는 아니다.
1,991개 테스트 통과 기록도 효율성·tail coverage·학술 독창성을 대신하지 않는다.

### 2.3 이번 계획이 예전 작은 격자 재시작과 다른 점

이미 10월20일에 fine 인증과 작은 모델 개발을 분리했다. “작은 격자부터 다시 한다”만으로는
새 방향이 아니다. 이번에는 다음을 추가로 고정한다.

- 초극단 `K/S0=.01`이 아니라 **독립 기준을 예산 내 검증할 수 있는 task**에서 기하 가설을 식별한다.
- bank 정보량과 family 표현능력을 별도 대조한다. 둘을 동시에 바꾸고 원인을 주장하지 않는다.
- 물리 기하와 비구조 기하, event 목적과 risk 목적을 혼동하지 않는 제거 대조를 둔다.
- 새로운 큰 모델보다 기존 구현을 묶는 작은 공통 benchmark 계층을 먼저 만든다.
- 무한 재설계 대신 제한된 한 사이클 후 계속/중단을 결정한다.

## 3. 수학적 계약

### 3.1 선언할 estimand

기본 대상은 `Z ~ p_N = N(0,I_(2N))`, 원래 BLP 격자 `N`, `epsilon=1`이다.
현재 terminal downside digital에서

\[
I_N=\Delta t\sum_{i=0}^{N-1}V_i,\quad
m_N=\log S_0-\tfrac12 I_N+\rho\sum_{i=0}^{N-1}\sqrt{V_i}\Delta W_i,
\]

\[
g_N(Z)=\Phi\!\left(\frac{\log K-m_N}{\sqrt{(1-\rho^2)I_N}}\right),
\qquad \mu_N=E_{p_N}[g_N(Z)].
\]

`I_N>0`, `|rho|<1`을 요구한다. 현 코드의 금리·배당 convention을 그대로 쓰고,
일반화할 경우 drift·할인·forward를 함께 명시한다. `mu_N>0`인 대상에서 상대오차를 정의한다.
local Gaussian 두 채널은 독립된 두 물리 Brownian이 아니다.
terminal에는 `V_0,...,V_(N-1)`이 쓰이며 마지막 residual `2N-1`은 payoff 미사용 음성 대조다.

### 3.2 bounded 실무 payoff

첫 practical payoff는 `g_N=E[(K-S_T)^+/K | Z]`로 정규화한 put으로 둔다.
이때도 `0<=g_N<=1`이며 원래 통화 단위 가격은 현 convention에서 `K*mu_N`이다.
할인율을 도입하면 할인 인자를 명시적으로 복원한다. digital 확률과 put 가격은 다른 estimand다.

call은 unbounded이므로 현재 bounded contribution·모멘트 보장을 그대로 재사용하지 않는다.
첫 사이클에서 call, barrier, Greeks, 시장 calibration까지 동시에 확장하지 않는다.

### 3.3 최종 estimator와 exact-density 범위

\[
\widehat\mu_N=\frac1n\sum_{i=1}^n g_N(Z_i)\frac{p_N(Z_i)}{q(Z_i)},
\quad Z_i\overset{iid}{\sim}q.
\]

- q는 학습·선택 후 동결된 정규화 밀도다. 최종 표본은 그 과정과 독립이다.
- 분모는 생성 label의 밀도가 아니라 **전체 mixture 밀도**다.
- fitting의 가중치 정규화는 가능하지만 final의 self-normalization·weight clipping은 금지한다.
- `q>=delta*p`, 기본 `delta=.1`을 유지한다. 이는 support·유한 모멘트 보호이지 낮은 상대오차 보장이 아니다.
- 학습 q에 조건부로 기대값이 `mu_N`이라는 수학적 불편성과, 유한 실행에서 그 평균을 잘 관측했는지는 별개다.
- 전 경로 finite-dimensional proposal을 adapted SDE drift나 연속시간 Girsanov control로 부르지 않는다.
- 원래 simulator의 adapted Itô convention·보정항을 proposal 설계와 함께 임의로 바꾸지 않는다.
- “exact”는 선언한 밀도·조건부식의 대수적 계약이다. float64와 수치 quadrature를 무오차라고 부르지 않는다.

### 3.4 사건 mass, 위험 mass, 평가 proposal의 구분

`f=g_N*p_N`, `mu=mu_N`, `J(q)=M2(q)`로 쓰면

\[
q^*=f/\mu,\quad J(q)=\int f^2/q,\quad
J(q)/\mu^2=1+\chi^2(q^*\Vert q).
\]

동결 q의 위험을 찾는 target은 `r_q=f^2/[q*J(q)]`다. 이는 사건 최적 q*와 다르다.
새 독립 `Z~G` bank에서

\[
H_q(Z)=\frac{g_N(Z)^2p_N(Z)^2}{G(Z)q(Z)},\qquad E_G H_q=J(q).
\]

`q>=delta*p`, `G>=gamma*p`이면 `H_q<=1/(delta*gamma)`다.
하지만 작은 J의 상대정밀도는 여전히 나쁠 수 있다. 보조 G/r만 개선한 결과는 사건 q 개선이 아니다.

기하 영역 `A_j`에 대해 `∫_Aj f / mu`, `∫_Aj f²/q / J(q)`, `∫_Aj q`를 각각 보고한다.
분자·분모의 불확실성과 공분산을 반영하며, 위험 기준이 부정확하면 mass/risk 차이도 불명확이다.

### 3.5 이미 알려진 이론과 새로 필요한 이론

위 IS/χ² 항등식, defensive bound, mixture convexity, 조건부 평균의 분산 감소는 기존 원리다.
독립 직교 Gaussian 분해 `Z=(X,U)`, `p=p_X*p_U`에서
`q=q_X*p_U`, `m2(X)=E_pU[g²|X]`라 하고 아래 E를 p_X에 대한 기대값으로 쓰면

\[
J(q)\ge (E\sqrt{m2(X)})^2\ge\mu^2
\]

인 relaxed representation floor도 이미 내부 oracle에서 검토했다. 현재 제한된 mixture가 equality를
달성한다고 가정하지 않는다. 기존 q의 평가에는 `E[g²|X]`, 조건부 평균으로 바꾼 사건 estimator에는
`E[g|X]²`가 필요하다. 수치 quadrature를 쓰면 그 오차·bias도 별도다.

새 이론의 후보는 **Volterra kernel/물리 시간창/잔차 상호작용이 이 gap 또는 J에 주는 정량 효과**다.
H·η·ρ·희귀도·N에 따른 상수가 실제 계산 구간에서 유용해야 한다. 의미 없는 상계만 나오면
“효율 정리”로 주장하지 않고 수치·메커니즘 연구 또는 엄밀한 한계 분석으로 분류한다.

## 4. 주 가설과 최소 대조 설계

### 4.1 검증할 단일 중심 가설

> Volterra의 사용 분산을 만드는 잔차와 그 직후 Brownian의 특정 상호작용이,
> 사건 학습에서 작게 보이는 위험 영역을 형성한다. 이 구조를 보존한 제한된 보정은
> 같은 정보·표현능력의 비구조 보정보다 전체 학습 비용을 포함한 오차–비용을 개선한다.

현재는 **가설**이다. selected extreme witness 몇 개나 과거 covariance의 낮은 RSE가 증명하지 않는다.
“event와 risk는 다르다”라는 일반 사실 자체를 독창성으로 쓰지 않는다.

### 4.2 먼저 분리할 두 원인

1. **자료 원인:** 같은 family에서도 bank의 탐색 범위를 바꾸면 독립 위험·오차가 개선되는가?
2. **표현 원인:** 같은 bank에서도 구조를 보존한 family가 압축 family보다 개선되는가?

최초 원인 실험은 `bank mechanism 2개 × family 2개`다. 기본안은 IID physical mixed bank와,
별도 CE/SMC로 적합·동결한 정규화 mixture G에서 **새로 뽑은 IID bank**다.
family는 부모 보존/제한 압축을 비교한다. bank 수, 정규화 상수의 취급, 실제 평가 비용을 명시한다.
SMC endpoint의 주변밀도를 알려진 G로 간주하지 않는다. endpoint particle bank를 학습에 사용하는
별도 대조가 필요하면 weighted empirical target 근사·의존성·normalizer 처리를 따로 정의한다.
그 입자에 밀도를 아는 IID bank의 importance weight 식을 그대로 적용하지 않는다.
fresh final 및 위 H_q의 평가는 실제 밀도를 계산할 수 있는 새 IID proposal에서만 수행한다.
먼저 같은 정보량 대조, 다음에 같은 실제 예산 대조를 구분한다.

### 4.3 구조 효과와 목적함수 효과의 분리

원인이 식별된 경우 다음 `2×2`를 사용한다.

| | 물리 기하 | 비구조 대조 기하 |
|---|---|---|
| event 목적 | E-P | E-C |
| risk-informed 목적 | R-P: 주후보 | R-C |

- `R-P vs R-C`: 같은 목적에서 Volterra 구조의 추가 가치.
- `R-P vs E-P`: 같은 물리 기하에서 목적 선택의 효과.
- `E-P vs E-C`: risk 목적 없이도 나타나는 구조 효과.
- `E-P vs R-C`만으로 목적 효과를 말하지 않는다. 두 요인이 동시에 달라진다.
- 부모 q0, strong static physical guide를 함께 둔다. 기존 q를 학습 보정이 반드시 이긴다고 가정하지 않는다.

비구조 기하는 같은 rank·covariance spectrum·성분 수를 유지하는 사전 고정 직교 방향으로 만든다.
미사용 terminal 좌표는 제외한다. 전역 무작위 회전은 시간 locality까지 바꾸므로 “물리 구조 전체의
가치” 대조이지 coupling 하나의 인과 대조는 아니다. coupling 효과에는 같은 시간창·marginal variance를
유지한 cross-covariance 제거 arm을 별도 작은 제거 대조로 쓴다.
whole별 control rotation seed를 사전 배정하여 한 회전의 운을 일반적인 비구조 기하 효과로 읽지 않는다.
event/risk arms의 초기값·반복 횟수·탐색 수와 정보량까지 맞춘다. 추가 control 비용도 포함한다.

후보 family를 계속 늘리지 않는다. **한 제한 family와 그 제거 대조**가 원칙이며, 원인에 맞는 재설계는
**FR0–FR4 전 사이클 합계 최대 1회**다. 단계가 바뀌어도 수정 기회가 충전되지 않는다.
비교 arm 수가 후보 발명 수와 같다는 뜻은 아니지만, arm 비용은 전부 장부에 포함한다.

## 5. 최소 알고리즘 설계와 재사용 범위

### 5.1 먼저 만들 것은 공통 실험 계층이다

새로운 대형 아키텍처보다 기존 physics/proposal을 같은 계약으로 연결한다.

```text
task·N·payoff 고정
  → reference-only feasibility
  → 매번 새 q0 학습 / static 기준 구성
  → discovery bank와 제한된 구조·대조 후보
  → 독립 selection 후 q 동결
  → 새 final estimator + 독립 검증
  → 전체 재학습 오차와 전체 비용
  → 봉인된 새 task에서 확인
```

### 5.2 제한된 보정 family: FR2 결과에 따라 한 번 확정

기본 출발점은 exact normalized blend `q_(a,theta)=(1-a)q0+a*r_theta`다.
q0와 r_theta 모두 natural floor를 가진다. `a=0`은 정확한 no-op다.

- 부모 mean·mode를 보존한 Gaussian mixture에 작은 rank covariance 보정을 적용한다.
- 물리 방향은 사용 residual–직후 Brownian과 **물리 시간** 창으로 정의한다.
- 초기 rank는 최대 4, 시간창 수는 최대 4를 운영 상한으로 둔다. 실제 성분·density 항 수도 함께 제한한다.
- 고유값·blend 후보는 작은 고정 집합으로만 선택한다. 시작안은 `a∈{0,.25,.5}`, covariance 고유값 `∈{1,3,9}`다.
- 모든 조합을 무조건 탐색하지 않는다. FR2로 정한 한 scale과 최대 두 비자명 blend만 FR3에서 평가한다.
- event/risk 비교에서는 mean·성분 수·rank·bank를 맞추고 목적만 바꾼다.
- bank가 정보를 제공하지 못하면 noisy M2를 더 최적화하지 않는다. 구조 고정 후보와 학습 후보를 구분한다.

첫 구현의 실제 학습 변수는 **고정 r_geometry와 q0 사이의 blend a 하나**로 제한한다.
scale·창·rank는 FR2에서 한 번 고정하고 FR3의 event/risk 비교 도중 바꾸지 않는다.
같은 `Z_i~G` training bank에서 event 목적은
`L_event(a)=-mean[(g*p/G)*log(q_a/p)]`, risk 목적은
`L_risk(a)=mean[g²*p²/(G*q_a)]`로 정의하여 같은 a 집합에서 최소화한다.
L_event에 q_a 대신 q_a/p를 쓴 차이는 a와 무관한 상수이며, fitting weight의 공통 정규화도 argmin을 바꾸지 않는다.
이것은 finite-bank direct-M2 목적이지 frozen-q0의 r_q0를 적합하는 것과 같지 않다.
선택 후 empirical 최솟값을 population M2로 보고하지 않는다.
각 objective가 제안한 a와 no-op을 독립 selection bank에서 사전 고정한 같은 규칙으로 비교한다.
selection이 둘을 모두 no-op으로 돌리면 목적 차이가 없다는 결과를 보존한다.
더 많은 weight·mean 학습은 이 제한 family의 근거가 있을 때만 별도 재계약한다.

`a<1`이면 `J(q_(a,theta))<=J(q0)/(1-a)`다. 이는 제한적 악화 상한이며 개선 보장이 아니다.
현재 373개 mean에 모든 시점 covariance를 곱해 11,905항을 만드는 구성은 “작은 수정”이 아니다.
큰 dictionary가 필요한 경우 compact parent와 strong static 모두에 대한 실제 비용을 먼저 확인한다.
더 작게 만든 family가 성능을 잃으면 density 비용과 표현 손실을 함께 보고한다.

기존 폭4 window는 구현 부품으로 재사용할 수 있으나, 원래 1.2배 gate 실패는 보존한다.
새 목적에서 새 예산을 사전 고정해 평가하는 것은 가능하지만, 기존 STOP을 성공으로 바꾸지 않는다.

### 5.3 기존 모듈

| 기능 | 재사용 파일 | 주의 |
|---|---|---|
| 조건부 payoff | `src/path_integral/volterra_conditional_payoffs.py` | digital·put·call 구분; bounded 가정 |
| cached digital physics | `src/path_integral/cached_volterra_terminal.py` | epsilon=1 downside 전용; task 변경 시 cache 재생성 |
| BLP FFT·coupling | `src/path_integral/rbergomi_fft.py` | 원래 covariance·Itô convention 보존 |
| physical guide | `src/path_integral/physical_window_guide.py` | 강한 static baseline; setup 비용 포함 |
| exact Gaussian transport | `src/path_integral/finite_rank_gaussian_transport.py` | 밀도·sampling·rank 일치 |
| CE baseline | `src/path_integral/baselines/weighted_conditional_ce.py` | 구현한 알고리즘 범위만 명명 |
| SMC | `src/path_integral/tempered_conditional_smc.py` | whole 정규화상수와 의존 bank 구분 |
| 기존 event/risk 보정 | `src/path_integral/risk_aware_model_correction.py` | 기존 K2 실패 보존 |
| 조건부 개입 | `src/path_integral/conditional_intervention.py` | mixture posterior label 사용; covariance 변경 시 조건부분포 재검토 |
| 잔차·window covariance | `src/path_integral/residual_brownian_covariance.py`, `src/path_integral/window_coupled_covariance.py` | N32 index를 physical-time 불변으로 오인 금지 |
| paired 위험 | `src/path_integral/paired_model_risk.py` | 공통 표본 covariance·분모 불확실성 |
| 비용·seed | `src/path_integral/research_cost_accounting.py`, `src/path_integral/seed_ledger.py` | 실패·선택·감사 비용도 기록 |

기존 `baselines/smoothing_rqmc.py`는 **3N 좌표의 rank-one 가격 smoothing**이다.
이를 새 **전체 가격 조건부 2N RQMC** baseline으로 그대로 이름만 바꾸지 않는다.
새 adapter는 같은 g_N에 scrambled Sobol을 적용하고, 독립 scramble 평균을 통계 단위로 사용한다.
정규 역CDF endpoint·유한정밀도 문제를 명시하고, 0/1 입력을 조용히 clipping하거나 좋은 scramble만 고르지 않는다.

### 5.4 최소 신규 파일 — 아직 미구현

1. `src/path_integral/finite_grid_geometry_protocol.py`: estimand·역할·예산·판정 schema.
2. `src/path_integral/finite_grid_geometry_benchmark.py`: 기존 physics/proposal/비용 adapter.
3. `src/path_integral/finite_grid_geometry_controls.py`: 필요한 경우에만 matched 기하·제거 대조.
4. `experiments/finite_grid_geometry_mechanism_study.py`: oracle·원인 분리·개발 실행.
5. `experiments/finite_grid_geometry_confirmation.py`: 동결 후보 하나의 새 확인 실행.
6. `configs/finite_grid_geometry/`: task·protocol·resource manifest. confirmation은 별도 파일.
7. 대응 protocol·density·statistic·runner 테스트. 같은 numerical core를 복제하지 않는다.

기존 abstraction으로 가능하면 신규 파일을 더 줄인다. 이 목록은 이 문서에서 생성한 파일 목록이 아니다.

## 6. 단계별 실행 로드맵

### FR0 — 주장·문헌·task·예산 계약

**목표:** 무엇을 성공으로 부를지와 무엇을 주장하지 않을지 먼저 고정한다.

구현·산출물:

1. estimand/주 가설/주 endpoint/강한 baseline/실패 조건 한 페이지 계약.
2. 기존 구현·결과와 새 작업의 차이를 표시하는 inventory 및 claim ledger.
3. 선행연구의 수식·가정·가장 가까운 알고리즘 비교표. 핵심 원문은 본문까지 확인한다.
4. task·seed-role·unit·예산·새 source snapshot을 담는 protocol schema.
5. `PASS / REJECT_CANDIDATE / UNRESOLVED_REFERENCE / UNRESOLVED_BUDGET / INVALID_IMPLEMENTATION`
   상태를 분리한다. 과거 P2 승인 field를 새 승인 field와 공유하지 않는다.

protocol의 필수 field는 `estimand_id, law_digest, task, N, payoff_scale, phase, role, whole_id,
proposal_digest, seed_key, parent_artifacts, planned_units, completed_units, max_work, max_wall,
reference_policy, allocation_rule, primary_comparison, multiplicity_rule, failure_policy`다.
예산 장부는 scalar/gradient/physics/density 호출과 실제 wall/RSS를 구분한다.
stage 결과가 없어도 실패 unit과 이미 소비한 비용은 저장한다. `redesign_count`는 전 사이클에서 공유한다.

단일 배포 workflow의 precision/accuracy 자격, 여러 whole 평균의 연구용 불확실성,
whole 간 실패율·timeout·fallback 정책, cell 단위 비용 endpoint를 별도 field로 봉인한다.
여러 whole을 합친 평균만 정확한 것을 단일 workflow의 정확도로 부르지 않는다.

gate:

- 새 기여를 한 문장으로 쓸 수 있고, 기존 일반 항등식의 재명명이 아니어야 한다.
- 예상 성능을 단정하지 않고 반증 가능한 기하 가설을 명시해야 한다.
- 학습·reference·selection·final·audit의 역할과 비용을 분리해야 한다.
- novelty 경계가 불명확하면 대형 구현 대신 문헌/전문가 검토에서 멈춘다.

### FR1 — 기존 known-answer 검증을 재사용하고 부족한 대조만 추가

**목표:** 정확한 답이 있는 곳에서 estimator·density·위험 진단을 검증한다.

기존 Gaussian bump, untouched-residual floor, 유한 전수합산, 미사용 terminal residual 검사를 재사용한다.
새로 필요한 것은 작은 event mass/큰 risk mass를 가진 2mode 예와, 같은 spectrum의 물리/대조 기하다.
1D/2D 독립 적분이나 완전한 식을 우선하며, mixture의 역수가 들어간 M2를 일반적으로 닫힌 식이라고 가정하지 않는다.
Volterra 선형 Gaussian feature를 쓴 toy는 좌표 해석 검사이지 rBergomi 성능 증거가 아니다.

검증:

- q=p, g=상수, a=0, 동일 mean/covariance dense Gaussian과의 일치.
- `sample → log_density` 일치, mixture 전체 정규화, natural floor, covariance SPD.
- 작은 mass mode를 누락하는 반례에서 진단의 탐지·오탐·불명확 비율.
- 실제 정답 기준 bias/RMSE·구간 coverage의 반복 평가와 반복 수 불확실성.
- 통계 코드가 알려진 나쁜 후보를 성공으로 표시하지 않는 negative control.
- float64 tolerance는 scale-aware로 정하고 서로 다른 reduction의 bitwise 동등성을 요구하지 않는다.

gate: numerical 계약 실패는 고친 뒤 새 버전으로 재검사한다. finite-run 평균 일치만으로 불편성을 증명했다고
쓰지 않는다. 기존 이미 통과한 toy를 새 학술 발견으로 집계하지 않는다.

### FR2 — 기준을 확보할 수 있는 Volterra 문제와 원인 분리

**목표:** 성능 탐색 전, 로컬 예산에서 비교가 실제로 가능한 task인지 확인한다.

아래는 최초 task manifest의 기본안이다. `S0=100, xi0=.04, epsilon=1`이며 시장 calibration 값이라고 주장하지 않는다.

| 개발 ID | H | η | ρ | T | K/S0 | N | payoff |
|---|---:|---:|---:|---:|---:|---:|---|
| D01 | .10 | 1.0 | -.70 | .5 | .80 | 32 | downside digital |
| D02 | .05 | 1.5 | -.70 | 1.0 | .60 | 32 | downside digital |
| D03 | .10 | 1.5 | -.70 | 1.0 | .50 | 64 | downside digital |
| D04 | .15 | 1.0 | -.70 | 1.0 | .80 | 64 | put/K |

이 값들의 희귀도·reference 비용은 아직 확인하지 않았다. “중간 난도 후보”이지 확률 구간의 보장이 아니다.
새 후보 성능을 보지 않는 **reference-only pilot**으로 기준의 오차·시간·메모리를 확인한다.
불가능한 셀은 실패·제외 사유를 공개하고, 사전 fallback task가 필요하면 최초 pilot 전에 순서를 고정한다.
후보가 잘되는 strike를 찾아 task를 고르지 않는다. 평가 가능한 일부 task의 성과는 그 범위에만 주장한다.
이 선택은 reference-feasible task 범위에 대한 연구다. 전체 희귀사건 영역을 대표한다고 주장하지 않는다.
pilot reference는 최종 accuracy reference에 합치지 않고, 동결 후보의 평가는 새 reference 자료로 수행한다.

reference 기본 조합:

- 같은 g_N의 conditional MC / 독립 scramble conditional RQMC.
- 필요하면 별도 CE/SMC 방식의 cross-check 하나. 둘 다 불안정하면 oracle로 부르지 않는다.
- 동일 simulator 공유의 한계를 적고 작은 raw/conditional·dense/FFT 법칙 검사를 별도로 유지한다.
- 난수 독립성과 탐색 기하의 독립성을 별도 field로 기록한다.

원인 실험 순서:

1. same-family/different-bank와 same-bank/different-family 대조를 분리한다.
2. group replacement와 training geometry 밖 bin을 사용하되 threshold는 discovery/calibration에서만 고정한다.
3. 위험 기여 reference가 불충분하면 group 순위를 확정하지 않는다.
4. 선택된 witness는 full-law 수치 감사에만 사용하고 population 빈도·기여를 추정하지 않는다.
5. 최소 한 digital과 한 practical bounded payoff에서 기준이 가능해야 이후 일반 실무성 연구로 확대한다.

gate: 적어도 한 유효한 Volterra 셀에서 검증 가능한 원인과 대조가 있어야 FR3으로 간다.
원인이 식별되지 않으면 한 번만 제한 재설계하고, 또 불명확하면 새 architecture를 추가하지 않고 방향 판단을 한다.
FR3의 두 셀 비교를 위해 두 번째 자격 셀도 필요하다. 한 셀만 확보되면 그 셀의 진단까지만 보고하고
나머지는 `UNRESOLVED_REFERENCE`로 남긴다. 미자격 셀을 강제로 넣지 않으며 범위 변경은 별도 사전 계약으로 처리한다.

### FR3 — 한 제한 family의 구조·목적 제거 대조

**목표:** FR2의 원인에 연결되는 실제 q 변경 하나를 시험한다.

- §4.3의 2×2를 사용한다. control geometry·rank·spectrum·표현능력을 사전 고정한다.
- 동일 train bank를 arms 간 공유해 paired 원인 실험을 할 수 있다. 이는 독립 confirmation이 아니다.
- selection bank는 train과 독립이며, 선택 후 final·heldout bank를 다시 새로 만든다.
- 1차에는 2개 셀×3whole을 사용해 구현·비용·효과 방향을 확인한다. 낮은 실패율 증거가 아니다.
- 최대 한 번의 원인 기반 수정만 허용한다. 수정 전 결과는 실패/개발 기록으로 남긴다.
- selection 기준이 불안정하면 q0/no-op을 우선하는 tie·fallback 규칙을 사전에 둔다.

gate: 숫자 하나의 큰 RSE 감소가 아니라, 독립 오차 평가와 구조 대조가 함께 지지되어야 한다.
비구조 대조와 같다면 “Volterra 특화 구조 효과”를 포기한다. static과 같거나 더 나쁘면 학습 성분의
가치가 없을 수 있다. density 비용이 이득을 지우면 단순화하거나 후보를 중단한다.

### FR4 — 전체 재학습과 고정정밀도 개발 benchmark

**목표:** 좋은 frozen q가 아니라 실제 workflow를 평가한다.

전체 unit은 `새 부모 생성 → bank → fit → selection → freeze → fresh final`이다.
기본 4개 개발 셀×10whole을 계획하되, FR2의 자격 셀과 FR0의 예산 계약으로 최초 실행 전에 확정한다.
학습 없는 static은 fresh sampling 반복이고, 10회 학습으로 부르지 않는다.
최종 8block은 하나의 학습에 조건부인 평가 block이며 8whole이 아니다.

baselines:

1. 같은 g_N의 conditional MC 및 2N conditional RQMC.
2. 정확한 밀도의 충분히 튜닝한 기존 CE, 필요하면 고정 일정 SMC.
3. strong static physical guide.
4. q0·동일 capacity 비구조 대조. 문헌 알고리즘 전체를 구현하지 않았으면 그 이름을 빌리지 않는다.

fixed-precision 설계:

- 독립 allocation pilot에서 final sample count를 정하고 본 평가 전에 동결한다.
- whole별 표본 수가 달라지는 경우도 독립 pilot과 사전 고정 rule만 사용한다. final 관측값으로 추가하지 않는다.
- 개발 target은 상대 SE 5%, 확인 target은 2%를 기본안으로 둔다. 이는 상대 RMSE 정리가 아니다.
- reference SE 기본 목표는 해당 ε의 1/4 이하이며, reference CI·불확실성도 같이 공개한다.
- 평균 일치 margin과 검정 power는 두 estimator의 오차를 반영해 사전 계산한다. SE보다 훨씬 좁은
  equivalence margin을 두고 실패를 구조 문제로 오인하지 않는다.
- 정확도는 이론 계약 + 독립 기준과의 일치 + whole 반복 오차/집중/민감도에서 평가한다.
- 단순 t/delta/부트스트랩 CI는 tail 누락을 배제하지 못한다. 알려진 정답에서 calibration을 점검하고,
  계산 가능한 셀에서는 bounded finite-sample interval도 대조한다. 그 상계가 느슨하면 그대로 적는다.
- 후보별 M2 전체를 정밀 인증하는 과제를 다시 전역 선행 gate로 만들지 않는다. **M2 기하를 주장하는
  곳에는 M2 증거가 필요하지만**, 독립 whole 오차로 판단한 유한격자 계산 성과와는 별도 claim이다.
- 독립 기준·whole 오차 모두 불안정하면 동일정확도 비용 우위를 선언하지 않는다.
- precision을 만족할 때까지 표본을 추가하고 통상 fixed-n CI를 쓰지 않는다. 적응 정지가 필요하면
  적절한 anytime-valid 절차와 예산을 새로 설계한다.

오차 qualification은 **각 단일 workflow의 최종 estimator**와 그 전체 반복 분포를 대상으로 한다.
10whole 평균의 RSE가 2%여도 한 번의 workflow가 2%를 달성했다고 표시하지 않는다.
cell/method qualification에는 단일 workflow의 precision·reference 일치·실패 비율과 그 불확실성을 모두 반영한다.
허용 실패 비율·검정력·평가 방식은 FR0에서 정하며 10회 결과로 낮은 실패율을 증명했다고 주장하지 않는다.
미자격 whole은 제거하지 않는다. 사전 fallback이 있으면 실제 추가 비용·최종 출력까지 평가하고,
없으면 censored failure로 남긴다. 성공 whole만 골라 비용비를 계산하지 않는다.

gate: 실용적인 개선 기준의 기본안은 개발에서 선택한 strongest qualified comparator 대비
**speedup = T_comparator / T_candidate >= 1.5**다.
이는 보편적 학술 기준이 아니라 복잡도 증가를 정당화하기 위한 운영 목표다. 관측 점추정과 불확실성을
함께 보고하고, FR5의 primary 주장·비교·반복 수를 이 결과로 한 번 결정한다. 1.5배 미만이라도 새로운
유용한 정리가 있으면 별도 이론 기여를 검토하되, 효율성 성공으로 바꾸지 않는다.
이는 공통 정확도 계약을 충족한 고정 allocation workflow의 실제 비용 비교이며, 각 방법의 최적 표본 수나
이론적인 최소 필요시간을 찾았다는 뜻은 아니다. 여기서 strongest는 평가한 개발 baseline 집합 안의 선택이다.

### FR5 — 새 task·새 전체 학습의 봉인 confirmation

**목표:** 선택과 평가를 분리하고 개발 셀 overfit을 확인한다.

최종 후보 1개, primary comparator, 정확도 계약, primary cost endpoint, 검정 hierarchy를 봉인한다.
baseline은 개발 자료에서 정한 규칙으로 동결하고 동일한 tuning 기회를 부여한다.
새 `H/η/ρ/T/K/N` 조합 최소 4개, 기본 whole 20회/셀을 출발안으로 하되 필요한 precision/power와 비용으로
최초 confirmation 전에 정한다. 20회가 작은 실패율을 보장하는 것은 아니다.

- 2개는 새 digital 조건, 1개는 새 normalized put, 1개는 격자·기억 강도 변화 조건으로 구성한다.
- 최소 N64와 N128을 포함한다. 정확 task 숫자는 FR0의 별도 holdout manifest에 보관한다.
  해당 task는 FR1–FR4의 학습·튜닝·selection·성능·reference 평가 어디에도 사용하지 않는다.
  후보·분석·allocation 규칙 동결 후 처음 실행한다.
- holdout allocation pilot이 필요하면 알고리즘 동결 후의 사전 정의된 후보-맹검 절차로만 수행하고,
  pilot과 confirmation 자료를 분리한다. 봉인된 reference feasibility를 충족하지 못하면 미자격/불명확으로 남기며
  그 셀을 잘되는 다른 task로 교체하지 않는다.
- 과거 K=1 canonical/high-η는 **이미 관측한 stress test**다. unseen confirmation으로 부르지 않는다.
- 새 train/selection/final/reference seed와 source/proposal digest를 사용한다.
- 여러 primary 셀의 동시 주장은 family-wise 보정 또는 사전 hierarchy를 사용한다. 유리한 셀만 평균내지 않는다.
- 실패·timeout·미자격 셀은 표에 모두 남긴다. 성공한 실행만의 조건부 median을 headline으로 쓰지 않는다.
- confirmation을 본 뒤 모델·threshold·task를 바꾸면 해당 세트는 개발 자료가 된다. 새 확인 비용이 필요하며 자동 반복하지 않는다.

gate: 정확도 자격을 얻은 적어도 두 방법 사이에서만 실제 비용비를 주장한다.
한 셀의 성공은 그 셀의 성과다. task 일반화 주장은 사전 정의한 다중 셀 판정을 통과해야 한다.
확인 실패 시 새 이름의 kernel을 추가하지 말고 FR7에서 범위를 줄이거나 중단한다.

### FR6 — 이론·격자 확장·실무 비용의 연결

**목표:** 수치 개선을 설명하고 유효 범위를 정한다. FR4 이후 이론 작업과 병행할 수 있다.

1. 표현 floor/조건부 변동이 실제 어느 물리 창·잔차 구조에 연결되는지 정량화한다.
2. 새 lemma가 필요한 경우 가정, 사용 norm, integrability, 희귀도에 따른 상수 폭발을 적는다.
3. N32/64/128에서 같은 physical-time 규칙의 효과·rank·밀도 비용을 비교한다.
4. local white-coordinate의 같은 variance 확대를 mesh-invariant 물리 perturbation이라고 부르지 않는다.
5. 인접 격자는 올바른 joint covariance coupling으로 비교하고 mesh sensitivity만 보고한다.
6. 작은 adjacent correction으로 무한 continuum remainder를 인증하지 않는다.
7. 반복 workload의 amortization은 실제 재사용 시나리오에서만 별도 계산한다.

연속시간 이론은 tail localization·weak cancellation·tail/error correlation 등 **새 논증**의 가능성을
전문가와 검토하는 별도 트랙이다. 기존 global-envelope 상수/r scan은 재개하지 않는다.
이 트랙의 성공을 주 finite-grid 논문의 필수조건으로 되돌리지 않는다.

### FR7 — 논문/전환/중단 결정과 재현 패키지

| 최종 증거 | 결정 |
|---|---|
| 구조적 설명 또는 새 유용한 정리 + 새 task에서 총비용 이득 | 알고리즘 중심 원고·재현 패키지 진행 |
| 강한 재현 가능한 계산 이득, 이론은 기존 원리 적용 | 계산·응용 기여로 정직하게 범위 설정 |
| 구조 특화 효과 없음, 단순 방법이 우세 | 단순 방법 채택; 특화 모델·학습의 우월성 주장 중단 |
| 새 일반 한계/진단 보증은 있으나 계산 이득 없음 | 별도 방법론 논문 가치 검토; 음성 결과만으로 게재 가정 금지 |
| 오차·비용·기여 모두 불충분 | 해당 계열 중단, 문제 자체의 더 큰 전환 |
| 예산/기준 부족 | 불명확 상태로 종료; 성공/실패를 억지로 확정하지 않음 |

원고 이전에 수치확률/rough volatility 연구자에게 **핵심 정리 하나, 핵심 결과 그림 하나,
closest-work 비교표**의 반박 검토를 받는 것을 권한다. 최상위 저널은 유용한 새 지식·강한 증거가
갖춰졌을 때 목표로 삼는다. 코드·문서·검사 수 또는 AI의 긍정 평가로 논문 수준을 정하지 않는다.

## 7. 통계·비용·운영 공통 계약

### 7.1 독립성과 정보 흐름

```text
reference-only feasibility ───────────────┐
discovery → training → selection → freeze → fresh final
                                      └──→ fresh heldout diagnostics
새 confirmation: 위 과정을 새 task/whole/seed로 다시 수행
```

참고값을 알고 있는 것과 그 자료로 후보를 선택하는 것은 다르다. confirmation reference 결과는
후보와 allocation 동결 후 공개하는 것이 기본이다. role별 source·artifact 의존성을 기록한다.
학습 bank를 arms 간 공유하면 그 의존성을 paired 분석에 반영한다. geometry partition은 train/calibration에서만
고정하고 `out-of-training-geometry` bin을 남긴다. support 밖이라는 잘못된 표현은 쓰지 않는다.

독립 IID final, whole 학습, SMC whole, QMC scramble은 서로 다른 통계 단위다.
같은 seed의 replay는 새로운 독립 반복으로 더하지 않는다. seed 충돌 검사는 난수 독립성의 수학적 증명이 아니다.

### 7.2 primary 오차·속도 보고

- fixed n의 관측 SE, 독립 reference와의 오차, whole 간 분포를 분리한다.
- reference를 정확한 참값으로 넣어 RMSE를 계산하지 않는다. reference 오차·공유 covariance를 전파한다.
- paired M2 difference는 같은 경로 covariance를 포함한다. ratio는 보조이며 작은 분모 문제를 표시한다.
- 알려진 정답이 아닌 곳의 coverage는 “참 coverage”로 부르지 않는다.
- 개발 5–10whole은 신호 확인용이다. 0/5 실패의 독립 Bernoulli 단측 95% 상한은 약45%다.
  0/20도 약14%이므로 “대부분 항상 성공”이라는 신뢰도 주장은 별도 설계가 필요하다.
- timing의 1.5배 point estimate를 통계적으로 확정된 1.5배 하한으로 바꾸지 않는다.

### 7.3 총비용 정의

\[
T_{total}=T_{offline}+T_{bank}+T_{fit}+T_{selection}+T_{inference}+T_{required\ restart}.
\]

- 단독 배포에 필요한 모든 단계가 대상이다. 사전 고정한 재시도 정책의 실패 비용도 포함한다.
- 학습 없는 static에 불필요한 SMC 비용을 부과하지 않는다. static geometry setup도 숨기지 않는다.
- 연구에서 공유한 bank 비용과 각 방법을 단독 실행할 때의 비용을 함께 공개한다.
- 평가 전용 reference·risk audit은 배포 비용과 별도 표기하되 **연구 총비용에는 포함**한다.
- 실제 알고리즘이 reference에 의존하면 그 비용은 배포 비용에도 포함한다.
- `cost * M2`와 `cost * (M2-mu²)`를 혼동하지 않는다. plug-in 외삽을 실제 fixed-precision 측정이라고 부르지 않는다.
- warmup·I/O·import·archive 포함 범위를 고정한다. throughput 점검과 논문 timing은 별도다.
- CPU/BLAS thread·dtype·batch·하드웨어·method 순서를 고정하고 순차 timing한다.
- timeout은 censoring·미자격으로 기록한다. 좋은 실행만의 시간비를 전체 성능으로 보고하지 않는다.

### 7.4 첫 제한 사이클의 자원 상한 기본안

| 단계 | 새 full payoff 평가 상한 | 계산 wall 상한 | 초과 시 |
|---|---:|---:|---|
| FR0 | 0, 문헌·inventory 중심 | active 검토 1–2일 배정 | 기여 불명확이면 구현 확대 중단 |
| FR1 | 2,000,000 equivalent 이하 | 30분 | oracle 단순화/불명확; 임의 증량 금지 |
| FR2 | 8,000,000 | 1시간 | 기준 없는 셀 분리; 다른 reference 연속 발명 금지 |
| FR3 | 20,000,000 | 2시간 | 후보 중단 또는 한 번의 작은 수정 |
| FR4 | 60,000,000 | 3시간 | development 결론; confirmation 자동 진입 금지 |

이는 실측 예측이 아니라 **연구비 손실을 제한하는 제안 cap**이다. full/scalar/gradient/density 호출을 따로 센다.
첫 사이클 physics 상한 합계는 90,000,000회다. wall·메모리·평가수 중 먼저 도달한 제한을 따른다.
tests·문헌·감사·실패 복구 비용은 별도 기록하며 이 표 밖이라고 무제한 허용하지 않는다.
실측 preflight가 cap 안의 유효 비교를 불가능하게 하면 표본·셀을 몰래 줄이지 않고 protocol을 다시 결정한다.

FR5/FR6 예산은 pilot 기반 job 수·메모리·필요 precision을 확인한 뒤 별도 봉인한다.
FR0 job matrix에는 FR2의 bank×family 4arms, FR3의 목적×기하 4arms와 q0/static,
FR4의 최종 후보 1개·필요 control·baseline·전체 학습 반복을 전부 합산한다.
원인 실험의 모든 arm을 모든 후속 셀에서 계속 반복하지 않는다. FR3에서 후보 하나를 고정한 뒤
FR4는 primary 후보와 필요한 comparator만 남겨 cap 안에 드는지를 preflight로 검사한다.
현재 요청은 로드맵 작성이므로 실험·클라우드 과금·장시간 자동 실행 승인이 아니다.
노트북 CPU float64로 시작하며 GPU/RunPod 도입을 선행조건으로 두지 않는다. 기존 covariance에서는
physics보다 density가 지배한 사례가 있으므로 실제 profiler로 병목을 확인한다.
microbatch와 실제 RSS를 검사한다. 예를 들어 N4096, batch8192의 2N float64 좌표만 약512MiB이며
중간 tensor까지 포함하면 훨씬 커진다. 운영 중단은 통계 실패와 구분한다.

### 7.5 일정의 기본안

첫 방향 재검증은 active 연구 10일 내외의 제한 사이클을 제안한다.
1–2일차 FR0, 3일차 FR1, 4–5일차 FR2, 6–7일차 FR3, 8–10일차 FR4와 계속/중단 판단이다.
이는 성공 기한이나 자동 예약이 아니다. 앞 gate가 막히면 다음 날의 대형 작업을 시작하지 않는다.
FR5 이후의 장시간 실험은 첫 사이클의 정보 가치와 비용을 보고 별도 결정한다.

## 8. 오류 방지와 단계별 검토 체크리스트

각 단계 종료 시 독립 코드/수학 검토 후 다음을 기록한다.

- [ ] estimand·payoff·N·parameter·epsilon·discount binding이 맞는가?
- [ ] log-domain overflow/underflow·NaN·Inf를 명시적으로 다뤘는가?
- [ ] target을 바꾸는 clipping·숨은 truncation·잘못된 compensator가 없는가?
- [ ] density가 normalized full mixture이고 sampler와 같은 분포인가?
- [ ] geometry control의 rank/spectrum/capacity와 실제 비용이 기록됐는가?
- [ ] covariance 변경 후 기존 identity-mixture conditional sampler를 잘못 재사용하지 않았는가?
- [ ] training/selection/final/reference의 자료·seed·artifact 역할이 분리됐는가?
- [ ] event 위험과 M2 평가 위험, m1²과 m2를 혼동하지 않았는가?
- [ ] QMC point/SMC particle/final block을 독립 whole로 세지 않았는가?
- [ ] 선택 후의 최소 empirical risk를 불편한 최종 성능으로 보고하지 않았는가?
- [ ] CI·다중성·공유 reference·optional stopping을 설계와 맞게 처리했는가?
- [ ] 낮은 RSE/좋은 ESS/acceptance/ancestry 하나로 coverage를 선언하지 않았는가?
- [ ] 실패·timeout·미자격·source 변경·replay 비용을 누락하지 않았는가?
- [ ] finite-grid 성공을 continuum·시장 예측·production 인증으로 확대하지 않았는가?
- [ ] 새 lemma의 가정과 novelty를 기존 문헌과 대조했는가?

소프트웨어 순서는 작은 oracle → 관련 회귀 → Ruff/Mypy → 전체 테스트다.
timing 중 전체 pytest나 다른 무거운 실험을 병렬 실행하지 않는다.
임의 source의 실험 중 수정 대신 버전을 분리하고 checkpoint/source digest를 고정한다.
환경 의존성은 필요성이 확인되지 않으면 업그레이드하지 않는다.

## 9. 채택·중단 결정표

| 관측 | 해석 | 행동 |
|---|---|---|
| bank 변경만 효과, family 변경은 무효 | 탐색 자료가 주병목일 가능성 | bank 개선만 남기고 구조 모델 주장 축소 |
| 같은 bank에서 family 차이 재현 | 표현 제한 가설 지지 | 원인이 되는 작은 family 변경만 진행 |
| R-P와 R-C가 동일 | 물리 특화 추가 가치 미입증 | Volterra 구조 우월성 주장 중단 |
| event와 risk 목적이 동일 | risk 학습 추가 가치 미입증 | 더 단순한 목적 사용 |
| q0/static이 새 후보보다 유리 | 복잡도 가치 없음 | baseline 채택 또는 후보 폐기 |
| M2만 불안정, direct 반복 오차는 검증 가능 | 위험 기하 주장에 한계 | 그 claim 보류; 계산 성과는 별도 근거로 평가 |
| direct·reference 모두 불안정 | accuracy 미자격 | 속도 주장 중단, 예산 내 불명확으로 종료 |
| 낮은 분산이나 큰 이동만 관측 | proxy 개선 | 알고리즘 성공으로 집계하지 않음 |
| 새 task에서 개선 소멸 | 개발 overfit 또는 제한된 유효 범위 | 확인 실패 공개, 자동 재튜닝 금지 |
| 한 번의 제한 재설계 후에도 원인·이득 없음 | 추가 투자 근거 부족 | 계열 중단·문제 전환 검토 |
| continuum 상수만 개선 | 이론 수치 보강 | 주 알고리즘 성공이나 continuum 인증으로 승격 금지 |

## 10. 논문 산출물의 최소 구성

1. **문제·기여 한 문장:** 새 알고리즘/정량 구조/실용 이득 중 실제 입증한 것.
2. **알고리즘 하나:** 입력·학습·선택·fallback·sampling·full density·비용의 pseudocode.
3. **수학 계약과 필요한 정리:** 기존 항등식과 새 lemma를 분리.
4. **핵심 메커니즘 그림:** 사건 mass와 위험 mass의 차이 및 control 대조. 불확실성 표시.
5. **전체 학습 결과:** 오차·실패·비용 분포, frozen-q 성능과 구분.
6. **accuracy-qualified 비용 곡선:** 비용이 적더라도 부정확한 방법은 speedup 비교에서 분리.
7. **새 task 확인과 한계:** stress 실패·격자 민감도·실무 범위 포함.
8. **재현 패키지:** protocol/source/config/seed/artifact digest·실패·감사·실행법.

정적 guide만으로 실용 이득을 얻으면 neural이라는 이름을 보존하려고 학습을 추가하지 않는다.
상위 저널 가능성은 위 결과와 closest-work 차이에 따라 판단한다. 현재 게재 수준을 보장하지 않는다.

## 11. 근거와 선행연구 경계

### 내부 근거

- [10월7일 구조 검토·원래 개선계획](../reviews/MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md)
- [작은 격자 전체 학습 결과](../reviews/SMALL_GRID_MODEL_RESTART_RESULT_2026-10-20_KO.md)
- [dictionary 학습 결과](../reviews/PHYSICAL_DICTIONARY_MODEL_LEARNING_RESULT_2026-10-20_KO.md)
- [잔차 구조·known-answer 검토](../reviews/EXACT_RESIDUAL_STRUCTURAL_ORACLE_RESULT_2026-10-21_KO.md)
- [잔차 수학 계약](../theory/EXACT_RESIDUAL_STRUCTURAL_CONTRACT_2026-10-21_KO.md)
- [계보·오류 calibration 반례](../reviews/GENEALOGY_ERROR_CALIBRATION_RESULT_2026-10-23_KO.md)
- [conditional intervention·부호 평균](../reviews/CONDITIONAL_INTERVENTION_AND_TWO_TRACK_RESULT_2026-10-26_KO.md)
- [다중 시점 covariance](../reviews/MULTITIME_RESIDUAL_BROWNIAN_RESULT_2026-10-26_KO.md)
- [grouped density·window의 미평가 상태](../reviews/GROUPED_DENSITY_AND_WINDOW_RESULT_2026-10-27_KO.md)
- [최신 HMC·continuum 한계](../reviews/P2_HAMILTONIAN_AND_TAIL_BOUND_RESULT_2026-10-28_KO.md)

### 외부 원문: 확인한 경계와 후속 의무

- [McCrickerd–Pakkanen, Turbocharging Monte Carlo pricing for the rough Bergomi model](https://arxiv.org/abs/1708.02563):
  조건부 lognormal 구조·분산 감소·실행시간을 고려한 rough Bergomi 계산의 선행연구다.
- [Uribe 등, Cross-entropy-based importance sampling with failure-informed dimension reduction](https://arxiv.org/abs/2006.05496):
  CE와 intrinsic low-dimensional failure 구조 결합은 이미 연구됐다.
- [Agapiou 등, Importance Sampling: Intrinsic Dimension and Computational Cost](https://arxiv.org/abs/1511.06196):
  target/proposal 차이와 IS 비용을 연결하는 일반 이론의 선행 경계다.
- [Llorente 등, Optimality in Noisy Importance Sampling](https://arxiv.org/abs/2201.02432):
  noisy evaluation의 변동을 고려한 최적 proposal 설계가 기존 주제임을 보여준다.

이번 문서 작성에서는 위 원문의 서지·초록과 내부 이론/결과를 대조했다. 이는 체계적인 최신 문헌고찰이나
novelty 인증이 아니다. FR0에서는 가까운 논문 본문·수식·가정을 읽고, 이후 원고 전에 다시 최신 연구를 점검한다.

## 12. 첫 실행 묶음과 현재 완료 상태

바로 다음 구현은 다음 순서다.

1. FR0의 estimand/claim ledger·task·role·budget schema와 테스트.
2. 기존 adapter 목록을 실제 API와 binding 검사로 연결.
3. FR1에서 필요한 negative control만 추가하고 기존 oracle 회귀 실행.
4. FR2 reference-only pilot을 봉인해 실행한 뒤 평가 가능한 문제와 원인 실험을 결정.
5. 그 결과 없이 새 covariance·operator·큰 학습부터 시작하지 않음.

현재 체크포인트:

- [x] 사용자가 승인한 주 목표 변경을 문서에 명시.
- [x] 과거 P2 미해결·실패 기록을 보존하는 정책 명시.
- [x] 중복 회피, 수학·통계·비용·단계 gate와 중단 기준 작성.
- [ ] FR0 실행 계약·schema 구현 및 봉인.
- [ ] FR1–FR7 실제 실행·검증·최종 결정.

**이 문서 작성으로 모델 개선·확인 실험·논문 완성이 이루어진 것은 아니다.
이번 요청 범위는 로드맵 작성이며, 기존 코드·실험 설정·README·과거 산출물·Git history는 변경하지 않는다.**
