# 감사 결과에 따른 연구 재설계 및 단계별 실행계획

작성일: 2026-09-27

계획 기준: `3f9047d8e586cb707cf8fe2a889f928bef9fb0f0`

대상: Neural Path Integral의 V16 이후 연구
문서 상태: **실행 전 계획서. 아래 작업과 실험을 완료했다는 보고서가 아니다.**

주 근거는 [연구 방향 재검토 보고서](../reviews/RESEARCH_REORIENTATION_REVIEW_2026-09-27_KO.md), [수치·감사 재검증 자료](../reviews/V16_REORIENTATION_EVIDENCE_WITH_OPERATOR_2026-09-27.json), [재현 스크립트](../reviews/reproduce_v16_review.py)다. 감사 시작 시 소스는 `da91604`이며, 이 계획의 기준 커밋에는 해당 감사 문서가 포함되어 있다.

## 1. 결정 요약

### 1.1 무엇을 연구할 것인가

주 연구 질문을 다음으로 고정한다.

> Gaussian Volterra 모형의 희귀사건을 조건부 적분한 뒤, 남은 분산을 만드는 경로 방향을 찾아 최소한의 측도변환으로 줄일 수 있는가? 그 방향을 찾고 학습하는 비용까지 포함해 강한 기존 방법보다 유리한가?

권장 작업용 제목은 **Conditional Risk Geometry and Reliable Importance Sampling for Gaussian Volterra Rare Events**다. 제목과 모델 이름은 결과보다 먼저 확정하지 않는다.

선택하는 조합은 다음 두 트랙이다.

- **주 트랙 A:** 조건부 second-moment 기하 → 계산 가능한 방향 선택·적합 → 최소 transport → 실제 비용 검증.
- **지원 트랙 B:** 유한격자 추정 오차와 mesh bias 분리 → 필요한 범위의 Volterra 특화 오차 분석.

신경망은 주 트랙의 필수 구성요소가 아니다. 먼저 단일 task 추정기를 정리하고, 여러 서로 다른 task에 반복 사용하는 경우에 한해 별도 amortization 실험으로 복귀한다. 양자 요소, 대형 flow, 새로운 route의 계속된 추가에는 현재 예산을 배정하지 않는다.

### 1.2 무엇이 달라지는가

현재처럼 셀별로 좋은 구성요소를 더하는 대신, **좋아진 이유와 실패한 이유를 설명할 수 있는 하나의 알고리즘**을 만든다. 논문 성공 조건은 코드 증가나 내부 gate 개수가 아니다.

1. 정의와 구현이 일치하는 추정기.
2. 강한 비교기와 같은 조건에서 확인한 유효 범위.
3. 기존 IS/FIS/noisy-IS 원리를 넘는 명확한 기여.
4. 그 기여에 맞는 완전한 증명 또는 충분히 강한 수치 증거.

오류가 전혀 없거나 최상위 저널에 반드시 게재된다는 보장은 할 수 없다. 대신 오류를 발견하면 다음 단계의 주장을 차단하고, 실패를 보존하며, 재현 가능한 근거로 진행 여부를 결정한다.

### 1.3 실행 순서

| 단계 | 핵심 산출물 | 다음 단계 진입 조건 |
|---|---|---|
| R0 | 신뢰 가능한 감사기·weighted conditional CE·공통 통계/비용 계약·주장 정정표 | 잘못된 결과를 거부하고 비교 정의가 일치 |
| R1 | 네 병목 가설을 구별한 진단 보고서 | 원인별 증거 또는 명시적 미해결 판정 |
| R2 | 원인에 맞춘 최소 candidate 하나 | 독립 개발 검증에서 정확성 유지와 비용-분산 개선 신호 |
| R3 | 미사용 task·독립 학습 반복·실제 시간 confirmation | 사전 기준에 따른 성공/부분 성공/실패 판정 |
| R4 | mesh 및 실무 대상의 오차-비용 분석 | 주장하는 대상과 총 오차 범위가 일치 |
| R5 | 논문 기여표·증명·재현 패키지·투고 판단 | 새 기여와 증거가 일치하고 중요한 미해결 공백이 없음 |

주 의존관계는 `R0 → R1 → R2 → R3 → R5`다. R4의 기초 coupling 검증은 R0 이후 병행할 수 있고, 최종 대규모 refinement는 R2 이후 진행한다. R5에서 연속시간 효율을 주장하려면 그 주장에 해당하는 R4 증명을 완료해야 한다. **일반적인 joint mesh/noise 정리 전체를 주 트랙의 무조건적 선행조건으로 삼지는 않는다.**

## 2. 출발점: 유지할 자산과 폐기할 해석

### 2.1 감사로 확인한 현재 위치

| 확인된 기록 | 앞으로의 해석 |
|---|---|
| 7개 최종 셀 중 4개에서 work-proxy 기준 우위, 3개는 fallback | 제한된 개발 기록. 전체 우월성이나 실제 시간 가속의 증거는 아님 |
| canonical 셀의 100-query proxy 비율 약 5.978 | 현재 비교 정의에서의 점추정. 새 weighted conditional CE와 실제 시간으로 재검증 |
| 높은 η / 강한 음의 ρ / joint regime의 비율 약 0.471 / 0.605 / 0.639 | 필수 실패 진단 대상. 결과표에서 제외하지 않음 |
| 최종 variant마다 하나의 학습된 proposal, 그 아래 32개 evaluation cluster | 32번의 독립 학습 성공으로 해석하지 않음 |
| operator 실험의 함수평가 감소, 세 seed의 `cold 시간 / neural 초기화+보정 시간`은 약 0.818 / 0.676 / 0.831 | 이 기록은 teacher/training 비용도 제외한 비교이며, 신경망의 end-to-end 시간 이득은 아직 입증되지 않음 |
| 과거에 반복 확인한 OOD 셀 | 앞으로는 development/regression 셀. 미사용 holdout으로 재사용하지 않음 |

숫자의 상세 정의와 원본 파일 연결은 감사보고서를 기준으로 한다. 이번 계획은 새 성능 실험 결과를 추가하지 않는다.

### 2.2 유지할 것

- Gaussian Volterra 유한격자 법칙과 독립 가격 driver의 조건부 적분.
- 정규화 밀도를 계산할 수 있는 Gaussian/finite-rank mixture proposal.
- 실제 sampling density를 사용한 ordinary IS와 defensive mass.
- V14/V16, 실패 결과, 기존 비교기를 포함한 역사적 재현 자료.
- seed ledger, provenance, density·조건부 payoff 관련 기존 테스트.

### 2.3 당장 중단할 것

- `passed: true`, `proved`, 파일 존재만으로 과학적 결론을 확정하는 감사.
- 성능이 좋은 셀만 남기거나 불확실한 비교기를 탈락시켜 우위처럼 보이게 하는 집계.
- 고정된 하나의 proposal 재사용을 서로 다른 금융 task의 amortization으로 표현하는 것.
- 새로운 basis/mixture/신경망을 원인 진단 없이 한꺼번에 도입하는 것.
- 유한격자 exactness를 연속시간 무편향성 또는 현실 시장 예측력으로 확대하는 것.

## 3. 수학·실험의 비협상 계약

### 3.1 추정 대상

초기 대상은 기존 convention의 만기 left-tail bounded digital 확률이다. 모수, strike, maturity, forward variance, 이자율 convention, N, 이산화 scheme을 task의 일부로 고정한다.

참조 Gaussian 밀도를 `p_N`, 조건부 payoff를 `g_N`라 두면

\[
0\le g_N(z)\le1,\qquad \mu_N=E_{P_N}[g_N(Z)],\qquad Z\in\mathbb R^{2N}.
\]

원래 `3N` 난수 중 독립 가격 driver `N`개를 적분한 것이다. 남은 두 local 좌표는 동일한 volatility Brownian의 셀 내 관측을 표현한다. 임의로 독립 물리 driver 두 개로 바꾸지 않는다.

무이자 convention에서는 `I_N=Σ V_i Δt`, `J_N=Σ √V_i ΔW_i`에 대해

\[
g_N=\Phi\!\left(\frac{\log(K/S_0)+I_N/2-\rho J_N}
{\sqrt{(1-\rho^2)I_N}}\right).
\]

이는 기존 scheme의 adapted left-endpoint convention과 일치해야 한다. 다른 rate나 monitoring 정의를 도입하면 별도 task version과 oracle 테스트가 필요하다. 기본 영역은 `0<H<1/2`, `η>0`, `ξ>0`, `T>0`, `K>0`, `|ρ|<1`로 두고, 퇴화 경계는 별도 구현 없이 포함하지 않는다.

### 3.2 최종 추정과 안전장치

학습·선택을 끝낸 실제 proposal `q`에서 독립 최종 표본을 뽑아

\[
\widehat\mu_N=\frac1M\sum_{i=1}^M g_N(Z_i)\frac{p_N(Z_i)}{q(Z_i)}
\]

를 계산한다. 모든 mixture 성분을 포함한 전체 `q`가 분모다. 뽑힌 component의 밀도만 사용하는 방식은 허용하지 않는다.

`q≥δp_N`, `δ>0`이면 표본 기여 `X=g_N p_N/q`는 `0≤X≤1/δ`이고 `E_q[X²]≤μ_N/δ`다. 이는 유한 분산과 concentration에 유용하지만, 상대 오차에 대한 bound는 희귀 확률이 작을수록 약해질 수 있다. **defensive mass가 있다는 이유만으로 상대 효율을 보장하지 않는다.**

- 최종 inference에서 self-normalization, likelihood clipping, 유리한 표본 제거를 금지한다.
- 학습 목적의 가중 평균에 정규화된 importance weight를 쓰는 것은 허용한다. 최종 SNIS와 구분한다.
- basis나 covariance를 바꾸는 것은 proposal 변경이다. 참조 Brownian 법칙과 target은 고정한다.
- adaptive proposal이라도 최종 평가 전에 freeze한다. 평가 중 적응을 추가하려면 별도의 추정·분산 이론이 필요하다.
- finite-dimensional static transport를 무조건 adapted drift라고 부르지 않는다. static density와 Girsanov control은 서로 다른 정당화 경로다.

### 3.3 상태와 용어

결과는 `integrity`, `semantic_validity`, `statistical_evidence`, `theory_status`, `performance`를 별도 필드로 저장한다. `pass / fail / unresolved / not_applicable`을 구별한다.

- `exact_density`: 명시된 유한차원 proposal의 정규화 밀도를 계산 가능하다는 뜻.
- `finite_grid_unbiased`: 명시한 적분 가능성·독립성 조건에서 `μ_N`에 불편하다는 뜻.
- `holdout`: candidate·하이퍼파라미터 선택에 결과를 사용하지 않은 task.
- `external_reproduction`: 독립 실행의 환경·명령·입출력 근거가 있음. 파일 존재만으로 부여하지 않음.
- `confirmed`: 사전 등록된 특정 주장과 범위가 검증됐다는 뜻. 전역적인 우월성 표지가 아님.

## 4. 감사 지적사항을 작업으로 변환한 추적표

| ID | 문제 | 조치 | 검증 및 완료 근거 |
|---|---|---|---|
| A01 | stored pass와 ratio를 신뢰하는 감사 | R0.2에서 원시 통계·설정으로 재산출 | NaN, stale-pass, 잘못된 단위·키·참조 거부 |
| A02 | 기존 CEM과 표준 weighted CE의 차이 | R0.3에서 별도 baseline 추가 | 해석 가능한 Gaussian oracle, weighted moment 검증 |
| A03 | raw/conditional payoff 비대칭 | R0.3 및 R1의 matched-conditioning 비교 | conditioning 효과와 새 proposal 효과 분리 |
| A04 | comparator qualification 불일치 | R0.4 공통 판정기 | candidate/baseline 경로가 같은 입력에 같은 판정 |
| A05 | cost proxy와 실제 비용 불일치 | R0.5 실제 비용 schema·runner | fitting/selection/inference 포함 시간·메모리 |
| A06 | 단일 fit, 반복 사용된 OOD, 불확실한 reference | R3 학습 반복·새 task·reference 계약 | fit-level 통계와 독립성 manifest |
| A07 | T16-12B 양측 bound 상수 | R0.6 양측/단측 분리 | 정리-코드-테스트의 동일 수식 |
| A08 | T16-5/9/11 증명 연결고리 | R0.6 상태 정정, R4 필요한 범위 재증명 | 가정·의존 정리·증명 단계 ledger |
| A09 | 저차원 적합과 잔여 분산 | R1.2/R1.3, R2 | rank·basis·objective의 독립 대조 |
| A10 | mode 탐색 실패 가능성 | R1.4 | 독립 particle bank·ancestry·tail 기여 분석 |
| A11 | neural end-to-end 우위 미확인 | R5의 조건부 확장으로 이동 | teacher·training 포함 task-batch break-even |
| A12 | 유한격자 결과와 연속/실무 주장 혼합 | R4 | mesh bias·sampling error·용도 구분 |

## 5. R0 — 검증 기반과 공정한 비교를 먼저 복구

### R0.1 역사 보존과 새 실행 계약

1. 기존 결과, hash-bound config, theorem 문서를 덮어쓰지 않는다. 정정은 별도의 superseding claim ledger로 연결한다.
2. 새 결과에 `schema_version`, source commit, source dirty 여부와 patch digest, config digest, dependency versions, task ID, method ID, proposal digest를 기록한다.
3. `task × method × training_rep × evaluation_rep × stage × level`을 논리 키로 사용한다. 중복은 덮어쓰기 대신 오류다.
4. 기존 `SeedLedger`에 training/selection/final/reference/task-generation 역할을 명시한다. 임의 seed 덧셈 관례를 늘리지 않는다.
5. 같은 bank 또는 공통 난수를 쓴 비교는 의도된 dependence로 기록한다. seed가 다르다는 것만으로 모든 통계적 독립성을 증명했다고 쓰지 않는다.
6. deterministic replay와 통계적 재현을 구분한다. 하드웨어 차이의 부동소수점 오차 허용 범위를 명시한다.

### R0.2 의미 검증이 가능한 감사기

대상: `experiments/g11_v16_final_policy_audit.py`, `src/path_integral/provenance.py`, `src/path_integral/v15_result_audit.py`를 참고하되, 새 schema용 검증기를 독립 구현한다. 과거 schema는 legacy adapter로 읽는다.

검증 순서:

1. schema/type/range 확인 → 원본 및 재귀 binding hash 확인 → 참조 그래프의 누락·cycle·중복 확인.
2. task, N, payoff, convention, seed 역할, proposal identity가 비교 가능한지 확인.
3. 저장된 cluster/replicate 통계에서 평균, 표준오차, RSE, 비교 차이, 비용비를 재산출.
4. 재산출값과 표시값의 불일치를 탐지. 표시된 `passed`는 입력 근거로 사용하지 않음.
5. theory ledger의 범위와 실제 method 구성을 대조하되, 감사기가 증명을 자동 인증한다고 주장하지 않음.
6. 수치 유효성과 성능 판정을 분리하여 최종 decision trace 저장.

통계 저장 최소 요건:

- IID IS: 표본 수, 평균, 중심 제곱합, cluster 경계, likelihood normalization 통계, underflow/비유한 값 카운트.
- randomized QMC: 각 독립 scramble의 추정값과 점 수. Sobol 점들을 독립 표본처럼 SE 계산하지 않음.
- SMC reference: 독립 전체 SMC 실행별 출력과 설정. resampled particle을 독립 replicate로 취급하지 않음.
- 각 통계의 단위: 개별 기여 `X`, cluster mean, final estimator 중 무엇인지 명시.

가능하면 replay 가능한 seed와 proposal을 함께 저장한다. 전체 표본 저장을 강제할 필요는 없지만, 충분한 원시 요약도 없는 과거 결과는 `legacy_unverifiable`로 남긴다. 누락 정보를 추측해 pass로 바꾸지 않는다.

필수 adversarial 테스트:

- `NaN`, `Inf`, 음수 시간·SE·variance, bool 형태 sample count, 비정수 count, 0개 표본.
- accuracy z/RSE/normalization z를 극단값으로 바꾸고 stored-pass 유지.
- ratio만 NaN으로 바꾸기, 비용 방향 뒤집기, tolerance·query horizon·N 바꾸기.
- 같은 task/method 키 중복, 누락 comparator, reference hash만 정상이고 하위 config 변조.
- train/final stream 중복, proposal 또는 task가 바뀌었는데 오래된 summary 사용.
- SE=0: 해석적으로 상수인 경우와 희귀사건을 한 번도 보지 못한 경우를 구분.
- valid 해시를 가진 **의미상 잘못된 fixture**와 해시 자체가 틀린 fixture를 별도 검사.

저장 JSON은 비표준 NaN/Infinity를 허용하지 않는다. variance 누적은 안정적인 online/merge 방식과 작은 독립 oracle을 비교한다. 수치 문제를 넓은 clipping으로 감추지 않는다.

### R0.3 weighted CE 및 조건부 비교기

기존 `src/path_integral/baselines/cem.py`는 과거 재현용으로 유지하고, baseline ID에 unweighted heuristic임을 표시한다. 새 구현은 별도 `weighted_conditional_ce` 계열로 둔다.

현재 proposal `q_t`에서 뽑은 표본에 대해 최종 조건부 target 적합의 가중치는

\[
w_i=g_N(z_i)\,p_N(z_i)/q_t(z_i).
\]

중간 rare-event CE 단계에서 elite set `A_t`를 사용하면 `w_i=1_{A_t}(z_i)p_N(z_i)/q_t(z_i)`다. 최종 target과 중간 smoothing/elite schedule을 구분해 기록한다. Gaussian 평균·공분산은 이 weight의 정규화된 moment로 적합한다. 전체 Gaussian mixture에서 뽑았다면 weight의 분모도 전체 mixture다.

단계적 baseline family:

1. 조건부 mean-shift CE.
2. 조건부 low-rank covariance CE: 학습량에 비해 과도한 covariance를 방지하는 고정 shrinkage/eigenvalue 제약.
3. 작은 Gaussian mixture CE: pilot에서 필요성이 확인될 때만 사용하고 component 수·budget을 고정.

eigenvalue 제약은 명시적 proposal regularization이지 likelihood clipping이 아니다. 안정화된 covariance를 실제 sampling과 density 양쪽에 동일하게 사용한다.

또한 기존 적응 코드의 `softmax(β log(gp/q))`는 β와 q를 고정해 해석하면 `q^(1-β)(pg)^β`에 해당하는 weight tempering이다. 일반적으로 target-potential tempering `p g^β`와 같지 않다. 후자를 의도하면 `log(p/q)+β log g`가 필요하다. 표본 ESS로 선택한 β의 적응성도 별도 기록한다. 기존 방식을 무조건 오류라고 폐기하지 말고 **목표가 무엇인지 명명·분리·테스트**한다.

검증:

- 1차원 `X~N(0,1), X>a`에서 target mean `φ(a)/(1-Φ(a))` 및 variance `1+aλ-λ²`와 대조.
- 여러 shifted proposal 아래 weighted moment가 같은 oracle에 수렴하는지 확인. deterministic quadrature 테스트와 stochastic diagnostic을 분리.
- 작은 차원 Gaussian density를 독립 dense covariance 계산과 비교.
- 자연 proposal, mixture weight 경계, `δ=1`, 작은 positive δ, 상수 payoff, log-tail 테스트.
- 동일한 local 경로에서 독립 가격 noise를 반복 적분한 평균과 conditional CDF를 비교. 한 경로의 단일 raw payoff와 CDF가 같아야 한다는 잘못된 테스트는 금지.
- raw CE와 conditional CE를 각각 matched raw/conditional evaluation으로 비교. 주 비교에서는 모든 가능한 방법에 동일한 conditioning을 제공.

### R0.4 comparator와 reference 판정 통합

공통 판정 함수가 candidate와 comparator에 동일하게 적용돼야 한다. 판정 결과는 최소한 다음을 분리한다.

- target/proposal/통계가 올바른가.
- reference와 차이가 해석 가능한가.
- 차이를 논할 정도로 comparator와 reference가 충분히 정밀한가.
- 성능 비교가 가능한가, 아니면 unresolved인가.

큰 SE 때문에 z가 작아지는 것은 정확성 입증이 아니다. `z≤4`만으로 equivalence를 선언하지 않는다. reference 불확실성을 포함한 차이의 구간과 사전 equivalence margin을 사용하고, 이 절차도 유한 표본에서 완전한 무편향성을 증명하는 도구가 아니라는 점을 명시한다.

불확실한 baseline은 표에서 제거하지 않는다. reference 개선 또는 비교 불확실성 해소에 필요한 비용을 기록하고, 해결되지 않으면 해당 우위 주장을 보류한다.

### R0.5 실제 비용 계측

모든 방법에 동일 schema를 적용한다.

| 단계 | 비용에 포함할 것 |
|---|---|
| offline | teacher, 공통 basis 사전학습, 모델 학습; 재사용 가정 별도 |
| fit | pilot, SMC, gradient, covariance/basis fitting, optimizer |
| selection | candidate bank 평가 및 validation |
| inference | sampling, simulator, conditional payoff, 전체 density, 집계 |
| operational | 필요한 load/compile/I/O; 포함·제외 두 관점을 표시 가능 |

wall time, CPU time, peak memory, thread 수, CPU/OS/library, precision, batch size, warmup, power mode를 기록한다. 노트북 timing은 온도·백그라운드 작업 영향을 점검하고 method 순서를 무작위화한다. 시간 측정 실험을 같은 머신에서 동시에 돌리지 않는다.

proxy는 보조 설명 지표로만 유지한다. dense/low-rank projection의 `d×r`, simulator, k-means, SMC mutation 등 누락된 비용은 실제 시간으로 드러나야 한다. 연구 과정의 전체 탐색 비용과 한 번 배포할 때의 fit/selection 비용을 구분해 둘 다 공개한다.

**주의:** 계산시간과 표본값이 연관될 수 있으므로 시간이 끝날 때까지 나온 표본만 평균 내는 방식으로 fixed-budget 실험을 만들지 않는다. 독립 calibration에서 표본 수를 정한 뒤 그 수를 끝까지 실행하고 실제 시간을 보고한다. 시간 초과는 별도 operational failure로 남긴다.

### R0.6 이론 정정과 claim ledger

1. `defensive_proposal_selection.py`의 양측 동시 오차 및 oracle bound를 단측 UCB와 구분한다.
2. Maurer–Pontil의 해당 one-sided bound를 두 방향·J개 후보에 적용하는 양측 구현은 `log(4J/γ)`로 명시한다. 단측만 쓰는 다른 경로와 혼용하지 않는다.
3. 후보는 validation 전에 고정되어야 한다. 각 candidate의 risk 관측값은 `Y_j=g²(p/q_j)(p/G)`이고, `q_j≥δ_j p`, `G≥δ_G p`라면 상한은 `1/(δ_j δ_G)`다.
4. 같은 validation 표본을 후보 간 공유하는 것은 union bound를 막지 않는다. 그러나 IID 표본용 정리를 의존하는 SMC 입자나 QMC 점에 그대로 적용하지 않는다.
5. bound가 너무 커 선택 보장이 무의미하면 `vacuous`로 기록한다. 경험적 winner와 이론적으로 분리된 winner를 구별한다.
6. T16-5/9/11은 근거별 검토 상태로 재분류한다. 반례가 없는 증명 공백을 `false`라고 쓰지 않는다.
7. Gaussian measure equivalence, finite-grid identity, small-noise LDP, uniform mesh/noise, local optimizer 수렴을 별도 명제로 관리한다.

근거: [Maurer–Pontil, Theorem 4 / Corollary 5](https://arxiv.org/pdf/0907.3740). 기존 수식을 수정했다고 실제 coverage가 모든 실험 상황에서 보장되는 것은 아니다. boundedness·independence·고정 후보 조건을 함께 검증해야 한다.

### R0 종료 gate

- 위 실패 fixture가 모두 거부되고 정상 fixture는 통과.
- CE·density·conditional payoff oracle과 공통 qualification 테스트 통과.
- 과거 결과는 보존되며 정정표에서 제한된 주장으로 재해석.
- 공통 runner의 두 방법 smoke test에서 비용·seed·원시 통계가 완전하게 저장.
- Ruff, Mypy, 관련 테스트 및 전체 회귀 테스트 결과를 구분해 기록.
- 이후 새 모델의 정확성/우월성 결론을 내릴 수 있는 입력 계약이 마련됨.

## 6. R1 — 성능 병목을 분리하는 작은 실험

### R1.1 실험 규모와 통제

개발 셀은 기존 canonical, K=2, regular H, 높은 η, 강한 음의 ρ, joint regime의 **6개**를 기본으로 한다. 모두 개발 데이터임을 표시한다.

- N=16에서 full-rank sanity check, N=32에서 현재 연구와 연결.
- 주요 비교는 독립 학습 seed 5개. 이는 원인 탐색용이지 낮은 실패율의 입증이 아니다.
- N=32 rank 후보는 8/16/24/48/64, N=16에서는 rank≤32만 허용.
- 전체 Cartesian product를 실행하지 않는다. 우선 3개 대표 셀의 rank sweep → 원인이 보이는 조건만 6개로 확장.
- 동일 bank로 fit만 바꾸는 **기하 진단**과, 각 방법이 bank부터 새로 만드는 **end-to-end 비교**를 별도 표로 작성.
- 진단에 공유 bank를 쓰더라도 최종 held-out evaluation은 bank와 독립적으로 생성.

### R1.2 conditioning 및 subspace ablation

첫 비교는 `raw payoff + 기존 proposal`, `conditional payoff + 같은 proposal`, `conditional payoff + 새 proposal`이다. 첫 차이는 conditioning, 두 번째 차이가 추가 proposal 기여다.

같은 bank, rank, covariance family, 학습 budget에서 다음을 대조한다.

1. 현재 고정 basis.
2. weighted PCA 계열.
3. 선행연구 정의에 충실한 FIS 계열.
4. 아래 second-moment 정보를 사용한 실험적 방향.

FIS 구현에 approximation을 쓰면 무엇을 생략했는지 이름과 문서에 명시하고 gradient 비용도 포함한다. 단순 PCA를 FIS라는 이름으로 바꾸지 않는다. Weighted CE/FIS의 원래 목적과 구현은 [Uribe et al.](https://arxiv.org/pdf/2006.05496)의 식과 대조한다.

### R1.3 conditional second-moment 진단

직교 Gaussian 좌표 `Z=(U,V)`에서 `V`는 reference로 남기고 `U`만 바꾸는 family를 분석한다.

\[
m_1(u)=E[g_N\mid U=u],\quad m_2(u)=E[g_N^2\mid U=u],\quad
q=q_U\phi_V,\quad r=q_U/\phi_U.
\]

\[
M_2(r)=E[m_2/r],\qquad
\inf_r M_2(r)=\big(E\sqrt{m_2}\big)^2,\qquad
r^*=\frac{\sqrt{m_2}}{E\sqrt{m_2}}.
\]

반면 `r_KL=m₁/μ_N`이면

\[
M_2(r_{KL})-\mu_N^2
=\mu_N E\!\left[\frac{\operatorname{Var}(g_N\mid U)}{m_1(U)}\right].
\]

이는 진단의 출발점이며 그 자체를 새 정리라고 주장하지 않는다. [Llorente et al.의 noisy-IS optimal proposal](https://arxiv.org/pdf/2201.02432)과 직접 비교한다.

실행 명세:

1. outer `U`를 알려진 density `G_U`에서 생성하고 `φ_U/G_U`를 기록한다.
2. 각 U에서 새 reference `V`를 독립 생성해 m₁, m₂와 conditional variance를 추정한다.
3. inner 수 16/64/256의 개발 sweep으로 진단이 안정되는지 확인한다. 희귀 영역에서는 부족할 수 있으며, 이를 zero floor로 해석하지 않는다.
4. `sqrt(hat m₂)`의 concavity 때문에 plug-in은 일반적으로 편향된다. 이를 무편향 floor 추정치나 엄밀한 lower certificate라고 부르지 않는다.
5. outer 평균의 제곱, m₁ 분모의 비율에도 추가 편향이 있다. analytic toy, 독립 outer 반복, inner-size 안정성, uncertainty를 함께 보고한다.
6. deep tail에서 nested diagnostic이 예산 내 식별력을 갖지 못하면 `unresolved`로 남기고 독립 final second-moment 비교를 우선한다.

이 floor는 **reference complement를 유지하는 family**에만 적용된다. 다른 subspace의 성분이나 full-rank safety가 포함된 V5 전체 mixture의 lower bound로 사용하지 않는다. nested subspace에서 이상적 floor가 비증가한다는 사실도 실제 finite-sample 학습 성능의 단조 개선을 뜻하지 않는다.

### R1.4 objective·mode·비용 진단

| 가설 | 통제 실험 | 지지하는 관측 | 지지하지 않는 관측 |
|---|---|---|---|
| subspace 누락 | 같은 bank·family에서 rank/basis만 변경 | 새 방향이 held-out M₂를 일관되게 감소 | rank 증가에도 변화 없고 불확실성만 큼 |
| KL/M₂ 목표 불일치 | 같은 basis·family·bank에서 목적만 변경 | M₂ 적합이 독립 평가에서도 개선 | 학습값만 감소하고 평가에서는 소멸 |
| mode 탐색 실패 | 동일 예산 독립 SMC bank 및 mutation/bridge 비교 | bank별 contribution cluster 차이와 재현된 개선 | ESS만 높아지고 최종 IS 품질 변화 없음 |
| 계산비용 초과 | 동일 proposal의 단계별 profile | density/projection/학습이 이득을 상쇄 | 측정 잡음 범위의 차이 |

SMC에서는 ancestry, unique ancestors, resampling 수, mutation acceptance, log-normalizer 경로, tail 기여 concentration을 저장한다. cluster 모양이나 높은 ESS만으로 모든 mode를 찾았다고 주장하지 않는다.

`defensive only / learned only / defensive+learned / defensive+safety+learned`를 분해한다. learned-only에 support·moment 문제가 있으면 진단 전용으로 제한한다. 한쪽 ablation만 δ 또는 총 예산이 달라지지 않도록 한다.

### R1 종료 gate

각 가설을 `supported / not_supported / unresolved`로 판정하고, 상호작용이 의심되면 가장 작은 추가 대조만 수행한다. 원인이 불분명하면 R2의 큰 모델로 넘어가지 않는다. 진단 예산을 모두 사용해도 식별 불가하면 scope 축소 또는 연구 전환을 결정한다.

## 7. R2 — 최소 candidate 하나 구현

### R2.1 공통 구조

초기 candidate는 가능한 한 다음 범위에 둔다.

\[
q_\theta(z)=\delta p_N(z)+(1-\delta)
\sum_{k=1}^{K_c}\pi_k\mathcal N(z;B a_k,\ I+B(\Lambda_k-I)B^\top),
\quad B^\top B=I.
\]

여기서 Λ는 positive definite인 rank-r covariance이고, K_c는 성분 수다. 처음에는 한 성분을 우선한다. 둘 이상은 mode 진단이 지지하고 selection 비용까지 이득이 있을 때만 허용한다. 이 표기는 전체 구현을 강제하는 확정안이 아니라 **가장 작은 비교 가능한 family의 기본안**이다.

δ는 양수로 고정하여 출발하고 개발 단계에서 정한 값만 final에 사용한다. 특정 δ가 모든 rare-event regime에 최적이라고 가정하지 않는다. small-noise 정리에서 δ가 ε에 의존할 때의 조건은 별도다.

### R2.2 원인별 구현 분기

**A. subspace가 주원인일 때**

- 기존 basis의 complement에서 pilot residual direction 후보를 생성한다.
- QR/SVD로 직교화하고 rank 증가로 실제 `p_N` 법칙이 바뀌지 않는지 검사한다.
- 고정 rank 증가와 adaptive 방향 추가를 같은 rank·budget에서 비교한다.
- 개발 validation에서 얻은 한계 M₂ 개선과 추가 fit/inference 비용으로 rank를 선택한다.
- gradient/sensitivity는 후보 생성 proxy다. 그 크기가 최적 M₂ 감소를 보장한다고 주장하지 않는다.

**B. objective 불일치가 주원인일 때**

알려진 독립 pilot density G 아래에서

\[
L(\theta)=E_G\left[\frac{g_N(Z)^2p_N(Z)^2}{q_\theta(Z)G(Z)}\right]
=M_2(q_\theta)
\]

를 사용한다. Gaussian family와 defensive mass를 유지하고 positivity가 보장되는 parameterization으로 최적화한다.

- log-integrand는 `2 log g + 2 log p − log qθ − log G`다.
- Monte Carlo 목적의 log를 최적화해도, 그 log 값이 원래 risk의 무편향 추정치라고 쓰지 않는다.
- m₂의 미지 정규화 상수나 noisy network 출력을 최종 density 분모로 직접 쓰지 않는다.
- gradient를 analytic toy 및 finite difference와 대조한다.
- 같은 학습 표본에서의 과적합은 독립 selection과 final evaluation으로 검출한다.

**C. mode 누락이 주원인일 때**

- bridge/mutation/독립 bank 배분을 먼저 개선한다.
- 여러 bank를 합칠 때 실제 proposal의 mixture density와 모든 학습비를 계산한다.
- “많이 섞을수록 안전”이 아니라 같은 총 particle·mutation 예산에서 final risk가 개선되는지 확인한다.

**D. 비용이 주원인일 때**

- simulator/payoff/density의 중복 계산을 제거하고 factorization을 재사용한다.
- 정확한 동일 결과를 내는 vectorization과 batch 설계를 우선한다.
- training-only 통계를 final 경로에서 제거하고 불필요한 성분을 줄인다.
- 정확성 tolerances를 완화하거나 density를 근사해 성능을 만든 것은 별도 방법으로 취급한다.

처음에는 주원인 하나를 고친다. 두 원인의 결합이 필요하면 `기존 / A만 / B만 / A+B` 대조를 유지한다.

### R2.3 회귀 검증과 진입 기준

- sampling covariance와 dense Gaussian oracle 일치.
- mixture log density, p/q normalization, 상수 payoff, log-tail, full-rank 경계 테스트.
- CPU float64를 기준 구현으로 유지. 낮은 precision은 별도 accuracy 검증 후 도입.
- finite grid마다 frozen proposal의 독립 최종 평균이 reference와 일관되는지 확인.
- weighted conditional CE, 충실한 FIS 구현, V14와 같은 조건에서 비교.
- 학습 seed별 실패·시간 초과·불확실성을 삭제하지 않음.

내부 개발 진입 기준의 기본안은 **전체 비용을 포함한 정확도-비용 곡선에서 약 20% 이상의 개선 신호가 반복적으로 보이고, 사전 지정한 어려운 셀에서 명백한 정확성 실패가 없을 것**이다. 이는 유의성 검정이나 저널 합격선이 아니다. 효과가 작거나 불확실하면 이를 이유로 새 구조를 무한히 추가하지 않는다.

## 8. R3 — 최종 confirmation 설계

### R3.1 결과 보기 전 동결할 사항

1. candidate 알고리즘, 하이퍼파라미터 선택 규칙, fallback 규칙.
2. task generator의 영역·분포·seed, ID/경계/joint-OOD 정의.
3. baseline 구현·튜닝 예산·selection 규칙과 실패 처리.
4. 예산 지점, 표본 수 결정법, reference 방법, primary metric, 통계 단위.
5. 최소 유용 효과, equivalence margin, 다중 비교 방식, retry/timeout 정책.
6. source/config/proposal와 결과 파일을 연결하는 manifest.

과거 실패를 보고 만든 rule을 새로운 OOD 일반화의 근거로 쓰지 않는다. final 결과를 본 뒤 수정하면 그 세트는 development로 이동하고 새로운 confirmation version이 필요하다. 불리한 이전 version도 남긴다.

### R3.2 task와 반복 수: 기본 설계

- 본 confirmation 기본안: **24개 미사용 task = ID 8 + 경계 8 + joint OOD 8**.
- task별 **20개 독립 end-to-end 학습 반복**. pilot 정밀도와 비용으로 변경할 경우 final 결과 열람 전에 변경 사유를 고정.
- 각 fit의 final evaluation은 8개 독립 batch를 출발점으로 하되, batch당 표본 수는 별도 calibration으로 고정.
- QMC의 8개 반복은 독립 scramble이어야 한다. SMC reference는 별도 whole-run 반복을 둔다.
- 최소 비교군: candidate, weighted conditional CE, FIS 계열, V14, conditional RQMC. 기존 V16 V5는 대표 셀에서 역사적 비교를 추가.
- 위 다섯 방법을 전부 adaptive fit으로 보수적으로 세면 최대 `24×20×5=2,400` fit/evaluation 작업이다. RQMC처럼 fit이 없는 방법은 그 사실을 기록한다. 실행시간은 pilot 전에는 알 수 없다.

이 규모가 자원을 넘으면 사전에 12개 task 등으로 축소하고 주장 범위를 줄인다. 예산 때문에 줄인 실험을 동일한 강도의 검증이라고 표현하지 않는다.

task 영역은 R1에서 확인한 정의역·계산 가능성과 최종 주장 범위에 따라 구체적인 수치로 동결한다. 지금 근거 없이 금융 실무 분포를 만들어 넣지 않는다. rarity-matched 군을 만들 경우 별도 독립 pilot으로 strike를 정하고 그 비용을 포함한다. 희귀사건이 관측되지 않았다는 이유로 task를 조용히 제외하지 않는다.

20번 실패가 없었다고 실패율 1% 미만이라 주장할 수 없다. 독립 Bernoulli 가정에서 0/n 실패의 단측 95% 상한은 `1−0.05^(1/n)`이며 n=20이면 약 13.9%다. 낮은 실패율 주장은 추가 반복과 그 가정이 필요하다.

### R3.3 비용·오차 endpoint

주 결과는 **실제 총 wall time 대 relative RMSE/불확실성 곡선**이다. fit, selection, inference를 합산한다. 독립 calibration으로 고정한 표본 수의 실험을 완료한 뒤 시간을 측정한다.

권장 출발점은 3개 budget 지점과 상대 RMSE 5% 목표다. reference 정밀도가 이를 지지하지 못하면 목표 달성 여부는 unresolved다. 사전 등록한 곡선 범위에서만 목표 오차 도달 시간을 비교하며, 관측 밖 extrapolation을 실측 speedup이라 부르지 않는다.

보조 결과:

- 같은 M에서 단일 표본 second moment, estimator variance, fit 간 분포.
- fixed-grid accuracy 차이와 reference uncertainty.
- peak memory, tail contribution concentration, 실패율, 10% 하위 성능 quantile의 불확실성.
- 동일 task 재사용 workload와 서로 다른 task batch를 분리한 비용.
- proxy와 실제 시간의 차이. proxy는 headline speedup이 아님.

`(estimate−reference)²`에는 reference 오차도 포함된다. 독립 reference의 variance 보정이 가능해도 유한 표본 보정값은 음수가 될 수 있으므로 임의로 0에 잘라 정확한 MSE인 것처럼 쓰지 않는다. 원래 차이·reference SE·보정의 불확실성을 함께 제시한다.

### R3.4 통계 분석과 reference

- 학습 반복을 바깥 cluster, 그 안의 evaluation을 안쪽으로 구분한다.
- candidate와 baseline의 의도된 pairing을 유지한다. 하나의 reference를 공유하면 그 공통 오차도 공동 resampling/공분산에 반영한다.
- IID estimator에서 valid SE인지, QMC/SMC에서 올바른 replicate-level SE인지 검증한다.
- bootstrap/정규근사 CI는 방법과 한계를 표시한다. heavy tail에서 coverage를 보장하지 않는다.
- rigorous bounded concentration과 경험적 CI를 같은 종류의 보장으로 섞지 않는다.
- 여러 task/방법의 개별 우위를 주장하면 family-wise 또는 사전 지정한 다중 비교 절차를 사용한다.
- reference는 독립 SMC 반복, 작은 문제의 analytic/고정밀 oracle, 가능한 독립 알고리즘의 교차 확인으로 구성한다. 동일한 shared-code 오류 가능성도 기록한다.
- 기본 reference 목표는 비교 estimator SE의 1/5 이하로 두되, 이를 달성할 비용이 지나치면 결론을 보류한다. reference를 정확한 진리로 취급하지 않는다.

### R3.5 사전 의사결정 기준

| 판정 | 기준과 다음 행동 |
|---|---|
| 강한 성공 | primary workload의 목표 오차 도달 시간비에서 95% 하한>1, 점추정≥1.5를 내부 유용성 기준으로 만족; 정확성·실패 guardrail 통과 → R4/R5 |
| 제한적 성공 | 특정 사전 subgroup에서만 개선 → 그 범위의 알고리즘·정리로 논문 축소 |
| 불확실 | reference/CI/비용 측정이 결정 불가 → 미리 정한 추가 예산만 사용, 이후 unresolved |
| 실패 | 강한 조건부 baseline을 비용까지 포함해 이기지 못함 → 성능 우위 주장 폐기, 이론 또는 부정적 메커니즘 결과 검토 |

1.5배는 프로젝트의 투자 판단 기준이지 저널의 요구조건이 아니다. 평균 speedup이 좋더라도 실패 task와 tail regression을 별도 공개한다. comparator 중 누구를 기준으로 할지도 final 전에 정한다. validation으로 선택한 deployable baseline portfolio와 baseline별 결과를 함께 보고, test 결과를 보고 고른 oracle-best는 진단용이라고 표시한다.

## 9. R4 — mesh 및 이론 기여를 필요한 범위에서 완성

### R4.1 유한격자 이론의 목표

다음 항목을 구별해 proof ledger를 작성한다.

| 항목 | 상태/목표 | 필요한 근거 |
|---|---|---|
| conditional IS 불편성·defensive moment bound | 기존 원리, 구현과 연결 | target·density·독립성·적분 가능성 |
| restricted-family floor와 KL 잔여 분산 항등식 | 알려진 원리의 본 문제 적용 | family 범위와 0인 집합 처리 |
| finite pilot의 방향/모형 선택 오차 | 연구 목표, 아직 정리 아님 | 후보 복잡도·boundedness·학습/선택 분리 |
| H/η/ρ/rarity/rank와 상대 second moment 관계 | 핵심 독창성 후보 | Volterra 구조, tail 가정, 상수·uniformity |
| 격자 간 방향 안정성과 오차-비용 관계 | 확장 기여 후보 | 좌표 coupling 및 적절한 연산자/확률 norm |

새 정리의 가정을 결과를 본 뒤 맞추는 대신, 가정이 적용되는 task 영역과 반례 후보를 먼저 적는다. 절대 L² 오차가 작다는 결과를 `μ_N²`로 나눠도 유용한 상대 bound가 되는지 확인한다. μ_N이 작은 경우 폭발하는 상수를 숨기지 않는다.

**핵심 정리를 만드는 구체적인 순서**

1. 먼저 reference complement를 유지하는 단일-subspace family로 한정한다. mixture 전체에 적용하는 증명은 따로 둔다.
2. `s_B²(U)=Var(g_N|U)`, `D_B=E[s_B²/m₁]`를 정의하고 `m₁=0`인 집합의 기여는 0으로 정의한다. `0≤g_N≤1`이므로 그 집합에서는 g_N도 조건부로 거의 확실하게 0이다.
3. 이미 알려진 항등식 `Var_KL(X)/μ_N²=D_B/μ_N`를 구현 진단과 연결한다. 논문의 새 내용은 이 항등식 자체가 아니라 **Volterra 구조로 D_B를 제어하거나, 비용 내에서 낮추는 방법**이어야 한다.
4. Gaussian conditional variance에 대한 gradient 기반 상계를 출발 후보로 삼되, 필요한 Sobolev 적분 가능성과 실제 conditional payoff의 미분을 먼저 검증한다. 단순 sensitivity score와 증명된 상계를 구별한다.
5. 작은 integrated variance, `|ρ|→1`, 큰 volatility, 작은 m₁ 영역을 localization으로 분리한다. 그 영역을 구현에서 잘라내는 것이 아니라, 이론에서 complement의 weighted tail 기여를 따로 제어한다.
6. 목표 형태는 제한된 parameter 영역에서의 `D_B/μ_N ≤ A(H,η,ρ,rarity,N)·ε_B + R_B`다. 여기서 ε_B, A, R_B의 정확한 정의와 유한성·유용성은 **앞으로 증명할 대상**이다. 이 계획은 이 부등식이 이미 성립한다고 주장하지 않는다.
7. 고정 N의 완결된 결과를 먼저 얻고, 그 뒤 N에 대한 상수 안정성 및 방향의 mesh compatibility를 검토한다.
8. 이 경로가 유용한 bound를 주지 못하면 계산 가능한 risk-selection 보장 또는 명확한 family 한계 정리로 방향을 바꾼다. 의미 없는 상수를 숨겨 원래 정리를 유지하지 않는다.

각 proof package는 `정확한 명제 → 가정 → 알려진 정리와의 차이 → 증명 → 경계/반례 점검 → 검증 가능한 예측`의 여섯 부분으로 작성한다. 이 순서가 R1의 원인 진단과 맞지 않으면 이론도 수정한다.

### R4.2 continuous/joint theorem의 보수적 복구

- **T16-5:** ε에 의존하는 conditional payoff와 Itô 항을 포함하는 Laplace 단계에 적절한 joint LDP 또는 extended contraction 논증이 필요하다. Gaussian covariance 계산과 Cameron–Martin 경로의 연속성만으로 대체하지 않는다.
- **T16-9:** kernel Hilbert–Schmidt/L² 오차가 path supremum, stochastic quadratic variation, exponential localization을 자동 제어하지 않는다. 사용하는 norm과 tail estimate를 각각 증명한다.
- **T16-11:** 위 결과에 의존하는 uniform complexity 주장은 선행 증명 완료 전 보류한다.
- small-noise scaling convention, fixed-grid limit, N(ε)의 증가 조건, uniform constant를 별도로 고정한다.
- local Newton 수렴 조건을 실제 L-BFGS의 전역 성공 보장으로 바꾸지 않는다.
- safety가 필요한 정리를 safety=0 구성에 적용하지 않는다.

일반 정리가 막히면 `N 고정`, 제한된 parameter compact set, 특정 payoff/forward variance 등으로 범위를 축소한다. 제한된 정리를 완결하는 것이 증명 공백이 있는 보편적 주장보다 우선이다.

### R4.3 coupled mesh 실험

1. N=16/32에서 Brownian increment 및 BLP local-coordinate covariance의 coupling을 검증한다.
2. N=32/64/128, 필요 시 256으로 확장한다. coarse/fine Gaussian 배열을 단순 절단·재사용하지 않는다.
3. 같은 underlying Brownian 관측에 대응하는 joint covariance와 각 level의 marginal law를 검사한다.
4. `μ_2N−μ_N`의 coupled 추정과 uncertainty를 기록하고 독립 정밀 scheme/reference와 대조한다.
5. density가 다른 level proposal로 importance sampling하면 joint sampling law와 각 보정식을 먼저 증명한다.
6. MLMC/Richardson은 필요한 weak/strong rate와 비용 가정을 확인한 경우에만 도입한다.

\[
E[(\widehat\mu_N-\mu)^2]=\operatorname{Var}(\widehat\mu_N)+(\mu_N-\mu)^2
\]

는 유한격자 estimator가 μ_N에 불편한 경우의 분해다. numerical error는 별도다. 상대 총오차 목표를 세울 때 sampling, mesh, reference, numerical 항목의 budget을 따로 둔다. 예를 들어 실제 bias 상한 2%와 확률적으로 유효한 sampling 오차 상한 3%가 확보되면 삼각부등식으로 5%에 연결할 수 있지만, 인접 grid 차이가 2%라는 관측만으로 bias 상한이 증명되는 것은 아니다.

### R4.4 실무 연결의 최소 범위

- 극단 left-tail은 stress benchmark로 유지한다.
- 별도의 해석 가능한 strike/maturity/forward variance curve를 가진 bounded digital pricing task를 추가한다.
- 시장 calibration이 없으면 “calibrated production model”이라고 쓰지 않는다.
- risk-neutral 확률을 현실의 폭락 발생 확률 또는 physical VaR로 설명하지 않는다.
- unbounded call, Greeks, barrier로 확장할 때는 conditional formula·moment·monitoring bias를 새로 검토한다.

## 10. R5 — 논문으로 묶을지 판단

### R5.1 논문의 중심 문장

결과가 뒷받침하는 경우에만 다음 형태로 정리한다.

> 조건부 적분 이후의 잔여 second-moment 구조를 분석하고, Volterra 경로의 필요한 방향만 적응시키는 추정법을 제시한다. 명시된 모형·희귀도·격자 영역에서 정확성을 유지하며 학습을 포함한 비용을 줄인다.

다음 네 파일 묶음이 필요하다.

1. 기존 CE/FIS/noisy-IS/transport와의 정의·가정·기여 비교표.
2. 완결된 정리 및 적용 범위, 반례/실패 영역.
3. 1페이지 pseudocode와 재현 가능한 구현.
4. conditioning ablation, 강한 baseline, fresh task, 학습 반복, 실제 비용, mesh와 실패 결과를 포함한 핵심 실험.

새 명칭, Gaussian density의 exactness, 여러 방법을 routing한 사실만으로 독창성을 주장하지 않는다. 새 문헌 점검은 원문 수식·가정·알고리즘 단위로 수행하며, 출판 직전 다시 갱신한다.

### R5.2 결과별 투고 판단

- **Volterra 특화 정리 + 중요한 알고리즘 + 강한 비용/정확도 증거:** 상위 금융수학·계산수학 저널 검토 가능. 그래도 게재를 보장하지 않음.
- **강한 수치 개선, 이론은 주로 기존 원리:** 계산·응용 중심의 논문 범위로 정직하게 구성.
- **새로운 오차/불가능성 정리, 성능 우위 없음:** 이론 논문으로 재구성 가능성을 검토.
- **기존 방법 조합과 셀별 개선만 남음:** 최상위 저널의 주 기여가 부족. 추가 포장보다 문제 재선정.

독립 증명 검토와 독립 실행 재현을 권장한다. 이는 내부 품질 기준이지 모든 저널의 공식 제출 요건은 아니다. 검토 요청·데이터 공유·외부 연락은 별도 승인 후 진행한다.

### R5.3 선택적 확장 재개 조건

| 확장 | 재개 조건 | 반드시 포함할 비용/검증 |
|---|---|---|
| neural amortization | 단일 task solver가 정리되고 실제 반복 workload가 있음 | teacher+training+correction+inference, 새 task, break-even |
| nonlinear transport | Gaussian/subspace family의 한계가 진단으로 확인 | normalized density/Jacobian, 같은 budget baseline, support |
| MLMC/continuum complexity | coupling·rate·uniformity 근거가 확보 | level별 비용/분산/bias, 실제 총 RMSE |
| 비금융 일반화 | 같은 수학 구조를 가진 명확한 응용이 있음 | 새 응용의 target·baseline·의미 검증 |

## 11. 구현 파일 및 산출물 설계

아래 **신규 경로는 제안**이다. 실제 구현 전 기존 abstraction과 중복 여부를 검토한다. 이 계획 작성 시에는 생성하지 않는다.

| 묶음 | 기존 코드의 출발점 | 제안 산출물 |
|---|---|---|
| R0 증거 계약 | `provenance.py`, `seed_ledger.py`, `v15_result_audit.py` | `src/path_integral/research_result_contract.py`, `research_result_audit.py` |
| R0 공통 판정 | V16 comparator confirmation/development | `src/path_integral/comparator_qualification.py` |
| R0 CE | `baselines/cem.py`, `baselines/conditional_rbergomi.py` | `src/path_integral/baselines/weighted_conditional_ce.py` |
| R0 비용 | `baseline_framework.py`, 기존 runner | `src/path_integral/research_cost_accounting.py` |
| R0 주장 정정 | 기존 V16 theorem 문서/ledger | `docs/theory/POST_AUDIT_CLAIM_LEDGER.md` |
| R1 기하 진단 | `cameron_martin_basis.py`, `conditional_transport_adaptation.py` | `src/path_integral/conditional_risk_geometry.py`, 진단 runner |
| R2 candidate | `finite_rank_gaussian_transport.py`, `tempered_target_transport.py` | 원인에 맞는 작은 모듈과 독립 regression tests |
| R3 confirmation | 기존 confirmation runner의 공통 부분 | `experiments/conditional_risk_confirmation.py`, frozen config/manifest |
| R4 mesh | `rbergomi_cm_mesh.py`, `blp_cameron_martin_embedding.py` | coupling 검증 및 level-difference runner |

실험 config는 새 namespace `configs/post_audit/`, 결과는 기존 저장소 관례를 따르는 별도 post-audit 하위 경로를 사용한다. 파일 이름에 성공 여부를 미리 넣지 않는다.

모든 단계의 종료 보고서는 다음을 포함한다.

- 바뀐 정의·코드·이론 가정.
- 실행한 명령·환경·source/config digest.
- 통과/실패/미실행 테스트.
- 예상과 달랐던 결과 및 미해결 사항.
- 다음 단계 진입 여부와 근거.

## 12. 검증 체계와 실행 운영

### 12.1 테스트 계층

| 계층 | 예시 | 해석 |
|---|---|---|
| deterministic unit | density, Gaussian oracle, bound 상수, schema | 정해진 입력의 논리/산술 확인 |
| property/metamorphic | 직교 회전, 자연 proposal, mixture 재표현 | 불변성 위반 탐지 |
| stochastic diagnostic | normalization, analytic probability coverage | 통계적 증거, 증명 아님 |
| adversarial audit | stale-pass, NaN, seed reuse, task mismatch | 잘못된 결론 차단 |
| end-to-end regression | 작은 task 전체 pipeline | 모듈 결합 확인 |
| scientific confirmation | fresh task·학습 반복·실제 비용 | 제한된 과학 주장 평가 |

확률 테스트의 실패를 seed를 바꾸어 통과시키지 않는다. 원인 분석 후 사전 tolerance 또는 sample size를 정당하게 변경하면 변경 이력을 남긴다. 일반 단위 테스트와 장시간 과학 실험을 CI에서 분리한다.

현재 CI는 `main` push와 `main` 대상 PR에서 실행된다. 연구 branch에 push했다는 사실만으로 CI가 통과했다고 보고하지 않는다. 필요 시 manual trigger 또는 명시된 연구 branch 검증을 구현하되, 로컬 전체 검사와 원격 CI 결과를 별도 기록한다.

### 12.2 재개·중단·실패 보존

- 결과는 임시 파일 작성 → 유효성 확인 → 원자적 확정의 순서로 저장한다.
- 재개는 source/config/seed/proposal digest가 일치하는 완료 작업만 건너뛴다.
- 부분 실행을 완료 결과로 읽지 않고 checkpoint와 final artifact를 구분한다.
- timeout/OOM/numerical failure는 상태와 비용을 남긴다.
- 인프라 오류 retry와 알고리즘 실패 retry를 분리한다. 알고리즘의 자동 retry는 모든 비용을 포함하는 사전 정의된 정책이어야 한다.
- 좋은 seed가 나올 때까지 반복하거나 실패 task만 정의역에서 삭제하지 않는다.

### 12.3 자원 배분

처음은 노트북 CPU/float64로 한다. GPU나 cloud를 먼저 빌리지 않는다.

1. R0 smoke와 R1 대표 셀에서 fit/evaluation의 실제 시간·메모리를 측정.
2. `총 예상시간 = Σ(task, method, train_rep)의 fit+selection+evaluation + reference + 허용 retry`로 산정.
3. 전체 matrix 실행 전 비용표와 주장 범위를 검토.
4. compute 예산을 넘으면 task 수·rank sweep·confirmation 범위를 **결과 보기 전에** 조정.
5. cloud 사용·지출·외부 서비스 연결은 별도 선택과 승인 후 진행.

권장 작업량 추정은 R0 1–2주, R1 1–2주, R2 1–3주, R3 2–4주, R4의 범위 제한 분석 2–6주 이상이다. 일부 병행할 수 있으나 wall time 측정은 직렬로 수행한다. 이는 연구 인력 기준 추정이며, 코드 실행시간이나 정리 증명 완료의 보장이 아니다. pilot 후 다시 계산한다.

## 13. 중단·전환 규칙

| 관측 | 결정 |
|---|---|
| 감사/밀도/target가 틀림 | 성능 실험 중단, R0 복구 우선 |
| 새 conditional CE가 기존 우위를 설명 | 우위 주장을 수정하고 conditioning 이후 기여만 재평가 |
| rank 증가에도 held-out 개선 없음 | 고차원화 가설을 지지하지 않음; objective/mode/비용 진단 |
| pilot floor가 너무 noisy | floor 기반 확정 판단 중단; 직접 M₂ 검증 또는 범위 축소 |
| M₂ 감소보다 fit/density 시간이 더 증가 | 단순화 또는 개선안 폐기 |
| 새 holdout에서 이득 소멸 | 일반화 주장 철회, subgroup 범위 또는 문제 재선정 |
| proof에 희귀도 의존 폭발 상수만 남음 | 유용한 상대 효율 정리로 홍보하지 않음 |
| continuum 증명이 장기 정체 | finite-grid 논문 범위와 mesh의 경험적 한계를 명시 |
| 예산 내 성능·독창성 둘 다 확보 못 함 | route 추가를 멈추고 연구 질문 재선정 |

각 단계의 목적은 무조건 성공 판정을 만드는 것이 아니다. **실패를 통해 불필요한 방향을 제거하는 것도 올바른 완료 결과**다.

## 14. 바로 다음 구현 묶음

구현을 시작한다면 다음 순서를 따른다.

1. **R0.1–R0.2:** result contract, claim ledger 초안, adversarial fixture, 의미 검증 감사기.
2. **R0.3:** weighted conditional CE 및 1D Gaussian oracle. 기존 heuristic은 보존.
3. **R0.4–R0.5:** qualification 통합, timing/원시 통계/seed manifest 공통화.
4. **R0.6:** risk bound의 one-/two-sided 수정과 이론 상태 정정.
5. **R0 gate 보고서:** 전체 회귀 결과, 바뀐 해석, 미해결점. 여기까지는 새 모델 우위 주장 없음.
6. **R1 pilot:** 세 대표 셀, 작은 rank sweep, 독립 학습 반복으로 원인을 구별.
7. **R1 결정 보고서:** A/B/C/D 중 주원인을 선택하거나 unresolved로 중단. 그 뒤에만 R2 구현 범위를 확정.

이 계획의 다음 행동은 새로운 대형 V17을 만드는 것이 아니라, **현재 결과를 신뢰할 수 있게 만들고 새 방법이 필요한 정확한 이유를 찾는 것**이다. 그 이유가 증거로 확인됐을 때 가장 작은 개선을 구현하는 편이, 원래 목표인 의미 있는 경로공간 방법과 실용성, 높은 수준의 논문 기여를 함께 달성할 가능성이 높다.

## 15. 계획 작성 자체의 검토 기록

- 감사의 수치 기록과 새로운 실험 제안을 분리했다.
- 알려진 conditional-risk 항등식을 새 수학적 발견이라고 주장하지 않았다.
- weighted 학습과 최종 self-normalization, 두 종류의 tempering을 구분했다.
- nested plug-in 편향, SMC/QMC 의존성, 반복 학습과 평가 반복의 차이를 반영했다.
- fixed-time stopping과 reference 불확실성이 만드는 잘못된 결론을 방지하도록 설계했다.
- 유한격자·연속시간·실무 확장과 새 정리의 미증명 상태를 분리했다.
- 성공·실패·불확실 결과 각각의 다음 행동을 정했다.

이 검토는 실행 전 설계 검토다. 구현 검증·대규모 confirmation·완결된 신규 증명을 대신하지 않는다. 이번 산출물은 계획 문서이며, 모델 코드 수정·실험 실행·커밋·푸시는 포함하지 않는다.
