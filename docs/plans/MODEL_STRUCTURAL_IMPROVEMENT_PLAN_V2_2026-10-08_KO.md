# 모델 구조 개선 통합 실행계획 V2

## 문서 상태와 적용 범위

작성일: 2026-10-08.

**상태: 계획 작성 완료, 아래 신규 구현·실험은 미실행.** V2는 이 계획 문서의 버전이며 새로운 금융모형 또는 성능이 입증된 모델의 버전명이 아니다. 이번 문서 작성은 커밋·푸시·PR 병합·cloud 실행 승인을 의미하지 않는다.

현재 소스 기준: `aa9a21c2d9d5f16f1dc56798d476ee831f8556d0`. 실행 시 실제 HEAD·dirty diff·source snapshot을 다시 기록한다. 이 SHA와 이후 소스를 혼용하지 않는다.

통합한 근거:

1. [기존 구조 검토·개선계획](../reviews/MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md).
2. [위험 geometry 결과와 후속계획](../reviews/STRUCTURAL_IMPROVEMENT_RISK_GEOMETRY_AND_NEXT_PLAN_2026-10-07_KO.md).
3. [보조 위험분포 전체 재학습 안정성](../reviews/AUXILIARY_WHOLE_FIT_STABILITY_2026-10-07_KO.md).
4. [독립 reference·mesh·최소 보정 야간계획](../reviews/OVERNIGHT_GATED_EXECUTION_PLAN_2026-10-07_KO.md).
5. [OpenAI math 전이 가능성 조사](../reviews/OPENAI_MATH_TRANSFER_REVIEW_2026-10-08_KO.md).

이 문서는 위 계획들의 **앞으로의 실행 순서·신규 후보·판정 규칙을 갱신**한다. 과거 실패·결과·이론 상태를 덮어쓰지 않는다. 기존 실행 여부와 새 계획을 구분하며, 더 엄격한 정확성·표본 분리 계약은 유지한다.

아래 V2-P0–P9는 이 문서만의 단계 ID다. 과거 R/P/S/O 단계의 완료 선언과 혼동하지 않는다.

## 1. 연구 목표와 전략

핵심 질문:

> Gaussian Volterra 경로의 기억 구조와 희귀 기여 방향을 이용해, 정확한 finite-grid 평균을 유지하면서 반복 가능한 정확도를 더 적은 총 계산비용으로 얻을 수 있는가? 이 차이를 설명하는 유용한 수학적 관계를 증명할 수 있는가?

목표는 다음 세 가지를 동시에 충족하는 것이다.

- **정확성:** 선언한 target, density, 표본 역할, 불확실성이 일치한다.
- **효율:** 기존 최소 정적 방법과 강한 baseline보다 fixed-precision total work가 개선된다.
- **학술 기여:** 알려진 기법의 이름 변경이 아니라 Volterra 구조·희귀도·격자·weighted residual·비용 사이의 새 연결을 입증한다.

추천 전략:

**독립 기준값 → 정확한 격자 진단 → 작은 구조 기반 가이드 → bounded Stein 잔차 보정 → 필요한 경우에만 bank/신경망 확장 → 전체 학습·미사용 confirmation → 논문 판단.**

이 순서는 성능 성공을 보장하지 않는다. 실패 시 모델 크기 증가 대신 참값·격자·coverage·학습 비용 원인을 분리한다. 같은 개발 셀에서 성공 수치가 나올 때까지 무한 튜닝하지 않는다.

## 2. 현재 출발점: 완료와 미완료

| 항목 | 현재 근거 | 이 계획의 조치 |
|---|---|---|
| 조건부 finite-grid payoff·exact Gaussian mixture | 기존 구현·oracle·회귀 근거 있음 | 재사용, 새로운 API 경계만 추가 검증 |
| 부모/동일 family/단일 shift 원인 진단 | 기존 개발 실험 있음 | 처음부터 같은 실험을 불필요하게 반복하지 않음 |
| auxiliary 위험 평가 | 10개 고정 q의 50개 fresh auxiliary fit에서 v7 개발 안정성 통과 | 고정-q 경험적 결과로 유지 |
| static 대비 learned 추가 가치 | 관측 N·RSE²는 static-only가 유리 | static-only를 필수 baseline으로 승격 |
| 독립 μ/M₂ reference | 공통 guide에 따른 blind spot 가능, 인증 미완료 | 최우선 잔여 관문 |
| mesh 민감도 | 약 99%가 한 구간에 집중하는 기여 사례 | 정확한 adjacent coupling 후 별도 판정 |
| qα 원래 proposal 보정 | 기존 후속 문서에서 미실행 | reference/mesh 뒤 조건부 실행 |
| Stein 보정 | 원리·설계 조사만 완료 | 작은 analytic oracle부터 새 구현 |
| 전체 original-model 반복·confirmation | 미완료 | 새 parent 학습부터 검증 |
| 연속시간 정리·상위 저널 readiness | 미입증 | ledger 유지, 최종 근거로 판단 |

v7의 M₂ estimator RSE 중앙값 2.57%/2.53%는 원래 사건확률 μ의 RSE가 아니다. static-only 대비 learned blend의 관측 N·RSE² 비율 약 1.47/1.50은 점추정 진단이며 모집단 우위 정리가 아니다.

과거 문서의 “다음 단계” 문구를 현재 완료 상태보다 우선하지 않는다. 실행 직전 artifact inventory로 단계별 존재 여부·binding·감사 여부를 확인한다.

## 3. 추가·수정·보류할 내용

| 기존 계획에서 유지 | V2에서 추가·강화 | 지금 보류 |
|---|---|---|
| exact density·ordinary IS | bounded low-rank Gaussian Stein field | 대형 neural operator·차원 확대 |
| 부모 정보 보존·whole-training 반복 | raw M₂와 corrected M₂의 별도 계약 | 검증되지 않은 nonlinear flow |
| 독립 위험 평가·geometry 밖 기여 | signed estimator 전용 수치·통계 경로 | 로그오목 샘플링 정리 직접 적용 |
| μ/M₂ reference 분리 | point/window/distributed 가이드의 제한 대조 | W₂/KL를 분산 인증으로 대체 |
| adjacent Gaussian mesh 검증 | 작은 oracle의 finite region/tail 분리 | 양자/PEPS/renormalization 주장의 도입 |
| 같은 accuracy에서 실제 총시간 | CV 미분·feature 구축·selection 비용 포함 | 성능 증거 없는 neural CV |
| 실패·원자료 보존 | 선언한 theorem/assumption/implementation 계약 | 전체 외부 Lean library 이식 |

OpenAI math는 방법론적 참고자료다. 조사한 버전은 `adc7f1241b42e322a6451854ab7e4b4c146bf78a`이며, 원고의 정리를 우리 분포에 적용했다고 주장하지 않는다.

#139의 global strong-log-concavity 조건은 우리 구현에서 관측된 음의 Hessian 곡률과 맞지 않는다. #374는 Gaussian 전체 공간에 대한 M₂ 안정성 정리가 아니다. 자세한 가정·수치·source는 전이 조사 보고서에 보존한다.

## 4. 전체 단계와 의존관계

| 단계 | 목적 | 주요 산출물 | 다음 단계 조건 |
|---|---|---|---|
| V2-P0 | 상태·수학·표본 계약 봉인 | inventory, claim ledger, role schema | 정확성 오류 없음, historical/new 구분 |
| V2-P1 | 독립 raw μ/M₂ reference | 고정-q production corroboration | relevant q별 정밀도·상호 일치 |
| V2-P2 | exact coupling·mesh | 16/32/64 signed difference | marginal 검증 및 주장 범위 결정 |
| V2-P3 | 최소 physical guide·q 보정 | bounded candidate set·정확한 density | 같은 accuracy에서 개발 가치 |
| V2-P4 | analytic Stein core·toy | bounded field·zero-mean·signed numerics | 독립 oracle 및 bound 검사 |
| V2-P5 | fixed-q 실제 CV microstudy | risk-only와 event-CV 분리 결과 | 새로운 second moment·총비용 이득 |
| V2-P6 | 잔여 원인별 선택 확장 | bank augmentation 또는 neural CV 한 가지 | 최소 구조보다 독립 추가 이득 |
| V2-P7 | 전체 학습·fixed precision | fresh parent→final 반복·cost ledger | 안정성·동일 accuracy·실무 task |
| V2-P8 | 봉인 confirmation·외부 재현 | 미사용 task·강한 비교기·재현 패키지 | 사전 primary endpoint 판정 |
| V2-P9 | 수학·논문 기여 확정 | 증명·한계·novelty·원고 구조 | 결과에 맞는 투고 범위 |

P0 및 P4의 작은 oracle, P9의 문헌·증명 검토는 병행 가능한 저비용 작업이다. **P5의 실제 target 성능 실험과 P3의 원래 q 성능 비교는 P1/P2의 해당 관문 뒤에만 열린다.** 작은 analytic identity 검증을 reference 실패 때문에 금지할 필요는 없지만, 이를 실제 rare-event 성능 입증으로 쓰지 않는다.

P6는 필수 기능 목록이 아니라 조건부 분기다. P3 또는 P5에서 최소 구조가 충분하면 P6를 건너뛴다. P8 전에 알고리즘과 claims를 동결한다. P9의 novelty 검토가 불충분하면 비싼 confirmation 규모를 늘리지 않는다.

## 5. 공통 수학적 계약

### 5.1 추정 대상과 두 proposal

고정 N, d=2N에서

\[
p=\mathcal N(0,I_d),\quad 0\le g=g_N\le1,\quad
\mu=E_pg>0,\quad q\ge\delta_qp,\quad r\ge\delta_rp.
\]

q는 사건확률 final proposal, r는 second-moment 평가를 위한 auxiliary proposal이다. 두 분포를 이름만 바꾸어 같은 효율 주장으로 쓰지 않는다. 초기 δ_q=δ_r=.1을 유지하고, 변경은 별도 preregistration으로만 허용한다.

\[
W=g\,p/q,\quad E_qW=\mu,
\quad M_{2,raw}(q)=E_qW^2=\int g^2p^2/q.
\]

\[
Y=g^2p^2/(qr),\quad E_rY=M_{2,raw}(q),\quad 0\le Y\le1/(\delta_q\delta_r).
\]

defensive floor는 정확한 zero mean·identity covariance의 natural 성분으로 계산한다. near-zero mean이나 density diagnostic tolerance로 floor를 선언하지 않는다.

### 5.2 Gaussian Stein 보정

적절한 Gaussian integration-by-parts 조건 아래

\[
C_\theta=\nabla\cdot v_\theta-z^Tv_\theta,\quad E_pC_\theta=0.
\]

training/selection에 조건부로 q와 θ를 동결한 뒤 새 IID X~q에서

\[
W_\theta=(g-C_\theta)p/q,\quad E_qW_\theta=\mu.
\]

여기서 θ가 scalar coefficient를 포함하므로 별도 β와 θ를 중복 parameterize하지 않는다. θ=0이 raw baseline이다. 평균 0은 field 조건과 Gaussian p에서 얻는 것이며 사건/위험 target의 로그오목성을 요구하지 않는다.

### 5.3 반드시 구분할 세 가지 second moment

| 이름 | 적분 | 의미 |
|---|---|---|
| raw 사건 second moment | M₂,raw=∫g²p²/q | 기존 사건 estimator 위험 |
| corrected 사건 second moment | M₂,CV=∫(g−Cθ)²p²/q | 새 사건 CV estimator 위험 |
| auxiliary estimator second moment | E_r[Y²] 또는 E_r[Y_auxCV²] | M₂ 평가 자체의 정밀도 |

\[
\operatorname{Var}(\widehat\mu_\theta\mid q,\theta)
=\{M_{2,CV}(q,\theta)-\mu^2\}/n.
\]

**보조 M₂ evaluator를 개선했다고 M₂,raw가 작아지는 것은 아니다. 사건 CV를 적용한 후 M₂,raw만 측정해서 새 estimator 분산을 보고해도 틀리다.**

signed residual (g−Cθ)p/μ는 일반적으로 probability density가 아니다. 기존 비음수 g의 q*·Doob target을 이 식으로 단순 교체하지 않는다. P5는 normalized frozen q에서의 signed estimator를 사용한다. square residual을 새로운 SMC target으로 쓰려면 zeros와 MH convention까지 별도 계약이 필요하다.

P5에서 corrected second moment는 먼저 새 IID r 표본으로

\[
T_\theta=(g-C_\theta)^2p^2/(qr),\quad E_rT_\theta=M_{2,CV}
\]

를 평가한다. signed residual을 기존 `log_risk_potential(log_g, ...)`에 넣지 않는다. 그 함수는 0<g≤1 계약을 갖는다. residual=0은 정상일 수 있으며 기존 strictly-positive potential 계약과 다르다. corrected-target SMC 확장은 첫 버전에 포함하지 않는다.

### 5.4 raw M₂ 평가만 보정하는 경우

\[
h_q=g^2p/q,\quad E_ph_q=M_{2,raw},
\quad Y_{auxCV}=(h_q-C_\theta^{risk})p/r.
\]

이 estimator의 mean은 원래 M₂,raw이며 signed sample을 허용한다. 사건 CV field와 risk field를 별도 fit·digest로 관리한다. 동일 field를 두 목적에 자동 재사용하지 않는다.

### 5.5 bound와 signed inference

\|Cθ\|∞≤Bθ를 인증 가능한 analytic bound로 확보하면

\[
|W_\theta|\le(1+B_\theta)/\delta_q,
\quad |Y_{auxCV}|\le(1/\delta_q+B_\theta^{risk})/\delta_r,
\]

\[
0\le T_\theta\le(1+B_\theta)^2/(\delta_q\delta_r).
\]

이 bound는 finite variance 또는 유효한 bounded absolute CI의 출발점이다. 작은 상대오차를 자동 보장하지 않는다. 기존 Y≤100 규칙을 signed 보정에 재사용하지 않는다.

최종 estimate·sample·CI를 0으로 clipping하거나 음수 contribution을 삭제하지 않는다. signed SE는 signed mean과 sample second moment에서 올바르게 계산한다. 안정성을 위해 `SE/|sample mean|`와 별도로 `SE/reference mean` 및 평균의 부호를 보고하며, reference denominator 불확실성도 공개한다. 분산 공식의 음수를 무조건 0으로 처리하지 않고 수치 cancellation·입력 오류 여부를 감사한다.

### 5.6 finite-grid와 continuum

unbiasedness는 μ_N에 대한 것이다. coupled difference와 mesh 안정성은 continuous-time bias 상한이 아니다. 기존 T16-5/9/11 등 미완료 이론을 이 계획의 Stein 항등식으로 해결됐다고 표시하지 않는다.

## 6. V2-P0 — 상태·증거·실행 계약

### 구현 및 기록

1. 기존 stage별 완료 artifact, source ZIP, q/r digest, 감사 결과를 inventory로 작성한다. result 부재를 성공으로 처리하지 않는다.
2. 기존 완료된 parent/family/auxiliary 진단은 history로 재사용한다. source mismatch를 과거 hash 덮어쓰기로 해소하지 않는다.
3. 결과 schema에 `estimand_kind`, `estimator_kind`, `se_unit`, `parent_training_rep`, `auxiliary_fit_rep`, `cv_fit_rep`, `selection_rep`, `final_rep`, `reference_rep`, `mesh_pair`를 분리한다.
4. θ/U/center/lengthscale/bound 및 q/r/source/config digest를 저장한다. raw, risk-only-CV, event-CV를 다른 ID로 구분한다.
5. 각 단계에 `pass / limited_pass / fail / unresolved / not_run`을 사용한다. 실행 상한 초과와 정확성 실패를 구분한다.
6. 실행 시작 전에 candidate 수·pilot/final count·thread·메모리·wall/potential/derivative 상한·통계 endpoint를 봉인한다.

### 표본 역할

`design-pilot`, `parent-training`, `bank-fitting`, `cv-training`, `allocation-pilot`, `selection`, `geometry-calibration`, `reference-iid`, `reference-whole-smc`, `final-probability`, `final-raw-risk`, `final-cv-risk`, `mesh-augmentation`, `audit`를 분리한다.

paired 실험은 공유한 역할을 명시하고 joint covariance를 사용한다. historical final은 development 설계 근거일 뿐 새로운 selection/confirmation 표본이 아니다. frozen-q auxiliary 반복과 parent부터의 whole training 반복을 분리한다.

### 통과 기준

- input/source/role/digest 및 completion grid 감사가 유효하다.
- constant·density·raw risk oracle가 기존 동작과 일치한다.
- 추정 대상과 SE 단위를 잘못 연결한 runner를 거부한다.

오류 발견 시 성능 실험 중단 → 수정 → 작은 oracle → 관련 회귀 순서다. 이 단계 자체에 신규 대규모 학습은 필요 없다.

## 7. V2-P1 — 독립 raw 기준값

### 7.1 M₂,raw corroboration

고정된 두 개발 셀 × 5개 q를 각각 평가한다. 서로 다른 q의 M₂를 한 참값으로 합치지 않는다.

1. frozen static guide의 새 ordinary IID IS.
2. natural 시작·pCN-only risk-SMC.
3. pCN + exact static-guide independence MH risk-SMC 대조.

1과 3은 guide를 공유하므로 독립 blind spot 검증으로 둘만 세지 않는다. 2를 남긴다. shared payoff 구현에 대한 별도 low-dimensional quadrature oracle도 유지한다.

기존 야간계획의 local/global schedule 후보·실행 계수·pilot 분리 구조를 출발점으로 사용한다. 새 환경에서 실제 호출 수를 확인한 뒤 config를 봉인한다. 학습용 SMC 변경을 reference용 frozen schedule에 조용히 반영하지 않는다.

production 기본 출발점은 q별 local/global 각각 whole SMC 32회, static IID 1,048,576개다. **이 숫자는 예산 검토 전 확정 실행량이나 정밀도 보장이 아니다.** 독립 pilot에서 throughput·나쁜 쪽 whole-run 변동을 보고 필요한 count와 상한을 먼저 고정한다.

개발 기준:

- 경로별 RSE 2.5%는 기존 개발 진단 상한으로 유지하되, 실제 agreement를 위한 production allocation은 기본 목표 1.5%를 출발점으로 재설계한다. 달성할 budget이 없으면 unresolved로 남긴다.
- 10개 q의 pairwise 상대 agreement margin 10%; 30개 주 비교의 family-wise α=.05를 사전 선언.
- CI 상한까지 margin 안에 들어오는 equivalence 판정을 사용한다. 차이 CI가 0을 포함하는 것만으로 동등성을 선언하지 않는다.
- whole-run outlier, leave-one-whole-run-out, block contribution 민감도 기준을 production 전에 고정한다.
- 정규근사/whole-run bootstrap의 유한표본 한계를 공개한다. 작은 sample RSE를 oracle 인증으로 부르지 않는다.

allocation의 내부 정합성도 확인한다. 30개 비교의 Bonferroni z값은 약 3.144다. 두 독립 reference의 상대 SE가 각각 2.5%라면 차이의 CI half-width가 약 11.1%로, 점추정이 같아도 10% margin을 통과할 수 없다. 각각 1.5%일 때는 약 6.67%다. 이 계산은 같은 true mean·정규근사 아래의 설계 점검이다. 실제 차이, denominator 불확실성, desired power와 pilot 변동을 포함해 표본 수를 봉인하며, 1.5%만 만족하면 자동 agreement라고 하지 않는다.

production agreement의 개발용 기본 통계량은 양수 추정값 a,b의 symmetric relative difference d=2(a−b)/(a+b)다. delta-method SE²는

\[
16\{b^2SE_a^2+a^2SE_b^2-2ab\operatorname{Cov}(a,b)\}/(a+b)^4.
\]

독립 reference streams이면 covariance=0이다. shared random draws를 쓰면 covariance를 기록·추정한다. 기준은 |d|+z·SE_d≤.10이며, nonpositive/nonfinite mean 또는 SE 불안정은 unresolved다. 이 delta-method 판정은 엄밀한 distribution-free coverage가 아니며, whole-run/block bootstrap sensitivity와 함께 보고한다. 다른 ratio 정의를 쓰려면 production 전에 변경 사유·endpoint를 고정한다.

local만 불안정하고 shared-guide 경로만 일치하면 corroboration은 unresolved다. repair는 새 config의 제한된 한 묶음까지만 허용한다. 이후 동일 셀 무한 튜닝 대신 원인 보고 및 scope 재검토로 전환한다.

### 7.2 사건확률 μ reference

g-target whole SMC와 새 ordinary defensive/static IS를 별도로 생성한다. M₂ 기준값을 μ 기준값으로 대체하지 않는다. 공유 guide와 구현 의존성은 명시하고 independent low-dimensional oracle를 추가한다.

성능 비교에서는 reference SE ≤ 비교 method SE의 1/5를 목표로 한다. 이 조건이 불충분하면 더 강한 reference를 사전 새 설계로 생성하거나 comparison을 unresolved로 둔다. 좋은 method에 noise를 추가하거나 표본을 버려 조건을 맞추지 않는다.

reference 비용은 공개한다. 배포 algorithm이 reference를 selection에 필요로 하면 그 의존 비용을 배포 비용에 포함한다. common scientific benchmark 비용과 deployment 비용은 구분한다.

### 산출물과 출구

`raw_reference` 결과에는 raw normalizer moments, whole-run SE, IID block moments, seed/source/q/guide binding, 비용·실패를 저장한다. G-REF의 통과 범위는 q별 finite-grid 개발 corroboration이다. 한 q 실패를 다른 q의 pass로 가리지 않는다.

## 8. V2-P2 — 정확한 adjacent coupling과 mesh

### 구현 원칙

기존 `rbergomi_coupling.py`의 `adjacent_local_gaussian_coefficients`를 재사용한다. arbitrary local Gaussian mixture에서 fine 좌표와 augmentation을 다루는 adapter가 없으면 작은 별도 adapter만 만든다.

fine 두 구간의 Gaussian 관측을 (ΔW₁,L₁,C₁), (ΔW₂,L₂)로 구성한다. C₁은 coarse endpoint kernel의 첫 구간 적분이다.

\[
\Delta W_c=\Delta W_1+\Delta W_2,\quad L_c=C_1+L_2.
\]

**L_c=L₁+L₂로 대체하지 않는다.** conditional Gaussian augmentation covariance와 coarse whitening을 검증한다. fine proposal 아래 augmentation의 reference conditional law를 유지하면 joint likelihood는 p_f/q_f 하나다.

\[
D_{f,c}=(g_f-g_c)p_f/q_f.
\]

fine/coarse payoff에 서로 다른 likelihood를 붙여 paired correction을 만들지 않는다. coarse q를 별도로 쓴 estimator는 다른 실험이다.

### oracle 및 실제 진단

- marginal/cross covariance, ΔW aggregation, positive definiteness, quadrature tolerance.
- natural·작은 deterministic shift에서 dense Gaussian 식과 비교.
- constant·analytic Gaussian payoff의 signed difference.
- 독립 per-grid mean과 coupled marginal의 일치.
- augmentation seed와 fine path/label stream 분리.

개발 격자 N=16/32/64, adjacent pairs 16–32/32–64. 초기 pilot count 65,536/pair/cell, 이후 본 count는 기존 `{262,144,524,288,1,048,576}` 집합을 출발점으로 비용·정밀도를 보고 사전 봉인한다. pilot을 본 평가에 합치지 않는다.

기록: μ_N, signed difference/SE, reference 불확실성, relative difference, I/J/peak time/largest-cell share, μ/M₂ 기여별 집중, path×step 및 density cost.

### 분기

- coupling oracle 실패: paired 생산 결과 금지. 독립 grid 탐색만 별도 표시.
- adjacent relative 차이 약 10% 이상 또는 그 판단이 불확실: N=32 continuum/practical 해석 보류. fine-grid reference를 우선한다.
- 차이가 작음: finite-grid numerical stability 근거로만 기록한다. convergence rate·bias bound로 승격하지 않는다.
- 큰 민감도에서도 작은 toy/finite-grid Stein identity 연구는 가능하다. 다만 원래 계획의 실제-target 모델 확대·실무 성능 주장은 잠근다.

mesh 관문은 희귀 spike를 더 잘 샘플링하는 것이 물리적 모델 개선인지 coarse-grid 현상 학습인지 구분하기 위한 것이다.

## 9. V2-P3 — 작은 physical guide와 원래 q 보정

### 9.1 최소 후보 설계

먼저 fixed-q M₂ evaluator r에서 작은 가이드 대조를 실시한다. 효과가 확인되고 μ reference가 준비됐을 때만 원래 q proposal로의 전이를 별도 실험으로 평가한다.

초기 후보는 세 가지로 제한한다.

- G0: 기존 point/price uniform static guide.
- G1: component 상한을 맞춘 point/window/distributed uniform guide.
- G2: G1과 **동일한 Gaussian 성분**, 별도 training/selection으로 학습한 weights.

identity covariance, natural mass .1, 초기 전체 component 상한 373을 기존 guide와 맞춘다. G1은 무제한 성분 추가가 아니라 template 일부를 교체한다. 더 적은 성분을 쓸 수 있으며 density 비용을 함께 보고한다.

window는 물리적 시간에서 정의하고 refinement에도 같은 시간척도를 유지한다. simulator의 실제 B에 적용한 weighted rows로 방향을 구성한다. point peak 시점·window를 final 기여점에서 선택하지 않는다. training-only 설계라면 그 비용·digest·재학습 변동을 포함한다.

window/distributed 방향과 가격 방향의 직교성을 자동 가정하지 않는다. 선형 제약 A m=b를 정의하고, 일관성 b∈range(A), rank 및 Gram matrix를 검사한다.

\[
m_*=A^T(AA^T)^\dagger b.
\]

동일 κ라는 숫자만으로 feature 간 에너지·rarity가 같다고 부르지 않는다. ||m||²/2와 실제 field shift를 보고한다. 이 mean은 linear constraint의 최소 energy이며 nonlinear rare-event optimum이 아니다.

### 9.2 weights와 qα

fixed component density φ_j 아래

\[
s_w=\delta p+(1-\delta)\sum_jw_j\phi_j,\quad
w\in\Delta,\quad w_j\ge .01/J
\]

를 첫 weight-fit family로 사용한다. J=learned component 수다. natural component는 별도로 고정한다. 아래 lower-weight rule의 변경은 새 개발 config에서만 한다.

목표 integrand F를 고정하면 J_F(s_w)=∫F²p²/s_w는 affine positive density의 mixture weights에 대해 convex다. 사건 q_w 개선에서는 F=g이므로 J_F=M₂,raw(q_w)다. 고정 q₀의 auxiliary r_w 개선에서는 F=h_q₀=g²p/q₀이고 J_F는 **M₂ 평가 estimator의 second moment**다. 두 목적을 구분한다. derivative/Hessian 사용은 integrability를 확인한 뒤 정당화한다. known convex optimization을 신규성으로 주장하지 않는다. [He–Owen](https://arxiv.org/abs/1411.3954).

별도 frozen t에서의 training objective는

\[
\widehat J_F(w)=\frac1m\sum_i\frac{F(X_i)^2p(X_i)^2}{t(X_i)s_w(X_i)},\quad X_i\sim t.
\]

t의 full normalized density를 사용한다. auxiliary weight fit을 사건용 F=g objective로 실행하여 risk evaluator 개선이라고 해석하는 것은 금지한다. 동일 sample에서 empirical minimum이 생겨도 true risk 개선이 보장되지는 않는다. independent selection/final이 필요하다.

원래 q 개선에서는 기존 qα=(1−α)q₀+αr, α∈{0,.25,.5} 계획을 유지한다. matched-work event/risk correction을 비교하고 static-only q_V를 필수 baseline으로 둔다. α<1의 M₂(qα)≤M₂(q₀)/(1−α)는 개선 보장이 아니다. static-only를 α=1 공식에 넣지 않는다.

### 채택·중단

q/r 변경을 별도 결과로 저장한다. reference qualification·같은 component capacity·density 비용·전체 학습 실패율을 포함한다. G2가 G1보다 총비용에 유리하지 않으면 uniform weights를 유지한다. 기존 q가 부적격이면 q 대비 speedup을 승리로 집계하지 않는다.

P3에 성공한 최소 q/r를 동결하고 P5의 주 baseline으로 사용한다. 실패했다고 반드시 새 operator를 추가하지 않는다.

## 10. V2-P4 — bounded analytic Stein core

### 10.1 구현할 첫 field

U∈R^{d×k}, UᵀU=I, s=Uᵀz, k∈{2,4,8}을 작은 oracle에서 검사한다. 실제 microstudy에서는 selection으로 결정한 한 rank와 matched generic rank를 주 비교로 고정한다. 여러 rank의 final을 본 뒤 winner를 고르지 않는다.

feature dictionary는 physical point/window/price 선형 함수에서 만들고 QR/SVD로 whiten한다. rank tolerance와 순서·부호 convention을 고정한다. geometric feature와 physical driver의 의미를 혼동하지 않는다.

첫 analytic atom은 다음과 같다.

\[
a_j(s)=b_j\exp\{-\|s-c_j\|^2/(2\ell_j^2)\},\quad \ell_j>0,
\quad v_j(z)=Ua_j(U^Tz).
\]

\[
C_j(z)=-\left[b_j^Ts+\frac{b_j^T(s-c_j)}{\ell_j^2}\right]
\exp\{-\|s-c_j\|^2/(2\ell_j^2)\}.
\]

\[
C_\theta=\sum_j\theta_jC_j,
\quad B_j=\|b_j\|\left(\|c_j\|+(1+\ell_j^{-2})\ell_j e^{-1/2}\right),
\quad B_\theta=\sum_j|\theta_j|B_j.
\]

이 B_j는 ||t||exp(−||t||²/(2ℓ²))≤ℓexp(−1/2)에서 얻는 analytic upper bound다. 수치 grid에서 관측한 최대값을 global bound로 쓰지 않는다. Gaussian-window field와 그 derivative의 성장·적분가능성을 확인하고 integration by parts를 직접 증명한다. U의 complement에서 v가 사라지지 않아도 C_j가 s에만 의존하고 Gaussian 적분이 분리됨을 이용한다.

이 field의 zero-mean oracle에는 independent Gaussian moment 식도 사용한다. s~N(0,I_k), φ=exp(−||s−c||²/(2ℓ²))이면

\[
E\phi=(\ell^2/(1+\ell^2))^{k/2}
\exp\{-\|c\|^2/[2(1+\ell^2)]\},\quad
E[s\phi]=cE\phi/(1+\ell^2).
\]

따라서 E[(s−c)φ]=−ℓ²cEφ/(1+ℓ²)이고 E C_j=0이 직접 상쇄된다. 수식·구현의 sign/ℓ² convention을 이 식으로 대조한다. 학습 center·coefficients도 final 전에 frozen이면 conditional identity가 유지된다.

최초 dictionary는 deterministic center 0 및 training-only weighted center 소수로 제한한다. center/lengthscale 후보는 design-pilot에서 동결하고 최종 dictionary atom 수 ≤24를 출발점으로 둔다. mean shift가 큰 center의 B_j 및 수치 범위를 검사한다. center·basis를 학습하면 그 단계도 fit 비용·whole-training 변동에 포함한다.

### 10.2 convex coefficient fitting

frozen sampling density t, 목표 F∈{g,h_q}에 대해

\[
A_i=F(X_i)p(X_i)/t(X_i),\quad
B_{ij}=C_j(X_i)p(X_i)/t(X_i).
\]

\[
\min_\theta\frac1m\sum_i(A_i-B_i\theta)^2
+\lambda\|\theta\|^2,
\quad\sum_j|\theta_j|B_j\le B_{max}.
\]

event-CV는 t=q, auxiliary risk-CV는 t=r가 기본이다. 다른 t를 사용하면 별도의 정확한 reweighting objective를 도출한다. unweighted prior MSE를 q/r estimator variance 목적처럼 쓰지 않는다.

λ, B_max, 수치 scaling, atom 수는 별도 pilot/selection에서 결정한다. θ=0은 항상 feasible baseline이다. coefficient cap·ridge는 finite-sample 안정화 선택이며 true risk 감소 보장이 아니다. fit 수렴·KKT residual·조건수·rank·solver failure를 저장한다. θ 또는 optimizer가 실패하면 사전에 정의된 θ=0 fallback으로 돌아가고 실패·비용을 모두 기록한다.

정확한 population projection의 일반 identity는 알려진 회귀 원리다. unconstrained objective의 Gram G=E_t[BBᵀ], b=E_t[BA]에서 유효한 모멘트·range 조건 아래 θ*=G†b가 가능하지만, 이 식 자체는 Volterra 신규 정리가 아니다. empirical G를 population 값으로 취급하지 않는다.

### 10.3 signed numerical implementation

새 signed estimator는 기존 positive log-contribution summarizer를 그대로 재사용하지 않는다.

- field Cθ, residual g−Cθ, weight p/q 또는 p/r를 역할별로 계산한다.
- log-CDF와 signed residual의 subtraction/cancellation을 stable signed-log 또는 검증된 동등 식으로 처리한다.
- positive/negative sums, signed mean, second moment, unbiased sample variance, cancellation diagnostics를 저장한다.
- reference float64 dense 식 및 작은 high-precision oracle와 대조한다.
- exact zero residual, 매우 작은 positive g, 음수 residual, mixed signs, duplicate atoms, rank deficiency, 극단 center/ℓ를 테스트한다.
- NaN/Inf/허용하지 않은 numerical range는 명시 실패다. 임의 clipping으로 target을 바꾸지 않는다.
- field의 exact divergence와 autograd trace, central FD를 작은 k에서 비교한다. 첫 버전에서 Hutchinson/random trace를 쓰지 않는다.

원래 raw estimator 구현은 regression baseline으로 남긴다. signed 지원 때문에 기존 positive-potential SMC API를 느슨하게 바꾸지 않는다.

### 10.4 oracle와 통과 기준

1. p=N(0,I), g=c constant: raw mean c, Stein mean 0, event-CV mean c.
2. one/two-dimensional Gaussian-CDF payoff: independent quadrature로 μ, M₂,raw, M₂,CV 및 cross moments 확인.
3. identity/shift/mixture q와 r에서 full-density reweighting 항등식.
4. q=r=p이면 각 식이 ordinary Gaussian Monte Carlo/control variate 식으로 환원.
5. nonorthogonal U를 거부하거나 올바른 generalized Jacobian 경로를 별도로 검증.
6. analytic bound, coefficient cap 및 signed CI용 endpoint 검사.
7. frozen θ와 독립 final seed를 감사. final-informed θ를 intentional invalid case로 거부.
8. 작은 high-risk mode를 fitting에서 빼 둔 toy에서 CV가 coverage 인증이 아님을 재현.

구현 tolerance는 oracle 정확도·dtype·수치 범위에서 정하고 test config로 고정한다. stochastic mean 일치 하나만으로 boundary identity를 증명하지 않는다. deterministic 수식·quadrature·통계 검사를 함께 사용한다.

G-STEIN 통과는 field와 estimator의 correctness다. variance 감소·실제 rare-event 효율은 P5에서 별도로 검증한다.

## 11. V2-P5 — fixed-q 실제 CV microstudy

### 11.1 두 목적을 다른 실험으로 실행

**A: auxiliary risk-only CV**

- 원래 q는 변하지 않는다.
- raw M₂ evaluator vs bounded generic CV vs bounded Volterra CV.
- 목표는 M₂ 평가 정확도와 총비용이다.
- 성공해도 사건 estimator q의 성능 개선으로 쓰지 않는다.

**B: event-CV**

- frozen q, 새 event-CV field 및 final X~q를 사용한다.
- raw W vs generic bounded CV vs Volterra bounded CV를 평가한다.
- 별도 IID r에서 Tθ로 M₂,CV를 평가한다.
- M₂,raw도 같이 보존하여 estimator가 바뀌었음을 명확히 한다.

generic baseline은 같은 Gaussian-window atom·rank·atom 수·regularization·fit budget을 갖는 DCT/좌표 feature 등으로 고정한다. physical geometry 효과와 단순 CV 효과를 분리한다. strong standard CV/control-functional comparator의 상세 범위는 P9의 원문 대조 후 confirmation 전에 결정한다.

### 11.2 pilot와 본 평가

처음에는 각 개발 셀 parent replicate 0을 **ID 기준**으로 사용한다. 좋은 q를 성능으로 고르지 않는다. 이 pilot은 historical development이며 confirmation이 아니다.

1. deterministic dictionary 및 fit config를 봉인한다.
2. 새로운 CV training·selection·allocation pilot을 생성한다.
3. 선택된 field와 baseline을 동결한다.
4. 새로운 final-probability, final-raw-risk, final-cv-risk 표본을 생성한다.
5. 성공 신호가 있을 때만 10개 고정 q 모두에서 fresh CV 전체 fit 5회로 확대한다.

pilot count는 현 노트북 throughput에 맞춰 정한다. 기존 계획의 `{65,536,262,144,1,048,576}`은 allocation 시작 후보이며 signed field 비용을 측정한 뒤 최종 상한을 다시 봉인한다. N을 무조건 늘려 미관측 tail 문제를 해결했다고 하지 않는다.

조건부 fit 평가에서는 raw/CV에 공통 final draws를 사용하는 paired contrast가 가능하다. 서로 독립이라고 가정한 SE를 적용하지 않는다. selection과 final은 공유하지 않는다. 추가 independent final blocks와 r-based second-moment audit로 covariance/variance 계산을 교차 확인한다.

### 11.3 통계·비용 endpoint

primary development endpoint: 같은 accuracy qualification에서의 fixed-precision total wall-time ratio. pilot 단계의 variance×per-sample-cost는 allocation proxy로만 보고한다.

CV에는 setup 비용이 있으므로 V×total-time 같은 무차원으로 정리되지 않은 혼합 지표를 최종 efficiency 주장으로 쓰지 않는다. 목표 relative SE ρ에 대한 예측은

\[
n_{forecast}\approx V/(\rho^2\mu^2),\qquad
T_{forecast}=T_{offline}+T_{fit}+T_{selection}+c_{infer}n_{forecast}
\]

로 구분한다. 이는 독립 pilot 기반 forecast이며 실제 qualified run 시간과 다르다. CV를 여러 task에 재사용하면 offline amortization을 workload 크기별로 별도 보고한다.

secondary: raw/corrected M₂, whole-fit mean 변동, empirical SE, independent reference 차이, worst-case fit, field bound·coefficient norm·condition number, μ/M₂ tail contributions, training variance, peak RSS.

### 11.4 채택 기준

- 모든 correctness·reference 계약을 충족한다.
- 개선 방향의 사전 정의된 독립 반복 CI와 실질 효과 크기를 만족한다. 기존 약 20% total-cost 감소는 개발 확대 신호로 유지하되 저널 합격선이나 증명 기준으로 쓰지 않는다.
- 필요한 반복 수와 CI/primary comparisons/multiplicity 처리는 본 평가 전에 power·budget 분석으로 봉인한다.
- 최소 두 method가 같은 accuracy 자격을 얻은 셀에서만 cost ratio를 발표한다.
- sample RSE가 줄어도 independent corrected-risk 평가 또는 whole-fit 결과가 불일치하면 unresolved다.
- Volterra CV가 generic CV보다 추가 이득이 없으면 physical-feature 우위 가설은 미지원이다. raw 대비 개선과 구분한다.
- fitting+derivative 비용이 이득을 상쇄하면 θ=0 또는 작은 analytic method를 유지한다.

signed CV의 음수 sample/estimate를 숨기지 않는다. 값이 음수라는 이유만으로 unbiasedness 위반은 아니지만, rare positive mean에 대한 부족한 precision·수치 문제를 구분하여 판정한다.

## 12. V2-P6 — 필요한 경우에만 확장

### 분기 A: bank coverage가 계속 실패할 때

Gaussian augmentation을 검토한다.

\[
f_\beta(x)=p(x)h(x)^\beta,\quad
\widetilde\pi_\beta(x,y)\propto f_\beta(x)\mathcal N(y;x,\tau I).
\]

y|x exact Gaussian, x|y exact MH ratio를 사용하는 target-invariant transition을 먼저 증명한다. 전체 composed kernel의 detailed balance/invariance 및 SMC normalizer 계약을 검사한다. y를 적분하면 원래 x marginal이므로 payoff smoothing으로 target을 바꾸지 않는다.

global convexity·mixing time·독립성은 보장하지 않는다. 현재 타깃의 Hessian 전역 하한이 없으므로 proximal Gaussian 항만으로 로그오목이라고 선언하지 않는다. MCMC bank는 fitting 전용이고 IID final 대체가 아니다.

pCN / 기존 exact independence MH / augmentation을 동일 potential·밀도 비용·시간 상한으로 비교한다. whole bank 반복, 기여 mode overlap, 이동 거리, genealogy, 최종 q/r 위험을 함께 본다. ancestry 또는 acceptance만 개선되면 채택하지 않는다.

### 분기 B: analytic field 표현력 한계가 증거로 지지될 때

작은 neural field만 검토한다. 단순 training loss 감소가 아니라 analytic CV 대비 잔여 corrected-risk·새 task 재사용 비용으로 필요성을 입증한다.

첫 neural 후보는 bounded smooth core에 Gaussian envelope를 곱하는 등 field/derivative 성장 조건을 명시해야 한다. tanh 등 bounded activation만으로 divergence까지 전역 bound가 자동 확보되는 것은 아니다. network weight norm·Jacobian bound 및 envelope를 함께 검사한다.

first-order finite k exact divergence를 유지한다. high-dimensional random trace, unconstrained ReLU field, arbitrary flow는 별도 검증 없이 넣지 않는다. parameter cap·generic neural CV baseline·training-inclusive 비용을 포함한다.

### 확장 제한

두 분기를 동시에 붙이지 않는다. baseline/A/B/A+B가 필요한 경우 새 bounded development config로 따로 설계한다. scope가 커지거나 별도 설치·외부 자원이 필요하면 추가 승인과 예산 결정이 필요하다. 확장 실패를 기존 성공 record에서 삭제하지 않는다.

## 13. V2-P7 — original-model 전체 학습과 실무성

### 독립 반복 단위

각 반복에서 parent CE/SMC → bank → mixture/guide → CV fit → selection → allocation → final 및 risk audit를 새로 실행한다. 고정 q 아래 CV fit만 새로 한 5회 결과를 이 반복으로 세지 않는다.

개발 whole-training 5회는 구현·실행 관문이다. 이후 20회는 독립 안정성 평가의 출발점이며 모든 task에서 무조건 충분하다는 뜻이 아니다. failure-rate 주장에는 별도 반복 수를 계산한다. 실패 0/20의 단측 95% 상한이 약 13.9%라는 기존 계획의 주의사항을 유지한다.

### accuracy와 총비용

- development 목표 probability RSE 5%, 기존 equivalence margin 25%는 상대 RMSE≤5% 인증과 다르다.
- paper용 target accuracy·equivalence margin은 practical meaning 및 reference budget으로 사전 재설계한다.
- reference variance와 shared-reference covariance를 분석에 반영한다.
- finite-training 실패·timeout·fallback·retry 비용을 전부 남긴다.
- offline, guide probing, bank, fit, CV derivatives, allocation, selection, inference, I/O, warmup 포함 여부를 분리한다.
- 실제 공유 실험비와 standalone deployment 비용을 모두 기록한다.
- reference가 scientific validation에만 쓰이는지 deployment selection에 필요한지 구분한다.
- timing은 같은 환경·thread·전력 조건에서 순차 실행한다. 다른 무거운 작업과 동시에 돌리지 않는다.

새 training에도 static-only보다 learned addition이 불리하면 최소 구조를 논문 main method로 선택하고 negative result를 공개한다. 신경망이 없어도 유효한 연구 기여는 가능하다.

### 실무 workload

S0=100, K=1 stress cell과 분리하여 해석 가능한 strike ratio·maturity·forward variance의 bounded digital pricing workload를 정의한다. μ가 극소인 stress benchmark와 보통 pricing을 같은 성능 집계로 합치지 않는다.

위험중립 확률과 현실 폭락 확률/physical VaR를 혼동하지 않는다. 실제 calibration data가 없으면 production calibration 성과를 주장하지 않는다. unbounded call payoff나 다른 path task로 확장할 때 기존 g≤1 bound·conditional formula를 그대로 재사용하지 않고 별도 계약을 만든다.

## 14. V2-P8 — confirmation과 외부 재현

알고리즘·candidate family·routing·selection·budget·endpoint를 동결한 뒤 실행한다. 기본 출발점은 기존 24개 미사용 task(ID/boundary/joint OOD 각 8개), task별 20개 whole-training 반복이다.

canonical/높은 η 개발 셀은 새 seed라도 미사용 confirmation task가 아니다. 비용 때문에 task/repeat를 축소하면 결과 열람 전에 이유와 범위를 봉인한다. 여러 seed로 실패 셀을 대체하거나 성공 task만 추려 headline을 만들지 않는다.

필수 baseline:

1. 같은 conditional payoff의 강한 CE/SMC mixture.
2. 최소 static physical guide.
3. bounded generic CV와 해당 범위의 표준 Stein/control-functional 방법.
4. Volterra CV, q correction only, CV only, 선택된 combination.
5. paper claim에 필요한 충실한 failure-informed CE/conditional RQMC/V14 등. 일부 모듈만 구현한 것을 원문 전체 알고리즘으로 부르지 않는다.

모든 baseline을 동시에 무한 확장하지 않는다. 문헌·pilot에서 강한 최소 집합을 동결하고 capacity·tuning budget·workload를 공개한다. 금융 전용 주장이면 그 범위의 강한 비교기를 우선한다.

primary task aggregation, paired design, multiple comparisons, failure handling, total-time CIs를 사전 정의한다. 실패 셀을 포함한 qualification fraction과 시간 분포를 함께 보고한다. “성공한 셀에서만 빨랐다”를 전체 superiority로 쓰지 않는다.

외부 재현은 clean environment/source snapshot에서 같은 config 및 source/binding/seed 감사부터 시작한다. 하드웨어가 다르면 절대시간과 speed ratio 조건을 분리한다. 독립 구현 oracle·별도 연구자 재현이 없으면 그 완료를 주장하지 않는다.

## 15. V2-P9 — 이론·독창성·논문 의사결정

이 단계는 마지막에만 시작하지 않는다. P0/P4와 함께 시작하고 P8 전에 최소한의 중심 주장과 선행 경계를 고정한다.

### 15.1 알려진 정확성 명제와 신규성 후보를 구분

| 유형 | 대상 | 연구 기여 해석 |
|---|---|---|
| 기본 계약 | conditional Gaussian law, exact IS, Stein mean 0 | correctness 기반, 새 원리 아님 |
| 기존 방법 | convex mixture/CV fitting, ridge/projection, pCN/MH | 선행연구와 명시 대조 |
| 신규 후보 A | kernel/time scale가 weighted Stein residual을 제어 | 새 가정·유용한 상수·counterexample 필요 |
| 신규 후보 B | 희귀 기여 구조의 mesh 변화와 최소 guide coverage | finite-grid/continuum 분리 필요 |
| 신규 후보 C | 보정 이득과 offline amortized total work 연결 | algorithm·통계·비용 가정 필요 |

Gaussian prior의 Poincaré/LSI를 사건·위험 law의 mixing guarantee로 해석하지 않는다. W₂/KL/overlap는 coverage 진단이며 relative IS variance의 충분한 인증이 아니다. raw IS의 상대 분산에는 χ²(q*||q)가 직접 연결된다.

### 15.2 기존 complement 이론도 유지한다

기존 계획의 q_U(u)p_V(v) family와 conditional projection 분석을 폐기하지 않는다. Gaussian independent U/V 분해, m₁(u)=E[g|U=u], s²(u)=Var(g|U=u) 아래 marginal q_U를 자유롭게 고를 수 있는 이상적 KL projection은

\[
q_{KL}(u,v)=p_U(u)m_1(u)p_V(v)/\mu,
\quad D_U=E_{p_U}[s^2(U)/m_1(U)].
\]

적절한 적분가능성 아래 raw IS의 relative variance는 D_U/μ다. m₁=0인 집합은 conditional g=0이므로 해당 기여를 0으로 정의한다. 이 projection은 일반 variance-optimal proposal, 실제 제한 Gaussian fit 또는 defensive mixture와 같은 대상이 아니다.

Volterra kernel·rank·rarity가 D_U를 제어하는지 분석하는 기존 목표를 P9의 parallel theory track으로 유지한다. Stein field의 weighted residual 이론과 관계를 검토하되 두 항을 근거 없이 더한 새 variance 식을 만들지 않는다. 실제 q에 어떤 family 가정이 성립하는지 먼저 대조한다.

### 15.3 쓸 수 있지만 단독으로 약한 bound

E_pCθ=0, q≥δp이고

\[
\epsilon^2=E_p[(g-\mu-C_\theta)^2]
\]

가 유한하면

\[
M_{2,CV}\le(\mu^2+\epsilon^2)/\delta,
\quad \operatorname{Var}(\widehat\mu_\theta)/\mu^2
\le\{1/\delta-1+\epsilon^2/(\delta\mu^2)\}/n.
\]

이는 직접적인 defensive/projection bound다. ε²/μ²가 희귀도에 따라 폭발하면 유용한 relative-efficiency theorem이 아니다. μ를 정확히 아는 이상적 field를 실제로 계산했다고 가정하지 않는다.

더 중요한 새 목표는 실제 weighted norm

\[
J_{q,N}(\theta)=\int(g_N-C_\theta)^2p_N^2/q_N
\]

를 kernel·feature rank·training error·rarity·grid와 연결하는 것이다. prior-L² bound만으로 모든 q에서 강한 성능을 보장한다고 주장하지 않는다. Gaussian whole-space q/p의 성장과 field envelope 제한도 이론에 포함한다.

### 15.4 증명·반례 체크리스트

- Sobolev regularity와 Gaussian integration by parts 경계 항.
- 작은 integrated variance, 작은 conditional mean, |ρ|→1의 퇴화.
- physical-time feature의 refinement, U_N consistency와 rank 변화.
- 희귀도·H·η·ρ에 따른 상수 및 uniformity의 실제 범위.
- full space와 restricted Gaussian/mixture family의 projection 차이.
- training objective·finite-bank 탐색 실패·generalization error.
- event-CV의 signed residual과 corrected-risk moment.
- cost theorem의 density component 수·derivative·fit·reuse 횟수.
- kernel가 nonlocal이라고 양자 area law나 renormalization theorem을 가정하지 않음.

grid/refinement 정리가 실패하면 고정 N 결과로 좁힌다. uniform task theorem이 너무 강하면 선언한 parameter 범위 정리와 empirical OOD 분석으로 구분한다. 이를 계획 성공을 위해 “증명 완료”로 바꾸지 않는다.

### 15.5 선행연구 대조

최소한 [Oates–Girolami–Chopin](https://arxiv.org/abs/1410.2392), [Wan 등](https://arxiv.org/abs/1806.00159), [Müller 등](https://arxiv.org/abs/2006.01524), [He–Owen](https://arxiv.org/abs/1411.3954), [Rotskoff 등](https://proceedings.mlr.press/v145/rotskoff22a.html), [conditional neural CV](https://proceedings.mlr.press/v337/siahkoohi26a.html)를 theorem/algorithm/assumption/cost별로 대조한다.

rough-volatility conditional pricing·low-rank CV 조합의 가까운 문헌은 전이 조사 보고서의 목록을 이어서 전문 확인한다. 전문을 못 읽은 초록 자료는 novelty 인증 근거로 쓰지 않는다. 원고별 citation 및 실제 복사 대상 license도 확인한다.

### 최종 분기

| 결과 | 논문 방향 |
|---|---|
| Volterra 특화 정리 + 독립 accuracy + 반복 가능한 총비용 개선 | 이론·계산 통합 상위 저널 원고 후보 |
| 강한 수치 개선, 원리는 기존 조합 | 계산·응용 기여와 범위를 명확히 한 원고 |
| physical CV 추가 이득 없음, 유용한 새 한계 정리 있음 | 한계·기하 분석 중심 검토 |
| reference/mesh 미해결 또는 새 task 개선 소멸 | superiority 주장 보류, 질문·scope 재설정 |

논문 제목은 증거 뒤에 결정한다. `Mesh-aware Volterra Stein Residual Correction...`은 작업용 후보이며 mesh 정리·Stein 추가 가치가 없으면 headline으로 사용하지 않는다. 상위 저널 채택 가능성을 확률로 보장하지 않는다.

## 16. 구현 파일 묶음과 검증 순서

아래 신규 경로는 **제안**이다. 실행 전에 existing abstraction을 검색하고 충분하면 재사용한다. 이 계획 작성에서 생성한 코드 파일은 없다.

| 묶음 | 기존 출발점 | 제안 신규 경로 |
|---|---|---|
| 역할·estimand 감사 | `research_result_contract.py`, `seed_ledger.py` | 기존 schema 최소 확장 및 테스트 |
| independent raw reference | `conditional_second_moment.py`, weighted SMC | `experiments/post_audit_v2_independent_reference.py` |
| local adjacent adapter | `rbergomi_coupling.py` | `src/path_integral/volterra_adjacent_local_coupling.py` |
| mesh diagnostics | 기존 payoff·simulator | `experiments/post_audit_v2_mesh_diagnostics.py` |
| 다중척도 가이드 | `volterra_excursion_guide.py` | `src/path_integral/volterra_multiscale_guide.py` |
| 고정 component weights | `weighted_bank_mixture.py` | `src/path_integral/risk_calibrated_mixture_weights.py` |
| bounded field | physical B, Gaussian field 수식 | `src/path_integral/gaussian_stein_fields.py` |
| signed estimator·corrected risk | raw density/SE abstractions | `src/path_integral/signed_stein_estimators.py` |
| CV fit | coefficient regression | `src/path_integral/weighted_stein_fit.py` |
| microstudy | frozen q/r restoration | `experiments/post_audit_v2_stein_microstudy.py` |
| 전체 반복·총비용 | 기존 fixed-precision runner | `experiments/post_audit_v2_whole_training.py` |
| 선택 확장 | 기존 invariant kernels | P6 필요 시에만 새 파일 결정 |

configs는 `configs/post_audit/structural_v2_*_v1.yaml`, results는 `results/post_audit/structural_v2_*_v1.json[.gz]`를 제안한다. 기존 `v1` artifact를 덮어쓰지 않고 새 suffix를 사용한다. stage report·theorem ledger·source ZIP·failure ledger를 함께 저장한다.

검증 순서:

1. 수학 명세·수식·sign·measure·normalizer 검토.
2. deterministic dense/analytic oracle 및 FD/autograd 대조.
3. 작은 stochastic oracle와 finite-region/tail bound 분리.
4. role/seed/digest·invalid-input·failure-cost integration.
5. 제한 pilot 및 production config 봉인.
6. 새 표본 production, independent result audit.
7. 관련 회귀 → 전체 pytest/Ruff/mypy → diff/문서 검사.
8. phase report와 다음 관문 판정. correctness·효율·novelty 완료를 분리.

전체 test 통과는 유한 입력의 구현 근거이며 모든 수학적 오류가 없다는 보장이나 새 정리의 증명이 아니다. Lean은 작은 명세를 별도로 검증할 선택 도구이며 필수 설치·전체 이식은 하지 않는다.

## 17. 노트북 예산·재개·실패 대응

기존 야간 6시간 30분/상한 7시간 30분 설정을 새 CV·확장 단계 전체에 자동으로 적용하지 않는다. 추가 업무가 생겼으므로 단계별 fresh pilot 후 실제 budget을 봉인한다. planning-only 상태에서 긴 실행이나 automation을 시작하지 않는다.

노트북 기본 운영:

- torch threads=1, batch≤8,192를 초기값으로 삼고 메모리 pilot에서 확인.
- RSS 운영 상한 min(4 GiB, 물리 RAM 25%), free disk 10 GiB 미만이면 큰 artifact 전에 보류.
- potential calls 외에 path×step, density evaluations/component count, feature projection, divergence, fit iterations 및 I/O 시간을 계측.
- scientific timing은 순차 실행. CPU/GPU 교체는 동일 비교의 별도 환경으로 구분.
- paid cloud, 패키지/드라이버 설치, OS 설정 변경, git mutation은 별도 요청 없이 하지 않음.
- phase별 `max_wall_seconds`, `max_potential_evaluations`, `max_samples`, `max_whole_replicates`, `max_candidate_count` 설정.
- pilot·실패·warmup·retry도 실제 소비량으로 기록. test suite의 미계측 호출은 0으로 쓰지 않음.

재개는 완료 whole job·source/config snapshot·seed ledger·fit/final ID를 확인한 뒤 시작한다. interrupted run을 다시 돌리면 기존 부분 산출물을 삭제하거나 완료 반복으로 합치지 않는다. 프로세스 중단이 random stream/estimator를 어떻게 바꾸었는지 기록한다.

| 관측 | 조치 |
|---|---|
| density/target/sign/seed 오류 | 생산 실험 중단, oracle부터 수정 |
| 독립 reference 미확보 | raw/CV superiority 잠금, small toy·진단만 진행 |
| mesh 큰 차이 | fine-grid reference·주장 범위 우선 |
| static만 강함 | 학습·신경망 추가를 기본 채택하지 않음 |
| raw M₂ 개선으로 corrected 위험 오표기 | 결과 계약 수정 후 재감사·필요 시 새 평가 |
| CV training loss만 감소 | 새 reference/final/총비용 확인 전 성공 아님 |
| signed bound/경계 증명 불충분 | 해당 field 미채택 |
| generic CV와 차이 없음 | Volterra feature 우위 미지원 |
| ancestry만 개선 | augmentation 미채택 |
| cost 또는 budget 상한 도달 | unresolved·정확한 재개 지점 기록 |
| confirmation 보고 rule 변경 | 기존 세트 development 전환, 새 세트 필요 |

실패를 숨기지 않는 것은 목적이 아니라 원인 분리를 위한 계약이다. 제한된 repair 뒤에도 같은 조건이 해결되지 않으면 다음 algorithm을 무한 추가하지 않고 연구 질문/계산예산/target 범위를 재검토한다.

## 18. 단계별 완료 체크리스트와 바로 다음 실행

- [ ] V2-P0: 기존/new 상태 inventory, 역할·estimand schema, claim ledger.
- [ ] V2-P1: relevant q별 독립 raw M₂ 및 별도 μ corroboration.
- [ ] V2-P2: exact adjacent coupling 및 16/32/64 scope 판정.
- [ ] V2-P3: 제한 physical guide/weights·조건부 qα 대조.
- [ ] V2-P4: bounded analytic field·signed estimator·독립 oracle.
- [ ] V2-P5: risk-only 및 event-CV의 corrected-risk·총비용 결과.
- [ ] V2-P6: 필요성 있을 때만 선택 확장. 불필요하면 `not_required` 기록.
- [ ] V2-P7: parent부터의 whole training·practical finite-grid workload.
- [ ] V2-P8: 동결 confirmation·외부 재현.
- [ ] V2-P9: 유용한 정리·novelty·한계 및 원고 판단.

체크박스는 이 V2 계획의 acceptance 산출물 상태다. 기존 v7/과거 R단계 결과가 없다는 뜻은 아니다. 파일 생성만으로 체크하지 않는다.

**실행 요청을 받으면 시작할 첫 묶음:** P0 inventory 및 원래 estimator 계약 감사 → P1 independent reference runner/pilot → P2 coupling oracle. 병행 가능한 작은 작업으로 P4 analytic field 수식·oracle와 P9 선행연구 대조를 진행한다. 큰 실제-target CV·neural·bank 확장은 앞선 관문 없이 시작하지 않는다.

## 19. 계획 자체의 이론적·기술적 재검토

이번 통합에서 다음을 명시적으로 점검·보강했다.

1. q와 r, μ와 M₂,raw, M₂,CV, auxiliary variance를 분리했다.
2. Stein mean 0은 Gaussian prior에서 증명하며 non-log-concave 사건 law에 잘못 적용하지 않는다.
3. orthonormal U의 divergence와 bounded atom 공식을 명시했다.
4. signed estimator에 positive log-summary/SMC 계약을 재사용하지 않는다.
5. old M₂를 새 사건 CV variance로 쓰는 오류를 금지했다.
6. fine/coarse coupling에 coarse-kernel augmentation과 한 likelihood를 유지한다.
7. 작은 oracle·empirical development·confirmation·수학 증명 수준을 구분했다.
8. cost proxy, fixed-precision actual time, shared/inherited 비용을 구분했다.
9. convex fitting·Stein CV를 새 원리로 주장하지 않으며 novelty는 별도 gate다.
10. 미완료 mesh·reference를 더 큰 모델로 우회하지 않으며 조건부 확장과 실패 출구를 남겼다.
11. auxiliary mixture weights에는 F=h_q, 사건 mixture weights에는 F=g를 사용해 서로 다른 위험 목적을 분리했다.
12. 30개 comparison의 10% agreement margin과 reference RSE allocation이 양립 가능한지 확인하고, 기존 2.5% 상한만으로 통과를 기대하는 설계를 보완했다.

계획 작성 중의 제한된 수치 대조: 위 analytic atom의 7차원/feature rank 3, 세 lengthscale와 세 deterministic point의 9개 경우에서 full autograd divergence와 닫힌 식의 최대 차이는 약 3.89×10⁻¹⁶이었다. 이는 수식 구현 가능성의 작은 cross-check이지 P4의 전체 oracle 통과, global bound의 수치 인증 또는 CV 성능 실험이 아니다. reference allocation의 Bonferroni z와 half-width 숫자도 독립 계산해 확인했다.

이는 계획 단계에서 발견 가능한 오류를 줄이는 구조다. 아직 생성하지 않은 코드, 미실행 target 실험 또는 미증명 정리의 무오류를 보장하지 않는다. 각 단계의 독립 검토와 반례·회귀·새 표본 검증으로 판단을 갱신한다.
