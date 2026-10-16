# 야간 제한 실행계획: 독립 위험 검증 → mesh 진단 → 조건부 최소 보정

작성일: 2026-10-07

계획 작성 시 상태: **계획만 작성. 야간 실험·자동화·커밋·푸시는 당시 시작하지 않음.**

기본 실행 창: 최대 6시간 30분, 제한된 실패 대응 포함 절대 상한 7시간 30분. 사용자의 수면시간이나 완료시간을 보장하는 의미는 아니다.

## 1. 이 계획이 따르는 기준과 현재 출발점

상위 계획은 [모델 구조 재검토와 개선계획](MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md)이다. 이 문서는 그 계획의 S2·S3·S5를 현재 결과에 맞게 야간 작업으로 분해한다. 기존 관문을 완화하거나 새로운 모델 확장을 승인하는 문서가 아니다.

함께 반영한 후속 근거:

- [위험 geometry 결과와 D0–D3 실행계획](STRUCTURAL_IMPROVEMENT_RISK_GEOMETRY_AND_NEXT_PLAN_2026-10-07_KO.md).
- [전체 auxiliary 재학습 안정성 결과](AUXILIARY_WHOLE_FIT_STABILITY_2026-10-07_KO.md).

현재 확인된 사실:

1. N=32, 두 개발 셀의 고정 q 10개에서 auxiliary bank·fit·final을 새로 실행한 50회가 모두 개발용 위험 정밀도 관문을 통과했다.
2. risk-fit/static blend의 M₂ 추정 RSE 중앙값은 canonical 2.57%, 높은 η 2.53%다. 이것은 사건확률 μ의 RSE가 아니다.
3. 동일 표본 수로 보정한 관측 분산은 static Volterra guide 단독이 더 낮다. learned blend의 추가 효용은 미입증이다.
4. 두 estimator가 static 성분을 공유하므로 서로의 일치를 독립 reference 인증으로 쓰지 않는다.
5. 큰 기여 경로에서 한 시간 구간이 integrated variance의 약 99%를 차지하는 사례가 남아 있다. 보조분포 안정화가 discretization 민감성을 제거하지 않았다.
6. original q의 사건확률 효율, 전체 original-model 재학습의 안정성, 연속시간 정확성, 저널용 confirmation은 미완료다.

따라서 야간 목표는 **독립 위험 검증을 한 단계 진전시키고, 유한격자 현상과 mesh 문제를 분리하며, 조건이 갖춰진 경우에만 작은 q 보정을 평가하는 것**이다. 아침까지 저널 수준 완성이나 성능 우위를 보장하지 않는다.

## 2. 작업 순서와 시간·계산량 상한

| 순서 | 작업 | 시간 상한 | potential 평가 상한 | 다음 결정 |
|---|---|---:|---:|---|
| O0 | 증거 봉인, 이론·oracle·runner 계약 보강 | 45분 | 200만 | 정확성 문제 없을 때만 큰 실험 |
| O1 | static IID IS와 별도 risk-SMC의 독립 교차 검증 | 120분 | 1억 6천만 | 위험 reference 관문 판정 |
| O2 | 정확한 adjacent coupling과 N=16/32/64 mesh 진단 | 90분 | 7천만 | coarse-grid 민감성·미해결 항목 판정 |
| O3 | **조건부** μ reference 확보 및 최소 qα 보정 pilot/반복 | 90분 | 1억 8천만 | 같은 정확도 자격의 비교만 허용 |
| O4 | 전체 검증·결과 감사·보고서·재개 지점 정리 | 45분 | 대규모 새 실험 없음 | 성공/실패/미해결을 분리해 인계 |

기본 합계는 6시간 30분이다. O1 실패 시 한 번의 제한된 별도 repair 묶음에 최대 60분을 추가할 수 있지만, **전체 7시간 30분 및 4억 5천만 potential 평가 상한**을 넘지 않는다. 단계별 상한은 다른 단계의 남는 예산을 자동으로 가져와 늘리지 않는다. 실패·pilot·warmup·reference 비용을 전역 ledger에 별도로 기록한다.

시간은 예약한 작업 창이지 예측 소요시간이 아니다. 신규 pilot이 throughput과 메모리를 보여준 뒤 production count를 봉인한다. 남는 시간은 더 많은 후보를 임의로 찾는 데 쓰지 않고 검증·문서·재현성에 우선 사용한다.

potential 상한은 새 stress-target 실험의 실제 payoff 평가 호출에 적용한다. oracle·전체 회귀 테스트는 별도 검증 시간/작업 ledger로 구분한다. 기존 test suite의 전체 내부 호출을 계측하지 못했다면 미측정으로 표시하지, 0회로 합산하지 않는다. guide basis probe·density 계산·압축·I/O는 해당 작업량과 시간으로 별도 집계한다.

### 노트북 운영 조건

- 유료 cloud/Runpod, GPU 대여, 패키지·드라이버 설치, OS 전원 설정 변경은 하지 않는다.
- 시작 전에 실제 CPU/RAM, free disk, 실행 중 무거운 작업, Python 환경을 읽기 전용으로 점검한다.
- scientific timing은 한 방법씩 순차 실행한다. 전체 pytest·압축 등 무거운 작업을 timing 실험과 동시에 실행하지 않는다.
- torch thread 수는 기본 1로 고정한다. method별 임의 변경은 금지한다.
- 배치 크기는 기본 8,192 이하이며, RSS 운영 상한은 `min(4 GiB, 물리 RAM의 25%)`로 설정한다. OOM 방지를 위한 운영 기준이지 성능 비교의 새 표본 선택 규칙이 아니다.
- 시작 free disk 10 GiB 미만이면 큰 artifact 생성 전에 중단/축소 사유를 기록한다. 기존 evidence를 지워 공간을 만들지 않는다.
- 절전·종료·연결 중단이 발생하면 지속 실행을 보장할 수 없다. 재개는 저장된 ledger와 완료된 whole job에서 한다.
- 커밋·푸시·PR 병합은 하지 않는다. 사용자가 별도로 요청할 때만 한다.

## 3. O0 — 실행 계약과 작은 수학적 검증

### 3.1 evidence 및 역할 분리

1. 현재 branch/HEAD/dirty diff, 환경, 입력 artifact SHA와 전체 q digest를 기록한다. 기존 변경을 되돌리지 않는다.
2. 대용량 raw JSON은 그대로 보존한다. `.json.gz`와 `.source.zip`의 checksum 및 byte-identical roundtrip을 검사한다.
3. 새 단계별 config·source snapshot·result 이름을 사용한다. 완료·실패 artifact를 덮어쓰지 않는다.
4. 최소 역할을 분리한다: `design-pilot`, `reference-smc`, `reference-iid`, `parent-training`, `correction-training`, `allocation-pilot`, `selection`, `final-probability`, `final-risk`, `mesh-augmentation`, `audit`.
5. 기존 v6/v7 final은 설계 근거인 development로 남긴다. 새 reference·selection·final에 다시 넣지 않는다.
6. 단계를 실행하는 동안 그 snapshot의 runtime source/config를 수정하지 않는다. 다음 변경은 실행·저장 종료 후 새로운 snapshot으로 진행한다.

### 3.2 반드시 검사할 항등식

고정 q, p=N(0,I), 0≤g≤1, q≥δp에서

\[
h_q=\delta g^2p/q,\quad E_p h_q=\delta M_2(q),
\quad Y_r=g^2p^2/(qr),\quad E_rY_r=M_2(q).
\]

`q,r≥0.1p`이면 `Y_r≤100`이다. 작은 상대 SE는 이 bound에서 자동으로 따라오지 않는다.

static guide의 각 mean μ와 finite-grid Volterra matrix B에 대해

\[
E[Y]=B\mu,\quad\operatorname{Cov}(Y)=BB^T,
\quad E[V_j]=\xi_j\exp(\eta(B\mu)_j).
\]

위 식은 simulator의 Gaussian variance compensation과 동일한 convention 아래에서 사용한다. row variance·정규화가 다르면 먼저 맞춘다. integrated-variance 평균도 같은 left-point 일정으로 합산한다. 작은 shift·toy에서 독립 Gaussian mgf 계산과 simulator를 비교하며, 큰 κ에서 단순 MC 평균의 불안정을 식의 실패로 오인하지 않는다.

추가 검사:

- near-zero mean을 exact natural component로 세지 않는 density floor.
- full-mixture log density, covariance unchanged, 다음 price coordinate와 과거 Volterra 방향의 직교성.
- pCN/global independence MH acceptance의 상세 균형.
- SMC constant potential 및 smooth Gaussian oracle의 normalizer, mutation/resampling 없는 MC 일치.
- log-domain estimator에서 log 평균을 raw 평균처럼 읽거나 self-normalization하지 않는지 검사.

**정확성 오류 발견 시:** 생산 실험은 보류하고, 원인·수정·회귀 검증부터 완료한다. 성능을 얻기 위해 payoff/weight를 clipping하지 않는다.

## 4. O1 — 독립 M₂ reference 교차 검증

### 4.1 서로 다른 추정 경로

고정된 10개 original q에서 다음 세 경로를 비교한다.

| 경로 | 추정 방식 | SE의 독립 단위 |
|---|---|---|
| R-IID | frozen static guide에서 ordinary IID IS로 M₂ | 새 IID draw 또는 사전 고정 독립 block |
| R-SMC-local | natural 시작, pCN만 사용하는 risk-SMC normalizer/δ | **전체 SMC 실행** |
| R-SMC-global | 동일 risk target, pCN + exact static-guide independence MH | **전체 SMC 실행** |

R-IID와 R-SMC-global은 guide geometry를 공유한다. 난수를 공유하지 않더라도 공통 blind spot 가능성이 있다. 따라서 guide를 쓰지 않는 R-SMC-local 대조를 남긴다. 세 방법도 payoff/model 코드 일부를 공유하므로 독립 구현·연속시간 진실을 인증하지 않는다.

SMC terminal particle을 IID reference 관측으로 취급하지 않는다. normalizer가 작은 것 자체를 mixing 난이도나 편향의 증거로 쓰지 않는다.

### 4.2 pilot: 최대 세 개 matched-work schedule

pilot q는 성능으로 고르지 않고 각 개발 셀의 `parent_training_rep=0`으로 고정한다.

공통 설정: particles=256, temperatures=96개(0과 1 포함), bridge power=2, mutation=8, pCN scale=.35, stratified resampling.

| 설정 | resampling 간격 | global 이동 |
|---|---:|---|
| A | 2 | 없음 |
| B | 4 | 없음 |
| C | 4 | mutation 중 매 2번째 |

전체 replicate당 potential은 현재 코드의 일정에서

\[
256[1+(96-1)8]=194,816.
\]

각 셀·schedule에서 whole SMC 8회: `2×3×8×194,816=9,351,168` potential. 실제 실행 중 호출·실패 비용을 대조해 이 계수를 확인한다. endpoint mutation schedule이 변경되면 production 전에 계수와 config를 다시 봉인한다.

pilot 기록: 전체 normalizer 분포, RSE, 단일 whole replicate의 기여 집중, weight ESS, ancestry, local/global acceptance, 이동 거리, Volterra peak 시간, I·J, 독립 실행별 모드·tail 기여, wall time/RSS. ancestry 하나로 winner를 정하지 않는다.

A/B 중 production local schedule은 두 셀의 worst-case precision/cost와 혼합 진단으로 정한다. reference 평균에 가장 가까운 schedule을 고르지 않는다. C는 global 대조로 남긴다. 모든 pilot 결과는 공개하며 production에 합치지 않는다.

### 4.3 새 production allocation

- 기본 출발점: 모든 10개 q에서 local/global 각각 whole SMC 32회.
- 예상 potential: `10×2×32×194,816=124,682,240`.
- R-IID는 모든 q에서 **새 1,048,576개** 표본을 기본안으로 둔다. 합계 10,485,760회, 사전 고정 block과 전체 moments를 기록한다.
- 위 기본안은 pilot과 합쳐 약 1억 4,452만 potential로 O1 상한 안이다. 정밀도 달성 보장은 아니다.
- pilot에서 이 설계가 부족하거나 120분/메모리 상한 밖이라고 판단되면 실행 전에 범위와 unresolved 사유를 봉인한다. 낮은 RSE가 나올 때까지 final을 임의 연장하지 않는다.
- 필요 반복 수의 `CV²/r²` 외삽은 allocation 참고만 사용한다. 작은 pilot의 CV를 참값으로 취급하지 않는다. 여유계수와 상한을 결과 열람 전에 명시한다.

### 4.4 개발용 reference 관문

각 q를 따로 판정한다. 서로 다른 q의 M₂를 공통 참값으로 합치지 않는다.

1. 정확한 likelihood·source/q binding·완료 job·seed 분리가 모두 유효.
2. production RSE 목표: 각 reference 경로 **2.5% 이하**. 부족하면 unresolved다.
3. 세 경로의 pairwise agreement는 상대 margin 10%를 기준으로 평가한다. 비교 차이의 SE는 독립 전체 단위에서 계산한다.
4. 10개 q×3개 pair=30개 주 비교를 사전 선언하고, 정규근사 판정에는 `z=Φ⁻¹(1−.05/(2×30))`를 사용한다. 실제 차이와 불확실성 상한이 margin 안인지 검사한다.
5. whole-replicate outlier·leave-one-whole-replicate-out 민감도·새 IID block 불일치가 크면 낮은 sample RSE만으로 pass하지 않는다. 민감도 기준은 pilot 후 production 전에 봉인한다.

이 gate는 **bounded finite-grid development corroboration**이다. Bonferroni 보정이 heavy-tail 유한표본 정규근사를 정확한 coverage로 바꾸지 않는다. bootstrap도 whole SMC 단위로 하며 누락 tail을 배제하는 증명으로 쓰지 않는다. distribution-free 상대오차 인증 또는 oracle라는 명칭은 사용하지 않는다.

### 4.5 실패 대응

- likelihood/weight 오류 → 큰 실험 중단, O0로 돌아가 수정.
- local SMC 불안정, global만 안정 → guide 의존성을 공개하고 reference gate 보류.
- schedule 간 차이 또는 whole-run tail 불안정 → 한 번만 제한된 별도 repair pilot을 허용. 최대 60분, 기존 실패 보존, 새 config·새 whole runs.
- 추가 변경은 유효한 kernel·새 고정 bridge 범위 안이다. sample mean을 맞추려고 rule을 바꾸지 않는다.
- 끝까지 미해결이면 O3의 q 보정·승리 판정을 열지 않는다. O2 및 증거·이론 작업으로 진행한다.

## 5. O2 — 정확한 coupling과 mesh 민감도

### 5.1 질문과 추정 대상

주 질문은 `μ_16, μ_32, μ_64`와 adjacent difference가 어떻게 달라지는가다. 서로 다른 grid의 q/M₂를 같은 추정 대상으로 비교하지 않는다. 현재 original N=32 모델을 임의로 N=64로 옮겨 같은 q라고 부르지 않는다.

N=128, 연속시간 bias theorem, asymptotic rate fitting은 이번 핵심 야간 범위가 아니다. adjacent difference가 작더라도 연속시간 bias 상한을 선언하지 않는다.

### 5.2 기존 정확한 Brownian coupling 재사용

출발점: `src/path_integral/rbergomi_coupling.py`의 `adjacent_local_gaussian_coefficients`, 기존 conditional payoff와 simulator.

fine 첫 구간의 `(ΔW₁,L₁,C₁)` joint Gaussian과 둘째 구간의 `(ΔW₂,L₂)`를 사용한다. `C₁`은 coarse endpoint kernel에 대한 첫 구간 적분이다.

\[
\Delta W_c=\Delta W_1+\Delta W_2,\qquad L_c=C_1+L_2.
\]

**L_c=L₁+L₂로 대체하지 않는다.** fine local white 좌표와 독립 augmentation 좌표에서 이 joint Gaussian을 생성하고, coarse local pair를 coarse Cholesky로 whiten한다. coarse local integral의 cross-kernel covariance와 quadrature tolerance를 검사한다.

fine proposal을 joint law로 lift할 때 augmentation의 conditional reference law는 유지한다. joint density ratio는 fine full-mixture ratio다. correction은

\[
D_{f,c}=(g_f-g_c)\,p_f/q_f
\]

라는 **하나의 공통 likelihood**로 계산한다. fine/coarse 각각 다른 likelihood를 correction에 붙이지 않는다.

우선 구현·검증할 oracle:

- natural fine/coarse 각 marginal covariance와 simulator 일치.
- joint cross covariance, coarse increment aggregation, covariance positivity.
- identity/작은 deterministic mean shift 아래 독립 dense Gaussian 식과 density/weighted moment 비교.
- constant 및 작은 analytic Gaussian payoff의 signed difference.
- augmentation seed가 fine path/label/reference와 겹치지 않는지.

이 adapter는 신규 구현일 수 있으며 기존 causal-control MLMC sampler와 arbitrary local Gaussian mixture의 API가 자동 호환된다고 가정하지 않는다. oracle가 실패하면 production coupled 결과를 만들지 않는다.

### 5.3 pilot 및 본 진단

- 두 개발 셀, grid 16/32/64, coupling pair 16–32/32–64.
- 각 pair·cell에서 작은 독립 pilot을 먼저 실행한다. 초기 65,536개의 coupled draw를 출발점으로 둔다.
- pilot 뒤 본 count를 `{262,144,524,288,1,048,576}` 중 봉인한다. 본 결과를 보고 count를 늘리지 않는다.
- fine/coarse payoff 두 호출을 모두 세고, grid별 scalar potential 수 외에 `path×step`, guide component 수, density wall time을 공개한다.
- 추가 independent-per-grid estimate는 coupling marginal sanity check에 사용한다. 독립 차이에 paired SE를 적용하지 않는다.

보고 항목: μ_N·SE, signed adjacent mean·SE, 상대 grid difference와 불확실성, I·J·max variance·peak time·largest-cell share, 상위 contribution, 위험 concentration, wall time/RSS.

### 5.4 결과 해석과 중단

- operational warning: adjacent relative mean difference가 10% 이상이거나 그 구분 자체가 불확실하면 N=32의 continuum/practical 해석을 보류한다. 이 10%는 이론적 bias bound나 논문 합격선이 아니다.
- 성능 비교 전에 상대 difference CI·SE와 표본 precision을 같이 본다. 값이 작거나 CI가 0을 포함한다는 이유로 mesh bias=0이라 쓰지 않는다.
- coupling이 미검증이면 independent-per-grid 탐색만 남긴다. 이를 exact paired mesh 결과로 부르지 않는다.
- 큰 mesh 민감도가 나타나면 저해상도 모델 확장보다 fine-grid reference·coupling·이론을 다음 우선순위로 올린다.

## 6. O3 — 조건이 갖춰진 경우에만 최소 보정

### 6.1 진입 조건

O1의 relevant q에 대한 독립 위험 corroboration, O2의 finite-grid coupling 계약, 새 사건확률 μ reference의 adequacy가 필요하다. mesh 민감도가 크면 이번 단계의 큰 비교를 생략하고 원인/mesh 검증으로 전환한다.

**M₂ reference를 μ reference로 사용할 수 없다.** O1 성공만으로 사건확률 accuracy 자격은 생기지 않는다.

μ reference는 g-target whole SMC와 새 ordinary static-guide IS를 별도로 생성해 교차 확인한다. reference SE가 method SE의 1/5 이하라는 상위 계획 조건을 유지한다. 높은 정밀도 method 때문에 reference가 상대적으로 불충분해지면 qualification을 보류하며, 좋은 method에 인위적으로 noise를 넣거나 표본을 버리지 않는다.

이 단계의 μ reference pilot·production·학습·선택·final 합계가 O3 상한 안에 들어가지 않으면 numerical dominance 비교는 시작하지 않는다.

### 6.2 후보는 작은 고정 집합

처음에는 셀당 새 parent 1회 pilot, 조건이 유지되면 셀당 **original parent 생성부터 새 5회**를 출발점으로 둔다.

각 새 parent q₀에 대해 다음 6개 candidate를 준비한다.

1. q₀ 그대로.
2. `q₀`와 event-bank/static safeguard r의 α=.25 mixture.
3. 같은 event correction의 α=.5.
4. matched-work risk-bank/static safeguard correction의 α=.25.
5. 같은 risk correction의 α=.5.
6. **static-only q_V** — 현재 가장 중요한 추가 baseline.

\[
q_\alpha=(1-\alpha)q_0+\alpha r,
\quad\alpha\in\{0,.25,.5\}.
\]

event/risk bank의 potential 예산·component 상한·rank·shrinkage·static safeguard 질량을 동일하게 맞춘다. static-only는 fitting이 없어 실제 total cost가 작으므로 이 차이를 숨기지 않는다. component 수 차이와 density cost를 명시하며 모든 차이를 risk-target 효과로 귀속하지 않는다.

q_V는 자체적으로 exact defensive proposal이다. α=1에 `M₂(qα)≤M₂(q₀)/(1−α)`를 적용하지 않는다. α<1의 이 식도 개선 보장이 아니라 악화 상한이다.

### 6.3 통계 역할과 비용

- parent·event/risk fitting·allocation·selection·final probability·final risk 표본을 모두 분리한다.
- selection 전에 후보 6개를 모두 동결한다. event/risk 각 family에서 α=.25/.5 중 하나를 별도 selection 표본의 위험·목표 정확도 forecast·setup/inference 비용으로 고른다. 선택 목적함수·query 수·tie-break와 uncertainty 처리 방식은 selection 시작 전에 고정한다.
- primary final 비교군은 `q₀`, `static-only`, `선택된 event correction`, `선택된 risk correction`이다. 모든 final을 본 뒤 가장 좋은 α를 다시 선택하지 않는다. 선택되지 않은 후보의 selection 기록도 보존한다.
- selection에서 μ 또는 M₂로 만든 ratio/표본 수 forecast는 불편 추정량이나 정확한 RMSE라고 부르지 않는다. final accuracy용 독립 reference와 selection용 추정값을 분리한다.
- 최초 allocation pilot 16,384개/후보를 출발점으로 하되, count 후보 `{65,536,262,144,1,048,576}`와 안전계수·상한을 final 전에 봉인한다.
- 필요한 count가 상한보다 크면 해당 후보를 unresolved로 남긴다. RSE 목표를 낮춰 통과시키지 않는다.
- 기본 개발 probability RSE 목표 5%, 기존 equivalence margin 25%를 유지한다. 이는 relative RMSE≤5% 인증이나 저널용 최종 자격이 아니다.
- 이미 공개한 canonical/높은 η 셀의 새 seed 실험도 development다. 이들을 미사용 confirmation task라고 부르지 않는다.
- 전체 parent·bank·fit·selection·inference·실패·retry를 포함한다. 실제 공유 실험비와 각 method 단독 deployment 비용을 구분한다.
- 비교 순서·warmup·thread·I/O 포함 여부를 봉인하고 순차 timing한다. 단일 관측 시간만으로 speed theorem을 주장하지 않는다.

### 6.4 판단

- 최소 두 method가 같은 accuracy 자격을 얻은 셀에서만 fixed-precision total wall-time을 비교한다.
- q₀가 부적격이면 qα의 q₀ 대비 가속비를 winner로 발표하지 않는다. 적격인 static-only와 correction의 비교는 별도로 가능하다.
- 약 20% total-cost 개선은 추가 개발을 위한 실용적 신호일 뿐, 통계적 증명/저널 기준이 아니다.
- correction이 static-only보다 비용·안정성에서 유리하지 않으면 learned addition 우위 가설을 미지원으로 남긴다. 추가 operator/rank로 덮지 않는다.
- 저널용 24-task×20-whole-training 확인 실험은 이번 밤에 시작하지 않는다.

### 6.5 O3가 잠긴 경우의 대체 작업

억지로 다음 모델을 실행하지 않고 다음을 수행한다.

1. O1의 whole-run tail·mixing 및 O2의 spike/mesh 실패 원인 보고서.
2. fixed-q에서 event mass와 M₂ 기여를 구분하는 새 calibration/discovery 진단. 기준을 새 final에 맞춰 다시 움직이지 않는다.
3. static guide의 field mean/energy/시간 위치가 payoff 기여와 연결되는 finite-grid 수식·toy 검증.
4. 기존 claim ledger의 unresolved theorem과 이번에 실제로 검증한 명제를 분리한다. 알려진 IS/MH/최소 에너지 식을 새 정리로 포장하지 않는다.
5. 노트북 예산으로 부족했던 exact job/count/precision과 다음 실행에 필요한 계산량을 정리한다.

이 대체 작업이 더 유용하지 않거나 실행 상한에 도달하면 보고서를 완료하고 종료한다. 긴 시간 자체를 목표로 불필요한 계산을 늘리지 않는다.

## 7. O4 — 아침에 확인할 산출물

### 예정 파일: 실행 때 생성하며 현재 완료로 간주하지 않음

기존 abstractions를 우선 재사용한다. 아래 새 경로는 제안이며, 실제 구현 전에 중복을 확인한다.

- `experiments/post_audit_r2_independent_risk_reference.py`: O1 실행·산술/seed 감사.
- `src/path_integral/volterra_adjacent_local_coupling.py`: 필요할 경우 O2 local mixture/augmentation adapter.
- `experiments/post_audit_r2_mesh_excursion_diagnostics.py`: O2 pilot·production·검증.
- `experiments/post_audit_r2_guarded_correction.py`: O3가 열릴 때만.
- 관련 `tests/`, 단계별 `configs/post_audit/r2_overnight_*_v1.yaml`.
- `results/post_audit/r2_overnight_*_v1.json[.gz]`, source ZIP, 실패·whole-job ledger.
- `docs/reviews/OVERNIGHT_EXECUTION_REPORT_2026-10-07_KO.md`: 최종 설명·결정표·재개 지점.

### 최종 감사

- 입력·source/config·q/guide/proposal binding, 새 seed와 역할의 완전성.
- whole-run와 particle/batch SE 단위, log/raw normalizer conversion.
- completion grid, comparator 누락, stale pass, failed work/time, memory/cap 검사.
- exact coupling marginal/cross covariance와 하나의 joint likelihood.
- independent accuracy 및 reference uncertainty가 들어간 판정 재계산.
- 관련 oracle·단위·통합 테스트 → 전체 pytest → Ruff → mypy → `git diff --check`.
- gzip roundtrip 및 재현 명령. raw evidence는 변경하지 않는다.

### 보고서가 답해야 할 질문

1. static guide의 낮은 M₂ RSE를 다른 추정 경로가 실제로 지지했는가?
2. 학습 bank의 혼합/기여 모드 문제는 어떤 형태로 남았는가?
3. N=32의 큰 기여는 격자를 바꾸어도 유지되는가? 차이를 충분히 정밀하게 측정했는가?
4. learned correction은 static-only보다 같은 accuracy에서 실제로 유리한가, 아니면 불필요한 비용인가?
5. original-model fresh training부터 final까지 검증한 항목과 fixed-q 조건부 결과가 무엇인가?
6. 다음에는 모델을 개선해야 하는가, reference/mesh를 개선해야 하는가, 주장을 좁혀야 하는가?

결론은 `통과 / 제한적 통과 / 실패 / 미해결 / 실행하지 않음`으로 구분한다. 모든 정확성·이론 오류를 배제했다거나 독립 연구 재현을 완료했다고 쓰지 않는다.

## 8. 최종 범위

이번 계획은 **실행 승인 전 문서**다. 실행 요청이 오면 O0부터 시작해 관문에 따라 진행한다. 앱·컴퓨터의 무기한 작동, 예산 초기화, 모든 단계 성공은 보장하지 않는다. 실행 중 새 권한이 필요한 외부 자원·데이터·조작이 발생하면 가정으로 처리하지 않는다.

가장 바람직한 아침 결과는 "더 큰 모델을 만들었다"가 아니라, **독립 위험 검증과 mesh 검증을 통해 다음 연구 선택의 근거를 확보했고, 가능한 경우 최소 보정의 정확도·총비용을 공정하게 비교했다**는 것이다.
