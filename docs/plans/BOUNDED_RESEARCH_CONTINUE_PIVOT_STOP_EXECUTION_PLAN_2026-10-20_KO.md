# 제한된 판별 연구: 계속·전환·중단 실행계획

작성일: 2026-10-20

문서 상태: **실행 전 계획안**. 이번 요청에서는 계획만 작성하며 새 학습·physics 실험을 실행하지 않는다.
기존 결과·감사·코드를 수정하지 않고, 커밋·푸시도 하지 않는다.

## 0. 결론과 이번 연구의 종료점

이번에는 “어떻게든 성공할 때까지 모델을 개선한다”를 목표로 하지 않는다.
**한 번의 사전 고정된 판별 사이클**로 다음 투자가 합리적인지 결정한다.

핵심 질문:

> 같은 물리 경로 사전에서 mixture-label 변동을 제거하면, 위험 평가가 안정되고
> 학습 제안분포의 독립 위험·정밀도·총비용에 재현 가능한 이득이 생기는가?

이 질문은 bank와 geometry를 **부분적으로 분리**한다. 전체 원인을 완전히 식별하지는 않는다.
층화 실패는 geometry 부족의 증명이 아니다. 층화 성공도 dictionary 충분성의 증명이 아니다.
Gaussian 성분 내부의 residual rare tail은 이 실험에서도 남을 수 있다.

이번 사이클의 산출물은 새 모델 이름이 아니라 다음 셋 중 하나의 판단 보고서다.

- `CONTINUE_CANDIDATE`: 제한된 개발 증거가 있어 별도 확인 연구를 설계할 가치가 있음.
- `PIVOT_WEIGHT_ONLY`: 평가 가능한 범위에서 가중치 학습의 실익이 부족해 이 branch를 종료.
- `STOP_UNRESOLVED_BUDGET`: 이번 하드웨어·예산에서 판별 불가. 자동 증설·재시도 없음.

어느 분기도 기존 P1/P2/large-training 자격이나 저널 제출 자격을 부여하지 않는다.
“비유의함”을 “효과 없음의 증명”으로, “관측 RSE 감소”를 “확정 속도 개선”으로 쓰지 않는다.

## 1. 출발점: 무엇이 확인됐고 무엇이 아닌가

직전 실험은 두 N32 stress 셀 ×5독립 학습, 같은373성분에서 uniform/event-KL/risk-M₂를 비교했다.
계획된263units와9,503,232payoff calls, 별도524,288calls 재현 감사가 완료됐다.

| 항목 | 현재 증거 | 의미하지 않는 것 |
|---|---|---|
| high-η 관측 RSE | uniform10.11%, event-KL4.43% 중앙값 | population variance/총비용 우위 확정 |
| 독립 위험 비교 | 30쌍의 복합 감소 판정0건 | 효과0의 증명 |
| risk train ESS | 16,384개 중1.12–16.73 | 충분한 coverage 또는 SMC ancestry |
| 큰 단일 기여 | train 위험 합의 최대94.67% | 그 경로가 population 위험의94.67%라는 뜻 |
| 최적화 | 20fits 모두256상한, 엄격한 tolerance 미충족 | 학습이 전혀 안 됐거나 optimizer만 고치면 해결 |
| 검증 | 1,613tests/린트/타입/대표 replay 통과 | 새로운 수학적 원리·실용성 입증 |

원본 anchor:

| 파일 | SHA256 |
|---|---|
| `results/post_audit/dictionary_model_learning_v1/session_20261020/result.json` | `3964e74bb620da10b34a5c8256a46286b3a46a8eb5ab676031314fe647273033` |
| 같은 directory의 `audit.json` | `7183ff46af14cb4fcdd9b768fac9a6bf54063d43ca682f0bd618d0190e011de2` |
| `results/post_audit/dictionary_learning_diagnosis_v1/session_20261020/result.json` | `6393fe9415a52059fb6c827a4e0d6119a25f500bcd532df694eca11c48f9dd6f` |
| 같은 diagnosis directory의 `audit.json` | `3808603a303d1e0e2a2230fca6a5ca45b47f9e646fe8a8bd76c7f247c9774475` |

직전 보고서: `docs/reviews/PHYSICAL_DICTIONARY_MODEL_LEARNING_RESULT_2026-10-20_KO.md`.

## 2. 범위 고정: 추가하지 않을 것

- N32, dim64, 기존 canonical(H=.05,η1.5) 및 high-η(η2.0) 두 셀만 사용한다.
  spot100, threshold1, maturity1, xi.04, rho−.7과 finite-grid conditional g를 유지한다.
- 373개 mixed-window Gaussian component의 각 mean/identity covariance를 고정한다.
  weight가 바뀌면 전체 mixture mean/covariance는 바뀐다.
- natural .1, uniform shifted floor .1, adaptive .8의 기존 domain을 유지한다.
- K2 재압축, covariance/rank 증가, 새 window/진폭, 새 operator/신경망, quantum 요소,
  SMC schedule 탐색, grid 증가, GPU·유료 cloud 도입은 이번 사이클에서 제외한다.
- 신규 생산 학습 objective는 **event-KL 하나**로 제한한다. 직전 완료 결과를 보고 지금
  고른 선택이다. risk-M₂를 유리한 seed에서 되살리지 않는다. risk-M₂ 수학/기존 결과는
  oracle와 frozen-family 진단에 남기되 새 생산 학습 factor로 추가하지 않는다.
- 기존의 α∈{0,.25,.5}, tie 시 작은α, FW최대256/line search40/tolerance1e-6을 유지한다.
  optimizer 상한 도달 여부를 공개한다. 반복 수를 늘려 통과시키는 별도 탐색은 하지 않는다.

실무적 주장도 제한한다. 이번 target은 두 stress probability다. 일반 pricing,
calibration, Greeks, 실제 거래비용 절감이나 시장 적합성은 별도 검증 전 주장하지 않는다.

## 3. 전체 순서와 단계별 문턱

| 단계 | 수행 | 다음 단계 조건 | 새 Volterra payoff calls 상한 |
|---|---|---|---:|
| D0 | 기존 checkpoint의 direct X² 민감도와 증거 점검 | 증거 연결·집계 정합성 통과 | 0 |
| D1 | 층화 수학·통계 core, Gaussian oracle, 검정력/비용 점검 | 구현/정답/예산 사전 점검 통과 | 0 |
| D2 | 기존 frozen q family의 IID 대 비례층화 평가 | 사전 정한 개발 평가가능성 gate 통과 | 952,320 |
| D3 | 조건부 새 bank 학습·선택·pilot·새 confirmation | 자동 연장 없이 최종 판정 | 12,522,240 |
| D4 | 전체 감사·회귀·비용/신규성/계속·전환·중단 보고 | 사이클 종료 | 아래 별도 감사 예산 |

공통 resource preflight는 별도4,960calls다.
**생산 최대13,479,520calls**, 대표 replay 별도 최대1,000,000calls를 예약한다.
합계 신규 Volterra 생산+감사 hard cap은 **14,479,520calls**다.
D2에서 실패하면 D3를 실행하지 않는다. 미집행 예산은 다음 탐색으로 자동 이전하지 않는다.

시간은 소요 보장이 아니라 상한이다. 생산 D2는900초, D3는3,600초,
preflight/준비는300초, 전체 생산 phase는7,200초다. 감사는별도1,800초,
회귀테스트는별도3,600초 상한이다. CPU float64/1thread, RSS4GiB, cloud 없음.
설계·구현 작업은2–5집중 작업일의 예산안으로 잡되, 초과가 예상되면 범위를 다시 결정한다.

## 4. D0 — 새 표본 없이 direct M₂의 불안정성 복원

### 4.1 입력과 수행

완료된30개 최종 estimator의 각8개 `final*.finish.json`을 읽는다. journal envelope,
manifest, identity, count, proposal digest, seed digest를 검증한 후 아래를 계산한다.

각 block b의 표본 수 n, final contribution X=g/Dq에 대해:

\[
\overline{X^2}_b=\exp(\mathrm{log\_sum\_squares}_b-\log n).
\]

- 8개의 같은 크기 독립 block에서 관측 direct M₂ 평균과 block-based SE/RSE.
- whole-block leave-one-out, 최대 block 기여, 상·하4block 평균 차이.
- 저장된 `log_maximum`으로 최대 관측 X² 기여 비중:
  `exp(2*max(log_maximum_b) − logsumexp(log_sum_squares_b))`.
- 독립 G-heldout M₂와의 평균/불확실성 비교. 두 추정법의 sampling law와 SE 단위를 분리.
- 확률 RSE가 낮지만 X² block 민감도가 큰 사례, 두 M₂ 평가가 모두 불안정한 사례를 표로 정리.

### 4.2 이 단계에서 금지하는 해석

저장된2차 moments만으로 X²의 raw IID SE는 계산할 수 없다. 그 계산에는 X⁴ moment가 필요하다.
8block RSE를 raw IID RSE로 표시하거나 확률 RSE를 빌려 쓰지 않는다.
원래 seed를 재생해서 X⁴를 얻는 것은 새로운 physics 비용이므로 D0에는 포함하지 않는다.
8block bootstrap/LOO는 관측 민감도이지 미관측 tail의 부재 증명이 아니다.

### 4.3 산출물과 gate

- 새 zero-physics 분석 module/test/result/audit. 입력240checkpoint의 hash도 source/manifest에 연결.
- parent result의 merged log second moment와 block에서 복원한 값이1e-12 absolute log tolerance 내 일치.
- 전체30estimators/240blocks가 모두 처리돼야 한다. 일부 성공 prefix 보고 금지.
- 증거 손상/집계 불일치면 D1으로 넘어가지 않고 원인을 보고한다. 원본 파일은 고치지 않는다.

## 5. D1 — 추정량과 학습 목적을 먼저 검증

### 5.1 정확한 비례 층화

\[
G=\sum_{j=0}^{372}\pi_jp_j,\quad \pi_0=.1=124/1240,\quad
\pi_j=.9/372=3/1240.
\]

최소 balanced tile1,240개는 natural124 + 각 shifted3개다. 여기서 사용할 주요 크기는:

| 용도 | 전체 N | natural | 각 shifted | balanced block |
|---|---:|---:|---:|---:|
| D2 block | 29,760 | 2,976 | 72 | block 자체가 전체373strata 포함 |
| D3 train/common selection/risk pilot | 19,840 | 1,984 | 48 | risk pilot은8×2,480 |
| D3 risk confirmation 최소 | 39,680 | 3,968 | 96 | 8×4,960 |
| D3 risk confirmation 최대 | 158,720 | 15,872 | 384 | 8×19,840 |

IID 대조도 같은 N을 쓴다. 다만 IID labels는 무작위, 층화 labels는 고정이다.
층화 block은 매번 모든 성분을 포함해야 한다. component별 정렬 배열을8등분한 것은
독립 balanced block이 아니므로 그 block SE를 사용하면 안 된다.

`fixed_stratum_reference.proportional_guide_counts()`는 N=3,720m만 지원한다.
29,760에는 재사용 가능하지만19,840에는 최소 tile1240을 지원하는 새 helper가 필요하다.
기존 frozen module은 수정하지 않는다.

### 5.2 학습 목적과 likelihood

D_G=G/p, D_w=q_w/p다. IID의 관측별 mass a_i=1/N이고,
층화 stratum j의 관측별 mass a_ji=π_j/n_j다.

\[
L(w)=-\frac{\sum a_i(g_i/D_{G,i})\log D_w(Z_i)}{\sum a_i g_i/D_{G,i}},\qquad
\widehat J(w)=\sum a_i\frac{g_i^2}{D_{G,i}D_w(Z_i)}.
\]

L의 정규화는 fitting에만 사용한다. J와 최종 사건확률에는 estimated normalizer를 넣지 않는다.
전체 normalized mixture density를 쓰며 component 단독 density로 나누지 않는다.

**이번 정확 비례 설계에서는 a_ji=1/N이므로 기존 equal-row FW loss 재사용이 수학적으로 가능하다.**
nonproportional counts는 이번 생산에서 금지한다. 일반화한다면 log a_i를 gradient weight에
포함하고 별도 테스트해야 한다. “층화니까 기존 IID loss/SE를 모두 그대로 쓰면 된다”는 주장은 틀리다.

q_w domain은 w0=.1, wj=.1/372+.8vj, v simplex다. selection 후 qα≥.5G이며
모든 q,G≥.1p다. 0≤g≤1이면 final X≤10, heldout H_q≤100이지만 상대 정밀도는 보장하지 않는다.

### 5.3 층화 평균·분산·paired covariance

같은 새 경로에서 모든 q의 H_q=g²/(D_GD_q)를 계산한다.

\[
\widehat M_q=\sum_j\pi_j\bar H_{q,j},\qquad
\widehat{\operatorname{Var}}(\widehat M_q)=\sum_j\frac{\pi_j^2s_{q,j}^2}{n_j}.
\]

paired Y=H_a−H_b에 대해서도 층별 Y를 먼저 만들고 같은 식으로 분산을 계산한다.
relative reduction r=(M_a−M_b)/M_a의 delta SE는
각 층의 influence `Y−r_hat*H_a`에 대해
`SE(r_hat)=sqrt(Σ_j π_j²*s²_(influence,j)/n_j)/M_a_hat`로 계산한다(M_a_hat>0).
제곱근 전 합을 M_a_hat²로 나눈 값은 variance이지 SE가 아니다.
candidate covariance를 버리고 두 SE를 독립으로 더하지 않는다.

`stratified_summary()`는 positive moments 합산에 재사용하되, signed paired difference와
cross-products는 새 core가 필요하다. 기존 `paired_risk_reduction()`에 층화 rows를 IID로 넣지 않는다.
raw stratum-based SE와8balanced-block SE를 모두 보고한다.

고정 integrand와 정확 비례 allocation에서는:

\[
\operatorname{Var}_{IID}(\hat M)-\operatorname{Var}_{strat}(\hat M)
=\frac1N\operatorname{Var}_{J\sim\pi}(E[H\mid J])\ge0.
\]

이것은 label-count 변동 제거에 대한 기존 분산 분해다. 새 정리라고 주장하지 않는다.
특정 유한 run의 관측 RSE가 항상 작아지거나, 학습된 q까지 좋아진다는 보장도 아니다.

### 5.4 Gaussian oracle와 필수 테스트

새 모델 탐색이 아니라 구현 검증용2D bounded Gaussian bump 문제4개를 고정한다.
p=N(0,I2), g=Σa_l exp(−||z−c_l||²/(2s_l²)), Σa_l≤1을 쓴다.
μ는 기존 `GaussianBumpOracle.log_endpoint(1)`의 analytic 값을 사용한다.

| toy | target | sampling dictionary(자연 성분 외) | 검증 목적 |
|---|---|---|---|
| A | c=(3,0),a=1,s=1 | (±3,0),(0,±3) | 잘 포함된 단일 기여 |
| B | c=(±3,0),a각.5,s각1 | A와 동일 | 두 기여를 동시에 보존 |
| C | c=(2,0),(0,5),a=.999/.001,s각.75 | A와 동일 | 작은 target 성분의 위험을 진단할 수 있는지 |
| D | c=(0,4),a=1,s=.75 | (±3,0),(±6,0) | 사전에 없는 방향의 한계를 알고리즘 오류와 구분 |

각 toy의 M₂가 무엇을 지배하는지는 수치 oracle로 확인하고 이름만으로 가정하지 않는다.
M₂ mixture integral은 analytic μ와 다르다. adaptive quadrature와 별도 Gauss-Hermite
48/96/192점 수렴을 대조하며 상대 차이≤1e-5를 요구한다. quadrature의 reported error는
엄밀한 interval proof로 부르지 않는다. box truncation을 쓰면 q≥.1p,g≤1로
M₂ omitted tail≤10P_p(outside box)를 따로 상계하고 기록한다.

toy sample 검사는4cases×2bank designs×16independent repetitions×16,000개다.
toy의4shift에서는 .1/.225에 맞춰 natural1,600, 각shift3,600을 쓴다.
이 표본 크기는373성분용 표와 별개다. 따라서 toy sampling2,048,000 logical evaluations다.
추가 quadrature/재현 포함 synthetic logical payoff cap4,194,304. Volterra calls는0이다.
D1 oracle 실행 wall cap은1,800초다. 같은 quadrature node의 g는 캐시할 수 있지만,
여러 q의 density scoring과 optimizer wall을 비용 ledger에서 숨기지 않는다.

필수 테스트:

1. unequal allocation 장난감에서 π_j/n_j 가중 평균·분산·paired covariance의 수기식 일치.
2. 비례 counts에서는 기존 KL/M₂ loss와 generalized weighted loss의 gradient/해 일치.
3. signed difference/negative difference/정확 no-op/constant-per-stratum/zero variance 처리.
4. X²와 X⁴, probability SE와 M₂ SE, stratum SE와 balanced-block SE 단위 분리.
5. log-space 계산과 직접 계산 일치, covariance matrix 대칭/PSD roundoff 검사.
6. 잘못된 label-count, 빠진 stratum, 잘못된 block slicing, count/seed/hash 변조 거부.
7. optimizer FW gap을 finite-bank bound로 검증. 작은 dictionary에서 별도 constrained solver와 대조.
8. known μ와 numerical M₂ 대비 반복적 오차·보고 SE의 calibration을 공개.
   16회 coverage 비율을 정확한95% 보장으로 해석하지 않는다.

통계 core/derivative/density/count의 오류가 남으면 D2 금지다. 수치 oracle가 수렴하지 않는 toy는
임의로 지우지 않고 D1 failure로 기록한다. 실제 Volterra seed나 geometry를 고르는 데 toy를 쓰지 않는다.

### 5.5 실행 전 검정력 점검

실제 성능을 보기 전에 fixed toy와 기존 완료자료만으로 아래 budget이20% 수준의
M₂ 감소를 구분할 수 있는지 대략 점검한다. 유리한 observed reduction을 분모로 넣어
sample size를 정하지 않는다. 기존 불안정한 variance로 계산한 power는 참고치다.
검정력 표는 참고자료이며 재량적인 통계 gate로 쓰지 않는다. 실행 gate는 D1의 구현·수학·자원
검증 및 이후의 명시된 D2 기준이다. 검정력이 낮게 예측돼도 예산을 자동 증설하지 않는다.

## 6. D2 — 새 학습 전에 frozen-family 평가가 가능한지 검사

### 6.1 실험 설계

- 두 셀, IID G와 정확 비례층화 G, 각8fresh blocks×29,760 = 각238,080개.
- 총2×2×8×29,760=**952,320calls**, 32sampling units.
- 각 셀의 q family는 uniform + 직전 event5개 + risk5개, 총11ID를 모두 유지한다.
  α0로 density가 같은 ID도 보존하되 계산 중복은 digest로 공유할 수 있다.
- 모든 경로에서 μ와11개 M₂를 계산한다. q당 density 비용은 별도 기록하며 payoff calls와 혼동하지 않는다.
- 두 sampling designs의 seed는 독립. cross-design covariance는0이다.
  같은 design 안에서는 모든 q가 같은 경로를 쓰므로 paired covariance를 보존한다.
- 기존 평균/단일 RSE를 좋은 reference로 가정하지 않는다. 두 design은 같은 G이므로
  서로 다른 물리 메커니즘의 reference 검증으로 세지 않는다.

### 6.2 D2→D3 개발 평가가능성 gate

전체32units를 완료한 뒤에만 판단한다. 아래는 **새 판별 연구의 development gate**이지
기존1.5% reference gate나 P1/P2 판정을 완화하는 기준이 아니다.

두 셀의 **층화 design**에서 모든11M₂와 μ에 대해 다음을 요구한다.

- μ RSE≤5%, M₂ RSE≤20%.
- balanced-block RSE≤2×해당 raw IID 또는 stratum-based RSE.
- maximum LOO relative shift≤15%, 관측 최대 단일 가중 contribution share≤10%.
- 첫4/뒤4balanced blocks의 평균 차이도 공개한다. 이 수치로 후반 표본을 증설하지 않는다.
- IID design의 같은 지표도 모두 공개한다. IID의 gate 미충족만으로 층화의 구제 가능성을 버리지는 않는다.
- 24estimands(2cells×[μ+11M₂])의 IID/strata independent mean 대조에는
  symmetric relative difference `d=2*(M_IID−M_strat)/(M_IID+M_strat)`를 쓴다.
  독립delta SE는 `4*M_IID*M_strat/(M_IID+M_strat)^2 * sqrt(RSE_IID^2+RSE_strat^2)`다.
  family α=.05,24intervals, `z=Φ^−1(1−.05/(2*24))`를 쓴다.
  adjusted two-sided normal interval이 d∈[−.20,.20]과 완전히 분리되면 contradiction으로 실패.
  **분리되지 않음은 equivalence나 정확도 인증이 아니다.**

관측 집중 기준은 tail 누락을 배제하지 못한다. 층화 design이 gate를 충족하지 못하거나
위 contradiction이 발생하면 **D3를 실행하지 않고 `STOP_UNRESOLVED_BUDGET`으로 종료**한다.
층화는 통과하고 IID는 실패하면 `stratified_only_feasibility`로 명시하고 조건부 D3를 허용한다.
이때도 IID를 독립 자격 reference로 세지 않는다. “geometry가 나쁘다”는 판단이 아니라 현재
비교의 신뢰도에 관한 투자 판단이다. q ID·seed·목표·기준을 바꾸지 않는다.

예산 한계도 명시한다. 직전 IID M₂ RSE 약43%/131,072를 단순 N^−1/2로 외삽하면
238,080에서 약32%여서20% gate는 쉽게 통과할 기준이 아니다. 외삽은 tail을 놓친 pilot에
민감하고 보장도 아니다. 층화가 충분히 개선하지 못하면 여기서 멈출 수 있다.

### 6.3 기여 분해

각 target에 대해 전체 관측 변동을 between-label 및 within-component로 분해한다.
각 component의 mean/variance/기여와 물리 family/window/진폭별 합을 기록한다.
component concentration을 실제 사건 mode의 발견률로 부르지 않는다.
within-component 집중이 남으면 이 연구가 제거한 것은 label noise뿐이라고 결론 내린다.
새 clustering이나 geometry 선택을 이번 자료에서 수행해 같은 자료로 확인하지 않는다.

## 7. D3 — 통과한 경우에만 같은 geometry에서 새 bank 학습

### 7.1 세 모델과 독립 반복

2cells×5independent repetitions, 총10개 task-repetition을 모두 계획한다.

- `U`: 학습 없는 fixed uniform G.
- `E-IID`: 새 IID G train19,840개로 event-KL fit.
- `E-STRAT`: 새 비례층화 G train19,840개로 event-KL fit.

각 반복에서 두 fit은 같은 geometry/domain/초기점/optimizer limits를 쓴다.
train draws는 서로 독립이며, sampler design 외에는 바꾸지 않는다.
두 fit에 **공통인 새 층화 selection G19,840개**로 α를 각각 선택한다.
selection을 공통으로 고정해 train-bank 변화와 selection 설계 변경을 혼동하지 않는다.
α0은 정확한 U이며 no-op의 별도 final 우연은 개선으로 세지 않는다.

train2개+common selection1개이므로 실제 work는10×3×19,840=**595,200calls**.
두 workflow 각각에는 공통 selection 생성비를 전액 부과한다. 실제 연구 work에서는 한 번만 센다.

### 7.2 sampling pilot과 단 한 번의 allocation

proposal과 α를 먼저 동결한 뒤 다음 fresh pilots를 실행한다.

- probability: 각30개 estimator에 IID q16,384개, 총**491,520calls**.
- risk: 각10반복에 IID G와층화 G 각각19,840개, 세 q를 모두 같은 경로에서 평가.
  총**396,800calls**. 각bank는8개balanced/동일크기 blocks로 기록한다.
- pilots는 allocation용이며 confirmation 평균/SE에 합치지 않는다.
- 10반복의 pilots를 모두 완료한 다음 allocation manifest를 한 번 생성·봉인한다.
  유리한 fit만 추가 학습하거나 prefix 성공으로 다음 단계에 진입하지 않는다.

allocation은 추정 RSE r에 대해

\[
N_{req}=\operatorname{round\_up}_{unit}\left(2N_{pilot}\frac{r_{safe}^2}{\epsilon^2}\right),
\quad r_{safe}=\max(r_{raw/stratum},r_{8block}).
\]

probability: ε=.05, unit16,384, 최소65,536, 최대262,144.
risk: ε=.20, unit9,920(8balanced tiles), 최소39,680, 최대158,720.
실제 배정은 `N=max(N_min,N_req)`이고 N>N_max이면 미배정이다.
N_max로 잘라 배정한 뒤 precision 목표를 충족할 예측인 것처럼 쓰지 않는다.
각 risk repetition의 두 heldout designs에는 **같은 N**을 배정한다. N은 층화 pilot의
세 q 중 최악으로 정한다. IID pilot의 더 큰 예측치는 별도 공개하되 그것만으로 층화 confirmation까지
cap 초과로 만들지는 않는다. 층화를 primary risk evaluator로 사전 지정한 설계이며,
IID precision을 보장하는 allocation은 아니다.

이 식은 pilot 예측이지 precision 보장이 아니다. zero/비유한/미정 variance를
유리한 N으로 바꾸지 않는다. cap 초과는 `unallocated_precision_budget`으로 기록한다.
해당 비교는 실패/미해결로 세며 seed나 fit을 교체하지 않는다. 예산 초과 rows도
전체10반복 보고서와4/5 판정의 분모에 남긴다.

미배정의 전파 규칙도 고정한다. risk allocation은 세 q의 최악값으로 정하므로 하나가
cap을 초과하면 그 반복의 **공유 risk-bank group 전체**를 미배정 처리한다.
가장 어려운 q를 빼고 N을 다시 계산하지 않는다. probability allocation 실패는 그 q가
필요한 모든 비교를 평가불가로 만든다. U가 미배정이면 U 대비 두 비교 모두 평가불가다.
다른 q의 예정된 자료를 추가 학습·seed 교체에 전용하지 않는다.

confirmation 전에 전체 allocation/status를 봉인하므로, 계획상 미배정 row는 명시적인
conditional skip이며 완료 prefix를 사후 골라내는 행위가 아니다.

### 7.3 독립 confirmation

- probability는 각 selected q에서 새 IID draws. train design이 층화여도 final은 IID로 유지한다.
- 최대30×262,144=**7,864,320calls**. 각 estimator는8동일크기 independent blocks다.
- risk는 두 heldout designs의 새 G draws. 모든 q를 양쪽에서 공통 경로로 평가한다.
- 최대10×2×158,720=**3,174,400calls**. 각bank는8balanced/동일크기 blocks다.
- N을 채운 뒤 precision이 나빠도 새 block을 붙이지 않는다. pilots나 과거 결과와 합치지 않는다.
- probability final에서 X 및 X²의 수기 SE, X²·X⁴ sufficient moments,
  risk bank에서 H 및 모든 후보 cross-products를 저장한다.
  이후 raw M₂ SE/paired covariance를 복원할 수 있게 하되 raw path 전체 저장은 요구하지 않는다.

### 7.4 비교 family와 반복 성공 기준

세 쌍 U→E-IID, U→E-STRAT, E-IID→E-STRAT ×10반복=30pairs를 사전 고정한다.
화살표 왼쪽은 baseline, 오른쪽은 candidate다. Y=H_baseline−H_candidate가 양수면 candidate에 유리하다.
각 pair의2heldout designs×2SE단위(raw 또는stratum/8block)×2endpoint(absolute/relative),
총**240intervals**를 family로 하여 Bonferroni normal/delta 진단을 적용한다.
family α=.05이며 `z=Φ^−1(1−.05/(2*240))`를 사용한다.
부호·방향은 config에 고정하고 반대 방향 결과도 그대로 보고한다.
모든 interval은 **점근적 개발용 normal/delta 진단**이며, 유한표본 coverage 보장이 아니다.
positive evidence와 아래 negative-effect veto에는 같은 조정된 two-sided intervals를 쓴다.
별도의 무보정 one-sided test를 추가하지 않는다.

층화 primary evaluator에 의한 후보의 반복 성공 조건:

1. no-op이 아니며 배정·단위·감사가 완료됨.
2. U 대비 M₂ reduction 점추정≥20%이고 adjusted absolute/relative lower가 모두0보다 큼.
   층화 heldout의 stratum-based 및8balanced-block SE 단위 모두 충족해야 함.
3. 층화 heldout에서 해당 비교의 각 M₂가 개발 RSE≤20%, block/LOO/단일 기여 gate를 충족.
   IID heldout의 raw와8block **어느 한 SE 단위에서라도**, absolute/relative upper가
   둘 다0미만이면 유의한 반대 효과로 보아 실패 처리한다.
   IID의 미정밀/비유의함을 일치나 독립 재현으로 바꾸어 표현하지 않는다.
4. 후보와U의 fresh probability final이 개발 RSE≤5%, blockRSE≤2rawRSE를 충족.
5. 아래의 비용 비교도 유리해야 실용 후보로 분류. probability precision 통과를 true accuracy로 부르지 않음.

양 셀에서 각각5회 중4회 이상이1–4를 충족해야 개발 후보가 된다.
IID에서도 해당 precision·감소 판정이 통과한 반복 수를 따로 기록해
`single_design_screening`과 `dual_design_development`를 구분한다.
dual-design 표시는 같은 반복에서 두 designs 모두 통과한 교집합이 양 셀 각각4/5 이상일 때만 쓴다.
single-design 후보는 후속 확인의 가치만 뜻하며 robust risk 우위나 독립 reference 인증이 아니다.
threshold 초과 seed/fit은 분모에서 빼지 않는다. 하나의 좋은 셀/반복만으로 채택하지 않는다.
E-STRAT와E-IID 대조는 bank 설계 효과를 별도로 해석한다. E-STRAT의U 우위만으로
층화 train 자체의 인과 효과를 입증했다고 쓰지 않는다.

bank 설계 기여의 반복 기준은 baseline=E-IID, candidate=E-STRAT로 고정한다.
위1–4에서 U를 E-IID로 바꾼 동일 규칙(점감소≥20%, 조정 lower 양수, primary precision,
양 모델 final precision)을 양 셀 각각4/5 이상 만족해야 한다. 두 selected proposal의
전체 means/weights digest가 같으면 pairwise no-op이며 성공으로 세지 않는다.
이 기준을 못 넘으면 학습 후보가 U를 이겨도 bank 설계 기여는 미해결로 남긴다.

FW가 엄격한 tolerance에 못 미치면 그 사실과 finite-bank gap bound를 함께 공개한다.
결론은 고정 예산 학습 알고리즘에 관한 것이며, 그 dictionary의 population 최적해에 관한 것이 아니다.

## 8. 비용: 실제 측정과 예측을 분리

각 단독 workflow의 비용 ledger는 setup, geometry/cache, train bank+component matrix,
fit, selection bank/scoring, allocation pilots, final inference, I/O를 포함한다.
U에는 train/fit/selection 비용을 부과하지 않는다. risk audit/reference 연구비는 실제
연구 ledger에 포함하되 배포 inference 비용과는 따로 보여준다.
같은 dictionary여도 density/sampler wall이 같다고 가정하지 않는다.

새 final의 배정 count가 달라도 같은5% 개발 precision 자격을 얻은 rows에서만
`offline+fit+selection+probability-pilot+final` 실제 wall을 비교한다.
이는 **독립 참값으로 accuracy가 인증된 fixed-precision benchmark가 아니라,
동일 개발 precision 목표를 달성한 실측 workflow 비교**라고 표시한다.
precision 자격이 다른 rows에서 속도 배수를 발표하지 않는다.

실용 개발 후보는 위 성능 반복 성공과 함께 각 셀에서 **같은 반복 ID의 U와 후보**를 비교한
workflow wall 비 `T_candidate/T_U≤.8`까지5회 중4회 이상 충족해야 한다.
matched rows는 사전 성능·precision 기준으로 정하고 유리한 시간만 따로 고르지 않는다.
비용 분포/범위와 실패비용을 함께 공개한다.
비용이 상쇄되지 않으면 “통계적 후보지만 실무 효용 미입증”으로 분류한다.
단일 노트북 wall 중앙값을 보편적인20% population speedup 증명으로 쓰지 않는다.

보조 break-even 예측은
`Tq(ε)≈Bq+cq*ceil[(M₂(q)/μ²−1)/ε²]`로 계산할 수 있다.
μ/M₂ 불확실성을 전파하고, μ의 하한이0이거나 위험이불안정하면 예측 불가로 둔다.
이 예측으로 confirmation count를 다시 바꾸지 않는다. 반복 사용10/100회 amortization은
같은 q/같은 task에서만 보조표로 제시하고 one-shot 성공 판정을 대신하지 않는다.

## 9. 최종 계속·전환·중단 표

| 결과 | 결정 | 다음에 허용되는 일 |
|---|---|---|
| D2 primary 평가가능성 실패/contradiction, 또는 D3에서 양 셀4/5의 primary 평가가능한 비교 자체가 확보되지 않음 | `STOP_UNRESOLVED_BUDGET` | 실패 보고·보존. 현재 weight-only 확장 중지. 추가 예산은 별도 결정 |
| 위험 평가는 안정됐지만 두 학습법 모두4/5 반복 기준 미달 | `PIVOT_WEIGHT_ONLY` | U를 기준선으로 유지. geometry 연구는 별도 새 가설/계획으로만 |
| 학습이 위험·precision에서 통과하지만 비용 이득 없음 | `PIVOT_PRACTICAL_CLAIM` | 실제 사용 목적 재검토. 실무용 성공·대규모 학습 주장 금지 |
| E-STRAT가 반복 성능·비용 기준 통과, E-IID와의 차이도 반복됨 | `CONTINUE_CANDIDATE` | bank 설계 후보의 별도 fresh confirmation/격자 이전 계획 |
| 학습 후보는 통과하지만 IID/strata 학습 차이는 미해결 | `CONTINUE_MODEL_ONLY` | learning 후보만 보존. bank 메커니즘 기여/새 원리 주장 금지 |
| 데이터·density·SE·source binding 오류 | `INVALID_PROTOCOL` | 오염 결과로 성능 결론 금지. 원본 보존 후 수정 재실행 여부를 별도 결정 |

판정 우선순위는 INVALID → D2 gate → D3 평가가능성(precision/배정/감사) → 반복 성능 → 반복 비용이다.
D3에서1회 미배정이 있다고 전체 즉시 실패하지는 않지만 그 row는 성공으로 세지 않고4/5 요건의
실패 분모에 남긴다. 평가가능한 rows 부족과 평가가능했으나 우위를 못 보인 것을 각각 STOP과 PIVOT으로 분리한다.
single/dual-design evidence 등급은 위 계속 결정에 반드시 덧붙이며, 등급이 낮은 결과를
독립 확인 성공으로 바꾸어 표현하지 않는다.

negative 결과가 나온다고 즉시 다른 모델10개를 시도하지 않는다. 이번 사이클은 여기서 종료한다.
위 표는 연구 투자 결정이며 실제 효과0/불가능성 정리가 아니다.

## 10. geometry를 다음으로 연구할 때의 별도 조건

이번 데이터만으로 “큰 모델/새 operator가 필요하다”는 결론을 내리지 않는다.
geometry 연구가 다음 투자 후보가 되려면 다음 중 하나가 필요하다.

- 실제 위험을 지배하는 **반복 가능한 잔차 특징**이 독립 표본에서 검출되고,
  현재 single-template dictionary가 그 특징을 왜 표현하기 어려운지 설명 가능.
- 알려진 문제에서의 geometry lower bound/counterexample와 실제 Volterra 문제의 연결.
- data-driven target mass를 사용한 bound라면 그 mass 불확실성과 coverage를 포함.
  finite-bank optimum을 population optimum으로 대체하지 않음.

조건을 만족하더라도 한 개의 사전 지정한 multi-window/잔차 변경만 별도 연구로 제안한다.
새 geometry를 고른 자료와 confirmation 자료는 분리한다. 이 계획에는 그 구현·실험비가 없다.

## 11. 구현 파일과 재사용 경계

새 파일 제안(아직 구현되지 않음):

| 파일 | 역할 |
|---|---|
| `src/path_integral/balanced_mixture_bank.py` | 최소tile/count/label/seed/chunk 계약 |
| `src/path_integral/stratified_paired_risk.py` | signed/positive sufficient moments, stratum covariance, paired delta |
| `src/path_integral/direct_second_moment_diagnostics.py` | 기존 final X² block 분석 |
| `src/path_integral/bounded_decision_protocol.py` | counts/conditional allocation/gates/비용 status |
| `experiments/post_audit_bounded_direct_m2.py` | D0 zero-physics runner |
| `experiments/post_audit_bounded_bank_oracles.py` | D1 toy/통계 검증 |
| `experiments/post_audit_bounded_frozen_bank_contrast.py` | D2 frozen-family contrast |
| `experiments/post_audit_bounded_bank_learning.py` | D3 conditional fresh learning/confirmation |
| 대응하는 `tests/test_*.py` | 수기식·oracle·오류주입·replay 회귀 |
| `configs/post_audit/bounded_bank_decision_v1.yaml` | 본 계획 수치·roles·counts·caps·decision 고정 |

기존 cached Volterra evaluator/PhysicalWindowGuide/mixture density/대상함수/FW core는
재사용하되 frozen files를 수정하지 않는다. 기존 canonical stratification pilot은
다른 guide/기존 q family에 hard-code되어 있어 새 실험으로 그대로 실행하지 않는다.

기존 dictionary `bank()`와 `stream_digest()`의 CHUNK2048 배수 강제를 그대로 쓰지 않는다.
새 bank는 명시적 마지막 chunk와 절대 sample-offset seed ledger를 지원한다.
block/count가 달라져도 seed가 충돌하지 않게 phase/cell/rep/design/role/block/component를 포함한다.
seed목록의 전체 uniqueness를 sampling 전에 검사한다. IID path/label generator를 분리한다.
층화는 component label이 stochastic인 척 label RNG를 가짜로 기록하지 않는다.

## 12. 예산 산술과 실패 처리

| 용도 | 산술 | 최대 calls |
|---|---|---:|
| preflight | 2cells×2designs×1,240 | 4,960 |
| D2 | 2cells×2designs×8×29,760 | 952,320 |
| D3 train/selection | 10×3×19,840 | 595,200 |
| D3 probability pilots | 10×3×16,384 | 491,520 |
| D3 risk pilots | 10×2×19,840 | 396,800 |
| D3 probability confirmation | 10×3×262,144 | 7,864,320 |
| D3 risk confirmation | 10×2×158,720 | 3,174,400 |
| **생산 합계** |  | **13,479,520** |
| 별도 representative replay cap | 미실행 phase에는 사용하지 않음 | **1,000,000** |

대표 replay는 D2의각cell/design1block, D3rep0두셀의train+selection/pilots,
각모델 probability confirmation1block, 각heldout design1block을 미리 지정한다.
preflight4,960calls까지 전부 재현하면 모두 최대 count일 때 예상696,672calls로 cap 안이다
(preflight 제외 열거분691,712). 탈락 unit은 성공 seed로 대체하지 않는다.
실제 replay 범위와 calls를 receipt에 기록한다.

저장/통계 계산이 추가됐다고 payoff count를 늘려 세지 않고, q가11개라고 g를11번 평가한
것처럼 세지도 않는다. 반대로 setup/preflight/재생성/실패 중 이미 실행한 g는 숨기지 않는다.
quadrature/synthetic evaluations, regression tests, import/archive, 분석 wall도 별도
연구 ledger에 기록한다. 위 hard cap은 신규 Volterra 생산+감사 범위이며 모든 회귀테스트의
내부 physics 호출까지 계측한 전역 합계라는 뜻은 아니다.

추가 모델·sampling 디자인·seed·표본·격자 증설은 이번 예산에 포함되지 않는다.
OS/배터리 정책 변경이나 background cloud 실행 없음.

## 13. 재현성과 리뷰 체크리스트

실행 전:

- [ ] D0/D1 구현 후 수학·기술 검토. sampler와 denominator가 같은 normalized law인지 확인.
- [ ] counts/allocation/cost 산술을 unit test로 검증. partial chunk와balanced-block 구성 검사.
- [ ] 원본 anchor/source archive/hash/코드/계획/config/test를 봉인.
- [ ] **D2의 첫 표본 전에 D3 runner까지 구현·테스트하고 동일 source snapshot에 봉인.**
  D2 결과를 본 뒤 D3 정책/표본수/학습목적/판정을 수정하지 않음. 기술 오류 수정이 필요하면
  기존 실행을 INVALID로 보존하고 별도 protocol 결정 없이 같은 연구를 재실행하지 않음.
- [ ] 새 namespace `bounded-bank-decision-v1` 아래 phase별train/selection/pilot/confirmation 분리.
- [ ] existing nonempty result directory는 덮어쓰기/암묵적 resume하지 않음.
- [ ] preflight는 precision 선택에 쓰지 않고 resource forecast에만 사용.

실행 중/후:

- [ ] append-only journal에 reservation/start/finish/failure/실제 work와wall 기록.
- [ ] D2 전체 family 이후 gate 판정. D3 전체 pilots 이후 allocation manifest를 한 번 봉인.
- [ ] 사전 규칙의 unallocated row는 실패 분모에 유지. partial family를 성공으로 보고하지 않음.
- [ ] 모든 unit의count/normalized weights/strata counts/α binding/role/seed/digests 검증.
- [ ] candidate cross-products와se 단위를 모든반복에서재집계; 대표 replay와전수 metadata 검사를구분.
- [ ] optimizer cap/실패/NaN/자원 초과/신뢰할 수 없는 tail을 숨기지 않음.
- [ ] final I/O 이후에도 실제 phase wall을 다시 확인. 초과시 violation marker와자격실패.
- [ ] CI 범위 Ruff/Mypy, 전체pytest/coverage, 수기식·oracle·replay 검증.
- [ ] 검토된 결과만README/보고서에 반영. 코드검사 성공과과학적성능 성공을분리.

현재 계획은 one-shot/fail-closed다. 중단 unit의 같은 seed를 조용히 재실행하지 않는다.
복구가 필요하면 완료 unit 보존/중단 work 처리/새 family 범위를 사전 고정한 별도 protocol이
필요하며 자동재시도 예산은 없다. 중단 자체는 모델의 수학적 실패와 별도로 분류한다.
커밋·푸시는 이 계획 작성의 범위가 아니다.

## 14. 논문 가치 평가: 성능 성공 뒤에도 남는 문턱

mixed proposal의 weight 최적화/convexity나 비례층화의 분산 감소는 기존 원리다.
단순히 이번 실험이 통과했다는 이유로 새로운 수학적 모델이라고 쓰지 않는다.

- [He–Owen, Optimal mixture weights in multiple importance sampling](https://arxiv.org/abs/1411.3954):
  mixture allocation과convex 최적화의 선행연구 경계.
- [Owen–Zhou, Safe and effective importance sampling](https://artowen.su.domains/reports/seis.pdf):
  defensive/multiple-mixture 접근과 안전성 논의의 경계.
- [Agapiou et al., Importance Sampling: Intrinsic Dimension and Computational Cost](https://arxiv.org/abs/1511.06196):
  target/proposal 차이, 표본 복잡도와 효용 평가의 맥락.
- [Hartmann–Richter, Nonasymptotic Bounds for Suboptimal Importance Sampling](https://epubs.siam.org/doi/abs/10.1137/21M1427760):
  제안분포 근사 오차와path-sampling 위험의 선행연구 경계.

이번 사이클 종료시 novelty 표에는 각 주장마다 기존 원리/새 구현/새 실험 사실/
검증할 새 명제를 구분한다. 실패 진단을 논문으로 전환하려면 “중요도 샘플링이
불안정할 수 있다”를 반복하는 것이 아니라, Volterra 구조에 특화된 새로운 조건,
반례/한계 또는 그것을 해결하는 재현 가능한 방법이 필요하다.

top journal을 목표로 계속하려면 후속으로만 다음을 요구한다.

1. Volterra 구조와 연결되는 비자명한 기여·정확한 적용 조건·선행연구 대비 차별화.
2. 새 task/새 seed/더 미세한 격자에서의독립확인. N32 결과를continuum으로확대하지않음.
3. 실제사용 문제·계약범위와효과크기, 동등accuracy에서training-inclusive 총비용.
4. 기존CE/SMC/물리정적·해당분야경쟁방법과의공정한benchmark.

이 후속 전체 연구는 이번 cap에 포함되지 않는다. 현재 증거만으로 게재 가능성이
높다고 평가하거나, 음성 결과만으로논문기여가자동완성됐다고말하지않는다.

## 15. 최종 납품물과 착수 순서

착수는 D0 → D1 → gate → D2 → gate → 조건부D3 → D4 순서다.
어느 gate든 실패하면 **검토·판정 보고서를 작성하고 이 사이클을 종료**한다.

최종 납품물:

1. source/config/hash에 연결된 결과·allocation·status·cost ledger.
2. direct X² 민감도, IID/층화 covariance, 학습 bank 집중도 및 독립 final 결과.
3. 가설별 채택/기각/미해결 표와원인설명. geometry 탓으로단정하지않음.
4. 계속/전환/중단 결정, 남은 scientific/technical/실무 한계와논문주장 가능범위.
5. 전체 회귀 및대표수치재현 감사. 모든전역자격false유지.

**이 계획의 성공 기준은 “모델이 반드시 이기는 것”이 아니라,
다음 투자를 정당화하거나 중단할 수 있는, 사전 기준에 따른 결정에 도달하는 것이다.**

## 16. 계획 검토 기록과 미실행 상태

계획 작성 중 통계 설계와 구현 경계를 독립적으로 재검토했고 다음을 반영했다.

- 373성분의 최소 비례 tile은1,240임을 확인. 기존3,720전용 helper와2,048배수 bank를
  무조건 재사용하지 않도록 명시했다.
- 층화 relative delta의 variance/SE를 구분하고 제곱근 식을 수정했다.
- equal-row 학습 재사용은 정확 비례 allocation에만 한정했다.
- IID만 실패했을 때 층화 구제 가능성을 판별할 수 있도록 primary/secondary design을
  사전 지정하고, 독립 확인 성공과 single-design 신호를 구분했다.
- 후보 대비 기준선, head-to-head, no-op, 4/5, 배정 불가 전파, 반대 효과 veto와
  비용 기준의 판정 순서를 명시했다.
- 생산13,479,520, 대표 replay696,672, 감사cap포함14,479,520calls를 독립 산술로 확인했다.
- D2 전에 D3까지 source/policy를 동결해 결과에 맞춰 후속 실험을 바꾸지 않도록 했다.

이는 계획의 확인 가능한 정합성 검토이며 모든 경우의 무오류 증명은 아니다.
새 module/runner/config/test의 구현과 oracle 실행은 아직 하지 않았다.
이번 작성에서는 이 MD 파일만 새로 만들었고, 기존 미커밋 작업을 보존했다.
커밋·푸시·새 학습·Volterra payoff sampling은 수행하지 않았다.
