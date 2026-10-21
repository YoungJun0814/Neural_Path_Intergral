# 원래 모델 개발 재시작: 구현·전체 실행·실패 원인 분리

2026-10-20. 결론: **진행방식 개선과 실제 q 보정 구현은 완료했다. 이번 K2 보정의
안정적 성능 개선은 입증하지 못했으므로 production winner로 채택하지 않는다.**

## 1. 이번에 실제로 바뀐 것

기존에는 N4096 독립 SMC reference를 완전히 자격화하기 전까지 원래 q 개발도
대기했다. 이제 다음 두 트랙을 분리한다.

- 모델 개발: N32에서 새 전체 부모 학습·실제 사건 proposal 보정·독립 final/risk 대조.
- 제출 검증: fine reference/all-q M2/mesh/continuum/confirmation gate는 그대로 미완료.

이는 실패 기준 완화가 아니다. 새로운 개발 precision 5%는 기존 1.5% reference
gate를 대체하지 않는다. 과거 실패·결과·봉인 runtime을 수정하지 않았다.
기존 q를 고정해 보조 r만 고친 최근 단계와 달리, 이번에는 새 q0와 실제
q_event/q_risk를 만들어 사건확률을 추정했다. 대규모 신경망은 추가하지 않았다.

## 2. 구현과 수학적 검토

- [사전 고정 실행계획](../plans/SMALL_GRID_MODEL_RESTART_2026-10-20_KO.md)
- [수학적 계약과 exact-risk 분석](../theory/SMALL_GRID_MODEL_RESTART_CONTRACT_2026-10-20_KO.md)
- core: `src/path_integral/risk_aware_model_correction.py`.
- paired diagnostic: `src/path_integral/paired_model_risk.py`.
- runner: `experiments/post_audit_small_grid_model_restart.py`.
- zero-physics 실패 분석: `experiments/post_audit_model_restart_diagnosis.py`.

새 SMC q0는 기존과 같은 3islands×256particles, 49points, β=(k/48)^4,
2mutations, scale .35, resample4, stratified, K≤2 shift/identity covariance다.
과거 좋은 부모를 골라 재사용하지 않고 두 셀에서 각각 5회 처음부터 학습했다.

독립 physical mixed-window G에서 8192training draws를 만들고,
event weight g/(G/p), risk weight g²/[(q0/p)(G/p)]로 각각 K≤2 교정 분포를
동일 규칙으로 적합했다. 독립 selection 16384개에서 α∈{0,.25,.5}를 선택했다.
최종 qα=(1−α)q0+αr는 일반적으로 6성분이며 natural을 합치면 5성분이다.
fitting family K2와 배포 mixture K2를 혼동하지 않는다. static G는 373성분이므로
동일 capacity 대조가 아니라 practical baseline이다.

모든 최종 추정은 full-mixture likelihood를 사용한 ordinary IS다.
학습 가중치 정규화는 fitting에만 사용하며 final에는 self-normalization이나
clipping을 하지 않았다. 모든 실제 q,G≥.1p로 support를 유지했다.

exact r0∝g²p²/q0에 대해서는 J(r0)≤J(q0),convex blend의 J비증가가 성립하는
조건을 따로 검토했다. 그러나 **finite bank/K2 Gaussian 적합에는 이 보장이
상속되지 않는다.** risk-target fitting을 직접 population M2 최적화라고 부르지 않는다.
이 항등식의 독창성이나 새 정리 발견도 주장하지 않는다.

## 3. 전체 고정 실험

각 arm/부모의 final은 새 IID 8blocks×32768=262144개다. 네 arm은 parent,
static, event, risk다. 별도 새 G 131072개에서 네 q의 M2를 공통 경로로 평가했다.
선택·최종 사건 표본·heldout 위험 표본의 seed와 역할을 모두 분리했다.

- 10whole-parent fits,40final estimators,320final blocks,10heldout banks.
- **343journal units,12,772,352payoff evaluations**(preflight512포함).
- phase **244.039초**. phase 준비/I/O를 포함하고 import/source archive·사후
  replay·회귀테스트는 제외한다. 통제된 논문용 시간 benchmark가 아니다.
- torchCPUfloat64,1thread. 중간 결과에 따른 count/seed/config 변경 없음.
- static5회는 fixed geometry의 fresh sampling 반복이지5회 학습이 아니다.

## 4. 직접 사건확률 추정 결과

아래는5부모의 관측 RSE 중앙값과 범위다. rawRSE≤5% 및 blockRSE≤2rawRSE를
새 개발 precision 표시로 사용했다. reference accuracy/equivalence 인증은 아니다.

| 셀 | arm | RSE중앙값 | RSE범위 | 개발 precision통과 |
|---|---|---:|---:|---:|
| canonical | parent | 6.51% | 4.89–12.48% | 1/5 |
| canonical | static | 4.41% | 3.37–21.07% | 3/5 |
| canonical | event | 6.15% | 3.77–11.38% | 2/5 |
| canonical | risk | 11.02% | 4.28–51.31% | 1/5 |
| high-η | parent | 10.81% | 5.37–72.28% | 0/5 |
| high-η | static | 9.59% | 5.32–25.05% | 0/5 |
| high-η | event | 9.30% | 5.29–39.72% | 0/5 |
| high-η | risk | 9.21% | 6.15–28.30% | 0/5 |

canonical risk의 최대 RSE 51.31%나 high-η parent의 72.28%를 이상치로 제외하지
않는다. 혼합의 이론적 M2 상한과, 서로 다른 유한 final 표본에서 관측한 RSE 비는
다르다. 이번 큰 변동만으로 이론적 상한의 위반이라고 할 수 없다.

event의 선택 α는 canonical [.25,.5,0,.25,.25], high-η [.25,.25,.25,.25,.5]였다.
risk는 canonical [.25,.5,.5,.25,.25], high-η에서는 모두 .25였다.
canonical/rep2/event는 α0이므로 parent와 정확히 같은 밀도다. 독립 final의 RSE가
조금 달라져도 새로운 모델 개선으로 세지 않는다.

## 5. 독립 위험 비교와 ‘낮아 보이는 분산’의 재검토

같은 새 G 경로에서 H_q=g²/[(G/p)(q/p)]를 계산했다. 모든 60쌍에서 공통 경로의
공분산을 반영하고 Bonferroni-adjusted normal/delta 진단을 사용했다.
raw IID와 8block 모두에서 absolute difference와 relative reduction의 lower가
양수인 경우에만 개발 감소 신호로 표시했다. 이 normal 근사는 유한표본의
tail coverage나 population dominance 증명이 아니다.

- parent 대비 static 감소 신호: canonical 1/5, high-η 4/5.
- parent 대비 event 감소 신호: canonical 1/5, high-η 0/5.
- parent 대비 risk 감소 신호: canonical 1/5, high-η 4/5.
- 감소 신호와 final 개발 precision을 **함께** 통과: canonical 각 1/5,
  high-η 모든 후보 0/5. 사전 기준인 두 셀 각각 4/5에 도달한 후보는 **없다**.

relative ratio만 보면 parent/static이 거의 99% 개선처럼 보이는 경우도 있으나,
분모 위험의 큰 불확실성 때문에 absolute 차이 판정이 실패했다. 유리한 ratio만
채택하지 않는다. 학습 보정이 static보다 작은 M2를 보인 point estimate도
10반복 모두에서 없었다. 그러나 static 자신의 M2 추정도 불안정하므로 이를
확정적인 성능 우월성이나 속도 배수로 바꾸어 쓰지 않는다.

별도 post-hoc 분석에서 canonical parent의 heldout M2 / direct-q 관측 제곱
모멘트 비는 반복별 약 142–553배, 중앙값 501배였다. event는 중앙값 684배,
risk는 125배다. 두 M2 평가가 모두 불확실하므로 **모집단 위험이 정확히
501배라는 뜻이 아니다.** direct 표본만으로 위험을 과소 관측할 수 있다는 경고이며,
편향 입증도 아니다.

heldout 위험 RSE 중앙값은 다음과 같아 이번 bank도 oracle이 아니다.

| 셀 | parent | static | event | risk |
|---|---:|---:|---:|---:|
| canonical | 35.16% | 61.58% | 37.45% | 36.14% |
| high-η | 13.06% | 47.47% | 12.55% | 23.12% |

static의 낮은 위험 점추정이나 risk의 일부 개선 신호를 독립 reference 자격으로
승격하지 않는다. 새로운 low-RSE만 찾는 반복도 하지 않는다.

## 6. 무엇이 병목으로 더 분명해졌는가

### 확인된 사실

8192개 train G 경로의 weight ESS는 event에서 canonical 22.63–51.91,
high-η 28.57–92.03, **risk에서는 5.23–16.39와 3.69–26.69**였다.
high-η의 한 risk fit에서는 한 경로가 bank weight의 51.33%를 차지했다.
SMC 부모 island 최종 계보도 canonical 11–21/256, high-η 19–34/256이었다.
SMC 조상 수와 IID weight ESS는 서로 다른 지표지만 둘 다 자료의 집중을 드러낸다.

### 검증이 필요한 해석

이 집중된 자료를 K2 shift로 압축하면 다양한 Volterra 경로 구조를 보존하기
어렵다는 가설이 지지된다. 그러나 이번 실험만으로 bank 문제와 K2 표현 문제를
완전히 분리하지 못했다. “공분산 rank만 올리면 해결”, “신경망 크기만 부족”,
“새 모드 누락을 증명”이라는 결론은 내리지 않는다. 선택 bank도 위험 기여가
희소할 수 있어 α 선택의 population 최적성은 없다.

## 7. 실제 비용과 다음 개발 결정

단독 workflow에 전액 부과한 setup+부모+bank+fit+selection+fixed-final 시간의 중앙값:

| 셀 | parent | static | event | risk |
|---|---:|---:|---:|---:|
| canonical | 3.015s | 7.836s | 7.031s | 7.043s |
| high-η | 3.711s | 5.437s | 4.407s | 4.684s |

static에 부모 학습비를 부과하지 않았고 learned에 G/train/selection 비용을 누락하지
않았다. heldout 연구 평가는 배포 inference에 필요 없어 workflow 표에서 제외했지만
실제 payoff count에는 포함했다. parent와 static은 복잡도와 accuracy가 다르므로
이 표로 fixed-precision 속도 우위를 선언할 수 없다.

이번 결과에 따른 채택/중단:

1. **채택:** fine-reference와 독립인 작은 q 개발 트랙, 공통 독립 heldout 위험 대조,
   no-op 식별, 전체 학습·선택·inference 비용 분리. 검증기만 개선하는 직렬 순서는 끝낸다.
2. **채택하지 않음:** 현재 K2 event/risk 보정을 완성 모델·production winner로 등록하지 않는다.
3. **우선순위 하향:** β power/resampling 반복 검색, 큰 신경망·operator 추가, 무조건 표본 증설.
4. 다음 모델 가설은 **학습 자료를 작은 Gaussian 두 개로 압축하지 않고 물리적 다중경로
   dictionary를 보존한 채 weight를 학습하는 것**이다. event-KL 대 risk-M2를 같은
   dictionary·같은 bank·같은 compute에서 분리하고 정적 균등 baseline을 반드시 둔다.
   이는 기존 mixture 기법을 사용하는 후속 가설이지 독창성이 증명된 새 모델이 아니다.
5. 그 실험은 이번 train/selection/final을 재사용하지 않고 새 계획·새 bank·새 final로
   별도 고정해야 한다. 이번 음성 결과를 근거로 실험을 무한 자동 연장하지 않았다.
6. 진전의 최종 기준은 새 학습 반복에서의 위험·정확도·총비용이며 test 수나 문서 수는 아니다.
   학습이 정적 baseline을 이기지 못하면 물리 기반 방법을 주연구로 바꾸는 선택도 유지한다.

현재 scope는 N32 stress-probability 개발이다. N4096 reference와 수치를 직접
비교하거나 일반 pricing/calibration 실용성으로 확대하지 않는다. 모든 전역 자격은 false다.

## 8. 재현·검증 기록

결과: `results/post_audit/small_grid_model_restart_v1/session_20261020/`.
SHA256: `ab3ca01060ba1fdbc4ef5cdaf1b9370dbeda4adaa57cd93dba63689cbf19682f`.

감사 상태: `small_grid_original_model_source_seed_fit_selection_final_risk_replay_pass`.
대표 2부모의 SMC+fit+selection, 8final blocks, 2heldout banks를 재현했다.
추가 payoff 평가는 719360회로 생산 비용과 구분했다. normalizer/계보/파라미터/
bank digest/선택/density/seed/moments/paired 위험 통계가 일치했다.

추가 실패 분석은 original result+audit SHA에 묶은
`results/post_audit/model_restart_diagnosis_v1/session_20261020/result.json`이며
새 physics 평가는 0회다. 이 분석은 post-hoc 진단이지 새 사전 등록 검정이 아니다.

최종 검증:

- CI와 같은 전체 pytest: **1,575 passed, 460.80초**, source line coverage 84%.
- CI 범위 Ruff 통과, Mypy **206 source files** 통과.
- 새 runner/진단 직접 Mypy 2files 통과. 새 core들은 전체 Mypy 범위에도 포함됐다.
- source archive 1080entries를 확인했고, 봉인된 `src/`·`experiments/` runtime의
  실행 후 변경은 0건이다. 결과 SHA와 343개 complete units/실제 work도 재확인했다.
- zero-physics 진단을 original summary에서 다시 계산해 정확히 일치시켰다.
- P1/P2/large-training false, `git diff --check` 통과.
- HEAD `9cafe59b689ce8a015a7e79fea5f8ba2a8dba00c` 유지, staging은 비어 있다.
  커밋·푸시는 하지 않았다.
- 기존 requests 의존성 경고는 남아 있다. 실험 중 의존성 업그레이드나 OS 정책
  변경을 하지 않았다. 이 작업 범위 밖 legacy helper 문제까지 해결했다고 주장하지 않는다.

검사 범위에서 수치·회귀 오류를 발견하지 않았다는 뜻이지, 모든 입력에서의 무오류,
미관측 tail의 부재, 새 모델의 효율 우위 또는 논문 완성을 증명한 것은 아니다.
