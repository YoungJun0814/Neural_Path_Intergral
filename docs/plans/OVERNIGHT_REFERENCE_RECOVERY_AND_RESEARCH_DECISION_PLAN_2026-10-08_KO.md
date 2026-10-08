# 야간 실행계획: reference 신뢰성 회복과 다음 연구 방향 결정

작성일: 2026-10-08. 계획 ID: `overnight-reference-recovery-v1`.

**현재 상태: 계획 문서만 작성했다. 이 문서의 신규 구현·실험·야간 실행·예약·커밋·푸시는 시작하지 않았다.**

후속 실행 기록(2026-10-08): 사용자 실행 요청에 따라 기본696-unit 개발 grid·검증을 수행했다. marginal 성능 재현/독립 reference 관문은 실패했고, N4의 I/O 포함 wall cap 초과를 발견·기록·수정했다. 원자료와 사전 계획은 보존한다. 아래 계획 당시 상태를 실제 완료 선언으로 읽지 말고 [실행 결과 보고서](../reviews/OVERNIGHT_REFERENCE_RECOVERY_RESULT_2026-10-08_KO.md)를 우선한다. 사용자는 최종 commit/push도 별도로 허용했으며, 이는 scientific gate 변경이 아니다.

이 계획은 노트북에서 한 번의 긴 작업 세션을 수행하기 위한 것이다. 정상 작업 목표는 6시간 30분, 오류 복구·검증 여유를 포함한 절대 상한은 7시간 30분이다. 이는 소요시간 예측이나 성공 보장이 아니다. 빨리 끝나면 조기에 종료하고, 끝나지 않으면 완료·실패·미해결을 구분하여 인계한다.

## 1. 이번 밤에 해결하려는 문제

현재의 핵심 문제는 새로운 신경망을 추가하지 못한 것이 아니라, **현재 모델의 성능을 믿을 만큼 독립적인 기준 추정값을 만들지 못한 것**이다.

이번 밤의 목표는 다음 여섯 가지다.

1. 중단해도 과학적 계약과 소비 예산을 잃지 않는 실행기를 만든다.
2. 어려운 경로를 놓치는 현상을 참값이 알려진 작은 문제에서 구분한다.
3. 같은 수학적 법칙을 유지하면서 불필요한 계산을 줄인다.
4. 마지막 pair의 `L=1` marginal reference가 기존 full-reference IS보다 실제로 유용한지 새 표본으로 비교한다.
5. 초기 population·resampling·mutation 중 어느 부분이 희귀 기여를 놓치는지 제한된 대조로 분석한다.
6. 다음 날 무엇을 채택·폐기·보류할지 판단할 수 있는 결과와 오류 검토를 남긴다.

**밤이 끝났다는 이유로 reference 관문 통과, 주모델 성능 개선, 박사급 기여 또는 최상위 저널 제출 준비 완료를 선언하지 않는다.**

## 2. 근거 문서와 우선순위

우선 적용할 근거는 다음과 같다.

- [최신 구현·실험 결과](../reviews/REFERENCE_REDESIGN_IMPLEMENTATION_AND_EXPERIMENT_REPORT_2026-10-08_KO.md).
- [reference 재설계 사전 계획](STRUCTURAL_V2_REFERENCE_METHOD_REDESIGN_2026-10-08_KO.md).
- [reference 구현 계약 감사](../theory/REFERENCE_REDESIGN_IMPLEMENTATION_AUDIT_2026-10-08_KO.md).
- [모델 구조 개선 통합 V2](MODEL_STRUCTURAL_IMPROVEMENT_PLAN_V2_2026-10-08_KO.md).
- [최초 구조 검토·개선계획](../reviews/MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md).

2026-10-07의 야간계획에 있던 reference → mesh → q 보정 순서를 지금 그대로 실행하지 않는다. 후속 실제 결과에서 reference 관문이 실패했으므로, 이번 계획은 그 실패 이후의 진단·복구 분기다. 과거 계획·실패 artifact를 수정하거나 삭제하지 않는다.

현재 알려진 HEAD는 `aa9a21c2d9d5f16f1dc56798d476ee831f8556d0`, 브랜치는 `research/v10r1-full-latent`다. 실행 직전 실제 값을 다시 확인한다. 작업 폴더에는 기존 미커밋 변경이 다수 있으므로 전부 사용자 작업으로 보존한다.

## 3. 현재 상황: 완료와 미완료를 분리

| 항목 | 현재 근거 | 이번 밤의 해석 |
|---|---|---|
| exact marginal·조건부 evaluator·ellipse core·production 경로 | 구현됨 | 새로 처음부터 만들지 않고 계약을 재사용 |
| 현재 소스 회귀 | 최신 보고서 기준 pytest 1,233개, Ruff, mypy 178파일 통과 | 과거 검증 결과이며 이번 변경 후 다시 검사 |
| 마지막 pair 내부 표본 증가 | `L=4/16/64` 모두 비용 개선 관문 실패 | 같은 실험 반복 금지 |
| 초기·중기·후기 고정 pair | 세 후보 모두 비용 개선 실패 | 새 pair 탐색을 무제한 추가하지 않음 |
| guide-free ellipse | 일부 RSE 감소, 비용 증가; 전체 bundle cap 실패 | 빠른 독립 reference가 입증된 것은 아님 |
| fresh reference production | 미실행 | 강제로 실행하지 않음 |
| production allocation | `production_config=null` | 기존 승인 경로 잠금 유지 |
| V2-P2 mesh | `p2_authorized=false` | 이번 밤에는 열지 않음 |
| 주모델 q·새 bank·CV 성능·confirmation | 이번 reference 재설계에서 미수행 | reference 실패를 우회하여 실행하지 않음 |

### 현재 결과에서 중요한 숫자

- canonical 마지막-pair M₂ RSE는 `L=1` 약 20.92%, `L=64` 약 21.57%였다.
- high-η는 약 21.88%와 21.91%였다. 내부 반복을 늘려도 주요 잔여 분산이 줄지 않았다.
- canonical `L=64`의 개발 분해는 inner-mean CV² 약 0.0322, outer CV² 약 95.4740이었다. high-η는 약 0.00527과 98.6516이었다. 작은 pilot의 추정이지 모집단 인증이 아니다.
- ellipse/pCN의 관측 `RSE² × wall` 비율은 완료된 셀에서 약 4.00–9.11이었다. 같은 accuracy 자격이 없어 정식 성능 승패가 아니다.
- canonical risk의 최종 initial ancestor 수는 pCN 4–7, ellipse 3–9였다. ancestry는 독립 표본 수와 같지 않다.
- 34개 production job 중 33개만 forecast한 경우에도 약 **185.54시간·172.07억 FFT+CDF 단위**였다. 실제 실행치도 완전한 전체 forecast도 아니다.
- 기존 production 상한은 **2시간·1.6억 단위**다. 야간 세션이 길다는 이유로 이 상한을 완화하지 않는다.

따라서 “시간이 많으니 기존 production을 끝까지 돌리기”는 이번 계획의 액션이 아니다.

## 4. 작업 범위와 금지 사항

### 수행 범위

- 실행 안정성, source/config/seed 봉인, 동일-law cache, bounded profiling.
- analytic toy oracle, fresh 개발 IS 대조, 제한된 초기 coverage 진단.
- 수학적 분산·비용 관계 정리, 반례 검토, 기존 claim ledger와의 연결.
- unit/integration/artifact audit·전체 회귀·아침 인계 보고.

### 이번 밤에 하지 않는 것

- 기존 실패 v1 파일 덮어쓰기, 실패 run 삭제, 좋은 seed로 교체.
- 실패한 high-η μ ellipse 부분만 채워 기존 kernel v1을 성공으로 재분류.
- 새 temperature/pCN/kernel 후보를 결과가 좋아질 때까지 반복 탐색.
- `L>1`·고정 block 실패의 비교 기준을 사후에 바꾸기.
- q 재학습, 차원 확대, neural operator, 실제-target Stein CV 성능 실험.
- mesh·whole-training·sealed confirmation·논문 superiority 주장.
- cloud/유료 자원, 패키지·드라이버 설치, OS 전원 설정 변경.
- 커밋·푸시·PR 병합·브랜치 변경.

작은 독립 수학 oracle와 증명 검토는 reference 실패와 관계없이 가능하다. 그것을 실제 금융 rare-event 성능의 성공으로 쓰지는 않는다.

## 5. 전체 순서와 시간 배치

| 단계 | 우선순위 | 정상 배정 | 핵심 결과 | 실패 시 |
|---|---:|---:|---|---|
| N0 | 필수 | 15분 | 현 상태·수학·환경·예산 봉인 | 실험을 시작하지 않고 원인 보고 |
| N1 | 필수 | 60분 | durable checkpoint·누적 예산·재개 검사 | 실제 장기 실험 금지 |
| N2 | 필수 | 45분 | shifted/bimodal analytic oracle | 해당 sampler 실제 실험 금지 |
| N3 | 필수 | 60분 | 동일-law cache·비용 분해 | 원래 evaluator 유지, 최적화 실패 기록 |
| N4 | 높음 | 75분 | fresh full-r 대 marginal-reference 비교 | 유효한 완료 범위만 개발 기록, 자격 미부여 |
| N5 | 높음 | 60분 | 초기 coverage/island 대조 | 새 sweep 없이 원인·미해결 기록 |
| N6 | 필수 | 30분 | 수학·원인·다음 설계 결정 | 증명 미완료를 미완료로 명시 |
| N7 | 필수 | 45분 | 전체 검증·산출물 감사·인계 | 실패를 고치거나 정확한 blocker 보고 |
| 예비 | 안전 여유 | 60분 | 오류 복구·재검증 | 과학 표본·후보 추가에 사용 금지 |

정상 배정 합계 390분, 예비 포함 450분이다. N4/N5가 보류되면 시간을 무작정 Monte Carlo로 채우지 않고 N6/N7을 앞당긴다.

실제 timing 실험은 한 프로세스·CPU 1 thread·순차 실행한다. pytest, 다른 수치 실험, 무거운 병렬 분석을 같은 timing 구간에 겹치지 않는다. 수식 검토·코드 검토는 독립적으로 진행할 수 있지만 동일 파일을 동시에 수정하지 않는다.

마지막 45분은 새 과학 job 착수 금지 구간이다. audit·검증·체크포인트·인계 보고만 한다.

## 6. 공통 예산과 노트북 보호

### 6.1 예산 계층

아래 숫자는 **신규 야간 개발 suite**의 상한이다. 기존 production 예산과 별개이며 기존 gate를 변경하지 않는다.

| 단계 | 신규 과학 work 상한: FFT+CDF | 과학 실행 wall 상한 | 운영 원칙 |
|---|---:|---:|---|
| N0 audit | 500,000 | 120초 | 기존 자료 재생성 금지 |
| N1 checkpoint toy | 500,000 | 120초 | 작고 결정적인 중단 테스트 |
| N2 oracle | 4,000,000 | 600초 | toy potential 호출도 별도 계측 |
| N3 profile/cache | 4,000,000 | 600초 | warmup·오류 비용 포함 |
| N4 fresh IS | 32,000,000 | 1,800초 | 고정된 balanced job grid |
| N5 coverage | 8,000,000 | 1,200초 | 대조 하나만, 추가 sweep 없음 |
| N6 수식 검토 | 신규 MC 없음 | 해당 없음 | 참값 없는 실제 성능 주장 금지 |
| N7 artifact audit | 1,000,000 | 120초 | 전체 테스트 비용은 별도 |
| 합계 | **50,000,000** | **4,560초** | 미사용 예산 단계 간 이전 금지 |

추가 전체 보호 상한은 과학 실행 누적 7,200초, 전체 세션 27,000초다. 단계 상한과 전체 상한 중 먼저 도달한 조건을 적용한다. 표의 wall은 코딩·import·문서·전체 테스트 시간이 아니라 과학 runner 구간 상한이다. 모든 작업의 총시간은 별도 기록한다.

toy는 FFT/CDF가 없을 수 있으므로 `toy_potential_evaluations`를 독립 ledger로 기록하고 N2 최대 4,000,000회를 적용한다. toy가 무료인 것처럼 FFT+CDF=0만 보고 반복하지 않는다. 테스트 내부 미계측 potential 호출은 0으로 기재하지 않고 `not_instrumented_test_work`로 구분한다.

### 6.2 자원 조건

- Torch/BLAS 등 실제 사용 라이브러리의 thread convention을 시작 시 확인·기록한다. scientific timing은 1 thread.
- RSS 상한은 `min(4 GiB, 물리 RAM의 25%)`다. 시스템 전체 여유 메모리도 검사한다.
- 과학 batch 초기 상한은 8,192다. 큰 batch가 이득이어도 RSS pilot 없이 올리지 않는다.
- free disk 10 GiB 미만이면 새 큰 artifact 실행을 보류한다.
- 신규 야간 artifact 총 증가량은 2 GiB 이내를 기본 상한으로 둔다. 초과 예측이면 raw 저장을 줄일 새 명세를 먼저 봉인한다.
- 파일은 workspace 내부 새 run 디렉터리에만 생성한다. 기존 결과를 압축·삭제하여 자리를 만들지 않는다.
- 전원·절전·프로세스 종료 가능성은 시작 조건에 기록한다. 사용자 설정을 임의 변경하지 않는다.
- 예산 확인은 큰 연산 전에도 수행하지만 wall의 완전한 hard-real-time 보장은 아니다. bounded chunk와 종료 여유를 사용한다.

실제 머신이 느리거나 메모리가 부족하면 **통계 결과를 보기 전 allocation 단계에서만** 균형 잡힌 규모로 축소할 수 있다. 그 결정과 config를 저장한다. 본 실행 도중 낮은 RSE·좋은 성능을 보고 표본 수를 바꾸지 않는다.

## 7. 모든 단계가 지켜야 하는 수학·통계 계약

### 7.1 원래 target

현재 finite-grid terminal downside, ε=1, left-point rBergomi 구현에 한정한다. N=32인 실제 개발 셀에서 local Gaussian 좌표는 d=2N=64다.

\[
p=\mathcal N(0,I_d),\quad 0\le g\le1,\quad D_q=q/p,
\quad \mu=E_p[g],\quad M_2(q)=E_p[g^2/D_q].
\]

q는 평가할 원래 고정 proposal, r은 reference 생성 proposal이다. q와 r을 혼동하거나 q를 새 marginal proposal로 교체하지 않는다. exact defensive mass·mixture weights·q digest를 보존한다.

### 7.2 마지막 pair와 새 비교의 의미

z=(A,B), B는 마지막 두 Gaussian 좌표이며 p=p_Aφ₂다. r_A는 r의 **정확한 marginal**이다.

full-r IS:

\[
X_\mu(Z)=g(Z)p(Z)/r(Z),\qquad
X_{M_2}(Z)=g(Z)^2p(Z)^2/[q(Z)r(Z)],\quad Z\sim r.
\]

새 marginal reference:

\[
Y_\mu(A)=\overline g(A)p_A(A)/r_A(A),\quad A\sim r_A,
\]

\[
Y_{M_2}(A,B)=\frac{p_A(A)}{r_A(A)}\frac{g(A,B)^2}{D_q(A,B)},
\quad A\sim r_A,\ B\sim\varphi_2.
\]

여기서 \(\overline g=E_{\varphi_2}[g\mid A]\)는 구현된 해석적 CDF 식이다.

- μ에서는 Y가 full-r X의 조건부 평균이므로 같은 outer marginal에서 Rao–Blackwell 분산 비증가가 성립한다. 시간 효율까지 보장하는 정리는 아니다.
- **M₂의 L=1은 내부 평균을 정확히 적분한 Rao–Blackwell estimator가 아니다.** reference density를 r에서 r_Aφ₂로 바꾼 한 번의 inner IS이다. full-r 대비 분산 감소 보장은 없다.
- gbar², q_A 분모, r(A,0), 마지막 y 삭제는 원래 M₂를 바꿀 수 있으므로 금지한다.
- rank-zero identity-covariance mixture 지원 범위를 벗어나면 거부한다. finite-rank covariance를 조용히 잘라 쓰지 않는다.

### 7.3 추론 단위·표본 분리

- ordinary IS의 단위는 독립 full draw 또는 독립 outer draw다.
- nested L개의 inner draw를 nL개 IID 단위로 계산하지 않는다.
- SMC는 독립 whole-run normalizer가 단위다. terminal particle·ancestor·island particle을 IID로 취급하지 않는다.
- toy, calibration, throughput pilot, selection, development-replication, 실제 reference 역할을 분리한다.
- 방법별 독립 seed namespace를 사용한다. 공통 난수를 사용한다면 covariance를 저장하고 독립 스트림용 equivalence 함수를 재사용하지 않는다.
- 새 개발 표본을 과거 micro/kernel/block/repair 결과에 pooling하지 않는다.

### 7.4 정확성과 관측 정밀도를 구분

- 낮은 sample RSE, 두 방법의 비슷한 mean, 많은 ancestry는 미발견 모드 부재 증명이 아니다.
- shared static guide를 쓰는 두 IS 방법의 일치는 guide-independent 검증이 아니다.
- fixed-schedule SMC의 normalizer 이론과 유한 run의 안정적 tail 탐색은 별개다. 느린 mixing만으로 estimator가 이론적으로 biased라고 단정하지 않는다.
- potential에 상수를 곱하면 normalized bridge target은 변하지 않는다. normalizer의 작은 절대값만을 난이도의 원인으로 설명하지 않는다.
- finite-grid unbiasedness는 연속시간 가격에 대한 unbiasedness가 아니다.
- 통계적 CI·bootstrap·관측 sensitivity는 명시한 유한표본 근사이며 distribution-free 인증으로 포장하지 않는다.

## 8. N0 — 시작 전 상태와 실행 manifest 봉인

### 해야 할 일

1. 실제 HEAD·브랜치·dirty diff·Python 경로/버전·Torch·NumPy·SciPy·thread·CPU/RAM/free disk를 기록한다. 토큰·credential·개인 환경 전체를 dump하지 않는다.
2. 기존 micro/kernel/block/allocation artifact와 source ZIP hash를 inventory에 연결한다.
3. `production_config=null`, `p2_authorized=false`, 기존 실패 사유가 유지되는지 검사한다.
4. 두 개발 셀의 task/parameter/N와 parent rep별 fixed q·static guide를 exact binding한다.
5. 신규 예상 job 전체, ID·role·seed namespace·count·cap·source/config digest를 manifest에 저장한다.
6. 새 runner와 현재 runner를 구분한다. 현 `post_audit_v2_reference_redesign.py`는 종료 후 최종 JSON을 한 번 쓰므로 아직 durable checkpoint가 아니다.

### 출구

source/target/q/seed 역할·환경·저장 경로가 유효할 때만 N1로 진행한다. 기존 결과의 source와 현재 source가 다르면 버전 차이를 기록하지, 옛 hash를 다시 생성하여 덮어쓰지 않는다.

## 9. N1 — 장시간 실행 전에 checkpoint·재개부터 구현

### 9.1 최소 구현

기존 [reference_shards.py](../../src/path_integral/reference_shards.py)의 `write_json_atomic_nonoverwriting` 등 저장 primitive를 재사용한다. 현 Windows 파일시스템에서 atomic/non-overwriting 동작도 작은 테스트로 확인한다.

새 실행기는 최소 다음을 제공해야 한다.

- run manifest: 전체 expected grid, source ZIP SHA, config SHA, q/guide/task digest, 역할, count, thread, 예산.
- complete shard: IID 고정 batch 또는 완성된 whole SMC run.
- sufficient moments: count, log sum, log sum squares, 최대 contribution, 독립 block mean.
- work: 실제 FFT·CDF·density component·toy 호출·warmup·실패·retry 비용.
- status: `complete / pending / interrupted / protocol_failure / invalid_artifact`.
- cumulative budget: 완료/미완료 시도와 이전 세션의 소모량을 모두 합산.
- append-only failure/event ledger와 atomic completion manifest.
- resume 검사: checksum·identity·예상 unit 수·seed·중복·source/config 변경을 거부.

각 unit의 work 예약과 시작 이벤트를 계산 전에 원자적으로 기록한다. 강제 종료로 실제 소모량을 확정하지 못한 시도는 무료로 취급하지 않고 예약된 work와 확인 가능한 wall 상한을 보수적으로 비용에 반영한다. 소비량이 미확정인데 예산이 충분하다고 가정하여 재개하지 않는다. variable-work kernel도 bounded chunk/attempt별 선기록이 필요하다.

source가 바뀌면 같은 stochastic run을 이어 붙이지 않는다. 새 source epoch로 분리하고 기존 결과의 재사용 가능 여부를 따로 감사한다.

### 9.2 중단과 재시작 규칙

1. **완료 shard만** 동일 manifest로 재사용한다. 중복으로 합산하지 않는다.
2. 미완료 unit은 estimator에 넣지 않는다.
3. 재개가 정확하려면 RNG, particle/weight, bridge index, ancestry, kernel state, 비용까지 복구해야 한다.
4. 첫 구현은 complete-unit boundary 재개를 우선한다. 임의 trajectory 중간 복구를 지원한다고 선언하지 않는다.
5. 비과학적 중단의 동일 seed 재계산을 허용하려면 deterministic replay와 누적 비용이 먼저 검증되어야 한다. 미완료 원기록은 남긴다.
6. ellipse rejection cap, nonfinite, target/density 오류는 resource pause로 바꾸지 않는다. 실패를 없애기 위한 재추출 금지.
7. cap/중단 때문에 고정 grid를 끝내지 못하면 완료 prefix만으로 reference 또는 효율 자격을 부여하지 않는다.
8. 나중에 수정한 코드·다른 seed로 성공한 실행은 새 실험이다. 기존 실패를 지우거나 정밀도 자료에 선택적으로 섞지 않는다.

### 9.3 필수 테스트

- 정상 단일 실행과 complete-boundary 중단·재개 결과의 동일성.
- 저장 중 강제 예외, 존재 파일 덮어쓰기, corrupt shard, digest mismatch, duplicate unit, missing seed 거부.
- 중단 이전 소모량이 resume 후 budget에서 차감되는지.
- 실패 prefix가 평균·SE·pass 판정에 포함되지 않는지.
- 큰 다음 unit이 남은 work/time에 들어오지 않으면 시작하지 않는지.

**N1 gate:** 이 항목을 통과하기 전 실제 장시간 N4/N5는 시작하지 않는다. 해결이 길어지면 야간 결과는 실행 안정성 개선으로 제한한다.

## 10. N2 — 참값이 알려진 어려운 toy에서 오류와 coverage 분리

현재 중심 0 Gaussian stationarity/normalizer 검사는 재사용한다. 같은 쉬운 검사를 많이 반복하는 대신 **shifted rare bump와 비대칭 bimodal bump**를 추가한다.

### 10.1 정확한 oracle

표준정규 prior 아래

\[
g(z)=\sum_{k=1}^K a_k\exp\{-\|z-m_k\|^2/(2s_k^2)\},
\quad a_k>0,\quad\sum a_k\le1.
\]

그러면 0<g≤1이고

\[
\mu=\sum_k a_k
\left(\frac{s_k^2}{1+s_k^2}\right)^{d/2}
\exp\left\{-\frac{\|m_k\|^2}{2(1+s_k^2)}\right\}.
\]

q=p인 M₂에는 pair 적분을 사용한다.

\[
\tau_{ij}=1+s_i^{-2}+s_j^{-2},\quad
b_{ij}=m_i/s_i^2+m_j/s_j^2,\quad
c_{ij}=\|m_i\|^2/s_i^2+\|m_j\|^2/s_j^2,
\]

\[
M_2(p)=\sum_{i,j}a_i a_j\tau_{ij}^{-d/2}
\exp\{-\tfrac12(c_{ij}-\|b_{ij}\|^2/\tau_{ij})\}.
\]

τ에는 prior precision 1이 이미 들어 있다. prior를 두 번 더하지 않는다. 모든 ordered pair를 합하거나 i<j 교차항에 2를 곱한다. logsumexp 등으로 안정화하고 별도 square-completion/저차원 수치 oracle로 식을 교차검사한다.

μ endpoint의 posterior component는 mean `m_k/(1+s_k²)`, covariance `s_k²/(1+s_k²) I`이며 component mass는 `a_k Z_k/μ`다. 서로 겹치는 component의 mass와 단순 공간상의 mode-region mass를 같은 것으로 쓰지 않는다.

**주의:** bimodal `(Σ bumps)^β`의 fractional β normalizer는 이 Gaussian-mixture 닫힌 식으로 계산되지 않는다. endpoint μ/M₂ oracle만 정확하다. 중간 true bridge mass가 필요하면 별도 적분·오차 계약이 필요하다.

### 10.2 유한 후보와 선택 규칙

- d=2에서 centered control, shifted bump, asymmetric bimodal의 세 문제만 둔다.
- m/s/a는 실제 SMC 출력 전에 analytic μ/M₂와 prior-region mass를 보고 고정한다.
- μ와 q=p M₂를 별도 target으로 검사한다.
- oracle 구조·상수·density 오류는 deterministic 검사로 판정한다.
- sampler의 관측 mean 오차·whole-run skew는 사전 고정된 반복 수와 CI/sensitivity로 분석한다. “5 SE 안에 들어올 때까지 반복”하지 않는다.
- 쉬운 centered toy만 통과하고 shifted/bimodal에서 불안정하면 actual reference 안정성 통과로 쓰지 않는다.

기본 toy 명세는 다음과 같다. 이 값은 아직 실행한 결과가 아니며, N0/N2의 analytic 검산 후 실제 sampler 출력 전에 고정한다. 구현 불가·수치 문제로 바꾸면 이유와 새 manifest를 남긴다.

| toy | a | m | s | 목적 |
|---|---|---|---|---|
| centered | 1 | (0,0) | 1 | 기존 쉬운 control |
| shifted | 1 | (6,0) | 0.5 | prior에서 드문 posterior 중심 |
| asymmetric bimodal | (0.000001, 0.999999) | ((0,0),(6,0)) | (0.25,0.5) | 작은 중심 bump와 중요한 먼 bump의 경쟁 |

coverage용 sampler 대조는 N5와 같은 단일 population/island 설계만 사용한다. 기본 반복은 `3 toy × 2 target × 2 allocation × 8 whole replicates`, 각 allocation 총 particles 512, levels 64, mutation 1이다. 기본 toy potential forecast는 3,096,576회로 N2의 400만 상한 이내다. 준비·검증의 추가 호출도 계측하여 남은 한도에 포함한다. endpoint 비교 12개 family의 α=.05와 근사 uncertainty 규칙을 사전 선언하되, 8-run 결과를 sampler unbiasedness의 엄밀한 통계 증명으로 사용하지 않는다.

### 10.3 coverage 측정

사전 정의한 box/projection region R의 prior mass p_R를 Gaussian CDF로 정확히 계산할 수 있다면, 초기 M개 IID particles의 hit 확률 `1-(1-p_R)^M`과 관측을 비교한다.

initial hit은 성공의 필요조건도 충분조건도 아니다. mutation으로 나중에 진입할 수 있고 resampling으로 이미 발견한 영역을 잃을 수도 있다. 따라서 first-hit β, pre/post-resampling region mass, mutation 진입/이탈, endpoint contribution, whole-run normalizer를 함께 기록한다.

### 출구

정확성 실패면 N5 실제 sampler 진단을 잠근다. coverage 성능 실패 자체는 숨기지 않고 N6의 원인 판단 자료로 남긴다. toy 성공은 실제 Volterra 문제의 해결 증명이 아니다.

## 11. N3 — 같은 law를 유지하는 계산 최적화와 profiling

### 11.1 먼저 측정할 것

- batch 크기 1/2/4/16/128/1,024/8,192에서 simulator 고정 준비·FFT·CDF·density 시간을 분리.
- cache 구성 시간, cold/warm timing, Python callback/검증 overhead.
- ellipse의 angle attempt 수, active batch 분포, β별 wall/potential 소비.
- 실제 메모리, source/import/archive·I/O와 scientific wall의 분리.

각 profile 횟수·방법 순서·warmup은 먼저 고정한다. 더 빠른 측정치만 고르지 않고 중앙값·분포와 전체 비용을 저장한다. 시스템 부하가 바뀌면 해당 timing session을 따로 표시한다.

### 11.2 우선 구현할 cache

1. BLP local covariance/Cholesky·historical kernel·kernel FFT.
2. deterministic variance compensator·고정 시간 스케줄.
3. mixture log weights·mean norms·고정 projection 준비값.

cache key는 최소 N, dt/T, H, dtype, device, convolution convention과 실제 의존 파라미터/텐서 digest를 포함한다. 변동하는 ξ/η/ρ/task에 의존하는 값을 파라미터 없이 재사용하지 않는다. cache는 immutable하며 용량 상한과 invalidation 검사가 있어야 한다.

기존 `rbergomi_fft.py`는 simulator 호출에서 kernel을 만들고 convolution 내부에서 kernel FFT를 반복할 수 있다. 이를 첫 최적화 대상으로 삼는다. broad rewrite보다 기존 경로와 대조 가능한 작은 adapter를 우선한다.

### 11.3 조건부 후순위 최적화

Gaussian Volterra 선형 driver map L에 대해

\[
L(x\cos\theta+\nu\sin\theta)=L(x)\cos\theta+L(\nu)\sin\theta
\]

를 이용한 ellipse candidate cache는 가능하다. 하지만 비선형 variance·I·J·CDF는 각 candidate에서 다시 계산한다. 이전 candidate의 volatility를 고정하면 다른 법칙이다.

이 최적화는 기본 cache와 oracle가 끝나고 시간이 남을 때만 작은 prototype으로 검토한다. 두 최적화를 동시에 바꿔 원인을 섞지 않는다. 승인되지 않은 speedup을 전제로 N5 예산을 늘리지 않는다.

### 11.4 정확성 gate

- 동일 innovations에서 N=1/4/32, H/η/ρ 변형, cold/warm cache 경로와 log potential을 원래 구현과 대조.
- direct convolution·현재 FFT·cached FFT를 각각 비교.
- cache key 변경/미변경, dtype/device, unsupported task를 검사.
- floor·signed ρ·full q norm·마지막 y 의존성을 유지.
- clipping, memory truncation, KL truncation, float32/fastmath 도입 금지.
- cache-only 버전은 RNG 호출을 바꾸지 않아야 한다. 더 깊은 연산 재배열에서 floating-point accept 경계가 바뀌면 원인·허용 오차·분포 검증을 별도 명시한다.

**성공의 의미:** 같은 law의 비용 감소다. 초기 coverage, 필요 whole-run 수, reference 정밀도가 해결됐다는 뜻은 아니다. cache가 이득 없거나 정확성에 의문이 있으면 원래 evaluator로 N4를 수행한다.

## 12. N4 — fresh full-r 대 marginal-reference 비교

### 12.1 새로운 가설을 분리

과거 가설: “같은 r_Aφ₂ 기준에서 L>1이 L=1보다 비용 proxy를 20% 줄이는가?” → 실패.

이번 가설: “full-r보다 r_Aφ₂ 기반 reference가 원래 μ/M₂를 같은 work에서 더 효율적으로 평가하는가?” → 아직 미검증.

이번 가설이 성공해도 과거 실패를 재채점하지 않는다. r을 공유하므로 guide-free corroboration도 아니다.

### 12.2 본 실행 전 고정할 설계

1. canonical/high-η, parent rep0의 고정 q로 시작한다.
2. 각 셀에서 μ 두 방법, M₂ 두 방법: 총 8개 job.
3. N3와 별도 seed의 throughput-only pilot으로 complete batch 비용을 측정한다. mean/variance를 본평가 selection에 쓰지 않는다.
4. 같은 FFT+CDF work의 fixed count를 먼저 봉인한다. 기본 후보는 **job당 총 262,144 units = shard당 8,192 units × 32 shards**다. 8-job 한 묶음은 256 shards, 두 묶음은 512 shards다. 이 숫자는 최종 실행량이 아니라 preflight의 출발값이다.
5. fresh development-replication도 같은 8-job grid·새 namespace로 별도 봉인한다. 두 묶음을 구분하며 global sealed confirmation이라고 부르지 않는다.
6. 이 두 묶음은 기본 count에서 약 838.86만 FFT+CDF 단위가 예상된다. 실제 counted work가 다르면 보수적으로 다시 forecast한다.
7. resource forecast가 부족하면 **출력 통계를 보기 전** 두 방법/두 셀을 균형 있게 줄여 config를 확정한다. 실행 중 count를 바꾸지 않는다.

μ의 target은 q와 무관하다. 여러 parent q가 있다고 μ를 여러 독립 target처럼 불필요하게 반복하지 않는다. M₂는 parent별 q digest가 달라 별도 target이다.

### 12.3 측정과 판정

- sample mean, IID unit SE, CV², 최대 share, 상위 1% share, block mean·leave-one-block-out.
- contribution의 시간대 spike·I/J·log-density 방향을 calibration에서 고정한 bin에 요약.
- unit variance × 실제 unit cost를 주 비용 proxy로 사용한다. 같은 target의 두 noisy sample mean²을 서로 다르게 분모로 써 우위가 왜곡되지 않도록 한다.
- 전체 wall에는 cache 구축·sampling·density·CDF·moment merge·I/O를 분리하여 포함한다.
- 같은 work 비교와 같은 wall의 forecast를 별도 제시한다. 실제 fixed-precision time 측정이라고 부르지 않는다.
- independent blocks 기반 uncertainty/sensitivity를 함께 보고한다. bootstrap이 unseen tail을 인증하지 않는다는 한계도 표시한다.
- μ와 M₂의 두 셀 총 네 primary comparison family, α=.05를 사전 지정한다. 채택을 추론적으로 주장하려면 적용한 simultaneous uncertainty 규칙을 미리 구현·봉인한다. CI가 불안정하면 point-estimate 개발 후보로만 남긴다.

평균·분산의 sufficient moments만으로 상위 1% share와 geometry별 기여는 복원할 수 없다. 사전 고정한 top-k heap 또는 제한된 log-contribution 배열, geometry-bin별 log sums를 별도 저장한다. 기본 count에서 scalar float64 기여 262,144개는 약 2 MiB이므로 full path 대신 제한된 기여 배열 저장을 검토할 수 있다. heap 크기는 예상 total count의 1% 이상으로 봉인한다. 근사 quantile/구간 요약을 쓰면 근사와 오차를 명시한다.

개발 채택 후보 기준은 같은 target에서 비용 proxy ≤0.8, fresh replication의 방향 일치, target/binding 감사 통과다. 이것만으로 accuracy qualification 또는 reference 인증을 부여하지 않는다. mean disagreement는 평균내어 해결하지 않고 별도 문제로 남긴다.

### 12.4 다른 fixed q로 제한 확장

rep0의 두 셀 M₂에서 모두 개발 채택 기준이 충족되고 forecast가 cap 이내일 때만 parent rep1–4의 **M₂ 두 방법**을 평가한다. 전체 16-job grid와 count를 결과를 보기 전에 봉인한다. 기본 count에서 약 838.86만 추가 FFT+CDF 단위다.

한 q만 좋아서 해당 q를 고르는 것이 아니라 선언한 전체 grid를 완료한다. 실패/미완료가 있으면 확장 전체의 자격은 미해결이다. 이는 frozen-q robustness 진단이며 parent부터 전체 재학습을 반복한 결과가 아니다.

기본 두 묶음과 조건부 확장의 예상 합계는 약 1,677.72만 단위이며 3,200만 phase cap보다 작다. 실제 throughput/밀도 비용·forecast 검사를 통과한 경우에만 착수한다.

### 출구

- 유용하면 다음 reference 설계의 guide-based 후보로 남긴다.
- marginal 방법이 악화되면 캐시/조건부 μ의 범위만 남기고 L=1 risk 대체는 폐기한다.
- 두 방법이 낮은 RSE로 일치해도 공통 guide blind spot은 미해결이다.
- 어느 결과에서도 기존 34-job production 또는 P2를 자동 해제하지 않는다.

## 13. N5 — 초기 coverage·genealogy를 한 가지 대조로 분리

### 13.1 왜 새 kernel sweep이 아닌가

이미 pCN/bridge bounded repair와 ellipse 대조를 수행했다. 이번 질문은 “더 좋은 parameter가 무엇인가?”보다 **동일 초기 총 population에서 전역 resampling이 rare contribution을 어떻게 살리거나 잃는가?**다.

N2 toy에서 먼저, 그 다음 실제 rep0의 μ/M₂에서 다음 단일 대조를 수행한다.

- A: 한 population의 particles=512.
- B: 독립 island 4개 × particles=128, island 간 resampling 없음.
- 총 초기 particle 수·고정 β schedule·mutation·resampling convention은 동일.
- 고정 64개 β points, bridge power 2, mutation steps 1, pCN scale .35, stratified resampling every 4를 새 diagnostic configuration으로 제안한다. 실행 전 코드 convention과 work forecast를 확인하여 봉인한다.
- 방법당 독립 8 whole replicates, 실제 셀 2개 × estimand 2개.

이것은 새 진단이며 옛 128-particle kernel pilot의 누락 자료를 채우는 실험이 아니다. β/pCN scale/새 kernel을 추가 탐색하지 않는다. 본진단을 보고 island 수나 particles를 바꾸지 않는다.

현 SMC의 final mutation 없음 규칙에서 pCN potential 호출은 `P × [1+(levels-2)×mutation_steps]`다. 위 grid의 실제 Volterra FFT+CDF 예상은 약 **412.88만 단위**다. source convention이 다르면 다시 계산하고 800만 cap 안에서 전체 grid가 가능할 때만 시작한다.

### 13.2 island estimator 계약

각 island가 같은 normalizer Z의 유효한 SMC estimator라면 B의 whole-unit estimator는 네 normalizer의 산술 평균이다. log 값은 logsumexp로 평균한다. unnormalized log normalizer 자체를 평균하지 않는다.

표준오차 단위는 **8개의 독립 aggregate whole replicates**다. 4×8 island를 임의로 32개의 비교 단위로 바꾸지 않는다. within/between-island variation은 별도 진단으로만 보고한다.

한 aggregate replicate에서 island 하나라도 실패하면 나머지 세 normalizer만 평균하여 complete unit으로 만들지 않는다. 실패 비용·완료 부분을 보존하고 aggregate 전체를 미완료/실패로 표시한다.

총 초기 IID particle이 동일하므로 지정 region의 최초 hit 확률은 A와 B에서 원칙적으로 같다. island 자체가 초기 rare-region 발견 확률을 자동 증가시키는 것이 아니다. 차이는 이후 resampling/genealogy/mutation에서 찾아야 한다.

### 13.3 기록할 진단

- β=0 log-g/log-risk 분포, maximum, 사전 고정 region 최초 hit.
- 최초 8개 β 구간과 정해진 후속 구간의 incremental ESS·최대 weight.
- resampling 전후 region mass·ancestor·island별 contribution.
- mutation acceptance·이동 크기·region 진입/이탈·first-hit β.
- 전체 whole-run mean/SE, normalizer 분포, maximum-run share, leave-one-run-out, 사전 block 민감도.
- toy endpoint의 analytic component/region mass와 관측 차이.
- 실제 경로는 spike time·window·I/J·density와 rare tail contribution의 경험적 관계.

cluster/bin/threshold는 calibration 후 fresh 진단 전에 고정한다. contribution을 본 뒤 유리한 cluster만 만들지 않는다. geometry clustering은 원인 진단이지 빠진 모든 mode가 없다는 보증이 아니다.

SMC terminal mode/region contribution을 normalizer와 연결할 경우 **최종 weighted empirical measure**를 사용한다. resampled particle 개수만으로 exact mode contribution을 선언하지 않는다.

### 판정

8-run 실제 mean·RSE·skew만으로 bias 제거나 precision qualification을 선언하지 않는다. 이 대조는 원인 구분용이다. islands가 도움 없거나 더 나쁘면 새 island 수 sweep을 시작하지 않고 기록한다.

oracle·법칙·정확성 오류면 sampler를 수정하고 작은 oracle부터 다시 검사한다. 실제 coverage만 실패한 경우 “될 때까지 반복”하지 않고 N6에서 새로운 질문/예산/독립 방법을 결정한다.

## 14. N6 — 결과를 수학과 다음 설계로 연결

### 14.1 이번 밤에 정리할 작은 분산·비용 관계

fixed outer proposal과 IID inner 구조에서

\[
V(L)=V_{out}+V_{in}/L.
\]

실제 비용이 `c_out + L c_in`으로 근사되고 모든 상수가 양수인 범위라면 fixed-budget variance proxy는

\[
F(L)=(V_{out}+V_{in}/L)(c_{out}+L c_{in}),\qquad
L_* = \sqrt{\frac{V_{in}c_{out}}{V_{out}c_{in}}}.
\]

실제 정수·L≥1·최대 L에서 선택하고, 비용의 batching 비선형성은 직접 측정으로 확인한다. 음수 noise-subtracted Vout를 0으로 잘라 L*를 만들지 않는다. 0인 경계의 경우는 따로 분석한다.

이 관계는 last-pair 증가가 실패한 이유를 설명하는 기존 nested-MC 분해의 응용이다. 이 식 자체를 새로운 수학적 발견이라고 주장하지 않는다. 적용 범위·반례·측정된 outer floor를 정리한다.

### 14.2 더 본질적인 후보는 수식 수준에서만 검토

outer variance가 계속 지배하면 사전 고정한 여러 시간대/선형 Gaussian 방향을 joint하게 조건부 적분하는 후보를 검토한다.

- 고정 orthogonal transform에서 Gaussian 조건부 법칙·reference marginal·full-q 분모를 유도한다.
- Volterra 미래 변화와 실제 path 재계산 비용을 포함한다.
- path마다 최대 spike 좌표를 고르는 방식은 선택 조건을 유도하기 전 금지한다.
- nonlinear mixture의 정확한 conditional component weights/covariance를 확인한다.
- 조건부 분산 감소와 계산비용 감소를 분리한다.
- 새 실제 sampler/모델 대규모 구현은 하지 않고 다음날 제안의 수학·실행 가능성만 작성한다.

### 14.3 결과에 따른 다음 판단

| 관측 | 다음 우선순위 | 하지 않을 해석 |
|---|---|---|
| cache만 개선, coverage 그대로 | cheap evaluator 재사용 + 독립 설계 검토 | 모델/정밀도 성공 아님 |
| marginal M₂가 두 셀·다른 q에서 유용 | guide-based reference 후보로 유지 | guide-independent 검증 아님 |
| μ만 개선 | μ evaluator 별도 채택 후보, risk 설계 분리 | μ 결과로 M₂ 인증 금지 |
| toy에서도 rare mode 소실 | 초기 bridge/resampling의 정확한 원인부터 설계 | Volterra만의 어려움으로 돌리지 않음 |
| toy 양호, 실제 coverage 불안정 | 실제 geometry/여러 시간대 joint 방향 검토 | toy pass를 실제 pass로 대체 금지 |
| islands가 유용 | 한 번의 fresh 확대 feasibility 설계를 다음 단계로 제안 | 밤에 추가 scale sweep 금지 |
| 방법 mean이 불일치 | source/target 감사 후 coverage/불확실성 분리 | 평균내어 truth 만들기 금지 |
| 모든 새 후보가 실패 | target 범위·reference mechanism·계산예산 재설정 | 대형 신경망으로 우회 금지 |

claim ledger에는 `identity proved / tested implementation / empirical development / conjecture / not supported`를 분리한다. 논문 novelty는 알려진 smoothing·IS·SMC 이름을 바꾸는 데 있지 않고 구조·비용·오차 사이의 추가 관계에서 찾아야 한다. 그 관계가 아직 없으면 솔직히 미입증으로 남긴다.

## 15. N7 — 중간/최종 이론·기술 오류 검토와 아침 보고

### 중간마다 반복할 순서

1. 수식의 measure·sign·normalizer·scope를 코드 변경 전에 검토한다.
2. 작은 deterministic/analytic oracle를 먼저 실행한다.
3. invalid input·cap·seed·source/density 경계 테스트를 실행한다.
4. 관련 회귀가 통과하면 bounded 실제 개발 실험을 한다.
5. 결과를 별도 audit 경로로 재계산하고 실패/미완료까지 검증한다.

### 마지막 검사

```powershell
python -m pytest -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
git diff --check
```

신규 experiment의 mypy 대상과 원자 저장·cache·oracle·runner 테스트는 명시적으로 추가한다. 기존 artifact 감사는 아래와 같이 기존 자료를 덮어쓰지 않고 수행한다.

```powershell
python -m experiments.post_audit_v2_reference_redesign --audit results/post_audit/reference_redesign_micro_v1.json
python -m experiments.post_audit_v2_reference_redesign --audit results/post_audit/reference_redesign_kernel_v1.json
python -m experiments.post_audit_v2_reference_redesign --audit results/post_audit/reference_redesign_block_v1.json
```

새 야간 artifact에는 별도의 `--audit` 경로가 필요하다. 이 CLI는 아직 구현되지 않았으므로 실행 명령을 있는 기능처럼 제시하지 않는다.

전체 테스트 성공만으로 이론적 무오류·tail certification·저널 novelty를 선언하지 않는다. 실패한 테스트, 수정 전 실패, 환경 warning, 미실행 범위를 모두 보고한다. 수정이 원래 estimator를 바꾸면 이전 sample에 새 식을 소급 적용하지 않는다.

## 16. 신규 파일·산출물 제안

**아래 경로는 후속 실행 때 만들 제안이다. 이번 요청에서 생성한 것은 이 계획 MD 한 개뿐이다.** 기존 abstraction이 충분하면 이름을 더 늘리지 않고 재사용하되 실제 mapping을 기록한다.

| 묶음 | 제안 경로/출발점 |
|---|---|
| 야간 runner | `experiments/post_audit_v2_overnight_reference_recovery.py` |
| manifest·예산·재개 | 기존 `reference_protocol.py`, `reference_shards.py`, seed ledger 재사용; 필요 최소 adapter |
| cached evaluator | 기존 `rbergomi_fft.py`와 원래 evaluator에 additive adapter |
| toy oracle | 신규 Gaussian-bump oracle helper + 독립 tests |
| config | `configs/post_audit/overnight_reference_recovery_v1.yaml` |
| 모든 새 run 자료 | `results/post_audit/overnight_reference_recovery_v1/<run-id>/` |
| source·manifest·실패 | 해당 run 안의 source ZIP/manifest/shards/events/audit/allocation |
| 최종 보고서 | `docs/reviews/OVERNIGHT_REFERENCE_RECOVERY_RESULT_2026-10-08_KO.md` |

실제 완료 날짜가 바뀌면 결과 보고서 날짜도 바꾸고 session start/end UTC·local timezone을 기록한다. 기존 v1이 존재하면 새 ID를 정해 봉인하며 덮어쓰지 않는다.

### 각 phase report의 최소 필드

- 목적·가설·비교 기준·target scope.
- planned/completed/failed/pending grid와 전체 expected count.
- source/config/q/r/task/seed 역할·snapshot hash.
- 실제 unit·whole-run 수, mean/SE·sensitivity, inference limitations.
- FFT/CDF/density/toy/실패 비용, scientific wall·전체 wall, RSS/disk.
- 코드 변경, 발견 오류, 수정, 관련·전체 검증 결과.
- correctness/efficiency/coverage/novelty의 별도 status.
- 다음 작업, 금지 작업, 정확한 resume command/identity 또는 재설계 필요 사유.

## 17. 실패·중단·사용자 응답이 없을 때의 분기

| 상황 | 즉시 조치 |
|---|---|
| target/density/likelihood/sign 오류 | 실제 과학 job 중단 → 수식 → 작은 oracle → 회귀 |
| seed/identity/digest mismatch | 자료 사용 거부, 원기록 보존 |
| statistical cap·ellipse cap | protocol failure, 완료 prefix로 자격 부여 금지 |
| OS 중단·전원·프로세스 종료 | checkpoint 경계 확인, 동일 replay 가능한 범위만 재개 |
| 시간이 부족 | 새로운 N4/N5 job 중단, N7/인계부터 마무리 |
| resource forecast 부족 | `not_run_budget` 기록, 기존 production 상한 변경 금지 |
| 새 방법 성능 나쁨 | 원인·반례 기록, 다른 seed/parameter 탐색 금지 |
| 자동으로 해결하려면 범위 확장 필요 | 해당 분기 보류, 사용자가 결정할 질문을 보고 |
| 사용자가 자는 동안 답변 없음 | 범위 내 독립 검토·검증만 진행, 유료/설정/git 변경 금지 |

런타임·사용 한도·머신 중단으로 세션이 끝날 수 있으므로, 다음 세션이 이어 받을 수 있는 manifest와 체크포인트가 핵심이다. 계획 작성 자체는 야간 프로세스를 시작하거나 예약하지 않는다.

## 18. 아침에 확인할 완료 기준

### 반드시 남겨야 할 결과

- [ ] N0: 시작 상태와 기존 실패 관문을 정확히 봉인했다.
- [ ] N1: 원자 저장·중단/재개·누적 예산과 실패 보존을 검증했다.
- [ ] N2: shifted/bimodal μ/M₂ oracle와 coverage 진단을 수행하거나 정확한 실패를 기록했다.
- [ ] N3: cache의 law preservation과 실제 비용을 확인했다. 실패 시 원래 경로를 유지했다.
- [ ] N4: fresh balanced 대조와 replication을 완료했거나 `not_run/unresolved` 사유를 남겼다.
- [ ] N5: 사전 고정 대조 하나의 결과·skew·genealogy를 남겼다. 추가 sweep은 하지 않았다.
- [ ] N6: 현재 병목을 수학·실험 근거로 분리하고 다음 설계를 정했다.
- [ ] N7: 변경 범위 검증·전체 회귀·artifact audit·아침 보고서를 남겼다.
- [ ] 원래 reference production·P2·confirmation 잠금이 유지된다.
- [ ] 기존 변경과 실패 artifact가 보존되고 커밋·푸시는 하지 않았다.

체크박스는 구현/검증 evidence가 있을 때만 완료한다. 파일을 만들거나 계획을 읽은 것만으로 완료하지 않는다.

### 최종 요약의 형식

1. 이번 밤에 실제 완료한 것.
2. 새 비교의 결과: 무엇을 대상으로 어떤 비용/오차가 변했는가.
3. 해결하지 못한 것: 특히 독립 coverage와 reference 자격.
4. 발견한 오류와 수정 후 검증 결과.
5. 다음 우선순위 한 가지와 선택 근거.
6. 멈춘 지점·소모 예산·남은 전체 grid·재개 조건.

## 19. 계획 자체의 오류 검토

작성 시 다음 잠재 오류를 명시적으로 제거했다.

1. L=1 marginal proposal 변경을 M₂ Rao–Blackwellization이라고 오인하지 않는다.
2. μ의 조건부 평균을 제곱해 원래 M₂를 대신하지 않는다.
3. full q·정확한 r marginal·마지막 y dependence를 보존한다.
4. Gaussian-bump M₂에 prior precision을 두 번 넣거나 교차항을 빠뜨리지 않는다.
5. fractional β의 bimodal normalizer를 endpoint 공식으로 잘못 계산하지 않는다.
6. island 평균은 normalizer의 산술 평균이며 whole inference unit을 유지한다.
7. 같은 초기 총 population을 island로 나눈다고 최초 hit 확률이 증가한다고 주장하지 않는다.
8. cache에서 선형 driver만 재사용하고 비선형 미래 variance는 다시 계산한다.
9. cap에서 완료된 좋은 run만 남기는 선택·optional-stopping 문제를 숨기지 않는다.
10. 기존 185시간 forecast를 새 야간 실행 승인으로 해석하지 않는다.
11. 새 개발 채택 기준이 과거 inner-L 실패나 P1/P2 관문을 소급 변경하지 않는다.
12. 테스트·관측 정밀도·coverage·실제 모델 개선·novelty를 별도 판정한다.

이 검토는 알려진 설계 오류를 줄이는 작업이다. 아직 작성하지 않은 코드와 미실행 실험이 완벽하거나 반드시 성공할 것이라고 보증하지 않는다.

## 20. 한 줄 결론

**이번 밤은 “더 큰 모델을 만들어 성공 수치를 얻는 밤”이 아니라, 정확한 실행·공정한 fresh 비교·희귀경로 누락 원인 분석을 통해 다음 연구 설계를 선택할 근거를 확보하는 밤으로 운영한다.**
