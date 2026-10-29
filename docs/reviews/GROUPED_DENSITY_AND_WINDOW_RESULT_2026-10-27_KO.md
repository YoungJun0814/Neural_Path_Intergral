# 정확한 밀도 압축·다중 구간 결합 구현 및 검토 결과

2026-10-27 America/New_York. 기준 main `59f2809`. 이 작업에서는 커밋·푸시하지 않았다.

## 1. 결론

**밀도를 바꾸지 않는 계산 최적화와 equal-trace window proposal을 구현했다.
수치 정합성은 통과했지만, 봉인한 계산 속도 gate는 미달했다.
따라서 사전 계획대로 새 event/risk 표본 비교는 시작하지 않았다.**

- 동일 분포 밀도의 최대 log 차이: `7.105427357601002e-15`.
- AB/BA/AB speed ratio: `1.21293 / 1.04946 / 1.08466`, 중앙값 **1.08466배**.
- 사전 채택 기준: 중앙값 ≥1.2배. 첫 실행만 선택하면 안 된다.
- 판정: `STOP_EXACT_KERNEL_NUMERICAL_OR_SPEED_GATE`.
- 384개 과거 selected witness 슬롯 → cell/좌표 기준 170개 고유 경로.
  전부 물리량·밀도·저장 기여값을 다시 계산해 일치했다.
- 새 window proposal의 성능은 **미평가**다. 실패한 모델 성능 실험이라고
  부르거나, 구현 정확성만으로 개선됐다고 부르면 둘 다 잘못이다.
- P2·production 채택·대규모 학습은 모두 승인하지 않았다.

계획의 계산 gate는 연구 자원 사용에 관한 정책이지 수학적 유효성 조건이 아니다.
이번 STOP은 window 모델이 수학적으로 틀렸다는 뜻이 아니다.
반대로 약한 속도 개선을 P2의 본질적 해결로 해석할 근거도 없다.

## 2. 구현한 것

### 2.1 Exact local-mean grouping

`src/path_integral/grouped_covariance_density.py`

기존 공분산 보정은 시점 j의 mean 두 좌표에만 의존한다. 그 두 값이
정확히 같은 component끼리 원래 mixture log항을 먼저 합친 뒤 보정한다.
다른 좌표의 mean·원래 component weight·전체 mean dictionary는 그대로 남는다.

시점별 shifted component 372개가 5~139개 local-mean group으로 줄었다.
하지만 원래 component log항 계산과 group별 합산 비용은 남는다.
명목 항 수 감소가 그대로 실행시간 감소가 되지는 않았다.

- rounding·mean 근사·component pruning 없음.
- group별 logsumexp로 계산. 전역 최대값 기준으로 작은 group을 0으로 지우지 않음.
- base/개별 arm/전체 arm 평가 지원.
- 원본 means/weights를 복사해 외부 원본 변경으로 캐시가 어긋나지 않게 함.
- 기존 sampling은 변경하지 않음. 새 확률모형이나 학습법이 아니라 동일 밀도의 재계산법.

### 2.2 Equal-trace window covariance

`src/path_integral/window_coupled_covariance.py`

기존 local 방향은 한 residual/다음 Brownian 쌍을 함께 움직인다.
새 방향은 고정된 여러 쌍을 같은 부호로 함께 움직인다.
N32의 사용 시점 1~31을 폭4로 나누어 총8개 window를 만든다.
마지막 window는 29~31의 3시점이다. 미사용 좌표63은 제외한다.

각 방향 w는 단위벡터이고 covariance는 `I+8wwᵀ`이다.
기존 local과 고유값 9, trace 증가 8이 같다. 총 변동을 더 넣는 효과와
시간 결합 방향의 효과를 분리하려는 설계다.

원래 physical 373 means, weights, natural mass .1을 유지한다.
shifted mass의 절반만 window covariance mixture로 바꾼다.
명목 Gaussian 항 수는 3,349개로 base 373개, local 11,905개와 다르다.
capacity와 계산비가 같은 모델 비교라고 주장할 수 없다.

폭1이면 기존 local-coupled와 같은 분포가 된다.
폭4의 밀도·샘플링 구현은 독립 dense Gaussian oracle와 mean/covariance 검사로
확인했으나, **실제 rare-event 성능 또는 fine-grid 확장성은 확인하지 않았다.**

### 2.3 실행·중단·감사 기능

`experiments/post_audit_grouped_window_coupling.py`

- 새 namespace 및 과거 protocol 범위 seed 충돌 검사.
- 실행 전에 source/config/계획/기존 결과 및 감사 hash 봉인.
- 새 3,072개 좌표는 밀도 검사에만 사용. 새 event 정확도 표본으로 세지 않음.
- old/grouped 전체 arm AB/BA/AB 세 번 고정. 관측 후 반복 횟수 변경 없음.
- 조건부 80-unit 실험 branch: base/local/window direct와 known-mass H 및 R.
- risk는 full mixture 분모를 사용. self-normalization 없음.
- risk precision 미달이면 reference equivalence를 평가하지 않음.
- 이번에는 앞 단계 STOP 때문에 80-unit branch·preflight·production은 실행하지 않음.
  branch의 estimator 수식·작업 수·gate는 단위 테스트했지만 실제 end-to-end 실행 검증은 아님.

## 3. 계산 결과와 비용의 정확한 해석

| 순서 | 원래 전체 arm 초 | grouped 전체 arm 초 | old/grouped |
|---|---:|---:|---:|
| AB | 2.68801 | 2.21614 | 1.21293 |
| BA | 2.52059 | 2.40180 | 1.04946 |
| AB | 2.45342 | 2.26193 | 1.08466 |

각 row는 두 셀의 총3,072개 좌표에 대한 kernel 시간이다.
한 노트북의 짧은 순차 측정이며 OS/background 부하를 완전히 통제한 benchmark는 아니다.
가장 좋은 row만 고르거나 population 성능·고정정확도 속도 개선으로 해석하지 않는다.

grouped 개별 arm은 base .0412초, residual 1.7794초, independent 1.8884초,
coupled 1.8608초였다. 이는 단회 기술값이며 원래 개별-arm 최적 구현과의
동일 조건 paired 비교가 아니다. 별도의 scalar-kernel 우위 근거로 사용하지 않는다.

물리 계산 회계:

- kernel: 0회. 원래 밀도 coordinate-pass 12,544회, 약30.35초.
- selected witness 구조 검사: 170회, 원래 밀도 170 coordinate-pass.
- 새 event/risk 표본·사전 preflight·production: **0회**.
- 독립 재생 감사: 같은 170개 경로 물리 재생 170회, 밀도 좌표3,072개 재생.
- 총 실제 물리 호출은 **340회**이며, 모두 과거 selected witness의 검사·재생이다.
  이를 새 독립 accuracy/confirmation 표본 340개라고 부르면 안 된다.

## 4. Selected witness 구조 감사

84개 canonical, 86개 high-eta 고유 경로에 대해 다음을 계산했다.
마지막 미사용 variance가 아니라 **실제로 적분에 사용되는 32개 variance**의
질량 비중을 사용한다. effective duration은 `1/Σ share²`이며 단위는 grid step이다.

| 선택된 경로의 중앙값 | Canonical | High eta |
|---|---:|---:|
| 최대 한 구간의 variance 질량 | 64.04% | 87.15% |
| 상위4구간의 variance 질량 | 85.64% | 95.94% |
| effective duration (step) | 2.341 | 1.312 |
| window / local projection 제곱 에너지 | 48.30% | 39.78% |

여기서 local 공간은31개의 쌍 방향, window 공간은 그 안의8개 고정 방향이다.
비율은 latent 좌표 projection의 기하학적 기술값이다. 설명된 risk 비중이나
variance reduction 비율이 아니다. local 공간 밖의 좌표 효과도 남아 있다.

관찰된 사례에는 매우 짧은 spike가 있고, peak 시점도 여러 곳에 분산돼 있다.
동부호 균등 window 방향은 일부 구조만 담는다. 짧은 spike 또는 window 내부의
반대 부호 변동이 중요하면 희석·상쇄가 생길 수 있다.
따라서 단순 폭4 window가 핵심 해결책이라는 확증은 없다.

**선택 편향 주의:** 이 표는 과거 각 unit/target의 최댓값으로 선택된 사례다.
중복 제거도 그 선택 편향을 없애지 않는다. 모집단 mode 빈도·희귀 기여 질량·인과
기여도·후보의 일반화 성능을 추정하지 않는다. 이 사례로 폭·방향·seed를 맞춘 뒤
동일 자료에서 성능을 주장하지 않는다.

재생 최대 오차:

- log payoff: `0`.
- 저장된 log density: `8.881784197001252e-16`.
- grouped/원래 log density: `3.552713678800501e-15`.
- 저장된 log contribution: `3.552713678800501e-15`.

## 5. 이론 재검토

### 정합성이 확인된 범위

1. 동일 local mean group의 합은 대수적으로 원래 mixture와 같다.
2. `Σ=I+(λ−1)wwᵀ`의 inverse/determinant와 rank-one square-root sampling이 일치한다.
3. shifted mean을 covariance와 함께 잘못 변환하지 않는다.
4. natural component는 그대로 유지하며 전체 mixture 정규화가 보존된다.
5. `qW≥.1p`, `qW≥.5q0`, 따라서 `M₂(qW)≤2M₂(q0)`이다.
   이는 감소 보장이 아니다.
6. `0≤g≤1`과 방어 밀도 floor로 event 기여≤10, 보조 risk 기여≤100이다.
   유한 모멘트가 보장되지만, 평균이 아주 작으면 상대오차는 여전히 클 수 있다.
   현재 문제를 무한분산 문제로 설명하는 것은 부정확하다.
7. H는 세 proposal의 같은 질량 혼합이며 stratum별 known-mass covariance를 사용한다.
   실제 risk 실험은 미실행이므로 이 통계 코드의 테스트 통과와 empirical 검증은 구별한다.

### 아직 해결되지 않은 범위

- 새로운 population variance 감소 정리, uniform task theorem, 연속시간 법칙 없음.
- 폭4 window가 실제 risk를 줄인다는 근거 없음.
- guide-free R도 window 및 simulator를 공유한다. 난수 독립성만으로 구조적 독립 검증이 되지 않음.
- 희귀 기여 모드의 충분한 탐색과 정확한 M₂ 평가가 아직 미해결.
- N32 개발 결과로 N4096 정확도·mesh bias·continuum bias를 대체할 수 없음.
- 이 구현이나 약1.085배 kernel 속도만으로 탑급 저널 기여를 확보하지 못함.

## 6. 다음 우선순위: 채택·보류·중단

1. **채택: 밀도 정합성 검사·구조 감사 도구.** 기존 결과를 바꾸지 않는 진단 도구다.
   production density kernel으로의 교체는 보류한다.
2. **중단: 이번 protocol의 80-unit 새 표본 실험.** 속도 gate를 낮추거나
   같은 seed/횟수로 다시 실행해 통과한 값만 선택하지 않는다.
3. **보류: 동부호 폭4 window의 성능 주장과 큰 학습.** 구현은 남기되
   selected witness만으로 성공 가능성이 입증됐다고 보지 않는다.
4. **후속 계산 설계:** 별도 새 protocol에서 실제로 필요한 base/local/window
   scalar 밀도 소비량을 기준으로 구현·비용을 비교할 수 있다. 원래 full-four-arm
   구현 대비 단회 scalar 시간만 보고 우위를 선언하지 않는다. 이번 STOP은 그대로 보존한다.
5. **후속 본질적 연구:** 좁은 spike, window 내부 상쇄, 여러 시점에 걸친 효과를
   분리하는 독립 진단과 다른 geometry의 risk reference가 우선이다.
   예를 들어 사전 고정한 whitened subspace의 radial strata는 Gaussian 질량을
   정확히 알고 sampler/density를 구성할 수 있는 후보지만, angular blindspot이 남는다.
   아직 구현·효과 검증하지 않았으며 P2 해결책이라고 단정하지 않는다.
6. **P2는 별도 트랙 유지:** 기존 N4096 frozen-ensemble 방법 하나는 약0.9% 관측 RSE에
   도달했다. 남은 것은 자격을 얻은 다른 geometry 기준과의 대조, signed adjacent-mesh
   correction 및 continuum 상대 bias 검증이다. 모든 fine-grid 방법이 정밀도 미달인 것은 아니다.

무작정 차원·operator·폭 후보를 늘리기보다, 계산 절약과 risk 검증의 문제를
분리해야 한다. 다음 실험 역시 새 source/config/seed와 실패 기준을 먼저 고정해야 한다.

**우선순위 보정:** 1.2배 속도가 나올 때까지 kernel만 끝없이 수정하는 것은
P2 해결에 충분하지 않다. 이번 STOP은 유지하되, 후속 독립 protocol은 실제 필요한
단일/세-arm 계산량과 사전 wall/work 예산으로 실행 가능성을 판단해야 한다.
그 안에서 다른 geometry의 risk 검증을 먼저 다루고, kernel 최적화는 보조 과제로 둔다.
과거 full-four-arm 대비 1.2배라는 조건을 모든 향후 과학 실험의 필수조건으로
일반화하지 않는다. 이것은 이번 실패를 성공으로 바꾸는 재판정이 아니다.

## 7. 소프트웨어·무결성 검증

실행 전 신규·관련 기존 테스트 **67개 통과**. 이 중 신규 테스트는41개다.
독립 dense Gaussian 밀도, 부호±, 폭1 환원, 마지막 짧은 window, covariance trace,
샘플 mean/covariance, defensive bound, 극단 좌표, adjacent-float mean 구분,
원본 파라미터 변경, invalid 입력, full-mixture/known-stratum estimator 및 STOP gate를 검사했다.

별도 감사는 새 밀도 좌표 재생,170개 witness 전수 재생,source/config/hash,
kernel 판정 재계산을 통과했다. 새 event/risk unit은 없으므로 해당 array 검사·production
물리 재생 수는 모두0이다. timing 자체를 재현한 감사는 아니다.

전체 regression/static 완료 결과와 공개 파일 무결성 검사는 아래 최종 검증 절에 기록한다.
초기 일반 follow-imports typing 진단에서는 과거 봉인 실험4파일의 기존14개 typing 문제가
함께 노출됐다. 이 파일들을 바꾸지 않았다. 현 CI 범위와 새 runner의 별도 typing 검사를
구분해서 보고하며, 모든 과거 experiments가 typing-clean하다고 주장하지 않는다.

## 8. 재현과 봉인 증거

- [사전 실행계획](../plans/GROUPED_DENSITY_AND_WINDOW_COUPLING_2026-10-27_KO.md)
- [protocol](../../results/post_audit/grouped_window_coupling_v1/session_20261027/protocol.json)
- [kernel 결과](../../results/post_audit/grouped_window_coupling_v1/session_20261027/kernel.json)
- [selected witness 전수 자료](../../results/post_audit/grouped_window_coupling_v1/session_20261027/witnesses.json)
- [전체 결과](../../results/post_audit/grouped_window_coupling_v1/session_20261027/result.json)
- [재생 감사](../../results/post_audit/grouped_window_coupling_v1/session_20261027/audit.json)
- [봉인 소스](../../results/post_audit/grouped_window_coupling_v1/session_20261027/run.source.zip)

result SHA256: `96a7af4bded9745fcc4f1506365f8c725e14c456b012f3cac201015f4ef2d0f5`.
source SHA256: `3a8f5107c10b56d3d4f203307c082a16f087316840385efa45ae8f730872d5bd`, 봉인 대상1,155파일과 `RUN_CONFIG.json` 1개.
archive를 별도 폴더에 풀고 frozen config를 사용하면 수치 결과를 재생할 수 있다.
`run`은 동일 출력 폴더 재사용을 거부하고 `audit`도 기존 audit 파일 덮어쓰기를 거부한다.
기존 폴더에서 재실행해 기록을 덮어쓰지 않는다.

## 9. 최종 검증 완료 기록

- 전체 pytest: **1,939개 통과**, 508.33초, 관측한 process exit code **0**.
  Windows/Python3.10.11 로컬 실행이다. 이번 변경의 원격 CI는 실행하지 않았다.
- source statement coverage: 85% (23,326문장,3,456미실행).
  새 grouped density와 window covariance core는 각각98%다.
  커버리지 수치는 수학적 완전성이나 모든 오류 부재를 증명하지 않는다.
- CI 범위 Ruff 통과. Mypy 224개 source/main 파일과 새 runner 별도 검사 통과.
- 4개 봉인 archive의4,597개 member/CRC 및4,593건 current/archive file-byte 대조 통과.
  기존 intervention·selected-extreme·local-covariance 결과의 runtime source를 변경하지 않았다.
- receipt 추가 전 strict JSON5개·문서 local link175개·공개 text13개를 검사했다.
  archive 경로 침범·비정상 JSON·검사한 credential pattern·100MiB초과 신규 text·
  trailing whitespace는 발견하지 않았다. 이 패턴 검사가 완전한 보안 감사를 대신하지는 않는다.
- RequestsDependencyWarning은 기존 환경에서 그대로 관측됐으며, 이를 숨기거나
  무시하기 위해 의존성을 바꾸지 않았다. 테스트는 이 경고와 별개로 모두 통과했다.
- 최종 receipt·문서 링크 추가 후 무결성 재검사 결과는 `final_integrity.stdout.txt`에 보존한다.

[검증 receipt](../../results/post_audit/grouped_window_coupling_v1/session_20261027/verification_receipt.json),
[전체 테스트 로그](../../results/post_audit/grouped_window_coupling_v1/session_20261027/pytest.stdout.txt),
[정적 검사 로그](../../results/post_audit/grouped_window_coupling_v1/session_20261027/static.stdout.txt),
[봉인 소스·문서 검사 로그](../../results/post_audit/grouped_window_coupling_v1/session_20261027/integrity.stdout.txt).

결론은 **구현 정합성 검사를 통과한 도구와 새 후보를 확보했지만,
이번 계산 채택 gate는 실패했고 실제 rare-event 성능 개선은 입증하지 않았다**는 것이다.
README를 이 상태로 갱신했으며, 커밋·푸시하지 않았다.
