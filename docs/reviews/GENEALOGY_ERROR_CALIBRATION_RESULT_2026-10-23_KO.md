# 계보 진단과 실제 오차: 정답 기반 검증 결과·방향 판단

2026-10-23 America/New_York.
[사전 계획](../plans/GENEALOGY_ERROR_CALIBRATION_2026-10-23_KO.md)에 따른 synthetic 진단 실험이다.
기존 source/result/gate를 바꾸지 않았고 commit/push는 하지 않는다.

**핵심 결론:** 계보·nominal ESS·4회 반복 RSE는 경고 도구이지 정확도 인증서가 아니다.
기존 gate를 단순히 완화할 근거도, 통과만 하면 맞다고 말할 근거도 얻지 못했다.
실험 판정은 `CALIBRATION_COMPLETE_NO_RETROACTIVE_ADOPTION`이며 현재 Volterra STOP은 유지한다.

## 1. 무엇을 검증했는가

1D에서 p=q=N(0,1), 목표분포 h를 알려진 두 Gaussian mixture로 정했다.
`m2=Z*h/p≤1`이 되게 amplitude Z를 analytic Gaussian-ratio bound로 정했다.
그러면 기존 risk bridge의 endpoint는 정확히 Z*h이고, Z와 target moment를 closed form으로 안다.
g=sqrt(m2)로 두면 bounded conditional second moment인 예가 되지만 실제 금융 geometry는 아니다.

| 문제 | h의 means / weights / sd | 정확한 Z | E[X] | E[X²] | P(X>0) |
|---|---|---:|---:|---:|---:|
| Moderate symmetric | ±1.5 / 0.5,0.5 /0.7 | 0.0771073 | 0 | 2.74 | 0.5 |
| Rare symmetric | ±4 / 0.5,0.5 /0.5 | 1.165455e-5 | 0 | 16.25 | 0.5 |
| Rare asymmetric | -5,+6 /0.7,0.3 /0.5 | 6.282313e-11 | -1.7 | 28.55 | 0.3 |

별도 scalar adaptive quadrature는 tiny Z로 scale한 raw moment를 적분했다.
세 문제 모두 통과했고 최대 analytic 대조 오차는1.421e-14였다.
adaptive error proxy는 최대7.956e-11로, 비외향 반올림의 한계는 남는다.
tiny absolute error만으로 성공 처리하지 않았다.

각 문제32개 독립 replicate, 총96개를 실행했다.
기존 SMC core의128particle/32linear stage/기존3move/resampling index를 유지했다.
q=p이므로 여기서 component pCN은 prior pCN과 같으며,
이 실험은 이전 nontrivial-q oracle의 대체물이 아니다.

대조는 replicate당 독립 p-IID12,416개로 ordinary mean m2를 사용했다.
SMC와 potential context 수는 같지만 FLOP/wall 동일성을 주장하지 않는다.
IID의 normalized target moment는 SNIS 진단으로서 finite-N biased다.
Z 자체는 self-normalize하지 않았고 실제 event estimator는 변경하지 않았다.

## 2. 먼저 확인된 이론적 경계

### 조상 하나라도 final moment가 정확할 수 있다

동결된 과거 H가 final weights w, lineage, Zhat를 결정한다고 하자.
각 particle를 exact π에서 독립적으로 refresh하면
`E[Σw_i f(X_i')|H]=π(f)`,
`Var[Σw_i f(X_i')|H]=Varπ(f) Σw_i²`다.
lineage와 Zhat는 변하지 않는다.
따라서 **큰 initial ancestor ESS는 final moment 정확도의 이론적 필요조건이 아니다.**
또한 exact refresh 후에도 Zhat는 그대로여서, final 상태가 좋아져도 이미 누적한 normalizer가
고쳐지는 것은 아니다. 여기서는 exact target sampler를 알기에 가능한 반례다.
현재 Volterra에 이 sampler가 있거나 바로 사용할 수 있다는 뜻이 아니다.

component별 midpoint quantile를 결합한 constructed cubature에서도,
모든 lineage를0으로 표시해 ancestor ESS≈1이지만 세 문제의 target moment 기준은 통과했다.
이는 IID 표본/실제 SMC 결과가 아니며 normalizer는 공식으로 제공한 값이다.

### 조상이128개여도 중요한 모드를 놓칠 수 있다

왼쪽 component만128개 midpoint quantile로 만든 별도 bank는
uniform weight/ancestor128/nominal ESS128이었다.
하지만 true P(X>0)=0.5 또는0.3을 재현하지 못했고 모든 target-accuracy 기준에 실패했다.
movement trace를 조작해 full gate를 통과한 실제 SMC처럼 보고하지 않았다.
이는 genealogy/weight 지표만으로 target coverage를 보증할 수 없다는 구성 반례다.

## 3. 실제 반복 결과: bank와 normalizer의 정확도를 분리

사전 label은 |mean오차|≤0.25, |second오차|≤max(0.5,0.1*truth),
|오른쪽 질량오차|≤0.10을 모두 만족하는지를 뜻한다.
Z label은 상대오차≤20%다. 이는 유한 bank의 tolerance label이지 coverage 확률/오류0 인증이 아니다.

| 문제 | 기존 full gate 통과 /32 | SMC target label 통과 /32 | SMC Z label 통과 /32 | Exact-refresh target label 통과 /32 |
|---|---:|---:|---:|---:|
| Moderate symmetric | 32 | 30 | 32 | 30 |
| Rare symmetric | 19 | 7 | 15 | 9 |
| Rare asymmetric | 0 | 0 | 4 | 11 |

exact-refresh는 원래 weight/lineage/logZ/full-gate 표시를 그대로 유지했다.
weighted128개라 exact target draw라 해도 finite-N moment가 항상 tolerance를 통과하지는 않는다.
예를 들어 희귀 대칭 target의 분산은16.25여서 이상적인 uniform128 IID 평균도
표준오차 약0.356이며, 평균오차0.25 기준은 엄격하다.
따라서 모든 label 실패를 구현 오류/완전한 모드 누락으로 해석하지 않는다.
새 seed로 반복해 통과율을 올리는 추가 실행은 하지 않았다.

### 희귀 대칭 문제에서 gate 통과가 오차를 막았는가?

gate를 통과한19개 중 target label 통과는4개, 실패는15개였다.
그15개는 모두 mean 기준을 넘었고, 통과19개 중11개는 오른쪽 질량오차0.10도 넘었다.
Z 상대오차20%를 넘은 통과 bank는11/19였다.
반대로 gate가 reject한13개 중 target label에 맞은 bank3개와 Z label에 맞은7개도 있었다.
이는 **이 고정1D stress test와 사전 tolerance에서의 confusion count**다.
실제62D에서의 false-positive 확률/학습된 classifier 성능은 아니다.

희귀 비대칭에서는 gate가32개를 모두 reject했고 실제 target label도 모두 실패했다.
exact refresh만 추가하면 같은 reject 표시에서11개가 target label을 통과했다.
따라서 경고가 실제 어려움을 잡는 경우도 있으나, final moment에 대한 필요조건이 되는 것은 아니다.

ancestor ESS와 target error score의 Spearman 상관은 moderate0.004,
rare symmetric-0.342, rare asymmetric-0.656이었다.
Z 상대오차와는 각각-0.061,+0.240,+0.291로 달랐다.
기술적 순위 상관일 뿐 유의성/일반화/인과적 기전의 증명이 아니다.

## 4. 작은 RSE도 정답이 아니었다

32개 normalizer의 ordinary mean 결과:

| 문제 / 방법 | 32회 평균 Z의 실제 상대오차 | 독립 반복 RSE | 개별 target label 통과 /32 |
|---|---:|---:|---:|
| Moderate / SMC | 0.033% | 0.512% | 30 |
| Moderate / IID-IS | 0.149% | 0.261% | 32 |
| Rare symmetric / SMC | 12.52% | 5.89% | 7 |
| Rare symmetric / IID-IS | 10.20% | 14.81% | 1 |
| Rare asymmetric / SMC | 7.24% | 28.38% | 0 |
| Rare asymmetric / IID-IS | 89.94% | 34.26% | 0 |

평균 Z가 맞아 보여도 개별 bank가 target을 잘 나타낸다는 뜻은 아니다.
rare asymmetric SMC의32회 평균 오차가7.24%여도 RSE28.38%이고 최대 반복 질량23.25%였다.
이를 통계적으로 안정된 성공이라고 부르지 않는다.

사전에 정한 연속4replicate씩8group의 기존 normalizer gate도 확인했다.

- Rare symmetric SMC: gate 통과6/8group 중1개는 실제 Z오차29.95%, RSE16.14%였다.
- Rare symmetric IID-IS: 통과3/8group 중2개는 실제 Z오차24.39%/25.65%였다.
- Rare asymmetric IID-IS: 통과1/8group의 실제 Z오차는 **96.21%**인데 RSE는 **15.58%**였다.

마지막 사례는 작은 반복 변동이 심각한 과소추정을 걸러내지 못할 수 있음을 보여준다.
표본이 놓친 희귀 기여는 표본 RSE에 충분히 드러나지 않을 수 있다.
이 결과로 현재 Volterra에서도 같은 오류율이 발생한다고 수치적으로 추정하지는 않는다.

## 5. 이번 결과가 연구 방향에 주는 판단

1. **기존 gate 완화/폐지는 채택하지 않는다.** 낮은 계보가 필수 실패 조건은 아니라는 이유만으로
   현재 실패 bank를 승인하면 반대의 false-confidence 문제를 해결하지 못한다.
2. **bank gate와 normalizer/reference gate를 분리한다.** 실제 fitting bank에는 genealogy/weight/
   feature coverage의 경고를 유지하고, normalizer의 타당성은 독립 기준으로 확인한다.
3. **RSE만 낮추는 것이 다음 목표가 아니다.** 서로 다른 sampling geometry의 추정량,
   기여 구간별 질량·variance, fresh 반복이 동일 estimand를 지지하는지 확인해야 한다.
4. **다음 구현은 새 operator나 대규모 학습이 아니다.** 고정된 미탐색 잔차57을 구간별로
   직접 샘플링하는 독립 stratified-IS reference를 우선 검토한다.
   [다음 단계 설계](../plans/RESIDUAL_STRATIFIED_REFERENCE_2026-10-23_KO.md)에 식/가정/예산/중단 기준을 구체화했다.
   2026-10-26 검토에서는 defensive bound로도 희귀 M2의 유용한 상대 정확도가 자동으로
   인증되지 않는 점을 추가했다. 후속 pilot의 RSE·상호 비반증은 개발 진단이지 tail 인증이 아니다.
5. **논문 기여는 아직 입증 전이다.** 이 진단은 잘못된 성능 선언을 막는 기반이다.
   새 proposal의 population 효율/독창성 또는 최상위 저널 수준의 결과가 생긴 것은 아니다.

## 6. 계산·감사·기술 검토

생산 potential contexts2,383,872(SMC1,191,936+IID1,191,936), exact-target draws12,288,
Volterra physical0, scalar reference evaluations6,330이었다.
source freeze5.685초/reference0.220초/compute+artifact I/O29.569초,
end-to-end37.046초(마지막 receipt 직전)였다.
회귀검증과 일부 겹치며 독점 host benchmark가 아니다.

기존 source는 수정하지 않고 새 [진단 core](../../src/path_integral/genealogy_error_calibration.py),
[실행기](../../experiments/post_audit_genealogy_error_calibration.py),21개 사전 테스트를 추가했다.
실행 전 JSON roundtrip 일치를 검사해 tuple/list 차이를 통일했고 import 검사를 수정한 뒤 봉인했다.
frozen numeric source/원본을 소급 수정하지 않는다.

독립 reference/구성 반례와96개 replicate **전체 full replay** 감사가 통과했다.
모든 최종 상태·정규화상수·진단 label·confusion matrix·hash chain·seed와 집계가 일치했다.
감사 potential contexts2,383,872/exact-target draws12,288/wall31.502초였다.
생산+감사 합계 potential contexts **4,767,744**, exact-target draws **24,576**,
Volterra physical **0**, scalar reference evaluations **12,660**이다.
unit/full regression 계산은 위 연구 표본 예산과 별도다.
[전체 재생 감사](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/audit.json).

Ruff CI 범위, source220개 파일의 Mypy 및 새 실행기1개 별도 Mypy는 통과했다.
local Windows/Python3.10 검증이며 새 remote Linux/Python3.11 CI 결과가 아니다.
알려진 Requests dependency warning은 보존했고 dependency는 변경하지 않았다.
전체 회귀검증은2026-10-23 로그가93%에서 끝나고 종료 요약/exit code를 확인하지 못했다.
2026-10-26 재개 시 기존 session이 없고 일치하는 pytest process도 확인되지 않아,
원래 로그를 보존하고 별도 로그로 전체 재검증을 시작했다.
원인은 미확인이며 전원/OS 오류라고 추정하지 않는다. 기존 미완료 실행은 pass로 세지 않는다.
2026-10-26 새 회귀검증은 **1,849개 통과**,410.27초로 종료했다.
source statement coverage는85%(22,972 statements/3,443 missed), 새 진단 core는97%(112/3)다.
statement coverage는 모든 branch나 이론적 오류 부재의 증명이 아니다.
재개 후 Ruff/Mypy도 위와 같은 범위로 다시 통과했고 dependency 버전도 동일함을 확인했다.
[중단·재개 기록](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/validation/regression_resume_20261026.json).
[최종 회귀검증 receipt](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/validation/regression_review.json),
[완료 테스트 로그](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/validation/pytest-resumed-20261026-output.txt),
[재개 후 정적 검사 로그](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/validation/static-resumed-20261026-output.txt).

공개파일 검토에서382개 파일/strict JSON315개/로컬 링크220개와
9개 archive의 member10,170개를 확인했다. current 동결파일1,166개 hash도 일치했다.
credential pattern/위험 archive 경로/100MiB 이상 파일/trailing whitespace 검출은0이었다.
이는 secret 부재의 exhaustive 보장이 아니다.
당시 완료된 JSON/validation stdout324개에서 Git filter 이후 producer byte 보존을 확인했다.
[무결성 receipt](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/validation/integrity_review.json)에
진행 중 stdout을 제외한 원래 검토 범위를 기록했다. 재개 검토와 완료 로그의 최종 closure는 별도 기록한다.

2026-10-26 재개 시 원래9개 anchor/archive와 동결파일1,166개,96개 artifact를 다시 확인해
변경0을 확인했다. 추가 공개파일 검토에서는386개 파일/strict JSON317개/로컬 링크222개,
archive member10,170개와 완료 artifact의 Git blob328개를 확인했고 검출 오류는0이었다.
원래 미완료 pytest 로그의 SHA도 그대로다.
[재개 무결성 기록](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/validation/integrity_resume_20261026.json)에
완료 로그·새 receipt의 최종 closure 범위를 연결한다. commit/push는 하지 않았다.

[생산 결과](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/result.json),
[정답·구성 반례](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/reference.json),
[end-to-end 비용](../../results/post_audit/genealogy_error_calibration_v1/session_20261023/end_to_end.json).

Result SHA256: `30f93470c1118881240b5845336107b1e97ff9af62ac5226d88688e80d3ed638`.
Source ZIP SHA256: `0c3538b8b5086cdd35d317e1a58a05dd57aa3bd6897d63e8ca6bbc80664353d0` (1,140files).
Audit SHA256: `a269c3a7324dc94fa83caf771758d056ec00005ca63d273c901ef7ff239347f8`.
