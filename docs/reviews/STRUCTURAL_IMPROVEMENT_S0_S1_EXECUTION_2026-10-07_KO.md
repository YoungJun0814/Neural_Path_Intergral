# 구조 개선계획 S0/S1 실행 및 검토 보고서

작성일: 2026-10-07

유일한 실행 범위: [구조적 문제와 개선계획](MODEL_STRUCTURAL_REVIEW_AND_IMPROVEMENT_PLAN_2026-10-07_KO.md)

상태: S0/S1 구현·pilot·전체 부모 학습 5회 개발 진단. S2–S6 완료나 성능 입증을 뜻하지 않는다.

## 1. 결론

평가 파이프라인과 표현 family를 개선했지만, **효율 개선을 입증하지 못했다.** 현재 가장 중요한 변화는 낮아 보이는 pilot 분산을 최종 성능으로 오인하거나, 평가 상한 때문에 미완료된 후보를 모델 실패로 단정하지 않게 된 것이다.

새 부모 학습 10개(두 셀 × 5회), 후보 30개 중 23개는 allocation이 후보별 524,288개 상한을 초과했다. 나머지 7개는 봉인된 표본 수로 독립 최종 평가를 완료했다. 그중 방법 RSE 5% 조건을 만족한 것은 1개였지만, reference 정밀도와 equivalence까지 포함한 전체 qualification은 **0/30**이다. 23개를 통계적 실패로 세지 않는다. 상태는 `unresolved_sample_budget`이다.

따라서 fixed-precision 비용 우위, bank 문제가 해결됐다는 주장, 모델 차원 확대, confirmation 확대를 승인하지 않는다. 새 operator나 추가 이론을 임의로 붙이지 않았다.

## 2. 이번 구현

### S0 — 실험 계약

- 부모 학습부터 다시 시작하는 `parent_training_rep`와 `refit_rep`, `selection_rep`, `final_batch`, historical reference 역할을 구분했다. 실제 selection은 없으므로 `selection_rep=null`이다.
- 설정별 digest를 seed namespace에 넣어 첫 throughput pilot과 이후 5회 학습이 겹치지 않게 했다. 각 부모·bank·allocation·최종 표본은 다른 stream이다.
- 실제 실행 전 source/config/tests/의존성 선언/개선계획을 ZIP으로 저장한다. archive 전체와 파일별 SHA256을 기록한다. 결과 파일은 runtime source digest에서 제외한다.
- 과거 artifact의 digest와 결과를 덮어쓰지 않았다. 현재 소스가 이후 변경되더라도 당시 ZIP이 재현 기준이다. ZIP의 존재는 외부 재현 성공을 뜻하지 않는다.
- `unresolved_sample_budget`, `unresolved_reference_precision`, `insufficient_final_precision`, `equivalence_not_established`를 분리했다. 실패한 부모의 실행시간과 사용한 potential 평가도 보존한다.
- offline, fit, allocation/selection, inference 비용을 구분한다. 공통 부모 비용은 각 배포 알고리즘 비용에 포함하고, 공유 실행의 물리적 시간은 별도로 기록한다. 메모리는 주기적으로 관측한 process RSS이며 엄밀한 전체 peak가 아니다.
- README에서 V16 기록을 historical work-proxy 결과로 구분하고 현재 frontier를 post-audit 개발 단계로 바로잡았다.

### S1 — 세 가지 표현의 분리

| 후보 | 변경 | 유지하는 계약 |
|---|---|---|
| parent-as-is | SMC bank에서 학습한 부모 그대로 | natural 성분과 학습 mixture 전부 |
| same-family-refit | 같은 IID bank로 학습 성분 평균·상대 weight를 KL 재적합 | 성분 수, identity covariance, natural 질량 0.1 |
| single-shift | 같은 IID bank로 단일 shift KL 적합 | natural 질량 0.1, identity covariance |

현재 부모는 학습 성분 2개다. 재적합은 covariance/rank를 늘리지 않는다. 단일 shift에는 전체 64차원 평균 방향을 허용했으므로, rank 축소까지 동시에 섞은 비교가 아니다. 단일 성분과 두 성분의 capacity가 같다는 주장은 하지 않는다. 이 비교의 목적은 family 압축 손실 진단이다.

같은-family fitting은 초기 부모를 포함한 empirical KL loss 최저 iterate를 반환한다. 이것은 최종 M₂나 비용을 개선한다는 보장이 아니다. bank에서 정규화한 target weight는 fitting 전용이고, 최종 추정은 full-mixture likelihood를 이용한 ordinary IS이다.

모든 후보를 먼저 고정한 후, 각 후보의 새 pilot 65,536개로 표본 수를 봉인했다. 샘플 CV²로 `n≈2 CV²/0.05²`를 계산하고 8,192개 배수로 올림했다. 계수 2는 안전 여유이며 통계적 upper confidence bound가 아니다. 최종 결과가 나쁘다고 추가 표본을 연장하지 않았다.

### 작은 위험 potential 검증

`log h = log δ + 2 log g − log(q/p)`를 구현했다. 고정 q가 `q≥δp`, `0<g≤1`이면 `0<h≤1`, `E_p[h]=δ M₂(q)`다. 상수 payoff의 SMC normalizer와 1차원 Gaussian-CDF payoff의 독립 quadrature를 검증했다.

실제 rBergomi 후보의 risk-SMC 평가와 보정 mixture는 아직 실행하지 않았다. 이번 구현은 이론적 변환의 oracle 검증이며, 위험 평가의 효율성·신규성을 입증한 것이 아니다. nonfinite log값과 defensive bound 위반은 clipping 없이 거부한다.

## 3. 새 실험 결과

### 최초 throughput/allocation pilot

- 새 부모 2개, 후보 6개.
- potential 평가 760,320회, 물리적 실행시간 약 17.0초.
- canonical 부모만 상한 안에서 최종 212,992개를 평가했다. 평균 `4.24749e−8`, RSE **5.2713%**. 목표 5%와 reference SE 비율 조건을 만족하지 못했다.
- 다른 5개는 표본 수 상한 262,144개를 초과해 최종 평가하지 않았다.

### 새 전체 부모 학습 5회/셀

- 총 potential 평가 **5,014,016회**, 물리적 실행시간 **79.43초**.
- 시간에는 모델 학습·bank·pilot·최종 계산을 포함한다. Python import, source ZIP 생성과 결과 저장 I/O, 공통 historical reference 생성은 제외한다. 전력/CPU clock/warmup을 통제한 publication benchmark가 아니다.
- CPU 1 torch thread, 개발 셀 N=32, conditional dimension=64.
- 부모당 SMC island 3개 × 256 particles, 동일한 기존 bridge/resampling/mutation 설정. 부모마다 처음부터 난수를 새로 사용했다.
- 재적합 bank는 각 부모에서 뽑은 4,096개 새 IID 표본이다. target-weight ESS 범위 **1.14–52.16**. 이것은 학습 집중도의 진단이며 iid 표본 수의 대체값이나 최종 SE가 아니다.
- SMC ancestry는 canonical island별 10–17, 높은 η 19–29였다. 상관된 SMC bank의 ancestry와 새 IID 재적합 bank의 ESS를 혼동하지 않는다.

아래는 pilot에서 봉인한 필요 표본 수의 **중앙값**이다. 최종 precision이 입증된 sample complexity가 아니다.

| 셀 | 부모 그대로 | 같은-family 재적합 | 단일 shift |
|---|---:|---:|---:|
| canonical | 638,976 | 442,368 | 4,153,344 |
| 높은 η | 1,048,576 | 1,941,504 | 4,317,184 |

canonical에서는 mixture 보존이 압축보다 유리한 탐색 신호가 있다. 높은 η에서는 재적합의 계획 비용도 부모보다 커져, family 보존만으로 문제를 해결할 수 없다는 신호다. 학습 반복 5개와 불안정한 pilot만으로 우위 확률을 추정하거나 일반화하지 않는다.

최종 평가를 완료한 7개:

| 셀 / 부모 반복 | 후보 | 최종 표본 | 평균 | RSE |
|---|---|---:|---:|---:|
| canonical / 1 | 부모 그대로 | 163,840 | 5.30468e−8 | 14.42% |
| canonical / 1 | 같은-family | 401,408 | 4.27009e−8 | 7.94% |
| canonical / 3 | 부모 그대로 | 466,944 | 5.12715e−8 | 14.95% |
| canonical / 3 | 같은-family | 442,368 | 3.92909e−8 | 3.79% |
| canonical / 3 | 단일 shift | 221,184 | 4.20921e−8 | 7.20% |
| canonical / 4 | 같은-family | 335,872 | 4.07059e−8 | 7.64% |
| 높은 η / 3 | 같은-family | 245,760 | 3.50988e−6 | 7.34% |

모두 equivalence qualification에 실패했다. canonical / 3 same-family의 낮은 RSE만으로 성공이라고 판단하지 않는다. historical reference와 차이가 있고 reference 불확실성 조건도 만족하지 못했다. 이것은 bias의 증명도 아니다. tail 누락·표본 변동·reference 불확실성을 독립 위험 평가로 분리해야 한다.

특히 canonical 부모 반복 1과 3은 pilot RSE가 각각 5.50%, 9.38%였지만, 더 많은 새 최종 표본에서 RSE가 14.42%, 14.95%가 됐다. 단순 CV allocation의 tail-risk 한계를 보여주는 개발 증거다. 안전계수를 올리기만 하면 해결된다고 가정하지 않는다.

## 4. 기술적·이론적 재검토

- mixture likelihood는 기존 exact-density 구현을 그대로 사용한다. covariance가 identity인 새 fitting 밀도를 독립 dense Gaussian 식과 대조했다.
- natural mass 0.1을 고정했으며, bank fitting weight와 최종 importance weight를 분리했다.
- allocation과 최종 표본이 독립이고 후보는 미리 고정된다. 봉인된 n을 전부 완료한 경우의 ordinary IS 불편성 계약을 유지한다. 시간 초과의 부분 결과는 진단용이며 qualification하지 않는다.
- random stream은 역할·task·부모 반복·batch·설정으로 나뉜다. 전체 학습 5회는 개발 관문이며 신뢰성/실패율 증명을 뜻하지 않는다.
- source archive/config/reference/proposal binding, batch moment의 독립 재계산, allocation 올림, 정확도 플래그, cost 합, 누락 반복과 stale qualification을 감사한다. 이 감사의 통과는 통계적 성능의 인증이 아니다.
- 새로운 유한격자/연속시간 효율 정리, Gaussian Volterra 구조의 novelty, market calibration은 이번 변경으로 해결되지 않았다.

## 5. 다음 관문 — 원래 계획 안에서만

1. **S2 bank 대조:** 1×768 / 3×256 / 6×128 island를 같은 potential 평가 예산으로 사전 봉인한다. 모델 family는 유지한다. ancestry뿐 아니라 독립 위험·최종 기여·실제 비용까지 비교해야 한다.
2. **독립 위험 진단:** 고정 부모와 재적합 q에 대해 whole-replicate risk-SMC를 실행하고 direct IS second moment와 대조한다. 불안정하면 oracle로 취급하지 않고 unresolved로 남긴다. event-bank 대조 없이 risk 보정의 우위를 주장하지 않는다.
3. **reference 정밀도와 allocation:** 기존 reference SE는 비교 estimator SE의 1/5 이하 조건을 충분히 충족하지 못한다. 필요한 독립 reference budget과 다른 메커니즘의 cross-check를 먼저 봉인한다. 샘플 수 확대는 성능을 보장하지 않는다.
4. **진행 조건:** 최소 두 방법이 동일 accuracy 자격을 얻기 전에는 fixed-precision wall-time 우위와 새 confirmation을 생산하지 않는다. bank 또는 위험 진단에서 개선 신호가 없으면 같은 셀의 무한 튜닝을 중단한다.

이후 S3 20회·S4 미사용 task·S5 이론/novelty·S6 원고 방향은 해당 관문을 통과한 뒤 진행한다. 이번 단계에서 그 조건을 건너뛰지 않았다.

## 6. 산출물

- [실행기](../../experiments/post_audit_r2_family_diagnosis.py), [감사기](../../experiments/post_audit_r2_family_audit.py)
- [mixture 보존 fitting](../../src/path_integral/proposal_family_diagnostics.py), [위험 potential·allocation](../../src/path_integral/conditional_second_moment.py)
- [oracle/프로토콜 테스트](../../tests/test_r2_improvement_contract.py)
- [최초 pilot 설정](../../configs/post_audit/r2_family_diagnosis_pilot_v1.yaml), [5회 학습 설정](../../configs/post_audit/r2_family_diagnosis_five_fit_v1.yaml)
- [최초 pilot 결과](../../results/post_audit/r2_family_diagnosis_pilot_v1.json), [5회 학습 결과](../../results/post_audit/r2_family_diagnosis_five_fit_v1.json)
- 각 결과 옆 `.source.zip`에 실행 당시의 immutable content snapshot을 보관했다.

커밋과 푸시는 하지 않았다. 검증 결과의 최종 기록은 아래 실행 확인 항목을 참고한다.

### 실행 확인

- 새 oracle/프로토콜 검사: 10개 통과(감사기 변조 거부 검사 포함).
- 전체 Ruff: 통과.
- 전체 CI 범위 mypy: 168개 소스 파일 통과.
- 두 artifact의 source snapshot·산술 감사: 통과. 최초 2부모/6후보/최종1, 후속 10부모/30후보/최종7.
- 전체 회귀 pytest: **1,089개 통과**, 약 163초. 이후 감사기 정확도 재계산·변조 거부 검사를 강화한 변경은 관련 테스트 10개를 다시 실행해 통과했다.
- 전역 무오류 보장이나 top-journal 게재 가능성의 입증은 아니다.
