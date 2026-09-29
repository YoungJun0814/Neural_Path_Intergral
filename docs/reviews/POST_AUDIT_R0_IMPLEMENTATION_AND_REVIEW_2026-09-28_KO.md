# 감사 기반 R0 구현·이론·기술 검토 보고서

작성일: 2026-09-28

범위: [감사 기반 실행계획](../plans/POST_AUDIT_RESEARCH_EXECUTION_PLAN_2026-09-27_KO.md)의 R0.1–R0.6
판정: **R0의 검증 기반을 구현했다. 기존 V16의 성능 우위 주장은 새 기준에서 미확정이다.**

## 1. 구현 결과

| 작업 | 구현 내용 | 결과와 남은 범위 |
|---|---|---|
| R0.1 출처·역할 계약 | source commit/dirty/patch digest, runtime source-tree digest, config digest, runtime, task·payoff·proposal fingerprint, 역할별 seed ledger | 새 결과 형식에 적용. 부동소수점의 기계 간 동일성까지 보증하지 않음 |
| R0.2 감사기 | binding의 재귀 해시 검증, JSON/YAML 중복 키 탐지, raw cluster moment 재집계, 원본 표시값·비용비 재계산, 독립 unit 구별, 변조 테스트 | 새 결과는 검증. V16 과거 기록은 산술 일관성만 확인 가능 |
| R0.3 CE 비교기 | `2N` local 좌표, conditional Gaussian CDF, `p/q` 가중 elite 및 target 적합, mean shift와 선택적 low-rank covariance, 정확한 defensive mixture 밀도 | 방법 정의와 작은 oracle 검증 완료. 강한 baseline 성능 검증은 R1/R3 과제 |
| R0.4 공통 자격 판정 | candidate와 comparator에 동일한 RSE, reference precision, 사전 equivalence margin 기준 적용 | 새 결과 schema에 적용. V16의 오래된 비교 출력은 historical/legacy로 남음 |
| R0.5 비용 | offline/fit/selection/inference/reference별 wall·CPU·process peak memory·proxy 기록 | smoke의 timing은 계측 경로 점검용. 정밀한 실제 시간 우위는 아직 측정하지 않음 |
| R0.6 이론 | T16-12B의 양측 union bound 계수 수정, 미완성 연속정리의 주장 상태를 별도 ledger에 명시 | finite-grid 항등식은 유지. T16-5/9/11의 공백은 해결되지 않았음 |

기존 실험 결과와 원래 YAML ledger의 과거 기록은 그대로 보존했다. 현재 V16 최종 감사 엔트리를 다시 실행하면 새로운 legacy 검토 결과를 붙이고 `finite_grid_empirical_claim_authorized=false`로 처리한다. 출력도 기존 파일이 아닌 `results/post_audit/legacy_v16_reaudit_v1.json`에 기록하도록 했다.

R0 smoke는 커밋 전에 실행됐으므로 출처 기록에 `source_dirty=true`와 당시의 patch digest가 남는다. 별도의 runtime source-tree digest는 Git staging 상태와 무관하게 소스·실험 코드·설정 파일 내용으로 계산된다. 최종 커밋의 해당 내용과 비교할 수 있지만, 이 소규모 기록을 이미 커밋된 소스에서 수행한 공식 confirmation이라고 표현하지 않는다.

## 2. 감사 결과: 과거 주장과 새 기록의 관계

[과거 결과의 7개 셀](V16_REORIENTATION_EVIDENCE_WITH_OPERATOR_2026-09-27.json)의 bound artifact들을 새 adapter로 순회해 설정·결과·하위 baseline/reference 파일의 SHA-256을 확인했다. 셀 이름 중복, 평가 cluster 수, 평균, pooled/between-cluster SE, robust RSE, reference z, work-normalized variance, 각 baseline 대비 proxy 비율을 다시 계산했다.

재계산된 100-query proxy 비율은 canonical 5.978, `K=2` 1.280, `K=0.5` 5.023, regular-H 1.815, 높은 η 0.471, 강한 음의 ρ 0.605, joint extreme 0.639이다. 4개 셀의 점추정 proxy 우위와 3개 fallback이라는 기존 산술은 유지됐다. 그러나 old CEM은 unweighted 3N hard-event fit이고 V16은 2N conditional payoff를 사용한다. variant당 독립 proposal fit도 하나다. 따라서 새 ledger는 이 기록을 **역사적 개발 결과**로 분류한다.

기존 comparator JSON 일부에는 비표준 `Infinity`가 기록돼 있다. 새 post-audit JSON은 이런 값을 거부한다. legacy adapter에서만 읽기 위해 해당 양의 무한대 sentinel을 허용했으며, 이를 통계적 자격의 근거로 사용하지 않았다.

새 감사기는 저장된 `passed`나 표시된 성능비를 믿지 않는다. SHA가 일치하는 입력이라도 in-memory 표시값을 바꾸면 재계산과 다르다는 이유로 거부한다. `NaN` ratio, 큰 accuracy/normalization z, 오래된 RSE, 중복 method/evaluation key, task·dimension·config mismatch, 역할별 seed 재사용, 음수 비용과 방어 하한 위반을 다룬다.

한계도 명확하다. cluster의 `count/mean/m2`가 실제 표본에서 나왔는지를 외부 감사기가 혼자 증명할 수는 없다. source/config/proposal·seed manifest와 재실행 가능한 경로를 제공하고, R3에서는 독립 실행 및 선택적 원시 표본 replay로 이를 보강해야 한다. 해시 검증은 파일 무결성이지 실험의 진실성을 증명하는 장치가 아니다.

## 3. 조건부 CE의 수학적 검토

학습 표본 `Z_i ~ q_t`에서 현재 단계의 soft target을 적합할 때 `w_i=g_N(Z_i)p_N(Z_i)/q_t(Z_i)`를 쓴다. event까지 가는 중간 elite 단계는 `1_{A_t}(Z_i)p_N(Z_i)/q_t(Z_i)`를 쓴다. 가중 평균의 분모 `Σw_i`는 Gaussian **학습 파라미터**를 계산하는 정규화다. 최종 확률 추정은 `M^{-1}Σg_N p_N/q`이며 self-normalization을 사용하지 않는다.

제안 분포는 자연 표준 Gaussian과 학습 Gaussian의 positive mixture다. 고정 covariance-rank 옵션은 학습 방향의 covariance eigenvalue를 명시된 구간으로 제약한다. 실제 sampling과 밀도 계산에 동일한 covariance·가중치를 사용한다. 모든 성분의 밀도를 log-sum-exp로 계산하기 때문에 label별 단일 밀도를 분모로 쓰는 오류를 피한다. 자연 성분의 질량 `δ>0`는 `p_N/q≤1/δ`를 보장한다.

1차원 표준정규 오른쪽 tail에서 weighted fit의 평균·분산을 truncated-normal 식과 비교했고, 작은 Gaussian covariance와 mixture likelihood는 기존 독립 dense-oracle 테스트로 확인했다. 한 local 경로를 고정하고 독립 가격 noise를 반복했을 때 hard-event 평균이 conditional CDF와 부합하는지도 확인했다.

남은 제한:

- `g_N`가 machine float64에서 지수화되며 0으로 underflow될 정도의 극단 tail은 별도 log-domain 집계가 필요하다. 현 R0 smoke는 그런 영역의 성능 보장이 아니다.
- 중간 elite 선택은 최종 `g_N p_N` target이 아니라 **기록된 다른 단계 target**이다. target ESS가 충분할 때만 soft target fit으로 전환하고 `target_reached`를 보존한다.
- ESS threshold, covariance rank·shrinkage, defensive mass는 사전 지정된 비교 조건이어야 한다. 이번 smoke의 값을 성능 최적값으로 해석하지 않는다.
- small Gaussian mixture CE는 R1에서 mode 누락이 확인될 경우에만 추가한다. R0에는 자연 성분+학습 성분의 정확한 혼합을 구현했다.

## 4. 통계와 비용의 검토

새 schema는 IID path, randomized QMC scramble, 독립 SMC 전체 실행을 서로 다른 inferential unit으로 명시한다. QMC 점이나 resampled SMC particle을 독립 반복으로 계산하면 `raw_sample_count = unit_count × points_per_unit` 검사에 걸린다. 각 cluster의 centered second moment는 Chan 병합식으로 집계한다. `SE=0`인 zero-hit 결과는 RSE를 null로 두고 reference equivalence를 자동 승인하지 않는다.

새 공통 qualification은 estimator와 reference의 relative SE를 각각 확인하고, 차이의 구간이 사전에 정한 상대 margin 안에 들어가는지 검사한다. 작은 z 하나로 정확성을 선언하지 않는다. 이 절차 자체도 heavy-tail finite-sample coverage의 엄밀한 보증은 아니다. `qualification=unresolved`인 방법은 표에서 숨기지 않고 비교 결론을 보류한다.

총 method 비용에는 offline, fit, selection, final inference의 측정 시간을 더한다. reference는 공통 비용으로 별도 기록한다. process peak memory는 각 단계에 시점별로 읽은 **프로세스 생애 최대치**이며, 그 단계의 증분 메모리가 아니다. proxy work는 디버깅·기존 결과와 연결하는 보조 필드다. R0 smoke에는 warmup과 power mode가 계측돼 있지 않으므로 wall-time speedup을 주장하지 않는다.

고정 wall-time이 끝날 때 평균 계산을 멈추면 시간이 표본값과 연관될 때 편향이 생길 수 있다. 이후 성능 시험은 독립 calibration에서 표본 수를 정한 뒤 정해진 수를 완료하고 소요 시간을 측정해야 한다.

기록 형식 검사를 위해 `N=4`, `K=80`, 자연법칙과 weighted conditional CE를 같은 bounded terminal payoff에 적용했다. [재실행 가능한 smoke 결과](../../results/post_audit/r0_smoke_v1.json)의 독립 reference는 `0.131972 ± 0.003333`(표준오차), 자연법칙은 `0.114696 ± 0.006778`, CE는 `0.123975 ± 0.002701`이다. 두 방법은 각각 3개 cluster·768개 IID final path를 사용했고, CE는 2회·회당 512개 학습 표본을 사용했다. 내부 CE 단계에서 soft target 적합 ESS 기준에 도달했다고 기록됐다.

work-proxy variance 비율은 자연법칙/CE `2.700`으로 계산됐다. 해당 실행의 fit 포함 wall time은 자연법칙 약 `0.0112초`, CE 약 `0.0310초`였다. 짧은 실행에 warmup·power mode 통제가 없고 독립 학습 반복도 하나이므로 **proxy 개선이나 이 wall-time 차이 어느 쪽도 성능 결론으로 사용하지 않는다.** 새 감사기의 integrity·semantic validity는 통과했으며 `performance=unresolved`다.

## 5. 이론 정정의 검토

기존 selector가 사용한 Maurer–Pontil형 empirical-Bernstein 식의 `log(2J/γ)`는 한쪽 방향의 오차를 J개 후보에 합칠 때의 상수다. 기존 문서의 양측 simultaneous event와 oracle-excess 식에는 두 방향이 필요하므로 코드와 정리 노트를 `log(4J/γ)`로 수정했다. 상수 payoff·자연 proposal oracle 테스트에서 반경의 계수를 직접 확인했다.

bounded payoff, 후보 고정, IID validation, `Q_j≥δ_jP`, `G≥δ_GP`, 최소 2개 표본이 정리의 적용 조건이다. 실제 위험 상계가 매우 크면 certificate는 vacuous로 남긴다. V16 최종 route는 이 selector를 사용하지 않았으므로 기존 셀의 추정값은 이 상수 변경으로 재평가되지 않는다.

[새 주장 상태표](../theory/POST_AUDIT_CLAIM_LEDGER_2026-09-28.md)는 T16-5의 ε 의존 conditional payoff/Itô 항의 Laplace 연결, T16-9의 stochastic quadratic variation과 uniform joint grid 제어, 이를 사용하는 T16-11의 일부를 `review pending`으로 명시한다. 이 공백에 반례가 발견된 것은 아니다. `proved`라고 기록된 과거 YAML만으로 새로운 논문에서 그 결론을 사용하지 않는다.

## 6. 테스트 및 판정 기록

| 검사 | 결과 |
|---|---|
| R0 적대적·oracle 테스트 | 최종 수정 후 20개 통과 |
| 전체 저장소 회귀 테스트 | 1,061개 통과 (`pytest -q --cov=src --cov-report=term-missing`) |
| Ruff | `src tests experiments main.py train_driftnet.py` 통과 |
| Mypy | `src main.py train_driftnet.py` 및 수정한 experiment 모듈 통과 |
| CI형 coverage 명령 | 최종 코드에서 1,061개 통과, 전체 `src` line coverage 84% |
| 두 방법 공통 schema smoke | 새 감사에서 integrity·semantic validity 통과; performance는 unresolved |

테스트는 유한 입력에서 구현과 산술을 확인한다. T16-5/9/11의 새로운 완전 증명이나 R3의 통계적 우위를 대신하지 않는다.

## 7. R0 gate와 바로 다음 작업

R0의 범위인 **증거 계약, 공정한 조건부 비교기의 시작점, 이론 주장 정정**은 구현·검증했다. 이로써 R1의 작은 원인 진단을 시작할 수 있다. 그 전에 R0가 현재 모델의 우월성을 입증했다고 해석하면 안 된다.

R1에서는 기존 성공·실패 셀 6개를 개발 대상으로 삼고, 첫 3개에서 rank/basis, KL 대비 second-moment 목적, 독립 SMC bank의 mode, 실제 fitting/density 시간을 분리한다. 새 weighted conditional CE와 V14/FIS를 같은 조건부 payoff와 예산에서 비교한다. 5개의 독립 학습 seed는 원인 탐색용이며 최종 실패율 입증이 아니다.

과학적 미해결점은 모두 남겨둔다: 새 CE의 큰 N/deep tail 경쟁력, 유한격자 mesh bias, proof review pending, 정밀한 실제 시간, 여러 task의 독립 학습 변동. R1 증거 없이 모델 성분을 추가하지 않는다.
