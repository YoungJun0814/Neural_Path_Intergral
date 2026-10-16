# V2-P1 실행 결과와 P2 진입 판정

작성일: 2026-10-08. 근거: [통합 V2 계획](../plans/MODEL_STRUCTURAL_IMPROVEMENT_PLAN_V2_2026-10-08_KO.md), [P0 보고서](STRUCTURAL_V2_P0_EXECUTION_REPORT_2026-10-08_KO.md).

## 1. 결론

**P1의 runner·검증·새 pilot 실행은 완료했지만 G-REF는 미해결이다. P1 통과 조건을 충족하지 않아 P2는 실행하지 않았다.**

22개 작업이 계산 오류 없이 완료됐다. 그러나 충분히 정밀한 production reference를 만들기 위한 사전 allocation이 표본·potential·시간 상한을 초과했다. pilot 관측을 production으로 승격하거나, 정밀도·agreement 기준을 낮추거나, shared-guide 두 경로만 남겨 P2를 열지 않았다.

| 항목 | 상태 |
|---|---|
| 추정식·독립 단위·역할·source binding 구현 | 구현 및 관련 검증 완료 |
| 새 P1 pilot | 22/22 작업 완료 |
| pilot artifact 재감사 | binding·산술·역할 감사 통과 |
| 사전 production allocation | `unresolved_production_budget` |
| production μ/M₂ reference | `not_run` |
| relevant q별 G-REF | `unresolved` |
| P2 coupling·mesh 실행 | `not_run` — 사용자 지정 선행 조건 미충족 |
| 모델 성능 개선·저널용 confirmation | 실행하지 않음 |

테스트·감사 통과는 확인한 구현 조건의 근거다. 누락 tail의 부재, 독립 참값, 연속시간 정확성 또는 모든 오류의 부재를 인증하지 않는다. 커밋·푸시·cloud·패키지 설치는 하지 않았다.

## 2. 구현과 이론 검토

[reference 통계 모듈](../../src/path_integral/structural_v2_reference.py), [P1 runner](../../experiments/post_audit_v2_independent_reference.py), [사전 pilot config](../../configs/post_audit/structural_v2_p1_pilot_v1.yaml), [테스트](../../tests/test_structural_v2_reference.py)를 추가했다.

### 추정 대상을 분리했다

- 사건확률: μ_N=E_p[g_N]. static ordinary IS는 g·p/r, g-target SMC는 normalizer 자체를 추정한다.
- 고정 q의 raw second moment: M₂(q)=∫g²p²/q. static auxiliary IS는 g²p²/(q r), risk-SMC는 h=δ_q g²p/q의 normalizer를 **δ_q로 나누어** 추정한다.
- 10개 q의 서로 다른 M₂를 같은 참값처럼 합치지 않는다. μ는 같은 셀·격자의 공통 target이므로 두 셀에 별도 reference를 설계한다.
- final ordinary IS를 self-normalize하거나 payoff/weight를 clipping하지 않는다.

SMC의 SE 단위는 전체 SMC 실행이다. terminal particle이나 lineage를 IID 관측으로 세지 않는다. IID 전체 SE는 draw moments에서 계산하며, 별도의 동일 크기 block 간 변동·bootstrap은 민감도 진단이다. block SE를 전체 IID SE로 몰래 대체하지 않는다.

pCN MH acceptance에는 β(log h(candidate)−log h(current))를 사용한다. global independence MH는 정확한 full-mixture density ratio 보정을 추가한다. mutation/resampling/bridge 일정은 각 실행 전에 고정하며, reference 평균을 맞추는 방향으로 선택하지 않는다. 기존 weighted SMC 커널은 변경하지 않았다.

### 온도 convention과 호출 수

기존 helper의 `levels`는 bridge **구간 수**여서 `levels+1`개 점을 만든다. 새 계획은 **96개 온도 점**을 명시한다. 이를 혼용하지 않도록 새 adapter가 96개 점을 직접 만들며, 마지막 bridge transition에서는 기존 커널대로 mutation을 하지 않는다.

새 P1 whole run의 실제 호출 수는

`256 × [1 + (96−2)×8] = 192,768`이다.

계획의 194,816과 차이는 온도 점/구간 convention 때문이다. 기존 helper를 사용한 과거 `levels=96` 실행의 194,816은 그 **97개 점 convention에서는 올바르다**. 과거 비용을 오류로 재작성하지 않았다. 새 runner는 매 whole run에서 실제 호출 수와 192,768의 일치를 검사한다.

### pilot → production 경계

- 새 `allocation-pilot`·`audit` stream은 production `reference-iid`·`reference-whole-smc`와 분리된다.
- local-A/B 선택은 두 셀 중 나쁜 쪽 empirical CV²×whole-run 시간으로 한다. 다른 reference의 평균에 가까운 쪽을 선택하지 않는다.
- pilot은 production mean·SE에 합치지 않는다. production 결과를 본 뒤 count를 연장하지 않는다.
- raw pilot allocation은 외삽이다. 실행 가능한 경우에만 별도 production config를 봉인한다. IID count는 이때 동일 block·batch를 유지하도록 위쪽으로 반올림하고 sample/total budget을 다시 검사한다.
- production config의 전체 schedule·guide·endpoint·job grid를 pilot로부터 재생성하여 감사한다. 다른 endpoint로 바꾸어 통과시키지 못하게 한다.
- finite-grid development equivalence는 symmetric relative difference와 delta-method SE를 사용한다. risk 30개 비교와 μ 2개 비교는 각각 선언한 family다. 이 근사는 distribution-free coverage가 아니며 두 family를 합친 joint 95% 인증도 아니다.
- μ reference SE≤method SE/5는 특정 성능 비교 method가 생겼을 때 별도 확인한다. 이번에 그 자격을 확보했다고 표시하지 않는다.

## 3. 실제 실행과 원자료

[pilot 결과 JSON](../../results/post_audit/structural_v2_p1_pilot_v1.json), [실행 당시 source ZIP](../../results/post_audit/structural_v2_p1_pilot_v1.source.zip).

기존 고정 q: 두 개발 셀 × parent 5개. 새 학습은 하지 않았다. SMC pilot은 각 셀의 사전 고정 parent rep=0에서만 수행했다.

- M₂ IID: 고정 q 10개 × 새 65,536 draw.
- μ IID: 두 셀 × 별도 새 65,536 draw.
- risk SMC: 두 셀 × local-A/local-B/global-C × 전체 실행 8회 = 48회.
- μ SMC: 두 셀 × local-B/global-C × 전체 실행 8회 = 32회.
- 합계: IID 786,432 draw, whole SMC 80회, 작업 record 22개.
- 새 potential 평가 **16,207,872회**, path×step **518,651,904**.
- full-density component evaluation 계수 **2,620,887,040**. 이는 scalar density 호출 수와 다른 구성요소 단위의 작업 계수다.
- 실행 wall **427.45초(약 7.12분)**, sampled process peak RSS **748,331,008 bytes(약 713.7 MiB)**.
- seed stream **294개**, 명시 sample use **294개**, 공유 paired stream 0개.

wall은 runner의 과학 작업 구간이다. 입력 파싱·source 봉인 및 이후 tests/audit 전체 시간이 모두 포함된 end-to-end 배포 benchmark가 아니다. guide 구축용 simulator probe의 path×step은 각 row의 별도 계수로 남겼으며 potential 평가 합계에 숨겨 넣지 않았다. 기존 parent 학습 비용·pilot 비용을 새 모델 효율 이득으로 처리하지 않았다.

### parent rep=0의 pilot 값

**아래 평균은 reference 확정값이 아니다.** 다른 q의 M₂ 평균을 합치지 않는다.

| 셀 | estimand | 방법 | pilot 평균 | RSE |
|---|---|---|---:|---:|
| canonical | M₂(q₀) | static IID | 1.47754e−9 | 4.927% |
| canonical | M₂(q₀) | local-A | 1.47356e−9 | 11.521% |
| canonical | M₂(q₀) | local-B | 1.28053e−9 | 6.359% |
| canonical | M₂(q₀) | global-C | 1.57528e−9 | 5.179% |
| 높은 η | M₂(q₀) | static IID | 5.44735e−7 | 3.783% |
| 높은 η | M₂(q₀) | local-A | 6.11228e−7 | 8.240% |
| 높은 η | M₂(q₀) | local-B | 6.13510e−7 | 9.221% |
| 높은 η | M₂(q₀) | global-C | 6.03718e−7 | 2.745% |
| canonical | μ₃₂ | static IID | 5.13725e−8 | 10.897% |
| canonical | μ₃₂ | local-B | 5.23797e−8 | 6.420% |
| canonical | μ₃₂ | global-C | 5.35541e−8 | 4.595% |
| 높은 η | μ₃₂ | static IID | 5.16492e−6 | 4.110% |
| 높은 η | μ₃₂ | local-B | 4.47025e−6 | 2.059% |
| 높은 η | μ₃₂ | global-C | 4.76853e−6 | 4.734% |

전체 10개 q의 static IID M₂ RSE는 약 3.78–4.93%다. 이 범위는 사건확률의 RSE가 아니다.

### 혼합·기여 민감도

risk SMC의 initial ancestor 수는 canonical local-A 23–28, local-B 26–33, global-C 39–50; 높은 η는 local-A 30–38, local-B 32–44, global-C 43–54였다. μ SMC에서는 45–64 범위다. ancestor 수는 mixing/independence 인증이 아니며, 과거 다른 설정의 3–8과 단순 비교해서 방법 우위라고 주장하지 않는다.

canonical risk local-A의 leave-one-whole-run-out 평균 변화 최대값은 **7.64%**로, 사전 production 민감도 상한 5%보다 크다. 높은 η risk local-B도 약 **5.23%**다. 단지 계산량만 외삽하면 충분하다고 가정할 수 없다.

높은 η의 μ local-B RSE 2.059%는 낮아 보이지만 static IID와의 symmetric point difference는 약 **14.42%**다. pilot 불확실성까지 포함하면 10% equivalence가 성립하지 않는다. 이는 어느 방법의 bias를 증명하지 않지만 **낮은 sample RSE만으로 기준값을 선언하면 안 된다**는 구체적 근거다.

설명용으로 pilot static IID–local-B를 같은 delta 식에 넣으면 upper absolute difference는 canonical M₂ 약 39.45%, 높은 η M₂ 약 43.10%, canonical μ 약 30.29%, 높은 η μ 약 24.67%다. 이는 production gate를 실행한 결과가 아니라 pilot의 불충분함을 설명하는 탐색적 계산이다.

## 4. production 상한 판정

사전 target RSE 1.5%, safety factor 3, minimum whole SMC 32회, maximum 256회, maximum IID 8,388,608을 적용했다. noisy pilot 변동은 참값이나 보장된 production 정밀도가 아니다.

local schedule은 worst-cell precision/cost 규칙으로 **local-B**가 선택됐다. rep=0의 SMC 변동을 같은 셀의 다른 고정 q에 적용한 수치는 allocation forecast이며, 다른 4개 q의 실제 SMC 변동을 측정했다는 뜻이 아니다.

| 주요 forecast | 필요한 수 | 사전 상한 |
|---|---:|---:|
| canonical risk local-B / q | whole SMC 432회 | 256회 |
| canonical risk global-C / q | whole SMC 287회 | 256회 |
| 높은 η risk local-B / q | whole SMC 907회 | 256회 |
| 높은 η risk global-C / q | whole SMC 81회 | 256회 |
| canonical μ static IID | 10,379,264 draw | 8,388,608 |
| canonical μ local-B | whole SMC 440회 | 256회 |
| 전체 production potential | 1,767,353,600회 | 160,000,000회 |
| 전체 production wall 외삽 | 67,190.79초 ≈18.66시간 | 7,200초 =2시간 |

총량은 원래 pilot allocation forecast다. production 실행이 가능할 때 적용하는 equal-block 반올림 이전 값이며, 반올림은 추가 비용을 줄이지 않는다. 현재 allocation이 이미 미해결이므로 production config/source ZIP은 생성하지 않았다.

동일 precision을 확보할 예산이 없다는 것과 estimator가 수학적으로 잘못됐다는 것은 다르다. 현재 문제는 **참값 corroboration을 뒷받침할 정밀도·whole-run 안정성·기여 coverage 근거가 부족**하다는 것이다. 더 많은 sample이 평균 불일치를 반드시 해결한다는 보장도 없다.

## 5. 기술·이론 재검토와 검증

독립 검증에는 N=1 digital의 closed-form Gaussian 평균/pointwise formula, N=2 rough payoff의 별도 NumPy 수식 및 24/40차 Gaussian quadrature refinement를 포함했다. N=2 oracle의 ρ=0은 제한된 작은 사례이며 실제 64차원 rare-event 참값이 아니다. 생산 payoff와 같은 simulator를 호출해서 oracle이라고 한 것은 아니다.

추가 검사는 constant SMC normalizer, 실제 work 계수, log moment merge, 전체 SMC/동일 IID block 민감도, equivalence upper-bound, allocation 상한, pilot-only P2 잠금, moments/work/role/q-guide/normalizer/완료 grid/seed/unequal blocks/schedule 변조 거부다.

실행 당시 source/config를 ZIP에 보존했다. 실행 후 independent audit 및 production 경계 검사(equal-block count·전체 config binding)를 보강했다. 이 후속 코드 변경은 새 과학 표본을 만들거나 기존 결과/ZIP을 변경하지 않는다. 실행 당시 source와 현재 source가 동일하다고 주장하지 않는다. 재감사는 현재 코드를 이용한 binding·저장 moments 산술 검증이며 frozen simulator의 전체 numerical replay는 아니다.

관련 회귀 90개가 실행 전 통과했다. 최초 새 테스트 18개에서 이후 negative/production 경계 검사를 추가해 새 테스트는 22개다. **최종 전체 pytest 1,196개 모두 통과, 199.14초**였다. Ruff 전체 통과, mypy 173개 source file 통과, `git diff --check` 통과. requests dependency compatibility warning은 관측됐으며 설치 환경은 변경하지 않았다. 중복 실행한 targeted 테스트 수를 전체 unique test 수에 더하지 않는다.

## 6. 다음 우선순위 — P2가 아니라 P1의 제한 repair

1. **가이드 없는 SMC의 whole-run 변동을 먼저 줄인다.** 두 셀에서 한 번의 제한된 새 설계를 사전 고정한다. particle 수/bridge 수/mutation 수/pCN scale를 모두 동시에 무한 탐색하지 않고, 최대 두 matched-work 설계만 비교한다. whole SMC 수·actual calls·별도 pilot/final·wall cap을 봉인한다.
2. **평균 agreement와 rare contribution 탐색을 같이 본다.** leave-one-out/bootstrap, 독립 whole-run tail/모드 및 μ·M₂의 서로 다른 기여를 확인한다. ancestry·sample RSE 감소만으로 채택하지 않는다. guide-free 경로를 없애지 않는다.
3. **그 repair pilot 이후 새 allocation을 봉인한다.** 이전 pilot은 보존한다. 상한 이내에서 10개 q와 두 μ target의 production precision/equivalence를 달성할 것으로 판단되는 경우에만 새 production을 시작한다. 현재 약 18.7시간 외삽을 실제 필요한 시간의 확정값으로 사용하지 않는다.
4. **production G-REF를 통과한 범위에서만 P2를 연다.** 통과 셀을 다른 셀로 확장하지 않는다. 사용자가 요청한 전체 P1 조건을 충족하지 못하면 전체 P2 실행은 계속 보류한다.
5. 제한 repair 이후에도 부족하면 compute budget/개발 범위/reference 메커니즘을 재설계한다. 모델 차원·operator를 늘리거나 실패 셀을 몰래 제거하지 않는다. 외부 compute나 범위 변경은 별도 결정이 필요하다.

새 repair의 성능 성공은 보장하지 않는다. 이번 요청에서 이 별도 repair/큰 production을 자동 추가 실행하지 않았다. 계획의 중단·상한 규칙에 따라 결과와 다음 판단 근거를 먼저 남겼다.

## 7. 재현

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m experiments.post_audit_v2_independent_reference --audit results/post_audit/structural_v2_p1_pilot_v1.json
python -m pytest tests/test_structural_v2_reference.py -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
python -m pytest -q
```

이미 생성된 pilot output/ZIP은 덮어쓰지 않는다. fresh pilot 재실행은 새 config/output suffix와 snapshot 아래에서 한다. `--production-from`은 allocation이 미해결이면 실행을 거부하며, 현재 v1 pilot로는 production이 열리지 않는다.
