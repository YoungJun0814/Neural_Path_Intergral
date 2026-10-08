# V2-P1 단일 제한 repair 결과·재검토

작성일: 2026-10-08. [실행 전 봉인 프로토콜](../plans/STRUCTURAL_V2_P1_BOUNDED_REPAIR_PROTOCOL_2026-10-08_KO.md), [이전 pilot 보고서](STRUCTURAL_V2_P1_EXECUTION_AND_P2_GATE_REPORT_2026-10-08_KO.md).

## 1. 결론

**repair 구현·새 pilot·산술/역할 감사는 완료했지만, independent reference의 production allocation은 여전히 미해결이다. P2는 실행하지 않았다.**

입자를 늘리고 bridge/mutation 작업 배분을 바꾸는 두 near-matched-work local 설계가 뚜렷한 효율 개선을 입증하지 못했다. 이 결과는 SMC normalizer의 수학적 unbiasedness가 틀렸다는 뜻이 아니라, 현재 경로/커널/예산에서 정밀한 cross-mechanism reference를 확보하지 못했다는 뜻이다.

사전 기준을 낮추거나 guide-free 대조를 제거하지 않았다. 단일 repair 묶음이라는 중단 규칙에 따라 같은 셀의 schedule 탐색을 추가 실행하지 않았다. 모델 q·차원·operator·defensive floor는 변경하지 않았다. 커밋·푸시·cloud·패키지 설치도 하지 않았다.

## 2. 실행과 검증 범위

[새 결과 JSON](../../results/post_audit/structural_v2_p1_repair_v1.json), [실행 당시 source ZIP](../../results/post_audit/structural_v2_p1_repair_v1.source.zip), [config](../../configs/post_audit/structural_v2_p1_repair_v1.yaml).

- fresh whole SMC **192회**, IID **3,145,728 draw**, 작업 **24/24 완료**.
- SMC potential 36,683,776회, IID 3,145,728회, 종료 경로·조건부 payoff 진단 114,688회: 총 **39,944,192회**. 40,000,000 상한 이내.
- 실제 scientific runner wall **798.43초(약 13.31분)**. 3,600초 상한 이내.
- sampled process peak RSS **755,998,720 bytes(약 721.0 MiB)**.
- density component evaluation 계수 **5,831,892,992**. 단순 scalar density 호출 수가 아니다.
- seed stream/use **984개**, pairing 0개. 이전 pilot stream과 별도 config/protocol digest.
- 24개 결과의 source/config/q/guide/seed/whole-run moment/normalizer 변환/작업량/geometry 합계 재감사 통과.

추가 진단도 동일 target의 조건부 digital g를 평가하므로 비용에 포함했다. 종료 입자는 새 독립 inference unit이 아니며, 진단 표본 수를 whole run 수에 더하지 않는다. wall은 source 봉인·입력 파싱·이후 전체 tests/audit를 모두 포함한 end-to-end 배포 benchmark가 아니다.

## 3. parent rep=0 결과

아래는 **pilot 추정값**이며 production 기준값·confirmation 또는 성능 우위가 아니다. 다른 q의 M₂를 공통 참값으로 합치지 않는다.

| 셀 | estimand | IID RSE | local-A RSE | local-B RSE | global-C RSE |
|---|---|---:|---:|---:|---:|
| canonical | M₂(q₀) | 2.600% | 8.184% | 7.477% | 2.116% |
| 높은 η | M₂(q₀) | 1.918% | 8.226% | 6.055% | 3.199% |
| canonical | μ₃₂ | 5.494% | 4.873% | 3.276% | 3.688% |
| 높은 η | μ₃₂ | 1.713% | 3.057% | 3.635% | 3.600% |

주요 평균:

- canonical M₂: IID 1.46810e−9, local-A 1.39897e−9, local-B 1.67185e−9, global-C 1.55460e−9.
- 높은 η M₂: IID 5.98297e−7, local-A 7.24440e−7, local-B 6.27628e−7, global-C 6.18571e−7.
- canonical μ: IID 4.88497e−8, local-A 5.15285e−8, local-B 5.13717e−8, global-C 5.28058e−8.
- 높은 η μ: IID 4.62980e−6, local-A 4.85217e−6, local-B 4.68213e−6, global-C 4.72372e−6.

이전 pilot은 whole 8회·IID 65,536개였고 이번은 16회·262,144개다. RSE 감소만으로 algorithmic variance reduction을 주장하지 않는다. local 설계도 여러 work-allocation 항목을 함께 바꾸므로 입자 수 하나의 causal 효과를 분리하지 못한다. global-C 커널은 기존과 같으므로 새 RSE 차이를 repair의 새 방법 효과로 귀속하지 않는다.

## 4. 왜 아직 통과하지 못했는가

### local 비용·변동 점수는 사실상 동률

risk의 worst-cell empirical CV²×whole-run wall 점수는 local-A **0.28155**, local-B **0.28215**로 약 **0.22%** 차이다. 사전 exact-min 규칙으로 local-A가 선택됐지만, 이 정도 점차를 통계적·실무적 우위로 부르지 않는다. μ에서는 별도 같은 규칙으로 local-B가 선택됐다. reference 평균에 가까운 쪽을 선택하지 않았다.

### 필요한 reference 예산이 여전히 매우 크다

목표 RSE 1.5%·안전계수 3·기존 상한을 그대로 적용했다.

| forecast | 필요한 수 | 상한 |
|---|---:|---:|
| canonical risk local-A / q | whole 1,429회 | 256회 |
| 높은 η risk local-A / q | whole 1,444회 | 256회 |
| canonical risk global-C / q | whole 96회 | 256회 |
| 높은 η risk global-C / q | whole 219회 | 256회 |
| canonical μ IID | 10,559,488 draw | 8,388,608 |
| canonical μ local-B | whole 229회 | 256회 |
| 높은 η μ local-B | whole 282회 | 256회 |
| 전체 production 평가 | **3,166,747,136회** | 160,000,000회 |
| 전체 wall 외삽 | **73,871.07초 ≈20.52시간** | 7,200초 =2시간 |

이 외삽은 noisy pilot 기반의 계획값이지 실제 필요한 count/시간의 보장된 참값이 아니다. rep=0의 SMC 변동을 다른 고정 q에 적용한 부분은 forecast일 뿐 그 q의 실제 변동 측정이 아니다. IID equal-block 생산용 위쪽 반올림 전 값이며, 반올림이 비용을 줄이지 않는다. production config/표본은 생성하지 않았다.

### 입자 증가와 bridge 감소의 trade-off

새 local-A의 최소 incremental ESS fraction은 canonical risk에서 **8.3–12.5%**, 높은 η에서 **7.7–11.8%**였다. 그 최저 지점 β 중앙값은 약 **.00724**다. local-B의 risk 최소 ESS는 각각 **23.9–30.2%**, **19.0–25.9%**이고 β 중앙값은 약 **.00403**다. global-C는 약 49–59%/53–58% 범위였다.

이는 early bridge에서 큰 weight concentration이 생긴다는 관측이다. 입자 수를 늘린 대신 bridge/mutation 깊이를 줄인 설계가 그 어려움을 없애지 못했다. 단, 이 대조만으로 early ESS가 전체 변동의 유일 원인이라고 증명하거나 세 설정 변화의 causal 효과를 분리하지 않는다.

## 5. 기여 geometry에서 얻은 구체적 근거

새 [geometry 모듈](../../src/path_integral/structural_v2_terminal_geometry.py)은 peak time 4개 quartile × largest-left-variance share [.5,.9,.99] 경계의 **사전 고정 16구간**을 사용한다. fitted clustering·모든 경로 모드의 분해가 아니다.

canonical risk의 구간 6은 **peak가 만기의 25–50%에 있고 largest-cell share가 90–99%인 경로**다. 관측 M₂ 기여 비율은 local-A **40.97%**, local-B **41.79%**, global-C **40.70%**다. 세 방법 모두 이 거친 구간을 16개 whole run에서 관측했다. 따라서 주요 구간 자체의 완전한 미발견이 확인된 것은 아니다.

하지만 해당 구간 기여 SE를 전체 M₂ 평균으로 나눈 값은 local-A **6.57%**, local-B **4.22%**, global-C **1.50%**다. 전체 구간이 보인다고 기여량이 안정적으로 측정된 것은 아니다.

반면 canonical μ에서 peak가 마지막 quartile에 있고 share가 90% 이하인 구간 12+13은 기여의 약 **48.2–50.7%**를 차지한다. risk에서는 이 두 구간의 관측 기여 합이 약 **0.18–0.32%**다. weighted terminal largest-cell share도 risk 약 **.914–.918**, μ 약 **.558–.575**로 다르다.

높은 η에서도 risk의 weighted share는 약 **.966–.967**, μ는 약 **.760–.763**이었다. risk와 μ가 같은 경로 geometry를 요구한다고 가정할 근거가 없다.

비율은 `whole normalizer × terminal weighted mass`의 전체 실행 평균을 전체 평균으로 나눈 plug-in 진단이다. 비율 denominator의 불확실성을 포함한 CI가 아니며, coarse 구간 안의 다른 hidden mode·outside tail 누락을 배제하지 않는다. terminal ancestry/ESS와 particle count를 IID SE로 사용하지 않았다.

**연구적 의미:** event bank의 탐색 안정성을 raw M₂ reference의 탐색 안정성으로 전용하면 안 된다. 다음 reference 방법은 μ와 g²/q를 별도 대상으로 검토해야 한다. 이 진단 자체는 새 Volterra 효율 정리나 저널용 superiority 증거가 아니다.

## 6. 이론·기술 재검토

- μ, raw M₂, risk normalizer/δ_q 변환을 유지. q를 변경하거나 corrected M₂로 바꾸지 않음.
- 각 primary SE·bootstrap·모드 SE의 독립 단위는 whole SMC 또는 IID draw/block으로 구분.
- diagnostic partition이 data를 보고 이동하지 않음. 기여가 0인 구간을 missing-mode exclusion으로 쓰지 않음.
- natural-start guide-free local 대조를 유지. static IID와 global-C의 공통 guide 의존성을 공개.
- actual SMC·추가 진단 호출 수와 preset budget 일치. 실패 비용/부재를 성공으로 바꾸지 않음.
- 원래 pilot·JSON·ZIP을 보존. 새 stream·새 snapshot 사용.
- 실행 후 geometry partition/count/finite bound 감사 조건을 추가 보강했다. estimator/표본/저장된 과학 결과는 바꾸지 않았다. 실행 당시 ZIP과 현재 코드를 동일 source라고 주장하지 않는다.
- 실행 전 targeted 테스트 **26개 통과**. 최종 전체 pytest **1,200개 모두 통과, 199.14초**. Ruff 전체 통과, mypy **174개 source file 통과**, `git diff --check` 통과. targeted 실행 수는 전체 unique test 수에 더하지 않는다. requests dependency compatibility warning은 관측됐지만 패키지는 변경하지 않았다.

감사는 저장 moments·metadata·source binding의 재계산이며 모든 simulator 경로의 독립 numerical replay가 아니다. 테스트 통과가 전체 공간의 무오류·rare-tail coverage를 인증하지 않는다.

## 7. 다음 결정 — 이번 repair 이후 자동 추가 탐색 없음

1. **현재 schedule 계열의 tuning은 일단 중단한다.** 두 local 점수는 사실상 동률이고 production 예산이 크게 초과됐다. 더 큰 신경망·operator로 우회할 단계가 아니다.
2. **reference 메커니즘 변경의 별도 설계를 먼저 검토한다.** 예를 들어 Gaussian conditional Rao–Blackwellization은 후보이지만, μ와 원래 M₂(q)의 조건부 적분을 별도로 정의해야 한다. `g`를 조건부 평균으로 교체한 뒤 그 제곱을 원래 M₂라고 부르면 Jensen 차이로 estimand가 바뀐다. M₂ quadrature를 사용하면 수치 bias·Gaussian tail·비용을 별도 검증해야 하며, 아직 구현/성공한 방법이 아니다.
3. **새 메커니즘이 없다면 compute/연구 범위 결정을 분리한다.** 현재 약 20.5시간 외삽을 그대로 결제·실행 승인이나 성공 보장으로 쓰지 않는다. 더 큰 예산으로 production을 수행해도 agreement 실패가 사라진다는 보장은 없다.
4. **새 사전 계획·새 표본·G-REF 통과 뒤에만 P2.** 실패 셀 제거, margin 완화, 기존 관측의 confirmation 승격은 하지 않는다.

이번 요청에서 수행한 것은 한 번의 승인된 bounded repair와 재검토다. 다음 reference 방법의 선택·구현 또는 예산 확장은 별도 단계이며 자동 실행하지 않았다.

## 재현

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m experiments.post_audit_v2_independent_reference --audit results/post_audit/structural_v2_p1_repair_v1.json
python -m pytest tests/test_structural_v2_reference.py tests/test_structural_v2_terminal_geometry.py -q
python -m ruff check src tests experiments main.py train_driftnet.py
python -m mypy src main.py train_driftnet.py
python -m pytest -q
```

기존 output/ZIP은 덮어쓰지 않는다. 새 실행은 새 config/output suffix와 사전 snapshot을 사용한다. 현재 repair allocation으로 `--production-from`을 호출하면 생산 실행을 거부한다.
