# 유한격자 연구 재설정: FR0–FR2 구현·고정 실행·중단 결정

2026-10-28 America/New_York. 이 문서는 최초 로드맵의 상태표를 대체하는 새 실행 보고서다.
기존 protocol·실패·P2 threshold를 소급 변경하지 않는다.

## 1. GitHub 및 이번 범위

먼저 기존 P2 구현·실패 기록·새 로드맵을 main `51e665d705c049795f59755d1a950716f2833be5`에
커밋·푸시했고 원격 SHA 일치를 확인했다. 해당 커밋의
[GitHub CI](https://github.com/YoungJun0814/Neural_Path_Intergral/actions/runs/40804098936)는 성공했다.

푸시 전 발견한 Git 줄바꿈 정규화 문제는 `.gitattributes`에서 frozen stdout의 producer bytes를
보존하도록 보완했다. 191개 staged 파일은 raw/index object hash가 전부 일치했다.
5개 신규 archive의5,856개 entry에서 위험한 경로와 제한된 credential 패턴을 검사했으며 발견하지 못했다.
이는 모든 비밀정보 부재의 보증이 아니다.

그 뒤의 FR0–FR2 구현·결과는 별도 후속 작업이다. 위 원격 CI는 이 후속 작업의 CI가 아니다.

## 2. 실제 구현한 기반

- `finite_grid_geometry_protocol.py`: 원래 법칙·payoff·task digest, role/seed, 확인 task 접근 guard,
  phase 예산, 후속 실행의 누적 소비 인계, 전 사이클 재설계1회 계약.
- `finite_grid_geometry_benchmark.py`: 같은 g_N의2N conditional MC/RQMC, bounded put/K,
  task/cache binding, inverse-CDF endpoint 거부, 저장 통계 재계산.
- `finite_grid_geometry_controls.py`: 기존 정규화 mixture·fitting을 연결한 bank×family 대조,
  known-mass 층화 covariance, 두 risk reference의 paired contrast.
- 두 실행기: reference-only feasibility와 별도 봉인 bank/family 실험·대표 replay.
- config: D01–D04 개발과 C01–C04 확인 task를 분리. **확인 task 물리 평가0회**.

기존 3N rank-one smoothing RQMC를2N 전체 가격 조건부 baseline으로 이름만 바꾸지 않았다.
digital과 put/K를 같은 bounded 계약으로 구분했고, put 학습에 digital을 대신 쓰지 않았다.
일반 IS 항등식과 기존 Gaussian mixture를 새 수학 원리로 부르지 않는다.

개발 중 JSON 직렬화의 tuple/list 왕복 문제와 테스트 fixture의 범위 오류를 찾아 수정했다.
실험 source를 봉인하기 전에 관련 검사를 다시 통과시켰다. 기존 실험 결과를 고친 것이 아니다.

## 3. FR1: 알려진 정답과 음성 대조

기존 residual Gaussian oracle, finite-sum marginal risk, bounded bank 및 genealogy 검증을 재사용했다.
새 task의 cached/full evaluator·미사용 마지막 residual·거의 상수 변동성 Black–Scholes limit을 대조했다.

추가 1D bounded two-bump 예에서는, 작은 event mass 영역이 defensive q의 M2를 지배할 수 있음을
독립 적분으로 확인했다. 이 예는 통계 코드의 음성 대조이며 새로운 일반 정리나 Volterra 발견이 아니다.
독립 quadrature의 보고 오차와 잘린 Gaussian tail bound도 구분했다.

FR0/adapter·기존 oracle을 포함한90개 검사, 새 protocol/adapter/control54개 검사가 통과했다.
known-answer 검사 통과는 제안 기하의 성능을 보장하지 않는다.

## 4. FR2 reference-only pilot

후보를 학습하지 않고 사전 선언한 각 task에서 MC32×8192와 RQMC32×8192를 실행했다.
총2,097,152회 payoff 평가, 계측 phase wall24.14초다. startup/archive 전체를 포함한
통제된 속도 benchmark는 아니다. RQMC의 통계 단위는32개의 독립 scramble이다.

| task | payoff | MC mean | MC 관측 RSE | RQMC mean | RQMC 관측 RSE |
|---|---|---:|---:|---:|---:|
| D01 | digital | .07879417 | .394% | .07879888 | .116% |
| D02 | digital | .02597498 | .710% | .02582085 | .284% |
| D03 | digital | .01236169 | 1.065% | .01247079 | .705% |
| D04 | put/K | .02106404 | .469% | .02090748 | .178% |

관측 RSE1.25%와20% approximate agreement margin의 **개발 feasibility**를 네 셀 모두 통과했다.
대상은 finite-grid μ_N이고 D04는 확률이 아니라 정규화 가격이다. 이 평균을 정확한 참값으로 쓰지 않는다.
같은 물리 simulator/CDF를 공유하므로 이 일치만으로 공통 구현 오류나 모든 missing tail을 배제하지 못한다.

대표8unit을 같은 source/seed로 재생했고65,536회를 추가 사용했다. 결과 SHA256:
`76e4eb61e1729611aac5a8f7e0a1603fcccf175c859f61d57e4748b1ca7d4385`.
재생 표본은 새 독립 기준 표본으로 더하지 않았다.

## 5. FR2 bank × family: 실제 결과

[사전 계약](../plans/FINITE_GRID_FR2_BANK_FAMILY_PROTOCOL_2026-10-28_KO.md)에 따라
D02 digital과 D04 put/K에서 각각 새 전체 부모 학습3회, 총6whole를 실행했다.
각 whole에서 bank2개×family2개를 만들고 부모/static과 함께6arm을 새 IID final32,768개로 평가했다.
성공한 부모를 골라 재사용하거나 불리한 whole를 제외하지 않았다.

아래는3whole의 직접 final RSE 중앙값이다. 독립 risk reference RSE와 다르고,
참 RMSE·최적 필요 표본 수·학습비 포함 speedup을 뜻하지 않는다.

| arm | D02 | D04 |
|---|---:|---:|
| 새 parent K2 | .836% | .513% |
| static guide | 1.344% | 1.057% |
| CE bank / K2 유지 | .776% | .392% |
| CE bank / K1 압축 | .716% | .363% |
| static bank / K2 유지 | .813% | .475% |
| static bank / K1 압축 | .765% | .412% |

모든36개 final은 사전5% 개발 precision을 통과했다. 이것만으로 accuracy를 인증하지 않는다.
이번에는 **위험 평가가 비교 자체를 막는 상황도 아니었다.**

| 위험 평가의 최대 관측 RSE: 셀 내 모든 arm·whole | D02 | D04 |
|---|---:|---:|
| H: parent/static known-mass mixture | 2.30% | .79% |
| P: 별도 자연 Gaussian | 3.50% | 1.15% |

24개 대비 모두 risk precision10%와 H/P20% agreement 진단을 통과했다.
그러나 최소 관심 효과10% M2 감소와 다중성 조정 근사 lower 조건까지 통과한 반복 수는 다음과 같다.

| 사전 방향의 대비 | D02 /3 | D04 /3 |
|---|---:|---:|
| CE bank K2가 K1보다 우수 | 0 | 0 |
| static bank K2가 K1보다 우수 | 0 | 0 |
| K2에서 CE bank가 static bank보다 우수 | 1 | 2 |
| K1에서 CE bank가 static bank보다 우수 | 0 | 0 |

동일 대비가 두 task 모두에서2/3 이상이어야 한다는 조건을 충족한 것은 없다.
최종 판정은 **`STOP_NO_REPLICATED_BANK_OR_FAMILY_MECHANISM`**다.
방향을 뒤집어 K1 승리를 사후 확인 성공으로 세거나10% 기준을 낮추지 않았다.

학습 bank event-weight ESS는 D02 CE약1095–1246/static586–610,
D04 CE약2177–2213/static874–900이었다. 과거 극단 셀의 작은 ESS와 달리 수백~수천이었지만,
이것이 미관측 tail의 부재나 구조 효과를 증명하는 것은 아니다.

새 physics2,052,096회, phase wall56.46초. 대표 whole2개의 source/seed/fit/final/risk/판정을
동일하게 재생했고684,032회를 추가 사용했다. 모든 replay 값이 일치했다.
결과 SHA256: `ae24f53131c7e3dd20a1327fd3139a74a411a0befb4bd37b3e2d865af27b0573`.

pilot·본 실험·재생을 합친 FR2 실제 payoff 평가는 **4,898,816회**다.
예산8백만 안이며 재생을 독립6whole에 추가하지 않는다. 테스트·source 봉인·문헌 검토는 별도 연구비다.

## 6. 해석과 다음 gate

이번의 진전은 **비교 가능한 문제와 위험 평가를 확보해, 현재 가설을 실제로 반증할 수 있게 된 것**이다.
모델 성능 우위나 새 이론의 확보가 아니다.

확인한 범위:

1. moderate finite-grid task에서는 검증기 문제가 항상 전체 연구를 막지는 않는다.
2. 이 family와 task에서 K2 보존이 K1 압축보다 좋아야 한다는 가설은 지지되지 않았다.
3. CE bank가 static bank보다 일부 유리해도, 두 task의 반복 기준을 넘는 보편적 자료 효과는 확보하지 못했다.
4. 복잡도·성분 수 확대가 지금의 정당한 다음 행동이라는 근거가 없다.

결론 내리지 않는 것:

- 과거 K=1 극단 셀의 압축·탐색 문제가 모두 없다는 주장.
- 모든 K1이 모든 mixture보다 우월하거나 Volterra 기하가 무의미하다는 주장.
- event/risk 목적 또는 residual covariance의 population 우열. 이번에는 해당 FR3 실험을 하지 않았다.
- 같은 표본 수의 낮은 RSE를 학습비 포함 실제 speedup으로 바꾸는 주장.
- 유한격자에서 성공했다고 연속시간 bias·시장 예측·production을 보장한다는 주장.

**FR3의 구조 확대·FR4 대형 반복·FR5 확인은 시작하지 않는다.**
남은 재설계1회를 쓴다면 먼저 관측된 사실에 맞는 다른 원인 가설과 novelty 경계를 작성해야 한다.
예를 들어 단순 CE/K1이 strong conditional RQMC보다 실제 workload에서 가치가 있는지는 별도 비용 질문이다.
이것은 그 자체로 새 모델이 아니며, 더 큰 네트워크를 넣어야 한다는 근거도 아니다.
현재 중단을 숨기고 새 kernel로 자동 이동하는 행동은 하지 않았다.

## 7. 검증 상태와 재현 경로

- 새 source/계약·config는 실행 전에 봉인했다. 대표 replay 시 current/source archive의 일치를 검사했다.
- seed/task/law·학습자료·final·risk reference 역할을 분리했다. C01–C04는 평가0회다.
- 두 study 결과·개별 whole·통계/결정과 대표 physics replay를 감사했다.
- Ruff 통과. CI 범위 Mypy233개 source/main 파일과 두 runner의 직접 범위 검사가 통과했다.
- runner 직접 검사는 `--follow-imports=silent` 범위다. legacy experiment를 전이 검사하면 기존3개 파일의
  typing 문제8개가 드러났고, 이를 새 numerical core 문제로 오인하거나 기존 source를 몰래 수정하지 않았다.
- 전체 회귀는 **2,045 passed / 450.19초 / 실패0**, 별도의 전체 workflow 통합 검사는
  **1 passed / 40.49초**다. 이를 한 번의2,046개 전체 회귀 실행으로 부르지 않는다.
- 통합 검사는 별도 validation namespace를 사용해342,016회 payoff를 사용했다.
  연구 whole6개나 mechanism signal에 더하지 않았다. FR2 실험·감사와 이 검증을 합친 알려진 소계는
  **5,240,832회**이며, 기타 단위 테스트 내부 physics 호출은 이 소계 밖이다.
- 합친 coverage는 source85%(23,993 statements /3,497 missed), 새 core3개 합계93%다.
  protocol91%/benchmark94%/controls94%이며 coverage가 모든 오류 부재를 뜻하지 않는다.
- 원본 source archive2개·2,356entry를 검사했고2,354개 frozen-file/current-byte 비교가 일치했다.
  새 source snapshot 이후 추가한 통합 테스트는 과거 archive에 포함된 검사로 소급하지 않는다.
- 두 study의424개 role seed에 충돌이 없었다. 제한된 credential 패턴과 archive 경로 검사도 통과했다.
- 기존 RequestsDependencyWarning은 남아 있고 전역 Python 환경을 업그레이드하지 않았다.
- 자세한 명령·JUnit hash·artifact binding은 [검증 receipt](../../results/post_audit/finite_grid_bank_family_v1/session_20261028/verification_receipt.json)에 기록했다.

새 gate·이론 식과 코드 검사에서 발견한 오류는 수정했지만, 모든 입력의 무오류나 tail coverage를 보장하지 않는다.

실행:

```powershell
python -m experiments.finite_grid_geometry_mechanism_study
python -m experiments.finite_grid_geometry_mechanism_study --audit
python -m experiments.finite_grid_geometry_bank_family
python -m experiments.finite_grid_geometry_bank_family --audit
```

기본 출력 경로는 non-overwrite다. 이미 완료된 디렉터리에 처음부터 실행하면 거부한다.
위 명령의 처음 두 줄을 다시 실행해 새 표본으로 합치지 않는다. source가 바뀐 뒤에는 frozen archive에 맞는
별도 replay 환경이 필요하며, current-tree digest 검사는 그 불일치를 숨기지 않기 위한 것이다.

결과 위치:

- [reference-only 결과](../../results/post_audit/finite_grid_geometry_reset_v1/session_20261028/result.json)
- [reference 재생 감사](../../results/post_audit/finite_grid_geometry_reset_v1/session_20261028/audit.json)
- [bank/family 결과](../../results/post_audit/finite_grid_bank_family_v1/session_20261028/result.json)
- [whole 재생 감사](../../results/post_audit/finite_grid_bank_family_v1/session_20261028/audit.json)
