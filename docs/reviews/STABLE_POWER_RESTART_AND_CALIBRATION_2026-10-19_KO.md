# 전원 안정 확인 후 재실행·2-thread 교정 결과

2026-10-19. 앞선 실패를 보존하고 실행한 후속 단계의 종료 기록이다.
현재 complete-grid accuracy, P2 또는 대규모 학습의 통과 보고가 아니다.

## 1. 원래 고정 seed의 수치 기록1개 추가

사용자의 전원 안정 확인 후 AC/battery31%(실행 시작30%)를 확인했다.
[새 사전 고정 계획](../plans/STABLE_POWER_FIXED_GRID_RECOVERY_2026-10-19_KO.md)에 따라
원래22개 complete를 SHA/source/seed/count 검증 후 재사용하도록 했다.
남은10개 중 첫 high-η/more-temperatures/rep6의 계산을 완료했다.

- physical forward129,280/joint backward64,640회, wall366.4193초.
- 원래32개 seed 중 numeric complete record는 이제23개다.
- 해당 whole의 power_before30%,power_after26%로 사전 고정한 감소 허용1%p를 초과했다.
- 수치 record를 먼저 저장한 뒤 `power_failure.json`으로 중단했다. 새 result.json은 없다.
  기존 두 실패와 이번 운영 실패를 덮어쓰지 않았고 prefix 통계도 산출하지 않았다.
- 이1개 기록의 `check_unit`, source/archive/current critical runtime hash, checksum,
  고정 membership 및 비용 검사는 통과했다. full-grid statistical qualification이나
  numerical whole replay 통과를 의미하지 않는다.

데이터: `results/post_audit/p2_fine_resource_stable_recovery_v2/session_20261019`.

## 2. 단순히8thread를 재시도하지 않고 저부하 조건을 검증

[2-thread 교정 계획](../plans/LOW_POWER_BACKEND_CALIBRATION_2026-10-19_KO.md)을
결과 생성 전에 고정했다. 원래 high-η/N4096의 동일128개 입력에서8thread baseline과
2thread의 components/family log potential/capped gradient는 기존 허용오차
rtol5e-10/atol1e-8 안에서 일치했다. 같은 입력을2thread로 반복해 부하를 유지하면서
helper thread의 WMI query로 전원 상태를 관측했다. 희귀 사건 성능 표본은 생성하지 않았다.

| 항목 | 관측값 |
|---|---:|
| 완료 callback | 72 (8thread baseline1+2thread warmup1+stress70) |
| 실제 완료 Volterra physical forward | 18,432 |
| 실제 완료 joint backward | 9,216 |
| runner 내부 phase wall (CLI import 이전 시간 제외) | 124.0075초 |
| 부하 구간의 보고 battery percent | 24% 유지 |
| 처음 보고 RemainingCapacity | 18,558mWh |
| 마지막 보고 RemainingCapacity | 18,272mWh |
| 보고 용량 변화 | -286mWh |

용량 순서는18,558→18,511→18,431→18,272→18,272mWh다.60초 연속 부하와 그 전후의
telemetry를 포함한 관측이며, 이 차이를 정확히60초의 정밀 방전율로 환산하지 않는다. WMI는 모든 관측에서
PowerOnline=true/Charging=true/Discharging=false로 보고했다. 따라서 **충전 표시와
보고 용량의 추세가 맞지 않는다**. 이는 어댑터 불량의 확정 진단도,2thread 계산이
용량 감소의 유일한 원인이라는 실험적 증명도 아니다. 센서/펌웨어/갱신 지연 등도
배제하지 않았다. 다만 사전 고정한100mWh 감소 허용 조건을 통과하지 못했으므로
후속 장시간 실행을 승인하지 않았다. 결과에 맞춰 이 기준을 완화하지 않았다.

교정은 power gate에서 종료됐다. 계획된4차원 full-SMC cross-backend toy와
실제 Volterra whole의 cross-backend replay는 **미실행**이다.
고정 입력 수치 동등성을 전체 sampler 동등성/효율/accuracy 인증으로 확대하지 않는다.
failure.json에 source/config/raw power samples/완료 callback 수와 보수적 reservation을 보존했다.

데이터: `results/post_audit/low_power_calibration_v1/session_20261019/failure.json`.

## 3. 연구 계산 종료 후 진단

연구 Python 계산이 끝난 상태에서도20:03:59→20:04:04(미 동부 시간)의
RemainingCapacity는17,953→17,937mWh였다. 짧은 구간의 진단이며 정밀 전력 측정이나
지속적인 방전율 추정은 아니다. 브라우저·Docker·System 등 다른 프로세스의 CPU 활동도
관측했으나 그 활동을 원인으로 확정하지 않았다. 다른 프로그램을 종료하거나
OS power policy/critical battery action을 변경하지 않았다.

## 4. 완료한 코드와 검증 범위

- 새 복구 loader는 원래22개 기록과31개 evidence binding을 확인한다. 새 테스트9개와
  이전 복구 테스트4개, explicit Mypy가 통과했다.
- G0/G1 pilot는 아직 표본이0개다. 실제 첫 실행용 계획/안정전원 guard와 새 full-grid
  result+4whole audit receipt의 선행조건을 구현했다. runner2tests와 explicit Mypy 통과.
- 저부하 전원·수치 비교 helper2tests 및 explicit Mypy 통과. power telemetry의 missing,
  용량 감소, unknown 상태를 pass로 바꾸지 않는다.
- 소프트웨어 검사와 source 무결성은 위 운영 실패를 통과로 바꾸는 근거가 아니다.
  P1/P2/large-training flags는 여전히false다.

## 5. 다음 조건

장시간 계산 전에 실제 보고 용량이 회복·유지되는지 확인하거나, 사용자 확인과 추가
진단으로 이 센서 보고가 부정확한지 규명해야 한다. 충전 입력/어댑터·포트와 다른
고부하 작업의 동시 실행 여부를 확인할 필요가 있다. 사용자 소유의 다른 프로세스를
임의로 종료하지 않는다.

조건이 해결되면 source·기존 실패 비용을 보존한 새 저부하 protocol을 고정하고,
고정 입력뿐 아니라 whole-level backend replay를 수행한 뒤 원래 grid를 완성한다.
그 전에는 G0/G1 실제 대규모 비교나 새 모델 학습을 시작하지 않는다.
