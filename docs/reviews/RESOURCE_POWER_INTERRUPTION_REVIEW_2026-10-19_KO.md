# Resource-v2 / 고정-seed 복구: 전원 중단과 무결성 검토

2026-10-19. 원래 실패를 덮어쓰지 않는 종료 기록. 현재 새 대형 계산은 실행하지 않는다.

## 실제 완료·실패 내역

| 실행 | 완료 | 중단 | 집계/자격 |
|---|---:|---|---|
| resource-v2 | 20/32 whole | high-η/more-temperatures/rep4, wall1745.756초>600초 | result.json 없음, prefix 자격 없음 |
| 별도 sleep recovery | 원래 seed의2/12 whole | rep6의 end-power guard 위반 | result.json 없음, complete-grid 자격 없음 |
| G0/G1 physical-window actual pilot | 0/32 block | 선행 full-grid+감사 receipt 부재 | 미실행 |

전체 원래 seed32개 중22개만 complete다. 평균/분산/ancestry를 이 prefix에서 선택적으로
집계하지 않았다. P1/P2/large-training flags는 통과로 바꾸지 않는다.
실패한 프로토콜을 그대로 resume하거나 seed를 교체하지 않는다.

원래 v2 실패 구간에 Windows Kernel-Power42(Battery) 및 Power-Troubleshooter1의
저전원 기록을 확인했다. UTC21:28:28.651166–21:53:29.669933 약1501초다.
그 시간을 빼서 원래 wall cap을 통과시킨 것으로 취급하지 않는다.

복구 시 AC 연결로 보고되었지만 실제 배터리는27→24→21→17%로 감소했다.
완료된 두 whole의 wall은280.0436/356.1664초이고 첫 whole의 process CPU2153.5초다.
세 번째 whole는357.4303초 뒤 end-power check에서 실패했고 normalizer result는 저장하지 않았다.
2026-10-19 재개 시에도 AC 표시/battery13%를 관측했다. 충전 표시만으로 안정적인
전력 공급을 보장할 수 없다는 관측이며, 어댑터 불량/정격 부족을 원인으로 확정한 진단은 아니다.
OS 절전 정책이나 critical-battery action은 변경하지 않았다.

## 비용: 성공 표본과 실패 예약을 구분

- 복구의 두 complete whole: physical forward258,560회, joint backward129,280회.
- 새 failed whole: conservative physical reservation129,280회. 결과에 저장된 별도의
  actual backward count가 없으므로 그 값을 관측 count로 만들어 보고하지 않는다.
- 복구 ledger의 conservative physical 합387,840회. prep 포함 unit-wall 합993.6707초.
  이 wall은 phase 전체/I/O/이후 절전 시간을 포함한 값이 아니며 속도 벤치마크가 아니다.
- 원래 v2의 실패/계산비용은 원래 ledger에 별도로 보존했다. 복구 비용을0으로 놓거나
  재사용20개를 새 독립 confirmation이라고 부르지 않는다.

## 이론·기술 검토 결과

1. 원래 archive/config/manifest/checksum 및20complete+12pending의 고정 membership을 확인했다.
2. 복구 source archive1,052entries와 checkpoint checksums를 검사했다. 실제2complete의
   seed/component identity/forward-backward count는 원래 `check_unit` contract를 통과했다.
3. 원래 및 복구 result.json이 모두 없고, 실패 사유가 보존되어 있음을 확인했다.
   이 무결성 검사는 새로운 통계 precision/미관측 tail/4whole numerical replay의 대체가 아니다.
4. 새 geometry/runner/Gaussian-toy/power-guard targeted14tests 통과. 전체 CI Ruff와
   production Mypy199files, 명시적 신규3files Mypy 통과. 이 변경분의 전체 로컬 suite는
   배터리 상태 때문에 재실행하지 않았으며 이전1392-pass 결과로 새 tree를 인증하지 않는다.
5. G1의 모든 price 방향은 고정 양의 물리 duration을 갖는다. point-price noise limit과
   physical-window density continuity는 조건부 수학 결과이며 실제 rBergomi variance 개선은 미검증이다.

## 재개 우선순위

1. 사용자가 전원 어댑터/연결을 확인하고 배터리가 실제로 회복·유지되는지 확인한다.
   낮은 배터리에서 고부하 trial을 재시작하지 않는다.
2. 안정 전원 확보 후 별도 사전 고정 protocol을 작성한다. 필요하면1–2CPU threads의
   고정 입력 numerical equivalence와 소규모 전력/시간 calibration을 먼저 한다.
   기존 source/seed/failed records는 덮어쓰지 않는다. AC 표시만 확인하는 preflight의 한계도 보완한다.
3. 원래32개 grid의 완전한 별도 검증 자료를 확보하고 독립 whole 단위로 요약한다.
   재사용/새 trial/실패 비용을 구분하고4whole replay를 완료한다.
4. 그 후에만 사전 고정 G0/G1 reference pilot를 실행한다. reference precision과
   guide-independent agreement를 확보하기 전에는 P2와 대규모 학습을 열지 않는다.

코드·문서·실패 evidence를 main에 보존하는 것은 위 과학적 gate 통과와 별개의 작업이다.
