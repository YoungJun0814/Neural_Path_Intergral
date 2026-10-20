# 2-thread 저부하 backend의 수치·전원 교정

2026-10-19. 사용자의 전원 안정 확인 뒤에도8thread 실제 whole에서30→26%로 감소해
보호 조건으로 중단했다. 그 whole의 numeric record는 저장했지만 protocol은 실패다.
이 계획은 하드웨어 운영 조건만 검증하며 rare-event estimator 성능을 평가하지 않는다.

고정 high-η/N4096/128개 joint Gaussian 입력은 이전 backend calibration의 동일seed다.
8thread baseline1회와2thread warmup1회를 비교하고,2thread에서60초간 같은 입력의
gradient callback을 반복한다. 총최대514callbacks/physical131,584/backward65,792,
phase180초/RSS4GiB. 입력을 바꾸거나 성능 표본을 생산하지 않는다.
반복 횟수는 시간 제한 교정용이며 statistical optional-stopping estimator가 아니다.

components/family log potential/capped gradient를 기존rtol5e-10/atol1e-8로 비교한다.
2thread continuous stress 중 약20초 간격으로 Windows WMI BatteryStatus를 확인한다.
AC 연결,Discharging=false, 시작·끝native battery>=20%, 끝percent가 시작보다1%p 넘게
낮아지지 않음, 실제 RemainingCapacity가100mWh 넘게 줄지 않음을 모두 요구한다.
WMI가 없거나 표본이 부족하면 unknown을 pass로 바꾸지 않는다.
OS power policy와 critical-battery 설정은 변경하지 않는다.

추가로4차원 알려진 Gaussian normalizer의 full gradient-SMC를2/8threads 각각
4개 고정seed로 계산해 normalizer와 stage ancestry/acceptance를 비교한다.
이는 Volterra physical cost와 별도toy potential work로 보고한다.

모든 교정을 통과한 경우에만2thread의 새 사전 고정 복구 실행을 준비한다.
실제 Volterra whole의 cross-backend replay도 별도로 필요하다. 이전8thread source와
실패 기록을 보존하고 새로운32개 confirmation이라고 부르지 않는다.
실패하면 더 큰 실험을 시도하지 않고 전원 관련 관측을 보고한다.
