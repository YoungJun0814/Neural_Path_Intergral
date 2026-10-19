# Resource-v2: 배터리 절전 후 전체 고정-grid 데이터 복구

2026-10-16. 원래 v2의 cap/manifest/source/failed finish를 바꾸거나 resume하지 않는다.
원래 실행은 운영 실패이고 result.json은 없다. prefix를 통과 결과로 발표하지 않는다.

## 확인한 사건

32개 whole 중20개 완료 후 high-η/more-temperatures/rep4가 wall1745.756초로
원래600초 cap을 초과했다. Windows Kernel-Power42의 이유는 Battery이며,
Power-Troubleshooter1은 UTC21:28:28.651166에서21:53:29.669933까지 저전원
상태였음을 기록한다. 약1501초의 suspension이 있었다. 이를 빼서 원래 cap을
통과했다고 재계산하지 않는다. 현재 배터리27%, BatteryStatus2(충전)를 확인했다.

## 새 artifact의 정확한 범위

새 directory `results/post_audit/p2_fine_resource_sleep_recovery_v1/session_20261016`.
원래 v2의 **동일32개 seed와 수학적 알고리즘**을 유지한다. 새로운 trial의32개
독립 표본인 것처럼 부르지 않는다. 원래20개 complete whole를 hash-bound evidence로
읽고, 결과가 저장되지 않은 rep4 및 미실행11개만 원래 seed로 처음부터 계산한다.
seed 교체, failed whole의 prefix 사용, 평균을 보고 종료, 최종 표본과의 pooling은 없다.
임의의 source-changed resume가 아니라 별도 append-only recovery artifact다.

원래 archive/config/manifest/21개 finish를 SHA256으로 묶고 canonical checksum을
검사한다. 계산에 사용하는 원래 archive의 src/experiments/config/runtime 경로가
현재 workspace와 동일한지 확인한다. 다른 새 파일은 원래 실행에 import되지 않는다.
새 source archive/계획과 계산된12개 unit도 별도 저장한다. 원래20개 숫자는 수정하지 않는다.

## 비용과 실행조건

미완료12개: more-temperatures4회 + more-particles8회.
새 physical forward1,537,024, joint backward768,512,
time-step forward4,721,737,728. 상한physical1.6m/phase21600초/whole1200초/prep30초/RSS4GiB.
TorchCPUfloat64/8threads, 원래 exact gradient-MH/bridge/particle/seed 전부 동일.
단일 대형 로컬 실험으로 실행하며 각 unit 직전에 Windows AC 연결 및 배터리>=20%를
확인한다. 배터리 critical action이나 OS 전원 정책은 바꾸지 않는다.
새 실패가 생기면 다시 표본을 교체하거나 무한 재시도하지 않고 중단·보고한다.

원래 완성될 전체32개 physical4,108,288회에 더해 버려진 failed whole의 conservative
reservation129,280회를 별도 표시한다. 새 계산비용과 기존 재사용 비용을 합산해
fresh independent32개라고 부르지 않는다. 실제 failed backward는 원본 finish에
없으므로64,640회를 독립 observed count로 보고하지 않는다.

## 결과·감사·다음 단계

전체32개가 모였을 때만 원래 고정 요약/selection rule을 적용한다. probability precision,
paired difference, ancestry, safety3 work forecast를 개발용으로 보고한다.
원래 prefix20개 또는 canonical만 따로 qualification하지 않는다.
wall은 충전/배터리/절전이 섞인 측정치여서 방법간 속도 비교와 fixed-precision wall
주장에 사용하지 않는다. 원래 요약의 wall forecast도 explicitly unqualified로 표시한다.

완료 후 각cell/design rep0의4whole numerical replay(physical513,536/back256,768)를
별도 감사한다. 이4개는 원래 재사용 whole도 포함해 정확한 kernel/source/seed를 확인한다.
모든 P1/P2/large-training flags는false다. 새 독립 confirmation은 향후 별도다.

G0/G1 reference pilot의 선행조건은 이 복구된 전체32개 artifact와 감사로 명시적으로
변경한다. 아직 G0/G1 실제 표본을 생성하지 않았으므로 결과를 보고 guide를 바꾸는
것이 아니다. G0/G1 pilot의 원래 count/namespace/geometry/추정식은 그대로다.
