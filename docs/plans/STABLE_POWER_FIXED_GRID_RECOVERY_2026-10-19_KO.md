# 안정 전원에서 원래32개 고정 seed grid 완성

2026-10-19. 사용자가 전원 안정화를 확인했다. 재확인 시 AC 표시/battery31%이며,
기존 HEAD/파일 변경이 없었다. 원래 두 운영 실패는 그대로 보존한다.

## 변경하지 않는 것

원래 namespace `paired-fine-resources-v2`,2cells×2designs×8whole=32개 seed를 유지한다.
fine4096/coarse2048 exact Gaussian augmentation, guide-free full reverse-MH correction,
particles/temperatures/mutation/float64/8threads/normalizer identity는 전부 동일하다.
원래 source의 모든 src/experiments/config runtime 파일과 기존 복구 loader의 SHA를 검사한다.
실패 기록을 삭제하거나 같은 directory에서 resume하지 않는다. 완전한 데이터 복구이지
새로운32개 confirmation이 아니며 P1/P2/large-training flags는 계속false다.

## 고정 재사용과 계산 범위

- 원래 v2 complete20개와 sleep-recovery-v1 complete2개를 checksum/source/count
  검증 후 재사용한다. 표본 평균/분산을 보고 포함 여부를 정하지 않는다.
- 원래 high-η/more-temperatures rep6,7 및 more-particles rep0..7의 **10개**를
  원래 seed로 처음부터 계산한다. 실패한 rep6의 partial result는 사용하지 않는다.
- 새 physical forward1,278,464/joint backward639,232/time-step forward3,927,441,408.
  prep0nonlinear simulations. caps physical1.3m/phase21600초/whole1200초/prep30초/RSS4GiB.
- 전체32개의 유효 physical4,108,288회와 두 discarded failed reservation258,560회를
  구분한다. 최초resource-v1 등 다른 실험 비용은 이 합계에 포함하지 않는다.
- 새 directory `results/post_audit/p2_fine_resource_stable_recovery_v2/session_20261019`.

## 전원·중단 관리

시작 시AC/battery>=30%, 각whole 전후AC/battery>=25%를 요구한다. 한whole 동안
배터리2%p 이상 감소하면 완료된 numerical record를 저장한 뒤 별도 power_failure를
남기고 중단한다. 끝의 power check 때문에 계산 완료 normalizer를 버리지 않도록,
숫자 record와 운영 유효성 판단을 분리한다. 저전력 실행을 성공으로 재분류하지 않는다.
OS 전원 정책/critical battery action은 변경하지 않는다. 다른 대형 실험과 겹치지 않는다.
wall에는 pause/I/O도 포함하며, 이전처럼 절전 시간을 빼서 cap을 통과시키지 않는다.
새 실패 발생 시 무한 재시도/seed 교체/부분 집계 없이 원인과 기록을 보고한다.

## 요약·검증·후속

모든32개가 모인 뒤에만 기존 요약 및 safety3 timestep-cost selection rule을 적용한다.
μf/μc RSE, paired difference/uncertainty, initial ancestry, sensitivity, fresh precision
work forecast를 보고한다. 여러 전원 상태의 wall은 방법 간 속도 비교에 사용하지 않는다.
32개 whole와 독립 fresh-confirmation을 혼동하지 않는다. 작은 adjacent difference는
continuum relative-bias bound가 아니다.

완료 후 각cell/design rep0의4whole를 고정8threads로 replay한다.
감사 physical513,536/backward256,768회는 본 실행과 별도다. source/archive/원본 SHA,
normalizer identity/seed/수량/전체 요약/전원·운영 flags를 검증한 receipt를 저장한다.

그 뒤 실제 표본 생성 전 G0/G1 pilot의 선행 anchor를 이 새 result/receipt로 바꾼다.
G0/G1의 geometry/count/namespace/μ 추정식/비교 family는 원래 계획대로 유지한다.
실행 전 전원 guard를 추가하되 original simulator/payoff/density는 바꾸지 않는다.
reference precision과 guide-independent agreement 없는 대규모 학습은 시작하지 않는다.
