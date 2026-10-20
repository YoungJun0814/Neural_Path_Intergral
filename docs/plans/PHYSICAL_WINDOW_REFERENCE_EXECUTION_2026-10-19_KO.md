# G0/G1 reference pilot: 실제 첫 sampling 실행 조건

2026-10-19. 아직 실제 G0/G1 표본은 없다. 기존
[고정 pilot 계획](PHYSICAL_WINDOW_REFERENCE_PILOT_2026-10-16_KO.md)의 geometry/seed/count/
float64/8threads/paired likelihood/비교 family/통계 요약/caps를 그대로 유지한다.
변경 사항은 실패한 선행 실행 대신 새 complete-grid 복구의 anchor를 사용하는 것과 전원 guard다.
resource의 통계 prefix 결과를 보고 guide/amplitudes/budgets를 조정하지 않았다.

선행 파일은 `p2_fine_resource_stable_recovery_v2/session_20261019/result.json` 및
그 SHA와 고정4whole numerical replay를 묶은 `audit.json`이다. 둘 다 완성·검증되어야 실행한다.
새 directory는 `physical_window_reference_v1/session_20261019`이며 원래
`physical-window-reference-v1` namespace를 유지한다. 다른 실행 결과와 pooling하지 않는다.

시작 시AC/battery>=30%, block 전후AC/battery>=25%를 요구한다. block 동안2%p 이상
감소하면 numerical record를 먼저 보존하고 별도 power_failure로 중단한다.
OS 전원 정책은 바꾸지 않는다. resource 계산/감사와 동시에 실행하지 않는다.

원래2cells×2guides×8blocks, block당16384fine samples/chunk512를 전부 계산한다.
전체 physical1,048,576/density524,288/time-step3,221,225,472.
caps physical1.2m/phase7200초/block180초/prep60초/RSS4GiB 유지.
G0/G1은 각각373개 normalized Gaussian components, natural.1,
vol amplitudes2/4/6/8 및 price0/2/4로 동일하다. 원래 q/driver/payoff는 바꾸지 않는다.

완료 후4block replay 감사 physical131,072/density65,536회를 별도로 수행한다.
μ의 IID RSE와 block sensitivity, raw max-share, paired small-difference uncertainty,
G0/G1 μ agreement, safety3 fresh-precision forecast를 함께 검토한다.
G0/G1은 geometry를 공유하므로 서로의 agreement만으로 blind spot을 배제하지 않는다.
guide-free SMC reference와의 교차 검증과 새 fixed-count confirmation이 별도로 필요하다.
P1/P2/large-training flags는 이 pilot에서false로 유지한다.
