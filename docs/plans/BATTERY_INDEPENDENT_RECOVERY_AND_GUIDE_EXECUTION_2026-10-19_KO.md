# 배터리 조건을 제외한 우선순위 실행

2026-10-19. 사용자가 배터리를 고려하지 말고 계속 계산하며 커밋·푸시하지 말라고 요청했다.
새 실행은 배터리/AC 상태를 조회하거나 중단 조건으로 쓰지 않는다. OS power policy와
critical battery action을 변경하는 요청으로 해석하지 않는다.

## 우선순위1: 원래32개 seed grid 완성

원래20개+첫 복구2개+최근 numerical whole1개=23개를 source/SHA/seed/count 검증 후
재사용한다. 최근 whole는 numerical record가 완전하지만 당시 power protocol은 실패했다.
그 실패를 성공으로 덮어쓰지 않고, 이번 정책 아래의 별도 complete-grid recovery에서
수치 기록만 재사용한다. 포함 여부는 평균/분산을 보고 결정하지 않는다.

남은 high-η/more-temperatures rep7과 more-particles rep0..7의9개를 원래 namespace
`paired-fine-resources-v2`와 동일seed로 계산한다. dtypefloat64/8threads, N4096/2048
exact Gaussian augmentation, full reverse-MH correction, particles/bridge/mutation은 그대로다.

새 directory: `results/post_audit/p2_fine_resource_completion_v3/session_20261019`.
신규 physical1,149,184/backward574,592/time-step3,530,293,248.
prep0nonlinear evaluations. cap physical1.2m/phase21600초/unit1200초/prep30초/RSS4GiB.
배터리와 무관하게 numerical finite/identity/source/count/memory/wall 검사는 유지한다.
OS sleep에 의한 wall 초과를 성공으로 바꾸거나 실패 seed를 교체하지 않는다.

모든32개가 모인 후에만 기존 요약/선택 규칙을 적용한다. wall은 여러 전원·운영 상태가
섞여 있어 비교에 쓰지 않는다. 새32개 confirmation이 아니라 원래 grid의 완전한 수치 복구다.
기존 운영 실패 flags와 P1/P2/large-training=false를 보존한다.

## 우선순위2: 재현 감사와 연구 판단

각cell/design rep0의4whole를 같은8threads로 replay한다. 감사 physical513,536/
backward256,768은 따로 기록한다. archived source/config/anchor/member/normalizer identity,
숫자·비용·전체 요약을 대조한다. 전원은 검사하지 않는다.
μf/μc RSE, ancestry, paired small-difference upper와 safety3 fresh precision forecast를 검토한다.
작은 adjacent difference나 allocation forecast를 실제 precision/continuum bias 보증으로 부르지 않는다.

## 우선순위3: 사전 고정 G0/G1 reference pilot

위 full-grid result와 audit receipt가 완성된 뒤 기존 G0/G1 numerical implementation을
그대로 사용하되 새 실행 wrapper에서 배터리 조건만 끈다. 실제 표본은 아직0개이므로
변경 전 결과를 보고 설정을 고르는 것이 아니다. geometry/373components/natural.1,
amplitudes/seed namespace/2cells×2guides×8blocks/각16384draw/chunk512는 그대로다.
physical1,048,576/density524,288/time-step3,221,225,472. 기존 phase7200초/block180초/
prep60초/RSS4GiB 유지. 완전한32blocks와 고정4block replay를 마친 후 비교한다.

reference agreement/precision이 부족하면 원인·실제 비용과 다음 방법을 보고하며
NN/operator/차원을 늘려 문제를 덮지 않는다. 최소 두 독립 메커니즘의 같은 accuracy
자격과 새 confirmation 없이는 저널용 효율 우위를 주장하지 않는다.

모든 변경·결과는 로컬에만 남기고 커밋·푸시하지 않는다.
