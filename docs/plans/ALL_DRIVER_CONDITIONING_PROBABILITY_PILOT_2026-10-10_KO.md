# Exact all-driver conditioning: N32 μ 비교 pilot

2026-10-10. 새 표본 생성 전 고정한다. P2 및 대규모 학습 승인 실험이 아니다.

원래 BLP/Itô finite-grid law를 그대로 유지하며, 가격 잔차 Gaussian을 모든 left
Volterra driver에 조건부로 적분해 μ의 좌표를 64에서 31로 줄인다.
직접 유도와 oracle는 [이론 검토](../theory/ALL_LEFT_DRIVER_EXACT_PRICE_CONDITIONING_2026-10-10_KO.md)를 따른다.
기존 q의 M2를 reduced G²로 바꾸는 것은 금지한다. q 자체도 변경하거나 학습하지 않는다.

2 original cells × full conditional payoff / all-driver payoff ×8 independent whole=32.
양쪽 모두 natural-start 1024 particles /64 quadratic temperatures /4 exact gradient-MH
moves(.35, drift cap20), every4 unbiased stratified resampling, final mutation 없음.
외부 guide 없이 μ-only target을 사용한다. 차원과 payoff evaluation을 제외한 설정은 같다.
새 namespace `driver-conditioned-probability-v1`, 과거 결과와 pooling/seed 교체 없음.

whole당 254,976 forward/actual backward. 총 8,159,232 forward/backward.
geometry 준비는 analytical rows/QR/kernel뿐이며 basis-path simulation call 0.
cap 10m forward, 10m backward, phase1800초, whole30초, RSS4GiB.
동시 실행 노트북 wall-time이므로 속도 benchmark로 사용하지 않는다.

32 whole 완료 후 μ RSE, independent normalizer agreement(2 comparisons, margin10%),
LOO/max whole share/ancestry를 보고한다. 작은 pilot의 통과는 publication reference
qualification이 아니다. 직접 score cost와 차원 축소도 실제 total-work 개선과 구분한다.
source/model/input/seed/endpoint/cost 감사와 각 셀-방법 rep0의4whole replay를 수행한다.
P1/P2/large-training flag는 false로 유지한다. 성공적인 variance forecast가 있어도
새 final-count validation을 먼저 사전 고정해야 한다.
