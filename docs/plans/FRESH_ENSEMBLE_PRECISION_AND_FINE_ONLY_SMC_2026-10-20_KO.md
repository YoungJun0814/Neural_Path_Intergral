# 새 앙상블 정밀도 검증과 fine-only 독립 SMC 대조

2026-10-20. 새 표본 생성 전 고정. main 9cafe59의 원격 CI는 모두 성공했다.
로컬 CPU float64/8 threads, 배터리 조회·OS 정책 변경·유료 자원 없음.
과거 source와 실패 결과는 보존한다. 이번 요청은 다음 단계 실행이며 새 커밋·푸시는 별도 요청 전 하지 않는다.

## 1. 질문

1. 고정된 두 bank 평균의 G1 ensemble이 새 표본에서 probability RSE≤1.5%와 민감도 조건을 충족하는가?
2. 같은 fine N4096 target을 계산하는 guide-free SMC와 일치하는가? SMC 자체의 정밀도·계보는 충분한가?

원래 q/그 M2, coarse/continuum error, 새 bank 학습 반복, 논문용 confirmation은 이 실험의 범위가 아니다.
같은 geometry의 fit끼리 비교하는 것으로 독립 기준을 대체하지 않는다. 전역 자격은 모두 false다.
I 단계는 전체 102개 block 종료 후 별도의 non-overwriting 단계 결과를 저장한다.
G 단계의 실패 여부와 무관하게 I의 완료 자료를 보존하되, 독립 corroboration으로 취급하지 않는다.

## 2. I: 앙상블의 새 fixed-count 검증

직전 full-path pilot의 ensemble 행에서 r=max(raw IID RSE, between-block RSE)를 사용한다.
새 block 수 = ceil[3×65,536×(r/.015)² / 8,192]. safety 3은 예측 여유이지 성공 보장이 아니다.
canonical 59 blocks=483,328 draws, high-eta 43 blocks=352,256 draws다.
총 final 835,584회. 과거 final/학습 bank와 독립이고 합치지 않는다. chunk=256.
새 namespace fresh-ensemble-precision-fine-smc-v1, stream/cell/block/chunk/role hash seed를 분리한다.
G1 means, ensemble weights, defensive masses, physical bins는 직전 감사된 상태 그대로 유지한다.
full mixture D로 ordinary mean g/D를 계산하고 모든 path의 bins를 기록한다.

각 cell의 별도 256 draw preflight(총 512)는 최종 성능 추정에 사용하지 않는다.
준비 120초, preflight 30초/cell, final 180초/block, phase I 7,200초, RSS4GiB.
preflight 시간×32>180초면 해당 producer를 시작하지 않는다. raw/between-block 예측을 낮게 골라 사용하지 않는다.

## 3. G: guide-free fine-only SMC 개발 대조

target은 원래 p(z) g_f(z)^β, z∈R^(2N)다. coarse target/augmentation을 제거하므로
fine 확률 μ_f 자체는 바뀌지 않는다. 과거 paired umbrella와 같은 estimator라고 취급하지 않는다.
정규화 상수의 product를 유지하고 whole-run 추정량의 ordinary mean으로 μ_f를 계산한다.
입자 간 독립 SE는 사용하지 않는다. 과거 G1 means/weights는 SMC 커널에 입력하지 않는다.

원래 검증된 gradient pCN-L full reverse-Gaussian MH correction, drift cap20(제안에만),
scale .35, quadratic fixed β, stratified resampling을 그대로 재사용한다.

| arm | particles | β points | nonfinal mutation 수 | resample every | forward/backward per whole |
|---|---:|---:|---:|---:|---:|
| 64points-2moves | 128 | 64 | 2 | 4 | 16,000 |
| 128points-1move | 128 | 128 | 1 | 8 | 16,256 |

2 cells×2 arms×8 독립 whole=32개. 본 실험 forward/backward 각각 516,096회.
두 arm은 약 15회의 resampling과 유사 총 mutation 비용을 맞추지만 완전한 단일요인 대조는 아니다.
과거 particles64의 paired 결과와 우위를 직접 판정하지 않는다.

별도 resource preflight는 cell별 particles32/β16/mutation1/resample4의 480회씩,
총 960 forward/backward이며 inferential whole에 합치지 않는다. whole300초·preflight30초,
전체 phase G10,800초, RSS4GiB. preflight 시간×16,256/480>300초면 G 본 실험을 시작하지 않는다.
I와 G는 timing 자원 경쟁을 피하도록 순차 실행한다. I의 통계적 성공 여부로 G seed/count를 바꾸지 않는다.
전체 I+G physical forward=1,353,152회(모든 preflight 포함), backward=517,056회.

## 4. 사전 판정

각 cell/method의 정밀도와 민감도는 별도로 계산한다.

- RSE≤.015. IID는 raw RSE, SMC는 독립 whole-run RSE.
- 최대 단일 inferential unit 비중≤.15, leave-one-unit-out 이동≤.05.
  IID unit은 equal-size block, SMC unit은 전체 whole다.
- IID에서 between-block RSE≤2×raw RSE.
- SMC ancestry/ESS/acceptance를 전부 보고하되 acceptance를 mixing 증명으로 취급하지 않는다.

fresh IID 대 SMC 2 arms × 2 cells의 4개 비교에 조정 10% equivalence를 적용한다.
정밀도 자격을 먼저 충족한 쌍에서만 accuracy-qualified corroboration으로 표시한다.
그 외에도 차이·구간은 공개하지만 unresolved와 detected disagreement를 구분한다.
ensemble 이전 pilot 대비 비교 2개는 별도 development family로 보고하고 독립 메커니즘으로 세지 않는다.
연속시간 bias, all-q M2, total fixed-precision wall-time 또는 P2로 자동 승격하지 않는다.

## 5. 감사·실패 처리

모든 고정 unit 완료 뒤 I는 각 cell 첫/마지막 block(총4), G는 각 cell/arm rep0(총4)를 replay한다.
I 추가32,768 physical, G 추가64,512 forward/backward, 합계 audit physical97,280회다.
source/config/seed/membership/work/moments/feature bins/SMC 단계 진단을 검증한다.
작은 문제의 fine-only target·gradient·normalizer 의미, cap/work·SE·자격 방어 검사를 먼저 수행한다.
그 뒤 전체 회귀검사와 결과 보고서를 작성한다.

같은 namespace의 seed 교체, 유리한 prefix, threshold 완화, 성공할 때까지 count 추가를 금지한다.
operational 실패는 보존하며 새 protocol 없이는 중단 unit을 바꿔 넣지 않는다.
통계적 실패 시 다음 설계 근거로 사용하되 실패 실험을 새 성공 표본에 합치지 않는다.
