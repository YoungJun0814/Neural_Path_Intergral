# Fine-only SMC 초기 bridge 고정비용 대조 — 실행 전 고정안

작성: 2026-10-20. 기존 성공/실패 결과는 변경하지 않는다. 커밋·푸시는 하지 않는다.

## 1. 질문과 범위

직전 fine-only N4096 SMC에서는 첫 계보 반감이 β=0.003968–0.016125에서
나타났고, 8회 독립 whole 기준 RSE가 11.58–24.20%였다. 높은 MH 수락률은
충분한 혼합을 보장하지 않는다. 이 관측만으로 편향이나 유일한 실패 원인을
단정할 수 없다. 이번에는 **초기 β 해상도를 늘리는 것이 동일 계산량에서
계보와 whole-normalizer 변동을 개선하는지**만 검사한다.

기존 두 schedule은 실제 resampling β가 거의 같았다. 이번에는
β_k=(k/63)^p, p∈{2,3,4}, k=0,…,63으로 바꿔 그 한계를 직접 대조한다.
β 지점과 실제 resampling β가 함께 변하므로 “mutation만의 효과”로 해석하지 않는다.

## 2. 변경하지 않는 이론·구현

- 원래 두 셀: canonical-h005-k1, high-eta200-k1.
- 동일 N4096 / 8192차원 Gaussian 좌표 / 원래 조건부 terminal CDF g_f.
- π_β ∝ φ(z)g_f(z)^β, Z_0=1, endpoint Z_1=Eφ[g_f].
- 기존 gradient pCN-L, 전체 역·정방향 MH 보정, proposal drift norm cap 20.
- likelihood·potential·기여값 clipping, self-normalized 최종 estimator 금지.
- G1/bank/학습된 guide는 SMC kernel의 입력으로 사용하지 않는다.
- 정규화상수는 단계별 weighted increment의 곱으로 유지한다.
- 모든 arm: 128 particles, 64 temperature points, nonfinal stage당 2 mutations,
  scale 0.35, 4 stage마다 stratified resampling. 마지막 stage에는 mutation/resampling 없음.

## 3. 고정 family와 난수

2 cells × 3 powers × 8 independent triplets = **48 whole runs**.
같은 (cell, replicate)의 세 power는 의도적으로 동일 seed를 쓴다(CRN).
초기 Gaussian, 각 stage mutation 및 resampling 난수 위치가 공통이다.
accept/reject 결과에 따라 난수 호출 수는 바뀌지 않는다. 따라서 arm 간 독립은
가정하지 않는다. 독립 표본 단위는 각 셀의 8 triplets이고, arm별 marginal은
8 independent whole이다. 셀·반복·preflight·과거 실험은 별도 namespace로 분리한다.

실행 순서는 cell → replicate → power. 결과를 본 후 count, seed, power,
판정 기준을 바꾸거나 실패 seed를 교체하지 않는다. prefix 결과로 판정하지 않는다.

## 4. 계산 예산과 중단

- whole당 forward=backward=128[1+62×2]=16,000.
- 최종 family: **768,000 forward / 768,000 backward**.
- 각 cell/power별 32 particles, 같은 64 points/2 moves의 별도 preflight:
  6×4,000=24,000 forward/backward. 통계에는 포함하지 않는다.
- 생산 총 **792,000 forward / 792,000 backward**.
- audit: rep0의 6 wholes를 재실행하므로 추가 96,000 forward/backward.
- CPU float64, 8 threads, RSS 4 GiB. 준비 120초, preflight whole 60초,
  final whole 300초, 전체 phase 14,400초. preflight wall×4≤300초여야 진행.
- 오류·예산초과·불완전 family는 보존하고 자격을 부여하지 않는다.
  학습·cloud·입자수 증설·새로운 대규모 실행은 자동으로 시작하지 않는다.

## 5. 사전 고정 진단과 판단

1. arm별 ordinary mean(Zhat), independent-whole RSE, max whole contribution,
   leave-one-whole-out shift, whole bootstrap, minimum ESS, final ancestors, 수락률, wall.
2. β={0.01,0.04,0.1,0.5,1}에서 마지막으로 기록된 β≤milestone의 계보를
   사용한다. 실제 사용 β도 기록한다. stage 사이 혼합을 추정하지 않는다.
3. precision: RSE≤1.5%, max whole share≤15%, max LOO shift≤5%를 모두 요구.
   8 wholes는 대규모 confirmation이 아닌 작은 개발 대조이며 tail 안정성 증명 아님.
4. frozen fresh IID reference 두 셀과 6회 Bonferroni-adjusted 10% equivalence.
   양쪽 precision/sensitivity와 equivalence를 모두 통과해야 scoped corroboration.
5. power2를 분모로 (mean(Z2)-mean(Zp))/mean(Z2)의 paired delta-method
   uncertainty를 4회 다중비교 보정한다. 이는 평균 차이 진단이지 variance 감소 검정 아님.
6. 개발상 후속 검토 후보: power3 또는 4가 **두 셀 모두에서** RSE를 낮추고,
   median final ancestry와 β≤0.04 median ancestry를 baseline 이상 유지해야 한다.
   통과한 power는 모두 보고하고 이 중 최적 power를 자동 확정하지 않는다.
   하나도 없으면 채택 없음. 새 independent confirmation 없이 성능 주장을 하지 않는다.
7. `p1_qualified`, `p2_qualified`, `large_training_authorized`는 항상 false.
   q 개선·all-q M2·mesh·continuum·논문 성능 입증과 혼동하지 않는다.

## 6. 검증과 산출물

실행 전 schedule/CRN/work/constant-potential/analytic-normalizer/complete-family
테스트 및 lint·types 확인 → source/config/계획/이론/parent SHA 봉인 → preflight
→ 전체48회 → 모든 checkpoint·seed·stage·work·summary 검증 → 대표6회 replay
→ 회귀테스트 → 결과 보고서와 README 업데이트. 기존 봉인 runtime은 수정하지 않는다.

실패하면 더 큰 계산으로 덮지 않는다. 어느 β에서 계보가 줄었는지, 전체
RSE·reference agreement·late-β tradeoff를 보고 다음 방법의 설계를 결정한다.
