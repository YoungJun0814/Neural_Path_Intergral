# P2 fine-mesh recovery: 2026-10-09 사전 고정 프로토콜

## 목적과 변경 불가 조건

N=16/32/64 pilot에서 큰 상대 차이가 발견되었다. 통과 기준을 낮추거나,
continuous model을 고정 격자 문제로 바꾸지 않고 같은 BLP 법칙의 N=32/64/128/256/512를
정확한 adjacent coupling으로 비교한다. 이것은 원인 진단/개발 pilot이며 P1/P2 qualification이나
대규모 학습 승인 자체가 아니다. K=1, H, eta, rho, xi, maturity와 원본 셀을 유지한다.

## 사전 고정 실행

- canonical 및 high-eta 원본 2셀, fine N=32,64,128,256,512: 총 10개 비교.
- 비교당 65,536 IID fine+augmentation 표본, 2,048 단위 32개 batch.
- 표본은 원래 p와 정확하게 정규화된 Gaussian 조건부 excursion mixture r에서 생성한다.
  r=.1p+.9/(6(N-1)) sum_{t,a} [p(.|Y_t>=a)+p(.|Y_t>=a,sign*Z_next>=1)], a=4,5,6.
- 모든 monitoring time 포함. fitted bank/학습 q/이전 guide를 이용하지 않는다.
- FFT density는 dense projection과 동등한 계산 최적화이다. 시뮬레이터 법칙을 바꾸지 않는다.
- fine p/r likelihood 하나로 g_f, g_c, g_f-g_c 모두 보정한다.
  추가 Gaussian residual은 원래 조건부 law로 뽑아 별도 likelihood가 없다.
- 국소 2x5 Gaussian whitening은 dense adapter와 동일하다. coarse local integral은 C1+L2.
- 비교당 signed stable-CDF difference, paired ratio 표준오차, 최대 기여 표본 비중을 기록한다.
- 추가 진단: 기여 가중 최대 variance-cell share. 이는 rare-event 기여 구조 진단이며
  최종 IS estimator를 self-normalize하는 것이 아니다. variance endpoint t=T는 적분에서 제외한다.
- work 한도 1,400,000 terminal potential 평가, wall 한도 1,800초, RSS 한도 4GiB.
  time-step 총합도 별도로 기록한다. 서로 다른 N에서 potential count만으로 비용 우위를 주장하지 않는다.
- namespace `p2-fine-excursion-v1`, 기존 pilot seed 재사용 금지.
- 10개 비교 normal-approximation simultaneous delta-method interval을 개발 진단으로만 사용한다.
  abs(relative difference)+z*SE <= .1 여부를 기록하되 continuum convergence 증명으로 부르지 않는다.
- archive/source/config/input SHA, checkpoint raw 값/seed/work를 보존하고 bounded replay를 수행한다.
- 실패나 높은 RSE도 모두 남긴다. 결과가 불리한 batch/seed/격자를 제거하지 않는다.

## 이론적 검토

BLP에서 standardized Volterra projection은 과거 local coordinates만 사용하므로 다음 Brownian
coordinate와 독립이다. 조건부 prior density는 indicator/정규 tail probability이다. 혼합 성분이
겹칠 때 density는 모든 성분을 합한다. defensive component로 r/p>=.1이므로 support를 잃지 않는다.
FFT convolution은 kernel approximation의 변경이 아니라 기존 convolution의 계산 방법이다.
BLP Gaussian driver의 mean-square convergence만으로 극단 tail probability의 상대 오차가
작다고 결론낼 수 없다. 따라서 mean-square variance 검증과 payoff mesh 검증을 별도로 한다.
원문: [Bennedsen–Lunde–Pakkanen](https://arxiv.org/abs/1507.03004).

## 결과별 다음 행동

1. finer difference가 작고 정밀도 충분: 독립 confirmation grid/새 표본으로 재검증한다.
2. 차이가 계속 큼: 학습 복잡도 증가를 중단하고 discretization error budget/가격 적분 및
   driver approximation의 분리 검증을 진행한다. 기존 q/bank는 N=32 전용 개발 자산으로만 남긴다.
3. tail concentration/RSE 때문에 차이 판단 불가: physical proposal 및 reference 개선 후
   새 고정 표본 계획으로 반복한다. 낮은 관측 분산만으로 pass하지 않는다.
