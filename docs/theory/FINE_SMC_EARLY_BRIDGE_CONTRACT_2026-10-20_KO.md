# 초기 bridge power 대조의 이론적 계약

2026-10-20. 본 문서는 새 schedule 실험 전에 고정한다.

## 대상과 정합성

고정된 Gaussian Volterra 이산화의 Gaussian 좌표 z∼N(0,I_8192)에 대해
g_f(z)는 직교 price noise를 적분한 원래 terminal-event 조건부 확률이다.
새 soft target·guide·coarse correction을 추가하지 않는다. 대상은 연속시간
확률이 아니라 기존 N4096 finite-grid μ_f=E[g_f]이다.

0=β_0<…<β_63=1인 모든 결정론적 schedule은 같은 endpoint를 갖는다.
단계 가중치는 exp((β_j−β_{j−1})log g_f)이며, 현재 normalized particle
weights에 대한 그 평균을 Zhat에 곱한다. resampling을 생략한 단계에서는
weights를 그대로 운반한다. 최종 추정은 independent whole의 Zhat를
산술평균한다. mean(log Zhat)의 지수나 최종 self-normalized payoff 평균은 쓰지 않는다.

각 단계의 기존 pCN-L proposal에는 forward/reverse Gaussian density와
φ(z)g_f(z)^β 비율이 포함된다. drift norm을 자르는 것은 proposal 변경이며,
같은 cap을 reverse drift에도 적용한 MH 보정이 필요하다. 기존 검증 core를
재사용한다. 이것은 finite-grid invariant kernel의 정합성이지 빠른 혼합 보장이 아니다.
고정 schedule, 적절한 불편 resampling, invariant mutations 아래 표준 SMC
normalizer 구조가 유지된다. 유한정밀도 수치오차가 없다는 정리는 주장하지 않는다.

## 초기 해상도와 비용

β_k=(k/63)^p에서 p를 키우면 초기 β increment는 작아진다. β≤0.04에
배치되는 nonfinal mutation stages는 p=2,3,4에서 각각12,21,28개다.
stage당2회이므로 각각24,42,56회 proposal/particle이다. 총124회는 같다.
반대로 후반부 increment는 커지므로 late bottleneck이 악화될 수도 있다.
resampling 횟수는 모두15회지만 실제 β는 다르다. 따라서 계보 개선은
이론적으로 보장되지 않으며 실험 질문으로 남긴다.

## CRN, 추정 단위, 오류 방지

세 arms에 공통 난수를 써도 각 arm의 marginal sampling law는 그대로다.
동일 schedule 길이·mutation 수·resampling stage index로 난수 호출 위치도
맞는다. MH acceptance가 달라도 난수 생성은 항상 수행되어 branch-dependent
소모가 없다. 공통 난수가 반드시 양의 상관이나 분산 감소를 보장하지는 않는다.
paired contrast는 동일 replicate 순서로 계산하고, 서로 다른 arm을 독립이라고
취급하는 SE 계산은 하지 않는다. frozen IID reference와 새 SMC의 난수는 분리한다.

SMC particles는 resampling 이후 독립 표본이 아니다. RSE 단위는8 independent
wholes이고, paired contrast 단위는8 independent triplets다. initial ancestry는
단조 감소하는 genealogy marker이며, mutation으로 새로운 계보 label을 만들지 않는다.
낮은 ancestry가 곧 편향, 높은 ancestry가 곧 mode coverage는 아니다.

8회 whole에 대한 normal/delta/bootstrap uncertainty는 개발 진단이다.
다중비교를 보정해도 희귀 미관측 꼬리·비정규성·작은 반복 수 문제는 사라지지
않는다. RSE/계보 기반 사후 후보는 independent confirmation이 필요하다.
본 실험은 학습 bank 품질, 원래 q의 M2, mesh error, continuum theorem을 해결하지 않는다.
