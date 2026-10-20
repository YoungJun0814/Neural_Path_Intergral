# P2 병목 해결책의 우선순위와 방어적 mixture 재배분 실험

2026-10-20. 새 실험 결과 확인 전 고정. 로컬 CPU float64, 8 threads.
커밋·푸시·유료 자원·배터리 조회 없음. 과거 runtime/결과 보존.

## 우선순위 판단

성공 확률의 수치 추정은 하지 않는다. 현재 증거, 변경 범위, 비용, 독립 검증 가능성을
함께 평가한 상대적인 실행 순서다.

| 순위 | 방안 | 근거와 한계 |
|---|---|---|
| 1 | 현재 G1 dictionary의 방어적 mixture 비율을 독립 bank로 학습 | window geometry의 개선 신호는 있지만 372 shifted 후보에 균등 배분한다. geometry/물리 법칙을 유지하며 비용 배분만 검증 가능. 없는 모드를 새로 만들지는 못함 |
| 2 | 물리적 기여 집중 분석 및 guide-free SMC와 새 독립 교차검증 | 서로 다른 탐색 실패를 확인하는 데 필수. 전체 SMC 표본 증설은 비용이 커 먼저 1순위 결과로 필요 예산/geometry 한계 판단 |
| 3 | 독립 bank 기반 새로운 window/scale geometry | 재배분이 일반화되지 않거나 tail 집중이 남으면 필요. 현재 평가 자료로 geometry를 고른 뒤 같은 자료로 성공을 주장하지 않음 |
| 4 | FFT 조건부 계산 preconditioner/가속 | 정확성 검사는 통과했으나 현재 pilot의 효과와 총비용 이득 미입증. 탐색 문제를 해결하지 못하므로 후순위 |
| 5 | 큰 신경망/차원 확대/대규모 학습 | 기준 정확도가 불확실한 상태에서 모델 확장은 성공 판정을 더 어렵게 함. 아직 승인하지 않음 |

## 1순위의 수학적 계약

p는 원래 2N 차원 표준 Gaussian, g는 원래 finite-grid terminal CDF다.
고정 shift component 밀도비 d_j=r_j/p, 원래 균등 G1 밀도비 D0=Σw0_j d_j.
학습 bank는 r0에서 독립적으로 생성한다. 다음 경험적 목적을 최소화한다.

Fhat(w) = mean_bank[ g(z)^2 / (D0(z) Dw(z)) ], Dw=Σw_j d_j.

고정 w일 때 E_r0 Fhat(w)=E_p[g²/Dw]이므로 second moment 목적에 대응한다.
단, 같은 bank로 고른 w의 training 목적은 낙관적일 수 있어 최종 성능 추정에 사용하지 않는다.
목적은 w에 대해 convex다. 이는 기존 mixture IS 최적화 원리이며 새 정리로 포장하지 않는다.
참고: He & Owen, Optimal mixture weights in multiple importance sampling,
https://arxiv.org/abs/1411.3954 .

자연 component w0=.1 고정. 372 shift 각각의 하한은 .1/372, 남은 .8을 simplex v로 배분한다.
즉 w_shift=.1/372+.8v. 원래 균등 G1이 feasible하고 모든 후보를 보존한다.
Dw≥.1이므로 확률 기여 g/Dw≤10. 원래 q, q의 M2 및 물리 법칙은 바꾸지 않는다.
비율을 자르는 clipping, empirical self-normalized 확률 추정, component만의 denominator는 금지한다.
최종 μ는 새 r_w 표본에서 ordinary mean[g/Dw]다.

학습은 deterministic Frank–Wolfe 최대 256회, 상대 경험적 gap 1e-6, line search bisection 40회.
gap은 경험적 문제에 관한 float64 진단이지 모집단 최적성/수치 interval 증명서가 아니다.

## 고정 실험

- 원래 canonical/high-eta, N4096, G1 means/geometry/amps 그대로.
- 두 셀 각각 독립 학습 반복 2개. 선택적으로 좋은 반복만 보고하지 않음.
- 각 반복: training 4,096 draws, validation 2,048 draws. validation은 학습/모델 선택에 사용하지 않고 목적 일반화 진단만 함.
- 각 반복의 final: 원래 uniform와 learned 두 arms, 각각 8 blocks × 4,096 draws.
- 총 새 물리 평가: 4×(4,096+2,048+2×8×4,096)=286,720회.
- chunk 256. 각 stream/role/cell/rep/block/chunk의 hash seed를 분리.
- final 표본은 학습·validation 및 다른 arm/반복과 공유하지 않음.
- preflight 각 셀 256 draws: 30초/cell, RSS 4GiB. 본 block 예측 16배가 180초를 넘으면 producer 시작 안 함.
- 준비 최대 120초, bank 180초/unit, final 180초/unit, 전체 phase 7,200초, RSS 4GiB.
- 실제 physical/density/fit/준비/평가 시간과 독립 block SE 및 기여 집중을 모두 기록.
- probability만 보는 development다. all-q/mesh/독립 탐색/continuum 자격 대체 불가.

## 판정과 분기

최소한 두 학습 반복의 모든 결과를 공개한다. 각 cell/rep의 final RSE, block sensitivity,
최대 기여 및 leave-one-block-out를 보고, uniform/learned의 독립 조정 비교는 family 4개로 계산한다.
두 반복 모두 관측 RSE 감소, training/validation 목적 일반화 및 집중도 개선 여부를 별도 표시한다.
기준 통과 여부와 단순한 개선 신호를 구분하며 observed RSE 비를 certified variance ratio라 부르지 않는다.

최종 RSE≤1.5%라도 이 실험만으로 P1/P2/large-training은 false다. 같은 geometry이고 반복이 2개뿐이다.
악화/일반화 실패면 은닉 geometry/tail 탐색을 다음 우선순위로 삼고 count/seed를 바꿔 반복하지 않는다.
모든 final을 마친 뒤 각 cell/rep의 training bank와 대표 final block을 재현 감사한다.
독립 작은 문제의 convexity/gradient/floor/ordinary-mean oracle와 전체 회귀검사를 수행한다.
