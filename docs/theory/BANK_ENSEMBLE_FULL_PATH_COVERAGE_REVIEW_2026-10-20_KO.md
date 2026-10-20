# Bank ensemble 및 전체 경로 bins: 수학·기술 검토

2026-10-20. 기존 fixed-dictionary IS와 원래 finite-grid 법칙을 유지한다.

## 앙상블의 정확한 의미

고정 shift means를 공유하는 두 proposal의 weights w1,w2에 대해 we=(w1+w2)/2다.
밀도비 De=(D1+D2)/2이므로 373개 component 안에서 같은 mixture를 정확하게 표현한다.
새 표본은 r_e=p De에서 생성하고 X_e=g/De를 ordinary mean한다. w1,w2의 학습
bank 및 과거 final과 분리한 새 표본이므로 bank 조건부 평균은 μ=E_p g다.

두 단일 fit과 ensemble이 같은 bank 정보를 공유해도, 별도 final stream을 사용하면
bank 조건부로 sample streams는 독립이다. 그 조건부 평균도 모두 μ로 같다.
이번 비교는 frozen weights에 조건부인 개발 실험이며 bank를 재학습한 확인 반복이 아니다.

각 z에서 1/[(D1+D2)/2]≤[1/D1+1/D2]/2이므로
F(we)≤[F(w1)+F(w2)]/2, F(w)=E_p[g²/Dw]다.
이는 최고 단일 fit 대비 우위나 모집단 mode coverage를 보장하지 않는다.
관측 RSE의 순서가 이 부등식과 다를 수도 있다. 별도의 새 수학 원리라는 주장이 아니다.

## 모든 경로의 유한격자 특성

v_i는 원래 simulator의 왼쪽 분산, h=T/N,
I=h Σv_i, κ=T max(v_i)/I≥1,
t_peak/T=argmax(v_i)/N이다. peak가 여러 개면 처음 나타난 index를 사용한다.
폭 N/32의 모든 격자 시간창에 대해 C32=max Σ_window v_i / Σ_all v_i를 계산한다.
N이 32의 배수이므로 시간창 폭은 정확히 T/32다.

prefix sum을 이용하지만 유한정밀도 차분 오차는 존재한다. 직접 sliding-window
계산과 작은 배열에서 대조하고 positivity/range를 검사한다. directed rounding이나
continuum spike-duration certificate를 제공하지는 않는다. κ와 C32는 현재 격자에 의존한다.

물리적 특성으로 만든 96개 bins는 sample들을 분할한다. 이 features는 분산 driver의
함수이므로 원래 conditional terminal CDF와 결합해 μ_b=E_p[g 1_bin_b]를 정의한다.
고정 bins를 사후에 결과에 맞게 바꾸지 않는다. 이 구간들은 학습된 물리 모드가 아니며
구간을 관측했다고 그 안의 모든 희귀 경로를 충분히 찾았다는 의미도 아니다.

## 구간 평균과 불확실성

전체 N_draw개 평가의 X_i=g_i/D_i에 대해 S_b=Σ X_i 1_b, Q_b=Σ X_i² 1_b다.
구간 확률 추정량은 μhat_b=S_b/N_draw다. bin sample 수로 나누면 다른 조건부
양을 추정하게 되므로 금지한다. Σ_b μhat_b=μhat이고 모든 분모는 원래 full mixture다.

S_b>0이면 observed RSE²=(N_draw Q_b/S_b²−1)/(N_draw−1)이다.
이는 bin 밖의 모든 sample이 0인 변수를 전체 N_draw개로 평가한 공식이다.
아예 관측하지 못한 bin에는 추정 SE=0/확률=0의 자격을 주지 않고 unobserved를 표시한다.
매우 작은 비영 기여는 exp에서 underflow될 수 있어 원래 log mean도 보존한다.

bins의 평균은 서로 독립이 아니다. bin별 SE를 독립이라고 가정해 합산하지 않는다.
전체 S, Q를 재구성해 원래 전체 평균과 moments를 검증한다.
구간 기여비 S_b/S 및 feature의 기여 가중 평균은 관측 진단이다. 이 비율을 최종
확률 estimator에 끼워 넣거나 unbiased feature mean/모집단 mode mass라 부르지 않는다.

## 검증 및 주장 한계

평균 weights의 floor/density identity와 convexity, 직접 sliding-window oracle,
상수 경로/단일 spike 구분, 경계 equality의 오른쪽 bin 처리, ordinary mean 및
zero-outside-bin SE, partition 손상 거부, 생산 코드의 작은 격자 재현을 검사한다.
전체 final 종료 후 각 cell/arm block 0을 replay하고 moments/모든 bin/feature 진단을 대조한다.

단일 fine 격자의 probability reference 개발이다. 기존 q의 M2를 바꾸지 않으며,
all-q, guide-free 탐색 교차검증, mesh/continuum, 총비용 우위와 P2 승인을 대신하지 않는다.
