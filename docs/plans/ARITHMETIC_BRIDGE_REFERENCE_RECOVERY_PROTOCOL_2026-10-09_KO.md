# Arithmetic mixture bridge: reference recovery protocol

先행 geometric/natural bridge의 high whole-run RSE를 낮추기 위한 새 개발 pilot이다.
기존 실패와 표본은 유지한다. mathematical oracle를 먼저 통과시킨다.

## 확률법칙과 정규화

b=p 또는 정규화된 physical excursion reference r로 고정한다.
F_beta=(1-beta)b+beta*p*H/c, c>0는 이전 독립 pilot에서 정한 고정 양수다.
Z_0=1, Z_1=E_p H/c이므로 마지막 normalizer에 c를 곱하면 같은 E_p H를 추정한다.
c가 참값과 다르더라도 endpoint 자체는 바뀌지 않는다. c는 정답/성능 증거가 아니라
bridge의 균형을 위한 튜닝 상수다. 현재 run의 결과에 맞추어 c를 변경하지 않는다.

incremental log weight=log(F_right/p)-log(F_left/p).
full 및 rank-two pCN MH ratio=Delta log(F_beta/p).
global r independence MH ratio=Delta log(F_beta/p)-Delta log(r/p).
산술/기하 bridge의 MH ratio를 혼용하면 오류이므로 서로 다른 계산을 사용한다.
SMC product-of-increments estimator는 유지하며 final mean을 self-normalize하지 않는다.
r initial component를 추가했다고 자연 시작 normalizer를 r expectation으로 바꾸지 않는다.

## 봉인 실행

원본 2셀, parent0 q의 mu 및 scaled risk H=delta_q*g²/Dq.
스케일 anchor는 앞서 완료된 directional pilot의 excursion-IID mu/M2 평균이며
config에 숫자/원본 artifact SHA256를 고정한다. risk scale은 M2 anchor에 delta_q를 곱한다.
anchor 파일도 source archive에 포함한다. pilot와 final sample stream은 다르다.

64점: beta0=0, beta_i=10^(-8+8*(i-1)/62), i=1..63.
natural 시작에서 첫 beta를 너무 크게 잡으면 극희귀한 p 표본의 큰 initial weight 때문에
새 병목이 생길 수 있어 1e-8부터 점진적으로 진행한다. 이 schedule은 결과 관측 전 고정이다.
particles1024, mutation4, scale.35, stratified resampling every4 stages,
full/directional pCN과 r-global move 0/1/2/3 배분은 directional protocol과 동일하다.
2bases×2targets×2cells×8 whole runs, 별도 fresh IID 두 대조를 포함하여 총 320단위.
전체 work17,367,296, cap40m/1800초/RSS4GiB. 기존 run과 서로 평균을 합치지 않는다.

Gaussian-bump oracle에서 c=참값의 .1배/10배 양쪽 모두 테스트한다. 이 테스트는
실제 tail coverage certificate가 아니며, 실제 8 whole-run의 낮은 sample variance도
생산 정밀도 증거가 아니다. full q matrix와 독립 fixed-count confirmation은 별도다.
선택/반복 개발으로 인한 multiple-testing 편향을 막기 위해 이 run은 development로 남긴다.

## 다음 결정

natural-base가 충분히 안정화될 경우에만 새 전체 2셀×5q pilot 및 독립 고정 precision run의
allocation을 작성한다. 실패하면 단순히 capacity/training epochs를 늘리지 않는다.
P2의 관측 mesh dependence도 별도 해결해야 한다. reference precision pass가 continuum
accuracy나 learned-model superiority를 자동으로 의미하지 않는다.
