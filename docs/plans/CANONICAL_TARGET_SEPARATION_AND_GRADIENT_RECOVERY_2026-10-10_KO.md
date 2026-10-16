# Canonical: target 분리와 정확한 gradient mutation 회복

2026-10-10. 아래 새 표본을 생성하기 전에 고정하는 개발 pilot다.
P1/P2 통과, 새 q 성능, continuum 정확도 또는 저널 confirmation이 아니다.

## 1. 실패와 가설

원래 5q 전체 allocation과 6-component umbrella를 모두 실제로 실행·감사했다.
canonical guide-free umbrella의 목표 RSE1.5%/safety2 forecast는
full707 / block777 whole로 256cap을 넘었다. 높은 η의 성공 가능성을 canonical에
외삽하지 않는다. 다음 두 가설을 독립적인 새 표본으로 확인한다.

1. probability g와 scaled-risk δg²/Dq의 형상이 달라 공동 umbrella의 tempering이
   risk exploration에 불리할 수 있다. μ를 분리한 risk-only family와 비교한다.
2. 무방향 pCN이 target의 작은 고확률 영역을 찾는 데 불리할 수 있다.
   외부 guide 없이 정확한 target gradient를 사용하는 MH mutation을 비교한다.

두 가설은 원인으로 확정된 사실이 아니다. 원래 q, payoff, 물리법칙은 변경하지 않는다.

## 2. 수학 계약

p=N(0,I64), Hj=δj g²/(qj/p), 0<Hj<=1. frozen 5q.
family weights는 기존 full-q pilot의 inverse scaled mean에서 이미 동결된 weights를
가져온다. risk-only는 그 중 risk 5개를 재정규화하며 새 pilot 중에는 변경하지 않는다.
probability-only는 H=g, weight1이다. μ+5risk control도 같은 동결 weights를 쓴다.

guide-free pCNL-type proposal:

    a=sqrt(1-s²), c=1-a, v(z)=fixed radial norm cap(∇log H(z),20)
    z'=a z+c β v(z)+s ε, ε~N(0,I)

gradient는 전체 Volterra 경로와 exact conditional payoff에 대한 derivative다.
이는 reference mutation이며 final pricing q를 anticipative하게 바꾸는 것이 아니다.
proposal drift만 제한한다. target/payoff/importance weight/contribution은 clipping하지 않는다.

target πβ ∝ p H^β에 대한 full forward/reverse Gaussian MH를 적용한다.
stable log ratio:

    β(h'-h) + β/(1+a)[(z-a z')·v'-(z'-a z)·v]
    + c β²/[2(1+a)] (||v||²-||v'||²).

plain pCN의 βΔh만 쓰면 gradient proposal에서 틀리다. reverse proposal에도
동일한 실제 capped v'를 사용한다. β=0 또는 v=0이면 prior-reversible pCN으로 환원된다.
현재 구현은 finite-dimensional exact MH다. dimension-independent mixing 정리가 아니다.
근거 방법 계열: [Cotter–Roberts–Stuart–White](https://arxiv.org/abs/1202.0709).

SMC는 고정 temperature·resampling calendar와 unnormalized product Zhat를 유지한다.
terminal component estimator는 Zhat ΣWi Hj/H이며 risk는 δj로 나눈다.
sum Wi Hj/H만 보고하는 self-normalized 최종 추정은 금지한다.
gradient callback은 sample-row separable이어야 한다. 물리 evaluator의 batch/singleton
gradient 일치와 finite difference를 검증한다. nonfinite derivative는 whole failure로 보존한다.

## 3. 사전 software gates

- expanded MH ratio와 직접 forward/reverse Gaussian density의 일치, antisymmetry.
- zero drift/β0 환원, norm cap의 실제 양방향 적용, 상세균형 산술.
- known Gaussian stationary law 및 multimodal family의 알려진 normalizer.
- constant family, frozen weights, potential/backward 별도 accounting.
- cached FFT/conditional CDF/q-density gradient의 finite difference 및 row separability.
- 실패·dimension/config validation. 통과 전 pilot 시작 금지.

## 4. 새 fixed pilot

canonical-h005-k1, N32, 4arms×8 independent whole=32wholes:

| arm | target | mutation / stage |
|---|---|---|
| family-pcn | μ+5risk | 8 full-pCN(.35) |
| risks-pcn | 5risk only | 8 full-pCN(.35) |
| risks-pcnl | 5risk only | 4 gradient-MH(.35), drift cap20 |
| probability-pcnl | μ only | 4 gradient-MH(.35), drift cap20 |

1024particles,64quadratic temperatures, every4 stratified resampling, final mutation 없음.
bank/static guide/Volterra-direction/global independence mutation은 사용하지 않는다.
gradient arm의 mutation 횟수가 작다고 비용이 같은 것은 아니다. forward/backward/wall을
모두 별도로 보고한다. namespace `canonical-separation-gradient-v1`; pilot/final/과거 stream과
합치지 않는다. pcn whole당508928forward, gradient whole당254976forward+254976backward.
preparation128; total12,222,592forward +4,079,616backward.
cap16mforward/5mbackward/1800seconds/RSS4GiB, whole당60seconds.
source/input/q/anchor/weights/seed 동결. 32whole 전부 완료한 후에 판단한다.

## 5. 진단, 선택과 후속

component별 mean/RSE/max whole fraction/LOO를 기록한다. whole이 SE 단위다.
normalizer Zhat의 whole RSE와 component ratio 변화를 별도로 기록한다.
상관된 output을 독립 반복으로 세지 않는다. RSE1.5%/safety2/min32/max256의
fresh-validation allocation forecast를 계산하고 arm별 최악 component로 필요whole을 정한다.
실제 forward/backward/total wall을 비교한다. pilot는 개발 선택이고 예측 가능≠정밀도 자격이다.
실패 arm을 삭제하거나 pilot를 final로 재사용하지 않는다.

risk-only 또는 gradient가 개선되고 forecast가 예산 안이면 별도 fresh validation을 동결한다.
개선되지 않으면 negative 결과를 보존하고 incremental ESS/temperature 구간과 mode/tail
coverage를 진단한 뒤 새 독립 메커니즘 또는 finite-grid research 범위를 선택한다.
gate를 임의로 낮추지 않는다. canonical reference가 성공해도 N4096 fine-grid uncertainty와
continuum bias는 별도로 해결해야 한다. bank quality, learned q, 대규모 학습은
해당 reference·mesh 관문 뒤에 진행한다.
