# Canonical gradient-MH: 자원 배치 대조 pilot

2026-10-10. 첫 target-separation32whole 완료·감사 후, 아래 새32whole 생성 전에 동결.

## 근거와 그대로 유지할 것

첫 pilot risk-only pCN 최악RSE9.071%, 필요586whole; gradient-MH 최악RSE6.464%,
필요298whole였다. gradient로 관측 개선이 있었지만256cap을 넘어 precision 자격이 아니다.
μ만 분리한 것은 risk pCN을 해결하지 못했다. common normalizer RSE도5.35%로 남았다.
현재 작은pilot의 sample variance를 population 값이나 새 모델 우위로 주장하지 않는다.

원래canonical,N32,5frozenq,scaledrisk5개,동결inverse-anchor weights,ordinary unnormalized
SMC estimator,guide-free target-gradient norm cap20,exact forward/reverse MH를 유지한다.
모든 arm의 final mutation 없음, every4 stratified resampling, whole이SE단위.

## 사전 고정4arms ×8whole

| design | particles | quadratic temperature points | mutation/stage | noise scale |
|---|---:|---:|---:|---:|
| more-moves | 1024 | 64 | 8 | .35 |
| more-particles | 2048 | 64 | 4 | .35 |
| more-temperatures | 1024 | 128 | 4 | .35 |
| larger-move | 1024 | 64 | 4 | .55 |

단순steps/particle수 이름으로 mixing 개선을 선언하지 않는다. firstpilot4mut .35는
개발anchor일 뿐, 새arm의 final 또는 통합대조 표본으로 pool하지 않는다.
namespace `canonical-gradient-resource-v1`, design별독립stream; firstpilot와 충돌 검사.

예상whole forward/backward는각508928/509952/517120/254976.
preparation128forward 포함 total14,327,936forward +14,327,808actual backward.
cap18mforward/18mbackward/1800초/RSS4GiB, whole60초. 완료prefix success 판단 금지.
failwhole를 다른seed로 대체하지 않는다. source/anchor/q/weights/config/seed와 모든 비용 보존.

## 판단과 후속

component별RSE/maximum whole/LOO, normalizerRSE, ESS구간/acceptance/ancestry를 확인한다.
RSE1.5%/safety2/min32/max256 shared-whole forecast를 동결된 기준으로 계산한다.
후보는wholecap과forecastforward160m/backward160m/wall1800초 모두 안인 경우만 고려한다.
뒤이어 μ/static corroboration 비용도 더해야 하므로 이조건만으로전체production을승인하지않는다.
feasible후보 중 worst-component CV² ×actual mean whole wall이작은설계를개발용으로선택한다.
동일 accuracy 자격을 얻기 전 speed superiority를 주장하지 않는다.

새선택설계는fresh validation표본으로 μ+5risk의외부mechanism과상호검증해야한다.
reference해결후에도P2 mesh/continuum,bank quality,learned added value는별도관문이다.
