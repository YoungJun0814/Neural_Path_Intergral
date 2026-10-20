# Fine-grid 독립 reference의 bridge/mutation 배분 pilot

2026-10-19. 과거32whole 및G0/G1 결과를 본 이후 새 개발 실험이다. confirmation아님.
2cells×2arms×8독립whole=32. seed namespace fine-bridge-mutation-allocation-v1.
CPUfloat64/8threads/N4096·2048 원래paired target H=(g_f+g_c)/2를 유지한다.
원래 pCN-L full reverse-Gaussian MH correction, drift norm cap20/scale.35,
quadratic fixed β schedule/stratified resampling/product normalizer를 재사용한다.

두arm: particles64, A=64temperature points/4mutations/resample-every4,
B=128points/2mutations/resample-every8. 비최종 transition당 mutation이고 각각
whole joint-forward/backward15,936/16,192, physical31,872/32,384다.
총 physical1,028,096/backward514,048/time-step3,158,310,912.
기존128particle 결과와capacity가달라 직접우위시험이아니다. 두arm사이에서만 입자수와
약15회 resampling 및전체연산량을 맞춰 bridge간격/mutation묶음의 결합변경을 평가한다.
particle당 mutation총량이비슷해도 correlation/mixing/ancestry가같지는않다.

cap phase14400초/unit600초/prep30초/RSS4GiB/physical1.1m. 배터리조회없음.
누락seed교체/유리한prefix선택/결과가좋아질때까지추가실행없음.
원래두result+두receipt를SHA로묶고 producer/runtime를archive한다.

보고: independentwhole SE(입자SE금지), fine/coarse probability,paired차이 비교는
**4cell-arm endpoint**로고정, ancestry/ESS/acceptance,whole maxshare/LOO,
safety3·최소32whole·RSE1.5% forecast, 실제비용. 선택은셀별더작은timestepforecast,
tie arm이름. 선택된arm도새확인전에는qualified아님. 모든P1/P2/training flags=false.
각cell/arm rep0의4whole numericalreplay별도physical128,512/backward64,256.
결과의10%동등성은기존G1과4개개발비교로전부보고하며, 불확실성과불일치를분리한다.

조건부Rao–Blackwell 후보는이번pilot에포함하지않는다. N4096 denseconditioning을
사용하지않으며 빠른조건부solve를증명/검증하지않고실제알고리즘처럼보고하지않는다.
이pilot와다른timing실험을겹쳐실행하지않는다. 커밋·푸시없음.
