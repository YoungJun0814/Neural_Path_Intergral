# Coverage-envelope recovery: 관측된 실패에 따른 새 사전 고정 계획

2026-10-10. 앞선 high-eta fixed-count 검증은 실패했다. excursion-IID의 μ에서 단일 표본이
17.806985% 기여하고 RSE17.814728%가 됐다. 이 경로의 max standardized Y=3.71754로,
threshold4/5/6 조건부 성분의 support 바깥이었다. logg=-1.863956, log(r/p)=log(.1).
재현 sample index2083, rep5/batch1을 포함한 raw artifact를 보존한다. 표본을 버리거나
임의 clipping하지 않는다. 이 원인 분석은 다음 설계용이며 실패 결과를 사후 수정하지 않는다.

## 물리적 대조 proposal

r=.4*r_excursion+.4*r_collective+.2*r_radial.
excursion은 이전과 같은 정확한 조건부 prior mixture이나 thresholds2/3/4/5/6으로 고정한다.
collective는 동일 model-law (Y_t,next-price) rank2 covariance25 Gaussian mixture다.
radial=.1p+.9N(0,2I). 각 reference가 .1p를 가지므로 전체 r/p>=.1.
모든 overlapping 성분의 density를 합산한다. 선택된 성분 density만 쓰지 않는다.
이 proposal은 학습 q/기존 guide/bank에 기반하지 않는다. 새 사건/쉬운 strike로 바꾸지 않는다.

## 검토해야 할 이론적 범위

세 성분 모두 정규화가 알려져 있으므로 mixture도 정규화가 정확하다.
radial logdensity=-d/2 log2+||z||²/4, deterministic sample scale=sqrt2이다.
중간 수준의 지속적 Volterra 상승과 넓은 Gaussian norm 꼬리를 보완하는 목적이다.
모든 tail mode coverage나 작은 finite-sample RSE를 증명하지 않는다. 참고 경로 하나의 density
증가는 확인하지만 그것을 모든 estimator variance 감소로 일반화하지 않는다.
탐색 결과로 선택한 proposal이므로 향후 journal confirmation과 development를 구분한다.

## 봉인 pilot

- 기존 두 셀, parent0 q, μ/M2(q), 원본 N32, target/defensive floor 불변.
- guide-IID와 envelope-IID 각각8×32768 draw, targets paired.
- natural/proposal-start geometric bridge 각 target/cell별8whole.
- 기존1024particles,64quadratic temperatures,4mutations,pcn.35,
  full/directional/global 0/1/2/3, stratified resampling every4, final mutation 없음.
- namespace coverage-envelope. 320complete units, work17,367,296/cap40m, wall1800초, RSS4GiB.
- 소스/설계/입력/q/seed/work 보존, pilot와 기존 결과 평균 합산 금지.
- 그 후에만 고정 precision 생산 표본 수를 사전 봉인한다. 실패 seed 교체 금지.
- 성공해도 원본전체2셀×5q P1 및 P2 grid error관문을 통과한 것은 아니다.
- latest fine-grid decomposition에서 high eta price-quadrature가 우세했다. 본 reference
  개선은 이 discretization bias를 고치는 방법이 아니므로 해당 후속 작업을 별도로 유지한다.

## Float64 repair amendment (새 v2 실행 전에 작성)

v1 실행 도중 radial scale 생성의 암묵적 float32 연산을 발견했다.
실행을 중단하고 전체 prefix를 수치 계약 미충족으로 무효 처리한다. 원자료/예약을 보존한다.
새 namespace `coverage-envelope-f64`, 새 directory v2를 사용한다. 성분/threshold/weight/표본수/
stage schedule은 바꾸지 않는다. float64 sqrt2 sample scale을 명시적으로 할당하고
독립 PRNG 재현으로 radial subset을 bitwise 검사한다. 모든 이전 실패는 보고서에 함께 공개한다.
