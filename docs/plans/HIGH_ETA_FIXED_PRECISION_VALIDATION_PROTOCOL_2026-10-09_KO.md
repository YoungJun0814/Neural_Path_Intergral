# High-eta parent0 fixed-precision development validation

산술 bridge pilot 완료 후, 새 표본 생성 전에 고정한 검증이다. 기존 모든 실패를 보존한다.
높은 eta 한 셀과 parent0 q만 대상이다. canonical 및 다른 4개 q의 통과로 일반화하지 않는다.
이 셀을 선택한 이유는 pilot forecast상 노트북 예산 안에서 precision 가능성이 있기 때문이다.
저널용 sealed confirmation이 아니며 개발 단계의 정확도 자격 검증이다.

- Original high-eta200-k1, N32, parent0 q 그대로. 같은 mu와 full M2(q).
- guide-IID 및 정확한 excursion-IID 각각 24 independent blocks ×65,536 =1,572,864 draw.
  4,096 checkpoint batch, mu/M2 paired. 다른 방법끼리 sample stream 독립.
- natural-start / proposal-start arithmetic bridge 각각 mu/M2 target별 64 whole SMC.
- particle1024,64점 log-beta(0 및 1e-8..1),4mutations,pcn.35,global every2,
  full/directional/global 고정 배분, stratified resampling every4, final mutation 없음.
- c는 앞서 directional pilot에서 고정한 high-eta excursion-IID 평균을 재사용한다.
  c, anchor hash, q digest, original input hash 및 소스와 이 문서를 새 archive로 고정한다.
- potential work=68,419,712 (prep128 포함), work cap80m, wall1800초, RSS4GiB.
  실패한 whole-run 교체/부분 통과/표본 수 조정/조기 성공 종료 없음.
- 4방법×2target 각각 RSE<=1.5%, maximum share<=1%(IID)/10%(whole SMC),
  maximum leave-one-whole-block-out shift<=5%를 확인한다.
- 4방법의 모든 6쌍×2target을 10% symmetric relative equivalence margin,
  comparisons=30 보수적 normal approximation으로 점검한다. covariance=0(독립 streams).
- 이 qualification은 finite-sample distribution-free coverage 증명이 아니다.
  IID와 whole-run sensitivity를 함께 공개하며 전체 P1/P2/full-training flags는 false로 둔다.
- 감사에서 source/input/q/seeds/raw arithmetic/work/완결성을 검증한다.
  전체 2셀×5q 관문은 별도로 남아 있다. 결과가 실패하면 실패 기록을 유지한다.
