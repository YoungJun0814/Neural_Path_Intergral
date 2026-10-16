# Coverage envelope fixed-count precision validation

2026-10-10. corrected-f64 envelope pilot 완료 후 fresh 검증 표본 생성 전에 동결한다.
원본 high-eta200-k1 N32 parent0 q만 대상으로 한다. 실패한 앞선 검증과 합치지 않는다.
pilot SHA256=b7074b2a135415bdac0cf62ec42494eab561f027e44a5b046b114f3a1c0d4f44.
이 셀만 고른 이유는 정밀도 목표에 대한 pilot forecast가 로컬 resource budget 내에 있기
때문이다. canonical 실패를 대신하거나 전체 P1/P2 관문을 열린 것으로 표시하지 않는다.

## 변경 불가 실행

- guide-IID / corrected coverage-envelope IID 각각96×32,768=3,145,728 draw.
  batch4096, block당8batch. mu 및 M2 paired.
- natural-start geometric bridge: mu176 / risk176 independent whole SMC.
- proposal-start geometric bridge: mu64 / risk64 independent whole SMC.
- 1024particles,64 quadratic temperatures,4mutations,pcn.35,
  mutation0 full-pCN /1 global /2 rank2-pCN /3 global,
  stratified resampling every4 stages, final transition mutation없음.
- phase cap160m potential evaluations /3600초/RSS4GiB. expected128,680,064 incl prep128.
- 별도 namespace `envelope-fixed-precision-v1`. snapshot/config/input/q/pilot anchor hash 보존.
- IID batch별 log_sum/log_sum_squares/log_maximum/count를 저장하며 full raw vectors는
  memory에 누적하지 않는다. batch moments는 exact frozen seed replay로 복원 가능하다.
- block96개 각각의 mean과 whole SMC를 독립 inference unit으로 sensitivity 분석한다.
- 각방법·target RSE<=1.5%, maxshare<=1% IID/10% SMC, max LOO block shift<=5%.
- 모든6method-pair×2targets의 10% symmetric relative equivalence를 comparisons30으로
  normal delta-method development 판정한다. distribution-free certificate나 journal confirmation이 아니다.
- source/archive/raw-moment checksum/seed/q/grid/accounting 및 선택한4IID batch numerical replay 감사.
- 고정 count 전체 완료 전 성공 종료 금지. 실패할 경우 sample 추가/seed 교체/관문 완화 금지.
- 성공도 parent0 high-eta 개발 검증으로 한정한다. 전체 2셀×5q reference, mesh
  accuracy, learned-model added value 및 bank quality는 별도의 필수 잔여 조건이다.
