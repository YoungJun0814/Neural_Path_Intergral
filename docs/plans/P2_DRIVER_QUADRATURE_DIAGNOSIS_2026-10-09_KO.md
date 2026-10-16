# P2 격자 차이의 driver / price-quadrature 분리 진단

기존 finer-mesh pilot를 관측한 후 세운 탐색적 진단이다. 새 표본을 쓰더라도 journal
confirmation으로 승격하지 않는다. 중간 payoff 정의나 불리한 결과를 사후 변경하지 않는다.

같은 fine Gaussian innovation와 augmentation에서 F=fine payoff, C=원래 coarse BLP payoff,
M=fine volatility를 coarse-left time에서 취하고 fine Brownian increments를 쌍으로 합쳐
가격 Euler/variance integration만 coarse 간격으로 계산한 payoff를 정의한다.
M은 diagnostic counterfactual이지 coarse BLP 법칙이나 새로운 최종 모델이 아니다.
F-C=(F-M)+(M-C)인 samplewise identity를 활용한다.
모두 독립 price Brownian을 정확한 Gaussian CDF로 적분하며 동일 fine p/r likelihood 하나를 쓴다.

- fine N=32,64,256,512; 원본2셀, 비교당65,536 draw, batch2048, 총256단위.
- proposal은 앞선 exact excursion mixture와 동일. 새 identity prefix `quadrature-driver`로
  기존 표본과 seed를 공유하지 않는다. augmentation은 독립 standard Gaussian.
- F-M을 price-quadrature 변화, M-C를 driver-discretization 변화로 명명한다.
  비선형 interaction은 이 순서에 할당되므로 유일한 causal decomposition으로 주장하지 않는다.
- stable signed log-CDF differences, paired ratio SE, samplewise triangle identity를 점검한다.
  0 signed correction의 log magnitude는 null로 저장 후 -inf로 복원하며 값을 clipping하지 않는다.
- potentials3×65,536×8=1,572,864, cap2m, wall600초, RSS4GiB.
  전체 time-step 작업도 기록한다. 절대 연속시간 bias의 추정량은 아니다.
- source/config/input SHA 및 raw checkpoint를 보존한다. bounded first-batch replay도 수행한다.

가격 적분이 우세하면 training dimension 증대 전에 temporal quadrature/refinement를 고친다.
driver 근사가 우세하면 BLP kernel와 exact-grid Gaussian driver 대조부터 한다.
어느 경우에도 fine mean의 낮은 관측 variance만으로 reference/continuum qualification을 선언하지 않는다.
