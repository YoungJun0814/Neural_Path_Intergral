# P2: 동일 coverage envelope로 더 미세한 격자 진단

2026-10-10. 이 문서는 아래 신규 표본 생성 전에 동결한다.

## 목적과 변경 불가 조건

앞선 N32–512 진단에서 high-eta의 가격/변동성 좌측 적분 간격 효과가
Volterra driver 근사 차이보다 크게 관측되었다. 학습량 증가는 이 오차를
제거하지 않는다. N1024/2048/4096에서 인접 격자 차이를 새 표본으로 측정한다.
원래 Itô 좌측 가격 적분, BLP hybrid kernel, 분산 보정, 원래 사건을 유지한다.
미래 변동성을 가격 Brownian increment에 곱하는 trapezoidal 변경은 하지 않는다.

- canonical / high-eta 원래 두 셀, fine N1024/2048/4096, coarse N/2.
- 셀·fine격자당 65,536 IID draw, 512 draw/batch, 128 batches. 전체 768 batches.
- r = .4 exact excursion(2/3/4/5/6) + .4 rank-two variance25 mixture + .2 radial mixture.
  각 내부 proposal은 .1 natural Gaussian floor. FFT는 밀도 계산만 가속하며
  Gaussian/conditional sampler와 확률법칙은 기존 corrected-f64 envelope와 동일하다.
- 전체 likelihood는 **fine joint latent에 대해 하나만** 적용한다. coarse white latent는
  fine latent와 독립 augmentation으로 정확한 adjacent BLP coupling을 구성한다.
  augmentation의 원래 Gaussian density는 target/proposal에서 상쇄된다.
- F-C signed difference는 안정적인 Gaussian CDF difference로 계산한다.
- 고정 namespace `p2-envelope-fine-grid-v1`; 이전 표본과 합치지 않는다.
- potential evaluations 786,432; fine+coarse time-step evaluations 1,409,286,144.
  analytic preparation은 potential call 0이지만 준비 시간/메모리는 포함한다.
- cap 800,000 potentials / 1,800초 wall / RSS guard 4GiB.
  batch 최대 wall20초, 준비 최대 wall60초. 완료 prefix를 성공 결과로 사용하지 않는다.
- before/after-batch RSS 검사는 모든 순간의 peak-RSS certificate가 아니다.
- source/config/input/seed/checkpoint hash와 전체 요약 산술을 감사한다.
  각 셀·격자의 첫512표본, 총3,072표본을 numerical replay한다.

## 판정 및 한계

paired delta-method로 relative fine-minus-coarse와 그 SE를 구한다.
10 comparisons Bonferroni normal quantile을 보수적으로 유지한다.
`abs(relative difference)+z*SE <= .1`은 작은 **개발용 인접 차이** 판정일 뿐,
continuum error 상계나 P2 통과 certificate가 아니다. IID sample RSE 자체도
아직 보지 못한 tail 기여를 보증하지 않는다. max-contribution share와 fine/coarse
RSE를 함께 기록한다. 모든 결과에서 전체 P1/P2 및 대규모 학습 authorization은 false다.

좋은 결과가 나와도 신규 grid별 reference 교차 검증, event/risk proposal 재학습,
training-bank ESS/ancestry 검토, 독립 반복, mesh trend와 상대 bias 이론이 필요하다.
bad result면 seed 교체·표본 추가·격자 선택으로 성공을 꾸미지 않고 원인을 기록한다.
