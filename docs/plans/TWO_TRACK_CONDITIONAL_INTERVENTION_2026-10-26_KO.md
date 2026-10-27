# P2 검증과 실제 사건 추정량 개발의 재분리

2026-10-26 America/New_York. 기존 작업은 main `bfaf890`에 푸시했다.
과거 source/result/STOP/gate는 보존한다. 이 문서는 새 개발 실험의 실행 전 계약이다.
잔차57 층화는 주력 방법에서 보류된 보조 후보로 내린다. 기존 계획은 소급 수정하지 않는다.

## 1. 서로 다른 두 목표

- 제출 검증: 원래 N4096 μ RSE 1.5%, 독립 탐색 기준의 일치, signed adjacent-grid
  correction, 남은 continuum 상대 bias, 새 task/독립 학습 confirmation이 필요하다.
- 모델 개발: N32 canonical/high-eta 동일 finite-grid target에서 실제 사건 추정량을
  개선한다. 이 개발 성공은 P2나 대규모 학습을 승인하지 않는다.
- 현재 fine reference의 정밀도 실패와 N32 M2 bank의 불안정성을 같은 문제로 부르지 않는다.
  새 N32 결과를 N4096 참값으로 사용하지 않는다. 모든 전역 qualification은 false다.

## 2. 과거 실패의 재사용 금지

late variance-time-center 특징은 독립 확인 4/8로 요구6/8에 미달했다.
같은 특징/threshold/seed를 바꿔 반복하지 않는다. ESS/ancestry를 정확도 인증으로 쓰지 않는다.
61번 수치 적분 pilot 및 전체 driver 주변화의 precision 실패도 유지한다.
이번에는 quadrature나 새로운 SMC schedule을 추가하지 않는다.

## 3. A: 정확한 conditional resampling으로 변동 원인을 직접 개입

고정 q는 identity-covariance 373-component Gaussian mixture다. group S의 보집합 R을
고정하면 component posterior는
`a_k(z_R) ∝ w_k exp(z_R·m_kR - ||m_kR||²/2)`다.
이 posterior에서 새 label을 뽑고 `z'_S=m_label,S+epsilon`으로 대체한다.
원래 생성 label을 고정하면 일반적으로 q(z_S|z_R)가 아니므로 금지한다.

z~q, z'_S~q(.|z_R)가 조건부 독립일 때, Y=g/Dq에 대해
`E[(Y(z)-Y(z'))²/2]=E[Var(Y|z_R)]`다.
이는 상관 특징 선별이 아니라 정확한 분포 보존 개입이며, total-effect 양은 가산적이지 않다.
큰 group의 값이 큰 것은 좌표 수의 영향일 수도 있어 좌표별 인과/중요도 순위로 부르지 않는다.

사전 group: (57), (61), (57,61), used odd residuals1..61,
even Brownian coordinates0..62, unused63(negative control).
각 셀8블록×1024 q표본. 같은 base를6개 group에 공유하고 base/각 replacement를 평가한다.
총 `2×8×1024×7=114,688` full Volterra evaluations.
log-space로 pair 차이 제곱, μ와 M2, group별 concentration/SE/블록변동을 저장한다.
unstable ratio나 작은 표본의 낮은 차이를 원인 부재로 확정하지 않는다.
이번 assay로 새 group을 사후 선택해 아래 B의 설계를 바꾸지 않는다.

## 4. B: quadrature 없는 실제 사건 추정량 후보

사전 주후보는 (57,61) sign orbit 평균, 단일57/단일61은 ablation이다.
q의 모든 component에서57/61/63 mean=0을 exact 검사한다.
identity covariance와 함께 이 조건은 q=q_R phi57 phi61 phi63 및 Dq의 독립성을 준다.

4개 sign 조합에서 g를 평가한다. 원래 Y=g/Dq, 후보는 `A=mean_sign(g)/Dq`다.
유한 대칭군 평균은 조건부 expectation projection이므로 E_q A=μ, M2(A)<=M2(Y).
단일 sign 평균보다 joint sign 평균의 M2가 크지 않다. 효과는 부호 변동만 제거하며
잔차 절댓값, 다른 좌표, 알려지지 않은 경로 mode의 변동은 남는다.
정확한 적분/zero-variance sampler/새 일반 수학 원리로 포장하지 않는다.

원래 M2 reference는 `mean_sign(g²)/Dq²`(q샘플),
joint 후보 M2는 `mean_sign(g)²/Dq²`다. 단일 후보 M2는 다른 잔차의 sign도 평균한다.
p샘플에서는 같은 분자의 denominator를 Dq 하나로 바꿔 동일 M2들을 평가한다.
`E[g²]`와 `E[g]²`를 혼동하지 않는다. risk의 orbit 평균은 reference의 분산 감소이며
원래 event estimator 변경과 구별한다. sign별 likelihood는 전체 mixture 밀도다.

새 namespace에서 각 셀 q16블록×8192, p16블록×2048을 고정한다.
총327,680 base contexts. q 결과는 실제 baseline/3후보 사건 기여도 함께 평가한다.
p 결과는 독립 geometry의 M2 대조로 쓰되 다른 방향의 공통 누락 가능성을 인정한다.
모든 최종 seed는 A/과거 실험과 분리한다. 고정수 전체 완료 전 채택하지 않는다.

### 빠른 구현과 수치 검토

57/61은 N32의 used variance29/31에만 영향을 주며63은 미사용이다.
두 좌표를0으로 둔 cache.quantities 한 번에서 4개 log-CDF를 재구성한다.
임의 finite context 및 양 셀에서 full simulator와 대조한다. density는 base 전체에서 평가한다.
sign 평균은 logsumexp로 계산하며 확률/contribution clipping은 없다.
적분 bias는 없지만 float64/FFT/CDF 수치오차는 별도 검사하고 무오류를 보장하지 않는다.

## 5. 고정된 개발 판정과 비용

- event 주후보: q raw RSE<=5%, block RSE<=2raw, 최대 한 경로 기여<=10%.
- M2 reference: q/p 각각 모든4endpoint raw 및16block RSE<=20%,
  최대 block 기여<=25%, leave-one-block-out 변화<=15%.
- 주후보 M2 상대감소는 q/p 각각 paired raw 및block normal/delta lower가20% 초과해야 한다.
  family4 comparisons(2cells×2designs)의 Bonferroni z를 사용한다.
- q/p M2의 symmetric relative difference에 family8 comparisons, 10% equivalence margin을
  개발 진단으로 적용한다. 소수반복/미관측tail에 대한 유한표본 coverage 보장이 아니다.
- 모두 충족해야 별도 confirmation 후보가 된다. production/P2 승인 자체는 아니다.
- 방법별 배포비는 warmup 후 별도 고정 AB/BA/AB 순서, cell당4096 새 표본에서
  baseline, joint kernel을 재계산해 기록한다. 이 타이밍은 sampling 정확도/속도우위 인증이 아니다.
  q 샘플링과 geometry 준비 비용도 따로 기록한다. 연구용 모든 ablation 비용도 숨기지 않는다.

## 6. 자원·봉인·감사

CPUfloat64,1thread,RSS4GiB,cloud/새 dependency/OS 정책 변경 없음.
주실험 full-call cap550,000, scalar-orbit cap1,600,000, phase wall1800초.
실행 전 비용 재검토에서 timing의 scalar도 cap에 포함했다. production은
A114688+B327680+timing49152+warmup1024=492544full-equivalent contexts,
orbit scalar CDF는 B1310720+timing98304+warmup2048=1,411,072다.
별도 structural preflight2cells×16contexts×(1profile+4full)=160full,
known-answer/샘플러 테스트는 연구 표본이 아닌 소프트웨어 검증으로 별도다.
감사 cap60,000full/120,000orbit/wall600초: A각셀block0 full replay14336,
B각셀q/p block0 replay20480, 모든array/hash/statistics/seed/결정 재집계.
source/config/계획을 실행 전에 archive로 봉인한다. 실패/serialization오류도 보존하고 덮어쓰지 않는다.

## 7. 결과에 따른 후속

1. joint sign 후보가 실제 비용·독립 위험에서 유망하면 새 confirmation과 fine-grid
   범용 odd-residual mapping 검증 계획을 별도로 만든다. N32 특수식을 N4096에 그대로 쓰지 않는다.
2. 효과가 작으면57 가설/주변화 추가를 확대하지 않는다. conditional intervention의
   나머지 group 결과가 해석 가능할 때만 joint geometry 후보를 새로 설계한다.
3. reference가 여전히 불안정하면 채택 보류다. 더 좋은 seed나 성공 bank만 골라 fitting하지 않는다.
4. P2 트랙은 독립적으로 fine μ/Δ/ref discrepancy의 기존 결과·자원 forecast를 정리한다.
   원래 gate를 낮추거나 이번 작은 후보를 fine qualification으로 바꾸지 않는다.
5. 전체 회귀/static/대표 수치재생 후 결과와 미해결사항을 README와 별도 보고서에 기록한다.
