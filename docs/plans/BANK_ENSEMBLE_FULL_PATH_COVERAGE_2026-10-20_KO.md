# Bank ensemble과 전체 경로 기여 진단: 새 고정 실험

2026-10-20. 이전 단계는 main a63825f에 푸시 완료. 이 계획은 새 표본 생성 전에 고정한다.
로컬 CPU float64/8 threads, 배터리 조회/OS 정책 변경/유료 자원 없음.

## 질문과 범위

이전 독립 bank 두 개에서 학습한 weights는 달랐고 high-eta 1은 단일 표본에
제곱 기여가 집중됐다. 같은 dictionary의 단순 평균 weights가 더 안정적인지 확인한다.
또한 top-16만으로 판단하지 않고 모든 최종 경로의 물리 특성별 기여를 기록한다.
새 geometry, q 학습, M2_q, mesh/continuum 또는 P2 승인 실험은 아니다.

## 이론 및 안전 조건

이전 audit된 w_fit0, w_fit1과 w_ensemble=(w_fit0+w_fit1)/2를 그대로 사용한다.
F(w)=E_p[g²/Dw]의 convexity로 F(ensemble)≤[F(fit0)+F(fit1)]/2다.
최고 단일 fit 대비 개선, coverage, finite-sample RSE 개선은 보장하지 않는다.
최종 표본은 bank·이전 final·다른 arm과 독립이다. 각 방법에서 full mixture Dw를
한 번 보정한 ordinary mean g/Dw를 사용한다. self-normalization/clipping 없음.
.1 natural 및 전체 shift floor를 보존한다. 기존 q와 과거 source/결과는 수정하지 않는다.

## 고정 arms와 count

- canonical/high-eta, fine N4096, 원래 G1의 373 means와 물리 법칙.
- arms: uniform, fit0, fit1, ensemble. 앙상블 비율은 1/2 고정; selection 없음.
- 각 cell/arm은 8 독립 blocks × 8,192 draws, chunk 256.
- 최종 physical 2×4×8×8192=524,288회.
- preflight: 각 cell 256 draws, total 512회, 30초/unit.
- 전체 물리 평가 524,800회. 준비 120초, final 180초/unit, 전체 phase 7,200초,
  RSS 4GiB. preflight score×32>180초이면 시작하지 않는다.
- namespace bank-ensemble-full-path-v1. cell/arm/block/chunk/role별 hash seed 분리.
- 마지막 block까지 모든 arms를 실행하고 prefix qualification/선택적 연장하지 않는다.
- 각 cell/arm의 block 0, 총 8개 block replay: 추가 physical 65,536회.

## 모든 경로의 feature와 고정 bins

각 finite-grid 경로의 I=dt Σv_left, t_peak=argmax(v_left)/N,
κ=T max(v_left)/I, 그리고 C32=max(T/32 시간창의 적분 v)/I를 기록한다.
κ와 C32는 해당 유한격자의 관측량이며 grid-independent/연속시간 spike 측도가 아니다.
C32는 FFT가 아니라 prefix-sum으로 계산하고 작은 배열의 직접 sliding-window oracle와 대조한다.

bins는 결과 확인 전에 고정한다.

- I: .25, .5, 1, 2, 4 경계의 6 bins.
- t_peak/T: .25, .5, .75 경계의 4 bins.
- C32: .25, .5, .75 경계의 4 bins.
- 총 96 joint bins. 경계와 정확히 같으면 오른쪽 bin으로 둔다.

모든 sample의 bin count, ΣX, ΣX²를 기록한다. bin μhat_b=ΣX 1_b / n이며
denominator는 bin의 sample 수가 아니라 **전체 n**이다. 합하면 전체 μhat가 된다.
bin SE도 전체 n의 zero-outside-bin 변수를 사용한다. 빈 bin을 확률 0/분산 0의 증거로
취급하지 않고 unobserved로 표시한다. weighted feature ratios는 관측 진단일 뿐
unbiased feature estimator나 unseen-tail certificate가 아니다.

계산량은 재사용한 원래 terminal quantities 한 번으로 계수한다. 별도 feature 호출을
물리 평가로 중복 계수하지 않되 feature 계산 시간은 score wall에 포함한다.

## 판정

raw IID RSE, 독립 block 민감도, 최대 기여, 전체 bin concentration과 top paths를 보고한다.
ensemble vs uniform/fit0/fit1, 두 cells의 총 6개 비교에 조정 equivalence를 적용한다.
관측 RSE² 비는 보장된 variance ratio/total-work speedup이 아니다.
1.5% 및 기존 precision 기준은 유지하지만 통과해도 같은 geometry이므로 전역 자격은 false다.
새 geometry를 bins에서 제안하면 이번 결과는 development 자료이며 새 confirmation이 필요하다.
단일 bank 재학습보다 ensemble이 유리한지는 실제 결과로 판단하고, 실패도 보존한다.

이후 필요한 독립 탐색 reference·all-q·격자 검증은 생략하지 않는다.
