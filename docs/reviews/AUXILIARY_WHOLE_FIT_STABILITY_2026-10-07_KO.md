# D1: 보조 위험분포 전체 재학습 안정성

## 목적과 성공의 의미

원래 사건확률 모델 q를 고정한 채 보조 위험 estimator를 검증한다. 원래 모델의 성능 개선이나 저널용 confirmation과 구분한다. 실패를 지우거나 가장 작은 RSE만 선택하지 않는다.

1차 실험은 2개 개발 셀 × 5개 고정 q × 5개 새 보조 학습 반복이다. 각 반복에 direct-q, event-r, risk-r를 모두 포함한다. event/risk bank는 같은 계산량·K≤2·rank 0·방어 질량 0.1을 사용한다. 각 최종 IID 표본은 131,072개이며 학습·다른 반복과 seed를 공유하지 않는다. 수치를 본 뒤 조기 종료하거나 표본 수를 늘리는 방식은 사용하지 않는다.

표본 RSE는 누락된 희귀 기여를 보증하지 않는다. 따라서 재학습별 M₂ 추정값의 분산을 함께 보고한다. q마다 참 M₂가 달라 서로 다른 q의 평균을 하나의 참값으로 합치지 않는다.

**중요한 분산 분해:** 고정 q에서 E[M̂₂|r]=M₂(q)이므로 Var(E[M̂₂|r])=0이다. 총분산은 E[Var(M̂₂|r)]이며, r 학습은 conditional variance와 tail coverage를 변화시킨다. 각 추정값의 IID SE를 제거한 between-run variance가 양수로 관측되더라도 이를 고유한 학습 평균 변동 또는 학습 편향으로 부르면 틀리다. 이는 작은 반복 수와 불안정한 SE 추정 아래의 진단용 초과 변동에 불과하다. 같은 r에서 독립 final 전체 반복도 평가하면 conditional sampling noise를 더 직접적으로 점검할 수 있으나, 유한 반복만으로 미관측 tail을 배제할 수는 없다.

개발용 안정성 기준은 각 q에서 5/5 risk-r가 RSE≤20%이고, 추정 평균의 max/min−1≤25%인 경우다. 모든 q를 공개하며 실패를 제외하지 않는다. 이 기준은 인증된 신뢰구간이나 oracle 판정이 아니다. D2로 진행하려면 별도 메커니즘과 충분한 정밀도·일치 조건도 필요하다.

## 실패 시 대응 순서

1. 낮은 ancestry/높은 반복 간 변동이면 bank 탐색·mutation을 개선한다. 단순 표본 수 증가로 덮지 않는다.
2. bank는 상대적으로 일관되나 큰 contribution이 남으면 proposal 형상 부족을 검토한다. event/risk에 같은 capacity와 학습 비용을 적용한다.
3. 독립 학습분포들의 사전 고정 동일 가중 ensemble을 검토할 수 있다. 이는 최종 표본을 보고 유리한 r를 선택하는 방법이 아니다. 정확한 full-mixture density를 사용해야 한다.
4. 추가 실험은 새 config·source snapshot·seed·final 표본으로 실행하며 이전 실패 artifact는 보존한다. 개발 성공까지 반복하더라도 미사용 task·새 학습 confirmation을 대신하지 않는다.

## 이론적 체크

고정 q와 정규화된 r에 대해 Y=g²p²/(qr), X~r이면 E[Y]=M₂(q)다. SMC 가중치는 r fitting 전용이며 최종 위험 추정에 self-normalization을 적용하지 않는다. q,r≥0.1p와 g≤1이면 Y≤100이지만 충분한 상대 정밀도를 보장하지 않는다.

동일 가중 ensemble r̄=(1/J)Σr_j에 대해 1/r̄≤(1/J)Σ1/r_j이므로 위험 estimator의 second moment ∫(g²p²/q)²/r̄는 개별 second moment의 평균 이하다. 이는 가장 좋은 개별 r보다 낫다는 보장이나 독립 미관측 모드 탐색 보장이 아니다.

추가 bank 대안으로 frozen auxiliary ensemble의 independence MH 이동을 검토한다. πβ∝p h^β, y~r일 때 log 승인비는 β(log h(y)−log h(x))−log(r/p)(y)+log(r/p)(x)다. πβ(x)r(y)min(1,πβ(y)r(x)/(πβ(x)r(y)))는 x,y 교환에 대칭이므로 detailed balance를 만족한다. 기존 pCN kernel과 사전 고정된 calendar로 합성해도 각 πβ에 대해 invariant다. 임의의 acceptance correction 없는 learned jump는 금지한다. 정상화 상수는 계산할 필요 없다.

사용한다면 이전 개발 bank에서 학습된 모든 r를 동일 가중으로 결합하고, 그 artifact SHA·정확한 guide density·guide 비용을 기록해야 한다. 이전 final 성능으로 guide를 cherry-pick하지 않는다. 새 bank와 새 final로 평가하며, 해당 guide의 이전 학습 비용을 새로운 총비용 주장에서 무료로 처리하지 않는다. 구현·실행 여부는 3차 결과 뒤에 결정한다.

## 실행 결과

1차 완료: 50개 학습 묶음, 150개 최종 estimator, 24,524,800 potential 평가, 463.34초. 모든 계산은 완료했지만 risk-r 정밀도 통과는 canonical 5/25, 높은 η 0/25다. RSE 중앙값은 각각 36.39%, 41.74%다. direct-q/event-r는 두 셀 모두 0/25다. 전체 재학습 안정성은 0/10 고정 q가 통과했다.

risk bank의 island별 초기 조상 수 중앙값은 canonical 4/128, 높은 η 9/128이다. weighted particle ESS 중앙값 326/512, 385/512가 이 독립성 부족을 나타내지 못한다. 따라서 더 작은 sample RSE를 찾아 반복하는 대신 bank 탐색을 먼저 개선한다. 조상 수만으로 실제 혼합을 확정하지도 않는다.

2차는 mutation 2→8, bridge power 4→2인 composite repair다. 변화의 효과를 단일 요인으로 귀속하지 않는다. 성분 수 K≤2, rank 0, 방어 질량 0.1, final 131,072개는 유지한다. 10개 q 전부에서 새 학습 3회씩 실행한다. 이는 탐색적 3회 묶음으로, 최종 5회 안정성 검증을 대신하지 않는다. 예상 potential 23,377,920회이며 상한 24,000,000회·1,200초를 봉인했다. 첫 실행의 throughput을 참고했지만 실행 시간을 보장하지 않는다.

2차 완료: 30개 학습 묶음, 90개 estimator, 23,377,920 potential 평가, 531.70초다. risk-r 정밀도 통과는 canonical 3/15, 높은 η 1/15, RSE 중앙값 37.12%·38.02%다. 안정성 통과 q는 0/10이다. risk bank 조상 수 중앙값은 11/128·15.5/128, weighted particle ESS는 470/512·482/512로 개선됐다. 다만 최종 위험 정밀도 개선을 입증하지 못했다.

3차는 **보조 r만** rank-16 Gaussian 공분산으로 확장한다. 원래 model q의 rank·차원은 변경하지 않는다. event/risk 양쪽에 동일한 K≤2, rank 16, shrinkage 0.5, variance [0.25,4], 방어 질량 0.1을 적용한다. 같은 개선 bank 설계를 사용하되 새 bank·새 최종 표본으로 10개 q에서 3회씩 실행한다. 이전 bank와 paired rank ablation은 아니므로 차이를 한 번의 paired 효과로 주장하지 않는다. 이 실험도 탐색적이며 5회 독립 재검증이 추가로 필요하다.

3차 완료: 30개 묶음, 23,377,920 potential 평가, 565.31초다. risk-r RSE 중앙값은 canonical 22.01%, 높은 η 25.29%, 통과는 7/15·6/15다. 형상 개선 신호는 있으나 안정성 통과 q는 여전히 0/10이다. 이전 묶음과 독립 bank의 탐색적 비교이므로 확정된 rank 우위를 주장하지 않는다.

4차는 위에서 증명한 independence MH 이동을 mutation 8회 중 2·4·6·8번째에 적용한다. 나머지는 기존 pCN이다. 3차의 모든 event/risk r를 q별·종류별 동일 가중으로 합성해 guide를 고정한다. guide를 final 성능으로 선별하지 않는다. 같은 rank-16 family로 새 bank와 새 final을 생성해 각 q에서 3회 검증한다. guide의 학습 비용은 별도 inherited cost로 기록하며, 현재 run 시간만으로 training-inclusive 성능을 주장하지 않는다.

4차 완료: 30개 묶음, 23,377,920 potential 평가, 819.12초다. risk-r 정밀도 통과 8/15·7/15, RSE 중앙값 19.87%·20.39%, 안정성 통과 q는 0/10이다. 전역 이동 승인율 중앙값은 event 약 12–13%, risk 약 3–4%였다. 추가 guide 학습 비용도 있으므로 이를 기본 bank 설계의 우승자로 채택하지 않는다. 정확한 MH kernel은 검증된 대안으로 남긴다. 특히 공통 guide에 조건부인 실험이므로 전체 guide 생성 변동까지 검증한 것이 아니다.

5차는 **공통 guide 없이 완전 새 학습**을 각 q에서 5회 수행한다. 각 반복에서 4개 새 SMC island를 만들고 pooled fit과 island별 fit을 계산한다. 최종 r는 pooled에 0.5, 각 island에 0.125를 부여한 정확한 mixture다. event/risk 양쪽에 같은 K≤10 learned components·rank 16·방어 질량 0.1이 적용된다. Jensen bound는 해당 분포들의 평균 second moment 대비 보장이며 pooled보다 개선된다는 보장은 아니다.

각 fit의 risk-r final은 1,048,576개, event-r와 direct-q는 131,072개다. 이는 risk precision 확보 실험으로 **동등 최종 비용의 speed benchmark가 아니다**. 최종 count는 v5 결과를 보기 전에 고정했고, 현 최종 표본으로 표본 수를 추가 조정하지 않는다. risk-r의 sample count와 family를 함께 바꾸므로 개선을 ensemble 단독 효과로 분리하지 않는다. 예상 potential 84,838,400회, 상한 90,000,000회·4,800초다. 기존 실행 throughput은 비용 참고이며 보장값이 아니다.

각 frozen r의 final을 사전 고정 4개 IID 블록으로 나누고 별도 moments를 기록한다. 이는 총 N에서 나눈 N/4개씩의 독립 블록이지 4회 각각 N개인 실행이 아니다. 전체 N의 IID SE와, 블록 간 큰 기여 집중을 함께 점검한다. five-fit 전체 기준 통과 없이는 안정성을 확보했다고 쓰지 않는다. 독립 risk 기준값·oracle·q 보정 readiness는 별도 관문으로 유지한다.

5차 완료: 50개 전체 학습 반복, 84,838,400 potential 평가, 2,417.93초다. risk-r의 RSE 중앙값은 canonical 12.08%, 높은 η 9.84%이며 정밀도 통과는 각각 17/25, 22/25다. 그러나 두 조건을 모두 만족하는 고정 q는 **0/10**이다. 일부 반복의 RSE가 여전히 약 52–53%이므로 평균적 개선을 안정성 확보로 해석하지 않는다. source archive·정확한 mixture 재구성·블록 moments·16,400개 seed stream 감사는 통과했다. 이는 통계적 인증이 아니다.

## 큰 기여 경로의 재현과 6차 설계

5차에서 각 셀의 최대 RSE 반복을 사후 진단 대상으로 선택하고, 가장 큰 contribution이 포함된 batch의 원래 seed를 재생했다. 재생 batch의 평균은 저장값과 일치했다. 이 선택은 진단용이며 성능 비교나 독립 confirmation으로 사용하지 않는다.

canonical의 최대 기여 경로는 standardized Volterra peak 약 6.13, integrated variance 약 2.99이며, 한 시간 구간이 integrated variance의 약 95.8%를 차지했다. 높은 η의 최대 경로는 각각 약 5.75, 7.07, 99.7%다. 두 경로에서 원래 q의 natural-component posterior는 약 96.0%, 99.9%다. 반면 auxiliary r의 natural posterior는 약 0.035%, 0.962%이므로 "r의 natural branch에서만 문제가 발생했다"는 설명은 맞지 않는다. fitted r가 상대적으로 희박하게 덮는 구조적 excursion을 발견한 것이다. 두 셀의 상위 5개씩만 분석했으므로 모든 중요 모드에 대한 결론은 아니다.

이처럼 한 구간이 지배하는 경로는 별도의 mesh sensitivity 관문이 필요하다는 경고다. 현재 finite-grid unbiasedness를 연속시간 rBergomi 정확성으로 바꾸어 주장하지 않는다. 연속시간 bias의 크기는 아직 측정하지 않았다.

6차는 학습 bank 없이 **결정론적 Volterra/다음 price-innovation Gaussian tilt**를 먼저 시험한다. B_i는 simulator의 left-monitoring Volterra 선형 연산자이고 u_i=B_i/||B_i||이다. 다음 price increment의 white coordinate e_i는 u_i와 직교한다. 평균 이동은 κu_i+γe_i이며 covariance는 identity다. 두 직교한 선형 평균 제약에 대한 최소 에너지 이동이며, Volterra 값을 고정하는 hard conditioning은 아니다. correlated price와 이후 volatility에 대한 효과는 원래 simulator가 그대로 계산한다.

κ∈{2,4,6,8}, γ∈{0,2,4}, 모든 stochastic left-monitoring 시간에 균등 가중하고 natural mass 0.1을 둔다. exact full-mixture density를 사용한다. 이 형태가 모든 excursion을 덮거나 효율적이라는 정리는 없다. 원래 q는 변경하지 않는다.

각 고정 q에서 5회 새 IID 평가, 각 estimator 131,072개, direct-q와 volterra-only 두 대조군을 사용한다. 이 반복은 **고정 r의 sampling 반복**이며 전체 재학습 5회를 대신하지 않는다. 최대 13,200,000 potential·1,800초를 사전 고정했다. 결과에 따라 다음 전체 fresh-fit 실험을 설계하되 6차 표본을 그 실험의 final로 재사용하지 않는다.

6차 완료: 13,107,200 potential 평가, 221.06초다. volterra-only의 정밀도 통과는 두 셀 모두 25/25이며, RSE 중앙값 canonical 3.06%, 높은 η 2.86%, 최대 4.18%·3.34%다. 고정 q별 5회 추정 평균의 max/min−1은 canonical 3.17–10.56%, 높은 η 4.93–7.95%다. direct-q는 두 셀 모두 0/25다. source·density·moment·3,200 seed 감사는 통과했다. 이 결과는 구조적 excursion을 덮는 설계에 대한 개발 신호이지 최종 oracle·미관측 모드 배제·새 학습 안정성의 증명이 아니다.

7차는 각 q에서 event/risk bank와 pooled/island fit을 처음부터 5회 새로 만든다. 공통 **학습된** guide는 사용하지 않는다. 최종 r=0.5 r_fresh-fit+0.5 r_Volterra로 고정한다. bank 설계는 5차와 동일하며 static global MH를 추가하지 않아, 이번 변화가 output safeguard라는 점을 명확히 한다. r≥0.5 r_Volterra이므로 ∫f²/r≤2∫f²/r_Volterra, f=g²p²/q다. 이는 모든 새 fit에 대한 second-moment 상한이며, 관측한 sample RSE를 모집단 보장으로 바꾸지 않는다.

50개 전체 새 학습 반복에서 risk-r final 262,144개, 나머지 direct-q/event-r/static-r는 각 131,072개로 봉인했다. 4개 고정 IID 블록을 기록하며 모든 q에서 기존 five-fit gate를 적용한다. risk 표본 수가 달라 동등 비용 속도 비교는 아니다. 예상 potential 52,070,400회, 상한 55,000,000회·3,600초다. 7차 final은 6차 final과 독립이며 두 결과를 합쳐 confirmation으로 쓰지 않는다.

## 7차 결과와 최종 판정

50개 whole fresh-fit 반복, 200개 final estimator, **52,070,400 potential 평가**, **1,467.57초**다. 새 bank·fit·final을 각 q에서 5회 실행했다. 원래 q는 변경하지 않았다. 학습/final의 8,400개 seed stream, full density 재구성, source/config/q binding, batch/block moments와 비용 감사가 통과했다.

| 셀 | M₂ estimator | final 표본/회 | RSE 중앙값 | 정밀도 통과 |
|---|---|---:|---:|---:|
| canonical | direct-q | 131,072 | 43.23% | 2/25 |
| canonical | event-fit + static | 131,072 | 4.38% | 25/25 |
| canonical | risk-fit + static | 262,144 | 2.57% | 25/25 |
| canonical | static-only | 131,072 | 3.00% | 25/25 |
| 높은 η | direct-q | 131,072 | 58.15% | 0/25 |
| 높은 η | event-fit + static | 131,072 | 4.07% | 25/25 |
| 높은 η | risk-fit + static | 262,144 | 2.53% | 25/25 |
| 높은 η | static-only | 131,072 | 2.93% | 25/25 |

**요청한 전체 보조 재학습 안정성 관문은 10/10 q에서 통과했다.** risk-r의 최대 RSE는 canonical 3.07%, 높은 η 3.04%다. 각 q의 5회 평균 max/min−1은 canonical 2.54–7.90%, 높은 η 5.25–9.43%로 모두 기준 25% 안이다. 실패를 제외하거나 기준을 변경하지 않았다. grid completeness 검사를 추가해 parent 누락이나 반복 ID 중복을 안정성 성공으로 처리하지 않게 했다.

하지만 **learned component의 추가 효용은 입증하지 못했다.** 표본 수가 다른 RSE만 비교하지 않고 N×RSE²의 중앙값을 비교하면 risk-fit+static은 canonical 173.67, 높은 η 168.14이며, static-only는 117.89, 112.31이다. 약 1.47·1.50배 큰 관측 sample-count-normalized variance이며 추가 fitting 비용도 있다. 이는 개발용 점추정 비교이며 confidence-bound dominance 정리는 아니다. risk-target fit은 event-target fit보다는 나은 신호지만, 구조 기반 분포 단독보다 우월하다고 쓰면 안 된다.

같은 q에서 whole-fit risk mean / static mean은 0.9666–1.0279 범위다. 두 estimator는 공통 static component를 사용하므로 이 일치를 **독립 메커니즘 두 개의 reference 검증**으로 세지 않는다. direct-q의 2개 precision pass도 미관측 tail을 배제하거나 accuracy 자격을 준 것이 아니다.

각 셀에서 최대 RSE인 7차 반복의 최대 contribution batch를 원래 seed로 재생했고 저장 평균과 일치했다. 조사한 상위 기여점에서 static-guide / blended-r log-density lift는 log(2) 이하로, density-floor 설계와 일치했다. 그러나 한 구간의 integrated-variance 비중이 약 99%인 경로는 여전히 존재했다. 보조분포가 이를 더 자주 샘플링하게 된 것이지, discretization 민감성을 해결한 것이 아니다. 이 재생도 사후 개발 진단이며 독립 confirmation으로 세지 않는다.

### 해결한 것과 남은 것

- 해결: 고정된 10개 개발 q·N=32에서 보조 위험분포를 완전히 새로 학습했을 때의 경험적 precision/mean stability. 구조적 volatility/price excursion을 덮는 auxiliary safeguard 및 정확한 likelihood 구현.
- 미해결: 독립 risk/reference 메커니즘, 원래 q의 사건확률 추정 효율, 학습된 부분의 추가 가치, bank 자체의 ancestry/mixing 한계, mesh bias, 미사용 셀 confirmation, 독립 연구 재현과 novelty.
- 현재 허용 주장: **finite-grid 개발용 M₂ 평가 안정화**. oracle 인증, original model dominance, top-journal readiness, 연속시간 정확성은 허용하지 않는다.
- 다음 방향: static physical guide를 강한 **보조 위험 기준선**으로 유지하고 별도 메커니즘 reference와 mesh 검증을 우선한다. 학습 복잡도를 늘리기 전에 learned addition이 이 기준선의 variance와 total work를 실제로 개선하는지 반증 가능하게 평가한다. 이번 작업에서는 D2/qα를 실행하지 않았다.

### 구조 기반 보조분포의 이론적 재검토

white coordinate z에서 두 orthonormal 방향 u,e의 선형 평균 제약 uᵀμ=κ, eᵀμ=γ를 만족하는 모든 μ는 κu+γe+w로 표현되며 w는 u,e와 직교한다. 따라서 ||μ||²=κ²+γ²+||w||²이고 사용한 μ가 최소 에너지다. 이것은 Gaussian linear constraint에 대한 명제이며 nonlinear payoff의 전역 최적 proposal 정리가 아니다.

identity covariance Gaussian mean-tilt의 정확한 density ratio는 exp(μᵀz−||μ||²/2)다. static mixture의 r_V/p는 natural mass와 이 비율들의 정규화된 가중합이다. 구현은 rank-0 성분을 vectorize하지만 혼합 성분 전체의 log-sum-exp를 사용한다. 선택된 성분의 density만 사용하지 않는다. 샘플·density·gradient를 이전 component-wise 구현과 비교하는 regression test를 추가했다.

Y=f/r, f=g²p²/q의 expectation은 모든 정규화된 r에서 M₂(q)다. natural mass가 있는 static mixture 및 learned/static blend는 p에 대한 절대연속성과 완전 support를 유지한다. q,r≥0.1p와 0≤g≤1 아래에서 Y≤100이다. 이 bound는 상대오차가 작다는 뜻은 아니다. SMC bank 가중치는 fitting에만 사용하고 final IID 평균에는 self-normalization을 적용하지 않는다.

7차는 static geometry가 모든 fit에 동일하지만 학습된 parameter는 공유하지 않는다. 따라서 고정 q·고정 deterministic geometry에 조건부로 새 bank/fit/final 전 과정을 반복한다. geometry의 task/grid 선택 불확실성이나 q의 최초 학습 변동까지 포함하는 end-to-end confirmation은 아니다. mesh refinement에서 component 수가 증가하므로 현재 비용·상한을 다른 grid에 그대로 외삽하지 않는다.

최종 result의 source archive, q digest, guide 재구성, mixture weights, seed ledger, batch/block moments, 실제 training/inference cost를 별도로 감사한다. 회귀 테스트 통과와 통계적 gate 통과를 서로 대체하지 않는다.

### 회귀 및 입력 검증

전체 pytest **1,125개 통과**(최종 전체 실행 183.65초), 전체 Ruff 통과, mypy **170개 소스 파일 통과**다. 보조분포에 대해 driver linearity, orthogonal minimum-energy constraints, 양/음/0 correlation의 price 방향, 혼합 density floor, component-wise 대비 density/gradient/sample 일치, invalid parameter 거부를 검사했다. runner에서는 static-only 반복을 전체 재학습으로 오인하지 않는지, learned/static full mixture 재구성, global MH guide binding, block moment 변조 거부, 실패 비용 기록을 검사했다.

마지막 수학적 edge-case 검토에서는 **거의 0인 mean을 정확한 natural component로 세어서는 안 된다**는 점을 강화했다. μ≠0이면 exp(μᵀz−||μ||²/2)는 Rᵈ 전체에서 양의 상수 하한을 갖지 않는다. diagnostic tolerance는 유지하되 `defensive_mass`는 tolerance=0의 정확한 natural 성분만 센다. 이번 실험의 natural 성분은 모두 정확한 0 mean·identity covariance라 수치 결과나 raw artifact는 바뀌지 않는다. 이 변경 뒤 관련 module/integration **26개 테스트를 추가 재실행해 통과**(24.38초)했고 Ruff/mypy도 다시 통과했다. 전체 실행 후의 이 작은 hardening과 추가 재검증을 구분해 기록한다.

이는 검사한 범위에서 회귀 오류를 발견하지 않았다는 뜻이다. 모든 입력·연속시간 극한·독립 연구 재현에 대한 무오류 보장은 아니다. 원래 q를 보정하거나 qualification을 개방하지 않았다.

### 대용량 evidence 보관

5차 raw JSON은 125,491,159 bytes, 6차는 131,766,729 bytes다. 상세 parameter와 whole-fit 기록을 지우거나 요약본으로 대체하지 않았다. 원본 JSON은 로컬에 보존하고 Git에서는 세 대용량 raw JSON 경로만 제외하며 `.json.gz`와 `.source.zip`을 보관한다. 5·6차 압축 사본을 해제한 byte stream은 원본과 완전히 일치했다.

- 5차 raw SHA256: `f1c35378f628b59e9099e1113b963f13dcd81a5f2306df26e60ff8b28c09a265`
- 6차 raw SHA256: `760ec55054483a3c2276881e784ba899f72702e0dae7a820a3d3c40502772a78`
- 7차 raw SHA256: `1baf805ebdc3c222c86d9b8dfff115c14576bc0b5fa16095b5cb253b2ea2d852`
- 7차 gzip SHA256: `747acadd89b280e2e953faa99eddca56cc5c2721093bf344347a845029cb8d92`

7차 raw JSON은 390,097,691 bytes이며 gzip은 33,189,001 bytes다. 이 압축 사본의 byte-for-byte roundtrip도 통과했다. 세 raw JSON 모두 수정하거나 삭제하지 않았다.

압축본은 다음과 같이 raw 파일을 복원하지 않고도 감사할 수 있다.

```powershell
python -c "import gzip,json; from experiments.post_audit_r2_auxiliary_risk import audit; print(audit(json.load(gzip.open('results/post_audit/r2_volterra_static_pilot_v6.json.gz','rt'))))"
```

압축은 기록의 수치·source·seed·digest를 변경하지 않는다. 압축 사본 생성이 커밋 또는 외부 업로드를 뜻하지도 않는다.

커밋·푸시는 요청하지 않았으므로 수행하지 않는다.
