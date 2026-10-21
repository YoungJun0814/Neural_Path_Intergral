# 제한 예산 연구 사이클 D0–D4 구현·실행·판정 보고서

작성일: 2026-10-20. 대상은 N32 유한격자 개발 문제이며 연속시간 모델 전체가 아니다.

## 1. 결론

**최종 과학적 판정은 `STOP_UNRESOLVED_BUDGET`이다.** D0 증거 재집계, D1 수학·통계 코어 검증,
D2 전체 frozen-family 실험 및 재현 감사를 완료했다. D2의 층화 primary 평가가능성 기준을
충족하지 못했으므로, 사전 계획대로 D3 신규 학습은 실행하지 않았다.

이는 구현이 중간에 방치됐다는 뜻도, 연구 전체가 수학적으로 불가능하다는 뜻도 아니다.
현재 guide와 이번 고정 예산으로는 새 학습의 성능을 신뢰성 있게 판별할 기반을 확보하지
못했다는 투자 결정이다. 같은 실험을 성공할 때까지 반복하거나 기준을 낮추지 않는다.

- D0: 기존 30개 estimator / 240개 checkpoint를 전수 결합·재집계, 추가 Volterra 평가 0건.
- D1: 고정 Gaussian toy 4개, IID/층화 각각 16회. 수치 적분 수렴 및 대표 8개 표본 재현 통과.
- D2: 2개 셀 × 2개 표본 설계 × 8개 완전 블록, **952,320건** 완료.
- preflight 4,960건, D2 대표 재현 감사 124,000건 포함 신규 Volterra 합계 **1,081,280건**.
- D3: 코드와 조건부 절차는 구현·단위 검증했으나, 실제 신규 학습·confirmation 호출은 **0건**.
  후속 코드 리뷰에서 비용 I/O 및 반복 evidence-grade 집계의 보완 항목도 발견해 생산 준비 완료로 표시하지 않았다.
- P1/P2 및 대규모 학습 자격은 모두 false. 새로운 모델 채택·논문 투고 준비 완료 주장은 없다.

실행 기준: [동결 계획](../plans/BOUNDED_RESEARCH_CONTINUE_PIVOT_STOP_EXECUTION_PLAN_2026-10-20_KO.md).

## 2. 이번에 구현한 것

| 구성 | 구현·검증 범위 |
|---|---|
| `balanced_mixture_bank.py` | 정확 비례 성분 수, 최소 tile 1,240, 마지막 partial chunk, phase/cell/rep/design/role/block/component별 seed |
| `stratified_paired_risk.py` | log-space 양의 충분통계, 교차 모멘트, 알려진 성분 질량 기반 평균·공분산, signed paired 차이와 delta 구간 |
| `direct_second_moment_diagnostics.py` | 기존 final 표본의 X² 블록 평균, leave-one-out, 최대 단일 X² 기여, 전·후반 민감도 |
| `bounded_decision_protocol.py` | 사전 표본 수·배정 상한·중단 기준·4/5 반복 판정·비용 기준 |
| D0/D1/D2 runner | source/config/input hash 결합, 기존 결과 불변, 일회성 append-only 실행·감사 |
| D3 runner/runtime | U/E-IID/E-STRAT 학습·선택 분리, 전체 pilot 후 단일 배정 봉인, fresh confirmation, 누락/미배정 실패 전파 |
| 테스트 | 수기 공분산·signed/no-op·독립 constrained solver·partial chunk·seed·domain/alpha 변조·조건부 전체 mock workflow |

373개 물리 경로 성분의 geometry는 바꾸지 않았다. 이번 변경의 중심은 새 거대 모델이 아니라
**성능을 잘못 좋아 보이게 만드는 불안정한 평가를 걸러내는 통계·실행 계층**이다.

D3가 구현돼 있다는 것과 실제 Volterra 학습 전체가 검증됐다는 것은 다르다. D3의 전체
분기는 단위/합성 mock 검사 대상이고, 이번에는 gate 때문에 생산 실행 증거가 없다.

## 3. D0: 확률 RSE와 M₂ 안정성은 다르다

기존 ordinary estimator X=g/Dq의 8개 독립 final 블록을 재집계했다.
직접 M₂는 E_q[X²]이며, 기존 checkpoint의 sum(X²)로 블록 평균은 복원 가능하다.
그러나 sum(X⁴)가 없으므로 **원시 IID 단위 M₂ SE는 복원 불가**라고 기록했다.
확률 SE를 M₂ SE로 잘못 재사용하지 않는다.

- 30개 직접 M₂의 블록 RSE 범위: **14.47–96.75%**.
- canonical uniform rep2에서는 단일 표본이 관측 X² 합의 **96.96%**를 차지했다.
- high-eta event-KL rep4: 확률 RSE **2.64%**, 직접 M₂ 블록 RSE **27.06%**,
  별도 held-out M₂ raw RSE **75.73%**.

따라서 낮은 관측 확률 RSE 하나를 population variance 개선이나 비용 우위의 증거로
취급할 수 없다. 서로 다른 M₂ 추정치의 차이도 그 자체로 bias의 증명은 아니다.

감사: 240개 checkpoint envelope·identity·count·seed·proposal binding 및 모멘트 재집계 통과.

## 4. D1: 정답이 알려진 문제의 검증

μ는 analytic Gaussian bump 적분, M₂는 Gauss–Hermite 48/96/192와 독립 adaptive quadrature로
대조했다. 네 toy 모두 요구 상대 차이 1e-5를 통과했다. GH192/adaptive 상대 차이는
약 3.12e-14–2.11e-11이다. [-12,12]² 바깥 M₂ 누락 상계는 7.11e-32다.
수치 적분기가 보고하는 오차를 엄밀한 interval certificate로 표현하지 않는다.

각 행은 16회 × 16,000개 표본이다. RMSE는 정답 대비 상대 오차의 반복 RMS이며,
포함 횟수는 점근적 nominal 95% 구간에 정답이 들어간 횟수다.

| Toy | 설계 | μ 상대 RMSE | M₂ 상대 RMSE | μ 포함 /16 | M₂ 포함 /16 |
|---|---|---:|---:|---:|---:|
| A | IID | 2.65% | 4.18% | 15 | 15 |
| A | 층화 | 2.54% | 3.50% | 14 | 14 |
| B | IID | 1.27% | 2.22% | 16 | 16 |
| B | 층화 | 1.12% | 1.87% | 16 | 16 |
| C | IID | 2.55% | 3.06% | 15 | 16 |
| C | 층화 | 3.38% | 5.38% | 13 | 13 |
| D | IID | 24.13% | 62.80% | 16 | 14 |
| D | 층화 | 14.11% | 38.72% | 16 | 15 |

16회 포함률을 정확한 95% coverage 보장으로 부르지 않는다. D처럼 guide에 없는 방향에서는
정답·공식이 맞아도 표본 오차가 클 수 있다. C의 이름만으로 작은 bump가 M₂를 지배한다고
결론 내리지 않았으며, 이번 결과로 그러한 지배 관계를 입증했다고 주장하지 않는다.

synthetic production/quadrature 2,433,350건 + 대표 replay 128,000건 = **2,561,350건**으로
별도 cap 4,194,304 이내다. Volterra 호출은 0건. 생산 계산 wall은 68.86초였다.

## 5. D2: 전체 frozen family 결과

두 셀마다 uniform U + 기존 event-KL 5개 + risk-M₂ 5개 ID를 유지했다.
동일한 density가 된 no-op ID도 제거하지 않았다. 설계별 238,080개 표본에서
μ와 11개 q의 M₂를 같은 경로로 계산했다. q마다 payoff를 다시 계산한 것처럼 비용을
11배로 세지 않았다. 7,214개의 PRNG seed role 중 충돌은 없었다.

### 5.1 요약

| 셀 | 설계 | μ RSE | 11개 M₂ RSE 범위 | μ 포함 gate 통과 수 /12 |
|---|---|---:|---:|---:|
| canonical | IID | 3.61% | 18.47–20.94% | 3 |
| canonical | 층화 primary | 4.07% | 37.14–66.48% | 1 |
| high-eta | IID | 3.99% | 27.01–37.56% | 1 |
| high-eta | 층화 primary | 15.82% | 86.95–92.93% | 0 |

층화 primary의 24개 항목 중 통과는 canonical μ **1개**뿐이다. IID canonical의 3개 통과도
μ, U와 U와 동일한 no-op ID를 포함하며, 독립적인 세 모델의 성공이 아니다.

### 5.2 Primary의 q별 실패 크기

모든 값은 %. LOO는 한 블록 제거에 따른 최대 상대 변화, 단일 기여는 전체 추정치에서
한 표본이 차지하는 최대 가중 비중이다. 사전 기준은 M₂ RSE≤20, LOO≤15, 단일 기여≤10이며
block RSE≤2×raw RSE도 동시에 요구한다.

| 셀 | q ID | raw RSE | block RSE | LOO | 최대 단일 기여 |
|---|---|---:|---:|---:|---:|
| canonical | U | 47.49 | 42.40 | 41.78 | 46.97 |
| canonical | event-kl-0 | 39.38 | 32.69 | 31.27 | 37.98 |
| canonical | event-kl-1 | 47.49 | 42.40 | 41.78 | 46.97 |
| canonical | event-kl-2 | 66.48 | 62.75 | 62.42 | 66.16 |
| canonical | event-kl-3 | 39.08 | 32.97 | 30.47 | 37.19 |
| canonical | event-kl-4 | 52.24 | 46.98 | 46.12 | 51.50 |
| canonical | risk-m2-0 | 37.14 | 30.28 | 28.98 | 35.64 |
| canonical | risk-m2-1 | 48.33 | 42.79 | 42.31 | 47.71 |
| canonical | risk-m2-2 | 60.27 | 56.05 | 55.61 | 59.80 |
| canonical | risk-m2-3 | 39.66 | 34.01 | 32.47 | 38.17 |
| canonical | risk-m2-4 | 45.82 | 40.07 | 38.92 | 44.74 |
| high-eta | U | 89.60 | 88.19 | 87.83 | 89.21 |
| high-eta | event-kl-0 | 88.52 | 86.93 | 86.41 | 87.98 |
| high-eta | event-kl-1 | 88.50 | 86.90 | 86.35 | 87.92 |
| high-eta | event-kl-2 | 86.95 | 85.08 | 84.24 | 86.09 |
| high-eta | event-kl-3 | 89.60 | 88.19 | 87.83 | 89.21 |
| high-eta | event-kl-4 | 92.93 | 92.01 | 91.85 | 92.76 |
| high-eta | risk-m2-0 | 91.08 | 89.88 | 89.62 | 90.80 |
| high-eta | risk-m2-1 | 89.28 | 87.82 | 87.40 | 88.84 |
| high-eta | risk-m2-2 | 87.17 | 85.35 | 84.59 | 86.38 |
| high-eta | risk-m2-3 | 89.60 | 88.19 | 87.83 | 89.21 |
| high-eta | risk-m2-4 | 92.72 | 91.76 | 91.61 | 92.55 |

high-eta μ도 RSE 15.82%, 단일 기여 14.94%로 실패했다. IID/층화 symmetric-difference의
24개 사전 조정 normal/delta 비교 중 high-eta M₂ 11개에서 contradiction flag가 켜졌다.
이는 **중단 veto**이며 두 설계의 실제 bias나 population disagreement를 증명하지 않는다.
특히 상대 차이가 포화되는 영역의 delta 근사와 관측 SE 자체가 극단값에 민감하다.
precision 실패만으로도 D2 중단은 이미 결정된다.

### 5.3 쉽게 해석하면

성분을 매번 골고루 뽑아도 **각 성분 안에서 매우 드문 고기여 경로**를 안정적으로 관측하는
문제는 남는다. 이번 high-eta 층화 M₂는 사실상 단일 표본에 지배됐다. IID 표본이 더 낮은
RSE를 보고했다고 IID가 더 정확하다고 인증할 수도 없다. 큰 기여를 놓쳤을 가능성과 우연히
관측했을 가능성을 함께 고려해야 한다.

층화 자료의 plug-in 분산 분해에서 M₂의 between-label 비율은 약 0.173–0.176%였다.
이 값은 잡음이 있는 성분 평균으로 계산한 **관측 설명량**이지 population 분산 비율이나
잔차 mode의 위치에 대한 증명이 아니다. 이를 geometry 불충분의 하한 정리로 바꾸지 않는다.

## 6. 이론적 재검토

기준 p는 표준 Gaussian 경로 좌표 law, g=g_N은 유한격자 조건부 사건 확률이다.
G=pD_G, q=pD_q로 두면 다음 항등식이 성립한다.

```
mu       = E_G[g / D_G]
M2(q)    = E_G[g^2 / (D_G D_q)] = E_q[(g / D_q)^2]
mu_hat   = sum_j pi_j * mean_i(Y_ji)
Cov_hat  = sum_j pi_j^2 * sample_cov_j(Y) / n_j
```

1. 샘플링 law와 full normalized mixture denominator를 일치시켰다. 뽑힌 성분의 density만을
   denominator로 사용하는 오류, 성분별 평균의 단순 평균, pooled IID SE를 사용하지 않았다.
2. n_j=N*pi_j인 정확 비례 allocation에서 각 행의 적분 질량은 pi_j/n_j=1/N이다.
   이 조건에서만 기존 equal-row event-KL/M₂ 목적함수 재사용이 맞다. 일반 불균등 allocation에는
   동일 논리를 적용할 수 없다. 불균등 수기 예제는 별도로 평균·공분산을 대조했다.
3. pi0=q0=0.1, g≤1이므로 Y≤10, 위험 integrand≤100이다. 절대 모멘트가 유한하다는 사실과
   작은 상대 오차를 이번 표본 수로 보장한다는 주장은 다르다.
4. 정확 비례층화의 population 분산은 IID의 within+between 분산에서 between 부분을 제거한다.
   그러나 독립적으로 뽑은 유한 표본의 **관측 SE가 항상 작아진다**는 보장은 없다.
5. paired 상대 감소율은 기준·후보 공분산을 함께 쓰는 delta 방법이다. signed 차이는 로그를
   취하지 않고 양의 두 모멘트를 안정적으로 스케일해 차감했다. 구간은 점근적 개발 진단이다.
6. event-KL finite-bank 목적의 target weights 정규화는 loss 계산이다. 최종 확률/M₂ estimator를
   self-normalize한 것과 다르다. 최종 ordinary estimator의 정규화 상수를 지우지 않았다.
7. FW gap/독립 solver 비교는 유한 학습 bank의 최적화 검증이다. population 최적해나 coverage의
   증명으로 확대하지 않는다. 이번에는 D3를 실행하지 않아 새 FW 수렴 결과도 없다.

검토·테스트에서 현재 **D0–D2 계산 결과**를 무효로 만드는 이론/수치 오류는 발견하지 못했다.
이는 모든 입력·미세격자·연속시간 한계에서 무오류라는 증명은 아니다.

## 7. 실행 편차와 해석 제한

다음 사항을 숨기지 않고 기록한다.

1. D0 최초 기동은 Windows 명령줄 길이 제한(WinError206)으로 **sampling 이전** 실패했다.
   실패 기록을 보존하고 archive-v2에서 checkpoint 입력 480개의 hash 결합을 유지한 채
   archive 인자 전달만 수정했다. 원본 연구 결과와 seed는 바꾸지 않았다.
2. D2의 최초 직접 파일 실행은 package import 오류로 sampling 이전 종료했다.
   코드 변경 없이 `python -m experiments.post_audit_bounded_frozen_bank_contrast`로 실행했다.
3. 계획 5.5의 **참고용 검정력 표를 D2 전에 별도 저장하지 못했다**. 이를 사전 검정력 검증을
   완료했다고 표현하지 않는다. 후술한 표는 D2 이후, 기존 자료와 D1 toy만으로 계산한
   post-hoc budget sensitivity다. D2/D3 gate·counts·source는 변경하지 않았다. 이 표는 원래도
   통계 gate가 아니지만, 계획의 절차적 편차인 것은 분명하다.
4. D2와 전체 회귀검사가 일부 동시에 실행됐다. D2 resource cap 판정에는 실제 경과 시간을
   사용했지만, 그 wall을 전용 장비의 공정한 모델 속도 benchmark로 해석하지 않는다.
   D3가 열리지 않아 practical speedup 후보 판정은 하지 않았다.
5. gate 차단 후 D3 소스 리뷰에서 `assess()`의 workflow wall이 train/fit/selection/pilot/final
   operation 시간은 합산하지만 checkpoint·배정·최종 결과 I/O를 arm별로 완전히 배분하지 않는
   점을 확인했다. 또한 개별 row의 `dual_design`은 계산하지만 `repeated_decision()`에 양 셀
   4/5 교집합을 집계한 evidence-grade 필드가 없다. 따라서 D3의 production-ready/완전 구현
   주장은 하지 않는다. 이번 D2의 수치·gate·중단 결정에는 사용되지 않은 코드다.
   결과 이후 동결 소스를 조용히 수정하지 않고 그대로 보존했다. 향후 D3를 재허용하는 별도
   계획이 생기면 I/O 배분 ledger와 반복 evidence-grade 집계를 먼저 수정·테스트·재동결해야 한다.

### 참고용 검정력 민감도 — 사후 분석, 실행 기준 아님

고정 효과는 **20% M₂ 감소**, z=3.70869(240개 two-sided 비교)이며 유리한 관측 감소율을
표본 수 분모로 쓰지 않았다. 과거 paired SE를 sqrt(N_old/158720)로 축소하는 stationarity
가정 아래 단일 상대 구간의 normal power proxy는 다음과 같다.

| 셀 | non-no-op 반복 | cap에서 proxy power 범위 | 가정상 80% 단일 구간에 필요한 표본 범위 |
|---|---:|---:|---:|
| canonical | 4 | 0.18–23.32% | 370,018–5,252,843 |
| high-eta | 4 | 0.77–11.86% | 514,779–1,987,554 |

두 정확 no-op 반복은 0 SE를 높은 검정력으로 해석하지 않고 제외·별도 기록했다.
이 계산은 새 층화 q의 실제 검정력, raw/block/absolute/relative의 동시 통과 또는 양 셀 4/5
성공 확률이 아니다. tail이 불안정하므로 표본 수 예측은 특히 신뢰가 약하며 **예산 증설 근거가 아니다**.

D1 toy의 평균 보고 M₂ RSE에서 두 endpoint가 독립·동일 CV라고 가정한 같은 fixed-effect
proxy는 A/B/C에서 99.9% 이상, D에서 약 0.46–0.48%였다. 이는 실제 후보를 학습해 비교한
실험이 아니며, toy가 쉽거나 어렵다는 점을 구분하는 가상 민감도 계산에 그친다.
계산식·입력 hash·행별 값은 `D4_descriptive_review.json`에 보존했다.

## 8. 감사와 기술 검증

| 단계 | 전수 확인 | 수치 replay 범위 | 결과 |
|---|---|---|---|
| D0 | 240개 기존 checkpoint 결합·재집계 | 신규 경로 없음 | pass |
| D1 | 4 toy identity/count와 source archive | 4 toy × 2 designs의 rep0, 128,000개 synthetic | pass |
| D2 | 36개 preflight/production unit, 373 strata counts, cross moments, 전체 요약·gate·source/config | preflight 전부 + 각 cell/design block0, 124,000개 Volterra | pass |
| D3 | gate 거부·mock 전체 흐름·배정/실패 전파 단위 검사 | 생산 replay 없음 | not run |

대표 replay와 모든 난수 표본 전수 재생은 다르다. D1은 모든 비대표 표본을 독립 재생한 감사가
아니며, quadrature 전 계산을 두 번째로 모두 실행한 것도 아니다. D2는 전수 저장 모멘트
재집계와 미리 지정한 대표 bank 수치 재생을 결합한 감사다.

전용 테스트 **33개 통과**. CI 범위 Ruff 통과, Mypy **211 source files 통과**.
5개 신규 runner/runtime의 별도 Mypy도 통과했다. 전체 pytest는 **1,646개 통과 / 709.06초**,
source coverage는 **85%**(22,059 statements, 3,394 missed)다.
정적 검사·단위 검사의 성공이 누락된 비용 ledger 기능까지 입증하지는 않는다.
기존 RequestsDependencyWarning은 별도 환경 경고로,
이번 연구 실행 중 의존성을 임의 업그레이드하지 않았다.

### 결과 anchor

| 결과 | SHA256 |
|---|---|
| D0 archive-v2 | `4b203d0b8d78316698fabe4ca6d81a2e231e97b84202dbc476a5233eef9b91ac` |
| D1 | `d9e0b71c8deba90ab8aa5ccfb2e8887d8dc02f0e29e5c914ff4d59438fc2aaeb` |
| D2 | `323b00268d797b6e70222a748ce500ee4c5f7f1ff9da1e6724acd750cb3ee1bb` |

결과 디렉터리는 각각 `results/post_audit/bounded_direct_m2_v1/session_20261020_archive_v2/`,
`bounded_bank_oracles_v1/session_20261020/`, `bounded_frozen_bank_contrast_v1/session_20261020/`다.
동결 코드·입력 archive와 audit receipt, D3_NOT_RUN 기록을 함께 보존했다.

## 9. 예산·변경 관리

| 항목 | Volterra calls |
|---|---:|
| D0 | 0 |
| D1 | 0 |
| preflight | 4,960 |
| D2 production | 952,320 |
| D2 representative audit | 124,000 |
| D3 | 0 |
| **이번 production + audit 합계** | **1,081,280** |

D3 최대 예산 12,522,240건은 미사용이며 다른 실험에 전용하지 않았다.
D2 본 계산·집계 wall 244.95초로 900초 cap 이내였다. import/archive/분석/회귀검사와
synthetic 작업은 구분한다. 위 숫자는 회귀테스트 내부 physics 호출까지 계측한 전역 합계가 아니다.
source freeze 이후 수치 구현을 결과에 맞춰 수정하지 않았다. 기존 dirty work를 보존했다.
커밋·푸시는 수행하지 않았다.

## 10. 가설별 판정과 다음 투자 경계

| 가설/주장 | 판정 | 이유 |
|---|---|---|
| 기존 final 확률 RSE가 낮으면 위험도 충분히 안정됐다 | 지지하지 않음 | 직접 X² 민감도와 held-out M₂ 불안정 |
| 비례 성분 층화만으로 이번 예산에서 frozen-family 평가를 구제한다 | 사전 기준 미달 | primary M₂ 22개 모두 실패, high-eta μ도 실패 |
| IID 평가가 더 낮은 RSE이므로 더 정확하다 | 판단 불가 | 독립 유한 표본의 tail 관측 차이와 불안정한 SE |
| E-STRAT 학습이 E-IID보다 낫다 | 미실험 | D3 미허용 |
| 현재 dictionary geometry가 본질적으로 부족하다 | 미해결 | 반복 가능한 잔차 특징/하한 정리 없음 |
| 최상위 저널용 새 수학 원리를 확보했다 | 아님 | 현재는 기존 원리의 구현·음성 개발 실험 |

이번 사이클은 여기서 종료한다. 가중치·rank·operator를 더 붙이거나 seed를 교체하는 작업은
이번 결과가 허용하는 다음 단계가 아니다. 별도 연구를 결정한다면 우선 한 가지 질문만
선택해야 한다: **위험을 지배하는 성분 내부의 잔차 경로 특징이 독립 자료에서도 반복되는가?**

그 후속 계획에는 진단 자료와 확인 자료 분리, tail-feature 저장 및 비용 계측,
정확 likelihood 유지, 한 개의 명시적 geometry 변경 또는 알려진 문제의 lower-bound/반례와
실제 Volterra 연결을 요구한다. 추가 sampling·geometry 선택·대규모 학습은 별도 예산과
계획 승인 없이 실행하지 않았다.

이 실패 보고 자체가 박사 논문이나 최상위 저널 기여를 자동 완성하지는 않는다. 앞으로의
논문 기여에는 Volterra 특유의 새 조건/한계 또는 재현 가능한 해결법, fresh task/격자 검증,
동등 정확도에서 학습비 포함 비용 이득과 경쟁 방법 비교가 여전히 필요하다.
