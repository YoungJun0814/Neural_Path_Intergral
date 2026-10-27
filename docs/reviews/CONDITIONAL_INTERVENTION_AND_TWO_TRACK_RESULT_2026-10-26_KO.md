# 방향 재설정 실행 결과: 실제 사건 추정량과 P2 검증의 분리

2026-10-26 America/New_York.

## 1. 결론

사용자 요청에 따라 기존388개 파일을 main `bfaf8906f2d820b8cff869b8ef8c6bfe628b3ab4`에
커밋·푸시했고 원격 SHA 일치를 확인했다. 해당 커밋의
[GitHub CI](https://github.com/YoungJun0814/Neural_Path_Intergral/actions/runs/40270150111)도 성공했다.
아래 후속 실험은 별도 미커밋 작업이다.

최근의 보조 검증기 개발만 반복하지 않고, **실제 사건 추정량을 바꾸는 부호 평균 후보**와
정확한 조건부 재표본추출 진단을 구현·고정 실행·재현 감사했다.
후보의 expectation 보존과 second-moment 비증가는 수학적으로 타당하지만,
이번 자료에서 위험 정밀도·독립 대조·사건 precision을 확보하지 못했다.
최종 판정은 **`NO_ADOPTION_COVERAGE_OR_PRECISION_UNRESOLVED`**다.

**P2/대규모 학습/production은 승인하지 않는다.** 불리한 seed나 경로를 제거하지 않았고,
조건을 낮추거나 성공할 때까지 표본을 추가하지 않았다.

## 2. 직전 방향 판단에서 보완해야 할 사실

잔차57 층화가 원래 P2를 직접 해결하지 못한다는 판단은 유지한다. 다만 최신 fine 기록을
정리하면 “fine μ 자체의 정밀도가 모든 방법에서 아직 실패”라는 설명은 부정확하다.

| P2 관련 항목 | 현재 증거 | 남은 것 |
|---|---|---|
| N4096 frozen-bank ensemble μ | canonical raw RSE0.902%, high-eta0.907%; block0.766%/1.058% | 같은 frozen geometry의 반복성이지 독립 탐색 검증 아님 |
| guide-free fine SMC | 초기 bridge 대조 최선 관측 RSE도 약5.93%/8.72%; 모든 설정 기존 결합 gate 실패 | 서로 다른 탐색 기준의 정확도·일치 |
| adjacent-grid correction | 일부 개발 진단만 통과; μ의 자격과 다름 | 정밀한 signed correction과 독립 per-grid 기준 |
| continuum 상대 bias | 해결되지 않음 | 적용 가능한 정리/상계 또는 별도 scope 결정 |
| N32 새 모델 | 학습/위험 bank 불안정; 강한 static 대비 안정적 추가 가치 미입증 | fresh 독립 학습·selection·final·risk 검증 |

근거: [fine ensemble/SMC 결과](FRESH_ENSEMBLE_PRECISION_AND_FINE_SMC_RESULT_2026-10-20_KO.md),
[초기 bridge 결과](FINE_SMC_EARLY_BRIDGE_RESULT_2026-10-20_KO.md).
이전의1.5% precision 실패는 원본대로 남기고, 나중의 회복 결과도 함께 반영한다.
서로 다른 grid/estimand/시점의 상태를 하나의 “실패”로 뭉뚱그리지 않는다.

## 3. 왜 또 특징 검색이나 SMC schedule 검색을 하지 않았는가

기존 late-time variance-center 특징은 독립 확인에서4/8로 요구6/8에 미달했다.
같은 특징을 다시 선택하거나 threshold만 바꾸는 실험은 하지 않았다.
새 operator, 큰 신경망, 새로운 SMC β schedule도 추가하지 않았다.

대신 다음 두 작업을 같은 사전 계획 안에서 수행했다.

1. **원인 개입:** 좌표 group을 실제 q의 조건부 분포에서 다시 뽑으면 사건 기여가 얼마나 변하는가?
2. **실제 추정량 후보:** 57/61의 네 부호 조합을 평균하면 같은 사건확률의 추정이 실용적으로 좋아지는가?

[실행 전 계획](../plans/TWO_TRACK_CONDITIONAL_INTERVENTION_2026-10-26_KO.md).
기존 잔차57 층화 계획은 삭제하지 않고 보류된 보조 후보로 남겼다.

## 4. 수학적 계약

### A. 조건부 mixture 재표본추출

원래 q는 identity covariance Gaussian mixture다. group S를 제외한 R 좌표를 고정하면
component posterior는 `a_k ∝ w_k exp(z_R·m_kR−||m_kR||²/2)`다.
이 posterior에서 새 label을 뽑고 S 좌표에 새 독립 Gaussian을 넣었다.
원래 생성 label을 고정하는 잘못된 conditional sampler를 쓰지 않았다.

Y=g/Dq에 대해 조건부 독립인 두 경로 Y,Y'는
`E[(Y−Y')²/2]=E[Var(Y|Z_R)]`를 만족한다.
이는 distribution-preserving total-effect 진단이며 좌표별 물리적 인과성 증명이 아니다.
group 값은 가산적이지 않고, 좌표 수가 다른 group을 공정한 단일 좌표 순위로 해석하지 않는다.

### B. 부호 평균 사건 추정량

모든 component에서57/61/63 mean이 정확히0임을 확인했다.
따라서 q는57/61의 독립 표준정규 인자를 가지고 Dq는 두 좌표의 부호와 무관하다.
원래 `Y=g/Dq` 대신 네 부호의 `A=mean_sign(g)/Dq`를 쓴다.

부호 대칭군 평균은 조건부 expectation projection이므로 E_q A=μ,
`E_q[A²]≤E_q[Y²]`다. 단일57 또는61 평균보다 joint 평균의 M2가 크지 않다.
이것은 알려진 대칭/조건부 평균 원리의 적용이지 새로운 일반 수학 원리 발견이 아니다.
q의 밀도/좌표 차원은 바꾸지 않았지만 **실제 사건 기여 함수는 바꾼 후보**다.

원래 q의 위험 reference는 `mean_sign(g²)/Dq²`, 새 후보 위험은
`mean_sign(g)²/Dq²`다. p에서 동일 위험을 추정할 때 Dq는 하나만 나눈다.
기존 M2 평가의 분산 감소와 사건 추정량 자체의 M2 감소를 구별했다.
empirical normalizer/self-normalization/clipping/numerical quadrature는 쓰지 않았다.

## 5. 고정 실행과 주결과

N32 canonical/high-eta 두 기존 stress 셀에서 실행했다.
원래 spot100/threshold1의 극단 하락 사건이며 일반 pricing/calibration 검증은 아니다.

- A: 셀당8×1024 base,6개 group replacement.114,688 full evaluations.
- B: 셀당 q16×8192 및 p16×2048,327,680 base contexts.
- timing: 고정AB/BA/AB,cell당4096context;별도warmup 포함.
- 새360개 seed role, 사전 최근10개 protocol에서 수집한2,157개 seed-context 정수와 충돌0.
  저장소 전체의 모든 과거 난수 독립성을 증명한 것은 아니다.
- 총80개 unit을 모두 완료했다. prefix 성공 판정이나 사후 sample extension은 없었다.

### 실제 사건 추정량: q 표본의 관측 RSE

각 행131,072개 base context, 동일 경로의 paired 후보다.
아래는 **사건확률 평균의 관측 RSE**이며 M2 reference RSE와 다르다.

| 셀 | 원래 | 57부호 평균 | 61부호 평균 | joint 부호 평균 |
|---|---:|---:|---:|---:|
| canonical | 7.22% | 7.18% | 7.09% | 7.06% |
| high-eta | 5.05% | 36.90% | 6.40% | 36.36% |

주후보 joint는 양 셀 모두 사전5% 개발 precision에 실패했다.
high-eta의 원래5.05%도5%로 반올림해 통과시키지 않는다.
관측 RSE 악화는 population 분산 비증가 정리의 반례가 아니다.
같은 orbit이 기존 기여에서는 거의 보이지 않던 큰 값을 드러낼 수 있다.

### 독립 위험 평가

| 셀/샘플링 | 원래 M2 관측 RSE | joint M2 관측 RSE | joint 관측 위험 감소 | 최대 단일 원래-risk 기여 |
|---|---:|---:|---:|---:|
| canonical/q | 44.64% | 46.02% | 3.12% | 34.57% |
| canonical/p | ≈100% | ≈100% | 52.28% | ≈100% |
| high-eta/q | 99.04% | 98.78% | 49.87% | 99.04% |
| high-eta/p | 99.20% | 99.20% | 0.37% | 99.20% |

두 셀×4endpoint의 q/p equivalence 판정은 모두 미달했다.
q/p의 전체4endpoint precision·block집중·LOO 복합 기준도 두 셀 모두 미달했다.
일부 paired risk ratio의 SE는 작아 보이지만, 같은 거대 기여 하나가 양쪽을 지배하면
ratio만 안정돼 보일 수 있다. 이를 population 감소율 확립으로 채택하지 않는다.
사전 원칙대로 reference accuracy 조건 실패가 채택을 막았다.

### A의 개입량은 아직 신뢰할 수 있는 순위가 아니다

`E[Var(Y|Z_R)] / E[Y²]`의 관측 비율:

| 셀 | 57 | 61 | 57/61 | used odd residual 전체 | Brownian 전체 |
|---|---:|---:|---:|---:|---:|
| canonical | .336 | .087 | .367 | 3.117 | .893 |
| high-eta | .029 | .032 | .052 | 1.066 | .767 |

모집단에서는 각 비율≤1이다.1을 넘은 관측치는 별도의 큰 replacement 기여와 불안정한
M2 분모로 인한 유한표본 경고이며, 정리 위반이나 “300% 제거 가능”이라는 뜻이 아니다.
assay의 M2 RSE 자체가 canonical65.55%/high-eta26.49%였다.
따라서 이 표로 Brownian/odd/residual 중 주원인을 확정하지 않는다.
미사용63의 개입 차이는 모든 경로에서 정확히0으로, 음성 대조군은 통과했다.

## 6. 극단 경로의 full-law 검증

주실험 후 각 셀×q/p에서 최대 original-risk orbit과 최대 절대 event 변화를 선택하고,
8슬롯을 별도 source/config로 동결해 full simulator로 재평가했다.
일부 슬롯은 동일 경로로,8개 독립 발견이 아니다.
이는 **사후 선택한 수치 감사**이며 독립 confirmation이나 발생 확률 추정이 아니다.

high-eta/q의 선택 경로는 z57=−4.73948,z61=.01158이었다.
z57 부호를 양으로 바꾸면 log g가 약−297.6에서−1.435로 변했다.
4orbit의 log Dq는 동일한−1.03777이었고, 사용 분산의 최대값은 약.414에서131.60으로 변했다.
이 경로가 joint event의 큰 기여와 original-risk 평가의 약99% 집중을 만들었다.

최대 full-law log 차이는2.132e-14였고,63 변경의 payoff 불변성도 통과했다.
preflight의 별도32context×4orbit에서는 최대6.822e-13이었다.
따라서 **이번 극단 기여는 단순한 slice/density 구현 불일치로 설명되지 않았다**.
모든 입력의 수치 무오류나 전체 물리 law의 독립 검증을 주장하지는 않는다.

57의 큰 절댓값이 실제로 중요해지는 사례는 얻었지만,canonical 최대risk 경로는
57/61을 바꿔도 거의 변하지 않았다.57만의 수정이 모든 셀의 주문제를 푼다고 할 수 없다.

## 7. 실측 비용과 계산 장부

동일4096context의 AB/BA/AB timing에서 median joint/raw scoring wall 비율은
canonical1.319/high-eta1.178이었다.sampling/setup은 별도로 기록했다.
동일 accuracy의 fixed-precision total-wall 우위를 보여주는 측정은 아니다.
이번에는 통계 조건이 실패했으므로 plug-in 개선율×비용으로 winner를 고르지 않는다.

| 범위 | full-equivalent context evaluations | orbit scalar CDF |
|---|---:|---:|
| 주실험A/B/timing/warmup | 492,544 | 1,411,072 |
| structural preflight | 160 | 128 |
| 대표6unit replay | 34,816 | 81,920 |
| 사후 극단 경로 감사·특징 재계산 | 80 | 32 |
| **합계** | **527,600** | **1,493,152** |

full-equivalent는 cache.quantities/log_probability 호출 경로 수다.4개의 저렴한CDF를
4회의 full Volterra simulation으로 세지 않았다.density FLOP나 source 봉인까지 포함한
전체 operation 수가 아니다.단위/회귀 테스트 내부 물리 계산도 이 연구 장부와 별도다.
주실험wall15.19초는 source/import/setup을 제외하고 preflight/저장/통계 집계를 포함한다.
CPUfloat64/1thread였다.production 중 전체pytest를 동시에 돌리지 않았다.

## 8. 채택·중단과 다음 우선순위

**채택하는 기반:** 정확한 conditional mixture resampling,sign-projection 후보 구현,
full-law 음성 대조군,event/M2 구분,잔차·Brownian group 개입,전체 비용/실패 보존.

**채택하지 않는 것:** joint sign을 production default/P2 돌파구로 지정하는 것,57이 전체
문제의 주원인이라고 확정하는 것,낮은ratio SE만으로 성공 선언,성공seed까지 반복하는 것.

다음 우선순위는 다음과 같다.

1. **현 후보 확대를 멈춘다.** N4096 sign 구현 확장이나57/61 수치 적분의 대규모화를
   이 결과만으로 실행하지 않는다.57 층화도 자동으로 주력에 복귀시키지 않는다.
2. **분포 설계를 재검토한다.** 다음 후보는 극단 경로 하나 주변에Gaussian을 놓는 것이 아니라,
   여러 시점의 국소 분산residual 대진폭과Brownian 결합을 정규화 가능한 밀도로 다뤄야 한다.
   예를 들어 자연성분을 남기는Gaussian covariance의residual-block 확장은 후보지만,
   아직 선택·구현·검증하지 않았다.전체 시점의 일률 확대 낭비와 고차원density penalty도 검토한다.
3. **새proposal 확인 단위를 고정한다.** 여러 새bank,동일target,full likelihood,
   strong static baseline,train/selection/final 분리,전체비용,미사용confirmation이 필요하다.
   이번 극단 경로는 설계에만 쓰고 성공 확인 데이터로 쓰지 않는다.
4. **P2 트랙은 독립 기준 부족을 주과제로 둔다.** 기존fine ensemble의0.9%precision을
   더 낮추기보다 별도geometry의qualified 기준 확보가 우선이다.
   기존SMC schedule 재검색이나 수천whole 자동실행은 하지 않는다.새 후보의 공통blind spot을
   작은known-answer와finite-grid에서 검토한 후fine 비용을 산정한다.
5. 그래도 독립risk/accuracy가 안정되지 않으면 성능우위 모델 주장 대신
   조건부분해·오차진단 연구로scope를 좁히는 결정을 명시한다.문서와 테스트 증가를
   모델 성능·독창성·최상위지 제출 준비 완료와 혼동하지 않는다.

## 9. 저장·재현·검증

[주실험 결과](../../results/post_audit/conditional_intervention_v1/session_20261026/result.json),
[실행 전protocol](../../results/post_audit/conditional_intervention_v1/session_20261026/protocol.json),
[대표 재현 감사](../../results/post_audit/conditional_intervention_v1/session_20261026/audit.json),
[극단 경로 감사](../../results/post_audit/conditional_intervention_v1/session_20261026/selected_extreme_audit/result.json).

- 80array hash·전체moment·전체판정을 재집계했고,지정6unit physics replay가 완전히 일치했다.
- 주실험snapshot1,143파일은 감사 시점에 불변이었다.후발 감사 코드는 별도snapshot에 저장했다.
- 주실험result SHA256: `d4d1215077eef40fc842f0f78668bd7ab742cc79379646b7c3779826d55b0619`.
- 주실험source SHA256: `992adb0c8f343abc7d2b0627a5955ac4804c57b0390d26af1eb734f50e3f5080`.
- 극단감사source SHA256: `c687fe5a12e1604ff3b46f524a8915edc645977c591ab5035fa96b1fe51aee68`.
- 새core/runner 전용22개와 극단선택1개 테스트를 실행 전에 통과했다.
  Ruff와 새3파일Mypy도 통과했다.
- 최종 전체pytest **1,872개 통과/386.90초**,source statement coverage85%
  (23,050statements/3,448missed),새core94%(78/5)였다.
- CI 범위Ruff,Mypy221source files 및 새runner2개 별도Mypy가 통과했다.
  [최종 검증receipt](../../results/post_audit/conditional_intervention_v1/session_20261026/validation/regression_review.json),
  [전체pytest 로그](../../results/post_audit/conditional_intervention_v1/session_20261026/validation/pytest-output.txt),
  [정적 검사 로그](../../results/post_audit/conditional_intervention_v1/session_20261026/validation/static-output.txt).
- 기존RequestsDependencyWarning은 환경 변경 없이 보존했다.이번 새 변경의remote CI는 미확인이다.
- 공개 대상180파일의strict JSON87개,archive/NPZ82개(2,451members),로컬링크174개와
  완료artifact171개의Git filter 이후 byte 보존을 확인했다.위험 archive 경로·100MiB 이상파일·
  알려진credential pattern·trailing whitespace 검출은0이었다.이는 비밀정보 부재의 전수 증명이 아니다.
  [공개 파일·동결 소스·완료 로그 검토](../../results/post_audit/conditional_intervention_v1/session_20261026/validation/integrity_review.json).

검사 범위에서 명백한 구현 불일치를 발견하지 않았다는 것과,모든 입력의 무오류·미관측tail 부재·
population효율·논문 완성을 증명했다는 것은 다르다.
