# 경로 dictionary 보존형 사건 모델: 전체 결과와 다음 결정

2026-10-20. 사전 고정한 다음 우선순위를 구현·전체 실행·재현 감사했다.
**사건-KL의 관측 precision 개선은 유망하지만, 독립 위험 감소와 반복 안정성을
함께 입증하지 못했다. 따라서 두 학습 모델 모두 production winner로 채택하지 않는다.**
기존 K2 실패, fine-reference 미해결, P1/P2/large-training false를 유지한다.

## 1. 실제로 진행한 다음 우선순위

직전 결과에서 중단한 K2 Gaussian 보정이나 SMC β 검색을 계속하지 않았다.
대신 물리 경로 373성분을 유지하고 **그 가중치만 학습하는 실제 사건 proposal q**를
구현했다. 보조 reference sampler만 바꾼 실험이 아니다.

- uniform: natural .1, 나머지 .9를 372shift에 균등 배분.
- event-kl: 같은 dictionary에서 event-weighted cross entropy를 최소화.
- risk-m2: 같은 dictionary에서 사건 estimator의 ordinary empirical M₂를 최소화.

학습 domain은 w0=.1, wj=.1/372+.8vj, v∈simplex다. 각 component의 mean/covariance와
좌표 차원은 세 모델에서 같고, component를 삭제하거나 두 Gaussian으로 압축하지 않았다.
혼합분포 전체의 mean/covariance는 weight가 달라지면 달라진다.
동일 train bank, 동일 최대256FW/40line-search, 다른 loss/gap 단위를 사용했다.
같은 최대 최적화 예산이지, 같은 실제 시간이나 같은 population 최적성은 아니다.

계획: `docs/plans/PHYSICAL_DICTIONARY_MODEL_LEARNING_2026-10-20_KO.md`.
수학 계약: `docs/theory/PHYSICAL_DICTIONARY_MODEL_LEARNING_CONTRACT_2026-10-20_KO.md`.
core: `src/path_integral/physical_dictionary_learning.py`.
runner: `experiments/post_audit_dictionary_model_learning.py`.

## 2. 사전 고정 실험을 끝까지 실행

N32 두 stress 셀 × 독립 train/selection 5회, 총10fits다. 기존 N4096 결과와는
다른 finite-grid target이며, 일반 pricing 실용성이나 continuum 정확성 검증은 아니다.

- train: 매 반복 새 IID uniform G 16,384개. KL/M₂가 같은 bank 공유.
- selection: 별도 새 IID G 16,384개. α∈{0,.25,.5}를 empirical M₂로 선택.
- 배포 qα=(1−α)G+αqfit. α0은 정확한 uniform/no-op.
- final: 각 셀/반복/모델 새 8blocks×32,768 = 262,144개.
- heldout: 별도 새 IID G 131,072개에서 세 q의 M₂를 공통 경로로 평가.
- 새 namespace와 역할별 seed 사용. 중간 count/seed 변경, 실패 교체, optional stopping 없음.
- CPU float64/1thread. 생산 **263units / 9,503,232payoff calls / 323.560초**.
  phase 시간은 준비/fit/sample/journal을 포함하고 import/source archive·사후 감사·
  회귀테스트는 제외한다. 통제된 저널용 시간 benchmark는 아니다.
- 대표2fits+6finalblocks+2heldout 재현은 별도 **524,288calls**.

모든 final은 전체 mixture likelihood를 쓴 ordinary IS다. 학습 weight 정규화와
최종 self-normalization을 혼동하지 않았다. likelihood clipping/추정 normalizer 없음.
q,G≥.1p, qα≥.5G를 유지한다. M₂(qα)≤2M₂(G)는 안전 상한이지 개선 보장이 아니다.

## 3. 직접 사건확률의 관측 precision

아래는 독립 5회 최종 평가의 RSE 중앙값/범위다. 개발 precision은 rawRSE≤5%와
blockRSE≤2rawRSE를 함께 요구한다. 정확도·불편성·미관측 tail coverage 인증은 아니다.

| 셀 | 모델 | RSE 중앙값 | RSE 범위 | 개발 precision 통과 |
|---|---|---:|---:|---:|
| canonical | uniform | 4.85% | 3.46–17.03% | 3/5 |
| canonical | event-kl | 4.18% | 2.29–6.01% | 3/5 |
| canonical | risk-m2 | 3.96% | 2.86–16.73% | 4/5 |
| high-η | uniform | 10.11% | 2.84–13.98% | 1/5 |
| high-η | event-kl | 4.43% | 2.64–7.03% | 4/5 |
| high-η | risk-m2 | 9.22% | 2.96–19.50% | 1/5 |

event-kl의 high-η 중앙값 감소는 관측된 개발 신호다. 그러나 서로 다른 유한 final
표본의 RSE 비를 population variance ratio나 속도 배수로 부르지 않는다.
canonical risk-m2/rep3의 16.73%, high-η risk-m2/rep2의 19.50%도 제외하지 않았다.

선택 α:

- canonical event-kl [.5,0,.5,.5,.5], risk-m2 [.5,.25,.5,.5,.5].
- high-η event-kl [.5,.5,.5,0,.5], risk-m2 [.25,.25,.5,0,.5].

세 no-op은 새 모델 개선으로 세지 않았다. 동일 q여도 새 final 표본의 RSE는 달라진다.
이전 K2 실험과 train 수/부모 구조도 다르므로, 역사적 표의 차이를 dictionary
보존만의 단일 인과 효과로 해석하지 않는다.

## 4. 독립 M₂ 비교가 채택을 막은 이유

heldout의 Hq=g²/[(G/p)(q/p)]는 같은 target M₂를 추정한다. 3pairs×10=30개
비교에 covariance-aware Bonferroni normal/delta 진단을 적용했다. raw IID와8block
각각에서 absolute/relative 감소 lower가 모두 양수여야 개발 감소 신호다.

**30쌍 중 이 복합 기준을 통과한 비교는 0개다.** 따라서 uniform 대비 감소 신호와
final precision을 함께 통과한 수도 모든 학습 모델/셀에서0/5다. 두 셀 각각4/5라는
사전 기준에 도달한 모델은 없다. normal/delta 근사 자체도 유한표본 tail certificate가 아니다.

heldout 위험 RSE 중앙값:

| 셀 | uniform | event-kl | risk-m2 |
|---|---:|---:|---:|
| canonical | 43.30% | 44.96% | 48.45% |
| high-η | 41.59% | 41.59% | 41.59% |

high-η의 같은 중앙값은 no-op을 포함한5개 값의 순서에서 나온 결과이지 모든
반복의 위험이 같다는 뜻이 아니다. 전체에서 한 heldout 경로가 관측 위험 합의
최대75.86%를 차지했다. 이번131,072개 bank도 신뢰도 높은 oracle이 아니다.

uniform 대비 M₂ 감소 점추정 범위는 event-kl −18.56~45.01%, risk-m2
−12.84~49.09%였다(no-op0포함). 일부 상대 감소 구간만 좋아 보여도 absolute 차이
불확실성이 커 복합 판정에 실패했다. 유리한 구간만 골라 승자로 등록하지 않는다.

heldout/direct 관측 M₂의 중앙 비는 canonical 약.947/1.060/1.169,
high-η 약.179/.456/.181(uniform/event-kl/risk-m2)였다. 이는 관측 제곱모멘트의
비일 뿐 정확한 population ratio나 bias의 증명이 아니다. 양쪽 모두 희귀 기여를
놓칠 수 있다. **사건확률 RSE를 direct M₂의 오차막대로 재사용하면 안 된다.**
현재 merged probability summary만으로 direct M₂의 SE는 복원할 수 없다.

## 5. bank 집중과 최적화 미수렴을 분리

### 자료 집중은 명백하다

| 셀 | event train ESS / 16,384 | uniform M₂ train ESS / 16,384 | 최대 M₂ train 경로 비중 |
|---|---:|---:|---:|
| canonical | 14.82–162.34 | 1.12–16.73 | 94.67% |
| high-η | 11.34–79.69 | 1.14–4.16 | 93.60% |

이는 SMC ancestry가 아니라 독립 IID bank의 contribution-weight ESS다.
16,384개가 있다고 해서 위험 학습에 그만큼 독립적인 유효 정보가 있다는 뜻은 아니다.
표본 수만 두 배로 늘리면 해결된다는 보장도 없다. 이번 namespace가 다른 실험과
ESS가 달라진 것을 원리적 개선/악화라고 단정하지 않는다.

### 최적화는 엄격한 tolerance에 도달하지 않았다

20개 fit 모두256iterations에서 멈췄다. event additive FW gap은
.0711–.0918, risk relative FW gap은 .1003–.1819로, tolerance1e-6 미충족이다.
최적 적합을 완전히 구한 뒤 표현력의 한계를 검증했다고 주장할 수 없다.

다만 “미수렴=아직 학습이 전혀 안 됨”도 틀리다. finite-bank convex objective F의
FW gap Δ는 F(v)−F*≤Δ를 준다. event에서는 보고된 gap이 additive loss bound다.
risk에서는 ρ=Δ/J이고, ρ<1이면 J*≥(1−ρ)J, 따라서 J/J*≤1/(1−ρ)다.
이번 최대ρ에서 이 비는 약1.223 이하다. 이는 **현재 동일 training bank의
고정 domain에 대한 수치적 bound**이며, population이나 선택 이후 qα의 성능 bound가 아니다.
부동소수점으로 계산한 gap을 machine-verified theorem certificate라고도 부르지 않는다.

raw risk training objective는 uniform 시작점 대비 약82–97% 감소했지만 heldout
복합 신호는0이었다. 그러므로 optimizer 반복만 무한히 늘리는 일을 우선하지 않는다.
엄격한 optimizer residual과 bank 일반화·geometry 문제를 별도로 다뤄야 한다.

## 6. 무엇을 학습했는가: 추가 physics 0회 사후 분석

별도 source/anchor에 묶인 `experiments/post_audit_dictionary_learning_diagnosis.py`가
raw fitted weights와 selected weights를 구분해 구조를 분석했다.

- shifted weights를 .9로 나눈 inverse-Simpson 유효 component 수는 raw event
  10.22–35.92, raw risk 3.68–35.42였다. 이는 IS sample ESS가 아니다.
- raw window mass는 event .514–.691, risk .476–.749였다. uniform은 .9×16/31≈.465다.
  point/window template 수가15/16이므로 애초 uniform family가50:50은 아니다.
- raw volatility amplitude8 mass는 약.036–.046으로, uniform .225보다 작았다.
  큰 진폭을 적게 골랐다는 서술이지 큰 진폭 경로가 물리적으로 불필요하다는 증명이 아니다.
- selected mixture는 baseline을 섞어 집중을 줄였다. α0에서는 정확한 uniform으로 복귀했다.

겹치고 상관된 Gaussian template의 mixing weight는 실제 사건의 mode 확률이나
경로 중요도 attribution과 같지 않다. component weight TV도 density TV와 다르다.
window feature의 평균 구간은 causal Volterra 방향의 전체 Brownian support가 아니다.
이 분석으로 새 가설검정이나 qualification을 만들지 않았다.

## 7. 비용과 채택/중단

각 단독 workflow에 full setup/train/fit/selection/fixed final 비용을 부과한 중앙값:

| 셀 | uniform | event-kl | risk-m2 |
|---|---:|---:|---:|
| canonical | 7.489s | 12.938s | 13.168s |
| high-η | 4.579s | 10.052s | 10.194s |

uniform에는 학습/선택 비용을 부과하지 않았다. heldout 연구 평가는 물리 calls에
포함하되 배포 workflow에서는 제외했다. bank/inference 내부의 payoff와 density
미세 시간은 합산 측정했다. 동일 accuracy 자격이 없어 fixed-precision speedup을 주장하지 않는다.

이번 결정:

1. **유지:** 물리적 다중경로 dictionary, ordinary full-mixture IS, 독립 학습/선택/final/
   위험 평가, small-grid 개발과 fine-grid 인증 분리.
2. **채택하지 않음:** 현재 event-kl/risk-m2 weights를 production winner로 등록하거나
   P2/대규모 학습의 근거로 쓰는 것. event-kl은 후속 비교대상으로만 보존한다.
3. **중단 유지:** K2 압축 복귀, SMC β 반복 검색, 큰 operator 추가, 성공할 때까지
   seed/표본/optimizer 조건을 바꾸는 탐색. 이번 음성 결과도 그대로 보존한다.
4. **다음 우선순위:** 새 모델을 키우기 전에 bank의 희귀 기여와 dictionary의
   geometry 한계를 분리한다. 아래 순서로 작은 새 프로토콜을 세워야 한다.

다음 순서는 (a) 완료 checkpoint의 X² block moment를 이용한 **추가 physics0회**
direct-M₂ 민감도, (b) 알려진 Gaussian 정답 문제에서 최적화 gap 및 component
coverage 진단 검증, (c) 동일 frozen geometry에 대해 component-stratified independent bank(층별 IID)와
random-mixture IID bank를 같은 총 budget/전체 likelihood로 대조하는 것이다.
정적 mixture의 층화 estimator는 각 component 평균을 사전 선언한 mixture mass로 합쳐야 한다.
이를 자동으로 n_j/N이라고 두거나 deterministic labels를 기존 equal-weight IID 학습/RSE
식에 그대로 넣으면 안 된다. 층별 표본 수를 반영한 variance와 paired covariance가 필요하다.
학습과 heldout도 별도 층화 family/표본을 사용해야 한다.

층화 자체가 미관측 residual tail을 해결한다고 가정하지 않는다. bank가 안정된 뒤에도
같은 dictionary가 부족할 때만 다중-window 결합/잔차 geometry를 제한적으로 검토한다.
이는 후속 설계 우선순위이며 **이번 실행에서 새 실험을 수행한 것으로 세지 않는다**.
이번 결과를 보고 같은 bank에 맞는 geometry를 고른 뒤 그대로 confirmation이라고
부르는 것도 금지한다. 더 미세한 격자/독립 메커니즘/reference/총비용은 여전히 별도 gate다.

## 8. 감사와 재현

생산 결과: `results/post_audit/dictionary_model_learning_v1/session_20261020/result.json`.
SHA256 `3964e74bb620da10b34a5c8256a46286b3a46a8eb5ab676031314fe647273033`.
감사 상태 `dictionary_model_source_geometry_seed_fit_selection_final_risk_replay_pass`.

계획/이론/source/config/직전 결과+감사를 봉인한 뒤 실행했다. 263개 start/finish,
work/cap/identity/seed/binding을 검사했다. 모든 fit의373weights/정규화/floor/α연결,
모든 final/heldout count,8block 구조/summary/RSE와 block paired 판정을 검사했다.
대표2fits+6finalblocks+2heldout의 수치가 재현됐다. 대표 replay이지 모든 raw draw의
독립 재계산은 아니며, 비재현 반복의 raw covariance는 별도로 복원하지 않았다.
실행 중 단일 실험 프로토콜의 source를 수정하거나 일부 unit을 교체하지 않았다.

사후 분석: `results/post_audit/dictionary_learning_diagnosis_v1/session_20261020/result.json`.
SHA256 `6393fe9415a52059fb6c827a4e0d6119a25f500bcd532df694eca11c48f9dd6f`.
별도 source/생산 result+audit SHA에 묶었고, 전체 재계산 감사가 통과했다. 추가 physics0회.

최종 검증:

- CI와 같은 전체 pytest: **1,613 passed, 363.74초**, source line coverage **85%**.
- CI 범위 Ruff 통과. Mypy **207source files** 통과.
- 새 생산/사후 분석 runner 직접 Mypy **2files** 통과.
- 생산 source archive1,088entries 중 수치 runtime460files의 실행 후 변경0건.
  사후 분석 archive1,088entries 중 수치 runtime461files의 실행 후 변경0건.
- 두 result SHA가 각 audit receipt와 일치함을 재확인했다.
- gradient/line derivative 유한차분, 알려진 Gaussian 적분, defensive floor,
  full density, seed 분리, no-op tie, 비용 누락, 불완전 family 거부,
  count/weight/α binding 변조 거부를 테스트했다.
- `git diff --check` 통과. HEAD `9cafe59b689ce8a015a7e79fea5f8ba2a8dba00c` 유지,
  staging은 비어 있다. 커밋·푸시는 하지 않았다.
- requests/urllib3/charset 의존성 경고는 기존 환경에 남아 있다. 이번 frozen
  실험 중 의존성을 업그레이드하거나 OS/전원 정책을 바꾸지 않았다.

검사 범위에서 발견한 구현·감사 문제를 수정하고 재검증했다는 뜻이지 모든
입력에서의 무오류를 증명했다는 뜻은 아니다. 특히 과학적 성능 미입증, tail coverage,
fine-grid/continuum, 독립 confirmation, 저널 제출 수준의 신규성은 여전히 미해결이다.
모든 전역 자격은 false다.
