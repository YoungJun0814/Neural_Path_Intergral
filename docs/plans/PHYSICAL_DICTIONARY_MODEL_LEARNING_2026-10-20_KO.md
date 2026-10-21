# 다음 우선순위: 경로 dictionary 보존형 사건 모델의 가중치 학습

2026-10-20. 기존 채택/중단 결정을 반영한 실행 전 고정 계획이다.
커밋·푸시는 하지 않는다. 기존 결과·source archive·gate는 수정하지 않는다.

## 1. 채택/중단의 구현

- 채택: 작은 N32 원래 사건 모델 개발과 fine-grid 인증을 분리한다.
- 유지: 새 train/selection/final/heldout, full-mixture ordinary IS, 모든 실패 기록.
- 중단: K2 보정을 production winner로 등록하지 않는다. 이번 실험에 부모 SMC,
  추가 Gaussian fit, β schedule 검색, 새 operator, 큰 신경망을 넣지 않는다.
- 다음 가설: 기존 373개 물리 경로를 Gaussian 두 개로 압축하지 않고 유지하면,
  같은 bank에서 학습한 적은 수의 mixture weights가 정적 균등 분포보다 유용한가?

이 실험은 새 dictionary geometry 발견이나 N4096 benchmark가 아니다.
N32의 기존 두 stress 셀(spot100, threshold1)을 사용한다. 일반 pricing 실용성,
연속시간 정확성, 최상위 저널 적합성을 자동 주장하지 않는다.

## 2. 동일한 세 모델

모든 모델은 mixed-window PhysicalWindowGuide의 같은 373개 identity-covariance
Gaussian means를 사용한다. amplitudes[2,4,6,8], price[0,2,4], natural mass .1.

1. `uniform`: 원래 G. 나머지 .9를 372shift에 균등 배분한다.
2. `event-kl`: 같은 IID G train에서 event-KL를 최소화한 weights.
3. `risk-m2`: 같은 IID G train에서 **실제 사건 estimator M2**를 최소화한 weights.

학습 domain은 w0=.1, wj=.1/372+.8 vj, v가 simplex다. 따라서 모든 기존
component가 양의 확률을 유지한다. event/risk의 표현력, bank, 최대 최적화 예산이 같다.
정적 baseline에는 불필요한 fit/selection 비용을 부과하지 않는다.

event-KL와 risk-M2는 같은 목적함수가 아니다. event-KL는 정규화한 train event
weight로 −log(q/p)를 평균하고, risk-M2는 ordinary mean g²/[(G/p)(q/p)]를 쓴다.
학습용 scale normalization을 최종 IS self-normalization으로 혼동하지 않는다.
두 목적은 이 fixed dictionary에서 convex지만 finite bank optimum은 population
optimum이나 미관측 경로 coverage를 보장하지 않는다. 신규성도 별도 입증 대상이다.

## 3. 고정 새 표본과 선택

2cells×5independent train/selection/final repetitions, 총10fits. static의5회는
fixed geometry의 새 sampling 반복이며 독립 학습5회가 아니다.

- train: IID uniform G16384개. 두 fit이 같은 bank를 공유.
- optimization: 동일 최대256FW iterations, line search40, gap tolerance1e-6.
  조기 수렴 시 실제 iteration/time을 공개한다. 같은 최대 예산이지 같은 실제 시간은 아니다.
- selection: 별도 IID uniform G16384개. 각 fit에 대해
  qα=(1−α)G+αq_fit, α∈{0,.25,.5}의 empirical M2로 선택한다.
  tie는 작은 α를 우선한다. α0은 정확한 uniform/no-op이다.
- 모든 weights 동결 후 각 arm 새 final8blocks×32768=262144개.
- 별도 heldout IID uniform G131072개(8equal blocks)에서 세 q의 M2를 공통 경로로 평가.
- train/selection/각 arm final/heldout 모두 새 namespace·별도 seed.
  과거 성공·실패 표본을 재사용하지 않는다. prefix 성공 판정, seed 교체, optional stopping 없음.

uniform은 모든 learned 후보와 **동일 geometry/capacity**이므로 이번 대조는
같은 표현에서 학습 목적의 추가 가치를 묻는다. 과거 K2 실험과 bank 수 등이
다르므로 두 실험 차이를 dictionary 보존의 단일 인과 효과로 주장하지 않는다.

## 4. 정합성과 판정

모든 q,G≥.1p, final contribution≤10, heldout risk contribution≤100.
또한 α≤.5이므로 qα≥.5G, M2(qα)≤2M2(G). 이 상한은 개선 보장이 아니다.
sampler와 density는 전체 normalized mixture를 사용한다. component denominator,
likelihood clipping, estimated normalizer 삽입은 금지한다.

- 비교는3pairs×10repetitions=30개 family. raw IID 및8block의 paired absolute
  difference와 relative M2 reduction에 covariance-aware Bonferroni normal/delta 진단.
- 양 단위 모두 adjusted lower가0보다 크고 non-no-op일 때 개발 감소 신호.
- final 개발 precision은 rawRSE≤5% 및 blockRSE≤2rawRSE.
- 양 셀에서 각각5회 중4회 이상이 감소 신호+final 개발 precision을 충족해야
  후속 후보. 이것은 confirmation이나 기존1.5%reference gate가 아니다.
- μ 참값 없이 같은 target의 M2 순서는 비교 가능하다. 그러나 heldout 자체도
  희귀 기여를 놓칠 수 있다. M2 RSE·최대 기여·LOO·block 민감도를 모두 공개한다.
- selection에서 좋아진 loss를 heldout 성능으로 보고하지 않는다. no-op의 새 final
  RSE가 우연히 좋아져도 모델 개선으로 세지 않는다.
- P1/P2/large-training flags는 모두 false 유지.

## 5. 비용과 제한

CPUfloat64,1thread,batch2048,RSS4GiB. OS·배터리 정책 변경 없음.
fit unit당32768calls, final총786432calls, heldout131072calls.
10반복×950272 + preflight512 = **9503232payoff calls**.
1prepare+2preflight+10fit+240final+10heldout=263units.

준비120초,preflight30초,fit300초,finalblock60초,heldout180초,
전체phase7200초. preflight의 finalblock 자원 예측이 초과하면 시작하지 않는다.
오류/예산초과/불완전 family는 보존하고 자격을 부여하지 않는다.
이번 runner는 단일 실행 프로토콜이다. 이미 내용이 있는 결과 directory를 재사용하거나
완료/중단 unit을 선택 재실행하지 않는다. 중단 시 기존 증거를 보존하며, 필요하면
별도 사전 고정 whole-family 프로토콜을 만들어야 한다. 자동 resume를 제공하지 않는다.
대표2fits+6finalblocks+2heldout replay는 별도524288calls다.

setup, train-bank(+payoff/full-density/component matrix), optimizer,
selection-bank, selection-density scoring, inference wall을 구분한다.
bank/inference 안의 payoff와 full-density 각각의 미세 타이밍은 별도로 측정하지 않는다.
공유 bank의 실제 물리 비용과 각 단독 workflow에 전액 부과한 비용을 함께 기록한다.
heldout 연구 평가는 실제 calls에 포함하되 deployment inference에는 부과하지 않는다.
정밀도가 다른 방법의 고정표본 시간으로 fixed-precision speedup을 주장하지 않는다.

## 6. 검증과 결정

이론/gradient/known-normalizer/정확한 density/무작위 역할 테스트 → source/config/
계획/이론/직전 감사 anchor 봉인 → preflight →263units전체 → summary·seed·work
감사 및 대표 replay → 전체 회귀테스트 → 결과/README 업데이트.

후보가 통과하면 동일 dictionary에서 무엇을 학습했는지 물리 특성별로 분석하고
새 independent confirmation과 N32→더 미세한 격자 이전을 별도 계획한다.
둘 다 통과하지 못하면 같은 dictionary의 weight/optimizer 검색을 반복 확대하지
않고, geometry의 누락과 bank의 희귀 기여를 분리하는 다음 질문으로 이동한다.
학습이 정적 baseline보다 유용하지 않다는 결과도 그대로 유지한다.
