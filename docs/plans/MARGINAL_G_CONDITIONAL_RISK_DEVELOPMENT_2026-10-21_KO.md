# 고정 q의 conditional marginal-G — bounded 개발 실행계획

2026-10-21 America/New_York. 최신 batch 결과의 다음 우선순위를 실행한다.
성공을 보장하거나 P2 gate를 완화하지 않는다. commit/push는 범위 밖이다.

## 1. 고정 estimand와 이론 검토

N32 canonical/high-eta 두 기존 셀, 기존373성분 사건q를 그대로 고정한다.
natural mass0.1, 나머지0.9/372이며 identity covariance다.
U=local61, unused63은 표준정규로 남기고 X는 나머지62개 좌표다.
57은 X에 포함한다. 이번에는joint2D 적분을 구현하지 않는다.

```
p=p_X phi(U)phi(z63), q=q_X phi(U)phi(z63), Dq=q_X/p_X,
m2(X)=E_phi(U)[g(X,U)^2], f(X)=m2(X)/Dq(X),
M2(q)=E_pX[f(X)], Y_G=f(X)/DG(X).
```

목표는 q의 probability estimator가 아니라 **고정 q의 M2 평가 정밀도**다.
자유로운 marginal optimum은 `G_X ∝ p_X f`; 수치m2를 사용하므로 exact unbiasedness 주장은 없다.
임의의 H를 sampling하는 경우의 `sqrt(E[g^4/DH])/Dq` 최적형과 혼동하지 않는다.
같은 q의 원래 M2에 m1²를 쓰지 않는다. 근거는 기존 조건부기대값/분산분해다.

새 standalone core는 normalized target fitting weights `f/Ds`를 **학습에만** 쓴다.
최종 M2는 ordinary mean `m2/(Dq DG)`이며 self-normalization을 하지 않는다.
샘플러/밀도는 같은 완전한 혼합분포를 사용하고 원래q의 defensive mass는 유지한다.
static finite-grid density이며 adapted drift나Girsanov theorem의 구현으로 부르지 않는다.
정확한 실수 density에서 Dq,DG≥0.1, m2≤1이므로 Y_G≤100이다.
이는 이 family에서 유한 분산을 확보하지만, 희귀 target 대비 작은 상대오차나 tail coverage를 보장하지 않는다.

## 2. 후보 family와 known-answer 선행 검증

training X에서 weighted two-cluster Gaussian을 기존 검증된 fitter로 생성한다.
cluster feature rank4, covariance rank4, shrinkage0.5, eigenvalue[0.25,4],
최소 cluster mass0.01, 최소2점, natural mass0.1을 고정한다.
unused61/63의mean은정확0, covariance directions도정확0으로full64에embedding한다.
cluster pruning은기존fitter의고정규칙이며결과에서그사실을보존한다.

새geometry는G평가자에만들어간다. 모든후보는 `G_alpha=(1-alpha)q+alpha Gfit`,
alpha∈{0,0.25,0.5}. reference/density의floor는최소0.1p다.
alpha 선택은 별도selection bank에서 아래의 ordinary second-moment를 최소화한다.

```
E_S[m2²/(Dq² Ds DG_alpha)].
```

한 표본의 f²와 f를 잘못 바꾸지 않는 test, covariance 임베딩/density불변성,
sampler전변환knownanswer, target식, concentration/precision veto,
JSON전체payloadroundtrip/NaN거부/예산선결제를먼저검증한다.
실제연구실행이전에plan/theory/core/test/runner/seed를sourceZIP에봉인한다.

## 3. 새 training bank와 선행 중단 gate

각셀2독립rep, rep당8block×256=2048개의fresh q draws. 총8192contexts.
같은형태의적분을batch128이하로수행한다. 모든좌표/profile/logdensity/logm2/error를보존한다.
과거262profile은학습·selection·새확인에재사용하지않는다.

각bank에서logweights=`logm2-logDq-logDs`(여기S=q)를계산한다.
fit하기전에다음을모두요구한다.

- weightedESS≥32 /2048, maximumweightfraction≤0.1.
- 8block의각기여를기록하고최대blockfraction≤0.5.
- 새적분의near-multiple/범위오류/envelope/Jensen실패는하나라도있으면STOP.
- mean targetmass의가중numerical errorproxy≤1e-6.

**개별moment가tail기준상대gate를못통과하더라도삭제하지않는다.**
고정tail upper와order-difference를logerror로보존해전체ordinarymean의오차proxy를전파한다.
tail-dominated tiny contribution의전체가중proxy가1e-6이하일때만bank-level수치gate를허용한다.
이는사전에명시한weighted aggregate numerical criterion이며rigorous bias certificate가아니다.
최대단일relativeerror/개별미해결수도같이보고한다. 구조적criticalpoint실패를이방법으로면제하지않는다.

네bank중하나라도실패하면 **Gfit/selection/final 전부실행하지않는다.**
모든네bank의실패/성공을보존하여사후좋은bank만선택하지않는다.
조건미달자료를클러스터실제학습성공으로부르지않는다.

## 4. 독립적인 raw reference 진단 — gate실패여부와무관하게정해진작업

각셀 raw G=q와 raw G=p 두방법을 각8block×512=4096freshpaths로계산한다.
전자는 g²/Dq², 후자는g²/Dq이며 **같은고정q의M2**를추정한다.
서로다른밀도에서의IID IS이지SMC reference나population정답은아니다.
residual numerical integration을사용하지않아독립적인오류경로대조가되지만,
두방법모두tailmiss할수있으므로일치만으로정답인증하지않는다.

둘다raw/blockRSE≤0.2,maxpathfraction≤0.1,blockLOO≤0.15일때만reference-qualified.
asymptoticlog差/combinedSE≤3.291을consistency경고기준으로고정한다.
저정밀도구간의넓은일치를성공으로세지않는다. 같은q의mean자체를개선성능으로비교하지않는다.

## 5. gate통과시에만실행할selection/final

네bank통과후각fit의별도S=G_0.5에서512selectiondraws.
각alphaordinarysecondmoment의weightedquadratureproxy≤1e-6을요구한다.
최소logobjective동률1e-12에는작은alpha선택. 선택에finalseed는사용하지않는다.
selectiondensity는후보마다바꾸지않고같은S로전수평가한다.

최종baselineq/selectedG를각8block×128=1024freshcontexts로독립평가한다.
새training/selection/final/reference의seed는namespace/셀/rep/role/block/path-label로분리한다.
각endpoint의raw/blockRSE≤0.2,maxpathfraction≤0.1,LOO≤0.15,
aggregatequadratureproxy≤1e-6을모두요구한다.
minimum2rep/2cell은 개발 최소 반복이지 publication confirmation 수준이 아니다.

네 fit 모두 선택 alpha>0, 양 arm과 두 raw reference가 모두 accuracy-qualified,
reference와 mean consistency, 고정 n의 block variance proxy가 각 fit에서20% 이상 개선이면
`CANDIDATE_FOR_SEPARATE_CONFIRMATION_ONLY`다. 어떤 경우에도 P2/D3는 false다.
아니면 해당 STOP/NO_ADOPTION을 기록한다. variance proxy는 population 우위의 증명이 아니다.
fixed-precision total-wall 비교는 하지 않으며, 자격을 얻은 별도 계획만 후속 후보로 삼는다.

## 6. 비용·감사·중단

production cap: physical40,000 / scalar120,000,000 / wall1800초.
training8192+raw reference16384=24576physical.
전체 gate 통과 시 selection2048+final8192로 최대34816physical.
profile은1context1full 계산, raw reference도1draw1full 계산이다.
batch padding/endpoint/warmup은 scalar에 과금한다. root/density 연산은 wall에 포함하지만 scalar node로 세지 않는다.

audit cap: physical3000 / scalar10,000,000 / wall600초.
training 각 rep block0의 처음16개, 총64 full profile을 재생하고 raw reference는 각 셀/방법
block0을512개씩 총2048 full 재생한다. 전체 저장 coordinate의 density/score/통계/seed/판정을 재계산한다.
selection/final이 있으면 각 fit selection의 처음8개(32)와 각 fit/arm final block0의 처음16개(128)를 재생한다.
audit physical 최대2272로 cap 이내다. fit 수치/parameter도 training bank에서 재생한다.
profile full 재구성은 위 고정 개수만 하되, 원래 padding을 맞추려고 각 해당 block의 첫128개
저장 profile 전체를 batch 재적분한다. 이 scalar 비용도 audit cap에 포함한다.

source/setup/training/fit/selection/reference/final/각 artifact I/O/aggregation/final I/O를 별도 기록한다.
end-to-end receipt는 최종 I/O 후 추가하고 실패도 기록한다. audit/test 비용은 별도 공개한다.
one-shot directory/no overwrite. 모든 저장 payload를 writer 전에 JSON 호환화하고 writer도 try/except 안에 둔다.
실패·미실행 예산을 다른 target에 유용하지 않는다. 관측 후 n/threshold/rank/cluster 증감 금지.
source hash/full regression/lint/types와 결과의 이론 재검토를 수행한다.

## 7. seedと主張の境界

독립 namespace `conditional-marginal-G-development-v1`을 사용한다.
이번 전체 stream 간 및 직전 Q1의32 seed와 충돌을 확인한다. repository 전체 seed의 망라 검사는 아니다.
다른 PRNG seed는 이론적 완전 독립의 증명이 아니라 재현 가능한 stream 분리 구현 계약이다.
기존 Q1 STOP, batch diagnostic-only 채택, P2/D3 보류를 유지한다.
이번 성과는 limited development의 채택/반증 판단까지다. top journal 달성, exact continuous law,
오차 없는 numerical M2, 실질적인 모델 속도 개선을 선취하지 않는다.
