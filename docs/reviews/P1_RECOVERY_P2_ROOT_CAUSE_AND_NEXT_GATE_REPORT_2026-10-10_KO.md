# P1/P2 회복: 원인, 구현, 결과와 다음 관문

작성일: 2026-10-10. 완료 결과 재검토: 2026-10-16.
새 연구방법의 우위나 논문 완성 보고서가 아니다.
원래 두 개발 셀, 고정 N32 q, exact conditional payoff, ordinary IS 계약을 유지한다.
실험의 과거 실패·수치 계약 위반·고정 예산을 보존한다.

## 1. 핵심 판단

현재 병목은 모델 용량보다 **reference의 희귀 기여 coverage와 격자 오차**다.
작은 pilot의 낮은 RSE가 큰 fresh validation에서 깨진 실제 사례를 확보했다.
그 원인을 seed replay로 찾아 normalized proposal의 집중 영역을 보강했다.
또한 mesh 차이를 price integration과 driver approximation의 순서 의존적 진단으로
분리했다. 더 많은 N32 학습은 이 격자 차이를 없애지 않는다.

corrected-f64 envelope 고정 검증과 guide-free block kernel pilot는 완료·감사했다.
원래10q의 fresh allocation pilot도 완료·감사했으나 전체 production allocation은 실패했다.
추정량을 바꾸지 않고 계산을 공유하는 frozen-family umbrella pilot도 완료·감사했다.
canonical에서는 이 방법도 allocation 실패였으나 high-eta는 예산 내 forecast를 얻었다.
따라서 원래 high-eta 5q 전체를 별도의 새 fixed-count validation으로 검증했다.
첫검증은guide-free risk4개의RSE미달로실패했다. 그실제변동을반영한새fixed재검증은
높은η의원래5q전체development precision/corroboration을통과했고numerical감사도통과했다.
canonical target분리/gradient-MH pilot와자원대조도완료했고128temperature 후보를선택했다.
canonical staticμ의known-component층화는실패했고exact last-pair conditioning을선택했다.
새 canonical 580 units의 3-method/five-q 전체검증은 완료·감사했으나
guide-free μ RSE 1.5066%가 고정 한계 1.5%를 넘어 전체 실패했다. 나머지 17개
정밀도 검사와 16개 일치 검사는 통과했다. 반올림해 통과시키지 않는다.
N1024/2048/4096 mesh 진단과 numerical replay 감사는 완료했지만 작은 차이 기준은 실패했다.
paired-gradient N2048/4096의 새 32 whole 개발 진단은 네 인접 차이 검사를 통과했다.
이는 μ 자체의 accuracy qualification이나 continuum bias 인증이 아니다.
전체 P1, P2, 대규모 학습, 저널 제출 준비도는 아직 인증하지 않는다.

## 2. 실제 해결한 구현 항목

1. Exact collective covariance reference: causal standardized Volterra direction와
   그 다음 orthogonal 가격 innovation의 rank2 covariance를 확대한다.
2. Exact conditional-excursion reference: Gaussian truncation의 알려진 정규화 상수와
   full overlapping-mixture density를 사용한다. empirical normalization은 없다.
3. Natural/proposal geometric SMC와 arithmetic bridge: 각각의 target 및 MH ratio,
   endpoint normalizer scale을 따로 검증했다.
4. State-independent rank2 pCN mutation: 올바른 Gaussian prior reversibility를 유지한다.
5. Exact adjacent coupling: coarse local integral은 C1+L2; augmentation을 포함한
   joint importance likelihood는 fine density 하나다.
6. FFT scoring / block coupling: 작은 격자에서 dense density·sampler·coupling과 비교했다.
   finer grid에서 likelihood 법칙을 바꾸지 않고 계산량을 줄인다.
7. Streaming moment validation: raw vectors를 계속 메모리에 누적하지 않으면서
   exact frozen seed replay, block sensitivity, complete-unit checkpoints를 유지한다.

이는 새로운 신경망 q의 구현이나 성능 개선이 아니다. q는 그대로 동결되어 있다.
reference r 개선과 원래 event estimator q 개선을 분리한다.

## 3. 실패를 통해 확인한 coverage 문제

원래 conditional-excursion(4/5/6) pilot의 high-eta μ RSE는 약1.60%였다.
fresh fixed-count validation에서는 μ RSE가 **17.81%**로 올라갔고, 한 draw가 전체
μ contribution의 **17.807%**를 차지했다. 같은 방법의 M2 RSE는 .91%였다.
따라서 risk evaluator가 안정적이라고 probability reference까지 안정적인 것은 아니다.

해당 draw를 exact seed로 복원했다:

| 속성 | 복원값 |
|---|---:|
| 최대 standardized Volterra monitor | 3.71754 |
| monitor >=2 / >=3 / >=4 개수 | 18 / 7 / 0 |
| integrated variance | 1.56282 |
| peak variance | 8.89965 |
| latent squared norm | 111.16596 |
| conditional event probability | 약 .155058 |
| 기존 log(r/p) | -2.302585 (defensive .1만 기여) |

이는 support=0이 아니라 집중 영역 바깥의 **중간 크기·지속형 excursion** 문제다.
경로를 제거하거나 contribution을 clipping하지 않았다.
새 envelope는 excursion(2/3/4/5/6), rank2 variance25, isotropic variance2를
.4/.4/.2로 섞는다. 같은 draw의 density는 약747.79배 증가하고 contribution은
1.55058에서 .00207355로 줄었다. **한 경로의 진단이며 population variance의
747배 개선을 뜻하지 않는다.**

근거:

- `results/post_audit/high_eta_precision_validation_v1/session_20261009/result.json`
- `results/post_audit/reference_coverage_failure_analysis_v2/session_20261010/result.json`

앞선 failure-analysis v1은 latent regeneration 4096건과 potential call194건을
같은 work field에 더했다. 원본은 보존하고 v2에서 이를 분리했다. 올바른 potential
count는194이지4288이 아니다. 재생성의 시간도 비용이며 비용이 사라진 것은 아니다.

## 4. 발견하여 고친 수치 계약 위반

envelope v1의 radial scale은 scalar torch.where 때문에 float32 sqrt2가 생성되었다.
log-density는 float64 variance2를 전제하므로 작은 오차라도 sampler-density 계약 위반이다.
해당 실행을 중단했고 prefix를 무효 처리했다. v2는 명시적 float64 scale을 사용하며
radial subset의 독립 PRNG reconstruction/bitwise equality regression을 추가했다.
v1의 interrupted reservation·결과를 삭제하지 않았다.

근거: `results/post_audit/p1_coverage_envelope_recovery_v1/session_20261010/INVALIDATED_NUMERICAL_CONTRACT.md`.

## 5. Corrected envelope pilot: 전역 개선은 아직 아님

각 IID method 8×32768 draw, SMC method/target/cell당8 whole-run.
전체320개 단위, potential17,367,296회. source/input/seed/summary/work 감사 통과.

| 셀 / 방법 | μ RSE | M2 RSE |
|---|---:|---:|
| canonical guide-IID | 6.794% | 2.250% |
| canonical envelope-IID | 10.379% | 4.961% |
| canonical natural-start SMC | 1.378% | 5.843% |
| canonical proposal-start SMC | 2.922% | 1.417% |
| high-eta guide-IID | 1.638% | 1.935% |
| high-eta envelope-IID | 2.219% | 3.183% |
| high-eta natural-start SMC | 4.559% | 4.795% |
| high-eta proposal-start SMC | 1.832% | 1.706% |

canonical IID 결과는 새 envelope가 오히려 더 불안정했다. 성공 경로만 강조하지 않는다.
pilot RSE로 production 성공을 보장하지 않는다. high-eta parent0만 fixed-count fresh
검증을 먼저 선택한 것은 local resource forecast 때문이다. canonical을 대체하지 않는다.

근거: `results/post_audit/p1_coverage_envelope_recovery_v2/session_20261010/result.json`.

## 6. 고정 정밀도 검증

상태: **high-eta parent0 개발 정밀도 기준 통과**, 전체2016units 완료 및 감사.
결과는 `envelope_precision_validation_v1/session_20261010/result.json`.
IID 두 방법 각각3,145,728 draw; natural-start SMC target당176 whole,
proposal-start SMC target당64 whole. 기대 potential128,680,064회.
actual potential128,680,064회, phase wall2839.79초(47.33분).
source/q/seed/moment/summary/accounting 및16384draw numerical replay 통과.

| 방법 | μ mean | μ RSE | M2 mean | M2 RSE |
|---|---:|---:|---:|---:|
| guide-IID | 4.67304e-6 | .497% | 6.15560e-7 | .561% |
| envelope-IID | 4.64477e-6 | .655% | 6.14559e-7 | .899% |
| natural-start guided SMC | 4.65888e-6 | .709% | 6.04907e-7 | 1.030% |
| proposal-start guided SMC | 4.73083e-6 | .610% | 6.05861e-7 | .639% |

8개method-target 모두RSE1.5% 이내. 최대 IID contribution share는 .1611%,
최대 whole-SMC share는1.7490%, 최대 LOO block shift는 .2700%였다.
6method-pairs×2targets 모두comparisons30 normal quantile 기준10% relative equivalence
통과(최대upper약5.881%). 앞선17.81% 실패보다 이번 observed tail sensitivity가 개선됐다.
그것이 population variance의 동일비율 개선이나 미관측 mode의 완전 배제를 증명하지는 않는다.

매우 중요한 범위:

- natural-start는 초기법칙이 p라는 뜻이다. mutation에는 envelope independence와
  Volterra direction을 사용하므로 **guide-free/local-only reference가 아니다**.
- original P1의 guide-independent/local-only corroboration을 이 방법으로 대체하지 않는다.
- high-eta parent0에서 통과해도 canonical 및 나머지8 q의 M2가 인증되는 것이 아니다.
- sample RSE·normal delta-method equivalence는 미관측 tail을 배제하는 엄밀한 상대 CI가 아니다.
- 진행 중인 공유 노트북 실험의 wall-time을 저널용 exclusive fixed-precision speed 비교로 쓰지 않는다.

## 7. P2: 시간 격자 문제

앞선 fresh N32–512 pilot에서 high-eta fine-minus-coarse / fine mean은 다음과 같았다.

| fine / coarse | relative difference | paired SE |
|---|---:|---:|
| 32 / 16 | -99.36% | 7.75% |
| 64 / 32 | -95.23% | 10.04% |
| 128 / 64 | -56.86% | 7.19% |
| 256 / 128 | -20.30% | 7.21% |
| 512 / 256 | -10.46% | 5.72% |

각 격자65536 draw. 전체potential1,310,720회. 마지막 차이도 불확실성을 포함하면
10% 이내라고 인증되지 않았다. canonical의 μ RSE는13–38%로 추가 불확실성이 컸다.

별도 price/driver 분해 pilot의 high-eta N512 결과는 price -14.53%(SE4.95%),
driver +.37%(SE.41%), total -14.16%(SE4.86%)였다. 앞선 pilot과 다른 새 표본이다.
순서 의존적 counterfactual 분해이지 두 독립 물리 원인의 유일한 분리가 아니다.

근거:

- `results/post_audit/p2_fine_mesh_recovery_v1/session_20261009/result.json` (20,480 paths replay)
- `results/post_audit/p2_driver_quadrature_diagnosis_v1/session_20261009/result.json` (16,384 paths replay)

corrected envelope / FFT scoring으로 N1024/2048/4096 새 진단을 완료했다.
각 셀·격자65536 draw,768 batch, potential786,432회, time-step evaluations
1,409,286,144회, phase wall647.59초. after-batch 관측 RSS 최대약1.63GiB
(모든 순간의 peak certificate 아님). source/input/seed/산술/3072paths replay 통과.

| 셀 | fine N | μ RSE | fine-coarse / fine | paired SE | abs차이+simultaneous z SE |
|---|---:|---:|---:|---:|---:|
| canonical | 1024 | 31.77% | +34.62% | 11.99% | 68.28% |
| canonical | 2048 | 26.51% | -44.80% | 42.20% | 163.25% |
| canonical | 4096 | 21.24% | +8.55% | 10.85% | 39.00% |
| high-eta | 1024 | 11.64% | -21.61% | 9.15% | 47.28% |
| high-eta | 2048 | 10.48% | -15.11% | 8.36% | 38.56% |
| high-eta | 4096 | 9.77% | +1.29% | 6.90% | 20.66% |

전부 개발용10% small-difference 기준 실패다. high-eta N4096의 작은 점추정을
mesh 성공으로 해석하지 않는다. canonical N4096에서는 한 draw가 fine contribution의
14.66%를 차지했다. proposal의 high-dimensional 효율도 별도 병목이다.
이전 N32 q/bank 재사용이나 같은 envelope의 dimension 확대만으로 해결되지 않는다.
근거: `results/post_audit/p2_envelope_fine_grid_v1/session_20261010/result.json`.

원래 adapted Itô left integration을 유지했다. 미래 volatility를 쓰는 naive
trapezoidal price increment로 격자 차이를 숨기지 않는다. 새 격자를 채택한다면
해당 grid의 q/bank/reference도 새로 검증해야 한다.

인접 차이가 작아져도 rough-volatility convergence rate와 tail sensitivity를 모르면
나머지 무한 mesh tail을 상계할 수 없다. finite-grid unbiasedness와 continuum
relative-bias certificate를 구분한다.

## 8. Guide-free kernel의 제한적인 재설계

기존 guide-free full-pCN만의 전역 갱신은 조건부 rare region에서 여러 좌표를
한꺼번에 바꾸므로 mixing 병목일 수 있다. 이 가설을 새 multiscale block-pCN으로
검증한다. state-independent random aligned block(크기2/8/32/64)만 갱신하는
prior-reversible kernel과 full-pCN을 번갈아 적용한다. fitted bank, Volterra 방향,
static event guide, independence proposal은 없다. frozen q는 risk target 정의에만 쓰인다.

Gaussian covariance 및 cross covariance symmetry, known Gaussian endpoint,
constant normalizer, 기존 weighted SMC와 full-arm 동작 일치의5tests 통과.
2cells×2targets×2arms×8whole의 fresh pilot 완료. potential32,571,392회,249.98초.
source/q/seed/summary/accounting 및8whole(추가4,071,424calls) replay 감사 통과.

| 셀 / target | full-pCN RSE | block-pCN RSE |
|---|---:|---:|
| canonical μ | 3.321% | 3.467% |
| canonical M2 | 6.573% | 3.672% |
| high-eta μ | 2.330% | 1.916% |
| high-eta M2 | 3.325% | 5.674% |

canonical risk에서는 개선, high-eta risk에서는 악화했다. universal improvement 아니다.
단지8whole의 variance 예측이므로 실제 precision qualification으로 승격하지 않는다.
**block이라는 이름이나 높은 acceptance
만으로 mode discovery·reference qualification을 선언하지 않는다.**
근거 계획: [guide-free pilot](../plans/GUIDE_FREE_MULTISCALE_BLOCK_RECOVERY_2026-10-10_KO.md).

### 8.1 원래10q 전체에서 재검증한 결과

parent0 pilot에서 cell/target별 observed CV²×wall이 작은 guide-free arm을 고정했다.
새 static IID/guide-free SMC/guided envelope SMC를 원래10q에 시행했다.
μ는 cell당 비교하며 static μ는5개 disjoint q-risk IID stream을 pool했다.
M2는 각자 q의 표본만 사용했다. 각q8whole/각IID262144draw.
전체816units, actual71,876,864potential calls,926.25초(15.44분).
source/anchor/q/seed/moment/summary/work와16384draw+4whole replay 감사 통과.

| 셀 | q | static M2 RSE | guide-free M2 RSE | guided M2 RSE | guide-free 필요whole forecast |
|---|---:|---:|---:|---:|---:|
| canonical | 0 | 2.414% | 4.883% | 1.578% | 170 |
| canonical | 1 | 2.073% | 6.036% | 2.199% | 260 |
| canonical | 2 | 2.258% | 6.776% | 1.903% | 327 |
| canonical | 3 | 2.067% | 9.589% | 1.973% | 654 |
| canonical | 4 | 2.414% | 7.928% | 2.905% | 447 |
| high-eta | 0 | 1.953% | 5.429% | 1.609% | 210 |
| high-eta | 1 | 1.958% | 2.321% | 1.017% | 39 |
| high-eta | 2 | 2.367% | 2.981% | 1.981% | 64 |
| high-eta | 3 | 2.008% | 1.950% | 1.550% | 32 |
| high-eta | 4 | 2.272% | 2.284% | 1.521% | 38 |

μ RSE: canonical static2.777% / guide-free3.027%, high-eta static.768% / guide-free2.129%.
RSE1.5%/safety2 allocation에서 canonical staticμ 필요8,986,624draw>8,388,608cap.
canonical q1..4 guide-free whole forecast는256cap 초과.
전체 forecast는**1,301,463,040potentials(약13.01억), wall3.03시간,
wall safety2 약6.06시간**으로 original160m/3600초 cap을 넘었다.
production을 강행하거나 failed q를 제외하지 않았다. 이는 forecast이지 실제 production 비용 아니다.
근거: `results/post_audit/full_q_reference_allocation_v1/session_20261010/result.json`.

### 8.2 계산 공유 후보: frozen-family umbrella

여러 q의 reference를 별도 SMC로 반복하면 같은 p,g 계산을 중복한다.
H0=g, Hj=δ_jg²/Dq_j로 정하고 고정positive w를 써H=Σw_jHj를 만든다.
SMC의 product normalizer Zhat와 terminal normalized weights Wi에서
Zhat_j=Zhat ΣWi Hj/H를 얻는다. M2에는 마지막δ_j 나눔을 적용한다.
Hj/H<=1/w_j 및 Σw_j Zhat_j=Zhat가 성립한다. final ratio만 쓰지 않으며
SMC unnormalized measure의 expectation identity를 사용한다.

known Gaussian endpoints/constant normalizers/ratio identity/기존scalar core 일치/
weight freezing 등의8core tests와2runner tests를 통과했다.
앞선 전체-q static means를 balancing anchor로 동결한 fresh48whole pilot를 완료했다.
actual20,365,440potential calls,280.72초. source/anchor/q/seed/endpoint identity/산술 및
6whole numerical replay(추가2,545,792calls) 감사 통과.
두 guide-free arm과 normalized-envelope guided arm을 구분한다. 여러 output의 SE는
particle가 아니라 independent whole 단위다.

| 셀 | method | μ RSE | 5risk RSE 범위 | 공유 필요whole forecast | 256cap 내 |
|---|---|---:|---:|---:|---|
| canonical | free-full | 2.408% | 7.740–9.966% | 707 | 아니오 |
| canonical | free-block | 3.780% | 8.230–10.453% | 777 | 아니오 |
| canonical | envelope-guided | 4.644% | 1.649–4.400% | 154 | 예 |
| high-eta | free-full | 3.106% | 1.987–2.547% | 69 | 예 |
| high-eta | free-block | 2.866% | 2.556–3.694% | 98 | 예 |
| high-eta | envelope-guided | 2.692% | 1.760–2.320% | 52 | 예 |

RSE1.5%/safety2 allocation의 최악6component 기준이다. 계산 공유는 비용 중복을 줄였지만
canonical mixing·whole variance를 해결하지는 못했다. 실제 uniform gain은 미입증이다.
high-eta는 staticIID40×32768 + free-full69whole + envelope52whole의 새 fixed validation을
선택했다. pilot와 평균내지 않는다. 실제 precision/민감도/상호 일치 후에만 해당 셀 scope를 갱신한다.
근거: `results/post_audit/umbrella_reference_pilot_v1/session_20261010/result.json`.
이것은 알려진 SMC 원리의 우리 family에 대한 적용이지 새 일반 정리의 발견 선언이 아니다.
상세: [umbrella 수학 검토](../theory/FROZEN_FAMILY_UMBRELLA_UNNORMALIZED_MEASURE_2026-10-10_KO.md).

### 8.2.1 높은 η의 5q 전체 fresh validation: 일치는 통과, 정밀도는 미달

fixed441units, actual49,685,632potential calls,766.39초(12.77분).
static1,310,720draw, free-full69whole, envelope52whole. pilot 표본과 pool하지 않았다.
source/anchor/q/seed/moment/endpoint identity/work/전체요약 및8192draw+2whole
numerical replay 감사 통과(추가772,224potential calls).

| target | static IID RSE | guide-free RSE | guided envelope RSE |
|---|---:|---:|---:|
| μ | .771% | .939% | .931% |
| M2(q0) | .865% | 1.613% | .911% |
| M2(q1) | .893% | 1.572% | .786% |
| M2(q2) | 1.019% | 1.508% | .870% |
| M2(q3) | .919% | 1.605% | .873% |
| M2(q4) | .958% | 1.472% | .940% |

primary μ pair와15risk pair의 simultaneous10% equivalence는 모두 통과했다.
최대 agreement upper는 약7.20%였다. IID maximum share .1074%,
whole maximum share2.2892%, maximum LOO shift .6179%로 민감도 기준도 통과했다.
그러나 q0/q1/q2/q3의 guide-free RSE가 사전1.5% 기준을 넘으므로
`high_eta_five_q_development_reference_pass=false`를 유지한다.
기준을 반올림하거나 69whole에 final 표본을 더 붙여 사후 통과시키지 않는다.
이는 경로 누락을 확정한 결과도 아니고 독립 정밀도를 모두 확보한 결과도 아니다.
다음 allocation에는 pilot8whole가 아니라 이 실패한69whole의 실제 변동을 반영하고,
별도 동결된 fresh namespace에서 전체 검증을 해야 한다. publication confirmation은
개발 중 재시도와 따로 봉인한다.
근거: `results/post_audit/high_eta_family_validation_v1/session_20261010/result.json`.

### 8.2.2 높은 η의 사전 재설계 fresh 검증: scoped 통과

실패한69whole의최악risk RSE로safety2를적용해free-full160whole를동결했다.
static40×32768/guided52whole는모두새stream으로재실행했고기존결과와합치지않았다.
fixed532units, actual95,998,080potential calls,1314.74초(21.91분).
source/anchor/q/seed/moment/identity/accounting 및8192draw+2whole replay 감사통과
(추가772,224potential calls). 개발재시도의선택과정이지publicationconfirmation이아니다.

| target | static IID RSE | guide-free RSE | guided envelope RSE |
|---|---:|---:|---:|
| μ | .805% | .623% | .997% |
| M2(q0) | .863% | 1.093% | .767% |
| M2(q1) | .887% | 1.067% | .831% |
| M2(q2) | 1.034% | .923% | .817% |
| M2(q3) | .910% | 1.038% | .811% |
| M2(q4) | .965% | .976% | .880% |

18method-target 모두RSE1.5%/민감도통과,primaryμpair+15riskpair도simultaneous10%일치.
이제high_eta_five_q_development_reference_pass=true다. 처음으로실제guide-free
corroboration을원래high-eta5q전체에서확보했다. finiteN32/development에만한정한다.
canonical/P2/large-training flags는false다. 확증용새셀·미관측tail·continuumbias를
이결과가인증하지않는다. originalq의eventperformance를개선한것도아니다.
근거: `results/post_audit/high_eta_family_revalidation_v1/session_20261010/result.json`.

### 8.3 과거 metadata 표기의 해석 오류

이전 parent0 recovery runner의 `guide_independence`는 단순히 method가 guide-IID가
아닌지를 표시했다. collective/excursion/envelope 및 그 mutation도 physical geometry를
쓰므로 이 field를 **guide-free 인증으로 읽으면 잘못**이다. 당시 scalar 계산/normalizer가
이로 인해 바뀌지는 않으며 authorization은 모두false였다. historical artifact의 field를
소급 덮어쓰지 않는다. 이번 새 full-q/umbrella schema는 `guide_independent_kernel`로
실제 kernel의 guidance 사용 여부를 명시하며 old ambiguous field를 관문에 사용하지 않는다.

## 9. 다음 행동의 우선순위

1. 위 고정 validation와 finer-grid 진단의 전체 완료·source/seed/moment/replay 감사.
2. 성공/실패 cell과 estimand를 분리한다. high-eta q0 성공만으로 P1 승인 금지.
3. guide-independent reference의 mixing/whole-run variance를 구조적으로 개선한다.
   단순히 natural-start라는 이름을 붙이거나 shared guide를 여러 estimator로 세지 않는다.
4. two-cell × five-q에서 새 allocation pilot; per-q risk를 직접 측정하고 μ는 cell별로
   검증한다. parent0 SMC 변동을 다른 q의 실제 정밀도로 대체하지 않는다.
5. allocation이 고정 cap 안에 들어오는 경우에만 새 namespace에서 production 전체를
   수행한다. 부족하면 cap 무시나 final 추가표본이 아니라 새 사전 설계로 돌아간다.
6. mesh의 관측 안정성과 남은 이론적 bias를 고려해 finite-grid 논문/continuum 논문
   주장 범위를 명시적으로 결정한다. N32의 q/bank를 미세 grid 인증 자산으로 쓰지 않는다.
7. reference + mesh의 해당 관문 뒤 bank quality, 최소 구조 q 보정, Stein CV,
   whole-training 반복, sealed confirmation으로 간다. 모델 크기 증가는 마지막 분기다.

SMC 초기 ancestry는 mutation이 있어도 genealogical label 기준으로 늘어나지 않는다.
낮은 ancestry만으로 bias를 판정하거나 mutation 후 label을 임의로 새롭게 붙여 개선했다고
보고하지 않는다. whole-run 변동, mode coverage, weighted tail contribution을 함께 본다.

### 9.1 이번에 착수한 다음 회복

canonical에서 μ와5risk 공동 target과 risk-only target을 새 표본으로 비교한다.
추가로 외부 guide 없는 gradient-MH를 구현했다. 방향을 따라 움직이더라도
full forward/reverse Gaussian density를 보정하므로 target은 유지한다.
proposal drift norm만 cap20으로 제한하고 g/target/weight는 clipping하지 않는다.
expanded MH와 직접density·상세균형, Gaussian stationary law, 다봉형 normalizer,
cached FFT/CDF/shift-density finite difference와 batch-row separability, constant/frozen
weights를 포함한10개 core/runner tests 통과 후4arms×8whole fresh pilot를 시작했다.
forward와 실제 autograd backward count를 분리한다. gradient가 항상 더 빠르거나
mode를 모두 찾는다는 주장은 없다. 기존 parent q를 개선한 것도 아니다.
계획: [target 분리 및 gradient 회복](../plans/CANONICAL_TARGET_SEPARATION_AND_GRADIENT_RECOVERY_2026-10-10_KO.md).

첫32whole 완료·감사: 기록된forward12,222,592 / actualbackward4,079,616,
phase163.76초. replay4whole 감사. 준비forward64회 과다계상은 아래정정서를따른다.

| arm | μ RSE | 5risk RSE 범위 | 공유 필요whole forecast |
|---|---:|---:|---:|
| μ+5risk full-pCN | 3.772% | 7.023–8.392% | 501 |
| risk-only full-pCN | 별도 | 7.044–9.071% | 586 |
| risk-only gradient-MH | 별도 | 4.618–6.464% | 298 |
| probability-only gradient-MH | 3.141% | 별도 | 71 |

μ를분리하기만하면risk가해결된다는가설은지지되지않았다.
gradient-MH는관측risk변동을줄였지만298>256로아직allocation실패다.
따라서particle2048 /mutation8 /temperature128 /larger scale.55의4개제한된
자원대조pilot도별도새stream에서완료했다. higheracceptance가독립정밀도증명은아니다.
실제q를개선한것도아니다.

| gradient design | 5risk RSE 범위 | 공유필요whole | wall forecast | 자원관문 |
|---|---:|---:|---:|---|
| 8moves | 4.024–5.193% | 192 | 2210초 | wall실패 |
| 2048particles | 6.880–7.345% | 384 | 3748초 | whole/work/wall실패 |
| 128temperature points | 2.240–3.730% | 99 | 1045초 | 후보 |
| larger scale.55 | 4.130–6.131% | 268 | 1401초 | whole실패 |

actual32whole, recorded14,327,936forward/14,327,808backward,297.53초.
4whole replay감사통과;forward의준비64과다계상은정정서를따른다.
128temperature의risk normalizerRSE는2.567%로낮아졌고fresh-validation forecast는
51,194,880forward+동일backward다. 아직실제referenceprecision이나speedup을인증한것은아니다.
staticμ/corroboration및μ-onlyreference비용을더한전체canonical설계가따로필요하다.
원자료: `results/post_audit/canonical_gradient_resource_v1/session_20261010/result.json`.

### 9.2 Canonical 준비 비용의 과다 계상 정정

one-cell prepare는64basis-path geometry만만드는데128을기록했다.
첫gradientpilot기록12,222,592→corrected12,222,528이다.
rawmanifest/checkpoint/source/result는덮어쓰지않는다. 이문제는추정값·seed·RSE·선택을
바꾸지않고budget을더보수적으로만들었지만`actual_work`표기오류이므로명시한다.
replayaudit의같은준비비용도64차감해1,527,872이다.
high-eta run은geometry64+static-guidegeometry64로128이정확하다.
근거: [gradient-MH 수학·비용 검토](../theory/GRADIENT_UMBRELLA_MH_AND_COST_REVIEW_2026-10-10_KO.md)
및해당run의`PREPARATION_COST_ERRATUM.md`.

### 9.3 Static reference의 실제 대조와 선택

known-component 층화는fullnormalizedmixture density를유지하는정확한ordinaryestimator다.
SE는stratum별독립variance를합쳐계산하고pooledIID variance로바꾸지않았다.
16blocks/476,224path-law calls/19.52초,59,520draw replay감사통과.

| static 방법 | μ RSE | μ allocation forecast |
|---|---:|---:|
| 새 ordinary IID | 5.321% | 6,011,520draw |
| fixed-proportional strata | 7.862% | 13,094,400draw(cap실패) |

비례층화의populationvariance는IIDmixture보다크지않지만개별pilot의observedsamplevariance는
오히려크게나올수있다. 이결과를원리의반례나층화가항상나쁘다는증거로해석하지않는다.
동일tail에대한finite-sample관측및실제cost개선이부족하다는negative개발결과다.
새IID의더낮은forecast만으로그전staticμ budget실패를해결했다고하지않았다.

다음pilot는기존exact last-pair cache로 **μ만**Rao–Blackwell적분했다.
gbar=Φ(a/sqrt(c²+b²)),prefixmarginal rU를정확히사용한다.
원래risk는fullg²/fullDq/fullDr를유지한다. gbar²를원래M2로표기하지않는다.
같은outer law에서Var(μ-RB)<=Var(μ-full)이지만엄밀한relativeCI나speedup보장은아니다.
8blocks/262,208path-law calls/10.95초,65,536draw replay및1,024full-vs-cache비교감사통과
(감사67,648calls).

| μ estimator | observed RSE | safety2 필요draw |
|---|---:|---:|
| full-g/full-r | 5.028% | 5,898,240 |
| exact last-pair /prefix-r | 4.688% | 5,144,576 |

paired relative 차이.737%,pairedSE1.728%,upper4.124%로10%개발일치.
samplingnoise가남아있으므로8block의variance를참값으로쓰지않는다.
새canonical검증은더보수적으로원래IIDcap전체8,388,608draw를고정했다.
refrisk99gradientwhole+μ71gradientwhole+guided154whole을각새stream에서검증한다.
total580units,116,953,216forward /69,298,176backward,3600초cap이다.
이는per-cell개발회복protocol이며oldertwo-cellproduction160mglobalbudget에소급끼워맞추지않는다.
근거:

- `results/post_audit/canonical_static_stratification_v1/session_20261010/result.json`
- `results/post_audit/canonical_last_pair_probability_v1/session_20261010/result.json`
- [fresh canonical 검증 계획](../plans/CANONICAL_FIVE_Q_FRESH_REFERENCE_VALIDATION_2026-10-10_KO.md)

완료 결과: phase 2492.83초, 17/18 precision checks 및 16/16 agreement checks 통과.
실패한 단 한 항목은 guide-free μ RSE 1.5066348% (>1.5%)다. static μ는 1.1978%,
guided μ는 1.0211%, guide-free five-risk는 1.2337–1.3570%였다.
source/anchor/q/seed/identity/moment/cost 및 65,536 IID draws +3 whole replay,
cache/full law 대조 감사를 통과했다. 감사 forward 1,094,784/backward 772,096회.
감사 통과는 통계 관문 통과와 다르며 scoped canonical flag는 false로 보존했다.

### 9.4 P2의 새 작은 진단

paired-grid는단순IID-envelope dimension확대대신joint Gaussian(zfine,ε)에
H=(gfine+gcoarse)/2를놓고gradient-MH/SMC를실행한다.
정확한coarsemarginal과기존Itôleft법칙을유지하고unnormalizedproductZhat로두μ를얻는다.
차이는두wholeoutput의covariance를포함한pairedSE로평가한다.
constantvolatilityfine/coarse동일성,jointaugmentation/FFTgradient의finite-difference및
singleton/batch일치,고정physical/jointcost검사를통과한뒤새32whole진단을시작했다.
N512/1024두셀의작은pilot이지P2confirmation/continuumbias인증이아니다.
계획: [paired-gradient mesh 진단](../plans/P2_PAIRED_GRADIENT_NORMALIZER_DIAGNOSTIC_2026-10-10_KO.md).

첫 32 whole-run은 완료했고 source/input/seed/joint identity/cost 및 각 셀-격자
rep0의 4 whole numerical replay 감사를 통과했다. 본 실험은 physical forward
2,039,808 / joint backward 1,019,904 / time-step forward 1,174,929,408회,
phase wall 476.48초다. 감사 비용은 physical 254,976 / backward 127,488회다.

| 셀 | fine N | fine μ RSE | (μf-μc)/μf | paired SE | 동시 상한 | 작은 차이 진단 |
|---|---:|---:|---:|---:|---:|---|
| canonical | 512 | 17.167% | -3.541% | 4.872% | 15.711% | 실패 |
| canonical | 1024 | 10.331% | -4.167% | 1.546% | 8.029% | 통과 |
| high-η | 512 | 12.684% | -18.821% | 4.593% | 30.294% | 실패 |
| high-η | 1024 | 13.770% | -10.655% | 3.978% | 20.591% | 실패 |

canonical N1024의 차이 진단이 개선됐지만 μ 자체의 정밀도, 8 whole의 tail coverage,
remaining continuum bias는 인증하지 못했다. high-η의 유의한 격자 변화도 남아 있다.
더 작은 μ RSE만으로 가격 정확도가 좋아졌다고 주장하지 않는다.
새 namespace에서 N2048/4096의 동일 32 whole 고정 pilot도 완료했다.
계획: [미세 격자 확장](../plans/P2_PAIRED_GRADIENT_FINE_EXTENSION_2026-10-10_KO.md).

| 셀 | fine N | fine μ RSE | relative difference | paired SE | 동시 상한 | 개발 차이 진단 |
|---|---:|---:|---:|---:|---:|---|
| canonical | 2048 | 12.382% | -3.162% | 2.541% | 9.508% | 통과 |
| canonical | 4096 | 6.805% | -2.758% | 1.221% | 5.809% | 통과 |
| high-η | 2048 | 6.557% | -1.237% | 3.277% | 9.423% | 통과 |
| high-η | 4096 | 15.382% | +0.913% | 2.685% | 7.619% | 통과 |

physical forward 2,039,808/joint backward 1,019,904/time-step forward 4,699,717,632,
phase2081.97초. 네 작은 차이 진단을 통과했지만 μ의 큰 RSE와 8 whole의 한계가
남아 있다. 이 결과만으로 미세 grid reference나 새 학습을 승인하지 않는다.
source/anchor/seed/joint identity/cost 및 고정 4 whole replay 감사도 통과했다.
감사 physical forward254,976/joint backward127,488회는 본 실험 비용과 별도다.

### 9.6 추가 이론 검토 및 작은 구현 후보

weak-error 문헌과 현재 coefficient를 직접 대조했다. 새 RMS historical weight
클래스는 monitoring driver의 분산을 정확히 맞추지만 원래 finite law와 다른 target다.
현재 H=.05의 terminal driver variance 부족은 N32 약 .057%, N4096 약 .035%로 작다.
이를 큰 mesh 차이의 원인이나 해결책으로 단정하지 않아 주 방법은 교체하지 않았다.
문헌 정리의 test function/scheme 가정이 우리 극단 digital target과 다르므로
H/2H/H+.5/min(3H+.5,1) 중 편리한 rate를 선택해 bias gate를 통과시키지 않는다.
근거: [weak-error와 moment matching 검토](../theory/P2_WEAK_ERROR_AND_MOMENT_MATCHING_REVIEW_2026-10-10_KO.md).

별도로 원래 BLP law의 모든 left driver에 가격 잔차를 조건부로 적분하는
`DriverConditionedTerminal`을 구현했다. N32 μ의 Gaussian outer dimension을 64→31로
줄이며, 원래 q의 M2를 바꾸지 않는다. Gaussian QR geometry와 직접 CDF 유도를
scalar Gauss–Hermite oracle로 확인했다. 이는 기존 conditional expectation 원리의
적용이며 새 일반 정리나 실제 성능 우위를 주장하지 않는다. 준비/score는 dense이고
N<=256로 제한했다. 2셀×2방법×8 whole의 새 μ-only 비교 pilot를 완료·감사했다.
근거: [exact all-driver conditioning](../theory/ALL_LEFT_DRIVER_EXACT_PRICE_CONDITIONING_2026-10-10_KO.md),
[고정 비교 계획](../plans/ALL_DRIVER_CONDITIONING_PROBABILITY_PILOT_2026-10-10_KO.md).

| 셀 | full64 μ RSE | conditioned31 μ RSE | 일치 상한(10% 기준) |
|---|---:|---:|---:|
| canonical | 3.727% | 3.774% | 12.841%: 실패 |
| high-η | 3.245% | 2.221% | 12.635%: 실패 |

32 whole/8,159,232 forward와 동일 backward/103.32초. 4 whole replay 감사 통과,
감사 forward/backward 각각1,019,904회. high-η의 관측 RSE는 줄었지만 canonical은
줄지 않았고 두 셀의 정밀도·일치 qualification은 실패다. natural-prior RB 분산 감소
정리를 서로 다른 MCMC/SMC whole variance의 단조 감소 정리로 혼동하지 않는다.

### 9.5 원래 계획과 승인 범위의 대조

기존통합V2의P1은staticordinaryIS /guide-freeSMC /guidedSMC의3risk대조와
별도2probability대조를요구했다. 새재설계는이역할을보존하되kernel과계산공유/μconditioning을
변경했다. 원래μ/M2/q와comparisonfamily30/2,margin10%,LOO5%를유지하고
actualprecision을옛2.5%max보다엄격한1.5%로검사한다.
반면새회복pilot의allocationsafety2와per-cellbudgets를옛safety3/통합productionmanifest와
동일하다고표기하지않는다. 특히high95,998,080+canonical계획116,953,216은
합계212,951,296으로옛global160m보다크다. 별도개발회복실험들이다.
historicalP1productionconfig의hash/schema를소급덮어쓰거나false승인하지않는다.
두셀검증성공시새method-recovery inventory와rawreferencebinding을먼저감사하고,
원래계획에서허용한fine-grid/scopediagnostic으로이어간다.
production model비교·large training·저널확증은별도승인관문을계속유지한다.

## 10. 검증 범위와 결론

paired-gradient 및 canonical validation runner까지 포함한 전체 테스트 1361개 통과
(288.23초). 당시 CI 범위 Mypy195 source files, CI Ruff 및 git diff 검사를 통과했다.
이후 추가한 finer runner 2개 테스트, moment-matched core 7개 테스트,
all-driver core 5개 oracle 테스트도 통과했다. H=.49의 대안 law 차이를 검사하는
테스트에서 기본 allclose tolerance가 너무 넓었던 문제는 절대 차이를 검사하도록
수정했다. 추정값을 바꾼 수정이 아니다.

2026-10-16 최종 재검토: 전체 pytest **1376개 통과(398.70초)**, CI Ruff 통과,
CI Mypy **197 source files 통과**, git diff 및 staged diff 공백 검사 통과.
canonical full validation, fine paired extension, driver-conditioned comparison의
완료 source/seed/whole numerical replay/cost 감사가 모두 통과했다. canonical의
통계적 실패와 두 conditioning 비교의 accuracy 실패는 그대로 보존한다.
로컬 검사는 Python3.10 환경이며 원격 Python3.10/3.11 CI 결과와 구분한다.

import를 무제한 따라간 실험 파일 Mypy 실행에서는 기존 비-CI 실험 dependency
4개 파일의 14개 오류가 노출됐다. 이번 신규 파일은 follow-imports=silent 개별 검사,
production source는 기존 CI 범위 전체 검사로 각각 통과했다. 저장소의 모든 과거
실험 파일이 정적으로 무오류라고 주장하지 않으며, 현재 실행 중인 source를
변경하거나 무관한 과거 코드를 이번 수정에 끼워 넣지 않았다.
소프트웨어 테스트·seed replay는 실제 tail 정밀도나 새 모델의 superiority 증명이 아니다.
상세 식과 가정은 [수학·수치 계약 검토](../theory/P1_RECOVERY_P2_NUMERICAL_CONTRACT_REVIEW_2026-10-10_KO.md).

최상위 저널을 겨냥하려면 아직 **독립 정확도, mesh/주장 범위, learned added value,
학습 포함 총비용, 새 수학적 기여와 외부 재현**이 남아 있다. 현재는 그 전제 조건을
회복하는 단계이며 연구 방향의 성공을 보장할 수 없다.
