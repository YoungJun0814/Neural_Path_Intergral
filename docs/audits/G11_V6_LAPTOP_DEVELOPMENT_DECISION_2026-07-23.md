# G11 V6 laptop development decision

작성일: 2026-07-23

판정: **두 개의 H=0.12, 10^-3 셀에서 Route A의 강한 개발 신호를 얻었지만,
qualification·confirmation 또는 저널 headline evidence는 아직 아니다.** Route B는
terminal inverse-slope theorem까지만 무조건부이고 coefficient/correction/complexity
theorem은 아직 열려 있다.

## 1. 현재 정책을 쉽게 설명하면

현재 V6은 신경망 하나가 확률을 바로 출력하는 모델이 아니다. rough Bergomi 경로의
희귀 하락 확률을 계산할 때 다음 절차를 하나의 사전 선언된 정책으로 묶는다.

1. 자연분포 pilot으로 사건이 얼마나 희귀한지 확인한다.
2. reference 값을 보지 않고 crude, DCS-SLIS 또는 Hybrid profiling 경로를 정한다.
3. task별 CEM control에서 만든 defensive proposal bank로 희귀 경로를 더 자주 만든다.
4. 사건을 직접 0/1로 세는 대신, 사건을 결정하는 한 Gaussian 좌표를 조건부로
   적분한다.
5. 여러 시간 해상도의 correction을 profile하고 예상 총 work가 작은 시작점을 고른다.
6. pilot과 완전히 다른 final seed로 고정된 정수 표본 수만 실행한다.
7. 학습·screening·profiling·final·checkpoint 비용을 모두 포함해 pure CEM과 비교한다.

수학적 핵심은 exact balance-mixture likelihood, Girsanov measure change, Gaussian
conditional integration, Rao--Blackwellization과 fine/coarse telescoping이다. 양자 진폭
또는 복소 확률을 사용하지 않는다.

## 2. 이번 실행 범위와 claim 경계

최종 24-cluster pilot의 공통 조건은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Hurst `H` | 0.12 |
| maturity | 0.25 |
| finest steps | 128 |
| tasks | terminal left tail, discrete lower barrier |
| nominal probability | 10^-3 |
| requested relative sampling RMSE | 20% |
| independent clusters | 24 |
| primary comparator | task-tuned pure CEM |

따라서 이 결과는 H, maturity, rarity와 두 task에 대한 **국소 개발 결과**이다. 세 H,
10^-2/10^-3/10^-4, 전체 accepted-cell matrix의 결과로 일반화하면 안 된다. 모든
artifact는 dirty development source, unfrozen config/manifest를 명시하므로 formal claim은
schema 수준에서 false이다.

## 3. Falsification-first 진행 기록

좋지 않은 결과도 다음 설계를 결정한 증거로 보존했다.

### 3.1 Reference V1에서 발견한 문제

단일 4,096-sample pilot의 분산에 1.5배만 곱한 allocation은 heavy-tail 변동을 충분히
보지 못했고, 5 million cap에서 DCS reference SE가 10^-5를 소폭 초과했다. V2는
8개의 독립 16,384-sample planning replicate, planning/final 분리, fixed final `N`,
method-level atomic checkpoint와 strict resume identity를 사용한다.

V2 two-cell reference는 모든 gate를 통과했다.

| cell | DCS estimate ± SE | DCS N | raw estimate ± SE | raw N | agreement z |
|---|---:|---:|---:|---:|---:|
| terminal | 0.0010224493 ± 7.10e-6 | 11,587,392 | 0.0009866804 ± 1.89e-5 | 3,157,407 | 1.768 |
| barrier | 0.0010225908 ± 6.73e-6 | 14,617,025 | 0.0010457991 ± 1.87e-5 | 4,024,476 | 1.170 |

두 method의 likelihood normalization도 통과했다. 이는 fixed-grid reference의 개발
증거이며 continuous-time truth는 아니다.

### 3.2 Hoeffding work certificate의 한계

Defensive likelihood의 bounded Hoeffding interval 자체는 수학적으로 유효하지만,
희귀 IS variance allocation에서는 upper bound가 실제 분산보다 수천 배 이상 커져
수억 sample을 요구했다. 따라서 이 interval은 안전/audit diagnostic으로 남기고,
실행 allocation과 work-regret claim에는 사용하지 않는다. V2 replicated planner는
명시적으로 `no finite-sample work-regret certificate`라고 출력한다.

### 3.3 Router final-sample floor 누락

zero-hit pilot에서 candidate point work가 mandatory minimum final sample을 청구하지
않아, profile cap이 실행 가능한 최소 work보다 작게 계산될 수 있었다. Direct와
Hybrid point-work 모두 final-count floor를 포함하도록 수정했고, sequential state가
look 사이에 floor를 바꾸지 못하도록 했다.

### 3.4 Median replicate variance 실패

기존 fixed proposal에서 `single_4` replicate variance가 약 10^-10부터 10^-3 이상까지
변했다. median은 드문 큰 분산을 제거해 final variance를 최대 수십 배 과소평가했다.
mean of independent sample variances로 바꾸고, 실제 achieved-RMSE gate를 별도로
유지했다. mean도 유한 표본 보장은 아니며 qualification에서 검증되어야 한다.

### 3.5 Proposal-bank mismatch

고정 bank `[-0.5,-2],[-1,-4]`는 task-tuned CEM control과 크게 달랐다. 세 CEM
development cluster의 componentwise median으로 task별 full control을 만들고,
`zero/half/full` defensive bank를 구성했다. V3 schema는 다음을 원본 artifact에서
다시 계산한다.

- raw SHA-256;
- pure-CEM-only 완료 레코드와 두 task coverage;
- componentwise median과 zero/half/full bank;
- training sample/work/wall/CPU 총합; 그리고
- 사전 선언된 amortization record count.

원본 학습 artifact SHA-256은
`7aa1fe54be0ccd9ae835ecec6e8c0d2cdbf802da83723607616d12a4a58dfbb5`이다.

### 3.6 Training amortization 과대계상

V6은 6회 CEM bank training 총비용을 16개 레코드에 정확히 배분하지 않고, source
record 한 개의 평균 training 비용을 각 레코드에 반복 청구했다. 정책에 유리한 오류는
아니며 실제보다 약 2.67배 과대계상했지만 exact-total-work 계약에는 맞지 않았다.
V3는 총 196,608 training samples와 176,160,768 work units를 실행 matrix에 정확히
한 번만 배분한다.

### 3.7 V7 정확도 실패와 V8 설계

정확 회계 V7, 8-cluster 실행에서는 barrier cluster 하나가 명목 relative SE
20.12%로 20% target을 넘었다. 해당 final term variance는 planning variance의 약
1.86배였고 4,096 minimum floor가 allocation을 지배했다. 결과를 보고 같은 protocol을
재튜닝하지 않았다.

새 protocol V8은 minimum final samples를 8,192로 고정하고 24개 새 seed cluster를
실행했다. 이 변경은 development 결과로 설계한 것이므로 V8 자체도 untouched
confirmation이 아니다.

### 3.8 Power-source 연결 오류

Confirmatory analyzer가 전달된 power artifact가 바로 그 baseline/policy pair에서
계산되었는지 확인하지 않았다. 이제 baseline과 policy의 canonical SHA-256을 power
artifact 내부 hash와 다시 비교하며 mismatch는 scientific gate를 fail한다.

## 4. 최종 24-cluster 개발 결과

### 4.1 실행 및 정확도

- baseline: 48/48 complete, censoring 0, design target 48/48;
- policy: 48/48 complete, censoring 0, design target 48/48, empirical plug-in target
  48/48;
- baseline plug-in target: terminal 24/24, barrier 22/24; 초과값은 20.03%와
  20.11%; 그리고
- 독립 JSON audit: baseline 48/48, policy 48/48 통과.

사전 선언된 cell-level co-gate 결과는 다음과 같다.

| method/cell | attainment | one-sided exact lower | empirical RMSE | bootstrap upper | tolerance | pass |
|---|---:|---:|---:|---:|---:|---:|
| CEM terminal | 24/24 | 0.883 | 7.70e-5 | 1.01e-4 | 2.50e-4 | yes |
| CEM barrier | 22/24 | 0.760 | 1.44e-4 | 1.85e-4 | 2.50e-4 | yes |
| V6 terminal | 24/24 | 0.883 | 1.26e-4 | 1.53e-4 | 2.50e-4 | yes |
| V6 barrier | 24/24 | 0.883 | 1.47e-4 | 1.84e-4 | 2.50e-4 | yes |

`minimum_target_attainment_rate=0.60`, one-sided exact level과 bootstrap seed/repetition은
분석 전에 config에 고정됐다. Bootstrap gate는 근사적 engineering qualification
criterion이지 distribution-free theorem이 아니다.

### 4.2 Training-inclusive efficiency

Equal-cell cluster log ratio는

`log(total work_pure-CEM / total work_V6-policy)`

로 정의했다.

- geometric mean ratio: **3.1308**;
- one-sided 95% lower ratio: **2.8095**;
- mean log ratio: 1.1413;
- cluster log-ratio SD: 0.3095; 그리고
- one-sided p-value against no saving: 2.20e-15.

이 수치는 두 development cell과 현재 operation-work metric에 한정된다. Wall-clock
headline, full matrix 또는 published neural/HJB competitor 우위를 뜻하지 않는다.

### 4.3 Power와 실행 중단점

효과를 `log(1.2)`로 cap하고 observed effect를 50% shrink하며 SD를 1.5배 inflate한
normal approximation은 **51 clusters**를 요구했다. 64-cluster 계획은 development
resource gate를 통과했고 보수적 projected work는 약 20.32 billion units,
projected wall time은 약 5.64시간이다.

24-cluster confirmatory-style 분석은 정확도, audit, protocol identity, power-source
binding과 one-sided efficiency lower gate를 모두 통과했지만 `24 < 51`이므로 최종
scientific gate는 실패했다. 이 fail은 의도된 동작이다.

## 5. 정확성에 대한 이론·기술 재검토

### 5.1 Finite-grid unbiasedness

Proposal training, screening, routing, profile selection과 allocation pilot이 만드는
sigma-field를 `G`라 하자. Proposal, route, selected candidate와 integer allocation은
`G`-measurable이고 final streams는 `G`와 독립이다. 각 final direct 또는 telescoping
estimator는 exact likelihood를 사용해 조건부 기댓값이 동일한 fixed-grid `p_L`이다.
따라서 tower property로 무조건부 기댓값도 `p_L`이다. Pilot은 final mean에 재사용되지
않는다.

### 5.2 Task-conditioned proposal의 타당성

Task label과 frozen control bank는 final random path를 보기 전에 정해진다. Task별
bank를 사용하는 것은 target leakage가 아니다. Balance-mixture likelihood를 모든
expert에 대해 정확히 계산하고 zero-shift mass 0.15를 유지하므로 defensive bound도
유효하다.

### 5.3 Replicated planning의 claim 한계

Independent planning replicate의 sample variance 평균은 각 proposal term variance의
불편추정량이지만, heavy-tail finite-sample upper confidence bound는 아니다. Factor 6과
minimum 8,192는 development engineering design이며 theorem이 아니다. 실제 accuracy
co-gate가 필요한 이유가 여기에 있다.

### 5.4 Durable resume

Baseline과 routed runner는 completed-record journal뿐 아니라 record 내부 final
sampling도 policy-hash-bound chunk checkpoint로 실행한다. Resume은 proposal, route,
candidate, allocation 또는 seed namespace를 바꿀 수 없다. 완료 레코드 journal을
원자적으로 쓴 뒤 record checkpoint를 제거한다.

## 6. Route B 상태

Positive piecewise direction에 대한 terminal slope

`B_n=sqrt(1-rho^2)sqrt(dt) sum_i u_i sqrt(V_i)`

의 모든 negative moment는 target rBergomi law에서 mesh-uniform하게 유한하다는 bound를
증명했다. Defensive proposal에서는 `L<=1/delta`로 likelihood-weighted corollary도
성립한다.

Full diagnostic은 432 records, 6개 H/task rate cell을 실행했고 pathwise exactness,
direction geometry, analytic inverse bounds와 empirical inverse moments gate를 통과했다.
그러나 stable rate window는 5/6만 식별됐고 H=0.30 terminal은 식별되지 않았다.
또한 DCS second-moment exponent에서 threshold-L2 exponent를 뺀 값이 모든 식별 셀에서
음수였다. 이는 현 범위에서 threshold coupling이 병목일 가능성을 지지하지만 proof는
아니다.

아직 필요한 증명은 다음과 같다.

- exact BLP coupling의 terminal intercept/slope `L^p` rate;
- numerator-envelope moment와 proposal-parameter uniformity;
- localization threshold 최적화와 terminal correction second-moment theorem;
- continuous target을 원할 때 weak-bias/cost exponent와 MLMC complexity; 그리고
- barrier active-time 및 fine-only mesh-enrichment theorem.

따라서 “unconditional rBergomi complexity theorem” 또는 barrier theorem은 아직
주장할 수 없다.

## 7. Artifact ledger

아래는 local development raw-file SHA-256이다. `tmp/` artifact는 formal publication
archive가 아니며 commit하지 않는다.

| artifact | raw SHA-256 |
|---|---|
| V2 reference | `4a5efeecc53eaf8d0fa30172d1e62e782a995c0763c8d9f8ec04179f6a5d2ff3` |
| 24-cluster CEM | `cfc07569e922ace96a06382f6f5e871e1f34747f08e483b7c25d7e15f8e1783a` |
| 24-cluster V8 policy | `939e0e0f8ca5b92b8a0654687357be22ea6c7533f3a9e66dbca10c3d16363298` |
| CEM independent audit | `30f8e2a1bce38ed477f5b49b081308b92aac68b33b19f2399b5db1a698172c2a` |
| policy independent audit | `56f1341df81ee310f8bfebb7f7b2e512c848f12decff6ac176eaf04c6a1fa455` |
| 64-cluster power plan | `a9fb7dc2d27b6643bf285ac6490f65be1061bbfd339805702fa882ff5a69490c` |
| 24-cluster analysis | `87bc6986527ff9cfb1d6be4144c6c977849347a7362144c752fbbb1722782969` |
| full Route B diagnostic | `73e8fcefecd7d8823f56e5e36434952a3c570ffa773c4d0a929dd9abdf6dffc3` |

Repository verification at this decision point:

- `python -m ruff check src experiments tests`: pass;
- `python -m pytest -q`: **481 passed**; 그리고
- `git diff --check`: pass.

## 8. 저널 수준까지 남은 필수 작업

### P0 — formal 실행 전

1. clean commit에서 full 18-cell calibration 결과를 재생성한다.
2. full accepted matrix에서 independent reference를 완성한다.
3. proposal-bank training/calibration cells와 qualification/confirmation cells 또는
   seed blocks를 명시적으로 분리한다.
4. V3 source artifact와 proposal bank를 publication archive에 포함한다.
5. crude, defensive CEM, fixed DCS-SLIS와 router/selector ablation을 mandatory secondary
   table로 실행한다.

### P1 — Route A qualification과 confirmation

1. heterogeneous matrix에서 먼저 독립 cluster variance와 power를 다시 추정한다.
   현재 51은 두 셀 전용 수치라 full matrix에 그대로 복사하지 않는다.
2. full-matrix power와 resource가 통과한 뒤 config/manifest/reference/audit hashes를
   outcome-blind freeze한다.
3. untouched confirmation을 실행하고 complete-case deletion 없이 모두 분석한다.
4. 별도 clean Linux/CPU 환경에서 같은 frozen protocol을 재현한다.
5. operation work 외에 frozen-thread wall time, CPU time, peak memory를 headline
   supplement에 보고한다.

### P2 — Route B

Terminal coefficient theorem을 먼저 완성한다. 이것이 실패하면 Route B는 general
localized-threshold theorem과 negative result로 범위를 낮추고, Route A의 finite-grid
computational paper와 분리한다. Barrier는 terminal theorem의 자동 corollary로 취급하지
않는다.

## 9. 객관적 논문 수준 판정

현재 코드·감사·실험설계는 박사과정 연구 인프라 수준이고, 두 셀 결과는 강한
proof-of-concept이다. 하지만 top journal 논문 완성본은 아니다.

- **지금 제출:** 이르다. 외적 타당성, untouched evidence, secondary baselines와
  coefficient theorem이 부족하다.
- **Route A full matrix + frozen confirmation + Linux reproduction 통과:** 강한
  computational finance/numerical methods 논문 후보가 된다.
- **Route B terminal rate/complexity까지 완성:** SIFIN/SISC급 이론·계산 결합 후보가
  된다.
- **Mathematical Finance급:** 위 결과에 더해 금융적 insight, 넓은 parameter
  robustness와 model-level theorem의 깊이가 필요하다.

현재 가장 정직한 결론은 “3.13배 개발 신호를 발견했고 정확성·감사 pipeline은
작동하지만, full-matrix powered confirmation과 핵심 coefficient theorem 전에는
우월성 또는 일반 복잡도 claim을 하지 않는다”이다.
