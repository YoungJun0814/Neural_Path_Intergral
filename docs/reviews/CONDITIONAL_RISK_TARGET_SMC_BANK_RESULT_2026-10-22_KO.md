# Conditional risk-target SMC bank — 실행·이론·기술 검토

2026-10-22 America/New_York. 사전 고정한16개 island의 계산을 모두 완료했다.
결론은 **`STOP_GENEALOGY_OR_MUTATION_UNQUALIFIED`**다.
수치 계산 완료는 성공했지만 bank 자격은 얻지 못했고, 새 G fitting/사건q/P2/D3를 채택하지 않았다.
재생 감사·전체 회귀검증의 종료 결과는 아래 검증 절에 별도 기록한다. 커밋·푸시는 하지 않는다.

## 1. 이번 방향이 무엇을 바꾸는가

앞선 marginal-G 연구는 q에서 독립 표본을 모아 `m2/Dq²`로 가중해 G를 학습하려 했다.
새2,048개 bank의 weighted ESS가2.36–14.56에 불과해 fitting을 중단했다.
이번에는 사건 q의 모델/373성분/차원은 그대로 두고,
**원하는 평가 target까지 점차 이동하는 SMC**로 bank를 생성했다.

역할을 혼동하지 않는다.

- q는 사건확률을 추정하는 기존 제안분포다.
- 이번 SMC는 **고정 q의 M2를 평가할 bank**를 만드는 개발 도구다.
- 실제 normalized G를 이 bank로 fit한 것은 아니다.
- SMC particle weights를 ordinary IID importance weights처럼 취급하지 않는다.

“SMC를 붙였다”는 사실 자체가 새 모델의 독창성이나 성능 우위는 아니다.
기존 수학을 정확히 연결하고, 그 방법이 현재 병목을 풀 수 있는지 검증하는 단계다.

## 2. 수학적 재검토

U=61은 적분하고 미사용63을 제외한 X62를 탐색했다. residual57은 X에 포함된다.
현재 finite-grid q의 각성분은 이 두 제외좌표에 정확한0 mean/identity covariance를 가진다.

```
Dq = q_X/p_X,
m2(X) = E_phi(U)[g(X,U)^2],
F(X) = m2(X)/Dq(X)^2,
gamma_beta(X) = q_X(X)*F(X)^beta.
```

beta0의 normalizer는1, beta1의 unnormalized density는 `p_X*m2/Dq`이며
normalizer는 같은 고정 q의 M2(q)다.
`q*(m2/Dq)^beta`로 만들면 다른 endpoint이므로 사용하지 않았다.

mutation은 제안분포별로 다른 MH ratio를 사용했다.

| 제안 | reference/성분 처리 | MH log ratio |
|---|---|---|
| 일반 pCN | p_X-reversible | `Delta logDq + beta*Delta logF` |
| 성분중심 pCN | q의 exact component posterior로 label Gibbs 후 해당성분 Gaussian에 reversible | `beta*Delta logF` |
| q 독립 제안 | normalized q_X에서 draw | `beta*Delta logF` |

성분중심 방법은 augmented target
`w_l*phi(x-mu_l)*F(x)^beta`에 대한 Gibbs/MH이며 marginal이 원하는 gamma_beta다.
label을 payoff가 큰 성분으로 임의 선택하지 않았다.
일반 pCN에 q-reference 수락비를 그대로 쓰는 오류도 피했다.

increment는 고정 Δbeta와 logF로 계산했고, resampling하지 않는 stage 사이에도 weight를 보존했다.
마지막 beta1에도 mutation을 수행하되 final resampling은 하지 않았다.
accepted global move가 있어도 initial lineage는 새 ancestor로 reset하지 않았다.

이 유도는 **현재 이산 Gaussian path model**에 한정된다.
continuous Volterra law의 exactness, adapted drift/Girsanov 조건의 증명은 아니다.
normalizer의 표준 SMC 성질도 exact potential 가정하의 결과와 실제 numerical m2 구현을 구분해야 한다.
이번 구현을 true M2의 certified unbiased estimator라고 주장하지 않는다.

수학 계약과 원문:
[동결 수학 계약](../theory/CONDITIONAL_RISK_TARGET_SMC_CONTRACT_2026-10-22_KO.md),
[Del Moral, Doucet & Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x),
[Cotter et al. (2013)](https://doi.org/10.1214/13-STS421).
기존 SMC sampler 및 Gaussian-reference 보존 설계를 현재 target에 적용한 것이며,
문헌이 현재 코드나 Volterra 성능을 검증해 준다는 의미는 아니다.

## 3. 사전에 고정한 대조와 gate

[실행계획](../plans/CONDITIONAL_RISK_TARGET_SMC_BANK_2026-10-22_KO.md)의 두 arm을 같은 작업량으로 실행했다.

- primary component: 성분중심 pCN → 일반 pCN → q 독립 제안.
- control prior: 일반 pCN → 일반 pCN → q 독립 제안.
- 각셀/arm4개 island, 총16개; particles128; temperatures `(j/32)^4`.
- scale0.35; 매4stage stratified resampling, 마지막stage 제외.
- 모든 island의 외부 PRNG stream8개씩을 미리 봉인했다. arm별 seed가 달라 paired CRN 대조가 아니다.
- source·계획·수학·parent hash를 계산 전 archive에 보존했다.

bank gate는 nominal particle ESS≥64뿐 아니라 ancestor≥16, weighted ancestor ESS≥8,
최대 ancestor 질량≤25%, max particle≤5%, distinct particle≥96,
마지막16stage acceptance≥10% 및 median cumulative squared jump≥0.5를 함께 요구했다.
독립4-island normalizer RSE≤20%, largest island fraction≤50%, logspread≤log4도 별도 기준이다.

distinct particle가 많아졌거나 resampling 뒤 weight가 같아졌다는 이유로 독립성이 생겼다고 하지 않는다.
이 gate도 보수적인 개발 기준이지 충분조건/필요조건의 mixing 정리는 아니다.
특히 initial ancestry를 보존하는 기준은 긴 mixing으로 회복된 particle 다양성을 과소평가할 수도 있다.
관측 후 이 이유로 기준을 완화하지는 않는다.

## 4. 결과와 사전 판정

전체16개 island는 모두 complete 상태였다. technical/resource/numerical 실패는 없었다.
평가한 총198,656개 proposed context의 개별 수치 gate가 모두 통과했고,
관측된 최대 relative numerical proxy는 **1.688e-12**였다.
이 수치는 엄밀한 bias bound가 아니다.

### 독립 island 수준의 안정성

| 셀 / 방법 | 개별 bank gate 통과 /4 | Normalizer RSE | 최대 island 질량 | logZ spread | Normalizer 자격 |
|---|---:|---:|---:|---:|---|
| Canonical / 일반 pCN 대조 | 0 | 27.33% | 41.97% | 1.571 | 실패 |
| Canonical / 성분중심 주후보 | 1 | 26.43% | 38.60% | 1.187 | 실패 |
| High eta / 일반 pCN 대조 | 0 | 29.19% | 45.44% | 1.366 | 실패 |
| High eta / 성분중심 주후보 | 2 | 63.03% | 71.57% | 2.802 | 실패 |

모든 normalizer가 RSE≤20%를 만족하지 못했다.
high-eta 주후보는 최대 island 질량≤50%, spread≤log4도 실패했다.
따라서 양 방법이 같은 accuracy 자격을 얻은 셀이 없어, **arm 간 일치/성능 비교는 실행하지 않았다.**
`accuracy_qualified=false, no_detected_contradiction=null`로 남겼다.
이전처럼 큰 SE에서 비반증이 나왔다는 것을 agreement 성공으로 표시하지 않았다.

### 주후보의 개별 bank gate

| 셀 / island | 초기 조상 수 | Weighted ancestor ESS | 최대 ancestor 질량 | Nominal particle ESS | 개별 gate |
|---|---:|---:|---:|---:|---|
| Canonical /0 | 39 | 4.95 | 40.12% | 33.59 | 실패 |
| Canonical /1 | 34 | 8.93 | 24.94% | 67.85 | 통과 |
| Canonical /2 | 30 | 4.46 | 43.03% | 75.37 | 실패 |
| Canonical /3 | 32 | 9.82 | 19.74% | 61.74 | 실패 |
| High eta /0 | 44 | 15.89 | 13.94% | 66.16 | 통과 |
| High eta /1 | 35 | 7.60 | 21.31% | 68.47 | 실패 |
| High eta /2 | 50 | 13.93 | 20.48% | 68.95 | 통과 |
| High eta /3 | 32 | 1.39 | 84.46% | 30.19 | 실패 |

모든 주후보의 초기 조상 수는16 이상이고 distinct particle는128이었다.
late acceptance는46.16–49.30%, median cumulative squared jump는190.81–472.08로
이동 관련 gate는 모두 통과했다.
**이번 STOP 라벨에 mutation이라는 단어가 들어 있지만, 실제로 실패한 것은 주로 계보/weight 집중이다.**
“입자가 전혀 안 움직인다”가 이번 결과의 진단은 아니다.
그렇다고 큰 Euclidean 이동량이 target mode 간 충분한 mixing을 증명하는 것도 아니다.

Canonical island2는 nominal ESS75.37로 높아 보여도 weighted ancestor ESS는4.46이었다.
High-eta island3은32개 조상이 남아도 한 조상 계열에84.46%가 집중됐다.
따라서 조상 **개수**, 조상 **질량**, 현재 weight ESS를 분리한 gate가 중요했다.
통과한3개 island만 모아 G를 fit하는 사후 선택은 하지 않았다.

이전 q-IID bank의 ESS2.36–14.56과 이번 SMC nominal ESS를 직접 비교해 개선 배수라고 부르지 않는다.
계산 대상/의존성/작업량이 다르고 SMC particle는 correlated하기 때문이다.

### 모드 존재 검사는 통과했지만 충분하지 않았다

사전 K3 feature-clustering의 최소 존재 기준은 양 셀 모두 통과했다.
다만 high-eta 주후보에서 세 번째 cluster의 질량은 island0/1/2에서
**4.16%,4.33%,5.54%**, island3에서 **91.18%**였다.
최소1% 존재 여부는 만족해도 cluster 질량의 정량적 반복 안정성까지 만족하는 것은 아니다.
그 island가 normalizer 평균의71.57%를 차지했다는 관측과 함께 보면
mode/contribution 불균형 가능성을 점검할 근거는 되지만, 누락 모드의 존재나 인과관계의 증명은 아니다.
현재 clustering은 bank 자체의 기술적 진단이며 held-out 확인이 아니다.

### 사후 stage 분석 — 새 표본0, 판정 변경0

저장된 trace를 관측 후 기술적으로 분석했다. 이것은 별도 confirmation이 아니며 gate를 바꾸지 않는다.
고정 schedule에서 처음16stage는 beta≤0.0625에만 할당됐고,
마지막4stage는 beta0.58618→1의 큰 target 변화를 처리했다.
주후보 weighted ancestor ESS는 stage16에서 canonical61.13–75.16/high-eta61.13–75.85였지만,
최종에는 각각4.46–9.82/1.39–15.89였다.
초반부터 numerical target이 붕괴한 것이 아니라 후반 target 집중 과정에서 반복 간 차이가 커졌다.

후반 성분중심 pCN acceptance는 canonical72.22–73.00%, high-eta68.55–74.41%였고,
q 독립 제안 acceptance도 각각15.87–19.09%,20.41–21.09%였다.
단순히 acceptance를 더 높이는 것만으로 해결된다고 단정할 수 없다.
stage 배치/conditional target geometry/계보퇴화는 후속 **가설**이지 이번에 검증된 치료책이 아니다.

## 5. 수치 정확도와 구현 검토의 범위

모든 proposed context에 대해 기존 analytic batch 적분을 사용했다.
MH potential에는 개별 상대 numerical gate≤1e-6을 요구했으며,
이전 IID fitting에서의 aggregate-only tiny-contribution 예외를 적용하지 않았다.
root/envelope/Jensen/overflow 미해결은 성공으로 clip하지 않는다.
실수연산의 tail bound와 구현의 order-difference/error proxy는 다른 것이다.

normalizer, final target moment, mixture별 detailed balance, final mutation,
lineage를 nominal ESS로 대체하지 않는 반례, seed/예산/JSON,
setup/source/protocol/result 저장 실패의 receipt 보존 등 **42개 사전 테스트**를 통과했다.

새 코어는 [`conditional_risk_target_smc.py`](../../src/path_integral/conditional_risk_target_smc.py),
실행기는 [`post_audit_conditional_risk_target_smc.py`](../../experiments/post_audit_conditional_risk_target_smc.py)다.
rejected proposal 전체를 파일에 중복 저장하지 않고 input/output tensor digest와 incremental hash chain,
평가 횟수/최대 numerical proxy를 보존했다. 전체 proposal 검증은 지정된 full replay 범위와 구분한다.

검토 중 다음 범위 차이도 확인했다.

- 계획의 feature 표준화 표현은 `std floor1e-8`로 읽히지만 구현은
  `sqrt(variance.clamp_min(1e-8))`, 즉 std floor1e-4다.
  이는 동결된 계획/구현 사이의 표현·parameterization 불일치다. 소급 수정하지 않았다.
  실제 feature scale의 최솟값이 canonical0.81037/high-eta0.99664여서 **이번에는 두 floor 모두 비활성**이었다.
  이번 cluster/STOP 수치에 영향이 없지만, 다음 버전은 floor가 variance인지 std인지 명시해야 한다.
- 사전 Gaussian normalizer/target-moment known-answer는 q=p인 경우였다.
  nontrivial mixture에서는 pairwise detailed balance와 density ratio를 검사했다.
  nontrivial mixture normalizer까지 독립적인 known-answer end-to-end로 확인한 것은 아니므로
  그 검증을 완료했다고 부르지 않는다.
- stage trace와 final particle는 atomic island artifact로 보존했지만,
  한 island 도중 OS/process 강제 종료까지 복구하는 stage별 durable checkpoint는 없다.
  이번 실행은 그런 중단 없이 끝났다. power-loss 무손실 복구를 보장하지 않는다.

## 6. 비용과 재생 감사

생산 physical은 **198,656**(16×128×97), scalar node는 **749,664,000**이었다.
이는 같은 상태가 다시 제안되거나 reject돼도 실제 평가한 work를 포함한다.
scalar는 g/적분 node이며 root/density/FLOP 전체를 뜻하지 않는다. 해당 overhead는 wall에 포함됐다.
production physical200,000/scalar800,000,000 cap 안에서 완료했다.

compute+island I/O483.751초, setup4.282초, source freeze7.692초,
protocol I/O0.0112초, island I/O0.3934초, 집계0.0416초,
result I/O0.00986초, end-to-end receipt 직전495.917초였다.
독점 host benchmark나 fixed-precision 효율 비교가 아니며 다른 프로세스 존재를 배제하지 않았다.

재생 감사는 **통과**했다. 모든16개 island의 final128개 상태를 다시 평가하고,
사전에 지정한 canonical/high-eta 주후보 island0 두 개를 동일 seed로 처음부터 재생했다.
최종 log m2의 최대 차이는 **0**이었으며, 두 full replay의 stage trace/최종 상태/판정이 일치했다.
전체 artifact hash, seed ledger, 집계·계보·모드 진단과 동결 source도 확인했다.
나머지14개 island의 rejected proposal 전체를 재생한 것은 아니며,
그 범위에서는 저장 hash chain과 final-state 재평가를 구분한다.

감사 비용은 physical **26,880**, scalar **101,385,216**, wall **73.145초**였다.
생산+감사 합계는 physical **225,536**, scalar **851,049,216**이다.
감사는 별도의 physical30,000/scalar120,000,000/wall600초 한도에서 수행했으므로
생산 cap 초과로 숨기지 않는다. unit/full regression 비용은 연구 표본 예산과 별도다.

[생산 result](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/result.json),
[end-to-end 비용](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/end_to_end.json),
[재생 감사](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/audit.json).

## 7. 채택/보류와 다음 판단

**채택:** 정확한 conditional-risk bridge, 제안별 MH 보정, normalized q density,
고정 schedule/seed/평가 work와 lineage를 분리하는 진단 도구.

**채택하지 않음:** 현재32stage/128particle SMC bank, component arm의 우위 주장,
일부 성공 island만 이용한 G fitting, 현재 M2 점추정의 정답 인증, 실제G/q 모델 교체.
P2/D3/대규모 학습은 보류다.

다음 작업은 무조건 particle/iteration 수를 늘리는 것이 아니라 **판단 범위를 줄여야 한다**.

1. **현재 설정을 고정된 실패 결과로 남긴다.** seed나 gate를 바꾸며 성공할 때까지 같은 실험을 반복하지 않는다.
2. **nontrivial mixture의 end-to-end known-answer를 먼저 보완한다.**
   Gaussian-mixture target/bridge의 normalizer와 moment를 독립 quadrature로 확인하고,
   맞는 target에 도달했는지와 Monte Carlo 변동을 분리한다.
3. **실제 반복 불안정을 다룰 가설은 하나로 한정한다.**
   저장 trace상 후반 beta 변화가 큰 점에 근거해, 별도 사전 설계에서 late-temperature 배치와
   resampling·mutation의 matched-work 대조를 검토할 수 있다.
   단순 final resampling으로 ESS를N으로 만들고 통과했다고 하는 방식은 배제한다.
4. **bank 자격과 실제 평가자 성능을 다시 분리한다.**
   혹시 수정 bank가 자격을 얻어도 fresh G fitting/selection/final/reference의
   end-to-end 확인과 lineage/모드·정밀도 확인을 별도 실시해야 한다.
5. **연구 중단/축소 판단을 명시적으로 유지한다.**
   이제 IID bank와 이번 target-aware bank가 모두 현 budget에서 실패했다.
   후속 한정 실험도 안정성을 얻지 못하면 다른 operator를 계속 붙이기보다
   이 방향을 주논문 경로로 유지할지, 오차인증/한계분석 중심으로 축소할지 재평가해야 한다.

이 후속 가설은 이번에 추가 실행하지 않았다. 이번 수치가 “SMC는 안 된다”,
“모든 marginal G가 불가능하다” 또는 “다음 schedule 변경이면 해결된다”를 증명하지는 않는다.
현재는 top-journal model novelty/확정된 population 효율·실용성 입증 상태가 아니다.

## 8. 최종 검증과 알려진 한계

전체 회귀검증 **1,792개 통과**,479.57초. source statement coverage는85%
(22,771 statements /3,438 missed), 새 SMC core는96%(156/7)였다.
Ruff의 CI 범위 검사, source218개 파일의 Mypy와 새 runner1개 별도 Mypy도 통과했다.
이것은 Windows/Python3.10의 local 검증이며 새 remote Linux/Python3.11 CI 결과가 아니다.
알려진 Requests dependency warning은 로그에 남겼고 dependency를 변경하지 않았다.
statement coverage는 모든 branch/이론적 오류 부재의 증명이 아니다.

[회귀검증 receipt](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/validation/regression_review.json),
[전체 테스트 로그](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/validation/pytest-output.txt),
[정적 검사 로그](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/validation/static_checks-output.txt).

동결 source·이전 anchor·공개파일 검토도 통과했다.
6개 원본 anchor/source ZIP, archive member6,753개와 current 동결파일1,150개의 hash가 일치했다.
현재 runtime tree도 이번 실행과 동일했다. 당시 공개파일169개/strict JSON122개/
로컬 문서링크192개를 확인했고, credential pattern/위험 archive 경로/
100MiB 이상 파일/trailing whitespace 검출은0이었다.
pattern 검토가 secret 부재의 완전한 보증은 아니다.
JSON 및 validation stdout128개에서 Git filter 적용 blob이 producer 원본 byte와 같음을 확인했다.

[무결성 receipt](../../results/post_audit/conditional_risk_target_smc_v1/session_20261022/validation/integrity_review.json)에
정확한 범위와 해시를 기록했다. `git diff --check` 통과, staged path0,
HEAD `f556424360b8307284b23a6cfc0efcdee7380060` 유지, 새 commit/push0이다.
오류가 전혀 없다는 보증 대신 위에서 검증한 범위와 남은 제한을 구분한다.

생산 result SHA256: `ac83eb5aafca2ca804ea779a9a0165b11299c6f3c20b094221a19fdbe2f4999a`.
Source ZIP SHA256: `eeb88b7d365a427e100d9242a35ebf1b33eb57723df5bbce89bec88776dfb46f`.
snapshot에는 source/config/tests/선택문서·parent anchor1,133개 파일이 있다.
Audit SHA256: `d641979704fdc808c67bb5e394659c8cda1cd31c495cfb83eaea3f1915e10f32`.
