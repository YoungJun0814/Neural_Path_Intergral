# 혼합분포 oracle → 단일 late-bridge 가설: 사전 실행계획

2026-10-23 America/New_York. 기존 2026-10-22 SMC STOP과 모든 원본을 보존한다.
commit/push, 사건 q 변경, 실제 G fitting, P2/D3/대규모 학습은 이 계획에 없다.
관측 후 seed·gate·schedule·예산을 바꾸어 성공할 때까지 반복하지 않는다.

## 1. 두 질문을 분리한다

A. 기존 SMC 구현이 **q가 실제로 혼합분포인 경우에도** 알려진 normalizer와 target moment를 재현하는가?

B. A가 통과하면, 기존 quartic temperature 배치 대신 선형 배치만 사용했을 때
같은 work에서 후반 target 집중/계보 퇴화가 완화되어 bank 개발 자격을 얻는가?

A의 성공은 B의 성공, Volterra 정확도, 효율 우위나 novelty의 증명이 아니다.
B가 실패하면 이번 한정 가설을 종료하고 주논문 경로의 유지/축소를 다시 판단한다.

## 2. 이론 재검토

기존 [계약](../theory/CONDITIONAL_RISK_TARGET_SMC_CONTRACT_2026-10-22_KO.md)을 그대로 사용한다.

`Dq=q/p`, `F=m2/Dq²`, `gamma_beta=q F^beta`, `Z1=∫p² m2/q`.

새 1D oracle은 `p=N(0,1)`, `q=0.2 N(0,1)+0.8 N(2,1)`,
`m2(x)=exp[-0.4(x-1)²-0.3]`이다. 이는 양수이고1 이하이며 q≠p다.
q≥0.2p이므로 gamma1≤5p. 따라서 |x|>T의 raw0/1/2 moment tail은 각각
`10 Phi(-T)`, `10 phi(T)`, `10[T phi(T)+Phi(-T)]`로 상계된다.
실수 산술의 이 bound와 코드의 비외향 반올림 수치는 구분한다.

SMC와 별도인 scalar math/SciPy 적분으로
`gamma_beta=q^(1-2 beta) p^(2 beta) m2^beta`를 계산한다.
beta=0,0.25,0.5,0.75,1의 Z, E[X], E[X²], P(X>0)를 구한다.
무한구간 adaptive quadrature와 [-12,0]/[0,12]의 독립 Gauss–Legendre256/512를 대조한다.
raw moment 차이≤1e-10, beta0 analytic Z/mean/second moment와 차이≤1e-10을 요구한다.
beta1의 생략 tail bound도 기록한다. 이는 arbitrary q/다차원 bias의 인증 정리가 아니다.

잘못된 `q(m2/Dq)^beta`의 endpoint는 p m2다. 그 normalizer와 Gaussian moment는
closed form이며 올바른 endpoint와의 차이도 명시해, target을 바꿔 맞춘 oracle을 배제한다.
기존 prior/component/global mutation의 MH 보정은 변경하지 않는다.

문헌 경계: [Del Moral, Doucet & Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x)의
SMC sampler 틀을 적용한다. 선형 schedule 자체는 새 수학적 원리가 아니며 최적 schedule이라는 주장도 없다.
adaptively chosen temperatures/optional stopping은 이번에 사용하지 않는다.

## 3. A: nontrivial-mixture known-answer 실험

- 두 arm prior/component × 두 schedule quartic/linear, 총4조건.
- schedule은 각각 `(j/32)^4`, `j/32`; particles512; scale0.35;
  매4stage stratified resampling, final resampling 없음; 기존3mutation/final mutation 유지.
- 조건당16개 독립 island, 총64개. 정확히3,178,496 Gaussian potential contexts.
- 새 namespace와 stream8개/island를 사전 ledger로 봉인한다.
  기존 marginal-G264/SMC128 stream과 대조한다. 전체 과거 PRNG 독립성 증명은 아니다.
- 독립 island Z의 ordinary mean을 사용한다. 각 조건의 RSE≤2.5%, 상대 Z오차≤4%,
  `|Zbar-Zref|≤4 SE+1e-4 Zref`를 모두 요구한다.
- final normalized moment의 island 평균에 대해 |mean오차|≤0.06,
  |second moment오차|≤0.10, |right-mass오차|≤0.04를 요구한다.
- 위 oracle 조건은 유한-N 코드 진단이고 normalized particle mean의 불편성 증명이 아니다.
  여러 gate의 familywise 통계적 유의수준이나 publication confirmation으로 해석하지 않는다.
- 잘못된 endpoint normalizer의 올바른 값 대비 상대 차이≥10%를 요구한다.
- Gaussian contexts cap3,200,000 / compute wall300초. 실제 Volterra physics는0이다.
- deterministic reference가 실패하면 random island를 실행하지 않고 STOP.
  그 외에는 사전64개를 완료하고 실패 조건만 골라 재시도하지 않는다.
- 전 island의 결과 digest/seed/weighted moment와 evaluation hash chain을 저장한다.
  감사는64개 전부 동일 seed로 재생하며 별도 contexts3,200,000/wall300초를 사용한다.
- source/config/이 계획/parent result+audit를 실행 전 archive로 봉인한다.
  A 결과와 감사 모두 통과해야 B를 실행한다.

## 4. B: schedule 하나만 바꾸는 matched-work 대조

이전 두 셀 canonical/high-eta, 고정373성분 사건 q와 X62를 그대로 쓴다.
두 schedule × 셀당4개 독립 island, 총16개를 **새 표본**으로 실행한다.

primary linear `j/32`, control quartic `(j/32)^4`.
모두 component arm: component-centred pCN → prior pCN → q independence.
particles128/scale0.35/32stage/매4stage resampling/final resampling 없음 등 나머지는 동일하다.
이는 resampling index와 kernel을 고정한 **temperature allocation 하나의 대조**다.
최적 resampling·mutation을 함께 찾는 탐색이 아니다.
arm별 seed가 달라 paired CRN이 아니며 이전16개 island를 control로 재사용하지 않는다.

실행 순서: 각 셀에서 replicate0..3, 그 안에서 quartic→linear.
물리 평가 총198,656, scalar cap800,000,000, physical cap200,000, compute wall2400초.
setup/source/protocol/island/result I/O/end-to-end 및 감사 비용을 따로 기록한다.
원래 numerical recorder를 그대로 사용하고 모든 proposed context에 개별 상대 proxy≤1e-6을 요구한다.
tiny contribution 예외/clip은 없다. global acceptance가 initial lineage를 reset하지 않는다.

### Bank/반복 자격

기존 gate를 그대로 사용한다: 각 primary island에서 initial ancestors≥16,
weighted ancestor ESS≥8, max ancestor mass≤25%, nominal ESS≥64,
max particle weight≤5%, distinct≥96, late acceptance≥10%, median cumulative squared jump≥0.5.

각 cell/schedule의4개 독립 Z에 RSE≤20%, max island fraction≤50%, logspread≤log4.
primary K3 feature 존재 gate도 기존 함수 그대로 사용한다.
**표준화 floor는 variance1e-8, 즉 std1e-4**임을 이번에는 명시한다.
cluster 최소 존재가 quantitative cluster-mass 안정성/모드 완전성이라는 주장은 하지 않는다.

primary8개 bank가 모두 통과하고 primary normalizer/mode가 자격을 얻어야 후보가 된다.
control normalizer까지 자격이 있을 때만 기존 3.291 combined-SE 차이 경고를 계산한다.
이는 equivalence나 ground-truth accuracy 증명이 아니다.
두 방법이 accuracy 자격을 얻지 못하면 speedup·fixed-precision total-wall 비교는 없다.
선형이 통과해도 **별도 fresh G pipeline을 설계할 후보**일 뿐 이번에 fit하지 않는다.

### 판정과 종료

자원/기술/수치 실패를 구분하고 기존 실패 island를 재시도하지 않는다.
primary bank gate 실패→`STOP_LATE_BRIDGE_BANK_UNQUALIFIED`.
primary normalizer/mode 실패→`STOP_LATE_BRIDGE_ISLAND_OR_MODE_UNRESOLVED`.
control 자격/차이 경고 실패→`STOP_LATE_BRIDGE_REFERENCE_UNRESOLVED`.
전부 통과→`LATE_BRIDGE_BANK_CANDIDATE_ONLY`.
모든 경우 actual G fits0, event q unchanged, P2/D3/large training false다.
후속 schedule/particle/scale 탐색으로 남은 budget을 전용하지 않는다.

## 5. B 감사·검증·알려진 제한

모든 artifact/hash/seed/stage/lineage/statistic/판정을 재집계하고 모든16 final bank를 재평가한다.
양 셀 **linear island0** 두 개를 full replay한다. replay physical cap30,000,
scalar120,000,000/wall600초. 예상26,880physical이며 final logm2 차이≤1e-8,
logDq≤1e-10을 요구한다. 나머지14개 rejected proposal 전체를 재생했다고 하지 않는다.
stage별 durable OS-kill 복구는 구현하지 않으며 atomic island 보존과 구분한다.

결과 후에도 frozen numeric source/plan/원본 결과는 소급 수정하지 않는다.
새 report/README/검증 receipt만 추가한다. 전체 pytest/Ruff/Mypy, source/anchor/hash/
strict JSON/문서 링크/공개파일 검토를 수행한다.

관측되는 RSE는4개 반복 기반이라 underpowered하고, 같은 누락 모드를 공유하는 경우도 배제하지 못한다.
quadrature proxy, Euclidean 이동, nominal ESS, ancestry gate 중 어느 것도 단독으로 mixing/정답을 보증하지 않는다.
오류가 없다는 절대 보증 대신 검증 범위와 남은 연구 한계를 명시한다.

## 6. Parent anchor

2026-10-22 SMC result SHA256:
`ac83eb5aafca2ca804ea779a9a0165b11299c6f3c20b094221a19fdbe2f4999a`.
해당 audit SHA256:
`d641979704fdc808c67bb5e394659c8cda1cd31c495cfb83eaea3f1915e10f32`.
