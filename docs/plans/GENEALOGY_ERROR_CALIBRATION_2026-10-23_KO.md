# 계보 진단과 실제 오차의 관계 — 사전 한정 검증계획

2026-10-23 America/New_York. 직전 late-bridge STOP과 모든 source/result를 보존한다.
이번 목표는 새 모델 채택이 아니라 **진단의 타당성**을 검토하는 것이다.
commit/push, 기존 gate 완화, 현재 Volterra bank fitting, P2/D3/대규모 학습은 하지 않는다.

## 1. 수학적으로 다른 두 질문

기존 bank의 final normalized target moment가 정확한가와 SMC normalizer가 정확한가는 다르다.
initial genealogy도 둘의 이론적 필요충분조건은 아니다.

F가 고정된 경우의 SMC sampler 틀과 normalizer/normalized empirical measure를 구분한다.
[Del Moral, Doucet & Jasra (2006), 저자 원문](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).
아래 exact-refresh 반례는 직접 유도하며 문헌이 현재 구현을 인증한다고 하지 않는다.

동결된 과거 정보 H가 final weights w_i, initial lineage와 Zhat를 정한다고 하자.
그 뒤 각 입자를 알려진 정확한 target π에서 **독립적으로** 새로 뽑는 kernel을 적용하면

`E[Σ w_i f(X_i') | H]=π(f)`,
`Var[Σ w_i f(X_i') | H]=Var_π(f) Σ w_i²`.

이때 weights/initial lineage/Zhat는 모두 그대로다.
따라서 낮은 initial ancestor ESS가 final moment 부정확성의 필요조건은 아니며,
반대로 final moment를 정확하게 만들어도 이미 누적한 normalizer 오류는 고쳐지지 않는다.
이 exact kernel은 target sampler를 아는 synthetic oracle에서만 가능하고 실제 Volterra 해결책이 아니다.
refresh 후 weights/lineage를 reset하지 않는다.

## 2. 정확한 rare/multimodal target 구성

p=q=N(0,1). 정규화된 target h는 같은 sd τ<1의 두 Gaussian mixture다.

`h(x)=Σ ρ_l φ_τ(x-μ_l)`,
`C_l=τ^(-1) exp[μ_l²/{2(1-τ²)}]`,
`Z=1/(Σ ρ_l C_l)`, `m2(x)=Z h(x)/p(x)`.

Gaussian ratio의 최대값은 C_l이므로0<m2≤1이고 gamma1=p*m2=Z*h다.
SMC의 numerical integration 없이 Z, mean, second moment, P(X>0)를 closed form으로 안다.
`g(x,u)=sqrt(m2(x))`로 두면 이 m2가 실제 bounded conditional second moment인 예를 만들 수 있다.
그러나 실제 금융 payoff의 geometry를 재현하는 것은 아니다.
정규화/오차 기준을 통과시키려고 target amplitude를 관측 후 바꾸지 않는다.

사전 세 문제:

| ID | means | target weights | sd |
|---|---|---|---:|
| moderate-symmetric | -1.5,+1.5 | 0.5,0.5 | 0.7 |
| rare-symmetric | -4,+4 | 0.5,0.5 | 0.5 |
| rare-asymmetric | -5,+6 | 0.7,0.3 | 0.5 |

실제62D의 대리모형이나 tail certificate가 아닌 **1D stress test**다.
양수 구간 질량은 latent component weight와 구분해 Gaussian CDF로 정확히 계산한다.
별도 scalar log-density와 adaptive quadrature로 Z를 scale한 raw0/1/2/right moment를 검사한다.
독립 적분 대조 오차≤1e-9를 요구한다. tiny Z에 absolute-only 기준을 쓰지 않는다.

## 3. 결정론적 반례 — 판정 기준을 바꾸는 근거가 아님

각 target의 component별64개 normal midpoint quantile와 weights ρ_l/64를 결합한다.
모든 lineage를0으로 표시한다. 이는 **constructed cubature**이지 IID 표본이 아니다.
target moment 근사와 genealogy가 다른 속성임을 보이는 데만 쓴다.

반대 bank는 왼쪽 component의128개 midpoint quantile, uniform weights,
lineage0..127로 구성한다. genealogy/weight 기준은 통과해도 오른쪽 target 질량을 놓친다.
movement trace를 조작해 full gate를 통과한 실제 SMC bank인 것처럼 만들지 않는다.

## 4. 실제 SMC + 독립 IID-IS + exact-refresh 대조

각 문제32개 독립 replicate, 총96개. 모두 새 stream을 사전 봉인한다.
기존 SMC core를 수정하지 않고 particles128,32stage선형,component arm,
scale0.35,매4stage stratified resampling/final resampling 없음을 그대로 사용한다.
q=p이므로 이 문제에서 component-centred kernel은 prior-centred와 같다.
한 replicate SMC work=12,416 potential contexts, 총1,191,936.

같은 replicate마다 **독립** natural-p IID12,416개로 Z를 ordinary mean m2로 추정한다.
추가1,191,936contexts, 총2,383,872. SMC와 동일 context 수이지만 FLOP/wall 동일성을 주장하지 않는다.
IID-IS의 normalized target moment는 SNIS 진단으로만 쓰며 finite-N biased다.
Z/M2 자체를 self-normalize하지 않고 event estimator로 쓰지 않는다.

각 SMC final bank 뒤에는 별도 label/noise seed의 exact-target128개로 refresh한 진단을 계산한다.
원래 weights/lineage/logZ를 보존한다. target draw12,288개는 potential-context와 별도로 과금한다.
실제 Volterra physical0이다.

## 5. 사전 오차 label과 비교

각 bank의 target-accuracy label은 다음을 모두 요구한다.

- |mean error|≤0.25.
- |second moment error|≤max(0.5,0.1*true second moment).
- |P(X>0) error|≤0.10.

normalizer-accuracy label은 |Zhat/Z-1|≤0.20이며 위 label과 별도로 저장한다.
이는 stress test용 해석 가능한 threshold이지 실제 모델의 accuracy 보증 기준이 아니다.
기존 full gate/ancestor-only/weight-only 판정과 위 target label/normalizer label의 confusion count를 기록한다.
false accept/reject 명칭은 이 사전 tolerance 기준에 한정하며 denominator0은 null로 표시한다.
threshold 최적화, ROC로 새 production gate 채택, 같은자료로 training/confirmation을 겸하지 않는다.

문제당32개 Z의 mean/RSE/정답 대비 오차와,
replicate를 사전 연속4개씩 나눈8개 group의 기존 normalizer gate를 확인한다.
**RSE/반복 gate 통과인데 실제 relative Z error>20%인 group**을 별도로 표시한다.
독립 label이라도 같은 모드를 놓치면 작은 RSE가 정답을 보장하지 않는다.

ancestor ESS와 target-error score 및 Z relative error의 Spearman 상관은 기술 통계로만 기록한다.
32개 stress-test 반복/3문제에서의 상관을 실제62D의 예측 성능이나 유의성으로 확대하지 않는다.

## 6. 실행·재현·중단

potential contexts cap2,400,000/target draws12,288/compute wall300초.
reference scalar quadrature count/setup/source/I/O/end-to-end 비용은 별도로 기록한다.
reference 검증 실패 시 random 실행0으로 STOP. 그 외 사전96개만 실행한다.
runtime 오류는 failure receipt로 보존하고 실패 job/seed를 재시도하지 않는다.
매 replicate를 atomic artifact로 저장하고 input/output hash chain과 최종 결과를 바인딩한다.
계산 도중 OS-kill에서의 stage별 durable recovery는 보장하지 않는다.

실행 전 source/config/tests/이 계획/parent result+audit를 archive에 봉인한다.
감사에서 독립 reference/반례와96개 replicate **전부 full replay**한다.
감사도 별도 동일 context/draw/wall cap을 쓴다. 모든 결과/집계/seed/hash를 대조한다.
수치 내용에서 wall은 재생 동일성을 요구하지 않는다.

검증이 완료돼도 판정은 `CALIBRATION_COMPLETE_NO_RETROACTIVE_ADOPTION`이다.
현재 Volterra STOP은 유지한다. 새로 알게 된 진단 한계는 향후 별도 설계에만 사용한다.
보고서에는 실제 Volterra의 다른 메커니즘 reference 선택과 예상 비용/한계를 구체화하되,
이번 synthetic 결과를 실제 reference qualification으로 대신하지 않는다.
전체 회귀검증/Ruff/Mypy/anchor·archive/hash/strict JSON/공개파일 검토를 마친다.

## 7. Parent anchor

late-bridge result:
`24b2b05b03e1e706f30c81d5f51c469f818930ab7ff835cc067a0e9719aaa721`.
audit:
`574f9cfaff2524201b0eb3cc18e9ece23153ccbb1ba78f3a0a616d38e9d26026`.
새 ledger는 직전 late-bridge/oracle 및 그 source인 SMC/IID ledger와 scoped collision을 검사한다.
이는 전체 과거 PRNG stream의 독립성 증명이 아니다.
