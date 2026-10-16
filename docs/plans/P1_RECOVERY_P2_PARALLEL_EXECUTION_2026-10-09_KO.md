# P1 회복과 P2 병행 실행 프로토콜

2026-10-09. 목표는 정확도 관문을 낮추지 않고 실제 reference 병목을 해결하며, 독립적으로 가능한 P2 coupling 구현을 진행하는 것이다. 기존 실패/원본 q/격자/target은 변경하지 않는다. 비용·시간·실패를 숨기지 않는다.

## 단계 A: 수학 및 구현

1. local block 대신 model-law Volterra monitor direction u_t와 그 다음 가격 innovation v_t를 함께 covariance 확대한다. u_t·v_t=0은 검사한다. 평균 0, 두 eigenvalue=25, defensive mass=.1, 모든 stochastic left times를 균등 혼합한다. q/bank/성능 결과로 방향·λ를 선택하지 않는다.
2. normalized proposal r에서 시작하는 SMC bridge πβ∝r^(1−β)(pH)^β를 구현한다. 초기 Z0=1, incremental potential=(H p/r)^Δβ. pCN MH acceptance는 βΔlogH+(1−β)Δlog(r/p), r-independence MH는 βΔ[logH−log(r/p)]다. natural 시작 πβ∝pH^β 대조도 유지한다. target H는 μ=g 또는 scaled risk δq g²/Dq이며 마지막 risk normalizer만 δq로 나눈다. self-normalization은 particle weights에만 사용하고 final normalizer를 ratio estimator로 바꾸지 않는다.
3. exact finite-grid adjacent adapter를 구현한다. fine 첫 pair의 기존 두 white 좌표를 유지하고 C1 residual용 새 독립 Gaussian을 보강한다. coarse L=C1+L2, Brownian increment는 합산 후 coarse whitening한다. augmented joint likelihood는 fine p/q 하나다.
4. Gaussian covariance·logpdf·known normalization·SMC toy oracle·coupling matrix AAᵀ+BBᵀ=I·shift law·원래 fine simulator identity를 테스트한다. fixed schedules, 독립 whole-run inference를 유지한다.

## 단계 B: 고정 pilot

원래 두 셀, parent0 q에서만 fresh pilot을 실행한다. 새방법: collective covariance IID, natural-start/global-covariance SMC, proposal-start bridge SMC. 기존 guide IID를 baseline으로 유지한다.

- IID: 방법당 8×32,768 draw; μ와 risk paired target임을 기록한다.
- SMC: target·method·cell당 8개 독립 whole-run. particles=1024, 64 quadratic temperature points, 4 mutations/transition, pCN=.35, 매 2번째 mutation global, stratified resampling every4, 마지막 transition mutation 없음. 호출=1024×(1+62×4)=254,976/whole-run.
- 새로운 namespace, immutable source/입력/config, complete-unit checkpoint, 전체 phase wall 1800초, work cap 40,000,000 potential evaluations. CPU float64 thread1. 현 laptop만 사용한다. cloud 비용·새 package 설치 없음.
- toy oracle는 phase A/B 별도 2D Gaussian bump의 known endpoint와 비교한다. 실제 tail certificate로 승격하지 않는다.
- RSE·maximum share·whole-run sensitivity·equivalence를 함께 보고한다. 방법의 평균을 다른 방법에 맞추지 않는다.

## 단계 C: allocation 및 다음 단계

pilot와 분리된 production을 실행하려면 사전 target RSE1.5%, safety factor2, reference agreement margin10%, 30 risk comparisons의 family-wise alpha.05를 유지한다. whole-run 최대256, IID 최대4,194,304/q, 전체 work 최대160,000,000, wall 최대3600초의 실행 가능성부터 계산한다. 이를 넘으면 생산 대형 run을 강행하지 않고 구조적 병목을 분석한다. pilot는 production 평균에 합치지 않는다.

P2의 작은 coupling oracle와 scoped mesh **pilot**는 reference가 미해결이어도 진행할 수 있다. 단독 prototype은 P2 전체 pass가 아니다. 실제 signed mesh correction의 오차와 독립 per-grid reference가 준비되기 전 continuum 안정성·학습 확대를 주장하지 않는다. training/confirmation의 gate는 변경하지 않는다.

방법이 실패하면 새 config·새 설명으로만 다음 설계를 진행하며, seed 교체·표본 삭제·threshold 완화·반복해서 유리한 결과만 채택하는 방식은 사용하지 않는다. 목표의 성공을 사전에 보장할 수 없다. 이번 run의 결과에 따라 다음 실험을 설계하는 개발과 봉인 confirmation을 구분한다.

이론 배경: [Del Moral–Doucet–Jasra, SMC Samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf). 새 bridge는 알려진 SMC 원리의 적용이며 새로운 일반 정리로 주장하지 않는다.
