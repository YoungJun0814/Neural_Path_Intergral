# Frozen-family umbrella reference: fixed development pilot

2026-10-10. 다음 신규 표본 전에 규칙을 고정한다. 알려진 답과 수치 계약의8tests를
통과한 core를 실제 원래 two-cell/five-q family에 적용한다. 신규 성능 주장이 아니다.

## 고정 target와 anchor

각 셀6개 component: H0=g, Hj=δ_j*g²/Dq_j,j=0..4.
새 full-q allocation pilot의 static IID μ/M2 mean을 positive balancing anchors로 쓴다.
anchor file=`results/post_audit/full_q_reference_allocation_v1/session_20261010/result.json`.
파일이 전체 완료된 후 SHA를 manifest/config/source에 봉인한다. μ의 anchor는 μ mean,
risk anchor는 δ_j*M2 mean. weights는 inverse-anchor를 logsumexp로 정상화한다.
pilot anchor의 정확도를 참값으로 가정하지 않는다. 작은/큰 오차는 효율을 바꿀 뿐
고정 H와 component endpoint의 expectation identity를 바꾸지 않는다.

새 `umbrella-reference-v1` namespace. 이전 final/pilot mean에 새로운 mean을 합치지 않는다.
q의 원래 .1 defensive floor와 exact density를 유지한다. fine-grid는 N32 그대로다.

## Methods / count / budget

- 원래2cells,5q family, method당8 independent whole-runs.
- free-full: natural start,8full-pCN(.35) mutations.
- free-block: natural start,full(.35)/multiscale-block(.7) 번갈아8mutations.
- envelope: normalized corrected-f64 envelope start,4mutations(full/global/rank2/global),
  proposal bridge,independence every2. 이것은 guide-free가 아니다.
- 전부1024particles,64quadraticpoints, every4stratified resampling,final mutation없음.
- 48wholes, component normalizers6개씩, 48×6 correlated outputs.
- calls=32×508928+16×254976+128analytic geometry prep=20,365,440.
  geometry128은 envelope2cells×64 simulator-basis probes. q/evaluator analytic prep0.
- cap25m potentials/1800초/RSSguard4GiB. whole30초, preparation30초.
- source/anchor/input/q/weights/seed/checkpoint completion 및
  terminal weighted-sum identity를 저장한다. 2cells×3methods rep0,6wholes replay.
- 고정 count 전체 완료 전 prefix로 성공·중단 판단 금지.

## 추정과 통계

Zhat product normalizer를 terminal weight ratio에 **반드시 곱한다**:
Zhat_j=Zhat ΣWi Hj/H. component δ_j correction 후 원래 μ/M2를 반환한다.
normalizer를 버린 self-normalized final IS estimator가 아니다. 각 output은 같은
whole의 correlated vector다. SE는 independent whole로 계산한다. particles를 IID로 세지 않는다.

각 method/cell의6개 output RSE/LOO/maxshare 및 allocationRSE1.5%,safety2를 기록한다.
공유-run production count forecast는6개 component 중 최대 required whole count다.
한 component의 통과로 다른 q를 대체하지 않는다. 최소32/max256whole.
full/block 선택은 이 파일에서 확정하지 않는다. 점추정 worst-component total work
forecast를 개발 선택 근거로만 사용하고 새 confirmation과 분리한다.

앞선 separate-target pilot와 비교되는 비용 절감은 workload 공유 후보이지
알고리즘 speedup/novelty 증거가 아니다. familyΣ w_j Zhat_j=Zhat identity와 bounded
Hj/H<=1/w_j는 correctness/finite-moment 근거이지 relative precision certificate가 아니다.
original P1의 static IID / guide-independent / guided corroboration과 family-wise30risk/2μ
comparison을 이후 **새 고정 production**에서 모두 유지한다. 모든 pilot authorization=false.
P2에서 드러난 mesh 문제를 해결하지도 않는다.
