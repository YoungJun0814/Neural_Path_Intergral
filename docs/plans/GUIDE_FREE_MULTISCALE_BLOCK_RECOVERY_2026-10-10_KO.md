# Guide-free block-pCN recovery pilot

2026-10-10. 이번 pilot의 실제 target 관측 전에 고정한다.

## 질문

자연분포 시작만으로 guide-independent라고 할 수 없다. 기존 recovery의
natural-start/envelope-global mutation은 원래 P1 local-only 대조를 대신하지 못한다.
아래 두 arm은 Volterra direction, fitted bank, event template, independence proposal을
전혀 사용하지 않는다. frozen q는 risk potential 정의에만 사용한다.

## 수학 계약

πβ ∝ p H^β, p=N(0,I64), H=g 또는 δq*g²/Dq. 두 arm 모두 Gaussian prior로 시작.
full-pCN은 기존식. block-pCN은 state와 독립으로 크기2/8/32/64 중 하나와 aligned
블록을 균등 선택하여 그 좌표에만 Gaussian pCN을 적용한다. 각 block은 orthogonal
coordinate projection이고 prior-reversible하다. state-independent mixture 역시
prior-reversible하므로 MH ratio는 βΔlogH다. random block의 확률이나 density를
생략한 일반 state-dependent proposal이 아니다. logpotential은 모두 평가한다.

arm A: 8 full-pCN mutations/transition, scale.35.
arm B: 8 mutations 중 짝수 index full-pCN(.35), 홀수 index block-pCN(.7).
결정론적 합성 kernel은 각 subkernel이 invariant이면 invariant다. 합성 전체의
reversibility를 가정할 필요는 없다. block 선택은 model geometry나 현재 payoff와 무관하다.

## 고정 실행

- 원래 canonical/high-eta, N32, parent0 두 q만. 2arms ×2targets ×2cells ×8 whole=64.
- 1024particles,64quadratic temperature points,8mutations, stratified resample every4,
  final transition mutation 없음. count=1024*(1+62*8)=508,928/whole.
- potential32,571,392회, analytic preparation potential0회(시간은 포함).
- 신규 namespace `guide-free-block-v1`, source/config/input/q/whole seed 고정.
- cap40m potentials /1800초/RSS guard4GiB. whole 최대30초, prepare30초.
- Gaussian prior covariance/pair-symmetry, constant normalizer, known Gaussian endpoint,
  full-arm의 기존 weighted SMC 동작 일치 검증 후 실행한다.
- whole-run이 독립 inference unit이다. terminal particles를 IID로 세지 않는다.
- max contribution share, LOO sensitivity, whole RSE, terminal ESS와 초기 ancestry,
  full/block acceptance를 기록한다. ancestry label을 mutation 후 새로 바꾸지 않는다.
- 사전 allocation forecast: RSE1.5%, safety2, minimum32/maximum256 wholes.
  forecast는 아직 production precision을 측정한 것이 아니다.
- 모든 단위와 source/input/q/seed/summary/work를 감사하고 각 arm/cell/target의
  첫 whole-run 총8개를 numerical replay한다. audit 비용은 inference 비용과 분리한다.

## 판정과 다음 분기

8개 whole로 실제 target variance나 tail coverage를 인증하지 않는다. full 대비
block observed RSE/whole-cost가 더 좋으면 새 two-cell/five-q pilot 후 고정 production을
검토한다. 반대 셀/target이 악화되면 그 실패도 기록한다. 상대 efficiency의 점추정만으로
새 방법 우위를 선언하지 않는다. 성공 seed 선택, 실패 삭제, 고정 count 연장 금지.
전체 P1/P2/larger training flag는 계속false다. 같은 기존 schedule을 반복해서
사후 운 좋은 결과만 채택하는 것이 아니라, 다른 prior-reversible kernel을 제한적으로
검증하는 development이다. 최종 confirmation은 별도 미사용 표본/설계가 필요하다.
