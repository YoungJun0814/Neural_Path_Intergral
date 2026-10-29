# P2 남은 두 의무: Gaussian-split HMC reference와 tail-aware proof 실행계획

2026-10-28 America/New_York. 기존 실패·통과 기준·봉인 source를 보존한다.
커밋·푸시·유료 클라우드·논문 범위 변경은 하지 않는다. 성공할 때까지 seed를
바꾸거나 임의로 표본을 추가하지 않는다. 코드 완료와 연구 문제 해결은 별개다.

## 1. 두 질문

1. 학습 dictionary/bank를 전혀 쓰지 않는 original N4096 reference의 precision을
   RSE1.5% 이하로 확보하고, 자격을 얻은 경우에만 기존 IID와10% equivalence를 검사할 수 있는가?
2. 기존 absolute error의 전역 CDF derivative bound 대신 tail-weighted 모멘트를
   보존해 원래 연속시간 확률의 **유용한** relative remainder를 증명할 수 있는가?

두 번째 의무를 첫 번째의 반복 수로 해결했다고 하지 않는다. μ/mesh/all-q M2도 구분한다.

## 2. 새 reference의 수학·구현 계약

`πβ(z)∝φ(z)g_N(z)^β`, N4096, original mean-cell BLP, K1,
H.05,η1.5/2,ρ−.7,ξ.04,S100,T1를 유지한다. 좌표8192개다.
raw Gaussian momentum을 독립 생성한다. kick/정확 harmonic rotation/kick을4회
실행하고 momentum을 뒤집은 proposal에 full Hamiltonian MH를 적용한다.
힘의 norm20 cap은 proposal에만 적용하며 실제 logg/normalizer는 clipping하지 않는다.
cap을 포함해 각 momentum shear는 가역·volume preserving이므로 유한차원 MH가 유효하다.
Gaussian energy 변화는 각 kick의 `p·δ+.5||δ||²`를 누적해 큰 norm들의 차감 오차를 피한다.
이것이 infinite-dimensional rBergomi mixing 정리나 새로운 HMC 원리라는 주장은 하지 않는다.

우선 역변환·dense Hamiltonian ratio·Jacobian·constant/analytic normalizer·prior
rotation·동일 callback 작업량을 검사한다. 같은 callback 예산의 기존 gradient MH4회를
대조군으로 둔다. Hamiltonian4step은4개의 gradient callback이며4개 독립 sample이 아니다.

## 3. 사전 고정 pilot

- namespace `p2-hamiltonian-tail-bound-v1` 및 pilot/final/preflight/bootstrap seed 분리.
- 두 셀×두 arm(`gradient-control`,`gaussian-hmc`)×8독립 whole=32whole.
- 128particles,64points β=(k/63)^3, stratified every4, scale.35.
- HMC step.35×4, control gradient4회. 마지막 stage mutation없음.
- whole128×(1+62×4)=31,872 forward/gradient-request rows.
- 32whole 합계1,019,904회. 별도 preflight는16particles/16points,
  각912회×4=3,648회. pilot 총1,023,552 physical forward.
- 8CPUthreads,float64,RSS4GiB,whole300초,pilotphase7200초,physicalcap1,050,000.
- 전체32whole 완료 전 통과 판정 금지. 두 arm의 seed는 독립이고 CRN이라고 하지 않는다.
- RSE≤.015,whole 최대기여≤.15,LOO≤.05. ancestry/ESS/acceptance/이동거리 모두 보고.

## 4. 조건부 새 precision 단계

primary는 사전에 HMC다. 두 셀 각각 pilot count8와 관측 RSE에서
`ceil(3×8×(RSE/.015)^2/8)×8`,minimum32,maximum128whole를 새로 배정한다.
둘 다128이하이고 pilot 평균 시간으로 예상한 총 새 wall≤10800초일 때에만 실행한다.
그렇지 않으면 STOP을 기록한다. control의 더 좋은 숫자로 primary를 바꾸지 않는다.

새 whole seed namespace role는 confirmation과 분리한다. 이 단계 이름은 fresh
precision-development이며 저널용 sealed confirmation이라고 부르지 않는다.
pilot/preflight를 합치지 않고 새 전량으로 precision/stability를 계산한다.
정밀도 자격 획득 후에만 기존 audited frozen-ensemble IID와2comparison 조정10%
equivalence를 검사한다. precision 실패면 equivalence는 not evaluated다.
primary failed이면 무한 증량/새 seed 구제 없이 증거를 보존한다.

## 5. continuum proof를 실제로 진전시키는 범위

기존 tail-weighted chain rule에서 Hölder와 Gaussian CDF의 해석적 envelope를
결합해 `|μ^b−μ_N^b|≤B_N(r)`를 유도한다. 이전 absolute bound에 비해 tail 크기를
보존하는 부등식이지만 현재 상수가 작다는 보장은 없다. 기존 inverse moments와
original BLP driver MSE를 사용하며 새로운 증명되지 않은 rate를 대입하지 않는다.

r∈{1.05,1.1,1.25,1.5,2,3,4,8},N∈{4096,8192,16384}를 두 셀에서 전부 계산한다.
이것은 deterministic analytic bound parameter scan이고 성능 seed 선택이 아니다.
가장 작은 것도 공개하되 모든 결과를 남긴다. 관측 μ_N plug-in은 certified interval이
아니다. 상계가[0,1]이면 유용한 relative remainder는 미해결로 유지한다.
불필요한 GPU나 새 operator로 이 증명 의무를 대체하지 않는다.

## 6. 감사와 종료 조건

실행 전 source/config/plan/theory/기존reference result·audit를 봉인한다.
seed ledger·과거scope충돌·모든 unit/result binding·whole/stage/work를 전수검사한다.
각cell×arm pilot rep0 네 whole를 재생하고, 새 precision이 있으면 각cell 첫 whole도 재생한다.
통계와 이론 계산을 다시 계산한다. audit physical cap200,000,wall1800초.
전체회귀/CI범위Ruff·Mypy/새runner타입/기존frozenhash를검사하고 보고서와 receipt를 쓴다.

P2 true는 independent-reference precision/equivalence 및 유효한 continuum relative
remainder 등 필요한 실제 의무가 모두 충족됐을 때만 가능하다. 이번 구현은 P2 flag를
자동 승인하지 않는다. 검토·계산으로 증명을 얻지 못하면 그 사실을 최종 보고한다.
