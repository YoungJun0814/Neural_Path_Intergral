# Exact Gaussian excursion conditioning 후속 프로토콜

2026-10-09. covariance pilot의 결과를 숨기지 않고 새 normalized sampling mechanism을 검증한다. 이전 pilot 평균을 production에 합치지 않는다. 현재 모델 q는 동결한다.

## 정확한 법칙

고정 prior-orthogonal (u_t,v_t), E_t,a={u_t·z≥a}, F_t={v_t·z≥1}, a∈{4,5,6}에서

    r=0.1p+0.9/(6K) Σ_t,a [p(.|E_t,a)+p(.|E_t,a∩F_t)].
    r/p=0.1+0.9/(6K) Σ_t,a [1_E/Φ(−a)+1_(E∩F)/(Φ(−a)Φ(−1))].

v_t의 부호는 가격 하락 방향(−signρ)이고 u_t·v_t=0을 검사한다. component label별 density가 아니라 모든 overlap의 밀도를 합산한다. thresholds는 단일 사전 고정 family이며 기존 평균에 맞추어 조정하지 않는다. covariance 확장과 달리 잔여 Gaussian 법칙을 그대로 유지하고 projection의 정확한 truncated-normal inverse CDF를 사용한다. 단일 threshold 선택이나 event를 원래 payoff로 대체하지 않는다.

proposal-density discontinuity는 IS와 MH를 무효화하지 않는다. natural component로 density는 양수다. prior pCN MH correction과 normalized-r independence correction은 이전 bridge 공식을 그대로 사용한다. log conditional tail은 알려진 정규화 상수이며 학습/SMC로 추정하지 않는다.

## 사전 검증

독립 2D quadrature normalization, analytic projected sampling moments, 전체 overlap density, known toy endpoint의 corrected IS/SMC, original simulator identity를 검사한다. inverse-CDF uniform=0이면 실패로 기록하며 clipping이나 sample 교체를 하지 않는다. finite-grid conditioning이며 continuous-time Gaussian-max의 정확한 hitting law라는 주장은 하지 않는다.

## 고정 실제 pilot

두 셀 parent0. excursion IID 8×32,768 draw와 natural/proposal bridge SMC 각 target마다8whole-run. SMC particles1024·64βquadratic·mutation4·pCN.35·globalevery2·resampleevery4를 유지한다. guide는 이전 fresh pilot baseline으로만 참조하며 새 gate에서 같은 stream 재사용을 하지 않는다. μ/M₂의 IID paired roles, 독립 whole-run, source/q/seed를 유지한다.

wall1800초, potential cap40m, fixed λ나 threshold sweep·seed교체·표본삭제 없음. 실패 시 다음 구조 검토로 전환한다. production은 q 전체의 새 pilot와 allocation을 봉인한 뒤에만 가능하다. pilot 기반 forecast를 관측 precision으로 표현하지 않는다.

## 추가 선택이 필요한 경우

IID inverse transform을 scrambled Sobol로 바꾸는 경우 independent scramble가 inference unit이다. Gaussian residual basis를 변경하는 경우에도 exact orthogonal law를 검증해야 한다. 이들은 이번 고정 pilot에 조용히 추가하지 않는다. 필요하면 별도 source/config로 사전 고정한다.
