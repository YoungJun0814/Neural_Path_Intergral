# Reference 회복과 원래 모델의 성능은 다르다

2026-10-16. 현재 진행 중인 실험의 prefix를 사용하지 않는다. 아래는 이미 완료한
N32 고정 five-q 자료의 계산상 함의다. 새 모델의 우위나 제출 준비 인증이 아니다.

## 1. 원래 q의 실제 병목

ordinary IID estimator Y=g(z)/Dq(z), z~q에서
Var(mean Y)/μ²=[M2(q)/μ²-1]/n이다. 따라서 CV²=M2/μ²-1가 크면 표본 수를 늘려도
실용적 정밀도에 도달하기 어렵다. reference의 M2 RSE와 이 **event estimator의 RSE**는
서로 다른 수치다. reference를 개선했다고 q가 개선된 것이 아니다.

완료 자료의 static μ와 독립 guide-free M2 점추정치를 사용한 plug-in forecast:

| N32 cell | 원래 five-q의 CV² 범위 | RSE5% 필요 IID draws | RSE1% 필요 IID draws |
|---|---:|---:|---:|
| canonical | 4.13–5.91 ×10^5 | 1.65–2.37 ×10^8 | 4.13–5.91 ×10^9 |
| high-η | 2.12–2.75 ×10^4 | 8.47–10.99 ×10^6 | 2.12–2.75 ×10^8 |

데이터: `canonical_family_validation_v1/session_20261010/result.json` 및
`high_eta_family_revalidation_v1/session_20261010/result.json`의 guide-iid μ,
free-full risk-q0..4. `ceil((exp(logM2-2logmu)-1)/target_rse²)`로 계산했다.
μ와 M2 추정의 불확실성, 미관측 tail, training/selection 비용을 포함한 보증이 아니다.
canonical의 전체 검증은 μ 한 항목의 precision 실패로 false인 상태를 보존한다.
이 값으로 fixed-precision wall-time 우월성이나 continuum 모델 성능을 선언하지 않는다.

## 2. 결과가 의미하는 것

원래 q는 여전히 극단 기여를 충분히 밀도 높게 덮지 못할 가능성이 크다. 이 자료는
특정 누락 mode의 완전한 식별이나 모든 training seed의 성능을 증명하지 않는다.
bank coverage·분산 목적함수·proposal 구조를 검토해야 하며, 더 큰 네트워크를 먼저
학습하는 것으로 이 수치가 저절로 개선된다고 볼 수 없다.

canonical static μ는 last-pair Rao–Blackwell estimator다. 그 관측 분산을 원래 full
static proposal의 M2와 동일하게 취급하지 않는다. N32와 N4096의 target도 다르므로
두 grid의 표본 수를 직접 모델 효율 비교로 쓰지 않는다.

## 3. 현재 순서와 다음 모델 개발의 조건

1. canonical μ의 fresh fixed-count 회복을 완료·감사한다.
2. fine-grid static ordinary IS와 guide-free paired SMC를 별도 측정한다. accuracy를
   확보한 동일 target에서만 비교한다. continuum bias와 finite-grid 정확도는 구분한다.
3. 같은 grid/target에서 정적 구조 proposal을 강한 baseline으로 포함한다. 미세 grid용
   implicit mixture는 큰 mean 행렬을 피하는 계산 구현이지 새로운 learned model은 아니다.
4. 그 후 fresh training bank의 weighted mode/tail coverage와 독립 validation을 확보하고,
   원래 q를 개선한다. 강한 static baseline을 이기지 못하면 그 실패도 보존한다.

학습 가능한 time-group mixture weights 등 낮은 복잡도 교정은 후보가 될 수 있다.
고정 component들에서 M2(q_w)=E_p[g²/(Σw_k D_k)]는 simplex weights에 대해 convex다.
그러나 finite bank에서의 최소화는 population optimum이나 일반화를 보장하지 않는다.
fresh validation/final samples, whole-training 반복, 독립 M2 reference가 필요하다.
이를 새로운 일반 convexity 정리의 발견으로 포장하지 않는다.

baseline q0와 교정 q1의 blend qε=εq0+(1-ε)q1이면 qε>=εq0이므로
M2(qε)<=M2(q0)/ε다. 이는 안전한 second-moment 상한일 뿐 우위 보장이 아니다.
예컨대 ε=.5라면 최악 M2를 baseline의2배 이내로 제한하지만 실제 개선은 별도 확인해야 한다.
이 문서는 그 학습을 이미 수행했거나 대규모 학습이 승인됐다는 뜻이 아니다.

미세 격자 static pilot 시작 후 추가로 대조한 scope 주의: N4096 all-monitor
mixture는49,141성분이다. 원래 P3의 matched373-component 모델 비교가 아니라
별도 reference 방법 진단이다. 같은 복잡도의 q에 대한 우위 증거로 쓰지 않는다.
향후 P3에는 기존 물리적 시간/성분 상한을 유지하거나 새 비교 계약이 필요하다.
이 주석은 frozen pilot의 사전 설계에 소급 추가하지 않는다.
