# 독립 reference 방법 재설계: 원래 estimand 보존과 조건부 비용 절감

작성일: 2026-10-08. 상태: **방법 재설계와 작은 identity 검사 완료; 신규 reference 구현·pilot·production 미실행.**

후속 실행 상태(동일 날짜): 생산용 core/실행기와 conditional·kernel·fixed-block 개발 실험을 구현·실행했다. 비용 개선 및 production allocation 관문은 실패했으므로 fresh production/P2는 미실행이다. 아래는 사전 계획을 보존한 내용이며, 실제 수치·오류 수정·검증 상태는 [실행 보고서](../reviews/REFERENCE_REDESIGN_IMPLEMENTATION_AND_EXPERIMENT_REPORT_2026-10-08_KO.md)에 기록한다.

이 문서는 [통합 V2 계획](MODEL_STRUCTURAL_IMPROVEMENT_PLAN_V2_2026-10-08_KO.md)의 P1 실패 이후 분기를 구체화한다. P2 이후 관문을 완화하거나 과거 실패를 성공으로 바꾸지 않는다. 대형 모델·새 operator·차원 확대·커밋·푸시는 이번 범위에 없다.

## 1. 판단과 현재 근거

주모델을 더 복잡하게 만들기보다 **원래 μ/M₂를 더 신뢰성 있게 평가하는 방법**을 바꾼다. 선택한 순서는 다음과 같다.

1. 마지막 Gaussian pair의 구조를 사용해 μ를 해석적으로 조건부 적분한다.
2. M₂는 같은 구조를 사용하되 **제곱과 원래 q의 분모를 그대로 보존**하는 unbiased nested estimator를 먼저 구현한다. 인증되지 않은 quadrature를 참값으로 채택하지 않는다.
3. 이 조건부 계산의 이득이 부족하면 시간대별 고정 block으로 확장한다. 미래 Volterra 경로를 정확히 재계산해야 하며 비용 이득이 없으면 중단한다.
4. 공통 정적 guide blind spot을 점검하는 별도 경로로 Gaussian-prior elliptical-slice SMC를 제한적으로 검증한다. 이것도 SMC이므로 독립된 종류의 normalizer 증명이라고 포장하지 않는다.
5. fresh production에서 원래 정확성 계약을 충족한 경우에만 P2를 연다.

[bounded repair 결과](../reviews/STRUCTURAL_V2_P1_BOUNDED_REPAIR_RESULT_2026-10-08_KO.md): 39,944,192 potential 평가, 약 798초. canonical risk의 guide-free local A/B RSE는 약 8.18%/7.48%, global C는 약 2.12%였다. 위험값 local A의 production 필요 whole-run은 canonical 1,429회, high-η 1,444회로 최대 256회를 넘었다. 전체 forecast는 약 31.67억 평가·20.52시간으로 1.6억 평가·2시간 cap을 넘었다. 이는 production 측정치가 아닌 pilot 기반 보수적 예측이다.

위험 contribution은 초·중반 spike에 집중되는 반면 μ contribution은 상대적으로 후반·분산된 경로에 집중했다. 따라서 마지막 pair 적분만으로 **핵심 위험 reference 문제가 해결될 것이라고 가정하지 않는다**. coarse geometry bin 관측은 미발견 mode 부재 증명이 아니다.

## 2. 변경하지 않는 수학적 대상

고정 finite-grid, ε=1, terminal downside task에 한정한다. 표준정규 local 좌표 z∈R^(2N)의 밀도를 p, 이미 독립 price driver를 적분한 조건부 사건확률을 g(z)라 한다.

    μ = ∫ g(z) p(z) dz
    M₂(q) = ∫ g(z)² p(z)² / q(z) dz = E_p[g(z)² / D_q(z)]
    D_q = q/p,  q ≥ δ_q p,  δ_q > 0.

q는 평가할 **원래 고정 proposal**이다. r은 reference 생성용 proposal로 별개다. q를 재학습·marginal proposal로 대체하거나 payoff를 더 smoothing한 모델의 M₂를 보고하면 기존 모델 성능을 검증한 것이 아니다.

μ와 M₂는 각각 별도 seed·표본 역할·오차·선택 규칙을 유지한다. ordinary likelihood, exact normalized mixture, defensive floor, whole-run SE, source snapshot, pilot/production 분리는 필수다.

## 3. 정확한 마지막 pair 구조

z=(A,B), A∈R^(2N−2), B=(x,y)=(마지막 Brownian 표준좌표, 마지막 local-integral 잔차좌표). p=p_A φ₂.

현재 BLP Cholesky 첫 행은 (√dt,0)이고 가격은 left-point variance를 사용한다. 따라서 마지막 pair는 마지막 가격 Brownian increment와 terminal volatility에 영향을 주지만, I와 마지막 가격에 사용되는 variance에는 영향을 주지 않는다. y는 이 terminal payoff에 영향을 주지 않는다. 다른 discretization·barrier·ε≠1에는 자동 확장하지 않는다.

    I(A) = dt Σ(i=0,…,N−1) v_i(A)
    J_prev(A) = Σ(i=0,…,N−2) √v_i(A) ΔW_i(A)
    a(A) = log(K/S0) + I(A)/2 − ρ J_prev(A)
    b(A) = ρ √(v_(N−1)(A) dt)
    c(A)² = (1−ρ²) I(A)
    g(A,x,y) = Φ((a(A)−b(A)x)/c(A)).

ρ가 음수여도 b를 절댓값으로 바꾸지 않고 이 부호 있는 식을 사용한다.

### 3.1 μ: 해석적 적분

    gbar(A) = E_φ₂[g(A,B)] = Φ(a(A)/√(c(A)²+b(A)²))
    μ = E_(r_A)[gbar(A) p_A(A)/r_A(A)].

r_A는 r의 **정확한 marginal density**다. B=0에서 full r을 평가하는 것은 marginalization이 아니다. 원래 full-r IS contribution을 r(B|A)에 대해 조건부 평균내면 위 식이 나오므로, 같은 outer marginal에 대해 분산은 증가하지 않는다. 이는 경로당 총시간 개선이나 다른 proposal 대비 우위 정리가 아니다.

### 3.2 M₂: 평균 후 제곱 금지

    H_q(A,B) = g(A,B)² / D_q(A,B)
    hbar_q(A) = E_φ₂[H_q(A,B)]
    M₂(q) = E_(r_A)[(p_A/r_A) hbar_q(A)].

gbar²/D_(q_A)로 바꾸면 원래 M₂와 다르다. q=p에서도 E[g²|A]≥E[g|A]²이고 일반적으로 부등호가 엄격하다. g가 y와 무관해도 q가 y에 의존하면 y를 분모에서 지울 수 없다.

## 4. 우선 구현: unbiased nested reference

Outer A_i∼r_A, inner B_il∼φ₂를 독립 생성하고 고정 L을 사용한다.

    Y_i = (p_A(A_i)/r_A(A_i)) (1/L) Σ(l=1,…,L) H_q(A_i,B_il)
    M₂_hat = (1/n) Σ_i Y_i.

Tonelli와 defensive bound로 E[Y_i]=M₂(q). 내부 수치적 quadrature 편향이 없으며, 각 outer unit 전체가 독립 추론 단위다. nL개의 inner contribution을 IID처럼 취급하지 않는다.

r_A≥δ_r p_A이면 0≤Y_i≤1/(δ_r δ_q). 이는 유한분산 보장이지만 희귀값에 유용한 상대오차·tail coverage 인증을 자동 제공하지 않는다. natural outer r_A=p_A도 유효하나 희귀 경로 탐색이 매우 비효율적일 수 있다.

    Var(Y_i) = Var_(r_A)[(p_A/r_A) hbar_q]
             + (1/L) E_(r_A)[(p_A/r_A)² Var_φ₂(H_q|A)].

이 분해의 비교 기준은 r_A(A)φ₂(B)에서 L=1인 estimator다. 원래 full-r estimator보다 항상 좋은 것이라는 주장은 하지 않는다. 공통 outer에 L=1,4,16,64를 비교해 감소 가능한 inner 분산과 남는 outer 분산을 분리한다. 파일럿에서 선택한 L은 production 전에 동결한다.

**비용 구조:** A의 경로는 한 번 계산하고, L개의 g는 위 식의 log-CDF로 재구성한다. 현재 rank-zero shift mixture에서

    D_q(A,B)=Σ_j w_j exp(m_j,A·A + m_j,B·B − ||m_j||²/2).

A 부분을 캐시하면 내부 계산에 FFT 경로 재실행이 필요 없다. density는 logsumexp, g는 log_ndtr, M₂ contribution은 log-domain에서 계산한다. 0으로 underflow한 contribution을 관측된 정확한 0으로 간주하지 않는다.

### exact marginal 구현 범위

첫 구현은 covariance=I인 shift mixture에만 적용하고 다른 covariance면 명시적으로 거부한다. r_A는 원래 weight와 잘린 mean으로 구성한 Gaussian mixture이다. marginal sample은 full sample의 prefix 또는 정확한 marginal sampler 둘 다 가능하나 seed 역할과 실제비용을 기록한다.

향후 finite-rank covariance는 Σ_AA와 Σ_B|A=Σ_BB−Σ_BA Σ_AA^−1 Σ_AB, component posterior weight까지 구현·검증한다. U의 행을 잘라 기존 orthonormal-U API에 그대로 넣지 않는다. frozen artifact의 rank/weight/dimension을 실행 전 검사한다.

## 5. 수치적 적분은 후순위

2차원 Gaussian에서 ||B||>R의 확률은 exp(−R²/2). 따라서 hbar truncation의 절대오차는 최대 exp(−R²/2)/δ_q이며, 원래 M₂의 전체 적분 bias도 이 상한 이하이다. outer weighted contribution의 개별 상한에는 추가 1/δ_r가 필요하다.

M₂≈10^−9인 셀에서 작은 절대오차도 큰 상대오차일 수 있다. 현재 pilot 평균을 참값으로 써서 상대오차 bound를 확정하지 않는다. GH order 증가·adaptive quad의 오차 추정·두 코드 일치는 rigorous enclosure가 아니다. 유효한 tail bound와 내부 quadrature enclosure 및 floating-point 오차 계약이 없는 경우 reference qualification에 사용하지 않는다. unbiased nested 방법을 먼저 선택한 이유다.

## 6. 마지막 pair가 부족할 때의 제한된 확장

마지막 pair가 담당하는 variance 비중이 작으면 inner 적분으로 outer spike 탐색 문제가 해결되지 않는다. 이 경우 early/mid/late의 **사전 고정 한 구간씩**을 후보로 둔다.

- 좌표 block은 개발 pilot에서 정하고 새 production 전에 고정한다.
- 해당 좌표를 바꾸면 그 이후 Volterra convolution·variance·I·J를 모두 정확히 재계산한다. 미래 variance를 고정한 마지막-pair 공식을 재사용하지 않는다.
- 경로별 최대 spike 좌표를 선택한 후 단순 φ 적분하면 선택조건이 누락될 수 있다. 조건부 selection boundary를 유도하지 않는 한 금지한다.
- block 확장은 unbiased inner MC로 시작한다. clipping·memory truncation·KL truncation·수치근사로 target을 바꾸지 않는다.
- 최대 3개 고정 block, 동일 총시간 cap. inner 분산이 줄어도 실제 wall-time×Var가 개선되지 않으면 중단한다.

이는 dimension 확대가 아니라 기존 좌표의 reference 계산 변경이다. 마지막-pair identity는 수학적으로 확정되지만 block 성능은 미지수다.

## 7. guide-free 대조: elliptical-slice SMC

각 bridge target π_β(z)∝p(z)h(z)^β에 대해 h=g 또는 g²/D_q를 사용한다. ν∼N(0,I), z(θ)=z cosθ+ν sinθ인 Gaussian ellipse와 slice threshold를 사용하면 Gaussian-prior posterior에 대한 elliptical slice transition을 구성할 수 있다. [원 논문](https://proceedings.mlr.press/v9/murray10a.html)은 이 구조의 근거다. 우리 희귀 M₂에 빠르다는 보장은 아니다.

핵심 계약:

1. 기존 fixed bridge/resampling normalizer를 유지하고 mutation만 별도 API로 대조한다. β schedule은 독립 pilot로 선택 후 고정한다.
2. log h는 원래 deterministic full-coordinate potential을 사용한다. noisy inner 평균을 무조건 log로 넣으면 다른 target을 만들 수 있다. pseudo-marginal 확장은 별도 증명 전 금지한다.
3. ellipse angle bracket·slice 변수·rejection shrink 로직을 원 알고리즘대로 검증한다. accepted move 횟수와 potential 호출은 다르므로 모든 시도·실패·시간을 센다.
4. rejection cap에서 현재값을 반환하면 invariant kernel이라고 가정하지 않는다. 예산 초과는 전체 실험의 protocol failure로 기록한다. 완료한 run만 선택적으로 평균내거나 완료될 때까지 교체해 조건부 표본을 만드는 것을 금지한다.
5. terminal particle을 IID처럼 사용하지 않는다. independent whole-run normalizer로 SE를 계산한다.
6. guide-free는 **공통 static proposal 미사용**이라는 뜻이다. risk target 자체가 원래 q에 의존하는 것은 필수이며, 이 의존성을 제거하지 않는다.

이는 pCN과 다른 이동 geometry지만 여전히 SMC normalizer다. nested ordinary IS와 결합해 estimator mechanism과 proposal geometry의 두 축에서 교차검증한다. 모든 방법이 동일 영역을 놓칠 수 있다는 한계는 남는다.

## 8. 후보 비교와 순서

| 후보 | 원래 μ/M₂ 보존 | 주요 기대 | 주요 위험 | 결정 |
|---|---|---|---|---|
| 해석적 마지막 pair μ | 정확한 identity | 추가 FFT 없이 smoothing | μ만 개선, M₂ 해결 아님 | 즉시 작은 core |
| 마지막 pair nested M₂ | unbiased, exact density 필요 | cheap inner로 위험값 분산 분해 | outer spike variance 잔존 | M₂ 첫 microstudy |
| guide-free ellipse SMC | invariant transition·fixed SMC 계약 필요 | static guide 밖 geometry 탐색 | 초기 bridge 붕괴·variable work | 작은 oracle 뒤 제한 대조 |
| 고정 early/mid block nested | 정확한 future 재계산 시 unbiased | 핵심 spike 좌표 평균화 | FFT 비용·차원·outer variance | 첫 microstudy 실패 시만 |
| certified quadrature | enclosure 확보 시만 | inner noise 제거 | tiny M₂의 상대 bias·구현비 | 후순위 |
| RQMC | 적절한 randomization 시 unbiased | smooth outer integrand | scramble SE·희귀 mode 누락 | 현재 후보에 추가하지 않음 |
| 추가 pCN/temperature sweep | target 자체 유지 가능 | 쉬운 변경 | 이미 bounded repair 소진 | 금지 |

Gaussian smoothing 자체는 새 원리가 아니다. [Bayer–Ben Hammouda–Tempone의 numerical smoothing 연구](https://arxiv.org/abs/2111.01874)는 조건부 적분과 smoothing이 알려진 흐름임을 보여준다. 우리 novelty 후보는 **원래 proposal의 M₂를 보존하는 조건부 residual 평가·Volterra 비용 분해·검증 가능한 실패 경계**에서 찾아야 하며, 아직 novelty가 입증된 것은 아니다.

## 9. 실행 단계와 통과/중단 규칙

### REF0 — identity·density 계약 (작은 계산만)

산출물: 마지막 pair μ evaluator, cached H_q evaluator, rank-zero marginal API, independent outer/inner ledger.

필수 검사: N=1/4/32, η=기본/높음, ρ=음/0/양, 여러 x/y에서 actual simulator 일치; exact μ oracle; q=p에서도 평균제곱 불일치 회귀; mixture marginal 정규화·sample moments·floor; unsupported task/covariance/ε 거부; extreme log-value·overflow 실패경로. 작은 Gaussian mixture analytic oracle의 M₂ reference도 준비한다.

### REF1 — 유한 microstudy (사전 고정 후 실행)

canonical/high-η, parent rep 0 개발자료만 사용한다. L=1/4/16/64, 4개의 독립 outer batch seed, batch당 최대 512 outer unit. 별도 nested pair replicates로 inner/outer 분산을 추정하며 negative noise-subtracted outer 추정값은 0으로 성공 처리하지 않고 불확실성으로 기록한다. 총 300만 potential-equivalent 평가·15분·RSS 4GiB 중 먼저 도달하는 한도에서 종료한다. 실제 FFT/CDF/density component/seed/RSS/wall-time을 모두 기록한다.

gain 판정: 두 셀에서 wall-time×unit variance 점추정 20% 이상 감소를 후보 진입 기준으로 쓰되 4batch만으로 통계적 우위 선언하지 않는다. L 증가에 따라 outer floor가 지배하거나 비용이 악화되면 마지막-pair M₂를 해결책으로 확장하지 않는다. tiny RSE와 blind-spot 진단은 동시에 본다.

### REF2 — guide-free kernel oracle와 bounded 대조

ellipse core를 Gaussian analytic posterior 및 known normalizer toy에서 검증한다. fixed β SMC 회귀, seed replay, variable call ledger, cap failure 테스트가 먼저다. 이후 두 셀에서 방법당 8 independent whole-runs의 feasibility pilot만 수행한다. 최초 cap: 500만 potential 평가·30분·4GiB. guide/template를 추가하지 않으며 실패한 whole-run을 빼고 평균내지 않는다. 이 규모는 qualification 아닌 비용·mixing 진단이다.

### REF3 — 필요 시 고정 block 한 번의 확장

REF1이 outer variance 병목임을 보이면 최대 3개 block 중 하나를 별도 microstudy로 선택한다. guide-free REF2 실패를 block의 성공으로 대신하지 않는다. 방법선택/seed/budget/source를 먼저 봉인하며 총 cap은 REF1과 동일하다. 실패하면 새 reference 설계 또는 계산예산 재승인으로 돌아가고 P2를 열지 않는다.

### REF4 — fresh production 및 기존 P1 gate

개발 microstudy 결과를 production에 pooling하지 않는다. production source·q artifact SHA·method·L/block·bridge·전체 budget·multiplicity family·termination을 새 config에 봉인한다. 기존 최소/최대 unit count, 1.6억 평가·2시간 cap과 safety factor를 유지한다. forecast가 넘으면 시작하지 않는다.

원래 P1 계약: 목표 allocation RSE 1.5%, qualification 최대 RSE 2.5%, 상대 equivalence margin 10%, family α=.05, risk 30/probability 2 비교; leave-one-out shift≤5%, maximum-unit share≤15%, block sensitivity 및 보수적 simultaneous uncertainty 확인. 통과는 낮은 sample RSE 하나가 아니라 **관련 모든 고정 q의 precision+equivalence+sensitivity+binding**을 뜻한다. 방법 수를 늘려 비교 family가 바뀌면 사전 multiplicity 계약을 재계산한다.

guide-free와 guide-based reference가 일치하지 않으면 평균내어 통과시키지 않는다. 한 방법만 accuracy를 얻으면 total-time 우위 비교 자격도 얻지 못한다. 통과 후에만 P2 mesh로 이동한다.

## 10. 구현 파일과 인터페이스 제안

- `structural_v2_conditional_reference.py`: scope guard, last-pair coefficients, exact log μ mean, cached full-q M₂ inner evaluation.
- `gaussian_mixture_marginal.py`: rank-zero exact marginal, natural floor, rank guard.
- `nested_reference_statistics.py`: outer-unit moment merge·nested work ledger·independent inner variance diagnostic.
- `elliptical_slice_kernel.py`: Gaussian-prior transition·variable-call result·explicit failure.
- 기존 `weighted_tempered_smc.py`: kernel adapter만 additive 변경, 기존 pCN 회귀 보존.
- 신규 runner/config/result/report는 `reference_redesign_*` prefix로 historical P1와 분리한다. 결과 schema에 estimand, marginal scope, inner L, outer count, mechanism, numerical bias status를 명시한다.

이는 예정 파일명이며 위 생산용 파일은 아직 구현하지 않았다. reference correctness를 검증하기 전에 주모델 q나 학습 bank를 수정하지 않는다.

## 11. 이번 검토에서 실제 확인한 범위

`tests/test_structural_v2_reference_redesign_identities.py`를 추가했다. N=1/4/32와 η=1.5/3.0의 실제 FFT simulator에서 마지막 pair를 바꾸어 I 불변성과 g의 signed affine-CDF 식을 검사한다. 별도 Gaussian-CDF identity와 평균/제곱 비가환도 검사한다.

이 검사는 reference production·희귀 tail coverage·ellipse invariance·분산 효율·연속시간 수렴을 검증하지 않는다. 이론은 finite-grid·현재 Cholesky·left-point variance·terminal threshold·ε=1에 한정한다. 수치 일치 테스트를 형식적 증명이나 무오류 보증으로 표현하지 않는다.

실행 결과: 신규 identity 검사 7개 통과, 신규 파일 Ruff 통과, tracked diff 공백 검사 통과. 이번 변경은 계획과 design-check 테스트이며 생산용 estimator를 변경하지 않았다. 전체 회귀의 이전 1,200개 통과 결과는 앞선 bounded repair 검증의 결과이고 이번 신규 전체 회귀 실행으로 표기하지 않는다.

## 12. 최종 결론

재설계의 핵심은 **μ에는 정확한 해석적 평균, M₂에는 원래 분모를 보존한 unbiased 조건부 내부 평균**이다. 마지막 pair는 안전하고 저렴한 출발점이지 성공 보장 수단이 아니다. 이후 분기는 계산으로 확인한 outer variance와 guide-free mixing 실패에 따라 결정한다.

현재는 신뢰할 reference를 만드는 단계이며, 새 학술 기여나 최상위 저널 제출 가능성이 입증된 단계가 아니다. 이 관문을 통과하지 못하면 모델 크기와 논문 주장을 키우지 않는다.
