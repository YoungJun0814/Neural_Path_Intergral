# Coverage·joint conditional 수학 검토

2026-10-09. 실험 전 고정한 [프로토콜](../plans/REFERENCE_COVERAGE_AND_JOINT_CONDITIONAL_PROTOCOL_2026-10-09_KO.md)에 대한 유도다. 아래 항등식은 기존 IS·조건부 기대값 원리의 적용이며 신규 학술 정리라는 주장이 아니다.

## 1. 평가 proposal과 모델 q의 분리

H_q=g²/D_q, D_q=q/p, Z=M₂(q)>0라 두면 risk-reference IS는 X=H_q p/r다. 정규화된 risk contribution 측도 ν_q(dz)=H_q(z)p(z)dz/Z를 정의하면

    Var_r(X)/Z² = ∫ν_q(z)²/r(z) dz − 1 = χ²(ν_q || r).

n개 IID draw의 상대분산은 이를 n으로 나눈 값이다. μ는 H=g, Z=μ로 같은 유도가 가능하다. 이 식의 ν_q는 실제로 모르는 normalizer를 포함하므로 최적 proposal을 이미 알고 있다는 뜻이 아니다. 학습 q가 좋은 것과 auxiliary r이 M₂를 잘 평가하는 것은 다르다. reference의 RSE가 작아도 finite sample에서 χ²를 과소평가할 수 있다.

## 2. 영역 coverage의 필수 제약

고정 영역 R에서 a=ν_q(R), b=r(R), 0<b<1이라 하자. R와 R의 여집합에 Cauchy–Schwarz를 적용하면

    χ²(ν_q || r) ≥ a²/b + (1−a)²/(1−b) − 1
                 = (a−b)² / [b(1−b)].

따라서 중요한 기여 영역에 proposal 질량이 거의 없으면 좋은 IID 상대분산이 불가능하다. 그러나 이것은 **하한**이지 정밀도 상한·충분조건이 아니다. 실제 Volterra의 a를 같은 실패한 sample로 추정하고 이를 참값으로 넣는 것은 인증이 아니다. 정확한 endpoint와 고정 영역 적분이 알려진 toy에서만 a를 oracle로 계산할 수 있다. a가 크다는 사실과 어느 proposal label이 선택됐는지는 별개다.

## 3. 독립 local scale mixture의 정확성 및 한계

서로 겹치지 않는 동일 크기 block b, block 수 K=d/b에서

    r = δp + (1−δ)/K Σ_k N(0, I+(λ−1)P_k).
    D_r(z)=δ+(1−δ)/K Σ_k λ^(−b/2) exp((1−1/λ)||P_k z||²/2).

P_k는 좌표 block의 orthogonal projection이다. logsumexp의 모든 component를 포함하며 label별 밀도로 나누지 않는다. 샘플은 p의 독립 잡음 중 선택 block만 sqrt(λ)배 한다. 정규화 상수는 알려져 있다. 따라서 exact normalized ordinary IS이며 학습이나 SMC normalizer가 없다.

r≥δp와 q≥δq p, 0≤g≤1에서 0≤X_μ≤1/δ, 0≤X_risk≤1/(δδq). 이 상한은 유한분산의 근거지만 상대 오차에 실용적인 상한을 자동 제공하지 않는다.

λ=25, b=2인 family가 모든 path geometry에 적합한 것은 아니다. 극단 경로가 여러 시간대에 분산된 경우 한 block 확대만으로 접근하기 어렵다. 예를 들어 ||z||²=36이 K=32개 block에 균등 분산되어 있으면 D_r≈0.1+0.9/25·exp(0.48·36/32)≈0.162다. 이는 해당 점의 prior보다 밀도가 낮다는 예시이지 실제 지배 경로의 형상을 증명한 결과는 아니다. 높은 차원의 모든 좌표를 무작정 크게 하는 것도 determinant 비용을 초래한다.

## 4. exact orthogonal shift-mixture conditional

고정 UᵀU=I, 행벡터 x=zU=(A,B), 원래 means를 m_jU=(m_Aj,m_Bj)로 둔다. identity covariance의 Gaussian은 회전 후 두 block이 component 내부에서 독립이다. mixture 전체에서는 label을 통해 의존한다.

    D_rA(A)=Σ_j w_j exp(A·m_Aj−||m_Aj||²/2).
    α_j(A)=w_j exp(A·m_Aj−||m_Aj||²/2)/D_rA(A).
    D_rB|A(B)=Σ_j α_j(A) exp(B·m_Bj−||m_Bj||²/2).
    D_r(z)=D_rA(A)D_rB|A(B).

α를 원래 w로 고정하면 일반적으로 잘못된 conditional law다. identity covariance가 아닌 경우 이 공식은 틀리므로 이번 구현은 nonzero rank를 거부한다. fixed rotation이 아니라 data-dependent 선택을 하면 별도 선택 조건부 법칙이 필요하다.

    E_rB|A[ H_q(z)/D_rB|A(B) | A ] = E_pB[H_q(z)|A].

outer weight 1/D_rA를 곱하면 original full-r IS와 같은 target다. exact inner 평균은 RB지만 유한 L은 자동으로 다른 proposal보다 낫지 않다. 전체 defensive floor가 있어도 conditional natural component 확률 α_0(A)는 A에 따라 매우 작아질 수 있다. 이를 고정 0.1이라고 취급하지 않는다.

코드에서 `split`은 zU, `join`은 [A,B]Uᵀ이며 scalar Jacobian 보정을 추가하지 않는다. 이는 orthogonal 변환의 |det U|=1을 사용한다. 수치 구현은 float64 orthogonality 허용오차 내 검증이지 모든 floating-point 입력에 대한 형식 증명은 아니다. exp posterior의 극소 확률 underflow 등 표준 유한정밀도 한계가 있다.

## 5. 구현 검증과 해석

독립 SciPy Gaussian logpdf, 단변량 적분 normalization, sampling marginal covariance, rotated mixture factorization, 조건부 posterior·평균·분산 및 원래 q를 유지한 risk integral을 검사한다. 새 conditional core는 여러 시간대 잔차를 다루기 위한 기반이며 실제 Volterra nested sampler를 생산용으로 배치한 것이 아니다. 일반 joint block의 H_q를 해석적으로 적분할 수 있다는 주장도 없다.

새 실제 개발 비교에서는 μ와 M₂를 같은 draw로 계산하고 pairing을 명시한다. 둘의 추정 오차를 독립이라고 가정하거나 M₂−μ²의 SE를 독립 공식으로 산출하지 않는다. 방법 간·반복 간 stream은 독립이다. 이번에는 original model variance의 결합 SE나 모델 훈련에 sample을 사용하지 않는다.

참고: 방어적 mixture는 [Owen–Zhou (2000)](https://artowen.su.domains/reports/seis.pdf)의 기존 연구와 관련된다. 그 논문의 control variates 또는 최적 mixture weight 결과를 이번 raw IS 코드가 구현했다고 주장하지 않는다. conditional Monte Carlo의 분산 감소는 [Owen의 해당 장](https://artowen.su.domains/mc/Ch-var-basic.pdf)에 정리된 기존 원리다.
