# 모든 left Volterra driver에 대한 exact Gaussian price conditioning

2026-10-10. 기존 finite-grid physics law와 payoff를 보존하는 별도 μ estimator core.
현재 작은 격자의 dense 검증만 구현하며 생산 q, M2, P2, large-training을 승인하지 않는다.

## 직접 유도

z~N(0,I_2N), left monitoring drivers Y=A z∈R^(N-1), price-driving increments
ΔW=B z∈R^N. A는 full row rank이고 B Bᵀ=hI다.
Aᵀ=U R의 reduced QR, UᵀU=I를 만들면 X=Uᵀz~N(0,I_(N-1)), Y=RᵀX.
z|X=U X+(I-UUᵀ)ε이므로
E[ΔW|X]=B U X, Cov(ΔW|X)=hI-BUUᵀBᵀ.

X를 고정하면 모든 left v_i=ξ exp(ηY_i-.5η² Var(Y_i))가 고정된다.
w_i=sqrt(v_i), I=hΣv_i라 놓으면 log S_N|X는 정규분포로

- mean=log S0-.5I+ρ wᵀ B U X;
- variance=I-ρ²||UᵀBᵀw||² >= (1-ρ²) I >0.

따라서 G(X)=Φ((log K-mean)/sqrt(variance))이고 E[G(X)]=μ_N다.
기존 orthogonal-price-integrated payoff g(z)에 대해 G=E[g(z)|X]이므로
자연 prior 아래 Var(G)<=Var(g)가 성립한다. 이것은 알려진 Gaussian conditioning과
Rao–Blackwell 원리의 구체적 적용이지 새로운 일반 원리의 발견은 아니다.

## 인과성 및 likelihood 구분

원래 SDE의 price sum은 left volatility를 사용한다. 미래 driver 정보를 conditional
expectation 계산에 사용하는 것과 미래 정보를 control에 사용하는 것은 다르다.
본 core는 전자의 exact disintegration이며 SDE를 anticipative integral로 바꾸지 않는다.
새 X proposal을 쓰려면 그 공간의 정규화된 density와 support를 명시해야 한다.

원래 q(z)의 M2(q)=E_p[g²/Dq]를 G²로 대체하면 일반적으로 틀리다.
E[g²/Dq|X]는 별도 conditional integral이다. 새 μ estimator의 분산이 낮아질 가능성과
기존 q의 성능 향상은 다르다. natural-law variance contraction도 서로 다른 IS
proposal의 total-work 우월성을 보장하지 않는다.

## 구현과 제한

별도 `DriverConditionedTerminal`, 2<=N<=256 제한. analytical A/B, reduced QR,
Gaussian covariance/orthogonality/variance bound 검사. 기존 cached law를 바꾸지 않는다.
준비의 dense QR는 O(N³), 행렬 저장과 path scoring은 O(N²)다. FFT 방식보다
항상 빠르거나 큰 N에 적합하다고 주장하지 않는다. 범용 barrier payoff에는 적용하지 않는다.

conditional scalar residual의 Gauss–Hermite quadrature와 원래 full-g를 대조하고,
driver preservation, row separability, directional gradient, rho=0 oracle를 확인한다.
추가로 원래 rare μ의 독립 whole-run reference와 새 seed로 대조해야 실제 precision을
판정할 수 있다. paired-grid로 확장하려면 두 grid driver를 함께 조건부로 처리하는
joint rank/covariance와 비용 계약을 새로 유도해야 한다. 단일 grid core만으로 P2를 승인하지 않는다.
