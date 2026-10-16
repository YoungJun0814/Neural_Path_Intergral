# 야간 reference 복구: 수학 명세와 주장 경계

작성일: 2026-10-08. 대상은 고정 finite-grid Gaussian Volterra terminal downside다. 아래 항등식·가정 검토는 실제 reference 정밀도 인증이나 새 정리의 학술적 독창성 선언이 아니다.

## 1. 추정 대상 보존

p=N(0,I), 0≤g≤1, q≥δq p, r≥δr p일 때

\[
\mu=E_p g,\qquad M_2(q)=E_p H_q,\qquad H_q=g^2/(q/p).
\]

q는 original event proposal, r은 보조 평가 proposal이다. reference를 바꾸어도 원래 M₂가 바뀌지 않는다. 반대로 보조 estimator의 분산이 감소해도 original q의 event variance가 줄었다는 뜻은 아니다.

\[
\operatorname{Var}_q(gp/q)=M_2(q)-\mu^2.
\]

학습 bank·모델 q는 이번 야간 연구에서 동결한다.

## 2. μ의 조건부 평균과 M₂의 density 변경

z=(A,B), p=p_Aφ_B이고 r_A는 정확한 marginal이다. original full-r ordinary IS contribution X=g p/r에 대해

\[
E_r[X\mid A]=\frac{p_A(A)}{r_A(A)}\int g(A,B)\varphi_B(B)dB.
\]

따라서 exact conditional μ evaluator는 full-r μ estimator의 Rao–Blackwellization이다. 유한분산이면 variance가 증가하지 않는다. sampling/evaluation/caching 비용을 포함한 시간 효율은 별도 측정 대상이다.

반면 실제 M₂ L=1 estimator는 A~r_A, B~φ_B에서

\[
Y=\frac{p_A}{r_A}\frac{g^2}{D_q}.
\]

이것은 sampling density r_Aφ_B를 쓰는 IS이다. E[Y]=M₂이지만 full-r risk contribution g²p²/(qr)보다 variance가 작다는 보장은 없다. gbar² 또는 q_A를 쓰면 원래 M₂를 보존하지 않는다.

defensive marginal r_A≥δr p_A이면 0≤Y≤1/(δrδq). 이 상한은 finite variance의 근거지만 tiny mean의 유용한 relative precision을 보장하지 않는다.

## 3. nested 분해와 비용 최적점

고정 A에서 독립 B_l~φ_B를 사용하여

\[
Y_L=w_A(A)\frac1L\sum_l H_q(A,B_l),\qquad w_A=p_A/r_A
\]

를 만들면 total variance 법칙으로

\[
V(L)=V_{out}+V_{in}/L,
\]

\[
V_{out}=\operatorname{Var}_{r_A}[w_A E_{\varphi_B}(H_q\mid A)],\quad
V_{in}=E_{r_A}[w_A^2\operatorname{Var}_{\varphi_B}(H_q\mid A)].
\]

한 outer unit 비용이 c_out+L c_in이고 고정 예산 W를 사용하면, rounding을 제외한 estimator variance는

\[
\frac{(V_{out}+V_{in}/L)(c_{out}+Lc_{in})}{W}.
\]

모든 상수가 양수일 때 미분으로

\[
L_*^2=V_{in}c_{out}/(V_{out}c_{in}).
\]

연속 최적점 근처 두 정수와 L=1·상한을 평가한다. 실제 batching·density 비용이 비선형이면 이 식은 모델일 뿐이며 측정한 discrete cost를 사용한다.

경계 사례:

- V_in=0, V_out>0, c_in>0이면 L=1이 최적이다.
- V_out=0, V_in>0이면 비용 모델에서 F(L)=V_in c_out/L+V_in c_in이므로 큰 L로 감소하나 총예산·outer count 제약을 포함해야 한다.
- c_in=0이면 L 증가가 비용 없이 내부 variance만 줄이지만 실제 CDF/density가 무료인지 별도 검토한다.
- sample noise로 추정한 V_out<0은 population 음수분산이 아니다. 0으로 잘라 infinite L을 선택하지 않는다.

이 분해/최적화는 알려진 conditional/nested Monte Carlo 관계의 적용이며 새로운 일반 원리로 주장하지 않는다.

## 4. Gaussian-bump oracle 검토

g=Σ a_i exp(-||z-m_i||²/(2s_i²)), a_i>0, Σa_i≤1의 μ는 Gaussian square completion으로 구한다. M₂(q=p)는 ordered pair 적분이다.

\[
\tau_{ij}=1+s_i^{-2}+s_j^{-2},\quad b_{ij}=m_i/s_i^2+m_j/s_j^2,
\quad c_{ij}=\|m_i\|^2/s_i^2+\|m_j\|^2/s_j^2,
\]

\[
E_p[g^2]=\sum_{i,j}a_i a_j\tau_{ij}^{-d/2}
\exp\{-\tfrac12(c_{ij}-\|b_{ij}\|^2/\tau_{ij})\}.
\]

prior precision 1을 중복하지 않는다. i≠j 교차항을 빠뜨리지 않는다. 실제 구현은 별도 SciPy 2차원 적분과 대조한다. fractional β의 (Σbump)^β를 이 endpoint Gaussian-mixture 공식으로 계산하지 않는다.

## 5. SMC와 island normalizer

fixed bridge/resampling과 prior-invariant mutation의 기존 SMC 계약을 사용한다. observer는 cloned particle/weight를 읽고 RNG를 소비하지 않으므로 선언한 알고리즘을 바꾸지 않는다. clone에 대한 mutation을 가하는 회귀에서도 원래 trajectory가 동일한지 검사한다.

island k의 normalizer estimator Zhat_k가 유효하면

\[
\widehat Z_{islands}=\frac1K\sum_k\widehat Z_k
\]

도 같은 mean을 가진다. 독립 island이면 variance=K⁻²ΣVar(Zhat_k)이지만, 같은 총 particle의 큰 population과 비교한 우위는 보장되지 않는다. 추론은 독립 aggregate whole-run별로 수행한다. 실패한 island를 제외한 평균은 정해진 estimator가 아니다.

총 M initial IID Gaussian draws가 같은 경우 fixed region R의 초기 hit 확률은 allocation과 무관하게 1-(1-p_R)^M이다. 이후 resampling·mutation과 region 소실을 따로 관찰한다. initial hit은 estimator 성공의 필요조건도 충분조건도 아니다.

작은 sample RSE는 unseen tail을 보지 못할 수 있다. 8-run bootstrap·정규근사·leave-one-run sensitivity는 개발 진단이지 unconditional tail certificate가 아니다.

## 6. 계산 cache와 law의 구분

scoped cache는 immutable 문제 파라미터에 묶인 준비값이다. N/T/H/η/ξ/ρ/task가 바뀐 객체에는 새 cache를 만든다. global LRU cache나 임의 parameter override를 지원하지 않는다. 내부 텐서를 외부에서 변경하는 API를 사용하지 않는다.

BLP kernel FFT·Cholesky·variance compensator와 mixture mean norm/log weight를 재사용한다. 모든 candidate의 비선형 variance·integrated variance·가격 drift·조건부 CDF는 다시 계산한다. clipping·float32·memory/KL truncation은 없다.

별도의 cached terminal evaluator는 사용하지 않는 spot/running-min/put/call 출력 계산을 생략하지만, original left-point log-price cumsum과 동일한 conditional digital law를 계산한다. 비교 oracle는 원래 simulator/payoff 경로다. deterministic agreement는 수치 구현 근거이지 모든 입력의 형식적 동등성 증명이 아니다.

## 7. 다음 joint-direction 후보의 정확한 출발점

고정 orthogonal U로 z=U_A A+U_B B를 나누면 prior는 독립 Gaussian blocks다. identity-covariance shift-mixture reference의 marginal과 conditional은

\[
r_A(A)=\sum_j w_j\varphi(A-m_{j,A}),
\]

\[
\alpha_j(A)=\frac{w_j\varphi(A-m_{j,A})}{r_A(A)},\quad
r(B\mid A)=\sum_j\alpha_j(A)\varphi(B-m_{j,B}).
\]

여러 시간대 joint 방향의 exact hbar=E_{φ_B}[H_q|A]는 적법한 RB 대상을 제공한다. natural B 대신 r(B|A)를 사용한 내부 contribution H_q φ_B/r(B|A)도 같은 conditional mean을 가진다. 이를 L=1로 쓰면 original full-r risk IS와 연결되며, 정확한 conditional 평균은 해당 full-r estimator의 RB가 된다.

그러나 hbar를 정확히 계산/샘플링하는 비용과 중요한 outer mode 탐색은 별도 병목이다. joint subspace가 항상 이득이라는 정리는 없다. path-dependent spike 선택은 고정 orthogonal 분할과 다르며 selection event의 조건부 법칙을 먼저 유도해야 한다.

이 후보는 다음 설계 검토용 수식이며 이번 밤에 새 실제 sampler를 추가하거나 reference 관문을 대체하지 않는다.

## 8. 완료 수준과 연구 주장

- exact algebra: 위 finite-grid 적분·분산·island 평균의 명세.
- tested implementation: deterministic oracle·seed replay·source/target audit.
- empirical development: fresh reference 비교·toy/실제 coverage 관측.
- unresolved: guide-independent rare-event precision, 전체 fixed-q production, mesh/continuum, 전체 학습 효율.
- not established: 새 모델 우위, 최상위 저널 novelty·채택 가능성.

최종 실험 수치·발견 오류·검증 상태는 야간 실행 결과 보고서에서 별도로 기록한다.
