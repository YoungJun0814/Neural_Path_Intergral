# Controlled Multilevel Volterra Path-Integral Estimator: 실행계획 v7

Date: 2026-07-17

Status: theory contract frozen before implementation

Execution note (2026-07-17): implementation and exactness gates passed, but the
frozen end-to-end work gate failed. See
[`docs/phase_reviews/G7_CONTROLLED_MLMC_FALSIFICATION_2026-07-17.md`](docs/phase_reviews/G7_CONTROLLED_MLMC_FALSIFICATION_2026-07-17.md).

## 0. 결정

Plan v6의 SDV residual 가설은 strong four-segment CEM을 이기지 못해 중단되었다.
Plan v7은 실패한 neural controller를 확장하지 않는다. 다음 두 자산만 결합한다.

1. finite-grid Volterra importance-sampling error/variance theory;
2. 검증된 time-piecewise two-driver CEM과 adjacent-level MLMC coupling.

주 연구 질문은 다음이다.

> 하나의 fine-grid causal change of measure가 rough-Volterra hard path-event의
> 발생확률과 adjacent-grid correction variance를 동시에 줄여 end-to-end work를
> 개선할 수 있는가?

## 1. 선행연구 및 claim 경계

이미 알려진 결과:

- BLP hybrid scheme은 singular recent-cell Wiener integral과 과거 kernel의
  step approximation을 결합한다: <https://arxiv.org/abs/1507.03004>.
- non-Lipschitz/digital/barrier payoff의 MLMC는 별도의 coupling 또는 smoothing이
  필요하다: <https://doi.org/10.1007/s00780-009-0092-1>,
  <https://arxiv.org/abs/2209.03017>.
- rough Bergomi 계열에서 MLMC 자체는 VIX payoff에 적용된 바 있다:
  <https://arxiv.org/abs/2105.05356>.
- multilevel importance sampling은 다른 확률계 계열에서 이미 존재한다:
  <https://arxiv.org/abs/2208.03225>.

따라서 다음을 신규성으로 주장하지 않는다.

- MLMC와 importance sampling을 단순히 함께 사용했다는 주장;
- BLP 또는 CEM 자체의 신규성;
- hard indicator에 smooth-payoff weak-rate theorem을 그대로 적용한 주장;
- finite-grid 결과를 continuously monitored event theorem으로 부르는 것.

조건부 신규성 후보는 다음 교집합이다.

- exact adjacent-grid BLP Gaussian coupling;
- one fine-grid likelihood로 가중하는 signed hard path-functional correction;
- disagreement-focused CEM 및 defensive marginal mixture;
- controlled correction variance와 total-work의 실증/이론 분해.

## 2. Target hierarchy

레벨 `ell=0,...,L`에서

$$
N_\ell=N_0 2^\ell,
\qquad h_\ell=T/N_\ell
$$

를 사용한다. 각 `P_ell`은 해당 격자에서 구현된 BLP rBergomi 법칙이다.
primary hard payoff는

$$
H_\ell=
1\left\{
\min_{1\le i\le N_\ell}S_i^{(\ell)}\le B_{hit},
\quad
h_\ell\sum_{i=1}^{N_\ell}
1_{\{S_i^{(\ell)}\le B_{stress}\}}\ge\tau_{occ}
\right\}.
$$

Primary target은 우선 finest declared grid의 `E[H_L]`이다. Continuous-monitoring
limit는 bias-rate 진단 없이 주장하지 않는다.

## 3. Adjacent BLP coupling

`h=h_ell`, coarse step은 `2h`, `alpha=H-1/2`라 하자. 한 coarse interval의
첫 fine cell `[0,h]`에서

$$
X_0=\int_0^h dW_s,\quad
X_1=\int_0^h(h-s)^\alpha dW_s,\quad
X_2=\int_0^h(2h-s)^\alpha dW_s
$$

를 joint Gaussian으로 정확히 샘플한다. 두 번째 fine cell `[h,2h]`에서는

$$
Y_0=\int_h^{2h}dW_s,\quad
Y_1=\int_h^{2h}(2h-s)^\alpha dW_s
$$

를 샘플한다. 두 cell은 독립이다. 그러면 coarse recent-cell pair는

$$
\Delta W^c=X_0+Y_0,
\qquad I^c=X_2+Y_1
$$

이고 정확히

$$
\operatorname{Var}(\Delta W^c)=2h,
$$

$$
\operatorname{Cov}(\Delta W^c,I^c)
=\frac{(2h)^{\alpha+1}}{\alpha+1},
$$

$$
\operatorname{Var}(I^c)
=\frac{(2h)^{2\alpha+1}}{2\alpha+1}
$$

를 만족한다. Fine recent-cell marginal도 기존 BLP와 동일하다.

단순히 fine local integrals를 더하면 첫 cell의 kernel endpoint가 달라지므로 이
공분산을 얻지 못한다. 해당 단순 합산은 금지한다.

## 4. Controlled coupled law

Fine cell `i`에서 bounded pre-increment control `u_i`를 사용한다.

$$
\Delta W_i^P=\Delta W_i^Q+u_i h.
$$

동일 Brownian Cameron--Martin shift는 모든 local kernel integral에

$$
u_i\int_{t_i}^{t_{i+1}}K(T-s)ds
$$

만큼의 mean shift를 준다. Brownian-bridge residual은 바뀌지 않는다. 따라서 shared
fine innovation space의 likelihood는

$$
L_\ell=\frac{dP}{dQ_\ell}
=\exp\left(
-\sum_{i=0}^{N_\ell-1}u_i\cdot\Delta W_i^Q
-\frac12\sum_{i=0}^{N_\ell-1}\|u_i\|^2h_\ell
\right).
$$

Coarse path는 shifted fine target coordinates를 aggregate하여 구성한다. Coarse에
별도 likelihood를 곱하지 않는다.

## 5. Exact finite-grid telescoping propositions

### Proposition V7-1: adjacent marginal exactness

Section 3의 joint Gaussian construction에서 fine marginal은 `P_ell`, coarse marginal은
`P_{ell-1}`의 implemented BLP law와 각각 동일하다.

### Proposition V7-2: controlled correction identity

bounded adapted fine control 아래

$$
Y_0=H_0L_0,
$$

$$
Y_\ell=(H_\ell-H_{\ell-1}^{coupled})L_\ell,
\qquad \ell\ge1
$$

이면

$$
E_{Q_0}[Y_0]=E_{P_0}[H_0],
$$

$$
E_{Q_\ell}[Y_\ell]
=E_P[H_\ell-H_{\ell-1}],
$$

따라서

$$
\sum_{\ell=0}^{L}E[Y_\ell]=E_{P_L}[H_L].
$$

이 명제는 finite-grid algebraic identity이며 continuous limit theorem이 아니다.

### Proposition V7-3: exact defensive mixture

$$
Q_{mix,\ell}=a_0P+(1-a_0)Q_{u,\ell},\qquad a_0>0
$$

에서 canonical fine target increments에 대해 모든 component density를 replay하면

$$
L_{mix,\ell}
=\left(a_0+(1-a_0)\frac{dQ_{u,\ell}}{dP}\right)^{-1}
$$

이고 `Y_ell L_mix` correction은 비편향이다. Final estimator는 self-normalize하지
않는다.

## 6. Correction-focused CEM

Level correction은 signed이지만 second moment의 support는 disagreement event

$$
D_\ell=|H_\ell-H_{\ell-1}|\in\{0,1\}
$$

이다. CEM은 `P(. | D_ell=1)`의 Gaussian mean-shift projection을 근사한다.
Segment `j`의 weighted MLE는

$$
u_{\ell,j}^{MLE}
=\frac{E_{Q}[D_\ell L
\sum_{i\in I_j}\Delta W_i^P]}
{|I_j|h_\ell E_Q[D_\ell L]}.
$$

최종 correction에는 `H_ell-H_{ell-1}`의 원래 부호를 유지한다. Disagreement sample이
부족하면 임의 soft proxy로 성공을 선언하지 않고 gate를 실패시킨다.

## 7. MLMC allocation and work

독립 level estimator의 single-path variance와 cost를 `V_ell,C_ell`이라 하면 variance
budget `epsilon_v^2`에 대한 표준 allocation은

$$
N_\ell
=\left\lceil
\epsilon_v^{-2}
\sqrt{\frac{V_\ell}{C_\ell}}
\sum_{k=0}^{L}\sqrt{V_kC_k}
\right\rceil.
$$

Predicted online work는

$$
W_{ML}=\epsilon_v^{-2}
\left(\sum_\ell\sqrt{V_\ell C_\ell}\right)^2
$$

이다. Training, pilot, allocation estimation 비용은 별도로 기록하고 repeated-query
break-even에 포함한다.

## 8. Hard-event theorem boundary

Hard barrier/occupation functional은 Lipschitz가 아니다. 연속극한 또는 rate theorem을
위해서는 최소한 다음이 필요하다.

1. barrier boundary anti-concentration;
2. occupation threshold 근처의 density 또는 crossing control;
3. coupled BLP strong/weak error;
4. likelihood의 uniform moment bound;
5. control energy의 level-uniform bound.

이번 구현은 위 가정을 자동으로 참이라고 선언하지 않는다. 먼저 finite levels에서
bias, disagreement probability, correction variance rate를 추정한다. Rate fit은 theorem이
아니라 assumption diagnostic으로 명시한다.

## 9. Implementation milestones

### M0: Gaussian coupling law

- triple-cell covariance positive definiteness;
- fine/coarse BLP local marginal covariance;
- null-controlled fine marginal equals standalone simulator statistically;
- null-controlled coarse marginal equals standalone simulator statistically.

### M1: controlled identity

- pathwise target = proposal + all kernel mean shifts;
- likelihood reconstruction;
- `E_Q[L]=1` diagnostic;
- bounded payoff and correction agreement with natural coupling.

### M2: telescoping

- natural adjacent corrections telescope to finest direct estimate;
- controlled corrections telescope within combined standard error;
- mixture and selected-component replay agree.

### M3: feasibility

- grids `N in {16,32,64,128}`;
- natural, shared event-CEM, correction-CEM mixture;
- five development seeds;
- variance, disagreement, cost, predicted work, break-even.

## 10. Stop gates

### G7-0: exact coupling

모든 covariance, marginal law, likelihood, telescoping test가 통과하지 않으면 성능실험을
하지 않는다.

### G7-1: multilevel feasibility

다음을 모두 요구한다.

- correction estimator difference `|z|<=3` versus natural coupling;
- controlled correction variance improvement on at least 2 of 3 nonzero levels;
- five-seed geometric predicted-work ratio versus finest single-level piecewise CEM
  `>1.25`;
- improving work seeds at least `4/5`;
- training-inclusive break-even `<=50` repeated queries;
- no likelihood normalization/replay failure.

실패 시 threshold, event, seed 또는 level set을 바꾸어 같은 claim을 재시험하지 않는다.

### G7-2: theorem development, conditional

G7-1 통과 시에만 boundary assumptions와 variance-rate proof를 추진한다. 통과 전에는
저명 저널 수준 theorem claim을 작성하지 않는다.

## 11. Non-negotiable technical rules

1. Fine/coarse correction pair에 하나의 fine-space likelihood만 사용한다.
2. BLP coarse recent integral을 fine local integral의 단순 합으로 만들지 않는다.
3. Coarse Brownian increments는 shifted fine target increments에서 aggregate한다.
4. Control은 current fine increment를 보기 전에 계산한다.
5. Final estimator에 self-normalization을 사용하지 않는다.
6. Signed correction의 부호를 학습 target 때문에 제거하지 않는다.
7. Online timing에서 augmented recording 조건을 모든 method에 동일하게 한다.
8. 새로운 seed split을 사용하고 Plan v6 untouched seeds를 재사용하지 않는다.
