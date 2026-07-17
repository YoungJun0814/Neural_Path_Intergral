# Boundary-Aware Conditional Volterra Bridge: 실행계획 v8

Date: 2026-07-17

Status: implemented; finite-grid exactness retained; G8 total-work claim falsified
Predecessor: `CONTROLLED_MULTILEVEL_VOLTERRA_PLAN_V7.md`

## 0. 연구 결정

G7은 exact adjacent BLP coupling, one-fine-space likelihood, correction-CEM의 수학적
정확성을 통과했지만 hard hit-plus-occupation correction의 자연분산이 격자 세분화와 함께
감소하지 않아 total-work gate를 실패했다. 따라서 controller의 차원이나 network 크기를
늘리지 않는다. 먼저 불연속 payoff가 만드는 fine/coarse disagreement를 조건부 평균으로
직접 제거하는 새로운 연산을 연구한다.

주모델 이름은 임시로 다음과 같이 고정한다.

> **CICVB: Coarse-Innovation-Conditioned Volterra Bridge branching**

연구 질문은 다음이다.

> BLP coarse innovation을 고정하고 그와 양립하는 fine Volterra bridge만 조건부로
> 재표본화하면, rough-volatility hard path correction의 분산과 end-to-end work를
> 유의미하게 줄일 수 있는가?

## 1. 선행연구 및 novelty 경계

- BLP hybrid scheme은 singular recent-cell Wiener integral과 historical step kernel을
  결합한다: <https://arxiv.org/abs/1507.03004>.
- Markov SDE digital option의 repeated path splitting/branching은 이미 존재한다:
  <https://arxiv.org/abs/2209.03017>.
- smooth test function에 대한 rough-Volterra PPDE/weak rate도 이미 연구됐다:
  <https://arxiv.org/abs/2304.03042>.

따라서 다음은 단독 신규성이 아니다.

1. 일반적인 conditional Monte Carlo 또는 Rao--Blackwellization;
2. 일반적인 path splitting;
3. BLP scheme 자체;
4. Girsanov importance sampling 또는 CEM 자체.

이번 연구의 잠재적 신규성은 다음 교집합에만 있다.

1. BLP local singular integrals까지 포함한 exact coarse-conditioned Gaussian bridge;
2. rough-Volterra hard path functional의 signed adjacent correction;
3. branch별 exact fine-space likelihood를 사용하는 controlled conditional estimator;
4. correction variance와 total-work에 대한 finite-grid 정리 및 경험적 rate;
5. 이후 boundary-conditioned Volterra operator로의 amortization.

`first` 또는 `unique`라는 표현은 별도 systematic review 전에는 사용하지 않는다.

## 2. Exact conditional Gaussian bridge

Fine step을 `h`, coarse step을 `2h`, `alpha=H-1/2`라 둔다. 한 coarse block의 첫
fine cell에서

$$
X_0=\int_0^h dW_s,\quad
X_1=\int_0^h(h-s)^\alpha dW_s,\quad
X_2=\int_0^h(2h-s)^\alpha dW_s
$$

를 사용하고 두 번째 fine cell에서

$$
Y_0=\int_h^{2h}dW_s,\quad
Y_1=\int_h^{2h}(2h-s)^\alpha dW_s
$$

를 사용한다. Fine innovation vector와 coarse innovation은

$$
F=(X_0,X_1,X_2,Y_0,Y_1)^\top,
$$

$$
C=AF=(X_0+Y_0,X_2+Y_1)^\top
$$

이다. `Sigma_F=Cov(F)`와 `Sigma_C=A Sigma_F A^T`라 두고

$$
K=\Sigma_F A^\top\Sigma_C^{-1}
$$

를 정의한다. 독립인 `F' ~ N(0,Sigma_F)`에서

$$
R=F'-KAF'
$$

를 만들고 `C ~ N(0,Sigma_C)`를 별도로 생성하면

$$
F^{(m)}=KC+R^{(m)}
$$

은 다음을 동시에 만족한다.

1. `AF^(m)=C` pathwise;
2. `F^(m)|C`는 정확한 conditional Gaussian law;
3. 서로 다른 branch의 residual은 `C` 조건부 독립;
4. 각 branch의 unconditional fine marginal은 기존 BLP fine law;
5. 공유 coarse marginal은 기존 BLP coarse law.

두 번째 독립 Brownian driver에는 동일하게 fine pair를 coarse sum에 조건화한 Gaussian
bridge를 사용한다.

## 3. Deterministic controlled law

첫 구현은 현재 practical winner인 deterministic time-piecewise two-driver control만
허용한다. Feedback control은 coarse target이 branch마다 달라질 수 있으므로 G8-0에
포함하지 않는다.

Fine controls를 `u_0,u_1`이라 하면 target fine increments는

$$
\Delta W_i^P=\Delta W_i^Q+u_i h
$$

이고 coarse target Brownian increment는

$$
\Delta W_c^P=C_0+(u_0+u_1)h.
$$

Coarse target local integral은

$$
I_c^P=C_1
+u_{0,1}\int_0^h(2h-s)^\alpha ds
+u_{1,1}\int_h^{2h}(2h-s)^\alpha ds.
$$

Branch `m`의 likelihood는 전체 fine proposal increments에서

$$
L^{(m)}=\exp\left(
-\sum_i u_i\cdot\Delta W_i^{Q,(m)}
-\frac12\sum_i\|u_i\|^2h
\right)
$$

로 계산한다. Coarse likelihood를 별도로 곱하지 않는다.

## 4. Branched correction estimator

공유 coarse innovation sigma-field를 `C_ell`이라 하고 branch 수를 `M`이라 둔다.
Branch별 signed contribution은

$$
Y_\ell^{(m)}=
(H_\ell^{(m)}-H_{\ell-1}^{c})L_\ell^{(m)}
$$

이고 parent contribution은

$$
\bar Y_{\ell,M}=\frac1M\sum_{m=1}^M Y_\ell^{(m)}
$$

이다.

### Proposition V8-1: finite-grid unbiasedness

각 branch marginal이 선언된 fine proposal이고 likelihood가 exact하면

$$
E_Q[\bar Y_{\ell,M}]
=E_P[H_\ell-H_{\ell-1}]
$$

이다. Branch 간 독립성은 unbiasedness에 필요하지 않지만 조건부 분산식에 필요하다.

### Proposition V8-2: conditional variance decomposition

`m(C)=E[Y^(1)|C]`, `v(C)=Var(Y^(1)|C)`라 두면

$$
Var(\bar Y_M)=Var(m(C))+\frac1M E[v(C)].
$$

따라서 variance는 `M`에 대해 단조 비증가한다. 그러나 irreducible term
`Var(m(C))`가 크면 work-normalized efficiency는 개선되지 않을 수 있다.

### Proposition V8-3: coarse-adaptive branching

Branch count `M(C)>=1`이 fine residual을 보기 전에 coarse path만으로 결정되면

$$
\bar Y_{M(C)}=
\frac1{M(C)}\sum_{m=1}^{M(C)}Y^{(m)}
$$

도 unbiased하고

$$
Var(\bar Y_{M(C)})
=Var(m(C))+E\left[\frac{v(C)}{M(C)}\right].
$$

Fine branch 결과를 본 뒤 branch count를 바꾸는 optional stopping 형태는 금지한다.

## 5. Boundary-aware allocation

초기 adaptive policy는 학습 모델이 아니라 coarse event margins만 사용한다.

$$
d_{hit}=\frac{|\min S^c-B_{hit}|}{s_{hit}},
\qquad
d_{occ}=\frac{|A^c-\tau_{occ}|}{s_{occ}}.
$$

Development seed에서 `v(C)`와 margin의 관계를 추정하고, sealed validation 전에
branching band와 branch count를 동결한다. Policy는 반드시 coarse path measurable이어야
한다.

후속 operator는 gate 통과 후에만

$$
(K_H,\zeta,\text{coarse path state})\mapsto
(\widehat v(C),M(C),u(C))
$$

를 amortize한다.

## 6. Cost contract

Parent 한 개의 비용을

$$
C_M=C_{trunk}+M C_{refine}
$$

로 분리 기록한다. Vectorization으로 wall-clock scaling이 sublinear일 수 있으므로 다음
두 비용을 모두 보고한다.

1. measured wall-clock cost per parent;
2. innovation/step work proxy proportional to `coarse_steps + M*fine_steps`.

Primary efficiency는 measured variance-times-wall-clock cost다. GPU batch saturation만으로
얻은 이점을 theorem complexity로 주장하지 않는다.

## 7. Milestones

### M0: Gaussian bridge oracle

- conditional projection identity `AF=C`;
- conditional mean/covariance Monte Carlo;
- branch residual conditional independence;
- fine and coarse unconditional BLP marginal agreement.

### M1: rBergomi path law

- `M=1` branched sampler와 G7 adjacent sampler의 law agreement;
- natural hard correction agreement;
- deterministic-control likelihood reconstruction;
- controlled hard correction agreement;
- variance decomposition numerical audit.

### M2: fixed branching

- levels `32,64,128`;
- branches `1,2,4,8,16`;
- natural and frozen G0 piecewise-CEM;
- variance, irreducible fraction, measured cost, work ratio.

### M3: boundary-adaptive branching

- development seed에서 coarse margin calibration;
- policy freeze;
- untouched five-seed validation;
- single-level CEM 및 G7 correction-CEM과 total-work 비교.

## 8. Stop gates

### G8-0: exactness

다음 중 하나라도 실패하면 성능실험을 금지한다.

- conditional constraint maximum error `<=1e-11`;
- marginal covariance relative error `<=5%`;
- natural/controlled estimator difference `|z|<=3`;
- direct likelihood reconstruction maximum error `<=1e-10`;
- branch variance가 이론적 분산식과 통계 오차 내 일치;
- `M` 증가 시 variance 비증가가 통계 오차 내 성립.

### G8-1: fixed-branch feasibility

- nonzero level 중 최소 2/3에서 best fixed `M`의 work ratio versus `M=1` `>1.10`;
- correction estimate consistency `|z|<=3`;
- likelihood normalization/replay 통과;
- irreducible conditional variance fraction이 모든 level에서 `>=0.9`이면 adaptive stage를
  중단한다.

### G8-2: adaptive branching

- five-seed geometric total-work ratio versus finest single-level G0 CEM `>1.25`;
- improving seeds `>=4/5`;
- training/calibration-inclusive break-even `<=50` queries;
- natural correction variance의 empirical positive decay 또는 명시적 rate theorem 후보;
- 실패 시 VFO, attention, quantum layer를 추가하지 않고 방향을 중단한다.

## 9. Claim boundary

G8-2 전에는 다음만 주장한다.

- exact finite-grid conditional BLP bridge;
- unbiased finite-grid branched correction;
- finite-grid variance decomposition.

다음은 주장하지 않는다.

- continuous-monitoring unbiasedness;
- rough hard payoff의 positive convergence rate;
- Markov digital branching 결과의 자동적인 Volterra 일반화;
- top-journal novelty 또는 최초성;
- neural/quantum advantage.

## 10. Execution decision — 2026-07-17

G8 was implemented and audited end to end. The exact finite-grid conditional
bridge, branchwise likelihood, and coarse-measurable variable allocation passed
their mathematical checks. Adaptive branching improved adjacent-correction
variance-times-cost by about `1.94x`, but the complete estimator achieved only
`0.299x` of the finest-level CEM efficiency, improved only `1/5` validation
seeds, and had correction-variance log-slope `-0.372`.

Decision: retain the exact bridge component, falsify the G8 total-work claim,
and do not add VFO/attention/quantum complexity to this architecture. Full
evidence and the post-hoc likelihood-gate correction are recorded in
`docs/phase_reviews/G8_VOLTERRA_BRIDGE_BRANCHING_FALSIFICATION_2026-07-17.md`.
