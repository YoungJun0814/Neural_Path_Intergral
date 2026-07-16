# Spectral Doob--Volterra Path-Integral Sampler: 실행계획 v6

Date: 2026-07-16

Status: theory-audited, pre-calibration gated protocol

## 0. Executive decision

Plan v6는 terminal two-tail neural search를 재개하지 않는다. 새로운 scientific
question은 다음과 같다.

> rBergomi의 discretely monitored path-dependent downside-excursion event에서,
> multi-scale Volterra lift와 positive conditional desirability moment regression으로
> 얻은 causal Gaussian control이 strong time-piecewise CEM보다 exact-estimator total
> work를 줄이는가?

가칭 모델은 **Spectral Doob--Volterra Path-Integral Sampler (SDV-PIS)**다. 여기서
`spectral`은 Volterra kernel을 positive exponential modes로 lift하는 것을 뜻한다.
Stochastic Koopman eigenfunction으로 Doob transform을 근사하는 일반 아이디어는 이미
존재하므로 spectral/Doob 자체를 신규성으로 주장하지 않는다.

이번 실행은 다음 세 결과 중 하나로 종료한다.

1. G0 실패: 선택한 task에서 time-dependent control 필요성이 없음;
2. G0 통과, G3 실패: path dependence는 있으나 SDV-PIS가 strong baseline을 못 이김;
3. G3 통과: sealed confirmatory와 논문 theorem package로 진행 가능.

Gate 실패 후 width, seed, threshold, objective를 바꾸어 같은 claim을 반복하지 않는다.

## 1. 선행연구 경계

### 이미 존재하는 결과

- neural Cameron--Martin drift를 이용한 option-pricing importance sampling과 optimal
  drift approximation: Arandjelovic, Rheinlander, Shevchenko, *Finance and
  Stochastics* 2025,
  <https://link.springer.com/article/10.1007/s00780-024-00549-x>;
- 과거 경로를 입력받는 sequence neural Girsanov importance sampling:
  <https://arxiv.org/abs/2007.02692>;
- stochastic Koopman eigenfunction으로 Doob transform을 근사하는 rare-event
  simulation: <https://arxiv.org/abs/2101.07330>;
- Schrödinger bridge, path-integral control, neural control과 Girsanov correction을
  결합한 sampler: <https://arxiv.org/abs/2111.15141>;
- rough-volatility PPDE의 random/reservoir neural solver:
  <https://arxiv.org/abs/2305.01035>;
- stochastic Volterra conditional expectation의 PPDE characterization 및 weak-rate
  analysis: <https://arxiv.org/abs/2304.03042>;
- rBergomi의 Markovian approximation:
  <https://arxiv.org/abs/2007.02113>.

### 따라서 주장하지 않는 것

- Doob transform, Feynman--Kac, spectral basis 또는 neural IS 자체의 신규성;
- path history를 network에 넣는 것 자체의 신규성;
- SOE/Markovian rough-volatility approximation 자체의 신규성;
- imaginary-time analogy가 실제 quantum algorithm이라는 주장;
- generic neural universal approximation theorem;
- continuous-time zero-variance control을 현재 network가 정확히 구현한다는 주장.

### 조건부 신규성 후보

다음 패키지가 함께 성공할 때만 기여 후보가 된다.

1. exact BLP target law를 변경하지 않는 causal Volterra spectral lift;
2. path-functional conditional Doob transition의 two-driver Gaussian KL projection;
3. positive desirability와 Brownian moment를 함께 학습하는 regression;
4. strong constant/time-piecewise CEM와 exact matched-work 비교;
5. all-path Girsanov/marginal likelihood와 finite-grid unbiased estimator;
6. physical kernel weights와 learned decision weights의 분리;
7. task와 grid에 따른 spectral-rank/error/work ablation.

## 2. Target finite-grid law

시간격자 `t_i=i*Delta t`, `i=0,...,N`에서 현재 검증된 controlled BLP rBergomi
target law를 그대로 사용한다.

$$
V_i=\xi\exp\left(\eta Y_i-\frac12\eta^2\operatorname{Var}(Y_i)\right),
$$

$$
\Delta B_i=\rho\Delta W_i^1+\sqrt{1-\rho^2}\Delta W_i^2,
$$

$$
\log S_{i+1}=\log S_i-\frac12V_i\Delta t+\sqrt{V_i}\Delta B_i.
$$

`Y_i`는 BLP `kappa=1` hybrid law로 계산한다. Controller의 SOE state는 feature일
뿐 target `Y_i`를 대체하지 않는다.

Proposal convention은 기존 저장소와 동일하다.

$$
\Delta W_i^P=\Delta W_i^Q+u_i\Delta t,
$$

$$
\log\frac{dP}{dQ_u}
=-\sum_i u_i\cdot\Delta W_i^Q
-\frac12\sum_i\lVert u_i\rVert^2\Delta t.
$$

Control은 `Delta W_i^Q`를 생성하기 전에 `F_i`에서 평가한다.

## 3. Primary path-functional task

Primary event는 discretely monitored downside excursion이다.

$$
A=\left\{
\min_{0\le i\le N}S_i\le B_{hit},\quad
\Delta t\sum_{i=1}^N1_{\{S_i\le B_{stress}\}}\ge\tau_{occ}
\right\},
$$

where `B_hit < B_stress < S0`.

이는 단순 terminal event와 달리 barrier hit 여부와 stress 상태 체류시간을 모두
요구한다. Monitoring convention은 right endpoint `S_1,...,S_N`다. `S_0`는
occupation count에 포함하지 않는다.

### Training payoff

Hard indicator는 evaluation에만 사용한다. Training에는

$$
G_{soft}
=\sigma\left(\frac{B_{hit}-M_N}{s_{hit}}\right)
 \sigma\left(\frac{O_N-\tau_{occ}}{s_{occ}}\right),
$$

$$
M_N=\min_iS_i,
\qquad
O_N=\Delta t\sum_{i=1}^N1_{\{S_i\le B_{stress}\}}
$$

를 사용한다. Soft payoff는 proposal training용이며 최종 hard estimator를 대체하지
않는다.

### 단 한 번 허용하는 calibration

별도 seed와 natural paths로 `(B_hit,B_stress,tau_occ)`를 선택해 hard probability를
대략 `1e-3`--`1e-2` 범위로 만든다. Calibration 이후 숫자, train seed,
development seed, sealed seed를 config에 고정한다. Development 결과를 본 뒤
threshold를 변경하지 않는다.

## 4. Exact finite-grid theory

### Proposition V6-1: adapted bounded control identity

각 `u_i`가 `F_i`-measurable이고 `||u_i||<=U`이면 finite Gaussian product law에서
`P`와 `Q_u`는 equivalent이고 위 discrete likelihood가 정확하다.

이는 구현된 target Brownian coordinates에 대한 유한차원 Gaussian mean shift다.
BLP local singular-cell auxiliary Gaussian의 conditional bridge residual은 proposal과
target에서 동일하며, first-driver mean shift만 deterministic local integral shift로
전달된다. 별도 likelihood term을 중복 추가하지 않는다.

### Proposition V6-2: path-functional estimator unbiasedness

Hard event가 전체 recorded path의 measurable function이면

$$
E_{Q_u}\left[1_A\frac{dP}{dQ_u}\right]=P(A).
$$

Controller 또는 desirability approximation의 정확도는 이 identity에 영향을 주지
않고 variance만 바꾼다.

### Proposition V6-3: randomized mixture identity

고정 prior `alpha_k>0`와 adapted proposal `Q_k`에 대해

$$
Q_{mix}=\sum_k\alpha_kQ_k,
$$

$$
\frac{dQ_{mix}}{dP}(X)
=\sum_k\alpha_k\frac{dQ_k}{dP}(X).
$$

따라서 all-expert balance likelihood estimator가 unbiased다. Retained label을 쓰는
component estimator도 unbiased지만 primary estimator는 variance 안정성을 위해
balance form을 사용한다.

### Proposition V6-4: one-step Gaussian Doob projection

`0<G<=1`인 soft payoff에 대해

$$
h_i=E_P[G\mid\mathcal F_i],
\qquad
m_i=E_P[G\Delta W_i^P\mid\mathcal F_i].
$$

Exact soft Doob transition을 covariance `Delta t I`인 Gaussian mean-shift family에
conditional information projection, 즉 `KL(Q*_i || Q_{u,i})` 최소화하면

$$
u_i^{proj}=\frac{m_i}{\Delta t\,h_i}
$$

가 된다. 이는 전체 zero-variance Doob transition과 동일하다는 뜻이 아니라 Gaussian
mean-shift family 안의 conditional moment projection이다.

### Proof boundary

V6-1--V6-4는 finite-grid proposition으로 문서화하고 Gaussian oracle과 numerical
identity test를 제공한다. 다음은 이번 구현에서 theorem으로 주장하지 않는다.

- hard-event continuous-time Doob drift의 존재/regularity;
- discontinuous barrier monitoring bias의 universal rate;
- learned spectral approximation의 nonasymptotic relative-variance bound;
- `dt -> 0`에서 learned controller parameter의 수렴;
- global optimality of CEM or SDV-PIS.

이 항목들은 논문용 추가 theorem target이지 구현 완료 주장에 포함하지 않는다.

## 5. Volterra spectral lift

Controller feature로만 다음 positive sum-of-exponentials state를 사용한다.

$$
Z_{i+1}^{(m)}
=e^{-\lambda_m\Delta t}Z_i^{(m)}
+\frac{1-e^{-\lambda_m\Delta t}}{\lambda_m\Delta t}\Delta W_i^{1,P}.
$$

Kernel feature는

$$
K(r)\approx\sum_{m=1}^Ma_me^{-\lambda_mr},
\qquad a_m\ge0,
$$

으로 fit한다.

- `lambda_m,a_m`: physical kernel approximation; fixed;
- network projection/gate: decision relevance; learned;
- BLP target Volterra path: exact implemented grid law; unchanged.

Rank `M`은 primary에서 8로 고정하고 ablation `M in {2,4,8,16}`을 별도 seed에서
수행한다. Rank별 kernel error, inference cost, hard variance를 함께 보고한다.

## 6. Causal augmented state

Time `i`의 feature는 다음 정보만 사용한다.

$$
X_i^{aug}=(
t_i/T,
\log(S_i/S_0),
\log(V_i/\xi),
Y_i,
\log(M_i/S_0),
O_i/T,
1_{\{M_i\le B_{hit}\}},
Z_i^{(1:M)}
).
$$

여기서

$$
M_i=\min_{0\le j\le i}S_j,
\qquad
O_i=\Delta t\sum_{j=1}^i1_{\{S_j\le B_{stress}\}}.
$$

`S_{i+1}` 또는 `Delta W_i`는 `u_i` 계산에 사용하지 않는다. Replay는 same target
path에서 state를 처음부터 causal하게 재구성한다.

## 7. SDV-PIS parameterization

Model은 positive desirability와 projected control을 출력한다.

$$
\widehat h_\theta(X_i^{aug})
=\epsilon_h+(1-\epsilon_h)\sigma(f_\theta(X_i^{aug})),
$$

$$
\widehat u_\theta(X_i^{aug})
=U\tanh\left(b_{piece}(t_i)+r_\theta(X_i^{aug})\right).
$$

`b_piece`는 strong time-piecewise CEM을 exactly represent하는 inverse-tanh anchor다.
Residual output layer는 zero initialize하므로 initialization에서 control이 baseline과
pathwise identical하다.

Desirability head는 proposal density에 사용하지 않는다. Final likelihood에는 실제
bounded total control만 사용한다.

## 8. Training law and objectives

Training behavior는 defensive exact mixture다.

$$
Q_{beh}=\alpha_0P+(1-\alpha_0)Q_{piece},
\qquad \alpha_0\ge0.1.
$$

All-expert marginal likelihood `L_beh=dP/dQ_beh`를 계산한다. Training에서만
self-normalized regression weights를 허용한다.

### Desirability regression

$$
\mathcal L_h
=E_{Q_{beh}}\left[
L_{beh}\sum_i(\widehat h_i-G_{soft})^2
\right].
$$

### Brownian moment regression

수치 conditioning을 위해 `Delta W/sqrt(Delta t)`를 사용한다.

$$
\mathcal L_m
=E_{Q_{beh}}\left[
L_{beh}\sum_i
\left\|
\sqrt{\Delta t}\widehat h_i\widehat u_i
-G_{soft}\frac{\Delta W_i^P}{\sqrt{\Delta t}}
\right\|^2
\right].
$$

### Baseline-preserving penalty

Low-desirability state에서 noisy moment target이 control을 파괴하지 않도록

$$
\mathcal L_{anchor}
=E[(1-G_{soft})\|u_i-u_i^{piece}\|^2]
$$

를 사용한다. Coefficients는 config에 사전 고정한다. Model selection endpoint는
training loss가 아니라 independent hard-event second moment와 total work다.

## 9. Strong baselines

모든 baseline은 same two target Brownian coordinates와 exact likelihood를 사용한다.

1. natural MC;
2. constant two-driver CEM;
3. time-piecewise two-driver CEM, primary 4 segments;
4. SDV-PIS initialized at the piecewise CEM;
5. ablation without SOE modes;
6. ablation without path-functional states;
7. optional defensive mixture, same mixture overhead for compared proposals.

Piecewise CEM의 segment control은 weighted Gaussian MLE다.

$$
u_{j}^{MLE}
=\frac{E_w[\sum_{i\in I_j}\Delta W_i^P]}
{|I_j|\Delta t}.
$$

## 10. Implementation milestones

### M0: analytic/discrete Gaussian oracle

- 1D Gaussian random walk with barrier-hit plus occupation event;
- backward Gauss--Hermite dynamic programming for `h_i`;
- exact projected drift `m_i/(dt*h_i)`;
- natural/proposal likelihood normalization;
- unbiased probability agreement;
- projected proposal second moment below a constant tilt.

Failure means sign, time index 또는 projection theory 오류이므로 rBergomi model을
진행하지 않는다.

### M1: path-functional engine

- hard event, soft payoff, score, occupation convention;
- prefix feature reconstruction;
- suffix perturbation cannot change earlier state/control;
- simulator and replay state identity.

### M2: strong baseline gate G0

- separate task calibration;
- constant and 4-segment CEM;
- 5 development seeds, same paths and timing protocol;
- path-dependence necessity test.

### M3: SDV-PIS development

- SOE spectral states and kernel error;
- exact baseline initialization;
- defensive-mixture training data;
- positive h and moment regression;
- 5 independent development seeds.

### M4: sealed confirmatory, conditional

G3 통과 시에만 20 untouched seeds에서 natural, constant CEM, piecewise CEM,
SDV-PIS를 평가한다. 실패한 development model에 confirmatory compute를 사용하지
않는다.

## 11. Correctness gates

### G1: Gaussian oracle

모두 통과해야 한다.

- DP probability versus high-path natural MC: `|z|<=3`;
- likelihood mean: `|z|<=3` from one;
- weighted probability: `|z|<=3` from DP reference;
- projection formula versus numerical conditional KL optimum;
- replay error `<=1e-10`;
- projected drift improves second moment over zero/constant reference.

### G2: rBergomi exactness

- zero residual equals piecewise CEM pathwise at initialization;
- simulator/replay controls pathwise equal;
- selected likelihood replay error `<=1e-10`;
- natural likelihood normalization;
- both hard and soft payoff finite;
- suffix perturbation adaptedness test;
- SOE feature does not alter target spot/variance law at zero residual;
- all controls finite and bounded.

## 12. Falsification gates

### G0: task requires temporal/path control

Before SDV training, piecewise CEM must beat constant CEM on development seeds.

- geometric raw-variance VRF `>1.25`;
- geometric online-work VRF `>1.10`;
- improvement on at least 4/5 seeds;
- no probability inconsistency `|z|>3` versus natural reference;
- piecewise training-inclusive break-even `<=50` queries.

If G0 fails, stop Plan v6 at the baseline result. Threshold redesign is not allowed after
the one calibration.

### G3: SDV-PIS development

SDV-PIS must beat strong piecewise CEM.

- geometric raw-variance VRF `>1.35`;
- geometric online-work VRF `>1.25`;
- improvement on at least 4/5 seeds;
- incremental training break-even `<=50` queries;
- maximum replay error `<=1e-10`;
- hard probability difference `|z|<=3`;
- contribution ESS not lower than 80% of piecewise CEM;
- both no-SOE and no-path-state ablations are not jointly equivalent to full model.

Any failure stops the positive model claim. One ablation outperforming the full model means
the corresponding claimed mechanism is removed from the paper.

### G4: publication confirmatory

Conditional on G3 passage:

- 20 sealed seeds;
- geometric online-work VRF `>1.50` versus piecewise CEM;
- improvement on at least 16/20 seeds;
- paired log-work 95% CI entirely above zero;
- `dt in {1/32,1/64,1/128}` conclusion consistency;
- at least one parameter-shift task without retraining;
- training-inclusive break-even reported;
- no evaluation seed used for selection.

## 13. Statistical protocol

- Primary unit: validation seed, not individual path;
- Primary endpoint: `log(work_piece/work_candidate)`;
- `work = single_path_variance * seconds_per_path`;
- timing: warm-up followed by 3 or more medians;
- probability SE: ordinary, non-self-normalized contribution SE;
- cross-seed mean SE: `sqrt(sum_s SE_s^2)/number_of_seeds`;
- report arithmetic and geometric summaries;
- report contribution ESS, top-weight concentration, event fraction, control RMS;
- training time excluded from online work but included in break-even;
- all smoke/development/confirmatory results labeled separately.

## 14. Technical failure modes and controls

| Failure | Prevention |
|---|---|
| future information in occupation/hit state | state updated only through `S_i` before control `u_i` |
| BLP law replaced by SOE | SOE is feature-only; simulator continues exact implemented BLP grid |
| local bridge likelihood double counted | likelihood only on two Brownian mean-shift coordinates |
| hard-indicator pathwise gradient | no pathwise indicator gradient; regression uses recorded target increments |
| self-normalized final estimator | self-normalization restricted to training regression |
| approximate h biases estimate | final estimator uses exact control likelihood, not h as a weight |
| mixture label likelihood error | all-expert marginal `logsumexp` replay |
| spectral modes use future increments | SOE state updated after current control evaluation |
| baseline weakened | 4-segment two-driver CEM fitted before neural training |
| post-hoc task tuning | one calibration seed, then immutable config hash |
| complex quantum weights | h is real positive; no amplitude interference claim |
| timing artifact | repeated warm-up/median and same dtype/device |
| false continuous-time claim | primary claim explicitly finite-grid/discrete-monitoring |

## 15. Publication decision

### Development failure

보존 가능한 결과:

- Gaussian path-functional Doob projection oracle;
- exact path-functional rBergomi likelihood engine;
- constant versus piecewise CEM falsification;
- negative SDV-PIS benchmark.

이 결과만으로 positive-model 저명 저널 투고를 주장하지 않는다.

### G4 success

논문 핵심은 “quantum-inspired neural network”가 아니라 다음으로 기술한다.

> Exact finite-grid importance sampling for non-Markovian Volterra path functionals
> using a causal positive spectral lift and projected conditional Doob moments.

Target venue는 theorem 강도에 따라 *Finance and Stochastics* 또는 *SIAM Journal
on Financial Mathematics*, computational theorem이 중심이면 *Journal of
Computational Physics* 또는 *SIAM Journal on Scientific Computing*을 검토한다.

## 16. Reproducibility contract

- 모든 config에 schema, protocol id, seed split, frozen flag;
- 결과 JSON에 config SHA-256;
- checkpoint에 architecture와 state SHA-256;
- CPU float64 primary correctness path;
- Ruff, Mypy, Pytest green;
- calibration, development, confirmatory artifact 분리;
- 결과를 본 뒤 gate나 endpoint 변경 금지.
