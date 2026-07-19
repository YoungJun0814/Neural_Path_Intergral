# Correction-Focused DCS-MGI-MLMC Research and Implementation Plan V11

Date: 2026-07-19
Status: proposed; implementation must follow the gates in this document
Primary objective: a defensible numerical-probability paper, not a post-hoc rescue of G10

## 0. Executive decision

The next model will be **DCS-MGI-MLMC**:

> Defensive Control-Span Marginalized Gaussian Integration embedded in a complete
> multilevel Monte Carlo estimator for rare path events in Gaussian Volterra models.

The paper will not claim that the G10 single-level estimator is universally superior.
That hypothesis failed its predeclared `2x` gate. The new primary object is the
fine-minus-coarse correction, where G10 produced a frozen aggregate work ratio of
`2.395x` and all finite-grid identities passed.

This is not yet evidence of an improved asymptotic MLMC complexity. Across the 12 G10
core regimes, the fitted marginalized correction-variance slopes ranged from about
`-0.061` to `0.897`. Those fits used a budget designed for falsifying constant-factor
work, not for precise rate estimation. V11 therefore separates three claims:

1. **Exactness claim:** finite-dimensional marginalization and telescoping are exact.
2. **Rate theorem:** under an explicit threshold-approximation assumption, smoothing
   changes the correction second-moment upper bound from order `h^r` to `h^(2r)`.
3. **Practical claim:** full MLMC uses less total work at matched RMSE and coverage.

Claims 2 and 3 must each pass their own gate. A constant-factor result cannot be
reported as a complexity-rate result.

No plan can guarantee that a theorem is novel, that an implementation contains no
unknown defect, or that a journal will accept the paper. This plan replaces that
impossible guarantee with proof obligations, independent oracles, frozen protocols,
explicit stop rules, and reproducibility requirements.

## 1. Research question

The primary question is:

> When a rare-event importance-sampling proposal is a defensive mixture of
> deterministic Gaussian shifts, can the complete proposal-shift span that drives a
> monotone path event be integrated out exactly at every MLMC level, while preserving
> unbiased finite-grid telescoping and improving correction work and variance decay?

The application question is:

> Does this construction give a reproducible reduction in total CPU/GPU work for
> discretely monitored rare downside events under rough Bergomi and a broader class
> of Gaussian Volterra volatility models?

The paper is about an estimator and its mathematical structure. CEM or a neural
network may generate a deterministic proposal schedule, but neither is part of the
exactness theorem. This distinction is mandatory.

## 2. Publication contribution hierarchy

### 2.1 Required main contributions

1. A general finite-dimensional **control-span marginalization theorem** for defensive
   Gaussian mixtures.
2. A **Rao--Blackwell theorem under the mixture proposal**, with the conditioning law
   derived rather than asserted.
3. A pathwise **defensive marginal-likelihood bound**.
4. An exact **multilevel telescoping theorem** allowing a different frozen proposal at
   each level while requiring one fine-level likelihood inside each correction.
5. A **threshold-rate theorem** showing `O(h^(2r))` for the marginalized correction
   second moment when the coupled scalar thresholds converge in `L2` at rate `r`.
6. A full adaptive MLMC implementation whose sample allocation uses independent
   pilot data and whose reported work includes pilot and proposal-calibration costs.
7. Frozen CPU reproduction and an independent GPU or second-environment reproduction.

### 2.2 Conditional strengthening contributions

These may appear only if their gates pass:

1. a rough-Bergomi corollary identifying `r` from model/discretization assumptions;
2. a matching or near-matching raw-correction rate that demonstrates exponent
   improvement, not merely an upper bound;
3. canonical or improved MLMC complexity for a declared range of roughness `H`;
4. an amortized time/parameter-conditioned neural control generator that reduces
   calibration cost without changing the estimator or likelihood.

### 2.3 Prohibited framing

- “Conditional smoothing is new.”
- “The model is unbiased for a continuous-time barrier” when only a finite grid was
  evaluated.
- “The variance exponent doubled empirically” without seed-clustered uncertainty.
- “Canonical complexity” unless the bias, variance, and cost assumptions are all
  proved or separately verified in the claimed setting.
- “Training-inclusive speedup” when CEM, pilot, selection, compilation, or failed-run
  costs were excluded.
- “Neural path integral” as a mathematical claim merely because a neural controller
  was used to output a deterministic schedule.
- Any rank-two extension as a rescue. G10 rank two is stopped.

## 3. Prior-art boundary that must be defended

The introduction and related-work section must treat the following as existing work:

1. the standard MLMC telescoping and complexity theorem;
2. conditional expectation and Brownian-bridge smoothing for discontinuous financial
   payoffs;
3. numerical smoothing by root finding and one-dimensional integration in MLMC;
4. multiple importance sampling and balance-mixture likelihoods;
5. importance-sampled MLMC in other stochastic systems;
6. hybrid/FFT simulation of Brownian semistationary and rough-Bergomi processes;
7. MLMC for rough-volatility/VIX functionals.
8. HJB-based hierarchical importance sampling for rare occupation time, including a
   common-likelihood MLIS construction and preprocessing-inclusive work.

The candidate novelty is narrower:

> exact removal of the event-driving proposal-control span from a defensive Gaussian
> mixture, with a bounded marginal likelihood and a common-coordinate multilevel
> correction specialization in a rough-Volterra simulator.

Common likelihood, occupation-time MLIS, and preprocessing-inclusive MLIS are prior
art and may not be presented as the contribution by themselves.

Before manuscript drafting, create `docs/literature/G11_NOVELTY_MATRIX.md` with at
least the following columns:

| Work | Model class | Discontinuous event | IS mixture | Span integrated | Exact marginal likelihood | ML correction | Rate theorem | Code |
|---|---|---|---|---|---|---|---|---|

Every novelty sentence in the paper must map to a row-wise gap in that matrix. Search
results alone are not sufficient; the methods and theorem statements of the closest
papers must be read.

## 4. Frozen mathematical convention

### 4.1 Fine-level Gaussian law

At level `l`, let

`X_l ~ P_l = N(0, I_(d_l))`.

For Brownian increments of size `h_l`, the implementation must use standardized
coordinates

`X_(l,n,k) = Delta W_(l,n,k) / sqrt(h_l)`.

Confusing raw Brownian increments with standardized normals changes both the mean
shift and its energy and is a correctness failure.

The target event at level `l` is `F_l(X_l) in {0,1}`. The finest finite-grid target is

`p_L = E_(P_L)[F_L]`.

Unless a separate weak-error theorem is proved, V11 estimates `p_L`, not a
continuous-time probability.

### 4.2 Adjacent coupling

For a dyadic hierarchy, define a normalized aggregation map `C_l` satisfying

`X_(l-1) = C_l X_l`, and `C_l C_l^T = I_(d_(l-1))`.

For each driver and coarse time cell,

`X_(l-1,n) = (X_(l,2n) + X_(l,2n+1)) / sqrt(2)`.

In raw increments this is the ordinary sum of the two fine increments. Unit tests must
check both representations.

The exact fine-level correction functional is

`Delta_l(X_l) = F_l(X_l) - F_(l-1)(C_l X_l)`.

Fine and coarse paths in one correction may not use independent Gaussian inputs,
proposal labels, controls, or likelihoods.

### 4.3 Defensive deterministic Gaussian mixture

At level `l`, expert `j` has a deterministic standardized mean shift `m_(l,j)` and

`Q_(l,j) = N(m_(l,j), I_(d_l))`.

The randomized proposal is

`Q_l = sum_j pi_(l,j) Q_(l,j)`,

where every weight is positive and sums to one. Expert zero is natural:

`m_(l,0) = 0`, `pi_(l,0) = delta_l`, and `delta_l >= delta_min > 0`.

For an SDE control `u(t)`, the standardized shift is `m_n = sqrt(h_l) u(t_n)`.
All controls used by the exact V11 branch must be deterministic functions of declared
task/model parameters and time. A path-dependent feedback control is out of scope.

The full component density ratio is

`D_(l,j)(x) = dQ_(l,j)/dP_l(x)`
`             = exp(m_(l,j)^T x - 0.5 ||m_(l,j)||^2)`.

The full balance likelihood is

`L_l(x) = 1 / sum_j pi_(l,j) D_(l,j)(x)`.

Raw estimators are ordinary sample means of `L_l F_l` or `L_l Delta_l`.
Self-normalization is prohibited.

### 4.4 Integrated subspace

Let `U_l` have deterministic orthonormal columns. Decompose

`X_l = U_l Z_l + R_l`,

with `Z_l = U_l^T X_l` and `R_l = (I - U_l U_l^T) X_l`.

For each proposal shift define

`a_(l,j) = U_l^T m_(l,j)`,
`r_(l,j) = (I - U_l U_l^T) m_(l,j)`.

The marginal component density ratio on the residual sigma-field is

`Dbar_(l,j)(R_l)`
`  = exp(r_(l,j)^T R_l - 0.5 ||r_(l,j)||^2)`.

The marginal mixture likelihood is

`Lbar_l(R_l) = 1 / sum_j pi_(l,j) Dbar_(l,j)(R_l)`.

The production implementation is rank one. `U_l = q_l` must be strictly positive in
the price-driver time coordinates, have zero entries in all other coordinates, and
have unit Euclidean norm. All price-driver proposal shifts must be collinear with
`q_l` within a scale-aware tolerance. Otherwise the exact scalar branch rejects the
configuration.

## 5. Theorem contract

Each theorem must be written first in a standalone mathematical note and then encoded
as property tests. Proof review is a gate, not a manuscript clean-up task.

### T11-1: Gaussian-mixture marginalization

Let `G_l(R_l) = E_P[Delta_l(U_l Z + R_l) | R_l]`, where `Z ~ N(0,I)` is under the
target law. If `Delta_l` is integrable, then for samples from `Q_l`,

`E_Q[Lbar_l(R_l) G_l(R_l)] = E_P[Delta_l(X_l)]`.

The proof must integrate the proposal joint density over `Z`; it must not replace the
proposal conditional law of `Z | R` with a target Gaussian by assertion.

The level-zero version replaces `Delta_0` with `F_0`.

### T11-2: Rao--Blackwell identity under Q

Define the raw balance-mixture contribution

`H_l = L_l(X_l) Delta_l(X_l)`.

Then

`E_Q[H_l | R_l] = Lbar_l(R_l) G_l(R_l)`,

and therefore

`Var_Q(Lbar_l G_l) <= Var_Q(H_l)`.

This theorem guarantees variance non-increase against the raw estimator under the
same defensive mixture. It does not guarantee lower wall-clock work.

### T11-3: defensive bounds

Because the natural component has weight `delta_l`,

`0 < L_l(X_l) <= 1/delta_l`,
`0 < Lbar_l(R_l) <= 1/delta_l`.

Both full and marginal bounds must be stated. The G10 paper draft must not imply that
only the marginal estimator is bounded.

### T11-4: exact multilevel telescoping

For independently estimated levels, each of which may use a different frozen
`Q_l`, `U_l`, and `delta_l`, define

`p_hat_L = mean(Hbar_0) + sum_(l=1)^L mean(Hbar_l)`,

where

`Hbar_0 = Lbar_0 E_P[F_0 | R_0]`,
`Hbar_l = Lbar_l E_P[F_l - F_(l-1) o C_l | R_l]`.

Then

`E[p_hat_L] = E_P[F_L]`.

Different proposals across levels do not break telescoping. Using separate fine and
coarse likelihoods inside one `Hbar_l` does break the declared estimator and is
prohibited.

### T11-5: exact scalar-threshold correction

Assume, conditional on `R_l`,

`F_l = 1{Z <= A_l(R_l)}`,
`F_(l-1) o C_l = 1{Z <= A_(l-1)(R_l)}`,

for the same scalar target normal `Z`. Then

`G_l(R_l) = Phi(A_l) - Phi(A_(l-1))`.

The implementation must evaluate this signed difference with tail-stable log-CDF and
survival-function formulas. Direct subtraction of two rounded CDF values is not
allowed in production.

For the rBergomi adapter, the threshold representation is valid only when:

1. `|rho| < 1` by a declared numerical margin;
2. variance is strictly positive and finite;
3. all post-initial affine slopes are strictly positive;
4. the initial spot is above the hit barrier;
5. occupation time lies in `(0,T]`;
6. fine and coarse use the same fine target coordinate;
7. the event is explicitly defined on the right-endpoint finite grid.

### T11-6: correction-rate upper bounds

Let `phi_max = 1/sqrt(2*pi)`. Assume

`||A_l - A_(l-1)||_(L2(P)) <= C_A h_l^r`,

and `delta_l >= delta_min`. Since `Phi` is globally `phi_max`-Lipschitz,

`E_Q[Hbar_l^2]`
`  <= (1/delta_min) phi_max^2 C_A^2 h_l^(2r)`.

Thus the marginalized correction has a second-moment upper bound of order
`O(h_l^(2r))`.

For the raw signed indicator difference,

`E_Q[H_l^2]`
`  <= (1/delta_min) phi_max E_P[|A_l-A_(l-1)|]`
`  = O(h_l^r)`.

These are upper bounds. “The exponent exactly doubles” additionally requires lower
bounds or non-degeneracy assumptions and cannot be claimed from T11-6 alone.

Thresholds equal to positive or negative infinity require a separate trivial-event
case; expressions such as `infinity - infinity` are forbidden. The flagship theorem
will assume finite thresholds almost surely and the code will branch explicitly for
extended-real inputs.

### T11-7: MLMC complexity corollary

Let the finest-grid weak bias, correction variance, and per-sample cost satisfy

`|E[F_L] - p| = O(h_L^a)`,
`V_l = O(h_l^beta)`,
`C_l = O(h_l^(-gamma))`,

with the usual MLMC moment and independence assumptions, including the standard
compatibility condition `a >= 0.5 * min(beta,gamma)` for the stated classical theorem.
If this condition fails, a generalized complexity result must be cited and proved
applicable instead of reusing the classical conclusion. For DCS-MGI,
`beta = 2r` is available only when T11-6's threshold assumption is established.

The standard conclusions are:

- `beta > gamma`: `O(epsilon^-2)`;
- `beta = gamma`: `O(epsilon^-2 log(epsilon)^2)`;
- `beta < gamma`: `O(epsilon^(-2-(gamma-beta)/a))`.

FFT convolution has an `N log N` cost. The proof and empirical model must retain the
log factor or justify a finite-range effective power fit. It may not silently call
FFT cost `O(N)`.

If the target is only the declared finest grid `p_L`, there is no continuous-time
bias term and no continuous-time complexity claim. Report the fixed-`L` work needed
for a prescribed sampling RMSE separately.

## 6. Path-functional scope

The implementation proceeds from the easiest theorem oracle to the intended hard
application. A failure in a harder task does not invalidate an earlier theorem, but
the paper must retain the failure.

### Tier A: Gaussian threshold oracle

- affine scalar digital event;
- analytic target probability;
- analytic raw and marginalized moments when available;
- arbitrary ambient dimensions and randomized orthogonal bases.

Purpose: prove density, likelihood, conditioning, and telescoping code independently
of rBergomi.

### Tier B: terminal digital and barrier-only rBergomi tasks

Implement separate immutable task types:

- `TerminalThresholdTask`;
- `DiscreteBarrierHitTask`.

Do not encode them through degenerate occupation parameters in
`DownsideExcursionTask`. Separate types prevent accidental changes to event semantics.

Purpose: establish threshold-rate behavior without the order-statistic discontinuity
of occupation time.

### Tier C: hit-plus-occupation task

Retain the current `DownsideExcursionTask` as the practical stress-risk application.
Its threshold is the minimum of the barrier threshold and an occupation order
statistic. Prove finite-grid scalar equivalence, but treat a strong threshold-rate
corollary as conditional until the order-statistic stability is established.

### Out of scope for V11

- continuously monitored barrier exactness;
- stochastic/path-dependent proposal controls;
- rank-two numerical integration;
- deterministic quadrature advertised as unbiased;
- PPDE or learned conditional-event surrogates;
- quantum terminology or complex-valued amplitudes.

## 7. Software architecture

G10 experiment scripts currently import private helpers from one another. V11 must
move reusable research logic into typed library modules before building a full MLMC
driver.

### 7.1 New core modules

#### `src/path_integral/gaussian_span_marginalization.py`

Responsibilities:

- validated `GaussianMixtureShiftSpec`;
- orthonormal span construction and residual projection;
- full and marginal component log-density ratios;
- full and marginal balance-mixture likelihoods;
- generic raw and Rao--Blackwell contributions;
- basis-rotation invariance diagnostics;
- natural-component bound diagnostics.

Core dataclasses:

- `GaussianMixtureShiftSpec`;
- `OrthonormalControlSpan`;
- `MarginalLikelihoodEvaluation`;
- `MarginalizedCorrectionEvaluation`.

The generic module must not import rBergomi code.

#### `src/path_integral/stable_gaussian.py`

Responsibilities:

- stable `log_Phi` and `log_survival`;
- signed `Phi(a)-Phi(b)`;
- scaled signed differences in log space;
- explicit behavior for finite and extended-real thresholds;
- SciPy high-precision oracle comparisons in tests.

Existing stable routines may be migrated with characterization tests. Public G10 APIs
remain as compatibility wrappers until V11 frozen validation is complete.

#### `src/path_integral/mlmc.py`

Responsibilities:

- hierarchy validation;
- level-zero and adjacent-correction protocols;
- independent pilot/final sampling;
- optimal integer sample allocation;
- variance, cost, bias, and RMSE accounting;
- fixed-finest-grid and continuous-target modes as separate types;
- checkpoint/resume without sample duplication;
- mergeable sufficient statistics using numerically stable online moments.

Core dataclasses:

- `MLMCHierarchy`;
- `LevelPilotStatistics`;
- `LevelAllocation`;
- `LevelEstimate`;
- `MLMCResult`;
- `WorkLedger`.

#### `src/path_integral/seed_ledger.py`

Responsibilities:

- derive seeds from `(protocol, role, regime, task, level, replicate, stream)`;
- allocate separate streams for proposal Gaussian, mixture labels, inner diagnostics,
  bootstrap, and reference simulation;
- reject duplicate derived seeds across the full manifest;
- serialize and hash the ledger.

Do not derive label seeds by adding an undocumented constant inside an experiment.

#### `src/path_integral/rbergomi_dcs_mlmc.py`

Responsibilities:

- adapt exact FFT BLP fine/coarse paths to the generic Gaussian interface;
- construct the fine standardized coordinate and normalized coarse aggregation;
- compute terminal, barrier, and hit-plus-occupation thresholds;
- evaluate raw and marginalized level corrections;
- expose audit and production modes;
- forbid separate coarse likelihoods.

### 7.2 Refactors

1. Keep `control_span_smoothing.py` behavior frozen behind characterization tests.
2. Extract generic density math without changing G10 numerical outputs beyond
   `1e-12` absolute/relative tolerance.
3. Move `_candidate`, `_regime`, hashing, timing, and statistics helpers out of
   `experiments/g10_*` private imports.
4. Mark rank-two functions as experimental/stopped; do not delete evidence.
5. Require `torch.float64` in all research-evidence runs. Float32 is allowed only in
   explicitly labeled throughput experiments after a float64 comparison.

### 7.3 Experiment entry points

- `experiments/g11_gaussian_oracle.py`
- `experiments/g11_threshold_rate_pilot.py`
- `experiments/g11_mlmc_development.py`
- `experiments/g11_baseline_reproduction.py`
- `experiments/g11_frozen_cpu.py`
- `experiments/g11_frozen_gpu.py`
- `experiments/g11_artifact_audit.py`

Every script must accept a config and output path. Evidence scripts may not contain
hidden parameter defaults that alter the frozen protocol.

### 7.4 Config and result schemas

Planned configs:

- `configs/g11_gaussian_oracle.yaml`;
- `configs/g11_threshold_rate_pilot.yaml`;
- `configs/g11_mlmc_development.yaml`;
- `configs/g11_frozen_cpu.yaml`;
- `configs/g11_frozen_gpu.yaml`.

Each evidence artifact must record:

- config byte SHA-256;
- canonical input-artifact hashes;
- source commit and dirty-worktree flag;
- Python, PyTorch, CUDA, OS, CPU/GPU, thread count, and package versions;
- complete seed-ledger hash;
- dtype and deterministic-algorithm setting;
- warm-up, compilation, pilot, calibration, online, and audit costs separately;
- all failed cells and exceptions;
- `allow_nan=false` JSON serialization.

## 8. Implementation milestones and gates

### M0: freeze the mathematical note and novelty matrix

Deliverables:

- `docs/theory/G11_THEOREMS.md`;
- `docs/theory/G11_PROOF_AUDIT.md`;
- `docs/literature/G11_NOVELTY_MATRIX.md`;
- this plan linked from the README.

Gate M0:

1. T11-1 through T11-5 have complete proofs with dimensions and measures declared.
2. T11-6 is proved as an upper-bound theorem without an unjustified exact-rate claim.
3. At least the closest numerical-smoothing, MIS, rough-simulation, and rough-MLMC
   papers have been read at theorem/method level.
4. No novelty sentence contradicts the matrix.

Stop if T11-1, T11-2, or T11-4 fails. Do not write implementation around a false
identity.

### M1: generic Gaussian oracle

Implement the generic span module and test dimensions `2, 3, 7, 32`, mixture sizes
`1, 2, 5`, span ranks `0, 1, 2`, and defensive weights
`0.05, 0.10, 0.20, 0.50`.

The rank-two cases here test only the general Gaussian density algebra with analytic
conditional functions. They do not revive the stopped rBergomi rank-two event solver.

Required oracle comparisons:

1. analytic Gaussian probabilities;
2. direct numerical integration in dimensions where feasible;
3. large independent target Monte Carlo with confidence intervals;
4. raw-versus-marginalized paired mean comparison;
5. full-density reconstruction from parallel and residual factors.

Gate M1:

- maximum float64 pathwise reconstruction error `<= 1e-11`;
- maximum likelihood-bound violation `<= 1e-12` after scale-aware tolerance;
- all analytic mean errors within `max(1e-12, 4*SE)`;
- no basis permutation, sign, or orthogonal-rotation invariance failure;
- property tests pass at least 500 randomized valid cases;
- all deliberately invalid specifications are rejected.

### M2: rBergomi adapter and event hierarchy

Implement the three task tiers, normalized fine/coarse mapping, and generic wrappers.

Gate M2:

- fine and coarse coordinates match to `1e-11` where the rank-one contract requires;
- hard events equal scalar threshold events pathwise in every valid test;
- full/marginal component and mixture densities reconstruct to `1e-11`;
- reference and FFT engines agree on a shared low-dimensional oracle to `1e-10`;
- G10 frozen aggregate estimates and exactness summaries reproduce within declared
  statistical/numerical tolerances;
- no output is silently relabeled as continuous time.

### M3: complete MLMC engine

Implement level zero, all adjacent corrections, independent pilot allocation, final
sampling, checkpoint/resume, and exact work accounting.

For a sampling-variance target `epsilon_s^2`, use the standard continuous allocation
before integer rounding:

`N_l proportional to sqrt(V_l / C_l) * sum_k sqrt(V_k C_k) / epsilon_s^2`.

The implementation must recompute the achieved variance after integer rounding.
Final samples must be independent of pilot samples. Discarded pilot samples still
count as work.

Gate M3:

1. analytic telescoping oracles pass;
2. split-run/checkpoint-resume results are bitwise identical when deterministic mode
   is enabled;
3. no seed is reused;
4. estimated sampling variance is no greater than the target after any authorized
   top-up;
5. top-up decisions depend only on independent accumulated statistics, not on whether
   a result looks favorable;
6. coverage on Gaussian oracles is between `93%` and `97%` over at least 1,000 cheap
   repetitions for nominal 95% intervals.

### M4: threshold-rate identification

The G10 `2,000 paths x 10 seeds` budget is insufficient to establish asymptotic
variance slopes. V11 will estimate, per level:

- `E|A_l-A_(l-1)|`;
- `E|A_l-A_(l-1)|^2`;
- raw and marginalized correction second moments;
- variance, kurtosis, zero fraction, and sign balance;
- cost as both empirical time and operation-scaled proxy.

Use at least six dyadic fine levels after the coarsest oracle level. Determine a
candidate asymptotic window using a predeclared stability rule:

1. at least four consecutive levels;
2. removing either endpoint changes the slope by no more than `0.15`;
3. cluster-bootstrap coefficient of variation for every included variance estimate is
   at most `20%`;
4. the window-selection rule is applied identically to raw and marginalized data;
5. if no window qualifies, report “rate unidentified.”

Regression units are independent seed clusters, not individual paths or levels.
Report bootstrap intervals for `r`, `beta_raw`, `beta_DCS`, and their differences.

Gate M4 for an asymptotic-rate claim:

- threshold `L2` rate lower 95% bound is positive;
- observed `beta_DCS` is compatible with `2r` under a predeclared equivalence margin
  of `0.25`;
- at least 80% of core regimes have positive marginalized variance slope;
- no hidden level truncation or favorable-window manual selection occurred.

Failure does not invalidate finite-grid exactness, but it prohibits a complexity-rate
headline.

### M5: baseline implementation and verification

Required baselines:

1. crude target Monte Carlo;
2. raw single-CEM importance sampling;
3. raw natural/CEM defensive mixture;
4. conditional smoothing under the natural law, without IS;
5. G9 MGVS;
6. G10 rank-one DCS-MGI;
7. raw defensive MLMC;
8. DCS-MGI-MLMC;
9. a faithful numerical-smoothing MLMC baseline based on the closest published
   method, to the extent its assumptions apply.

Optional baselines such as SMC, QMC, or flow proposals may be included only with
matched targets and complete tuning costs. A poorly implemented external baseline is
worse than omitting it and declaring the scope.

Baseline gate:

- reproduce at least one published or analytic test case per external method;
- document any model mismatch;
- same event, finest grid, confidence target, hardware, dtype, and stopping rule;
- paired randomness used where mathematically valid;
- no method receives an oracle parameter unavailable to competitors without a
  separate oracle-labeled result.

### M6: development-only selection

Development may choose only from predeclared sets:

- natural mixture weight;
- CEM piecewise-control resolution;
- level-specific versus hierarchy-shared deterministic schedule;
- batch size and production kernel;
- fixed-finest-grid versus continuous-target experiment mode.

The objective is training-inclusive expected work at a target RMSE, under hard
exactness and likelihood constraints. Hyperparameters are selected on development
regimes/seeds only.

Disallowed selection:

- choosing the best validation seed;
- changing the task threshold after looking at confirmatory efficiency;
- dropping a slow or unfavorable regime;
- selecting a variance-slope window manually;
- moving calibration cost outside the ledger after observing the outcome.

Gate M6:

- development correction work ratio `>=1.25x` including online overhead;
- no exactness or normalization failure;
- full MLMC engine meets its RMSE target in at least 90% of development repetitions;
- a cost-amortization break-even count is reported for any reusable calibration.

### M7: freeze the confirmatory protocol

Before confirmatory execution:

1. create `configs/g11_frozen_cpu.yaml` and hash its raw bytes;
2. freeze task/model regimes, hierarchy, seed ledger, tolerance points, replications,
   gates, baselines, and exclusions;
3. commit all code and configs;
4. record the source commit in the protocol;
5. verify the validation seed namespace is disjoint from all earlier phases;
6. run only smoke tests that use a separate smoke namespace;
7. prohibit code changes after the first confirmatory seed is inspected.

Any necessary bug fix invalidates the full confirmatory artifact and requires a new
protocol version and untouched seed namespace.

### M8: frozen CPU evaluation

The core design must cross:

- roughness levels including low, medium, and high `H` in `(0,0.5)`;
- multiple volatility-of-volatility and correlation values;
- terminal, barrier, and hit-plus-occupation tasks;
- target probabilities spanning approximately `1e-3` to `1e-6` where reliable
  references are computationally available;
- at least three sampling-RMSE or relative-RMSE targets.

Thresholds used to create rarity bands must be calibrated on disjoint data and then
frozen. The confirmatory result must retain cells that miss their intended rarity.

Primary frozen practical gates:

1. geometric raw-defensive/DCS correction total-work ratio `>1.5x`;
2. seed-clustered one-sided 95% lower bound of that ratio `>1.0x`;
3. correction improvement in at least 80% of core cells;
4. full DCS-MGI-MLMC/raw-defensive-MLMC total-work ratio `>1.5x` at the two smallest
   predeclared error tolerances;
5. one-sided 95% lower bound for the full-MLMC ratio `>1.0x`;
6. nominal 95% interval coverage between 90% and 98% per aggregate task family;
7. all exactness, likelihood, seed, hash, and reference-consistency gates pass.

The full-MLMC work ratio includes:

- CEM calibration;
- mixture-weight and hierarchy selection;
- pilot paths;
- failed allocations and authorized top-ups;
- path generation, marginalization, likelihood, and payoff evaluation;
- JIT/graph compilation when not reusable in the declared deployment scenario.

Report online-only work as a secondary decomposition, not the headline.

### M9: independent reproduction

Minimum acceptable independence:

- a fresh process/environment;
- independently generated seed namespace;
- artifact retrieval from the committed source;
- no reuse of in-memory tensors or cached pilot statistics;
- either a second CPU implementation path or GPU execution with independent kernels.

GPU timing must synchronize before and after measured regions. CPU timing must freeze
thread count and affinity where available. Both require warm-up and alternating
baseline/method order.

Gate M9:

- target estimates statistically consistent with CPU frozen results;
- exactness tolerances pass;
- efficiency direction agrees in at least 80% of core cells;
- any performance discrepancy greater than 20% is explained and retained.

### M10: optional amortized neural control

This milestone is authorized only after M8 passes without it.

A neural network may map frozen task/model parameters and normalized time to a
deterministic control schedule:

`u_theta(t; H, eta, rho, xi, T, task_parameters)`.

Requirements:

- no realized Brownian path input;
- schedule frozen before sampling;
- exact balance likelihood recomputed from the emitted schedule;
- natural defensive component retained;
- training, validation, and amortization costs reported;
- comparison against interpolation and regression baselines;
- out-of-distribution failures retained.

The neural extension is useful only if it lowers multi-task calibration cost or
improves proposal quality. It is not needed for estimator correctness and must not be
used to mask a failed core MLMC result.

## 9. Test plan

### 9.1 Unit tests

- standardized/raw-increment conversion;
- normalized dyadic aggregation;
- deterministic mean-shift energy;
- mixture-weight validation;
- orthonormality and span containment;
- marginal shift projection;
- full/marginal log-density reconstruction;
- natural-component bounds;
- stable signed CDF difference in both tails;
- extended-real threshold cases;
- integer MLMC allocation and achieved variance;
- online-moment merge correctness;
- seed-ledger uniqueness;
- artifact hash verification.

### 9.2 Property tests

- basis sign/permutation/rotation invariance;
- label permutation invariance;
- split/merge invariance of sufficient statistics;
- raw and marginalized mean equality within uncertainty;
- variance non-increase under the matched proposal;
- exact telescoping for random finite-dimensional functionals;
- scale invariance under equivalent standardized Brownian representations;
- defensive bounds over randomized valid mixtures.

### 9.3 Negative tests

The implementation must reject:

- path-dependent controls in the exact branch;
- zero or negative mixture weights;
- missing natural component when a defensive bound is claimed;
- non-collinear rank-one price shifts;
- mixed-sign or zero event direction;
- `|rho|` numerically equal to one;
- fine/coarse coordinate mismatch;
- separate coarse likelihood;
- self-normalized output;
- float32 evidence mode without explicit authorization;
- repeated seeds;
- config/source hash mismatch;
- NaN or infinite contribution;
- confirmatory execution from a dirty or wrong source commit.

### 9.4 Numerical tolerances

Use a scale-aware comparison

`|x-y| <= atol + rtol * max(|x|,|y|)`.

Default float64 oracle values are `atol=1e-12`, `rtol=1e-10`; path reconstruction
gates may use `1e-11`. Tolerances must be justified by conditioning and cannot be
relaxed after seeing a confirmatory failure.

## 10. Statistical analysis plan

### 10.1 Estimands

Primary estimands:

1. finest-grid rare-event probability;
2. correction variance and second moment per level;
3. variance-times-cost and total-work ratios;
4. full-MLMC achieved RMSE and interval coverage;
5. threshold, correction, and cost exponents with uncertainty.

### 10.2 Independence and clustering

- Paths are nested within seed replicate, regime, task, and level.
- Aggregate inference clusters by independent seed replicate.
- Levels sharing a hierarchy are not treated as independent observations.
- Multiple paths from one generator call are not pseudo-replicates for timing or rate
  uncertainty.

### 10.3 Confidence intervals

- Use seed-cluster bootstrap for aggregate work ratios and slopes.
- Use log ratios for positive work metrics.
- Report paired mean differences with paired standard errors.
- Construct probability intervals from independent replicate means using a declared
  Student-t or cluster-bootstrap procedure; do not rely on a naive path-level normal
  interval when contribution kurtosis is high.
- Use exact/binomial intervals for coverage proportions where appropriate.
- Retain both one-sided gate intervals and two-sided descriptive 95% intervals.

For an absolute-RMSE experiment, `epsilon` is fixed directly in the frozen config. For
a relative-RMSE experiment, use `epsilon = relative_tolerance * p_reference`, where
`p_reference` comes from an independent frozen reference artifact. A probability
estimated from the same final samples may not be plugged into their own stopping rule.

### 10.4 Multiple comparisons

The paper has one primary practical endpoint: full-MLMC total work at the two smallest
frozen tolerances. All regime/task breakdowns are secondary. If more primary endpoints
are introduced, control family-wise error or false discovery rate as predeclared.

### 10.5 References for probabilities near `1e-6`

A “gold” reference must combine independent estimators or an estimator with at least
fourfold smaller standard error than the method comparison. Its cost is reported but
not charged to deployment work. No tested method may tune on the final reference
noise.

## 11. Work and timing protocol

For each measured kernel:

1. warm up outside and record warm-up cost;
2. alternate method order by seed;
3. synchronize GPU streams;
4. fix CPU threads;
5. use identical batch sizes unless a separately reported optimized-throughput result
   is intended;
6. report median and distribution across independent repetitions;
7. record peak memory;
8. separate audit mode from production mode;
9. include data transfer when it belongs to deployment;
10. retain failures and out-of-memory events.

The primary work metric is measured total time at matched statistical accuracy.
`variance x online cost` is a diagnostic approximation, not a replacement for full
MLMC runs.

## 12. Error register

| Risk | Why it is serious | Prevention | Mandatory evidence |
|---|---|---|---|
| Using Q-conditional Gaussian instead of P-conditional Gaussian | biases marginalization | density-level proof of T11-1 | analytic mixture oracle |
| Raw vs standardized increment confusion | wrong shift and likelihood energy | typed conversion boundary | aggregation/energy tests |
| Independent fine/coarse likelihoods | breaks telescoping | one correction sample owns one fine likelihood | negative test |
| Span does not contain proposal shifts | omitted density factor | scale-aware residual gate | randomized reconstruction |
| Non-monotone event direction | scalar threshold is false | strictly positive direction gate | pathwise event replay |
| CDF tail cancellation | zero/incorrect correction | signed log-domain difference | high-precision tail oracle |
| Infinite thresholds | NaN differences | explicit extended-real cases | truth-table tests |
| Path-dependent expert | Gaussian deterministic-shift theorem fails | exact branch rejects it | negative test |
| Natural component omitted | bound is false | `delta_min` config invariant | bound test |
| Self-normalization | finite-sample bias | API returns contributions, never normalized weights | schema/test audit |
| Pilot reused adaptively | optional-stopping/allocation bias risk | independent final samples | seed-role audit |
| Seed collision | invalid independence | central ledger and uniqueness hash | full-manifest test |
| Noisy slope fit | false complexity claim | precision gate and clustered bootstrap | rate report |
| FFT called linear cost | incorrect theorem | retain `N log N` | timing/model comparison |
| Finite grid called continuous time | wrong estimand | separate result types | schema assertion |
| Validation retuning | optimistic bias | frozen config and new namespace after fixes | artifact audit |
| GPU asynchronous timing | false speedup | explicit synchronization | timing unit test/review |
| Calibration omitted | false practical claim | complete work ledger | break-even analysis |
| Baseline mismatch | artificial advantage | shared task/hierarchy protocol | baseline reproduction report |
| UTF-8 corruption | unreadable claims/evidence | UTF-8 read-back and replacement-character scan | artifact audit |

## 13. Stop, pivot, and success rules

### 13.1 Immediate stops

Stop V11 implementation if:

1. T11-1 or T11-4 is false under the intended proposal/coupling;
2. the generic Gaussian oracle cannot meet exactness gates;
3. the rBergomi event cannot be represented by one shared scalar coordinate;
4. confirmatory seed integrity is compromised and no untouched namespace remains.

### 13.2 Claim-specific stops

- If M4 cannot identify a stable rate, remove asymptotic-rate claims.
- If M8 correction work passes but full MLMC work fails, submit only a narrower
  correction/conditional-Monte-Carlo paper if novelty review supports it.
- If both correction and full MLMC practical gates fail, retain the theorem/software
  result but do not target a top numerical-finance journal with efficiency as the
  headline.
- If the closest prior paper already contains the same mixture-span theorem, stop the
  novelty claim and reassess before further compute.
- Do not add rank two, PPDE, flow, or quantum components after a failed frozen run.

### 13.3 Minimum success for a strong submission

All of the following are required:

1. T11-1 through T11-6 accepted after internal proof audit;
2. novelty matrix supports a precise non-overlap claim;
3. full implementation and tests are reproducible from a clean commit;
4. correction and full-MLMC frozen practical gates pass;
5. CPU and independent reproduction agree;
6. failed regimes and training-inclusive costs are reported;
7. manuscript claims stay within the finite-grid/general-theorem boundary.

### 13.4 Additional requirement for a top mathematical-finance claim

At least one is needed beyond the minimum success:

- a proved rough-Volterra threshold rate and resulting sharp MLMC complexity;
- a lower-bound/non-degeneracy result showing genuine exponent improvement;
- a theorem covering a meaningful class of non-Markovian Gaussian Volterra models,
  not only one rBergomi implementation.

Without this, SIAM Journal on Financial Mathematics or a strong computational-finance
journal may be realistic if experiments pass; a top theoretical mathematical-finance
journal should not be promised.

## 14. Manuscript structure

1. Problem and contribution boundary.
2. Gaussian Volterra discretization and defensive proposal.
3. Control-span marginalization and Rao--Blackwell theorems.
4. Multilevel telescoping and threshold-rate analysis.
5. Rough-Bergomi finite-grid specialization.
6. Algorithms and work accounting.
7. Gaussian and published-baseline validation.
8. Frozen CPU/GPU experiments.
9. Failure modes, limitations, and continuous-time boundary.
10. Reproducibility appendix with configs, seeds, hashes, and hardware.

The abstract may mention the `2.395x` G10 correction result only as preliminary
motivation until V11 independent frozen results exist.

## 15. Concrete execution order

The implementation order is non-negotiable:

1. Write and audit T11-1--T11-6.
2. Complete the novelty matrix.
3. Add characterization tests for G10.
4. Implement the generic Gaussian span module.
5. Pass Gaussian analytic/property oracles.
6. Refactor stable Gaussian arithmetic.
7. Implement task-specific rBergomi adapters.
8. Reproduce G10 without numerical drift.
9. Implement seed ledger and full MLMC engine.
10. Pass analytic telescoping and coverage tests.
11. Run the development-only rate study.
12. Implement and validate external baselines.
13. Select only predeclared development choices.
14. Freeze and commit the confirmatory protocol.
15. Run untouched CPU validation.
16. Audit artifacts before reading headline conclusions.
17. Run independent reproduction.
18. Decide claims using the predeclared gates.
19. Only then authorize the optional amortized neural controller.
20. Draft and independently proof-check the manuscript.

## 16. Estimated effort and resources

The estimates below are planning ranges, not deadlines:

| Phase | Researcher effort | Compute character |
|---|---:|---|
| M0 theorem/novelty | 1--2 weeks | light |
| M1 generic oracle | 1 week | light |
| M2 rBergomi adapter | 1--2 weeks | light/medium |
| M3 MLMC engine | 2 weeks | medium |
| M4 rate study | 1--2 weeks | medium/high |
| M5 baselines | 2--3 weeks | medium/high |
| M6 development | 1--2 weeks | high |
| M7 freeze/audit | 2--3 days | light |
| M8 frozen CPU | budget determined before freeze | high |
| M9 reproduction | 1 week plus compute | high |
| M10 optional neural | 2--4 weeks | high |
| Manuscript | 3--5 weeks | light compute |

The frozen path budget is not chosen to fit a desired result. It will be computed from
development variance estimates to achieve predeclared precision, capped by a resource
limit recorded before validation. If the cap is insufficient, the result is
“underpowered,” not “passed.”

## 17. Deliverable checklist

### Theory

- [ ] T11-1 marginalization proof
- [ ] T11-2 Rao--Blackwell proof
- [ ] T11-3 defensive-bound proof
- [ ] T11-4 telescoping proof
- [ ] T11-5 scalar-threshold proof
- [ ] T11-6 rate-bound proof
- [ ] T11-7 conditional complexity corollary
- [ ] proof audit and claim ledger

### Software

- [ ] generic Gaussian span module
- [ ] stable Gaussian arithmetic module
- [ ] seed ledger
- [ ] complete MLMC engine
- [ ] rBergomi adapter and three task types
- [ ] baseline implementations
- [ ] artifact-audit command
- [ ] CPU/GPU production timing paths

### Evidence

- [ ] analytic Gaussian oracle
- [ ] G10 characterization reproduction
- [ ] threshold-rate pilot
- [ ] baseline reproduction report
- [ ] frozen CPU report
- [ ] independent reproduction report
- [ ] training-inclusive work ledger
- [ ] full failure table

### Publication

- [ ] novelty matrix
- [ ] pre-registration/frozen protocol
- [ ] theorem-to-test traceability table
- [ ] manuscript claim ledger
- [ ] reproducibility README
- [ ] archival release with immutable hashes

## 18. Initial references for the formal review

The systematic review must start from, but not be limited to:

1. M. B. Giles, “Multilevel Monte Carlo Path Simulation,” Operations Research 56(3),
   2008. <https://doi.org/10.1287/opre.1070.0496>
2. C. Bayer, C. Ben Hammouda, and R. Tempone, “Multilevel Monte Carlo with Numerical
   Smoothing for Robust and Efficient Computation of Probabilities and Densities.”
   <https://arxiv.org/abs/2003.05708>
3. C. Bayer, C. Ben Hammouda, and R. Tempone, “Numerical Smoothing with Hierarchical
   Adaptive Sparse Grids and Quasi-Monte Carlo Methods for Efficient Option Pricing.”
   <https://arxiv.org/abs/2111.01874>
4. A. Bennedsen, A. Lunde, and M. S. Pakkanen, “Hybrid scheme for Brownian
   semistationary processes,” Finance and Stochastics 21, 2017.
   <https://doi.org/10.1007/s00780-017-0335-5>
5. F. Bourgey and S. De Marco, “Multilevel Monte Carlo simulation for VIX options in
   the rough Bergomi model.” <https://arxiv.org/abs/2105.05356>
6. N. Ben Rached, A.-L. Haji-Ali, S. M. S. Pillai, and R. Tempone, “Multilevel
   Importance Sampling for Rare Events Associated With the McKean--Vlasov Equation.”
   <https://arxiv.org/abs/2208.03225>
7. E. Ben Amar, N. Ben Rached, and R. Tempone, “Hierarchical Importance Sampling for
   Estimating Occupation Time for SDE Solutions.”
   <https://arxiv.org/abs/2509.13950>

## 19. Final decision rule

V11 is successful as a paper project only if the mathematics, full MLMC work, and
independent reproduction agree. The implementation is successful as research even if
it falsifies the practical hypothesis, provided the failure is preserved and the
claims are reduced accordingly.

The immediate next action is M0: formalize T11-1--T11-6 and build the novelty matrix.
No new confirmatory experiment is authorized before those two documents pass audit.
