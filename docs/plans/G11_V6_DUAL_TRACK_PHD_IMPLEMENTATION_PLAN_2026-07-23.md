# G11 V6 Dual-Track PhD Implementation Plan

Date: 2026-07-23

Status: implementation-ready development protocol; confirmatory protocol is not frozen

Target: one common estimator core supporting both a computational paper (Route A) and
a theory-led paper (Route B)

## 0. Executive decision

The next version should not be two unrelated projects. It should be one finite-grid,
exact-likelihood research program with two independently falsifiable contribution
tracks.

- **Route A — computational-method track:** demonstrate that a predeclared routing
  policy using crude MC, DCS-SLIS, and selective Hybrid DCS-MGI reaches an actual
  RMSE target with less *total* work than strong task-tuned CEM baselines on genuinely
  rare rBergomi events.
- **Route B — theory-led track:** turn the current conditional terminal rBergomi rate
  into a theorem whose assumptions are either proved for the implemented scheme or
  exposed as explicit, numerically falsifiable conditions. The discrete-barrier
  theorem is a second-stage extension, not something inferred from the terminal case.
- **Combined paper:** exact finite-grid estimator + model-level terminal analysis +
  achieved-RMSE rare-event evidence. This is the only route that justifies aiming at
  the strongest journal tier currently under consideration.

The implementation order is common-core first, then A and B in parallel only after
the relevant common gate passes. Route A may not use an unproved B theorem as evidence
of efficiency. Route B may not use an empirical rate plot as a substitute for a proof.

### 0.1 Why V6 is necessary

The completed V5 qualification established implementation correctness, not a
performance contribution:

- all 120 achieved-RMSE runs were accurate and independently reproducible;
- event probabilities were approximately 25%--28%, so the matrix was not rare;
- every run selected the finest-grid DCS-SLIS endpoint;
- selection consumed about 90.4%--91.9% of total work;
- Hybrid was about 52--59 times more expensive than crude MC and about 1.8 times more
  expensive than CEM in that moderate-event regime; and
- the terminal and barrier model-level rate theorems remain conditional.

Therefore V6 must solve the identified scientific bottlenecks. Adding neural layers,
dimensions, or quantum terminology would not solve them.

### 0.2 한국어 실행 요약

공통 코어를 먼저 고친 뒤 두 경로를 독립적으로 판정한다.

- 공통 코어에서는 실제 희귀확률을 갖는 셀을 만들고, reference를 보지 않는
  저비용 라우터, profiling 예산 상한, 실제 RMSE 할당, 완전한 비용·seed 장부를
  구현한다.
- A 경로에서는 `crude / CEM / defensive CEM / DCS-SLIS / V6 policy`를 동일한
  목표와 하드웨어 조건으로 실행한다. 학습·선택·실패·재시도 비용을 모두 포함한
  상태에서 V6가 CEM보다 유의하게 효율적인지를 검정한다.
- B 경로에서는 현재 조건부인 terminal rBergomi rate를 실제 코드의 방향,
  fine/coarse coupling, slope와 정확히 일치시키며 증명한다. Barrier는 active time과
  fine-only monitoring point 때문에 별도 정리가 필요하다.
- A만 성공하면 계산논문, B만 성공하면 이론논문, 둘 다 성공하면 결합 논문으로
  간다. 어느 한쪽의 실패를 다른 쪽의 증거로 덮지 않는다.
- 구현 우선순위는 `프로토콜 -> 희귀 셀/reference -> 라우터/selector -> 실제
  RMSE runner/auditor -> A qualification -> B terminal theorem -> power/freeze ->
  untouched confirmation/Linux reproduction` 순서다.

## 1. Intended contribution and claim ladder

Claims are promoted only in the following order. A higher claim is prohibited unless
all lower claims needed by it have passed.

| Level | Permitted claim | Required evidence |
|---|---|---|
| C0 | Code computes the declared finite-grid estimand | unit/oracle tests and schema audit |
| C1 | Estimator is exact/unbiased for the finite-grid target | exact likelihood and independent final-sample proof |
| C2 | Router-selected estimator remains unbiased | router is pilot-measurable; final seeds are disjoint |
| C3 | Requested RMSE is actually attained | executed integer allocations and independent references |
| A1 | Competitive in selected rare-event cells | all training, routing, profiling, retries, and final work charged |
| A2 | Computationally superior on the frozen primary matrix | predeclared paired primary endpoint and accuracy co-gate pass |
| B1 | Terminal rBergomi correction has a proved rate | every model assumption discharged or precisely stated |
| B2 | Complexity improvement follows | separate weak-bias/cost assumptions plus B1 |
| B3 | Barrier rate holds | active-time, small-slope, and mesh-enrichment obligations proved |
| AB | Combined top-tier submission claim | A2 and at least B1 pass under one consistent model/protocol |

There is no planned claim of uniform superiority over CEM, SLIS, or MLMC. There is no
continuous-time exactness claim unless a separate discretization/monitoring-bias
theorem is completed.

## 2. Publication fit and objective standard

The plan is aligned to journal standards, not to a guaranteed venue outcome.

- *SIAM Journal on Scientific Computing* requires a genuinely new numerical method
  with meaningful computational evidence and emphasizes reproducible research.
- *SIAM Journal on Financial Mathematics* accepts significant theoretical or
  computational advances in financial mathematics, not an incremental architecture
  swap.
- *Mathematical Finance* is plausible only if the financial insight and model-level
  mathematical result are strong; a benchmark-only paper is insufficient.

Official scopes: [SISC](https://epubs.siam.org/sisc/about),
[SIFIN](https://epubs.siam.org/journal/sifin/about), and
[Mathematical Finance](https://onlinelibrary.wiley.com/page/journal/14679965/homepage/productinformation.html).

The closest methodological comparison includes recent hierarchical importance
sampling work that explicitly accounts for preprocessing cost and studies the
SLIS/MLIS crossover. It must be treated as a direct baseline and novelty boundary,
not merely cited as related work:
[Hierarchical Importance Sampling for Rare Events](https://arxiv.org/abs/2509.13950).

## 3. Frozen mathematical object

### 3.1 Primary estimand

For the first submission, the primary estimand is the probability of a declared event
on the finest fixed grid `L=128` under the declared rBergomi discretization:

`p_L(theta) = E_P[F_L(theta)]`.

`theta` includes the model parameters, task, and event threshold. Terminal and
discrete-barrier events are separate tasks. A continuous-time probability is a
different estimand and is outside the primary claim.

### 3.2 Proposal family

At level `ell`, the target standardized Gaussian input is `P_ell=N(0,I)`. Every
defensive proposal is a deterministic finite Gaussian mixture

`Q_ell = sum_j pi_(ell,j) N(m_(ell,j),I)`

containing a zero-shift component of total weight `delta_(ell,0)>0`. The target over
proposal likelihood is evaluated exactly. The defensive bound is

`0 < L_ell <= 1/delta_(ell,0)`.

The code may use `1/min_j pi_j` as a conservative bound, but the statistical theory
and future implementation must expose the actual zero-shift mass. Conflating the
minimum component weight with the defensive mass is safe only as an overly
conservative bound and can make the selector needlessly expensive.

### 3.3 DCS restriction

The implemented exact scalar marginalization applies only when:

1. the controlled price-driver shifts lie in one common oriented rank-one span;
2. proposal controls are deterministic after training;
3. fine and coarse events are represented with the same standardized scalar
   coordinate; and
4. all residual likelihood factors are retained.

Feedback controls, a learned random direction that depends on the final path, or
inconsistent fine/coarse directions are outside the theorem. A higher-rank extension
requires a new multivariate conditional integral and is not part of V6.

### 3.4 Independence filtration

Let `G_dev` contain development data, `G_ref` reference generation, and `G_pilot`
proposal training, routing, profiling, and allocation pilots. The selected method and
integer allocation must be measurable with respect to these pre-final sigma-fields.
Final samples are drawn from a new seed namespace and are never reused in training or
selection. Conditional unbiasedness then follows by conditioning on the frozen
choice and applying exact likelihood/telescoping term by term.

This proof covers a random router as well as a random CEM proposal. It does **not**
cover recycling pilot observations into the reported final mean.

## 4. Common V6 architecture

```text
development calibration
        |
        v
independent reference bank -----> frozen cell manifest
        |                                |
        |                                v
        +----------------------> screening pilot
                                         |
                              frozen rarity/work router
                          /              |              \
                       crude          DCS-SLIS       Hybrid DCS-MGI
                          \              |              /
                           frozen integer allocation
                                         |
                               independent final seeds
                                         |
                       estimator + total-work + audit artifact
                                  /                 \
                              Route A              Route B
                         efficiency tests      theorem diagnostics
```

Reference values are allowed to score benchmark accuracy. They are not available to
the production router and may not select the best-looking method. The router sees
only its own screening/training pilot and frozen configuration.

## 5. Common-core work packages

### C0. Protocol, provenance, and schema freeze

Create a V6 namespace without reusing V5 seeds or artifact identities.

Planned schemas:

- `npi.g11.v6-cell-manifest.v1`;
- `npi.g11.v6-reference.v1`;
- `npi.g11.v6-routing-qualification.v1`;
- `npi.g11.v6-achieved-rmse.v1`;
- `npi.g11.v6-theory-diagnostics.v1`;
- `npi.g11.v6-power.v1`;
- `npi.g11.v6-confirmatory.v1`;
- `npi.g11.v6-independent-audit.v1`; and
- `npi.g11.v6-hardware-reproduction.v1`.

Every result records source commit, dirty-tree flag, config bytes and SHA-256, Python
and dependency versions, platform, BLAS/thread settings, device, wall time, CPU time,
peak memory, work ledger, seed ledger, and checkpoint hash. A dirty tree is allowed
for smoke tests but prohibited for qualification and confirmation.

Deliverables:

- `src/path_integral/v6_protocol.py` for schema validation and frozen identities;
- `configs/g11_v6/*.json` for development, qualification, power, and confirmation;
- tests rejecting unknown fields, duplicate cell IDs, invalid seed overlaps, and
  mismatched hashes.

Gate C0 passes only if an artifact round-trip is deterministic and an independent
parser rejects every deliberately corrupted fixture.

### C1. Rare-cell calibration without target leakage

The latest validated Hurst set is initially fixed to `H={0.05,0.12,0.30}`. Terminal
and discrete-barrier tasks are calibrated at nominal probability bands around
`1e-2`, `1e-3`, and a feasible `1e-4` sentinel on the 128-step grid.

Calibration is development work and uses a dedicated `g11-v6-calibration-*` seed
namespace. For each `(H,task,band)`:

1. bracket the threshold by monotone common-random-number evaluations;
2. solve for a threshold using a safeguarded bisection/interpolation method;
3. rerun an independent validation batch;
4. record a binomial confidence interval for crude Bernoulli paths where feasible;
   for the `1e-4` sentinel, use a separately validated exact-likelihood defensive
   estimator and its justified bounded interval rather than pretending its weighted
   samples are binomial; and
5. accept the cell if the interval lies inside the predeclared band and the model
   diagnostics pass.

Suggested probability bands, fixed before execution:

| label | acceptable reference interval | role |
|---|---:|---|
| moderate-rare | `[0.5e-2, 2e-2]` | router/crude crossover |
| rare | `[0.5e-3, 2e-3]` | primary qualification |
| sentinel | `[0.5e-4, 2e-4]` | hardest feasible stress cell |

These are qualification bands, not a claim that all tasks must attain exactly their
center. Thresholds are frozen after calibration. Outcome-dependent replacement of a
cell because a method performs badly is prohibited.

Use `H=0.07` only if a written model/discretization rationale is approved before any
method-performance result; it may not be substituted post hoc for `H=0.05`.

### C2. Independent reference bank

Build matched references for every frozen cell. The reference procedure may use a
high-cost defensive mixture and stratification, but it must have an exact likelihood
and an independent seed namespace.

For an intended absolute sampling RMSE `epsilon`, require

`SE(reference) <= 0.10 * epsilon`.

If this is unaffordable for a sentinel cell, the cell is labelled
`reference-infeasible` before performance confirmation. It may remain a qualitative
stress test but cannot enter the primary accuracy endpoint.

Required checks:

- two independent reference batches agree within their combined uncertainty;
- defensive and non-defensive valid estimators agree within uncertainty;
- likelihood normalization `E_Q[L]=1` is checked with uncertainty;
- event conventions and barrier inclusivity match the tested production code; and
- reference seeds do not occur in calibration, training, profiling, or final ledgers.

### C3. Rarity and economic router

The V5 failure shows that method choice needs a cheap first stage. Implement a frozen
router using only an independent screening pilot.

Inputs:

- crude Bernoulli count and an exact or rigorously covered confidence interval;
- cheap DCS-SLIS variance/cost profile;
- total work already spent; and
- user-requested `epsilon` or relative RMSE target.

Predeclared routing logic:

1. If the lower probability confidence bound is above `p_router`, route to crude MC
   unless the cheap DCS-SLIS profile has already certified lower total work.
2. If the upper bound is below `p_router`, compare DCS-SLIS with the legal Hybrid
   candidates under the remaining profiling budget.
3. If the interval straddles `p_router`, take at most one additional fixed-size look;
   after that use the predeclared conservative fallback, initially DCS-SLIS.
4. Never launch a profile if its minimum unavoidable setup cost exceeds the largest
   possible certified saving at the requested accuracy.

`p_router` is tuned on development seeds and frozen before qualification. A starting
candidate is `0.05`, matching the existing rare-event gate, but this value is not a
theorem and must survive qualification.

The router is a *policy*, so its screening and misrouting costs are included in Route
A. Its selected estimator stays unbiased because final samples are independent and
every route estimates the same `p_L` exactly.

New module:

- `src/path_integral/rarity_router.py` with typed frozen decisions, rejection reasons,
  work accounting, and serialization.

Unit/oracle tests:

- forced moderate, rare, and ambiguous cases;
- exact budget boundary and integer rounding;
- no reference probability in router inputs;
- identical frozen decision after serialization;
- pilot/final seed collision rejection; and
- conditional Monte Carlo oracle showing no selection bias.

### C4. Budgeted Hybrid selector

Retain the statistically safe V5 selector but stop profiling when the possible gain
cannot repay additional selection cost.

Implementation rules:

- start with the legal endpoint DCS-SLIS;
- eliminate candidates first by deterministic cost lower bounds;
- use fixed predeclared look sizes and familywise-error allocation;
- cap selection work by both an absolute budget and a fraction of the current
  achievable total-work estimate;
- if no candidate is certified better when the cap is reached, return DCS-SLIS;
- charge every candidate profile and failed/retried profile; and
- expose the zero-shift defensive mass rather than using the smallest mixture weight
  when forming bounds.

Do not replace the current Hoeffding argument with an empirical-Bernstein or
time-uniform confidence sequence formula until its assumptions, range, optional-look
coverage, and implementation have separate theorem and oracle tests. A statistically
invalid fast selector is worse than a slow valid one.

Qualification targets:

- median selection-work fraction below 25%;
- 90th percentile below 40%;
- median selected-to-oracle total-work regret no more than 10%; and
- no familywise coverage failure beyond the predeclared Monte Carlo tolerance.

The oracle uses held-out, oversized profiles only for scoring selector regret; it is
not available to the production decision.

### C5. Unified achieved-RMSE execution

Refactor the current `HybridTarget`, `prepare_hybrid_run`, and `execute_hybrid_run`
into a policy-level runner capable of executing crude, pure CEM, defensive CEM,
DCS-SLIS, or Hybrid under one target contract.

The preparation stage must freeze:

- cell and method-policy identity;
- trained proposal parameters;
- routing and selector decisions;
- variance/cost estimates and simultaneous intervals;
- target `epsilon`;
- integer allocation rounded upward;
- maximum final work and censoring rule; and
- an immutable preparation hash.

The final stage may resume a checkpoint but may not revise the proposal, route,
candidate, target, or allocation after seeing final samples.

Harden `HybridCheckpoint.from_dict` to the same strict parsing standard as the MLMC
checkpoint. Remove the current loose type paths around engine and seed role by using
explicit literal/enumerated types.

Headline results use executed work only. Profile-based projections are allowed only
in resource planning and must be labelled as projections.

### C6. Independent auditor

The auditor must recompute from raw sufficient statistics:

- seed disjointness;
- exact route and selector result;
- allocation and ceiling arithmetic;
- likelihood/telescoping identities;
- estimates, variances, confidence intervals, RMSE gates;
- all work components and censored runs;
- paired aggregates and familywise/multiple-comparison corrections; and
- every final pass/fail decision.

It must not import the production decision helper that it is auditing. Shared low-level
mathematical primitives are acceptable only when separately oracle-tested.

## 6. Route A — computational-method implementation

### A0. Precise research question

At a fixed 128-step rBergomi target and actual requested sampling error, does the
predeclared V6 policy reduce total computation relative to a strong task-tuned pure
CEM baseline while retaining valid accuracy across a heterogeneous rare-event matrix?

The unit of comparison is a paired seed cluster within a fixed cell. The proposed
method is the whole policy, including cases where it routes to crude or DCS-SLIS.

### A1. Baselines

All baselines must use the same simulator, path convention, hardware thread policy,
accuracy target, reference, and seed-cluster structure.

Required methods:

| ID | Method | Required treatment |
|---|---|---|
| A-B0 | crude MC | exact Bernoulli estimator; actual allocation |
| A-B1 | task-tuned pure CEM | strong shift family; training charged |
| A-B2 | defensive CEM | exact balance likelihood; training charged |
| A-B3 | DCS-SLIS | full profiling/allocation charged |
| A-P | V6 routed Hybrid policy | all routing/selection/training/final work charged |

The primary comparator is A-B1, frozen before confirmation. A-B0, A-B2, and A-B3 are
mandatory secondary comparisons and diagnostic ablations. Selecting the easiest
baseline after seeing outcomes is prohibited.

The closest published hierarchical/HJB importance-sampling method receives a
separate applicability audit. If it can be reproduced under the identical
non-Markovian rBergomi target, discretization, event, and work contract, add it as
`A-B4` and include its preprocessing. If it requires a different Markov state or a
lifted approximation that changes the target, reproduce it on its native benchmark
and explain the mismatch, but do not place incomparable numbers in the primary
table. Neural/flow baselines follow the same rule and must include training cost.

CEM qualification includes:

- event-specific threshold and task objective;
- several prespecified initializations;
- convergence/failure rules;
- likelihood normalization and effective-sample diagnostics;
- identical maximum training budget across compared cells; and
- a held-out final batch.

### A2. Development and qualification matrices

Phase A-D is explicitly exploratory:

- `H={0.05,0.12,0.30}`;
- terminal and discrete-barrier tasks;
- probability bands around `1e-2` and `1e-3`;
- feasible `1e-4` sentinels;
- relative RMSE targets initially 20% and 10%; and
- enough independent clusters to estimate log-work variance, not to claim superiority.

Phase A-Q is a new-seed qualification used for power and resource planning. It must
retain the intended scientific matrix. Cells may be excluded only for a predeclared
estimand/reference/model-validity/resource-feasibility reason, never because the V6
policy lost to CEM.

Phase A-C is untouched confirmation. The number of clusters is determined by the
frozen paired-power analysis. The old default of 20 clusters is retained only if it
provides at least 80% power for the declared practically meaningful effect.

Do not jump directly to `1e-6`. Promotion beyond `1e-4` requires a feasible reference,
stable likelihood moments, and a new resource audit. Otherwise the experiment risks
becoming an unfinishable demonstration rather than a scientific comparison.

### A3. Accuracy endpoint

For each method/cell, execute the integer allocation and compute empirical squared
error across clusters relative to the independent reference.

Required co-gates:

1. empirical target attainment rate meets its predeclared lower confidence bound;
2. the RMSE upper bound is no larger than the tolerance after including reference
   uncertainty, for example
   `sqrt((1.25 epsilon)^2 + SE(reference)^2)`;
3. interval coverage and likelihood diagnostics pass; and
4. no method-specific censoring is silently removed.

The constant `1.25` is an engineering qualification margin, not a mathematical
identity. It must be frozen with the protocol.

### A4. Primary efficiency endpoint

For every primary cluster calculate

`R = total_work(CEM) / total_work(V6 policy)`.

Analyze `log R` with equal cell weight so a large easy cell cannot dominate the
conclusion. The primary claim requires:

- all accuracy co-gates pass;
- the one-sided 95% lower confidence bound for the equal-cell-weighted geometric mean
  ratio exceeds 1; and
- if resources permit, the design is powered for a practical threshold of 1.20.

The exact paired model (cluster fixed effect, cell effect, or prespecified cluster
bootstrap) is selected by a blinded simulation/power study and frozen before A-C.
Heavy-tailed work ratios require a robust sensitivity analysis, but the primary
analysis cannot be changed after outcomes are visible.

Censoring policy:

- resource censoring counts as a primary failure for that method/cell;
- no complete-case deletion;
- no post-hoc matched subset; and
- retry work remains charged even if the retry succeeds.

### A5. Work definition

Report at least three work metrics:

1. **primary deterministic operation work:** simulated time steps, likelihood
   evaluations, conditional CDF calls, optimizer steps, and all profiles;
2. **wall-clock work:** end-to-end elapsed time under a frozen thread configuration;
3. **resource footprint:** CPU time and peak resident memory.

Total work includes environment startup attributable to a method, proposal training,
screening, routing, selection, allocation pilots, final samples, checkpoints, failed
runs, and retries. Shared one-time reference generation is reported separately and is
not charged to a deployed method, because it is used only for evaluation.

### A6. Ablations and falsification tests

Mandatory ablations:

- router off versus on;
- DCS smoothing off versus on;
- defensive component off versus on, where moments remain numerically safe;
- fixed DCS-SLIS versus selected MLMC start;
- profiling cap values fixed from development; and
- terminal versus barrier tasks.

The paper is weakened, not strengthened, if only favorable ablations are reported.
All registered ablations are included in the artifact regardless of outcome.

### A7. Route A stop rules

Stop the superiority claim and do not spend untouched confirmation resources if any
of the following occurs in A-Q:

- probability/reference validation fails;
- actual RMSE fails after one prespecified allocation recalibration;
- median selection fraction remains above 40%;
- V6 total work is not plausibly competitive with CEM on the intended rare matrix;
- required power exceeds the approved compute budget; or
- independent audit finds a seed, work, or likelihood inconsistency.

If accuracy passes but efficiency fails, Route A becomes a reproducible negative
result/benchmark and is not described as a superior method.

## 7. Route B — theory-led implementation

### B0. The theorem must match the code

Create a notation-to-code ledger before proving rates. For each symbol record its
array shape, grid scaling, source function, and coupling convention. At minimum map:

- fine/coarse Brownian inputs and FFT/BLP construction;
- volatility path and lognormal factors;
- oriented direction weights;
- terminal/barrier intercepts and slopes;
- residual mixture likelihood;
- threshold map; and
- new fine monitoring points.

Any proof using a normalized vector while the code uses grid-scaled weights, or a
different fine/coarse aggregation, is invalid even if the algebra is correct.

Freeze a theorem parameter domain before claiming uniform constants. At minimum,
require `H` in a compact subset of `(0,1/2)`, volatility and maturity parameters in
declared compact sets, and `|rho| <= 1-eta_rho` for some `eta_rho>0`. At `|rho|=1`
the independent price-driver slope vanishes, so the implemented scalar smoothing
argument does not apply.

Deliverable: `docs/theory/G11_V6_NOTATION_CODE_LEDGER.md` plus shape/property tests.

### B1. Direction regularity

For the implemented deterministic positive direction, prove or explicitly assume:

- positivity and orientation;
- unit normalization in the standardized Gaussian coordinates;
- fine-to-coarse aggregation consistency;
- level scaling of maximum and summed weights; and
- stability under the actual BLP/FFT coupling.

Required code tests verify the identities for every configured level and reject a
sign flip or inconsistent coarse direction.

Failure rule: if the desired uniform scaling fails, state the observed direction
class and prove only the theorem it supports. Do not silently replace the
implementation with a proof-friendly direction after Route A results exist.

### B2. Terminal slope lower-tail and negative moments

The terminal slope is a positive weighted sum of lognormal volatility factors in the
independent price-driver direction. Pointwise positivity is insufficient for a
uniform inverse-moment bound.

The proof program is:

1. write the exact implemented slope, including `sqrt(1-rho^2)`, `sqrt(V_t)`,
   `sqrt(dt)`, and direction weights;
2. identify a positive subinterval/block with controlled total direction mass;
3. lower-bound the full sum by that block;
4. control the negative moments or lower tail of the block using Gaussian exponential
   moments and the rBergomi covariance structure; and
5. show constants and exponents are uniform over the declared finite hierarchy and,
   if claimed, as mesh size tends to zero.

Required diagnostic artifact:

- empirical lower-tail curves for slope;
- inverse-moment estimates over multiple meshes;
- confidence intervals and sample-size sensitivity;
- comparison to the proposed analytic exponent; and
- a flag that empirical stability is supportive, never proof.

Failure rule: if a uniform negative-moment result cannot be proved, retain a theorem
conditional on an explicit lower-tail exponent. Route B may still be publishable as a
general threshold-smoothing theorem plus a rigorous identification of the missing
rBergomi condition, but not as an unconditional rBergomi complexity theorem.

### B3. Fine/coarse coefficient coupling

Prove `L^p` rates for terminal intercept and slope differences under the exact shared
Gaussian coupling.

Separate sources of error:

- approximation of the Volterra Gaussian process;
- exponentiation into variance;
- asset/log-price discretization;
- direction aggregation; and
- proposal-shift terms in the intercept.

For each source, specify the norm, exponent, parameter dependence, and moment order.
Do not infer a coefficient rate from a regression slope alone.

Implementation support:

- extend `rate_analysis.py` to emit raw paired moments by level;
- add manufactured Gaussian/constant-volatility oracles with known rates;
- add rBergomi coupling identities as deterministic tests; and
- report local slopes only with predeclared fit windows and uncertainty.

### B4. Margin-localized terminal threshold theorem

Use the already proved deterministic ratio localization:

- on the good event, intercept/slope errors control threshold error;
- on the bad small-slope event, retain an explicit probability term;
- under the exact change of measure, retain the defensive factor; and
- optimize the localization threshold `kappa(h)` only after all exponents and moment
  orders are stated.

The resulting theorem must distinguish:

1. an upper bound on the second moment of the DCS correction;
2. an asymptotic rate under model assumptions;
3. an empirical fitted exponent; and
4. a complexity consequence after adding bias and cost assumptions.

These are not interchangeable claims.

### B5. Weak bias and complexity

A correction-variance theorem alone does not prove MLMC complexity. Add a separate
weak-bias statement for the target sequence and a deterministic per-sample cost rate.

For finite-grid-only Route A, no asymptotic bias theorem is needed. For Route B
complexity, state the standard triplet:

- weak bias exponent `alpha`;
- correction variance exponent `beta`; and
- per-sample cost exponent `gamma`.

Then derive the complexity regime from a named MLMC theorem after checking its
assumptions. Never announce `O(epsilon^-2)` merely because a fitted `beta` appears
large.

If continuous-time terminal bias cannot be proved, Route B stops at a finite-hierarchy
second-moment result and says so explicitly.

### B6. Barrier extension

Barrier events require new mathematics and are not a corollary of B4. Prove or leave
conditional:

- probability that the active maximizing threshold occurs too early;
- uniform control of the slope after the chosen active time;
- the contribution of monitoring points present only on the fine grid; and
- monitoring bias if a continuously monitored barrier is claimed.

The barrier theorem is promoted only after separate numerical diagnostics falsify
neither the active-time nor mesh-enrichment assumptions.

Failure rule: Route A may still include a finite-grid barrier experiment if exactness
and accuracy hold. Route B then claims a terminal theorem only and labels barrier
rates open.

### B7. Route B theorem-to-test contract

| Proof obligation | Code-level falsification check | Passing meaning |
|---|---|---|
| direction positivity/normalization | exact array assertions | code matches stated geometry |
| coarse aggregation | paired-level identity test | common-coordinate theorem applies |
| likelihood cancellation | Gaussian mixture oracle | finite-grid DCS is exact |
| slope negative moments | multimesh tail diagnostic | assumption not empirically contradicted |
| coefficient `L^p` rate | held-out multilevel moments | proposed rate is numerically plausible |
| threshold localization | pathwise bound audit | deterministic inequality is implemented |
| barrier active time | active-index distribution | bad-event hypothesis is plausible |
| mesh enrichment | fine-only monitoring audit | missing barrier term is measured |
| weak bias | independent reference levels | complexity premise is plausible |

Passing an empirical check never upgrades an assumption to a proof. Failing a check
forces theorem revision or scope reduction.

### B8. Route B stop rules

- If notation and implementation cannot be made identical, stop theorem work and fix
  the code/protocol first.
- If the terminal slope admits no usable uniform lower-tail control, do not claim an
  unconditional rBergomi rate.
- If coefficient rates depend on proposal parameters not uniformly controlled, state
  that dependence and restrict the theorem.
- If weak bias is open, do not claim end-to-end asymptotic complexity.
- If barrier mesh enrichment is open, do not transfer the terminal rate to barriers.

## 8. Concrete repository changes

### 8.1 New modules

| File | Responsibility |
|---|---|
| `src/path_integral/v6_protocol.py` | strict schemas, cell identity, frozen hashes |
| `src/path_integral/rarity_router.py` | pilot-measurable routing and work budget |
| `src/path_integral/policy_allocation.py` | common target/allocation across all methods |
| `src/path_integral/v6_work_ledger.py` | deterministic operation and resource accounting |
| `src/path_integral/rbergomi_theory_diagnostics.py` | slope, coefficient, active-time, mesh diagnostics |

Do not fork a second simulator. These modules must call the existing rBergomi,
Gaussian marginalization, MLMC, and seed-ledger primitives.

### 8.2 Existing modules to extend

| File | Change |
|---|---|
| `src/path_integral/rbergomi_hybrid.py` | expose zero-shift mass; strict engine/role types |
| `src/path_integral/hybrid_allocation.py` | strict checkpoint parser; policy interface |
| `src/path_integral/robust_crossover.py` | budget stop and deterministic dominance |
| `src/path_integral/statistical_gates.py` | paired primary endpoint and censoring rules |
| `src/path_integral/rate_analysis.py` | raw `L^p` moments and uncertainty, not only slopes |
| `src/path_integral/threshold_stability.py` | pathwise theorem diagnostics |
| `src/path_integral/seed_ledger.py` | V6 namespace roles and cross-artifact collision audit |

### 8.3 New experiments

| Script | Phase |
|---|---|
| `experiments/g11_v6_rarity_calibration.py` | C1 |
| `experiments/g11_v6_reference.py` | C2 |
| `experiments/g11_v6_router_qualification.py` | C3--C4 |
| `experiments/g11_v6_theory_diagnostics.py` | B1--B6 |
| `experiments/g11_v6_baseline_qualification.py` | A1 |
| `experiments/g11_v6_achieved_rmse.py` | A2--A5 |
| `experiments/g11_v6_power_analysis.py` | A4 |
| `experiments/g11_v6_confirmatory.py` | A-C |
| `experiments/g11_v6_result_audit.py` | independent audit |
| `experiments/g11_v6_hardware_reproduction.py` | second environment |

Each full experiment must support `--smoke`, `--resume`, `--config`, and `--output`.
Smoke output uses a distinct schema flag and can never satisfy a scientific gate.

### 8.4 Test plan

Unit tests:

- strict schema and checkpoint rejection;
- router boundary cases and budget arithmetic;
- zero-shift mass and defensive likelihood bound;
- allocation ceiling and zero/near-zero variance cases;
- work-ledger completeness;
- seed-role disjointness;
- direction aggregation and normalization;
- threshold ratio localization; and
- terminal/barrier convention fixtures.

Oracle tests:

- one-dimensional Gaussian tail with analytic probability;
- Gaussian mixture full/residual likelihood normalization;
- exact scalar threshold marginalization;
- fine/coarse Gaussian telescoping;
- pilot-selected method unbiasedness;
- manufactured MLMC rates;
- random-router unbiasedness; and
- checkpoint/resume bitwise equivalence.

Integration tests:

- one cell through calibration, reference, routing, allocation, final, and audit;
- all five methods under one target contract;
- forced censoring and failed retry accounting;
- independent audit deliberately catches a modified seed, cost, likelihood, and gate;
- Windows and clean Linux path/config round-trip.

Statistical tests must not assert that a random estimate equals a fixed value exactly.
Use deterministic fixtures where possible and calibrated tolerance/power where
randomness is essential.

## 9. Seed and data-dependency plan

Use keyed seed derivation, never positional counter reuse. A seed key includes:

`protocol / phase / cell / cluster / method / role / look / level / replicate`.

Reserved roles:

- `calibration`;
- `reference_a`, `reference_b`;
- `router_pilot`;
- `proposal_train`;
- `selector_profile`;
- `allocation_pilot`;
- `oracle_score`;
- `final`;
- `audit_replay`; and
- `hardware_reproduction`.

Forbidden edges:

- reference data into routing or method selection;
- final data into proposal training or allocation;
- oracle-score profiles into production selection;
- qualification clusters into confirmatory results; and
- Windows final samples reused as Linux reproduction samples.

The auditor rejects both identical seed integers and identical derived random-stream
identities across forbidden roles.

## 10. Revised gates and milestone order

| Gate | Requirement | Failure action |
|---|---|---|
| G0 | novelty and closest-work matrix current | narrow novelty claim |
| G1 | V6 protocol/schema/seed tests pass | no experiments |
| G2 | rare-cell calibration and references pass | recalibrate before method tests |
| G3 | router and selector coverage/regret/work gates pass | simplify policy; no Hybrid claim |
| G4 | five matched methods attain actual RMSE in qualification | fix allocation or stop A |
| G5 | paired power and resource plan feasible | reduce scope before freeze, not after |
| G6-A | Route A qualification supports practical competitiveness | no superiority confirmation |
| G6-B | terminal theorem obligations proved or honestly delimited | conditional theorem only |
| G7 | untouched config, code commit, seeds, and analysis frozen | no confirmation |
| G8 | untouched confirmation and independent audit pass | report failure; no refreeze |
| G9 | clean Linux reproduction passes | no reproducibility claim |
| G10 | paper claim ledger matches artifacts and proofs | revise manuscript |

Recommended execution order:

1. C0 protocol and strict checkpoint work;
2. C1 calibration and C2 references;
3. B0--B2 notation/direction/slope work while references run;
4. C3 router and C4 selector qualification;
5. C5 unified achieved-RMSE runner and C6 auditor;
6. A1 baselines and A-Q qualification;
7. B3--B5 terminal coefficient/rate/complexity work;
8. paired power/resource decision;
9. optional B6 barrier theorem;
10. G7 freeze, A-C confirmation, Linux reproduction, and manuscript.

## 11. Error audit

### 11.1 Theoretical errors explicitly prevented

| Risk | Why it is wrong | Prevention |
|---|---|---|
| finite-grid exactness called continuous exactness | grid monitoring/discretization bias remains | estimand fixed at `L=128`; separate bias theorem |
| positivity called inverse-moment control | positive variables can approach zero too often | B2 lower-tail/negative-moment proof |
| terminal theorem copied to barrier | fine-only monitoring points and active time add terms | separate B6 obligations |
| empirical slope called a theorem | regression does not prove asymptotics | plots are diagnostics only |
| correction rate called complexity | weak bias and cost exponents are missing | B5 triplet required |
| proposal-conditional scalar treated as standard normal | mixture conditioning changes its law | exact likelihood cancellation retained |
| adaptive final reuse | selection can bias the reported mean | pilot/final separation and conditional proof |
| arbitrary multi-dimensional direction weighting | breaks common scalar threshold geometry | V6 remains common rank-one span |

### 11.2 Statistical errors explicitly prevented

| Risk | Prevention |
|---|---|
| testing on 25% events while claiming rare-event performance | calibrated `1e-2/1e-3/1e-4` bands |
| using predicted rather than achieved RMSE | execute integer final allocations |
| post-hoc best baseline | pure CEM fixed as primary comparator |
| dropping censored/failed runs | primary failure and charged retries |
| using reference probability in router | router input schema forbids it |
| reusing confidence bounds at optional looks | fixed look/alpha schedule |
| tuning and confirming on same seeds | separate development, qualification, confirmation namespaces |
| underpowered 20-cluster convention | new paired power analysis |
| ignoring reference uncertainty | explicit combined RMSE tolerance |
| unequal easy-cell dominance | equal-cell-weighted log-work endpoint |

### 11.3 Technical errors explicitly prevented

| Risk | Prevention |
|---|---|
| incomplete work ledger | centralized enumerated work categories |
| checkpoint changes frozen choice | immutable preparation hash and strict parser |
| duplicate random stream under different labels | keyed seed ledger and cross-artifact audit |
| Windows-only success | clean Linux scientific reproduction |
| production auditor shares decision bug | independently implemented audit arithmetic |
| `min(weight)` confused with defensive mass | explicit zero-component index and mass |
| smoke artifact treated as evidence | schema-level smoke prohibition |
| silent config drift | exact config bytes/hash stored |

### 11.4 Residual risks that cannot be removed by planning

- the V6 policy may still lose to a well-tuned CEM at `1e-3` and `1e-4`;
- the selector may remain too conservative even after routing;
- a usable uniform terminal inverse-slope bound may be mathematically difficult;
- barrier mesh enrichment may dominate the hoped-for smoothing rate;
- laptop compute may be sufficient for development but not for powered confirmation;
- a technically valid result may still be judged insufficiently novel by reviewers.

These are research outcomes, not implementation bugs. The plan handles them with
predeclared stop rules and claim reduction rather than hiding them.

## 12. Compute and environment plan

Laptop-first scope:

- all unit/oracle/integration tests;
- calibration pilot and `1e-2/1e-3` references;
- router/selector development;
- theory diagnostics on small-to-medium batches;
- one-cell end-to-end qualification; and
- resource measurement for scaling.

External compute is justified only after G3 and a resource forecast. Rent compute for
independent cluster parallelism, not to compensate for a selector known to waste 90%
of work. Freeze container, dependency lock, CPU/GPU type, thread policy, persistent
storage, checkpoint interval, and shutdown rule before a paid run.

The primary numerical method is CPU-friendly. GPU acceleration is optional for batched
proposal training, but a hardware change must not change the estimand, precision, or
random-stream contract. CPU operation work remains the cross-hardware scientific
metric; wall time is platform-specific.

## 13. Paper outcomes under every branch

| Route A | Route B | Honest outcome |
|---|---|---|
| pass | pass | combined top-tier candidate: exact method, terminal theory, achieved-RMSE evidence |
| pass | conditional/fail | computational paper; finite-grid and conditional theory only |
| fail | pass | theory/falsification paper; no practical superiority claim |
| fail | fail | reproducibility/negative benchmark; not a new superior model paper |

Potential targeting after evidence, not before:

- both pass strongly: SIFIN or Mathematical Finance depending theorem/financial depth;
- A strong with rigorous numerical analysis: SISC or SIFIN;
- B strong but A neutral: a mathematical-finance/numerical-analysis venue matched to
  the theorem's actual scope.

## 14. Definition of done

### Route A done

- frozen genuine rare-event matrix;
- strong matched baselines;
- actual achieved-RMSE execution;
- total-work accounting and censoring rules;
- paired powered primary analysis;
- untouched confirmation;
- independent audit; and
- clean Linux reproduction.

### Route B done

- notation-to-code identity;
- direction regularity;
- terminal slope lower-tail/negative-moment result;
- fine/coarse coefficient rates;
- localized terminal correction theorem;
- correctly delimited bias/complexity corollary; and
- every assumption recorded as proved, cited, or explicitly conditional.

### Combined paper done

- both definitions above;
- novelty matrix updated against closest contemporary work;
- manuscript claim ledger exactly matches theorem and artifact status;
- reproducibility package contains configs, raw sufficient statistics, hashes,
  analysis scripts, environment lock, and a permanent archive; and
- no statement depends on an unreported failed cell, retried run, or development
  choice.

## 15. Immediate implementation backlog

The first implementation cycle should contain only the following bounded items:

1. add strict V6 protocol/cell schemas and checkpoint hardening;
2. expose the true zero-shift defensive mass;
3. implement and oracle-test the pilot-measurable rarity router;
4. add selection work caps and deterministic candidate dominance;
5. create the rare-cell calibration and independent reference configs;
6. produce the notation-to-code ledger and direction tests;
7. run one terminal and one barrier cell through the entire smoke pipeline;
8. run the independent auditor on deliberately corrupted artifacts; and
9. issue a G1--G3 decision before any large experiment.

Only after these items pass should the project spend substantial compute on Route A
qualification or claim progress on Route B's rBergomi theorem.

## 16. Final methodological assessment

This dual-track direction is theoretically coherent because both tracks share the
same exact finite-grid measure-change and coupling, while their conclusions remain
independent. It is technically implementable because it extends the existing V5
modules instead of replacing a validated simulator. It is statistically defensible
because routing, proposal fitting, selection, allocation, and final estimation are
separated and every cost is charged.

It is not possible to certify in advance that the research is error-free or will be
accepted by a top journal. What can be made rigorous is the dependency structure:
every major claim has a proof obligation, a code-level falsification check, a frozen
statistical gate, and a failure action. Under that standard, this is the appropriate
next implementation plan for pursuing both the practical and theoretical paper
paths without overstating the present evidence.

## 17. Evidence inputs for this plan

This plan supersedes performance assumptions in the earlier V5 plan where they
conflict with completed V5 evidence. Its local evidence base is:

- `docs/audits/G11_V5_ACHIEVED_RMSE_QUALIFICATION_V2_DECISION_2026-07-22.md`;
- `docs/audits/G11_V5_FREEZE_READINESS.md`;
- `docs/theory/G11_V5_THEOREMS.md`;
- `docs/theory/G11_V5_PROOF_AUDIT.md`;
- `docs/literature/G11_NOVELTY_MATRIX.md`;
- `docs/literature/G11_BASELINE_SCOPE.md`; and
- `docs/plans/G11_V5_SUBMISSION_GRADE_IMPLEMENTATION_PLAN_2026-07-22.md`.

If later evidence contradicts a numeric planning threshold in this document, the
threshold must be revised in a versioned development protocol before confirmation;
the old result and rationale remain preserved.
