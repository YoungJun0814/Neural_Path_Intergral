# Exact Conditional Residual Path-Space Transport Research Plan V12

Date: 2026-08-11
Status: P0--P4 correctness implementation complete; P2 performance gate failed
Short name: **ECRPT**
Primary estimand: a declared finite-grid rare-event probability under a Gaussian
Volterra path law

## 0. Executive decision

V12 preserves the exact part of V10R1 and changes the optimization target.

V10R1 samples a full-path defensive CEM proposal and then integrates one event-driving
Gaussian coordinate.  V12 first performs the exact conditional integration and trains
the proposal only on the residual path integral that remains.  The core estimator is

\[
X=Ae+R,\qquad
g_\lambda(R)=P(F_\lambda(X)=1\mid R),\qquad
\widehat p=\frac1n\sum_{i=1}^n
g_\lambda(R_i)\frac{p_R(R_i)}{q_{\theta,\lambda}(R_i)}.
\]

The proposal is defensive,

\[
q_{\theta,\lambda}=\delta p_R+(1-\delta)\widetilde q_{\theta,\lambda},
\qquad \delta>0,
\]

and every reported estimate is an ordinary, non-self-normalized mean with the exact
residual likelihood.  A neural network may amortize deterministic proposal generation
over tasks, but it is not allowed to replace the likelihood or conditional integral.

The implementation is fail-closed.  Completing software does not authorize a positive
performance claim.  Development, qualification, and submission are separate states.

### 0.1 Implementation outcome (2026-08-11)

The software-completion branch in Section 13 is complete.  P0--P4 modules, three
development namespaces, exact result artifacts, and independent audits are present.
All three P2 variants pass exactness, paired-identity, and likelihood-normalization
gates.  All three fail the frozen performance gate.  Consequently P3/P4 are available
only as correctness infrastructure, qualification is not authorized, and no positive
performance or journal-submission claim follows from V12.

The tested proposal variants are:

1. one-pass residual Gaussian-mixture cross entropy (V1);
2. adaptive exact-importance-weighted residual CE (V2);
3. annealed `g^beta` bridge CE with `beta=0.25,0.5,0.75,1` (V3).

The retained nonlinear coupling flow is exact but not yet run in the 384-dimensional
matrix because its first dense implementation has quadratic dimension cost.  This is
an explicit optimization target, not hidden positive evidence.

## 1. Research questions

### 1.1 Mathematical question

Can exact conditional elimination of an event-driving Gaussian coordinate be combined
with a tractable residual path-space transport such that proposal approximation error
has an explicit estimator-variance interpretation?

For a nonnegative event integrand, the ideal residual density is

\[
q_\lambda^*(r)=g_\lambda(r)p_R(r)/p_\lambda,
\]

and the desired exact identity is

\[
\operatorname{Var}_{q}(Y)/p_\lambda^2
=\chi^2(q_\lambda^*\Vert q).
\]

For a signed multilevel correction `G_l`, the second-moment-optimal density is instead
proportional to `|G_l| p_R`.  A signed correction may not be used as a probability
density.

### 1.2 Numerical question

Does training on the conditional residual target reduce training-inclusive work versus
all of:

1. target-law conditional Monte Carlo;
2. smoothing RQMC;
3. raw defensive CEM;
4. V10R1 full-path CEM followed by DCS;
5. residual natural sampling;
6. a simple regression/interpolation amortization baseline?

### 1.3 Practical question

Can one task-conditioned proposal generator be reused across thresholds and model
parameters, with exact per-task likelihoods, so that amortization means different risk
queries rather than repeated evaluation of one unchanged query?

## 2. Claims and prohibited claims

### 2.1 Claims eligible after their gates pass

- finite-dimensional unbiasedness conditional on frozen training;
- unconditional unbiasedness under training/evaluation independence;
- exact defensive residual likelihood and pathwise likelihood bound;
- exact Rao--Blackwell identity against the paired raw estimator;
- exact optimal-residual-density and chi-square relative-variance identities;
- exact finite-grid scalar thresholds for explicitly supported tasks;
- exact common-coordinate finite-grid ML telescoping;
- measured training-inclusive improvement on a frozen, uncensored benchmark;
- task-amortized improvement only for declared in-distribution and held-out task sets.

### 2.2 Claims prohibited without additional proof

- continuous-monitoring unbiasedness from a finite monitoring grid;
- novelty of conditional Monte Carlo, line integration, CEM, MLMC, or neural IS alone;
- a zero-variance ML correction proposal based on a signed density;
- exactness after path-dependent direction selection on evaluation samples;
- a generic feedback-control likelihood when only deterministic Gaussian shifts are
  implemented;
- a universal rough-Bergomi rate inferred from fitted slopes;
- best-baseline superiority when a comparator is inaccurate or resource-censored;
- quantum amplitudes or Feynman terminology as a mathematical contribution.

## 3. Frozen mathematical convention

### 3.1 Target coordinates

For each finite grid and task, the simulator consumes a declared standard Gaussian
vector `X in R^d`.  For the single-grid BLP rBergomi adapter, `d=3N`.  Raw Brownian
increments and standardized latent coordinates may not be interchanged.

### 3.2 Integrated coordinate

`e` is a deterministic unit vector.  In rBergomi it has zero entries in every local
Volterra coordinate and strictly positive entries only in the independent price block.
Write

\[
A=e^T X,\qquad R=(I-ee^T)X.
\]

Under the target law, `A` is standard normal, `R` is a standard Gaussian on the
orthogonal subspace, and they are independent.

### 3.3 Residual proposal

The first exact implementation is a mixture of translations on the residual subspace:

\[
q_R=\sum_{j=0}^{J-1}\pi_j N_R(m_j,I_R),
\qquad e^T m_j=0,
\]

with `m_0=0` and `pi_0=delta`.  Its exact density ratio on the residual Gaussian
reference measure is

\[
q_R(r)/p_R(r)=\sum_j\pi_j
\exp(m_j^T r-\|m_j\|^2/2).
\]

The implementation must reject nonorthogonal means rather than silently project a
frozen proposal during evaluation.

### 3.4 Training

Training-only target samples may fit the residual proposal by weighted cross entropy:

\[
\min_\theta
-\frac{\sum_i g(R_i)\log(q_\theta(R_i)/p_R(R_i))}
       {\sum_i g(R_i)}.
\]

The unknown probability normalizer is unnecessary.  Direction selection uses a
separate training stream and a predeclared finite family of strictly positive price
directions.  The selected direction and fitted proposal are frozen before evaluation.

### 3.5 Multilevel correction

Fine and coarse events are evaluated on one fine latent probability space and one
common scalar coordinate.  For

\[
G_l(R)=\Phi(\tau_l(R))-\Phi(\tau_{l-1}(R)),
\]

the proposal-training weight is `|G_l|`, while the estimator retains the sign.  One
fine residual likelihood weights the entire correction.

## 4. Phase P0: repair comparison and allocation

### Deliverables

- streaming Chan/Welford sufficient statistics;
- a bounded, one-sided second-moment upper confidence bound;
- tail-safe pilot allocation based on a variance upper bound rather than plug-in
  variance alone;
- method-specific minimum final units;
- at least 16 independent RQMC randomizations whenever an RQMC variance is reported;
- an independent final certificate that cannot accept zero empirical variance as
  proof of zero estimator variance;
- explicit resource-censoring before an oversized final allocation is attempted.

### Mathematical safety rule

For `Y in [a,b]`, apply a one-sided Hoeffding bound to `Y^2` and combine it with the
range variance bound.  This is intentionally conservative.  A method that cannot
meet the requested accuracy within the resource cap is censored, not assigned a false
zero standard error.

### Gate P0

- zero pilot/final contributions give a strictly positive uncertainty upper bound;
- streaming and monolithic sufficient statistics agree to floating-point tolerance;
- chunking never changes inferential-unit semantics;
- RQMC points are never treated as independent replicates;
- seed namespaces for training, pilot, final, and audit are disjoint;
- the old four-unit pathology is reproduced as a negative test and rejected.

## 5. Phase P1: generic residual transport

### Deliverables

- validated residual Gaussian-mixture specification;
- exact sampling on an orthogonal Gaussian subspace;
- exact component and balance-mixture log likelihoods;
- stable nonnegative and signed contribution builders;
- defensive-bound diagnostics;
- weighted-cross-entropy fitting with a natural component retained;
- canonical serialization and proposal hash;
- analytic Gaussian half-space oracle.

### Gate P1

- direction norm and residual orthogonality errors `<=1e-11` in float64;
- density reconstruction error `<=1e-11`;
- likelihood normalization agrees with one within `4 SE` on independent samples;
- estimator agrees with the analytic probability within `4 SE`;
- paired raw-minus-conditional mean agrees with zero within `4 SE`;
- invalid weights, missing defensive component, nonorthogonal means, NaNs, and
  self-normalized requests fail closed.

## 6. Phase P2: rBergomi ECRPT adapter

### Deliverables

- terminal, discrete-barrier, and hit-plus-occupation finite-grid conditional values;
- training-only positive direction selection;
- residual-target mixture fitting;
- paired raw/ECRPT evaluation under the exact same residual proposal;
- full latent and path reconstruction diagnostics;
- log-domain conditional contributions;
- training, selection, likelihood, CDF, simulation, and evaluation work ledgers.

### Direction candidates

The initial family contains only deterministic positive price-block directions:

- flat;
- front-loaded exponential directions;
- back-loaded exponential directions;
- positive power-law directions.

No volatility coordinate may enter the integrated direction.  Candidate choice is
made only on a dedicated training stream.

### Development micro-study

Use three representative terminal cells spanning roughness and rarity.  Each cell has
independent training and evaluation clusters.  Compare residual natural sampling,
one-shift ECRPT, multi-shift ECRPT, V10R1 full CEM+DCS, conditional rBergomi, smoothing
RQMC, and defensive CEM where the shared protocol is applicable.

### Gate P2

All correctness requirements are mandatory.  Performance advancement additionally
requires:

- no inaccurate or resource-censored primary comparator in the gate set;
- geometric best-primary/ECRPT total-work ratio `>1.5`;
- one-sided 95% lower ratio `>1.0`;
- ECRPT beats its paired raw estimator;
- benefit appears in at least two of three representative cells;
- all training and direction-selection work is charged.

Failure closes the broad performance branch.  It does not invalidate exactness.

## 7. Phase P3: task-conditioned amortization

This phase is performance-authorized only after P2, but its correctness infrastructure
may be implemented and oracle-tested independently.

### Architecture

A small deterministic network maps normalized task/model features to residual mixture
translations and, optionally, positive direction logits.  The output is projected in
the declared residual subspace before it is frozen.  Evaluation uses only the frozen
numeric proposal and recomputes its exact density.

### Comparators

- per-task residual fitting;
- nearest-neighbor reuse;
- linear/ridge interpolation;
- per-task CEM;
- one shared neural generator.

### Gate P3

- exactness is invariant to generator architecture;
- held-out tasks were never used for training or normalization fitting;
- amortized total work includes bank construction and network fitting;
- performance is reported separately in-distribution and out-of-distribution;
- no extrapolation claim is made from interpolation-only evidence.

## 8. Phase P4: residual MLMC

### Deliverables

- adjacent BLP latent contract;
- common positive fine-price coordinate;
- exact fine/coarse scalar thresholds;
- stable signed CDF differences;
- `|G_l|`-weighted residual proposal training;
- common residual likelihood for the correction;
- level-zero plus adjacent-correction sampler compatible with the existing MLMC
  engine;
- exact finite-grid telescoping oracle.

### Gate P4

- fine and coarse coordinates agree to `1e-11`;
- pathwise hard corrections equal scalar-coordinate corrections;
- target means telescope on analytic and low-dimensional numerical oracles;
- no signed density is constructed;
- fitted rate claims remain descriptive until their model assumptions are proved.

## 9. Phase P5: theory strengthening

The top-journal mathematical route requires at least one nontrivial result beyond the
generic optimal-IS identity:

1. a stability bound translating residual transport error into estimator work;
2. a strict improvement condition for a restricted proposal family;
3. a rough Gaussian-Volterra threshold-rate theorem;
4. a mesh-stability theorem for task-conditioned residual transports;
5. a lower-bound/nondegeneracy result for the ML correction exponent.

The first implementation may state only finite-dimensional exactness.  Empirical
slopes do not promote an open model-rate premise.

## 10. Phase P6: frozen evidence

Only after development gates pass:

1. freeze cells, thresholds, seeds, source hash, proposal budgets, and work metric;
2. execute an uncensored qualification matrix;
3. independently audit every artifact without importing experiment aggregation code;
4. freeze a new confirmation namespace;
5. reproduce on independent physical hardware;
6. report failures, training dispersion, and break-even query counts;
7. decide the manuscript claim from the predeclared gates.

## 11. Required tests

### Unit and property tests

- subspace projection and sampling;
- mixture density and defensive bound;
- weighted fitting determinism and seed sensitivity;
- hash mutation detection;
- scalar threshold equivalence;
- log-tail stability;
- signed correction handling;
- streaming merge associativity to tolerance;
- allocation censoring and zero-variance rejection;
- task feature validation and held-out separation;
- checkpoint/resume if long runs are introduced.

### Negative tests

- direction contains a volatility coordinate;
- direction has a nonpositive price entry;
- frozen mean is changed during evaluation;
- evaluation samples are reused for direction or proposal fitting;
- residual mean is not orthogonal;
- natural component is absent;
- signed correction is normalized as a density;
- coarse and fine use different integrated coordinates;
- final zero sample variance certifies a positive rare probability;
- RQMC Sobol points are counted as independent units.

## 12. Error register

| Risk | Consequence | Prevention |
|---|---|---|
| direction enters the volatility block | scalar Gaussian threshold becomes false | adapter-level support check |
| frozen proposal projected after training | evaluated law differs from trained law | orthogonality rejection and hash |
| hard-event CEM cost excluded | false practical speedup | complete training ledger |
| plug-in variance from a rare pilot | severe underallocation | bounded second-moment UCB |
| final sample variance is zero | false target attainment | independent bounded certificate |
| signed correction used as density | invalid proposal | fit to absolute correction |
| path-dependent proposal under static formula | wrong likelihood | deterministic-task-only exact branch |
| direction selected on evaluation data | selection bias | disjoint seed roles |
| one task repeated and called amortization | false practical claim | held-out multi-task benchmark |
| finite grid called continuous | wrong estimand | immutable task metadata |
| fitted slope called a theorem | false complexity claim | premise provenance ledger |

## 13. Completion definition

Software completion requires all P0--P4 correctness modules, documentation, tests, a
laptop-sized development execution, and an independent result reconstruction.  A
positive performance completion additionally requires the P2 gate.  Qualification,
confirmation, and top-journal authorization are deliberately impossible to grant from
the development namespace.

The immediate execution order is:

1. freeze this plan and the theorem/claim contracts;
2. implement and test P0;
3. implement and oracle-test P1;
4. implement and test the rBergomi P2 adapter;
5. run the micro-study and apply its gate;
6. implement P3/P4 correctness infrastructure without overriding a failed P2 gate;
7. run the complete regression/static audit;
8. write the final development decision report.
