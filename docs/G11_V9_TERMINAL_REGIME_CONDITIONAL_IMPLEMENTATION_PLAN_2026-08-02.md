# G11 V9 Terminal Regime-Conditional Implementation Plan

Date: 2026-08-02

Status: **frozen new development protocol; not a continuation or relabeling of V8**

## 1. Scientific objective

V9 tests a narrower statement than V8.  The estimand is the finite-grid terminal
left-tail probability under the fixed 128-step rBergomi simulator.  The candidate
estimator is a defensive Gaussian-mixture importance sampler followed by exact
proposal-conditional integration of one event-driving Gaussian coordinate (DCS).

The scientific question is not whether DCS beats every method in every path task.
It is whether there are predeclared Hurst regimes in which DCS:

1. retains exact likelihoods and an ordinary, unbiased finite-grid mean;
2. reduces total work relative to the raw estimator using the same mixture; and
3. remains practically competitive with conditional lognormal MC, smoothed RQMC,
   and defensive CEM after all training, pilot, diagnostic, and final work is charged.

Barrier events, continuum unbiasedness, a uniform factor-two claim, and a submission
claim are outside V9.

## 2. Why this is a valid new protocol

V8 is design information.  Its results motivated the terminal-only scope and the
abandonment of the uniform factor-two boundary.  V8 records are not confirmation
data.  V9 uses a new claim contract, fresh reference estimates, a newly trained
proposal bank, disjoint seeds, and separate development and qualification outputs.

The twelve fixed events cover the Cartesian product

\[
H\in\{0.05,0.12,0.20\},\qquad
p_{design}\in\{10^{-2},10^{-3},10^{-4},10^{-5}\}.
\]

The numerical thresholds are inherited disclosed design constants, not claimed to
be exact population quantiles.  Every accuracy statement concerns the resulting
fixed-threshold probability.

## 3. Mathematical and cost contract

For an inferential unit with variance \(v\), work \(c\), target estimator variance
\(\tau^2\), one-time work \(C_0\), and repeated-query count \(K\), V9 reports

\[
n_\tau=\max\{2,\lceil v/\tau^2\rceil\},\qquad
W_K=C_0/K+n_\tau c.
\]

For IID estimators a unit is one path.  For RQMC it is one independently scrambled
Sobol randomization, never one Sobol point.  Proposal training, allocation pilots,
and likelihood diagnostics belong to \(C_0\).  The primary value is frozen at
\(K=100\); \(K=1,10,1000\) are descriptive sensitivity points.

For the same-mixture mechanism comparison, the efficiency ratio is
\(W_{raw}/W_{DCS}\).  For an external method \(m\), it is \(W_m/W_{DCS}\).  A value
above one favors DCS.  The best-primary ratio uses the smallest external work in
each cell/cluster, which is conservative for DCS.

## 4. Statistical unit and regime rule

Independent execution clusters are the inferential units.  The four fixed rarity
cells inside one Hurst group are averaged on the log-ratio scale inside each
cluster; they are not falsely treated as independent replications.  A one-sided
Student-t lower confidence bound is then computed across clusters.

Development selects an H group only if all correctness/resource gates pass and:

- geometric \(W_{raw}/W_{DCS}\ge1.10\) with lower bound at least 1.00;
- geometric \(W_{best}/W_{DCS}\ge0.80\) with lower bound at least 0.67.

Qualification repeats the same thresholds on fresh seeds and applies Bonferroni
coverage across the selected H groups.  Every selected group must pass.  Failure
closes V9; thresholds and groups may not be changed inside the namespace.

## 5. Execution phases and stop rules

### V9-P0: contract and implementation freeze

- audit the terminal-only roster and forbidden claims;
- unit-test work-to-target arithmetic and cluster-level confidence bounds;
- commit implementation before any V9 outcome is generated.

### V9-P1: independent reference

- generate new smoothed randomized-QMC references under the target law;
- retain every randomization-level value;
- independently reconstruct means, standard errors, seeds, and work;
- stop if any reference fails its predeclared relative-SE gate.

### V9-P2: proposal bank

- train every cell independently with fresh CEM seeds;
- average frozen replicate controls and form a defensive rank-one mixture;
- charge failed/used iterations and all training paths;
- audit schedules, weights, seeds, hashes, and cost conservation.

### V9-P3: development falsification

- execute paired raw/DCS and all three primary baselines on all twelve cells;
- use four independent clusters and fresh seeds;
- reconstruct target-work at every K;
- select H groups only through the frozen algorithm;
- stop if no group passes or any correctness/resource gate fails.

### V9-P4: conditional qualification

- run only automatically selected H groups;
- use eight new independent clusters and no refitting;
- apply the Bonferroni-adjusted frozen gate;
- authorize only a regime-conditional empirical claim if every selected group passes.

### V9-P5: final audit and closure

- independently recompute hashes, seed sets, work, aggregates, and decisions;
- run Ruff, mypy, focused tests, full pytest, and `git diff --check`;
- document a pass or falsification without weakening a gate;
- keep top-journal and submission authorization false pending external proof and
  novelty review.

## 6. Known limitations retained on purpose

This protocol cannot establish a continuum-time theorem, a barrier result, uniform
bounded relative error, or novelty by computation alone.  Timing is hardware-local;
algorithmic work is the primary portable cost.  Selection uses development data,
so only the fresh qualification stage can support the narrower empirical claim.
