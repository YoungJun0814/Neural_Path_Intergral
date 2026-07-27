# G11 V8 P4 Strong-Baseline Framework

Date: 2026-07-25

Status: common lifecycle and likelihood/cost oracles implemented; no performance
claim

## 1. Scope

P4 standardizes the interface

```text
train(task, budget, seed) -> frozen proposal
plan(task, proposal, pilot_seed) -> frozen integer allocation
estimate(task, proposal, final_seed) -> estimate, variance, work
audit(proposal, plan, estimate) -> pass/fail
```

`freeze_baseline_proposal` is the boundary after a method-specific trainer. It
records the declared training budget, every realized training-cost category, exact
parameters, task identity, training seed, and a canonical SHA-256 digest. P4 does
not replace the existing CEM trainer or pretend that freezing supplied LD/flow
parameters is itself training. Actual fresh task-tuned fits must be executed and
charged in the later experimental phases.

## 2. Required methods

| Method | Proposal law | Independent inferential unit |
|---|---|---|
| crude MC | target Gaussian | path |
| antithetic MC | target Gaussian | antithetic pair mean |
| conditional rBergomi | target residual plus conditional integral | residual path |
| pure CEM | one Gaussian shift | path |
| defensive CEM | positive Gaussian-shift mixture with natural component | path |
| smoothing RQMC | scrambled Sobol target marginals | independent randomization |
| large-deviation subspace IS | Gaussian-shift mixture | path |
| exact-likelihood flow IS | bounded nonlinear triangular coupling flow | path |

Every IS estimator uses an ordinary arithmetic mean of

\[
F(X)\frac{dP}{dQ}(X).
\]

Clipping and self-normalization are prohibited.

## 3. Exact likelihood oracles

For a Gaussian shift \(Q=N(m,I)\),

\[
\log\frac{dQ}{dP}(x)=m^\mathsf Tx-\frac12\|m\|^2.
\]

For a mixture,

\[
\log\frac{dQ}{dP}(x)
=
\log\sum_j\pi_j
\exp\left(m_j^\mathsf Tx-\frac12\|m_j\|^2\right).
\]

The defensive-CEM adapter requires a positive-weight zero-mean component. The
large-deviation adapter may use a nondefensive mixture, but it receives no
likelihood bound merely from passing the density oracle.

RQMC does not define an iid joint proposal density over all net points. Each
scrambled point has the target Gaussian marginal after inverse-CDF transformation,
so no IS likelihood is applied. Dependence is handled by treating one complete
randomization, not one point, as the inferential unit.

## 4. Nonlinear exact flow baseline

Split \(z=(z_a,z_b)\). The implemented bounded triangular coupling is

\[
x_a=z_a+\mu_a,
\]

\[
s(z_a)=s_{\max}\tanh(W_sz_a+b_s),
\qquad
t(z_a)=W_tz_a+b_t,
\]

\[
x_b=\mu_b+\exp(s(z_a))\odot z_b+t(z_a).
\]

Its inverse is explicit:

\[
z_a=x_a-\mu_a,\qquad
z_b=(x_b-\mu_b-t(z_a))\odot\exp(-s(z_a)),
\]

and

\[
\log\left|\det\frac{\partial x}{\partial z}\right|
=\sum_i s_i(z_a).
\]

Thus

\[
\log\frac{q(x)}{p(x)}
=
\frac12(\|x\|^2-\|z\|^2)-\sum_i s_i(z_a).
\]

The bounded log scale prevents an unbounded numerical Jacobian. This entangling
flow is deliberately labelled `baseline_only`; it may not be relabelled as a DCS
extension because its conditional event integral has not been analytically
eliminated.

## 5. Allocation and variance

Pilot and final seeds must be distinct from each other and from the training seed.
For pilot unit variance \(s^2\) and target estimator variance \(v_\star\),

\[
N=\left\lceil\frac{s^2}{v_\star}\right\rceil
\]

is clamped only by predeclared integer minimum and maximum counts.

The final variance is computed across independent units:

- iid methods: individual weighted or conditional contributions;
- antithetic MC: pair averages;
- RQMC: independently scrambled-net estimates.

Using the variance across dependent points inside one RQMC net is prohibited.

## 6. Cost contract

Every artifact explicitly records:

- training and screening samples;
- optimizer steps, hyperparameter trials, and failed restarts;
- pilot/planning raw samples;
- final raw samples;
- likelihood, CDF, and quadrature calls;
- algorithmic work units;
- wall, CPU, and GPU seconds;
- peak memory;
- billed compute cost; and
- metered energy.

For antithetic planning, raw samples equal twice the pair count. For RQMC they equal
randomizations times points per net. If GPU time is nonzero, either billed cost or
metered energy must also be nonzero.

## 7. Audit boundary

The lifecycle audit verifies:

- canonical proposal hash and frozen state;
- task/method/proposal identity through all phases;
- disjoint training, pilot, and final seeds;
- exact likelihood and ordinary mean;
- absence of likelihood clipping and self-normalization;
- correct inferential unit and integer allocation;
- raw pilot/final sample charges;
- likelihood/CDF/quadrature charges;
- training budget compliance;
- finite estimate and exact variance relation;
- flow baseline-only role; and
- heterogeneous compute-cost backing.

Passing these checks means that the baseline can enter a fair experiment. It does
not mean that its trainer is sufficiently tuned, its hyperparameter budget is fair,
or its achieved-RMSE performance is competitive. Those are frozen-matrix and
outcome questions for P5--P10.

## 8. P4 claim boundary

Authorized:

- the common baseline lifecycle contract;
- exact finite-dimensional likelihood oracles for every IS family;
- randomized-QMC and antithetic inferential-unit rules;
- nonlinear exact coupling-flow sampling and density;
- training-inclusive cost-schema audits; and
- construction of the P5 reference and benchmark matrix.

Not authorized:

- superiority of any method;
- a claim that the current one-layer coupling flow is the strongest possible flow;
- a claim that supplied LD/flow parameters constitute fresh training;
- reuse of development-trained proposals in qualification;
- within-net RQMC error bars;
- flow-as-DCS language; or
- omission of training, restart, screening, or accelerator costs.
