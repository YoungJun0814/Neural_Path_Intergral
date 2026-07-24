# G11 V8 Claim and Estimand Contract

Date: 2026-07-25

Status: P0 development contract; not an outcome freeze

Machine-readable source:
`configs/g11_v8/top_journal_claim_contract_v1.yaml`

## 1. Research object

The underlying financial model is rough Bergomi. The research contribution is an
estimator of rare path-event probabilities, not a replacement stochastic-volatility
model.

The current primary target is

\[
p_{\theta,n}(A)
=P_\theta(X^{(n)}\in A),
\qquad n=128,
\]

where \(X^{(n)}\) is the declared finite-grid path and \(A\) is a terminal or
discrete-barrier event. This is not the continuously monitored probability
\(P_\theta(X\in A_{\mathrm{continuous}})\).

## 2. Primary paper claim

The permitted primary claim is:

> For predeclared finite-grid rare terminal and discrete-barrier events under rough
> Bergomi, exact defensive-mixture DCS preserves the target expectation, removes the
> integrated-coordinate conditional variance, and is tested against predeclared
> strong comparators using achieved-RMSE, training-inclusive work.

The claim contains three contributions only:

1. exact defensive-mixture conditional path integration;
2. a model-level strict-improvement or mesh/rate theorem; and
3. frozen strong-baseline total-work evidence.

## 3. Exact estimator contract

Let

\[
Q=\sum_{j=0}^{J-1}\pi_j N(m_j,I),
\qquad m_0=0,\quad \pi_0=\delta>0.
\]

The likelihood is the target density divided by the complete mixture density:

\[
L(x)
=\left[
\sum_j\pi_j
\exp\left(m_j^\mathsf T x-\frac12\|m_j\|^2\right)
\right]^{-1}.
\]

The raw contribution is

\[
Y_{\mathrm{raw}}=L(X)\mathbf 1_A(X),
\qquad X\sim Q.
\]

For \(X=UZ+R\), the DCS contribution must satisfy

\[
Y_{\mathrm{DCS}}(R)=E_Q[Y_{\mathrm{raw}}\mid R].
\]

The proposal conditional distribution \(Q(Z\mid R)\) is generally a
residual-dependent mixture. It may not be replaced by an unproved standard Gaussian
law. The implemented analytic expression must follow from the exact likelihood
cancellation.

## 4. Comparator contract

- `fixed_raw_defensive` isolates the DCS mechanism under the identical proposal.
- `task_tuned_pure_cem` is a primary external adaptive-work comparator.
- `numerical_smoothing_rqmc` is a primary closest-method comparator.
- crude/antithetic MC, defensive CEM, large-deviation adaptive IS, and
  exact-likelihood flow IS are mandatory secondary comparators.

A baseline is not removed because it fails, reaches a work cap, or performs poorly.
Hyperparameter searches and failed retries are charged.

## 5. Inferential contract

- Independent seed clusters are the inferential units.
- Cells are equally weighted inside clusters for primary geometric effects.
- Paths are Monte Carlo samples, not inferential replicates.
- Multiple primary baselines require simultaneous one-sided confidence intervals.
- Reference uncertainty is included in achieved-error assessment.
- Clopper--Pearson attainment bounds may be called exact.
- Bootstrap RMSE bounds must be called nominal or approximate.

## 6. Claim promotion ladder

| Level | Permitted claim | Requirement |
|---|---|---|
| C0 | code targets declared finite-grid event | oracle and schema tests |
| C1 | finite-grid estimator is exact | likelihood and conditional-mean proof |
| C2 | DCS cannot increase variance | Rao--Blackwell proof |
| C3 | DCS strictly improves under stated conditions | nondegeneracy theorem |
| C4 | DCS has a model-level mesh/correction rate | threshold and mesh proof |
| C5 | end-to-end MLMC complexity improves | separate bias, variance, cost exponents |
| E1 | externally competitive at achieved RMSE | frozen total-work comparison |
| E2 | repeated-query advantage is reproducible | query-count curve and physical reproduction |
| TJ | top-journal package | C3 or stronger, E1, E2, novelty and external review |

No higher claim is allowed when a required lower claim is conditional or failed.

## 7. Prohibited claims

Until independently proved, the project may not claim:

- unbiased continuous-barrier estimation;
- universal rough Bergomi optimality;
- unconditional rBergomi MLMC complexity;
- superiority over all importance samplers;
- a neural-architecture contribution;
- a quantum or Feynman path-integral result;
- a successful hybrid router;
- independent physical reproduction from the current same-laptop Windows/WSL result;
  or
- exact coverage for percentile-bootstrap RMSE bounds.

## 8. P0 decision

The V8 route is authorized to implement theory and comparator infrastructure. No V8
qualification or confirmation outcome is authorized by this contract. Those phases
require later independent gates and a clean outcome-blind freeze.
