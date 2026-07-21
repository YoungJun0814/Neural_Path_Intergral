# G11 Novelty Matrix

Date: 2026-07-22
Status: primary-source audit updated through the full 2026 v2 occupation-time MLIS
paper; external expert review and citation update remain required before submission

## 1. Candidate novelty after audit

The literature already contains conditional/numerical smoothing, MLMC for
discontinuous functionals, multiple-importance-sampling balance denominators,
importance-sampled MLMC, rough-process simulation, rough-volatility MLMC, and a 2025
common-likelihood MLIS method for occupation time.

Therefore the defensible candidate contribution is limited to:

> Exact removal of an event-driving deterministic Gaussian proposal-control span from
> a defensive balance mixture, giving a residual mixture likelihood with a pathwise
> bound, combined with an exact common-coordinate scalar-threshold correction in a
> rough Gaussian-Volterra hierarchy.

Neither “common likelihood,” “occupation-time MLIS,” “conditional smoothing,” nor
“MLMC under rough volatility” is independently novel.

## 2. Comparison matrix

Legend: `yes` means the cited work contains the feature in a materially relevant
form; `partial` means related but not the same contract; `no` means it is not a stated
feature of the audited method.

| Work | Model class | Discontinuous/path event | IS mixture | Event-driving span integrated | Exact residual mixture likelihood | Common fine/coarse likelihood | ML correction | Rate/complexity theorem | Preprocessing cost | Code |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Giles (2008), MLMC path simulation | SDE discretizations | partial | no | no | no | n/a | yes | yes | no | yes |
| Giles/financial conditional-expectation MLMC line | Markov SDE options | yes | no | partial, payoff conditioning | no | n/a | yes | yes | no | yes |
| Bayer, Ben Hammouda, Tempone (2024), numerical smoothing MLMC | discretized SDEs | yes | no | one numerical smoothing variable | no | n/a | yes | yes | no | not confirmed |
| Bayer, Ben Hammouda, Tempone (2023), smoothing + ASGQ/QMC | option-pricing integrals | yes | no | one numerical smoothing variable | no | n/a | no/partial hierarchy | smoothness analysis | no | not confirmed |
| Sbert and Elvira (2022), generalized balance heuristic | generic Monte Carlo integrals | generic | yes | no | balance denominator, not span marginal | no | no | MIS variance theory | no | not confirmed |
| Ben Rached et al. (2025), multilevel IS for McKean--Vlasov | McKean--Vlasov SDE | rare Lipschitz observable | no defensive Gaussian mixture | no | no | partial shared controls | yes | yes | partial | not confirmed |
| Ben Amar, Ben Rached, Tempone (2025), hierarchical IS for occupation time | Markov SDE + occupation state | yes | no defensive Gaussian mixture | no | no | **yes** | yes | yes, including MLIS/SLIS condition | **yes** | not confirmed |
| Bennedsen, Lunde, Pakkanen (2017), hybrid scheme | Brownian semistationary/rough Bergomi | no rare-event estimator | no | no | no | n/a | no | strong MSE asymptotics | no | partial/public implementations exist |
| Bourgey and De Marco (2025 version), rough-Bergomi VIX MLMC | Gaussian rough forward variance | Lipschitz VIX option | no | no | no | n/a | yes | yes | no | not confirmed |
| Internal G9 MGVS | BLP rBergomi finite grid | hit + occupation | deterministic IS | arbitrary one direction | residual full mixture remains | yes | yes | no general theorem | selection excluded from headline | yes |
| Internal G10 DCS-MGI | BLP rBergomi finite grid | hit + occupation | defensive natural/CEM mixture | complete rank-one price-control span | **yes** | **yes** | adjacent correction only | finite-grid identities; no rate theorem | calibration separated | yes |
| Proposed G11 | deterministic Gaussian-shift mixtures + rough-Volterra adapter | scalar-threshold path events | defensive balance mixture | **yes** | **yes, bounded** | **yes** | full MLMC | conditional `h^r -> h^(2r)` upper-bound theorem | **yes** | planned release |

## 3. Closest-work notes

### 3.1 Bayer, Ben Hammouda, and Tempone: numerical smoothing MLMC

Primary source: <https://arxiv.org/abs/2003.05708>
Published version: SIAM Journal on Scientific Computing 46(3), 2024.

Audited overlap:

- discontinuous probabilities and densities;
- root finding in one selected random variable;
- one-dimensional numerical integration;
- MLMC correction smoothing;
- variance-decay, robustness, and complexity analysis.

Non-overlap candidate:

- G11 analytically integrates a proposal-shift span inside a defensive Gaussian
  balance mixture;
- G11 derives a residual target-over-mixture likelihood and defensive bound;
- the rBergomi specialization has a closed scalar CDF rather than generic root finding
  plus numerical quadrature.

Required baseline action: reproduce an applicable published example and implement a
faithful root/integration baseline for a shared task where assumptions allow.

### 3.2 Ben Amar, Ben Rached, and Tempone: occupation-time MLIS

Primary source: <https://arxiv.org/abs/2509.13950>

Audited overlap:

- rare occupation-time estimation;
- stochastic-control/HJB importance sampling;
- SLIS and MLIS;
- common-likelihood fine/coarse construction;
- preprocessing cost in total work;
- conditions under which MLIS beats SLIS;
- extension of MLMC complexity analysis.

Consequences for G11:

- common likelihood is prior art;
- occupation-time MLIS is prior art;
- preprocessing-inclusive comparison is expected, not optional;
- this paper is a mandatory direct baseline/related-work reference.

Non-overlap candidate:

- G11 uses a defensive mixture of deterministic Gaussian path shifts instead of an
  HJB feedback-control proposal;
- G11 analytically eliminates the event-driving control span and retains an exact
  residual balance-mixture likelihood;
- G11 targets non-Markovian Gaussian Volterra dynamics and a hit-plus-occupation
  event, with no finite-dimensional HJB assumption.

The 2026 v2 paper's estimator equations and complexity propositions were reviewed.
Its fine HJB control is reused on the coarse path, coarse increments are aggregated
from the fine increments, and one fine likelihood weights the fine-minus-coarse
observable. It also gives an explicit MLIS-versus-SLIS work condition. Consequently
common likelihood, shared-control coupling, and the crossover condition are confirmed
prior art rather than provisional novelty.

Required baseline action: compare DCS-MLMC against DCS-SLIS on the same finest-grid
estimand and proposal, and against task-tuned SLIS with all training cost included.

### 3.3 Multiple importance sampling

Primary source used for the modern balance-heuristic boundary:
<https://arxiv.org/abs/1903.11908>.

Audited overlap:

- randomized and deterministic mixture sampling;
- mixture density in the denominator;
- unbiased balance-heuristic estimators;
- variance analysis across proposal combinations.

Non-overlap candidate:

- orthogonal Gaussian marginalization of a selected proposal-shift subspace;
- a resulting lower-dimensional mixture ratio;
- coupling that conditional estimator to a multilevel scalar-threshold correction.

### 3.4 Rough simulation and rough MLMC

Primary sources:

- hybrid scheme: <https://doi.org/10.1007/s00780-017-0335-5>;
- rough-Bergomi VIX MLMC: <https://arxiv.org/abs/2105.05356>.

Audited overlap:

- non-Markovian Gaussian Volterra structure;
- discretization and strong-error analysis;
- multilevel complexity in rough models.

Non-overlap candidate:

- rare discontinuous path-event estimation under a defensive mixture;
- exact removal of the price-control span;
- residual likelihood and scalar-threshold correction.

The VIX MLMC paper concerns an integrated forward-variance functional and exact
correlated Gaussian sampling; it does not establish the G11 spot-path threshold rate.

## 4. Claims allowed by the current matrix

Allowed only as candidate language before external review:

- “We derive an exact residual balance-mixture likelihood after analytically
  eliminating the event-driving deterministic Gaussian control span.”
- “The natural proposal component yields a pathwise bound on the full and residual
  likelihoods.”
- “For a common scalar-threshold correction, the conditional second-moment upper bound
  depends quadratically rather than linearly on the threshold discrepancy.”
- “We specialize the construction to a non-Markovian rough-Volterra finite-grid
  simulator.”

## 5. Claims prohibited by the current matrix

- first occupation-time MLIS method;
- first common-likelihood MLMC importance sampler;
- first smoothing method for discontinuous MLMC payoffs;
- first MLMC method in rough Bergomi;
- first balance-mixture importance sampler;
- first integration over a Gaussian direction;
- continuous-time exactness;
- superior asymptotic complexity before threshold and bias rates are proved.

## 6. Remaining literature actions before submission

1. Review citations of the 2024 SIAM numerical-smoothing paper through the submission
   cutoff date.
2. Search specifically for Gaussian-mixture Rao--Blackwellization and marginal
   importance sampling over proposal parameters/subspaces.
3. Review adaptive importance-sampled MLMC for diffusions and statistical Romberg.
4. Search rough-volatility barrier, occupation, and rare-event simulation papers.
5. Ask an external expert to challenge the novelty statement before journal targeting.
6. Record search strings, databases, dates, inclusion criteria, and excluded near
   matches in a reproducible literature appendix.

## 7. M0 novelty decision

**Conditional pass.** No audited source has yet been found that combines all of:

1. a defensive deterministic Gaussian balance mixture;
2. exact marginalization of its event-driving shift span;
3. an exact bounded residual mixture likelihood;
4. a common-coordinate scalar-threshold ML correction;
5. a rough Gaussian-Volterra application.

However, the 2025 occupation-time common-likelihood MLIS paper is close enough that
novelty remains provisional until its full derivation is audited line by line. G11 may
proceed to theorem-oracle implementation, but a submission-level novelty claim is not
yet authorized.

## 8. Primary references

1. M. B. Giles, “Multilevel Monte Carlo Path Simulation,” Operations Research 56(3),
   2008. <https://doi.org/10.1287/opre.1070.0496>
2. C. Bayer, C. Ben Hammouda, R. Tempone, “Multilevel Monte Carlo with Numerical
   Smoothing for Robust and Efficient Computation of Probabilities and Densities,”
   SISC 46(3), 2024. <https://arxiv.org/abs/2003.05708>
3. C. Bayer, C. Ben Hammouda, R. Tempone, “Numerical Smoothing with Hierarchical
   Adaptive Sparse Grids and Quasi-Monte Carlo Methods for Efficient Option Pricing,”
   Quantitative Finance 23(2), 2023. <https://arxiv.org/abs/2111.01874>
4. M. Sbert, V. Elvira, “Generalizing the Balance Heuristic Estimator in Multiple
   Importance Sampling,” Entropy 24(2), 2022.
   <https://arxiv.org/abs/1903.11908>
5. N. Ben Rached et al., “Multilevel Importance Sampling for Rare Events Associated
   With the McKean--Vlasov Equation,” Statistics and Computing, 2025.
   <https://arxiv.org/abs/2208.03225>
6. E. Ben Amar, N. Ben Rached, R. Tempone, “Hierarchical Importance Sampling for
   Estimating Occupation Time for SDE Solutions,” 2025.
   <https://arxiv.org/abs/2509.13950>
7. M. Bennedsen, A. Lunde, M. S. Pakkanen, “Hybrid scheme for Brownian
   semistationary processes,” Finance and Stochastics 21, 2017.
   <https://doi.org/10.1007/s00780-017-0335-5>
8. F. Bourgey, S. De Marco, “Multilevel Monte Carlo simulation for options in the
   rough Bergomi model,” revised 2025. <https://arxiv.org/abs/2105.05356>
