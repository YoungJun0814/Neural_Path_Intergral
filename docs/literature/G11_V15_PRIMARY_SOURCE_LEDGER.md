# G11 V15 primary-source novelty ledger

Date: 2026-08-11

Status: internal primary-source audit complete enough to authorize development;
external novelty review pending; submission and qualification claims locked.

## Search protocol

The audit used journal/publisher pages, arXiv records and source manuscripts. Search
families were:

- `rough Bergomi rare event importance sampling conditional Monte Carlo`;
- `Gaussian Volterra rare event Cameron Martin importance sampling`;
- `large deviation adaptive importance sampling likelihood informed subspace`;
- `neural importance sampling option pricing Cameron Martin`;
- `non-Markovian stochastic control importance sampling neural rough volatility`;
- `Gaussian measure low-rank covariance perturbation infinite dimensions`.

Searches were performed on 2026-08-11. Secondary aggregators were used only to find
primary records and do not support novelty decisions.

## Decisive prior art

| ID | Primary work | What is already prior art | Consequence for V15 |
|---|---|---|---|
| L1 | Bayer, Friz, Gatheral, *Pricing under rough volatility* (2016), DOI `10.1080/14697688.2015.1099717` | rBergomi model and its financial motivation | The model itself is not a contribution. |
| L2 | McCrickerd, Pakkanen, *Turbocharging Monte Carlo pricing for the rough Bergomi model* (2018), DOI `10.1080/14697688.2018.1459812` | conditional lognormal estimator, antithetic/control-variate acceleration, runtime-adjusted variance | Integrating the independent price driver and conditional Black--Scholes evaluation are not novel. |
| L3 | Bayer, Ben Hammouda, Tempone, *Numerical Smoothing ...* (2023/2024), DOI `10.1137/22M1495718` | preintegration, smoothing of discontinuous probabilities, MLMC variance/rate analysis | Conditional smoothing and smoothed ML corrections are not novel. |
| L4 | Glasserman, Heidelberger, Shahabuddin, *Asymptotically Optimal Importance Sampling and Stratification for Pricing Path-Dependent Options* (1999), DOI `10.1111/1467-9965.00065` | Gaussian most-important path, LDP drift, curvature-informed stratification | A most-important Cameron--Martin path and quadratic directions are not novel by themselves. |
| L5 | Guasoni, Robertson, *Optimal Importance Sampling with Explicit Formulas in Continuous Time* (2008) | continuous-time variational drift and asymptotic optimality | A continuous-time LDP drift objective alone is not novel. |
| L6 | Robertson, *Sample path Large Deviations and optimal importance sampling for stochastic volatility models* (2010) | sample-path LDP and optimal drift for stochastic-volatility option pricing | Applying a generic LDP drift to stochastic volatility is not novel. |
| L7 | Jacquier, Pakkanen, Stone, *Pathwise large deviations for the Rough Bergomi model* (2018), DOI `10.1017/jpr.2018.72` | rBergomi RKHS and a particular small-time/rescaled LDP | The rBergomi RKHS and published rescaled LDP are not novel; its scaling must not be silently replaced by a fixed-strike claim. |
| L8 | Tong, Stadler, *Large Deviation Theory-based Adaptive Importance Sampling for Rare Events in High Dimensions* (2023), DOI `10.1137/22M1524758` | LDP initialization, low-dimensional likelihood-informed subspace and CEM | LDP plus adaptive IS or subspace CEM is not novel. |
| L9 | Uribe et al., *Cross-Entropy-Based Importance Sampling with Failure-Informed Dimension Reduction* (2021), DOI `10.1137/20M1344585` | failure-informed dimension reduction for CEM | Rare-event dimension reduction is not novel. |
| L10 | Hartmann, Richter, *Nonasymptotic bounds for suboptimal importance sampling* (2021/2023), DOI `10.1137/21M1427760` | proposal-error/relative-error bounds and high-dimensional fragility | A generic proposal-error bound is not novel; V15 needs a sharper conditional-Volterra specialization. |
| L11 | Arandjelovic, Rheinlander, Shevchenko, *Importance sampling for option pricing with feedforward neural networks* (2025), DOI `10.1007/s00780-024-00549-x` | Cameron--Martin spaces, neural approximation of optimal drifts and RN densities, LDP variational problems, path-dependent option experiments | Neural Cameron--Martin drift approximation and its universal approximation theorem are prohibited novelty claims. |
| L12 | Pinski et al., *Kullback--Leibler approximation for probability measures on infinite dimensional spaces* (2015), DOI `10.1137/140962802` | Gaussian variational approximation and covariance parameterization in function space | Low-rank Gaussian covariance approximation is not independently novel. |
| L13 | He, Zheng, Wang, *On the Error Rate of Importance Sampling with Randomized Quasi-Monte Carlo* (2023), DOI `10.1137/22M1510121` | RQMC rates under Gaussian/t proposals | Combining Gaussian IS and RQMC is not independently novel. |
| L14 | Leao et al., *Adaptive Learning via Off-Model Training and Importance Sampling for Fully Non-Markovian Optimal Stochastic Control* (2026), arXiv `2604.13147` | dominating training laws, RN weights, neural dynamic programming and nonasymptotic error bounds for non-Markovian/rough settings | Off-model neural learning and generic non-Markovian IS are not V15 novelty. |
| L15 | Gulisashvili, *Large deviation principle for Volterra type fractional stochastic volatility models* (2018), arXiv `1710.10711` | small-noise LDP for Volterra-type Gaussian stochastic-volatility models under explicit assumptions | A continuous Volterra small-noise LDP is prior art; V16 must align its exact family and numerical law rather than claim the LDP itself. |
| L16 | Guyader, Touchette, *Efficient large deviation estimation based on importance sampling* (2020), DOI `10.1007/s10955-020-02589-x` | general joint-LDP necessary/sufficient framework for logarithmic IS efficiency | Generic logarithmic-efficiency criteria are prior art; only the conditional-Volterra specialization and exact safety mixture may be studied as a combination. |
| L17 | Rozanov, *On the Density of One Gaussian Measure with Respect to Another* (1962), DOI `10.1137/1107006` | equivalence criteria and densities for Gaussian process measures | Gaussian covariance equivalence and determinant densities are classical prior art; V16 novelty cannot rest on Feldman--Hajek/Rozanov alone. |
| L18 | Bayer, Fukasawa, Nakahara, *On the Weak Convergence Rate in the Discretization of Rough Volatility Models* (2023), DOI `10.1137/22M1482871` | general and structure-dependent weak-error bounds for rough-volatility discretizations | V16 may use this as a boundary, but cannot assign its rates to the nonlinear conditional digital without matching the payoff and scheme assumptions. |
| L19 | Gassiat, *Weak Error Rates of Numerical Schemes for Rough Volatility* (2023), DOI `10.1137/22M1485760` | sharper rates for specified integrands, tests and hybrid-type schemes | The qualitative T16-7 limit is defensible; an explicit rate remains locked until the exact conditional payoff assumptions are checked. |

## Exact overlap boundary

The closest direct conflict is L11. It already proves that feedforward neural networks
can approximate Cameron--Martin importance-sampling drifts and associated densities,
and it discusses LDP variational problems. Therefore V15 may not claim:

- first neural Cameron--Martin importance sampler;
- first neural drift for financial option pricing;
- first neural approximation of an optimal Gaussian measure change;
- first LDP-neural importance-sampling combination;
- novelty from universal approximation alone.

L2 independently removes the price-driver noise in conditionally lognormal rough
volatility. Therefore V14's complete terminal price-driver conditioning is an exact and
useful specialization, but not by itself a top-journal novelty claim.

## Candidate residual contribution

Subject to external review, the only defensible V15 combination is:

1. condition on the Gaussian Volterra driver and analytically eliminate the entire
   independent terminal price driver;
2. formulate the corresponding conditional zero-variance target and prove the exact
   `chi-square(Pi* || Q)` relative-variance identity;
3. derive a reduced small-noise action in which the independent price control is
   analytically minimized out;
4. cover all dominating modes with an exact defensive balance mixture;
5. allow only equivalence-preserving Cameron--Martin shifts and finite-rank covariance
   perturbations, with a pathwise likelihood bound;
6. connect discretized BLP transports to a continuous Gaussian-Volterra target through
   a mesh-consistency/bias analysis;
7. optionally amortize certified modes across tasks, while exactness remains independent
   of the neural operator.

No single item in this list is assumed novel. The claim is the new conditional reduction,
theorems and financial consequences of their joint use.

## Prohibited claims

- first conditional Monte Carlo method for rBergomi;
- first numerical smoothing or preintegration method;
- first most-important-path or Cameron--Martin optimizer;
- first neural importance sampler for option pricing;
- first low-rank/failure-informed rare-event proposal;
- first large-deviation importance sampler for stochastic volatility;
- first Gaussian low-rank covariance approximation in function space;
- first RQMC importance sampler;
- continuous-time exact simulation;
- small-time rBergomi efficiency inferred only from the Jacquier--Pakkanen--Stone LDP;
- a top-journal novelty claim before external equation-level review.

## P0 decision

**Conditional development pass.** The internal audit found substantial prior art and
therefore narrowed V15. It did not find a source that performs the complete conditional
price-driver contraction at both the zero-variance-density and LDP-action levels and
then couples it to an exact defensive multimode finite-rank transport with a rough-
Volterra mesh audit.

This is not a novelty certificate. Two independent external reviews remain a hard
submission gate. Development may proceed because every final estimator remains exact
for any frozen proposal, but qualification and top-journal language stay locked.

## Primary links

- <https://doi.org/10.1080/14697688.2015.1099717>
- <https://doi.org/10.1080/14697688.2018.1459812>
- <https://doi.org/10.1137/22M1495718>
- <https://doi.org/10.1111/1467-9965.00065>
- <https://doi.org/10.1017/jpr.2018.72>
- <https://doi.org/10.1137/22M1524758>
- <https://doi.org/10.1137/20M1344585>
- <https://doi.org/10.1137/21M1427760>
- <https://doi.org/10.1007/s00780-024-00549-x>
- <https://doi.org/10.1137/140962802>
- <https://doi.org/10.1137/22M1510121>
- <https://arxiv.org/abs/2604.13147>
- <https://arxiv.org/abs/1710.10711>
- <https://doi.org/10.1007/s10955-020-02589-x>
- <https://doi.org/10.1137/1107006>
- <https://doi.org/10.1137/22M1482871>
- <https://doi.org/10.1137/22M1485760>
