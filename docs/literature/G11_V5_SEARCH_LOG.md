# G11 V5 Reproducible Literature Search Log

Date opened: 2026-07-22

Status: living pre-submission log; current search is sufficient for M0 development,
not a substitute for the final submission-cutoff search or external expert review

## 1. Research question

The search asks whether prior work already combines all of:

1. a defensive deterministic Gaussian balance mixture;
2. exact marginalization of its event-driving proposal-control span;
3. the resulting exact bounded residual mixture likelihood;
4. a common-coordinate scalar-threshold multilevel correction;
5. non-Markovian Gaussian rough-Volterra dynamics; and
6. an uncertainty-aware, preprocessing-inclusive choice between SLIS and MLIS.

Component-level prior art is expected and is not treated as novelty.

## 2. Databases and interfaces searched

- arXiv full-text/metadata search;
- publisher/DOI landing pages for known Finance and Stochastics, SISC, Operations
  Research, Statistics and Computing, and Quantitative Finance papers;
- citation trails already recorded in `G11_NOVELTY_MATRIX.md`; and
- repository-local reference and claim ledgers.

Final submission work must add MathSciNet, zbMATH, Web of Science or Scopus, and
Google Scholar citation searches when access is available.

## 3. Search strings used on 2026-07-22

- `rough Bergomi multilevel Monte Carlo barrier weak convergence`
- `conditional smoothing multilevel Monte Carlo discontinuous payoff barrier`
- `multilevel importance sampling rare events stochastic differential equations`
- `rough Bergomi hybrid scheme convergence`
- `weak convergence rough Bergomi`
- `stochastic Volterra Euler strong convergence rough Bergomi H rate`
- `rough volatility barrier option continuous monitoring bias`
- `Gaussian mixture Rao-Blackwellization importance sampling`
- `pilot uncertainty adaptive MLMC allocation confidence variance estimation`
- `hierarchical importance sampling occupation time SLIS MLIS common likelihood`

## 4. Included closest primary sources

| Source | Material overlap | Consequence for G11 |
|---|---|---|
| Giles (2008), MLMC path simulation, <https://doi.org/10.1287/opre.1070.0496> | optimal allocation and complexity | standard MLMC work coefficient is prior art |
| Bayer, Ben Hammouda, Tempone, <https://arxiv.org/abs/2003.05708> | one-dimensional numerical smoothing for discontinuous MLMC | smoothing discontinuities is prior art |
| Ben Amar, Ben Rached, Tempone v2, <https://arxiv.org/abs/2509.13950> | SLIS/MLIS, common likelihood, crossover, preprocessing | none of those components is independently novel |
| Bennedsen, Lunde, Pakkanen, <https://doi.org/10.1007/s00780-017-0335-5> | hybrid rough-process simulation | rough simulation is prior art |
| Neuenkirch, Shalaiko, <https://arxiv.org/abs/1606.03854> | strong approximation order barrier in a rough volatility model | do not assume a classical SDE rate |
| Bourgey, De Marco v3, <https://arxiv.org/abs/2105.05356> | MLMC for VIX options in rBergomi | rough-Bergomi MLMC is prior art |
| Bayer, Hall, Tempone, <https://arxiv.org/abs/2009.01219> | weak rates for a linear rough model | does not prove nonlinear rBergomi threshold rates |
| Sbert, Elvira, <https://arxiv.org/abs/1903.11908> | balance-heuristic MIS | balance denominators are prior art |
| Ben Rached et al., <https://arxiv.org/abs/2208.03225> | multilevel IS for McKean--Vlasov rare events | IS plus multilevel complexity is prior art |

## 5. Excluded or non-matched families

- Markovian HJB importance sampling without a validated rBergomi lift: method class is
  relevant, but the target and state contract are not matched.
- VIX-option MLMC: relevant rough hierarchy, but its observable is not the spot-path
  rare threshold used by G11.
- General sequential/subset simulation for PDE reliability: relevant rare-event
  computation, but no exact Gaussian control-span residual likelihood.
- Neural proposal learning: relevant amortization, but earlier internal neural tracks
  failed the strong training-inclusive baseline.
- Quantum/Feynman path-integral algorithms: different mathematical problem.

Exclusion is not a claim that these papers are unimportant. It records why they do
not currently eliminate the narrow G11 candidate contribution.

## 6. Current novelty decision

**M0 conditional pass for implementation.** No included source was found that contains
the full six-part combination in Section 1. The candidate contribution remains:

> exact removal of a deterministic event-driving Gaussian proposal-control span from
> a defensive balance mixture, producing an exact bounded residual likelihood and a
> scalar-threshold correction for a rough Gaussian-Volterra hierarchy, with an
> uncertainty-audited finite-level deployment rule.

This wording is provisional. In particular, the uncertainty-aware deployment rule
may itself overlap adaptive multifidelity pilot literature and must not be called new
until the final search is complete.

## 7. Required final search

Before submission:

1. search citations and citing papers for every source in Section 4;
2. search marginal importance sampling and Rao--Blackwellization over Gaussian
   proposal parameters and subspaces;
3. search adaptive MLMC/multifidelity pilot confidence and best-arm selection;
4. search continuous/discrete barrier bias in rough volatility;
5. record every query, date, database, inclusion decision, and close exclusion; and
6. obtain an external expert challenge to the one-sentence contribution.

If a closer source is found, update the novelty matrix and narrow the claim before
freezing any confirmatory protocol.
