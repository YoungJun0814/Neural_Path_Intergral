# G11 V8 Reproducible Novelty Matrix

Date: 2026-07-25

Status: **conditional pass for P2 theory implementation; submission novelty not
authorized**

Machine-readable ledger:
`configs/g11_v8/novelty_search_ledger_v1.yaml`

This audit asks a narrow question: after acknowledging component-level prior art,
does a defensible mathematical contribution remain for V8? It does not claim that a
finite search proves global absence of a closer paper.

## 1. Search boundary

The recorded search used arXiv, publisher/DOI pages, SSRN, and web queries restricted
to original papers or author records. It covers nine predeclared families:

1. rough Bergomi conditional Monte Carlo;
2. numerical smoothing with QMC, ASGQ, and MLMC;
3. multilevel importance sampling;
4. balance multiple importance sampling;
5. large-deviation and cross-entropy importance sampling;
6. adaptive mixture importance sampling;
7. exact-density flow importance sampling;
8. rough weak error and MLMC; and
9. learning-enhanced control variates.

The ledger stores query text, interface, access date, version, model, event, proposal,
density contract, conditional integration, theorem, work accounting, code status,
material overlap, and material non-overlap. Search and source coverage are checked by
an independent fail-closed program.

## 2. Closest-work matrix

| Work | Material prior art | Consequence for V8 | Role |
|---|---|---|---|
| McCrickerd and Pakkanen, [Turbocharging Monte Carlo pricing for the rough Bergomi model](https://arxiv.org/abs/1708.02563) | conditional lognormal pricing, controls, antithetics, runtime-aware rough Bergomi comparison | conditional Monte Carlo or rough Bergomi variance reduction is not new | mandatory secondary |
| Bayer, Ben Hammouda, and Tempone, [rBergomi ASGQ/QMC](https://arxiv.org/abs/1812.08533) | rough Bergomi QMC, Brownian bridge, Richardson extrapolation | QMC under rough Bergomi is not new; weak-rate conjectures may not be reused as proofs | secondary |
| Bayer, Ben Hammouda, and Tempone, [Numerical Smoothing with ASGQ/QMC](https://arxiv.org/abs/2111.01874) | root finding and one-dimensional preintegration before QMC/ASGQ | selected-variable smoothing is prior art and is the primary closest computational method | primary closest comparator |
| Bayer, Ben Hammouda, and Tempone, [MLMC with Numerical Smoothing](https://arxiv.org/abs/2003.05708) | smoothed discontinuous corrections, variance/kurtosis improvements, MLMC complexity | neither smoothing a discontinuity nor smoothing an MLMC correction is new | primary theory boundary |
| Giles, [Multilevel Monte Carlo Path Simulation](https://doi.org/10.1287/opre.1070.0496) | MLMC allocation and bias-variance-cost regimes | V8 may state end-to-end complexity only after verifying all required exponents | theory boundary |
| Sbert and Elvira, [Generalizing the Balance Heuristic](https://arxiv.org/abs/1903.11908) | unbiased multiple-proposal balance denominators and variance theory | the mixture denominator is prior art | theory boundary |
| Cappé et al., [Adaptive Importance Sampling in General Mixture Classes](https://arxiv.org/abs/0710.4242) | adaptive mixture proposals and a Rao--Blackwellization device | neither adaptive mixtures nor the phrase Rao--Blackwellization is new; the mathematical object must be distinguished | novelty boundary |
| Tong and Stadler, [Large-deviation adaptive IS](https://arxiv.org/abs/2209.06278) | large-deviation initialization, informative subspace, Gaussian CEM | rare-event subspace search and CEM are strong prior baselines | mandatory secondary |
| Ben Amar, Ben Rached, and Tempone, [Hierarchical IS for occupation time](https://arxiv.org/abs/2509.13950) | HJB SLIS/MLIS, common likelihood, preprocessing-inclusive work, crossover theorem | common likelihood, occupation-time IS, and training/preprocessing accounting are prior art | theory boundary |
| Gao et al., [NOFIS](https://arxiv.org/abs/2310.19167) | normalizing-flow proposals for nested rare events with importance weights | flow-assisted rare-event IS is prior art | mandatory secondary |
| Kruse et al., [Latent-space flow IS](https://arxiv.org/abs/2501.03394) | rare-event proposal exploration in invertible-flow latent space | a flexible flow proposal is a baseline, not the V8 contribution | secondary |
| Bayer, Hall, and Tempone, [Weak error under linear rough volatility](https://arxiv.org/abs/2009.01219) | rigorous weak rates for a linear rough model | its rates do not prove nonlinear rough Bergomi indicator bias | theory boundary |
| Bourgey and De Marco, [rBergomi VIX MLMC](https://arxiv.org/abs/2105.05356) | rough Bergomi MLMC complexity and analytic controls for a VIX functional | rough-volatility MLMC and controls are prior art; the observable differs | theory boundary |
| Jouravlev, [Learning-Enhanced Control Variates under Rough Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5668450) | pilot-trained centered controls, rough Bergomi path-dependent payoffs, equal-wall-time evidence | learned rough-volatility variance reduction is not new | secondary |

## 3. Surviving candidate contribution

The strongest defensible candidate after this audit is:

> Exact integration of an event-driving deterministic Gaussian shift span inside a
> defensive balance mixture, yielding an exact bounded residual likelihood, together
> with a rough-Volterra finite-grid strict-efficiency or rate theorem.

The contribution is a conjunction, not a collection of separately new ingredients.
It survives only if P2 establishes more than the classical Rao--Blackwell
non-increase:

- the exact target-over-full-mixture likelihood cancellation;
- the correct proposal-conditional law;
- the residual balance-mixture expression and defensive bound;
- nondegenerate strict improvement or a quantitative lower bound;
- measurable scalar-threshold lemmas for the declared events; and
- a rough-path theorem whose assumptions are actually verified.

If P2 produces only
\(\operatorname{Var}(E[Y\mid R])\leq\operatorname{Var}(Y)\), the result remains useful
but is unlikely to carry a top mathematical-finance journal submission by itself.

## 4. Claims explicitly removed

V8 may not claim to be the first:

- conditional Monte Carlo method under rough Bergomi;
- numerical smoothing method for discontinuous payoffs;
- QMC or MLMC method under rough volatility;
- balance-mixture importance sampler;
- mixture Rao--Blackwellization method;
- large-deviation or CEM rare-event sampler;
- flow-based rare-event importance sampler; or
- learning-enhanced rough-volatility variance-reduction method.

It also may not claim unconditional top-journal novelty. “No full combination was
located in this recorded search” is the strongest current wording.

## 5. Comparator consequences

The P0 roles remain correct:

- numerical-smoothing RQMC is the primary closest computational comparator;
- fixed raw defensive IS isolates the DCS mechanism;
- fresh task-tuned pure CEM tests adaptive training-inclusive work;
- published-style rough Bergomi conditional Monte Carlo is mandatory;
- large-deviation/CEM IS is mandatory;
- exact-density flow IS is mandatory; and
- learned control variates are a relevant secondary benchmark if their target can be
  matched without weakening the rare-event estimand.

Where numerical smoothing reduces algebraically to the same scalar Gaussian CDF as
DCS, the paper must report equivalence rather than manufacture a performance win. A
distinct RQMC comparison requires randomized replicates and an estimand on which
preintegration still leaves a nontrivial remaining-dimensional integral.

## 6. Limitations and final-search obligations

This P1 audit is reproducible but not exhaustive. Before submission:

1. run MathSciNet and zbMATH subject/author searches;
2. run Scopus or Web of Science citation and cited-reference searches;
3. search Google Scholar citing papers for every primary closest work;
4. search Gaussian-mixture marginalization, marginal importance sampling,
   preintegration, and subspace Rao--Blackwellization with broader terminology;
5. record all close exclusions and version changes through the submission cutoff;
6. ask an independent expert in Monte Carlo/rough volatility to challenge the
   one-sentence novelty statement; and
7. narrow or stop the paper if a materially closer construction is found.

## 7. P1 conclusion

**Conditional pass.** The search did not locate the full proposed combination, and
the predeclared P2 theory program may proceed. This is an inference from the recorded
search, not proof of absence. Submission-level novelty remains blocked on the P2
theorem, final database search, and external expert review.
