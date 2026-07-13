# Research Charter

> Status: active research specification<br>
> Effective date: 2026-07-13<br>
> Governing roadmap: [`PUBLICATION_GRADE_RESEARCH_PLAN.md`](PUBLICATION_GRADE_RESEARCH_PLAN.md)

## Primary research question

Can a memory-aware, amortized neural importance sampler estimate rare downside
events under non-Markovian rough-volatility models more accurately per unit of
total compute than strong classical and neural baselines, while retaining an
exact likelihood ratio for the chosen discretized model?

## Primary paper scope

- Heston is a Markovian validation model, not the main novelty.
- rBergomi is the main non-Markovian model.
- The proposed controller is conditioned on event parameters, maturity, and
  model parameters so it can be reused across tasks.
- Terminal downside events, barrier/drawdown events, and deep-OTM nonnegative
  payoffs are the primary functionals.
- Brownian changes of measure are required. Jump tilting is an optional
  extension after the rough-volatility results pass the correctness gates.

## Measure convention

- `M` denotes the financial target measure whose expectation is being
  estimated. It may be the physical measure for stress probabilities or a
  risk-neutral measure for pricing, but a single experiment must choose one.
- `Q_phi` denotes the neural proposal/sampling measure.
- `L_phi = dM / dQ_phi` denotes the likelihood ratio used in the estimator.
- Physical calibration and risk-neutral pricing must not share an unidentified
  drift or be presented as the same statistical problem.

## Claims allowed before the publication gates pass

- The repository is a research prototype for controlled-SDE importance
  sampling.
- The correlated-Heston Girsanov correction has targeted numerical tests.
- Reported results are exploratory unless produced by the frozen paper
  experiment pipeline with independent evaluation samples.

## Claims not allowed without new evidence

- A crash-frequency ratio is a speedup or variance-reduction factor.
- Girsanov reweighting removes time-discretization or model-calibration bias.
- The method predicts black swans.
- A single integrated-gradients example identifies a general market crash
  driver.
- The method is physics-informed unless a precise PDE, conservation law, or
  path-integral/control structure is part of the method and validation.

## Publication gates

1. **G0 — Claims:** public documentation contains no unsupported headline
   result.
2. **G1 — Correctness:** simulators, likelihoods, convergence tests, lint, and
   type checks pass.
3. **G2 — Markov benchmark:** frozen-control estimators are correct and
   competitive on analytic/Heston problems.
4. **G3 — Rough memory:** memory-aware control significantly improves at least
   two rare-event regimes over a state-only controller.
5. **G4 — Amortization:** unseen in-domain tasks retain prespecified
   performance and demonstrate a compute break-even advantage.
6. **G5 — Theory:** the estimator and approximation results have complete
   proofs matching the implemented discretization.
7. **G6 — Evidence:** preregistered multi-seed benchmarks and ablations are
   complete.
8. **G7 — Submission:** a clean-room reproduction regenerates the main tables
   and figures.

## Naming policy

The scientific method is called **memory-aware amortized neural importance
sampling**. `DriftNet` remains the implementation/project name. “Neural Path
Integral” is used as a scientific term only if the paper derives and uses the
corresponding path-measure or linearly-solvable-control formulation.
