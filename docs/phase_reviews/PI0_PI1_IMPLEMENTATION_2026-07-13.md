# PI0–PI1 Implementation and Error Review

Date: 2026-07-13<br>
Scope: research-specification freeze and Gaussian path-integral oracle

## Outcome

PI0 and PI1 are complete as a correctness foundation. The repository now has
one implementation-aligned convention for the target/proposal measures, soft
potential, Brownian likelihood, path action, tilted weight, PICE projection,
and estimator-divergence diagnostics. The Gaussian oracle recovers every
available analytic result.

This result does **not** make the project journal-ready. It establishes that
the next Heston and rough-volatility phases will not be built on an unresolved
likelihood sign, measure orientation, or objective mismatch.

## Implemented components

| Component | Result |
|---|---|
| Soft left-tail potential | Stable `softplus`, exactly equal to negative log-sigmoid |
| Brownian likelihood | Multidriver `log(dM/dQ)` with fixed sign convention |
| Path action | `Phi + sum(u dB_Q) + 0.5 sum(||u||^2 dt)` |
| Tilted weight | Log-domain `log(g dM/dQ) = -action` |
| Divergence diagnostics | Empirical relative variance, chi-square, Rényi-2, contribution ESS |
| Gaussian exponential tilt | Normalizer, optimal drift, PI gap, relative variance |
| Gaussian tail oracle | Stable continuous Doob drift using the inverse Mills ratio |
| Constant PICE | One- and multidriver reverse-KL weighted projection |
| Off-policy coordinates | Target Brownian to candidate residual reconstruction |

## Analytic checks

For \(g(B_T)=e^{aB_T}\), the implementation verifies

$$
\log Z=\frac12a^2T,\qquad u^\star=a,
$$

$$
\mathcal J_{\mathrm{PI}}(u)+\log Z=\frac12(u-a)^2T,
$$

and

$$
\frac{\operatorname{Var}_{Q_u}(gL_u)}{Z^2}
=e^{(a-u)^2T}-1.
$$

At \(u=a\), the log contribution is pathwise constant. At a nonoptimal
control, both likelihood normalization and the analytic relative variance are
recovered by Monte Carlo. A weighted off-policy PICE projection recovers
\(u=a\) from behavior-proposal paths.

For the Brownian hard left tail, the analytic drift is negative and matches a
central finite difference of the log conditional probability. The stable
`log_ndtr` formulation remains finite in a rare-tail test.

## Theoretical error audit

No unresolved error was found inside the implemented PI0–PI1 scope. The
following high-risk confusions are explicitly prevented:

1. The code returns \(d\mathbb M/d\mathbb Q\), never the inverse density.
2. PI forward KL, PICE reverse KL, chi-square, and Rényi-2 are not presented as
   interchangeable losses.
3. Soft path-law training and hard final estimation are separated.
4. A bounded finite-step Gaussian drift is not claimed to exactly represent a
   hard conditional law or an arbitrary discrete tilted transition.
5. PICE behavior weights are not multiplied by a candidate likelihood a
   second time.
6. Candidate score coordinates are reconstructed from the target Brownian
   path, not copied from behavior Brownian residuals.
7. Self-normalized PICE weights are restricted to training and are not called
   an unbiased final estimator.
8. Multidriver controls are defined in an independent Brownian basis.
9. Causality is a caller-side invariant: controls must be evaluated before the
   matching Brownian increments.

## Technical verification

Commands and outcomes:

```text
python -m ruff check src tests
All checks passed!

python -m mypy src
Success: no issues found in 25 source files

python -m pytest -q
119 passed
```

The repository-root command `ruff check .` also inspects historical helper
scripts outside `src` and `tests`. It reports 49 pre-existing issues in
`fix_viz.py`, `fix_viz_only.py`, `generate_concept_plots.py`, and
`update_maxiter.py`. They are unrelated to PI0–PI1 and were not modified in
this phase.

## Remaining limitations

- The new package is a verified mathematical core, not yet a trainable neural
  path-integral controller.
- Constant PICE is only the analytic projection gate; causal feedback PICE is
  still pending.
- Gaussian Monte Carlo checks validate signs and identities, not performance
  in stochastic-volatility tails.
- The existing Heston controlled simulator shifts only the spot Brownian
  coordinate. The full independent two-driver implementation is PI2.
- Controlled rBergomi BLP, its local-cell mean correction, memory feedback,
  and pathwise reconstruction remain PI3–PI4 work.
- No novelty or journal-level performance claim follows from PI0–PI1 alone;
  the publishable contribution must be demonstrated in rough-memory,
  amortization, theory, and work-normalized experiments.

## Next gate: PI2

PI2 should proceed in this order:

1. add a full two-driver independent-basis Heston simulator while preserving
   the current one-driver API as a baseline;
2. expose both proposal increments and reconstruct both target increments;
3. compare the accumulated likelihood with the generic multidriver core;
4. implement a soft Heston conditional-desirability oracle and validate its
   spot/variance gradients;
5. compare CEM, constant/affine/MLP, PI, feedback PICE, and \(J_2\) under a
   sealed train/validation/evaluation protocol;
6. perform time-step refinement before interpreting performance against the
   continuous-time Heston reference.

PI2 is complete only when reconstruction, normalization, unbiasedness,
oracle-direction recovery, and discretization-bias gates all pass.
