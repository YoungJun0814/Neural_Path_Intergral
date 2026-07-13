# Phase 0–1 Execution Report

> Date: 2026-07-13<br>
> Roadmap: `PUBLICATION_GRADE_RESEARCH_PLAN.md`<br>
> Scope: claim hygiene, simulator correctness, efficiency metrics, and quality gates

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| G0 — Claims | Passed | README no longer advertises 126x speedup, universal unbiasedness, black-swan prediction, or a population-level 60% XAI result |
| G1 — Core correctness | Functionally passed | 72 tests pass; Ruff and Mypy pass; Heston/rBergomi/jump regression tests pass |
| G1 — Coverage report | Pending local plugin | `pytest-cov` is declared in dev requirements and CI, but was not installed in the current desktop environment |
| G2 — Markov benchmark | Not started | Requires a validated trajectory-likelihood CEM and high-accuracy Heston references |

## Completed work

### Research scope and public claims

- Added `RESEARCH_CHARTER.md` with the primary question, measure convention,
  non-claims, naming policy, and publication gates.
- Reframed the README as a research prototype.
- Marked notebook figures as legacy exploratory outputs.
- Removed the crash-frequency ratio as an efficiency claim.
- Removed the one-state integrated-gradients result as a general crash-driver
  claim.
- Separated the terms financial target measure and proposal measure in the
  research charter.

### rBergomi hybrid scheme

- Removed fractional powers of invalid negative lags that produced NaNs.
- Corrected the earlier-cell kernel scale from `dt^(alpha+1)` to `dt^alpha`.
- Added the standard `sqrt(2H)` normalization.
- Added the deterministic variance of the discretized Gaussian Volterra
  process.
- Used that variance in the discrete Wick exponential, preserving the forward
  variance mean at finite step size.
- Restricted the public API to the actually implemented BLP `kappa=1` scheme.
- Made parameter overrides non-mutating and validated `H`, `xi`, and `rho`.
- Ensured a noninteger requested step reaches maturity exactly.
- Added covariance, refinement, forward-variance-mean, maturity, and parameter
  regression tests.

### Heston/Bates/SVJJ engine

- Preserved the raw variance Euler state internally for a true
  full-truncation recursion while returning the nonnegative effective variance.
- Replaced rounded horizons with equal steps ending exactly at `T`.
- Added input/model validation.
- Added the missing jump component to controlled Bates/SVJJ simulations.
- Preserved the same independent jump law under Brownian-only control, so no
  unsupported jump likelihood term is introduced.
- Replaced `n * Exponential` variance jumps with the correct Gamma sum for
  multiple SVJJ jumps.
- Added zero-control natural-vs-controlled jump regression tests.

### Objectives and metrics

- Corrected work-normalized VRF to

  ```text
  VRF_work = (variance_MC * cost_MC) / (variance_IS * cost_IS)
  ```

- Added tests proving that a slower sampler is penalized rather than rewarded.
- Clarified online versus end-to-end learned-proposal cost.
- Removed the invalid helper named `cem_step`; it paired elite labels and
  states from independent batches and was not a CEM update.
- Reclassified the softplus/KL loss as entropy-regularized stress generation,
  not a smooth log-indicator or variance-minimization objective.
- Updated the smoke benchmark to measure wall-clock training/evaluation costs
  and report online and single-batch end-to-end VRF separately.

### Repository quality

- Aligned the declared Python version with the code: Python 3.10+.
- Applied Ruff formatting and safe lint fixes to source and tests.
- Added Mypy to CI.
- Extended CI lint scope to experiments and the training entry point.
- Removed the remaining autograd-to-float test warning.

## Verification snapshot

```text
pytest -q
72 passed

ruff check src tests experiments main.py train_driftnet.py
All checks passed

mypy src main.py train_driftnet.py
Success: no issues found in 15 source files
```

Default CLI Heston smoke run:

```text
paths: 20,000
T: 0.5
terminal mean: 102.5575
terminal std: 13.5852
elapsed: 0.402 s
```

The small `N=1000` rare-event smoke benchmark is intentionally not a paper
result. It reported:

```text
MC contribution variance:       4.5931
constant-control variance:      2.7982
neural-control variance:        3.9934
neural online work VRF:         0.4042
single-batch end-to-end VRF:    0.00465
```

This is useful negative evidence: the current neural sampler is slower and
less work-efficient than naive MC in this small smoke configuration. It must
not be advertised as an improvement. G2 requires a redesigned training method,
strong baselines, independent seeds, and a preregistered compute budget.

## Remaining work before G2

1. Add an independent exact-covariance rBergomi reference implementation for
   small grids; current covariance tests validate the hybrid sampler against
   its deterministic discrete covariance and analytic normalization.
2. Add a high-accuracy Heston reference pricer and a validated QE/reference
   simulation scheme.
3. Implement a genuine trajectory-likelihood CEM baseline.
4. Add barrier bridge/continuity correction or explicitly restrict barrier
   claims to discrete monitoring.
5. Add log-domain likelihood diagnostics and contribution-based efficiency
   diagnostics.
6. Add local coverage tooling and set a justified CI coverage threshold after
   measuring uncovered scientific paths.
7. Build the frozen training/validation/evaluation split required for G2.

## Interpretation

Phase 0–1 removes the known false headline claims and repairs the reproducible
numerical failures found in the initial audit. It does not establish the paper's
novelty or performance claim. The project should now move to G2: analytic
Markovian references and a correct baseline/training comparison before any
memory-aware or amortized architecture is introduced.
