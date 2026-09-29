# G11 V15 implementation and verification report

Date: 2026-08-11

## Outcome

V15 is implemented as a complete finite-grid research pipeline from probability
contract through frozen qualification, offline audit, manuscript draft, and external
reproduction instructions.  Numerical qualification passes.  Top-journal readiness
does not pass because the central asymptotic and mesh theorems remain open.

## Implemented phases

| Phase | Deliverable | Verification outcome |
|---|---|---|
| P0–P1 | primary-source ledger, claim lock, small-noise probability contract | pass; neural CM drift itself removed from novelty claim |
| P2 | stable conditional digital/put/call oracle | pathwise, extreme-tail, parity, and limit oracles pass |
| P3 | DCT CM basis, finite-noise action, dual solvers, multistart modes | analytic minima and finite-difference gradients pass |
| P4 | defensive multimode finite-rank Gaussian transport | dense density, sample moments, normalization, and bound pass |
| P5 | theorem ledger T15-1–T15-9 | T15-1–4 proved; T15-5–8 open; G5 false |
| P6 | exact adjacent mesh study and pilot-frozen MLMC allocation | v1 fails; streaming v2 bias budget passes but rate remains unidentified |
| P7 | structure-preserving neural initializer | SPD/bounded outputs, teacher fit, corrector, and fallback pass |
| P8 | five-comparator ordinary-IS protocol | comparator-completeness and training-cost locks pass |
| P9–P10 | frozen development/qualification and offline audit | integrity/numerical pass; theory/top-journal fail |
| P11–P12 | reproduction protocol, manuscript, claim audit | package ready; external execution still pending |

## Errors found and corrected during implementation

1. SciPy L-BFGS-B aborted the Windows Python process inside a nonconvex multistart
   test.  It was replaced by PyTorch strong-Wolfe L-BFGS and a self-contained dense
   trust-region Newton audit solver.  Both recover analytic minima.
2. A `zip(strict=True)` input check paired `steps` with `steps[1:]`, causing a length
   mismatch.  It was corrected to `steps[:-1]` with `steps[1:]` and caught by the
   integrated suite.
3. The first direct experiment invocation lacked the project root on Python's import
   path.  Reproduction commands now use `python -m experiments...` exclusively.
4. Mesh v1 used too few samples to identify adjacent corrections and failed its bias
   gate.  The failed artifact was preserved.  Streaming sufficient statistics were
   added and v2 used 50,000 samples through 512 steps without selecting a favorable
   seed from v1.
5. The deepest small-noise v1 natural conditional reference missed the dominant
   region and failed accuracy at `z=4.88`.  The failed artifact was preserved.  V2
   added a disjoint, larger ordinary-IS stream from the same frozen exact proposal;
   primary/reference agreement is then below `z=0.59` at every epsilon.  The shared
   proposal family is disclosed as a limitation.

## Frozen numerical receipts

### Qualification

| Cell | Candidate accuracy z | likelihood normalization z | bound violation | strongest-primary/V15 work ratio |
|---|---:|---:|---:|---:|
| `H=.05,K=70` | 0.994 | 1.002 | 0 | 1.430 |
| `H=.12,K=50` | 1.260 | 0.049 | 0 | 4.689 |
| `H=.20,K=35` | 2.220 | 0.198 | 0 | 3.964 |

All five primary comparators are accuracy-qualified in all three cells.

### Small-noise v2

At epsilon `1, .5, .25, .125`, relative variances are approximately
`.535, .500, .530, .502`; observed second-moment exponent ratios are
`.923, .955, .972, .985`.  The last probability is approximately `1.34e-6`.

### Mesh v2

Adjacent correlations increase from `0.952` to `0.972`, and the conservative bias
proxy fits inside the requested RMSE budget.  The fitted correction rate is only
`0.124` and finest correction SNR is `1.75`; no rate theorem follows.

## Verification receipt

- Full repository tests: **992 passed**.
- Focused/new static type checks: pass.
- Ruff on CI paths (`src`, `tests`, `experiments`): pass.
- `git diff --check`: pass.
- Development and qualification offline audits: integrity and numerical pass;
  theory and top-journal gate fail exactly because G5 is false.

Repository-wide `ruff check .` additionally scans three legacy root utilities and
reports 49 pre-existing style findings in `fix_viz.py`, `generate_concept_plots.py`,
and `update_maxiter.py`.  CI does not lint those files, and V15 did not modify them.

## Final research judgment

This is now a technically credible doctoral working-paper core with unusually strong
exactness and falsification infrastructure.  It is not yet a top mathematical-finance
journal submission.  The next scientifically decisive work is a valid T15-5 proof,
not another neural architecture or an additional favorable finite-grid table.
