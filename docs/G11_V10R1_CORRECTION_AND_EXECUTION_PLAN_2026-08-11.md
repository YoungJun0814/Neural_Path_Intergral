# G11 V10R1 correction and execution plan

Date: 2026-08-11  
Target: defensible finite-grid terminal rare-event result, with journal claims
kept behind explicit gates

## 1. Why V10 must be rerun

Exploratory V10 trained a `3N` BLP CEM mean but passed only one coordinate from
each two-coordinate local BLP pair, plus the price coordinates, to its final
adapter.  Thus `N` learned local coordinates were discarded.  It also changed
the price portion after training to force a sign convention.  The evaluated
proposal was consequently not the proposal claimed by the training artifact.
Its bank and benchmark were generated from a dirty worktree and named a source
commit that did not contain the executed V10 code/configuration.  Those results
are useful diagnostics only and cannot support a paper claim.

V10R1 is a correction namespace.  It preserves V10 artifacts without rewriting
history.

## 2. Non-negotiable protocol

1. The exact BLP latent space has dimension `3N` throughout training, sampling,
   simulation, likelihood evaluation, hashing, storage, and replay.
2. The frozen proposal is never projected or modified after CEM training.
3. One positive price-block direction is used only as an integration basis.
4. The raw and DCS estimators use exact ordinary likelihoods; self-normalization
   is forbidden.
5. Candidate CEM training randomness is represented by independent proposal
   replicates, one per inferential cluster.
6. Candidate and defensive-CEM comparator use the same frozen training budget.
7. Training, allocation pilot, likelihood diagnostic, and final evaluation work
   are all included.  Algorithmic work is primary; measured time is secondary.
8. All claim-bearing runs require a clean committed tree, exact artifact hashes,
   immutable outputs, contiguous unique seeds, and an independent recomputation
   audit.
9. Development failure stops qualification.  No result can set top-journal or
   submission authorization to true.

## 3. Execution phases and gates

### P0 — mathematical and coordinate contract

- Freeze the target, proposal, scalar decomposition, exact residual likelihood,
  Rao--Blackwell identity, defensive bound, and claim exclusions.
- Gate: code-to-notation ledger has no dropped coordinate or unproved
  multiplicative claim.

### P1 — corrected estimator implementation

- Sample the complete defensive mixture in `R^(3N)`.
- Feed those exact coordinates to `simulate_latent`.
- Reconstruct both coordinates of every local BLP pair and every price normal.
- Evaluate full and residual mixture likelihoods independently.
- Evaluate the exact terminal threshold and stable normal CDF.
- Record raw and DCS costs separately.
- Gate: every reconstruction and density diagnostic is at numerical tolerance.

### P2 — adversarial tests

- Full-basis dimension and positivity tests.
- Analytic likelihood/component reconstruction tests.
- Paired raw-versus-DCS expectation test.
- Mutation of a coordinate discarded by V10 must change the result.
- Proposal serialization/hash replay test.
- Audit mutation tests must reject altered seeds, work, aggregates, proposal
  hashes, or decisions.
- Gate: Ruff, mypy, targeted tests, and full regression pass.

### P3 — clean-source reference re-audit and proposal bank

- Independently re-audit the existing clean-source V9 reference artifact.
- Freeze a fresh V10R1 bank config before seeing V10R1 outcomes.
- Train four independent full-`3N` proposals per cell on all 12 terminal cells.
- Rebuild all proposal hashes and deterministically replay every training run.
- Gate: exact roster, seed set, full dimension, bank hash, total cost, clean
  source, committed config blob, and replay all pass.

### P4 — resource micro-run

- Before the full matrix, run one representative cell with two tiny independent
  proposals and a reduced paired sample count as a non-claim-bearing smoke and
  resource test.
- It may select a feasible resource budget, but may not tune performance gates,
  choose favorable Hurst regimes, or enter the final estimator comparison.
- Gate: no correctness, audit, or memory failure.

### P5 — frozen development matrix

- Use 12 terminal cells: `H in {0.05, 0.12, 0.20}` and nominal probabilities
  `10^-2` through `10^-5`.
- Use four independent proposal/evaluation clusters per cell.
- Compare paired full-CEM raw and full-CEM DCS estimators against conditional
  rBergomi, smoothing RQMC, and independently trained defensive CEM.
- Require exactness, finite ordinary likelihoods, likelihood normalization,
  agreement with the independent reference, paired raw/DCS identity, and zero
  resource censoring before any efficiency result is read.
- Primary comparison is training-inclusive algorithmic work at 100 amortized
  queries.  Report 1, 10, 100, and 1000 queries.

### P6 — independent development audit

- Recompute work-to-target and every aggregate from raw records.
- Check seed order, uniqueness, and disjointness from bank/reference seeds.
- Bind each cluster to its exact independently trained proposal.
- Verify source commit contains the exact configuration bytes.
- Gate: any mismatch makes the artifact invalid, regardless of performance.

### P7 — qualification decision

- If and only if development and audit pass, freeze a fresh qualification
  namespace with disjoint seeds before inspecting those outcomes.
- Qualification must cover every Hurst group, not only a selected subgroup, to
  support a grid-wide claim.  A regime-conditional claim may name only a group
  prespecified by the development rule and must be labelled empirical.
- If development fails, record the falsification result and return to method
  development.  Do not recycle development seeds as qualification evidence.

### P8 — manuscript-level work still required

- Repeat across discretizations to separate estimator error from model
  discretization error.
- Add a second rough-volatility model or a demonstrably different path-dependent
  application.
- Supply asymptotic or non-asymptotic theory beyond Rao--Blackwell non-increase
  if a top mathematical-finance journal is the target.
- Run independent hardware replication and a genuinely held-out qualification.
- Compare with current peer-reviewed state of the art under identical estimands.

## 4. Stop rules

- Any coordinate/law/likelihood mismatch: invalidate the run and fix P1.
- Dirty source or absent committed config: do not execute claim-bearing code.
- Bank replay mismatch: do not execute the benchmark.
- Reference/correctness/resource gate failure: efficiency ratios are descriptive
  diagnostics only.
- Development efficiency failure: do not run qualification merely to search for
  a favorable seed.
- Qualification failure: no performance claim and no submission authorization.

## 5. Meaning of success

Passing P0–P6 establishes a reproducible, technically valid development result.
Passing a later P7 supports only the frozen finite-grid empirical claim.  It is
strong doctoral research infrastructure, but it is not by itself a top-journal
paper.  P8 remains necessary for that level.

