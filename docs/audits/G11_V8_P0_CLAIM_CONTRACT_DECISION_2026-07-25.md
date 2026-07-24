# G11 V8 P0 Claim-Contract Decision

Date: 2026-07-25

Decision: **PASS for development; not a qualification or outcome freeze**

## 1. Decision scope

P0 authorizes the V8 theory and comparator-infrastructure program. It does not
authorize a performance, novelty, continuous-time, complexity, or top-journal
claim. Those claims remain conditional on P1--P14.

The machine contract is
`configs/g11_v8/top_journal_claim_contract_v1.yaml`, with SHA-256:

```text
f73d2f1b89aebefa710e0315e0f0f8b1415062ec1595c2a86d2d457c613281cd
```

## 2. Contract fixed before V8 outcomes

- The estimand is a rough Bergomi probability on a declared 128-step grid.
- Terminal and discretely monitored barrier events are primary.
- Continuously monitored barriers and hit-plus-occupation events are not primary.
- The paper has exactly three proposed contribution slots.
- Fixed raw defensive IS is the mechanism comparator.
- Fresh task-tuned pure CEM is the adaptive-work comparator.
- Numerical-smoothing RQMC is the closest-published-method comparator.
- Comparator selection may not use V8 outcomes.
- Training, failed attempts, tuning, sampling, likelihoods, payoffs, and conditional
  integrations enter the work ledger.
- Independent seed clusters, not paths or cells, are the inferential units.
- Multiple primary superiority statements require simultaneous intervals.
- A full-path flow is a secondary baseline unless exact density evaluation and the
  DCS conditional integral remain tractable.
- Each phase permits exactly one commit after theory, technical, and test audits.

## 3. Machine audit

The fail-closed audit passed all 59 contract checks. It validates exact allowed keys
as well as values, preventing contradictory shadow fields from silently bypassing
the contract. Tests verify rejection of:

1. continuous-monitoring promotion;
2. a contradictory `continuous_time` alias;
3. a post-hoc fourth contribution;
4. replacement of the closest-method comparator;
5. exclusion of comparator training cost;
6. outcome-selected comparator status;
7. removal of the residual-flow tractability condition;
8. weakening of the external total-work gate;
9. an unknown schema; and
10. overwrite of an existing audit artifact.

The command-line path writes a failure report before exiting nonzero, so a rejected
contract remains diagnosable.

## 4. Theory audit

### Correct boundaries

- The current exactness claim is finite-grid only. No discrete barrier is described
  as a continuously monitored event.
- A positive target component in the defensive proposal is sufficient to make the
  finite-dimensional target absolutely continuous with respect to the proposal and
  to bound the exact likelihood by the reciprocal defensive weight.
- The planned DCS theorem is explicitly a proposal-conditional expectation. The
  residual-dependent conditional mixture may not be replaced by a target-law
  standard Gaussian.
- Rao--Blackwell non-increase alone is not treated as the new theorem. P2 must add
  strict nondegeneracy or a quantitative/model-level result.
- A correction-variance rate is not promoted to MLMC complexity without separate
  weak-bias and per-sample-cost exponents.

### Unresolved obligations

- P1 must show that the proposed novelty survives the closest conditional-smoothing
  and rare-event importance-sampling literature.
- P2 must prove the exact likelihood cancellation, conditional-mean identity,
  strictness conditions, and terminal/barrier threshold lemmas in the V8 notation.
- P3 must include fine-only barrier crossings and must prove or explicitly downgrade
  all mesh, weak-bias, and complexity statements.
- No arbitrary flow extension is valid unless its density and conditional integral
  are exact and tractable.

These are blocking future gates, not P0 failures.

## 5. Technical audit

- Canonical audit: pass.
- Targeted P0 tests: 13/13 pass.
- Contract corruption tests: pass.
- Full repository regression suite: 558/558 pass.
- CI-scope Ruff: pass.
- Mypy: pass with no issues in 82 source files.
- `git diff --check`: pass.
- The audit refuses to overwrite evidence and emits deterministic, finite JSON.
- No empirical V8 result was inspected or used to select a comparator or threshold.

## 6. P0 conclusion

P0 is theoretically scoped and technically enforceable. It is suitable as the
starting contract for the top-journal program, but it provides no evidence that the
program will pass P1 novelty, P2--P3 theory, or P8--P11 external-performance and
reproduction gates.
