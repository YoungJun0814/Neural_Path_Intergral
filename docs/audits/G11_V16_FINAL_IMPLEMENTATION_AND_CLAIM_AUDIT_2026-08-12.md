# G11 V16 final implementation and claim audit

Date: 2026-08-12  
Policy: `v16_hybrid_routing_v5`

## Decision

- Finite-grid named-cell empirical claim: **PASS**.
- Uniform OOD dominance: **NOT AUTHORIZED**.
- Top-journal submission: **LOCKED**.

The machine-readable audit binds the canonical confirmation, six-cell V5 OOD
confirmation, theorem ledger, claim contract, and novelty ledger by SHA-256.

## Confirmed dominance cells

| Cell | strongest-comparator/V16 training-inclusive WNV |
|---|---:|
| rough H=0.05, K=1 canonical | 5.978 |
| rough H=0.05, K=2 OOD | 1.280 |
| rough H=0.05, K=0.5 OOD | 5.023 |
| regular H=0.12, K=1 OOD | 1.815 |

## Correctness fallback cells

The rough/high-eta, rough/strong-correlation, and high-eta/strong-correlation cells
pass reference accuracy, robust uncertainty, likelihood normalization, and exact
likelihood-bound gates. Their descriptive comparator ratios are below one, so the
audit labels them `correctness_fallback` and excludes them from dominance claims.

## Error review

1. Final inference uses ordinary IS, never self-normalization.
2. Every Gaussian family is normalized and the full balance density is evaluated.
3. SMC training particles and validation samples are not final inferential units.
4. Routing uses declared structural inputs, not estimated probabilities or outcomes.
5. Comparator qualification uses the independent SMC reference.
6. Training and evaluation work are included at the primary query count.
7. Failed V10--V14 hypotheses remain preserved.

## Submission locks

The audit returns `top_journal_submission_authorized=false` because external novelty
review is 0/2, independent person/hardware reproduction is absent, and the
quantitative continuous relative-bias/end-to-end complexity gate is open. Passing
the internal finite-grid audit does not override these blockers.
