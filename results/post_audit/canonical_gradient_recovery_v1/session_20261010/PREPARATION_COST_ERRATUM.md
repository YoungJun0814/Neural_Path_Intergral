# Preparation cost erratum (2026-10-10)

The original source, manifest, checkpoints and result are preserved.
The canonical-only preparation constructs one envelope geometry on 64 basis
paths. It does not also construct a static guide. The recorded preparation
`actual_work=128` inherited the two-cell pilot count and overcounts by 64.

- Recorded total: 12,222,592 forward/path-law evaluations.
- Corrected total: 12,222,528 (12,222,464 estimator + 64 geometry).
- Recorded replay audit: 1,527,936; corrected replay audit: 1,527,872.
- Actual backward evaluations remain 4,079,616 (audit 509,952).

No estimate, seed, normalizer, gradient, summary or qualification is altered.
The budget guard was conservative; this is an accounting error, not a claim of
estimator correctness without the separate numerical audit. See the detailed
gradient-MH mathematical and cost review in docs/theory.
