# Preparation cost erratum (2026-10-10)

The original source, manifest, checkpoints and result remain unchanged.
The canonical-only preparation computes one envelope geometry on 64 basis paths,
not 128. The recorded preparation was overcounted by 64; this did not affect the
normalizers, derivatives, seeds, RSEs, selected design or conservative budget.

- Recorded forward/path-law work: 14,327,936.
- Corrected work: 14,327,872 (14,327,808 estimator + 64 geometry).
- Actual backward work remains 14,327,808.
- Recorded replay forward audit: 1,791,104; corrected: 1,791,040.
- Replay backward audit remains 1,790,976.

This is a disclosed accounting error. Separate source/seed/numerical audits have
passed; neither those audits nor the selected design certify rare-tail precision.
