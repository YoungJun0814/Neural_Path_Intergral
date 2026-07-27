# G11 V8 P5 Threshold-Manifest Binding Decision

Date: 2026-07-27

Decision: **PASS FOR INDEPENDENT REFERENCE EXECUTION; PERFORMANCE OPEN**

The 24 finite-grid thresholds were recalibrated in the P5-specific
`p5-threshold-calibration-development` namespace, independently of P7 development,
references, and future final samples.  The clean calibration source commit is
`fd9b6046568a2de8fb2f5abbe7f5f7b8d33ace4a`; the calibration result has no dirty
worktree flag and passes its complete-matrix, probability-band, simultaneous-interval,
precision, likelihood-normalization, positivity, and seed-separation gates.

The frozen threshold-manifest SHA-256 is

```text
165e2107a031e55acefe55cec5888e23ea8f648a072d0a0a28833ca969451ef8
```

The binding config and its fail-closed audit bind the P5 matrix, P6 statistical design,
calibration config, calibration result, source commit, manifest, and future reference
namespace.  A reference cannot silently switch a threshold, cell, seed family, or
source artifact.

This means only that the subsequent reference estimand is a declared 128-step event
conditional on this fixed threshold manifest.  It is neither a continuous-barrier
claim nor DCS performance evidence.  The raw crosscheck and DCS reference must still
reach their prespecified standard-error, normalization, no-censoring, and agreement
gates before P5 reference execution can pass.
