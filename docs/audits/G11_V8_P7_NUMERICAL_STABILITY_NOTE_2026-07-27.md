# G11 V8 P7 Numerical-Stability Development Note

Date: 2026-07-27

Status: **INITIAL PROPOSAL FALSIFIED; CONSERVATIVE BARRIER PROPOSAL UNDER DEVELOPMENT**

The first full P7 calibration run used the barrier mixture controls
`(0,0)`, `(3.2,-1.5)`, and `(6.4,-3.0)`.  It failed before any reference or
performance result was produced: at \(H=0.20\), discrete-barrier calibration,
batch offset 0, two spot paths underflowed to zero.  A second run then found the
same failure mode for the terminal third component `(6.0,-3.0)`: five paths
underflowed at \(H=0.20\), calibration offset 34,816.  DCS requires finite strictly
positive spot and variance inputs, so retaining or silently clipping those paths
would have invalidated the estimator.

The failed run is retained outside the repository as a machine-readable development
receipt.  The P7 runner now rejects both nonfinite and nonpositive paths and writes a
non-overwriting failure receipt rather than emitting a partial success artifact.

For the same H=0.20 seed streams, the replacement third component `(4.5,-2.25)`
produced finite, strictly positive paths across all 131,072 calibration and 65,536
validation paths for both task families.  This numerical check alone does **not**
establish rare-event coverage or reference accuracy.  The corrected proposal proceeds
only to a new full independent calibration/validation run, whose probability-band,
simultaneous interval, precision, and likelihood-normalization gates remain binding.

## Corrected development result

The corrected full 24-cell run passes all seven calibration gates.  Its candidate
threshold-manifest SHA-256 is

```text
909d19dd5ae39375be22252e35c6afbb2ed78b4adafdd4ad2146a7d415c43c69
```

The preserved development artifact is
`results/g11_v8_p7_calibration_development_v1_2026-07-27.json`, with SHA-256

```text
141900b52fc8ae3580009e4b08afb4f10d4bbfebba3f37c3847b0a18aace578c
```

It records `dirty_worktree: true` and is therefore development evidence only.  In
particular, it is not a clean threshold freeze, not an independent reference, and
not a DCS-versus-baseline performance result.  A later clean-source run must bind a
threshold-manifest hash before any independent reference or final sample is allowed.
