# G11 V5 Full Threshold-Diagnostic Decision

Date: 2026-07-22

Decision: **G1 full development diagnostic passed; G2--G3 remain conditional.**

The non-smoke diagnostic was executed from clean detached source commit
`0836751f0be073b7fa1bceb5b2b65d31879e77d2`. It is a falsification experiment,
not a theorem qualification and not untouched confirmatory evidence.

## 1. Execution identity

- protocol: `g11-v5-threshold-diagnostics-development-v1`
- config SHA-256: `302b6efdd8adffd19e6174e4dcb5cdd7da548766b98940343bdd26ab6a4d3c61`
- seed-ledger SHA-256: `5d96475bec41ac9dd47280d6914f2a36a7a241309b0e78ac31f4947669965102`
- result SHA-256: `10e5314c48013975cf1023713f952e233396a8c696dffba435cbc5dfff7df6a3`
- design: 3 fixed-one-factor H regimes, 2 event types, 5 adjacent levels,
  8 replicates, and 4,096 paths per level
- source clean: true
- smoke: false
- continuous-time claim: false

Every pathwise threshold reconstruction and likelihood exactness check passed, and
the artifact contains no recorded failure.

## 2. Descriptive rate estimates

The slopes below are log-log descriptive exponents over levels 2--5. They are not
confidence-bounded theorem estimates.

| H | task | DCS correction second moment | raw correction second moment | threshold L2 | fine-only mesh L2 |
|---:|---|---:|---:|---:|---:|
| 0.05 | terminal | 0.0440 | 0.0127 | 0.0455 | n/a |
| 0.05 | barrier | 0.0471 | -0.0058 | 0.0606 | 0.6992 |
| 0.12 | terminal | 0.2172 | 0.0764 | 0.2161 | n/a |
| 0.12 | barrier | 0.2253 | 0.0688 | 0.2277 | 0.7172 |
| 0.30 | terminal | 0.5913 | 0.2521 | 0.5944 | n/a |
| 0.30 | barrier | 0.6939 | 0.3164 | 0.6848 | 0.8392 |

The DCS and threshold second-moment exponents are empirically close to a `2H`
pattern in the terminal task and in the two roughest barrier regimes. The barrier
mesh-enrichment defect decays more rapidly in this experiment, but that observation
does not establish a uniform bound.

## 3. Falsification and claim decision

The experiment does not immediately falsify the proposed localized threshold
mechanism. It also shows why a blanket fast-rate statement would be wrong: the
H=0.05 correction decay is extremely slow, and the raw barrier correction is flat
within Monte Carlo noise.

The next mathematical target is therefore a conditional finite-grid upper bound
`E[Delta_l^2] <= C h_l^beta`, where `beta > 0` must be derived by optimizing the
good-event coefficient error against the small-slope and mesh-enrichment bad-event
terms. The observed near-`2H` pattern is a hypothesis to test, not an exponent to put
into the theorem in advance. Promotion to an unconditional rBergomi theorem requires
proving the needed inverse-slope, coefficient, localization, and enrichment bounds
for the exact BLP coupling and normalized rank-one direction used by the
implementation.

This artifact does not close:

- inverse moments of the terminal affine slope;
- a coefficient-rate theorem for the implemented fine/coarse coupling;
- early-active barrier probability bounds;
- uniform fine-only monitoring-enrichment bounds; or
- any continuously monitored barrier statement.
