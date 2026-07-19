# G11 M7 Freeze Readiness Decision

Date: 2026-07-19
Decision: **not frozen**; development checkpoint is ready to commit

## Scope eligible for a future frozen protocol

1. Primary (`H=0.12`) and high-H (`H=0.30`) regimes:
   terminal, barrier, and non-degenerate excursion tasks at the calibrated
   `10^-3`, `10^-4`, `10^-5`, and `10^-6` bands.
2. Low-H (`H=0.07`) regime:
   terminal and barrier tasks at all four bands. Excursion at `10^-6` is excluded
   because it failed the predeclared 1% occupation-exclusion gate. The negative
   calibration artifact remains in the audit manifest.
3. Fixed 128-step estimand only. No continuous-time label is permitted.
4. Three-component deterministic proposal with natural weight 0.10 and non-natural
   weights 0.30 and 0.60. Price shifts remain in one flat control span.
5. Float64 FFT production engine, independent pilot/final streams, minimum 32 pilot
   nonzero terms, 65,536 pilot cap, and allocation safety factor 4.0.

## Raw-baseline resource rule to freeze before execution

Rare raw corrections failed target RMSE in 9/12 development cells and produced
heavy-tail variance estimates that imply tens of millions of paths on some levels.
The confirmatory protocol must therefore declare a resource ceiling before any
confirmatory seed is opened.

- If both raw and DCS meet the target, report the ordinary matched-RMSE work ratio.
- If DCS meets the target but raw reaches the frozen ceiling first, report raw as
  resource-infeasible and provide only a censored lower bound. Do not assign an
  invented point speedup.
- The primary practical aggregate must state matched and censored cell counts.
- DCS target attainment must be at least 90% across core cells.
- A performance headline requires at least three matched cells, a geometric matched
  operation-work ratio above 1.25, and a one-sided seed-clustered uncertainty bound.
- Wall time and operation work remain separate endpoints. Resumed development wall
  time is not confirmatory timing evidence.

The numerical value of the raw resource ceiling and confirmatory repetition count
remain to be selected from available hardware capacity. Selecting them after looking
at confirmatory outcomes is prohibited.

## Preconditions still open

1. Commit all M0--M6 code, configs, tests, and development artifacts.
2. Put that exact source commit and every calibration artifact hash in the frozen
   config.
3. Generate and audit a new validation seed ledger namespace disjoint from Gaussian
   oracle, rate, rarity-calibration, common-event development, rare development, and
   smoke roles.
4. Choose the raw resource ceiling and confirmatory repetitions from an explicit
   CPU/GPU-hour budget.
5. Run only separate-namespace smoke tests after freezing.
6. Once the first confirmatory seed is inspected, any code change invalidates the
   protocol and requires a new version and seed namespace.

## Why the protocol is not auto-frozen here

Freezing an unaffordable raw ceiling or an arbitrary replication count would create a
statistical design defect, not rigor. The implementation is ready; the remaining
decision is a declared resource/statistical design choice that changes the
confirmatory claim. It must be made before execution and recorded explicitly.
