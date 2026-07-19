# G11 M7 Local Execution and Resource Protocol

Date: 2026-07-19
Hardware: ASUS TUF Gaming A16, Ryzen 9 8940HX, 16 physical cores / 32 logical
processors, 16 GB RAM, Windows 11
Decision: local CPU execution is qualified and the protocol is freeze-ready;
confirmatory seeds remain unopened

## 1. Outcome

The notebook can execute the frozen finite-grid G11 estimator. The production path is
CPU-only and float64, so the installed RTX 5070 Laptop GPU is intentionally unused.
The M7 runner, raw-baseline censoring rule, multi-regime task selection, provenance
checks, and disjoint qualification namespace are implemented.

The confirmatory protocol contains 640 paired cells: 32 regime/task/rarity cells and
20 independent seed clusters. No M7 confirmatory cell has been executed or inspected.

## 2. Local hardware measurements

The same one-cell development smoke protocol was run with identical config and seeds.
Outputs were written outside the repository.

| Requested threads | PyTorch threads | Estimator wall seconds | Process wall seconds |
|---:|---:|---:|---:|
| 8 | 8 | 8.574 | 28.143 |
| 16 | 16 | 10.264 | 18.100 |
| 32 | 16 | 10.449 | 18.131 |

The 32-thread request was capped by PyTorch at the 16 physical-core count. The
estimator portion was fastest with eight threads, so M7 freezes eight PyTorch/OpenMP
threads. Process-wall differences on this short smoke are dominated by interpreter
and import startup and are not used as estimator evidence.

A separate 8-thread process probe observed a peak working set of approximately
0.776 GB for the one-cell smoke. This does not prove the peak for the full protocol,
but the production sampler is chunked at 4,096 paths, so memory is bounded by a batch
rather than the total final allocation. The 16 GB notebook is acceptable provided
other memory-heavy applications are closed.

## 3. Resource calculation

The completed 12-cell rare-development matrix used 2,322.85 measured estimator wall
seconds. Its primary-H, natural-weight-0.10 endpoint subset required about 1,459
seconds for six cells. Adding the intermediate rarity bands and the low/high-H regimes
gives a planning range of roughly 2--3 wall hours per 32-cell seed cluster on the
notebook. This is an engineering extrapolation, not a result.

The frozen resource choices are therefore:

| Quantity | Frozen value | Reason |
|---|---:|---|
| PyTorch/OpenMP threads | 8 | Fastest measured estimator configuration; lowers heat |
| Confirmatory seed clusters | 20 | Enables seed-clustered uncertainty rather than a one-off matrix |
| Raw final paths per level | 10,000,000 | Predeclared baseline ceiling; DCS is not capped |
| Total process CPU budget | 512 CPU-hours | Covers the 320--480 CPU-hour planning range with limited reserve |
| Relative RMSE target | 20% | Same matched-precision target as rare development |
| Finest grid | 128 steps | Keeps every claim finite-grid explicit |

Actual process CPU time is recorded with `time.process_time()`. The old approximation
`wall time x configured thread count` is not used for budget enforcement.

If 512 process CPU-hours are exhausted, the runner stops at a completed cell boundary,
marks the protocol incomplete, and prohibits a performance headline. The budget may
not be enlarged after confirmatory outcomes are inspected under the same protocol ID.

## 4. Frozen task matrix

| Regime | Included tasks | Rarity bands | Cells per seed cluster |
|---|---|---|---:|
| low H=0.07 | terminal, barrier | 1e-3, 1e-4, 1e-5, 1e-6 | 8 |
| primary H=0.12 | terminal, barrier, excursion | 1e-3, 1e-4, 1e-5, 1e-6 | 12 |
| high H=0.30 | terminal, barrier, excursion | 1e-3, 1e-4, 1e-5, 1e-6 | 12 |

All low-H excursion cells are excluded. In particular, its 1e-6 cell failed the
predeclared 1% occupation-exclusion gate. The negative calibration artifact remains
tracked and is not repaired post hoc.

Thresholds are loaded from the three byte-hash-pinned calibration artifacts. The
runner also verifies their config hashes, finite-grid declaration, non-smoke status,
normalization, probability, precision, and seed-role gates before allocating a seed.

## 5. Raw-baseline censoring semantics

The MLMC mathematics module is unchanged. After an ordinary independent pilot creates
the required allocation, only a raw-baseline level above 10,000,000 final paths is
truncated by the orchestration layer.

Every cell exposes three distinct fields:

1. `allocated_work_ratio_raw_over_dcs`: diagnostic work spent, regardless of target;
2. `matched_work_ratio_raw_over_dcs`: populated only when both methods meet RMSE; and
3. `censored_work_ratio_lower_bound`: populated only when DCS meets RMSE and capped raw
   does not.

A censored ratio is never inserted into the matched geometric mean or the
seed-clustered speedup interval. This prevents a resource failure from becoming an
invented point speedup.

## 6. Statistical gates

A performance headline requires all of the following:

- all 640 cells completed with no unexpected execution failure;
- DCS target attainment of at least 90%;
- at least three matched-target cells;
- matched-target geometric operation-work ratio above 1.25;
- at least ten replicate clusters contributing matched ratios; and
- a one-sided 95% seed-clustered lower confidence bound above 1.0.

The cluster statistic takes the geometric mean of matched cell ratios within each
replicate, then applies a one-sided Student-t interval to log cluster ratios. Matched
and censored cell counts are always reported separately.

## 7. Source and seed freeze

The estimator core is pinned to commit
`e358a562df9637eeaa6db5826307471bf5c6e2ce` and ten core source-file SHA-256 values.
The protocol itself is pinned by the Git tag `g11-m7-confirmatory-v1-freeze`, avoiding
the circular impossibility of embedding a commit hash inside the commit it identifies.

The qualification protocol ID is `g11-m7-local-qualification-v1`; the confirmatory ID
is `g11-m7-confirmatory-v1`. Because the full protocol identifier is part of every
canonical seed key, qualification and confirmatory streams are disjoint.

Confirmatory execution additionally requires:

- HEAD exactly equals the frozen tag;
- a clean Git worktree;
- eight PyTorch threads; and
- output/progress paths outside the clean source worktree.

## 8. Qualification result

The one-cell qualification completed successfully in its non-confirmatory namespace:

- raw allocation was deliberately capped and missed target;
- raw was classified as resource-censored;
- DCS met target;
- matched ratio remained null;
- the censored lower-bound field was populated;
- two independent method ledgers were hashed; and
- no unexpected failure occurred.

This validates orchestration semantics only. It is not rare-event performance
evidence and must not appear in a paper result table.

## 9. Operational rules for the notebook run

1. Run on AC power in the highest-performance thermal profile.
2. Close browsers, notebooks, and other memory-heavy programs before starting.
3. Disable sleep and scheduled restart for the run window.
4. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 8 before
   Python starts.
5. Run from a detached clean Git worktree at the frozen tag.
6. Store output and progress outside that worktree.
7. Do not edit code, config, calibration artifacts, or the Git tag after the first M7
   seed is executed.
8. Back up the progress and final JSON before any later cloud reproduction.

## 10. Current decision

The local environment and M7 orchestration passed the 321-test full suite and focused
Ruff/mypy checks. The protocol is ready for its dedicated commit and freeze tag.
Confirmatory execution begins only from a clean worktree at that tag.
