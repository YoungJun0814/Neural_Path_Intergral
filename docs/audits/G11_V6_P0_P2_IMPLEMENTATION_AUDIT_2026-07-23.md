# G11 V6 P0--P2 Implementation and Error Audit

Date: 2026-07-23

Branch at audit: `main`

Recorded pre-Snapshot-A HEAD:
`05ad9c54f1884e42bc69eb8797ef494b363e26c3`

Status of the evidence in this document: provisional evidence generated before the
formal clean-snapshot sequence

## 1. Executive verdict

The P0--P2 **software and protocol implementation** is substantially complete and
passes local unit, integration, strict-model, and smoke checks. The scientific
program is not formally complete:

- P0 still needs a clean-current-commit proposal-training artifact and a full
  independent 18-cell reference;
- P1 still needs the powered heterogeneous qualification, outcome-blind freeze,
  untouched confirmation, and Linux reproduction;
- P2 has an internally consistent terminal proof candidate and full diagnostic,
  but still needs an independent line-by-line mathematical review;
- the barrier theorem remains open by design.

This distinction is mandatory. A dirty or smoke artifact is engineering evidence,
not paper evidence.

## 2. Critical defect found and corrected

Four rBergomi paths silently applied `max(V,1e-10)`. That changes the mathematical
model from standard lognormal rBergomi to a floored-volatility model and invalidates
an unqualified continuous-model theorem claim.

The floor was removed from:

- `src/physics_engine.py`;
- `src/path_integral/rbergomi_fft.py`;
- `src/path_integral/rbergomi_coupling.py`; and
- `src/path_integral/rbergomi_branching.py`.

`strict_lognormal_variance` now:

- evaluates `xi*exp(log_factor)` in the log domain;
- rejects nonfinite inputs;
- rejects underflow/overflow outside the declared normal floating-point range; and
- never silently changes the target variance.

The full 18-cell recalibration under the strict target passed. Its thresholds match
the earlier floored-code calibration to at most about `2.2e-14` in this parameter
matrix, so the floor did not affect the observed calibration draws. This numerical
agreement does not excuse the old model-definition error; the correction remains
necessary for a theorem.

## 3. P0 audit

### P0.1 Full 18-cell calibration

Status: **provisional pass; formal rerun required**

The strict run covered `3 H x 2 tasks x 3 rarity bands = 18 cells`, with 524,288
calibration and 524,288 independent validation paths per Hurst model. All six gates
passed:

- complete matrix;
- point probability bands;
- simultaneous probability bands;
- relative-SE limits;
- likelihood normalization; and
- calibration/validation seed-role separation.

Artifact:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_P0_artifacts\g11_v6_calibration_full_strict_provisional_2026-07-23.json`

SHA-256:

`c59dc0d6ac3a25c26c1ed92251d7f302c22ea6517e650eefa787640e8d1e2010`

It is provisional because `dirty_worktree=true`. The older clean calibration is
preserved for audit history, but it came from the pre-correction source and cannot
serve as the final strict-model artifact.

### P0.2 Full independent reference

Status: **full 18-cell reference pass**

The reference runner provides:

- independent DCS and raw estimators;
- pilot/final seed separation;
- likelihood normalization;
- combined-z agreement;
- SE contracts;
- hard maximum sample caps;
- durable per-cell checkpoints; and
- fail-closed resource censoring.

A qualification-config smoke run completed and correctly remained unqualified
because smoke caps cannot satisfy reference precision. The full accepted 18-cell
matrix can require up to 20 million final paths per cell and was not represented as
completed.

The first clean full attempt under `g11-v6-reference-qualification-v1` hit its
predeclared resource stop: the `h005-terminal-p1e-04` DCS planning request was
`40,440,463`, above the `20,000,000` cap. The run was stopped and preserved rather
than completing a protocol that could no longer pass. This is a reference-proposal
efficiency failure, not a V6 accuracy result. The corrective V2 reference keeps the
same SE/cap contract and uses the already frozen reference-free task-conditioned
proposal bank under a new seed namespace. See
`G11_V6_REFERENCE_V1_RESOURCE_FAILURE_2026-07-23.md`.

The replacement `g11-v6-reference-qualification-v2` retained every accuracy and
resource contract, reused the pre-qualification task-conditioned defensive bank,
and passed all 18 cells with both DCS and raw estimators. Its 36 method runs had no
resource censoring; the maximum requested final count was `458,389`, maximum
achieved/target SE was `0.824`, maximum absolute agreement z was `1.767`, and
maximum absolute normalization z was `2.536`. The artifact SHA-256 is
`786173bd1316d882f8abe049abbd2f12fc05c681864c8367b61c5cd612f73db7`.
See `G11_V6_REFERENCE_V2_DECISION_2026-07-23.md`.

### P0.3 Training/calibration/qualification/confirmation separation

Status: **implemented; provisional audit pass**

`g11_v6_split_audit.py` reconstructs every `SeedLedger`, verifies its embedded hash,
and rejects either:

- a repeated numerical seed; or
- a repeated protocol namespace

across declared research stages. Equal cell definitions are allowed only when the
canonical seed blocks are disjoint.

The provisional training/calibration audit passed:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_P0_artifacts\g11_v6_split_audit_training_calibration_provisional_2026-07-23.json`

It must be rerun after clean-current-commit proposal training, qualification, and
confirmation artifacts exist.

### P0.4 V3 proposal source and publication archive

Status: **implementation pass; provisional archive pass**

The V3 source audit:

- hashes the raw pure-CEM training artifact;
- rejects incomplete or non-pure-CEM source records;
- derives separate terminal/barrier median controls;
- reconstructs zero/half/full proposal banks;
- verifies every configured control to absolute tolerance `1e-9`; and
- reconciles samples, work, wall time, and CPU time to the source ledger.

Formal qualification and protocol freeze additionally require the training source
itself to declare a clean committed non-smoke provenance. A hash match alone is not
treated as sufficient. `g11_v6_materialize_proposal_bank.py` derives the V3 controls,
source hash, exact totals, and amortization count directly from that artifact.
`g11_v6_materialize_qualification_suite.py` then atomically derives the full policy,
selector-off policy, and fixed secondary-baseline configs from the same proposal
bank, so nondeterministic wall/CPU values are never copied by hand.

The final source contract is now stricter than the provisional baseline-derived
artifact. `g11_v6_proposal_training.py` trains only on the two prespecified
`H=0.12`, `p=1e-3` development cells, with three independent CEM fits per task. It
does not load a reference artifact or any qualification outcome. Its formal artifact
must reconstruct both strict ledgers, cover both task families, contain only
converged finite bounded fits, conserve every training-cost field, and carry clean
committed provenance. Qualification materialization rejects the older combined
baseline/evaluation schema as a formal source, although that schema remains readable
for historical development audits.

The shared training cost now uses exact quotient/remainder apportionment. This fixes
a real defect: 196,608 training samples are not divisible by the full confirmation
record count `18*64=1,152`. Integer samples and floating work totals are now
conserved exactly without dropping or inventing cost.

The deterministic ZIP builder fixes timestamps and permissions, hashes every entry,
replays every archived entry, and refuses overwrites. The final provisional replay
archive for this session verified the source contents, explicitly recorded
`proposal_training_source_formal=false`, and contains ten entries:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_P0_artifacts\g11_v6_publication_archive_provisional_v3_2026-07-23.zip`

Its adjacent `.receipt.json` records the archive and internal-manifest SHA-256
values. The hash is intentionally not embedded in this document because this
document is itself an archive entry.

The source training artifact itself was generated from a dirty older development
state. Therefore this archive is a replay test, not the final publication archive.

During the clean-snapshot audit, a V6-only CEM configuration defect was found before
formal proposal training. The implementation interprets `elite_quantile` as the
score CDF quantile and retains `score >= quantile`; consequently `0.90` selects the
intended upper approximately 10%, whereas the V6 files had declared `0.10` and
retained approximately 90%. All V6 baseline/training configs were corrected to
`0.90`, and a regression assertion now fixes that convention. No formal calibration,
proposal-training, qualification, or confirmation artifact generated while this
correction was uncommitted is admissible.

### P0.5 Mandatory baselines and ablations

Status: **implemented and smoke-audited; full matrix pending**

The first full baseline protocol (`g11-v6-baseline-qualification-v2`) was stopped
after 74 durable records. Its one-look `B^2` Hoeffding variance upper bound implied
about 1.45 billion final paths for every `1e-4` defensive-CEM record even under an
all-zero pilot, so resource censoring was mathematically unavoidable. The preserved
failure and the non-leaking structural second-moment correction are documented in
`G11_V6_BASELINE_V2_RESOURCE_DESIGN_FAILURE_2026-07-23.md`. The replacement V3
protocol uses the exact inequality
`Var(1_A dP/dQ) <= B P(A)`, a frozen rarity-band upper envelope, a four-standard-error
independent-reference eligibility certificate, a new protocol/seed namespace, and
an independently replayed design certificate. No V2 result is reused.

V3 subsequently stopped after 134 complete records when one crude final allocation
missed its empirical sampling-variance target by 8.69%. The pointwise 95%
Clopper--Pearson pilot design was not adequate for an all-record matrix gate. V4
extends the same frozen rarity-band contract to crude MC via
`Var(1_A) <= p_upper(1-p_upper)` while leaving the primary pure-CEM comparator
unchanged. See
`G11_V6_BASELINE_V3_EMPIRICAL_RMSE_FAILURE_2026-07-23.md`.

V4 then verified the crude correction but showed that using the worst-case
`B p_upper` envelope as the defensive-CEM allocation oversampled a `1e-4` sentinel
record by roughly 610 times relative to the minimum final count. V5 retains the
structural envelope as an audited diagnostic and gives defensive CEM the same
fivefold independent-pilot variance rule as pure CEM. This is a strong-baseline
fairness correction, not a change to the primary comparator. See
`G11_V6_BASELINE_V4_OVERCONSERVATIVE_DESIGN_2026-07-23.md`.

The mandatory comparison design is now:

| Contrast | Isolated contribution |
|---|---|
| crude vs pure CEM vs defensive CEM | importance-sampling training and defensive mixture |
| fixed raw defensive vs fixed DCS-SLIS | conditional smoothing |
| fixed DCS-SLIS vs selector-off router | rarity routing |
| selector-off router vs full V6 | capped hybrid selector |

The smoothing-off estimator uses the exact defensive-mixture contribution, not
self-normalization. Proposal training is charged under an explicit standalone
per-method amortization contract.

All qualification-config smoke runs:

- completed their matrices;
- were uncensored;
- attained design targets;
- passed independent record audits; and
- recorded positive CPU and absolute process peak RSS.

The full routed-policy smoke missed one empirical RMSE gate, while the selector-off
and secondary-pair smokes attained it. This is expected to be noisy with one smoke
cluster and is not promoted into a model comparison.

## 4. P1 audit

### P1.1 Heterogeneous qualification

Status: **configs and runners ready; scientific execution pending**

The source-independent qualification designs cover all 18 cells and 24 independent
clusters. Proposal-dependent V3 configs are deliberately materialized only after
clean proposal training. Together they cover:

- crude, pure CEM, and defensive CEM;
- full V6;
- selector-off V6; and
- fixed DCS-SLIS and fixed raw defensive.

Every result preserves incomplete/censored records; no complete-case deletion path
is available.

### P1.2 Power and resource decision

Status: **implemented and unit-tested; awaits full qualification inputs**

The power analysis:

- uses identical `(cell,cluster)` pairs;
- averages log work ratios equally over cells inside each cluster;
- applies predeclared effect shrinkage and SD inflation;
- computes a paired normal-approximation cluster requirement;
- applies per-cell attainment and bootstrap-RMSE co-gates; and
- plans 64 confirmation clusters only if every gate passes.

The resource decision was strengthened in this audit. It now requires positive wall,
CPU, and peak-RSS records, then takes the more conservative of:

- a predeclared work-throughput projection; and
- an observed p90 wall-time projection from qualification.

Projected CPU hours and the maximum absolute process high-water RSS are reported
separately. Peak RSS is not falsely treated as additive or baseline-subtracted
per-method memory.

### P1.3 Outcome-blind freeze

Status: **implemented; intentionally blocked while dirty**

The manifest freeze preserves every calibrated cell byte-for-byte at the semantic
level and records both the calibration source commit and freeze-tool provenance.
The protocol freeze:

- requires a clean tree;
- requires a qualified full reference;
- requires a passing full power artifact;
- checks manifest/reference estimand identity;
- verifies the V3 raw proposal source;
- rejects noninteger planned cluster counts; and
- hashes every frozen config and input.

The current provisional manifests explicitly record
`formal_freeze_tool_readiness=false`; they are not silently presented as formal.

### P1.4 Untouched confirmation and Linux reproduction

Status: **implemented but not executed**

The confirmatory analyzer is fail-closed on frozen hashes, identical complete pair
sets, accuracy co-gates, independent audits, and no resource censoring. The hardware
reproduction analyzer verifies a different OS, frozen protocol hashes, disjoint
seeds, scientific-gate agreement, and effect compatibility.

Neither can legitimately run before P0 reference and P1 qualification/power pass.

## 5. P2 audit

Status: **strict implementation and full provisional diagnostics pass; proof is not
yet journal-ready**

Completed:

- strict unfloored rBergomi variance;
- exact direction/coarse-pair convention checks;
- analytic inverse terminal-slope moment bound;
- raw coefficient moment diagnostics;
- terminal candidate rate contract for every `r<H`;
- threshold and DCS second-moment corollaries;
- conservative weak-bias/FFT-MLMC exponent ledger; and
- explicit barrier exclusion.

The full strict provisional diagnostic produced 432 records and passed every
implementation gate. It remains formally unqualified only because the source was
dirty.

The detailed mathematical review is in:

`docs/theory/G11_V6_TERMINAL_COEFFICIENT_PROOF_AUDIT_2026-07-23.md`.

The exact honest status is:

- inverse-slope theorem: internally proved;
- coefficient and continuous weak-rate theorem: proof candidate awaiting independent
  mathematical review;
- terminal empirical diagnostic: passed as falsification evidence;
- barrier theorem: open.

## 6. Local validation ledger

The following checks passed during P0--P2 implementation:

- freeze/routing targeted tests;
- manifest-freeze and split-audit tests;
- deterministic archive tests;
- secondary-baseline and smoothing-off tests;
- resource-supplement tests;
- 44 strict rBergomi/coupling tests;
- terminal theory unit tests;
- qualification-config end-to-end smoke runs;
- independent offline result audits; and
- Ruff checks on modified P0--P2 modules.

The final full-suite count is recorded only after the concluding regression run in
this work session; no earlier baseline count is reused as if it covered new code.

## 7. Formal execution sequence after commits are permitted

Two clean snapshots are required because the final proposal bank is itself an output
of development training.

### Snapshot A: code and development protocol

1. Commit the reviewed code and development templates.
2. Assert `git status --porcelain` is empty.
3. Rerun full strict 18-cell calibration.
4. Rerun clean task-conditioned pure-CEM proposal training.
5. Generate and archive the resulting source hash, controls, and exact cost totals.
6. Rerun the split audit over training and calibration.

### Snapshot B: frozen qualification protocol

1. Materialize all proposal-dependent V3 qualification configs from Snapshot A
   artifacts with `g11_v6_materialize_qualification_suite.py`.
2. Commit those frozen configs without changing estimator code.
3. Assert the tree is clean.
4. Freeze qualification and confirmation manifests with receipts.
5. Run the full independent 18-cell reference with durable checkpoints.
6. Build the clean publication archive and replay every hash.
7. Run 18-cell x 24-cluster qualification for every mandatory method/ablation.
8. Run independent audits and resource supplements.
9. Run heterogeneous paired power and accuracy co-gates.
10. Stop if any P0/P1 gate fails.

### Outcome-blind confirmation

1. Generate 64-cluster confirmation configs only from the passing power artifact.
2. Commit/freeze hashes before reading confirmation outcomes.
3. Run the complete confirmation matrix without deleting failures.
4. Run the confirmatory analyzer and independent audits.
5. Repeat the frozen protocol on clean Linux/CPU with disjoint seeds.
6. Publish terminal theory only after the independent proof review.

## 8. Stop rules

- Dirty source: no formal artifact.
- Smoke flag: no scientific gate.
- Strict variance failure: failed run; never clamp and continue.
- Full reference censoring: cell cannot enter a headline comparison.
- Any incomplete pair: no deletion and no power calculation.
- Qualification accuracy or resource failure: no confirmation freeze.
- Negative or underpowered efficiency direction: no superiority confirmation.
- Independent proof objection to O1--O5: conditional theorem only.
- Barrier diagnostic success without proof: finite-grid empirical result only.

## 9. Final P0--P2 classification

| Phase | Software/protocol | Provisional evidence | Formal scientific completion |
|---|---|---|---|
| P0 | complete | substantial pass | no |
| P1 | complete | smoke pass | no |
| P2 terminal | complete proof candidate | full diagnostic pass | no, independent review pending |
| P2 barrier | diagnostics only | H=0.30 instability observed | no |

This document records the pre-formalization audit. Later Snapshot A/Snapshot B
commits do not retroactively promote any dirty artifact listed here.
