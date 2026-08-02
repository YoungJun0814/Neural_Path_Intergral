# G11 V8 D1 Stage A Falsification and Remediation Audit

Date: 2026-08-02
Status: **Stage A mechanism continuation only; external-comparator remediation required**

## Decision

The representative-cell experiment supports continuation to a remediated Stage B
because the same-proposal raw/DCS mechanism passed and all exact-density checks
passed. It does **not** qualify any performance, P8, submission, or top-journal
claim.

The independent V2 audit passes every structural, seed, cost-arithmetic, aggregate,
and claim-lock check. The audit does not turn an exploratory result into a
confirmatory one.

## Frozen evidence

- valid V2 result:
  `results/g11_v8_d1_p7_falsification_stage_a_v2_2026-08-02.json`;
- independent V2 audit:
  `results/g11_v8_d1_p7_falsification_stage_a_audit_v2_2026-08-02.json`;
- invalid V1 output is preserved rather than repaired in place;
- V1 failure receipt burns all 504 V1 seeds and prohibits reuse of its namespace.

## What passed

| Check | Stage A observation | Interpretation |
|---|---:|---|
| paired raw/DCS records | 24 | four cells, two budgets, three clusters |
| external records | 114 | full declared Stage A roster |
| maximum exactness error | `4.62e-14` | below the frozen `1e-10` tolerance |
| external nonfinite weights | 0 | representable contribution path |
| flow inverse/Jacobian audit | pass | density implementation exactness only |
| geometric raw/DCS variance ratio | `2.229` | exceeds the frozen mechanism boundary `2.0` |
| Stage B mechanism continuation | pass | subject to remediation below |

The paired comparison uses the same frozen Gaussian-mixture proposal and the same
simulated paths. DCS differs only by analytically integrating the final independent
Gaussian label. Therefore the population identity

`Var(raw) - Var(DCS) = E[Var(raw | residual state)] >= 0`

is the appropriate mechanism statement. An individual small empirical cluster may
have a ratio below one without contradicting that identity; the predeclared Stage A
decision used the aggregate geometric ratio.

## What failed or remained blocked

1. Five primary records were resource-censored.
2. Only about 70.2% of likelihood-normalization diagnostics fell within the frozen
   absolute z-score boundary.
3. The original pure CEM implementation continued selecting a fixed extreme elite
   fraction after its empirical level had already entered the event. It therefore
   targeted an unintended stricter event and over-shifted the proposal.
4. A zero or near-zero rare-event pilot variance could allocate the minimum four
   final inferential units. That is an invalid achieved-precision inference, even
   though the final ordinary mean itself remains unbiased.
5. The one-layer flow retained exact likelihoods but collapsed empirically to zero
   estimates with extreme likelihood-normalization errors. It is stopped as a
   performance comparator for Stage B; its exactness test remains useful.
6. The DCS proposal-bank construction cost is not yet closed under a replayable
   common ledger and amortization rule.
7. Stage B, Stage C, T1, P8, P9, P10, and P11 are incomplete.

## Remediations implemented before a fresh stream

### Target-level CEM

CEM now advances through intermediate score levels only while the empirical elite
quantile is below the declared event level zero. Once that quantile reaches zero,
it fits the empirical event-conditional mean and stops. Actual completed iterations
are charged as training cost; the predeclared maximum-iteration work remains the
training budget ceiling.

This changes proposal efficiency but not estimator exactness: final data still use
an ordinary, non-self-normalized mean with the exact Gaussian density ratio.

### Fail-closed pilot allocation

The common executor now supports:

- a predeclared minimum number of nonzero pilot inferential units;
- a predeclared multiplicative pilot-variance safety factor;
- separate recording of empirical and planning variance; and
- an explicit `PilotSupportError` before final sampling when support is inadequate.

The pilot is independent of final data, so conservative sample-size selection does
not bias the final ordinary mean. Failed pilots must remain visible as failures and
their work must be charged.

## Verification

- focused baseline tests: 21 passed;
- complete repository regression: 853 passed;
- focused Ruff: passed;
- focused mypy: passed;
- independent D1 audit mutation tests: passed;
- non-standard JSON tokens: rejected by the audit loader.

## Authorization boundary

Authorized next action: open a fresh, disjoint remediation/Stage B development
namespace after freezing its config and source commit.

Not authorized: P8 qualification, a superiority statement, a journal-submission
claim, reuse of V1/V2 development outcomes as confirmation, or omission of failed
baseline work.

## V3 remediation-stream addendum

The clean-source V3 remediation stream is also closed as a failed protocol stream.
It completed 240 disjoint seeds and verified that target-level CEM, nonzero-pilot
support, and variance inflation execute correctly. However, the aggregate decision
code did not apply the already frozen `primary_accuracy_combined_z` threshold. Its
emitted Stage B authorization is therefore invalid and is superseded by
`g11_v8_d1_p7_falsification_stage_a_failure_receipt_v3_2026-08-02.json`.

The V3 numerical observations remain falsification evidence: fixed-identity pure
CEM can still miss important likelihood regions in the rarest cells, while
smoothing RQMC requires substantially more randomizations for some rare barrier
cells. The next protocol must retain those failures, apply accuracy only at a
predeclared operating budget rather than demanding that every budget-ladder point
succeed, and keep resource censoring separate from accuracy.
