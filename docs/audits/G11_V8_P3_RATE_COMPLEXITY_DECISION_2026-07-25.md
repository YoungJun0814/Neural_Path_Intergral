# G11 V8 P3 Rate and Complexity Decision

Date: 2026-07-25

Decision: **CONDITIONAL PASS WITH SCOPE DOWNGRADE**

## 1. Bound artifact

The P3 ledger is `configs/g11_v8/rate_complexity_ledger_v1.yaml`, SHA-256:

```text
a2e5e617077d26b06913fabffde290347482c3ad9db7a3909591dcf77d0cba22
```

It binds the P2 theorem ledger and the P3 proof-audit document. The independent
machine audit evaluates 46 checks.

The completed P3 verification record is:

- 55/55 P3-targeted and adjacent-regression tests passed;
- 644/644 repository regression tests passed;
- all 46 rate-ledger audit checks passed;
- the repository CI-scope Ruff check passed;
- the repository Mypy check passed; and
- `git diff --check` passed.

## 2. What P3 establishes

P3 establishes:

1. an exact pathwise signed terminal/barrier threshold decomposition into a
   coarse-active coefficient term, common-grid active-index switch, and fine-only
   mesh enrichment;
2. a fail-closed MLMC algebra certificate with separate
   \((\alpha,\beta,\gamma)\) evidence provenance;
3. the correct FFT logarithmic factors in every MLMC regime;
4. a conditional terminal rBergomi chain at every \(r<H\); and
5. an explicit refusal to transfer that chain to barriers.

The previous `maximum_exact_decomposition_violation` diagnostic only checked an
absolute deterministic upper bound. P3 adds the actual signed three-term identity
and makes the legacy field require both checks. This was a diagnostic semantic and
coverage defect, not an estimator bias.

## 3. Terminal verdict

The terminal proof candidate uses

\[
\alpha=r,\qquad
\beta=2r,\qquad
\gamma=1,\qquad
\kappa=1,\qquad
r=H-\varepsilon_H.
\]

Since \(r<H<1/2\), the conditional complexity algebra is

\[
O\!\left(\epsilon^{-1/r}\log(\epsilon^{-1})\right).
\]

This is not an \(O(\epsilon^{-2})\) recovery result. The weak-bias and
correction-variance premises remain conditional until the continuous/discrete
coupling, coefficient decomposition, and continuous-direction obligations are
proved line by line and independently reviewed.

## 4. Barrier verdict

The discretely monitored barrier retains:

- exact finite-grid likelihood and threshold identities;
- strict same-grid Rao--Blackwell improvement; and
- exact coefficient/active/mesh diagnostics.

It does not yet have:

- an early-active-time small-slope rate;
- a common active-index switch rate;
- a fine-only mesh-enrichment rate;
- a continuous-monitoring weak-bias rate; or
- an end-to-end complexity theorem.

The barrier is therefore a finite-grid experimental secondary task only.

## 5. Claim gate

| Claim | P3 decision |
|---|---|
| finite-grid signed decomposition | proved pathwise |
| terminal model rate | conditional |
| barrier model rate | open |
| terminal sampling-only MLMC complexity | conditional |
| barrier sampling-only MLMC complexity | prohibited |
| training-inclusive end-to-end complexity | open |
| training-inclusive superiority | open experimental question |

P4 strong-baseline implementation is authorized. Submission-level complexity
language, unconditional terminal rate claims, and every barrier rate claim remain
unauthorized.
