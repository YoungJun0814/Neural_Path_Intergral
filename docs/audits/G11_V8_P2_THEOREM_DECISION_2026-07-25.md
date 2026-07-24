# G11 V8 P2 Finite-Grid Theorem Decision

Date: 2026-07-25

Decision: **PASS for finite-grid C0--C3; P3 and submission theory remain open**

## 1. Bound artifacts

The theorem ledger is `configs/g11_v8/theorem_ledger_v1.yaml`, SHA-256:

```text
94301f1680805daec34e8219700a303dc71154f06ea758eb0a76adece2e8c2cb
```

The ledger separately binds the P0 claim contract, P1 novelty ledger, and P2 proof
document hashes. The independent audit passes all 28 checks.

## 2. Theoretical result

P2 proves, within the declared finite-dimensional scope:

1. the exact target-over-defensive-mixture likelihood and its \(1/\delta\) bound;
2. the exact proposal-conditional DCS identity after full likelihood cancellation;
3. the total-variance decomposition;
4. strict variance improvement for a finite scalar threshold; and
5. exact terminal and discretely monitored barrier threshold maps.

For a residual \(r\), threshold \(a\), target probability \(p=\Phi(a)\), proposal
conditional event probability \(s\), and residual likelihood \(\bar L\), P2 proves

\[
\operatorname{Var}_Q(Y\mid R=r)
\geq
\bar L(r)^2p(r)^2\frac{1-s(r)}{s(r)}.
\]

Positive Gaussian-mixture weights and finite thresholds imply \(0<p,s<1\), making
the bound strictly positive. This is stronger than the earlier non-increase
statement, but it is not yet a uniform rare-event variance ratio or mesh-rate
theorem.

## 3. Technical result

The log-space certificate:

- reconstructs posterior component weights from the exact residual mixture;
- computes \(\log\Phi(a)\), \(\log s\), and \(\log(1-s)\) stably;
- remains finite for tested thresholds down to \(-40\);
- agrees exactly with Bernoulli variance when \(Q=P\);
- lies below the exact conditional variance gap computed by independent SciPy
  quadrature for nontrivial mixtures;
- handles \(+\infty\) and \(-\infty\) without a false strictness claim; and
- reuses the production residual likelihood exactly.

The terminal and barrier tests verify closed-event ties, finite thresholds, initial
barrier hits, and fail-closed rejection of zero post-initial slopes.

The completed P2 verification record is:

- 30/30 P2-targeted tests passed;
- 605/605 repository regression tests passed;
- the repository CI-scope Ruff check passed;
- the repository Mypy check passed; and
- `git diff --check` passed.

## 4. Corrected prior-document error

The V7 Rao--Blackwell mechanism document omitted a plus sign in the written
total-variance formula. It incorrectly displayed the two nonnegative terms without
an operator. P2 restores

\[
\operatorname{Var}(Y)
=\operatorname{Var}(Y_{\mathrm{DCS}})
+E[\operatorname{Var}(Y\mid R)].
\]

The V7 code and empirical variance-decomposition computation already used the
correct arithmetic; this was a documentation error, not an estimator-code error.
A regression test now prevents recurrence.

## 5. Claim promotion

| Claim level | P2 status |
|---|---|
| C0 finite-grid target | proved |
| C1 finite-grid exactness | proved |
| C2 variance non-increase | proved |
| C3 strict improvement | proved for finite scalar thresholds |
| C4 model mesh/rate theorem | open |
| C5 end-to-end MLMC complexity | prohibited until P3 obligations pass |

## 6. Remaining theoretical risks

P2 does not establish:

- continuously monitored barrier exactness;
- nonlinear rough Bergomi weak bias;
- the fine-only barrier-crossing mesh term;
- a model-level correction-variance exponent;
- a per-sample cost exponent;
- a matching rare-event asymptotic ratio; or
- training-inclusive superiority.

The novelty of the strictness theorem also remains subject to the external expert
review required by P1. If P3 fails to add a meaningful rough-model rate or complexity
result, the strongest mathematical-finance journal route remains doubtful even
though P2 is correct.

## 7. P2 gate

P3 mesh, weak-bias, and complexity implementation is authorized. Submission theory,
continuous-time language, and top-journal completeness are not authorized.
