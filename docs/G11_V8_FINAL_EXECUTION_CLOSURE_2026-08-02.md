# G11 V8 Final Execution Closure

Date: 2026-08-02

Final status: **all work authorized by the frozen V8 gates is complete; V8 is closed
by scientific falsification, not by an implementation failure.**

## 1. Meaning of “complete”

The implementation plan contained conditional branches. Completing the plan does not
mean running later phases after an upstream gate fails. It means:

1. executing every authorized phase;
2. independently auditing its theory, code, seeds, costs, and outputs;
3. applying the predeclared pass/fail criteria without lowering them; and
4. stopping downstream phases when their preconditions are false.

V8 reached that terminal condition. Stage C and P8--P11 were not omitted; Stage B
formally prohibited them.

## 2. Phase outcomes

| Phase | Outcome | Main evidence |
|---|---|---|
| R2 independent references | passed | high-precision reference and independent aggregate audit |
| B1 external baselines | passed as implementation | defensive CEM, pure CEM, smoothing RQMC, exact-density flow and correctness oracles |
| D1 Stage A | passed after invalid namespaces were preserved and superseded | mechanism ratio 2.2027; primary max z 3.4381; Stage B authorized only |
| Costed DCS bank | passed | 24 cells, 48 converged training replicates, 407,552 paths, 199 updates, all work retained |
| D1 Stage B | scientifically failed | full 24-cell by 2-cluster matrix with 672 disjoint seeds |
| Independent Stage B audit | passed | recomputed the same failure; no implementation discrepancy found |
| T1 quantitative theorem | internally passed at finite-grid level | explicit rarity-dependent population variance-gap certificate and quadrature oracle |
| T1 terminal rate | internally closed for O1--O5 | terminal only, every (r<H); O6 external review remains |
| T1 novelty refresh | passed as an audit, blocked as a claim | seven new primary sources; top-journal and submission claims remain unauthorized |
| Final closure audit | passed | ten artifact hashes and 22 semantic/stop checks |

## 3. Binding Stage B facts

The full Stage B result is scientifically useful but negative for the broad V8
hypothesis:

- exactness passed; maximum error (4.55\times10^{-13});
- DCS absolute accuracy passed; maximum combined z (2.0487);
- proposal-bank training cost was closed;
- all external weights were finite;
- geometric raw/DCS variance ratio was (1.9086417<2.0);
- the strongest primary comparator accuracy z was (4.4115>4);
- 24 primary smoothing-RQMC records were resource-censored; and
- Stage C, P8, P9, P10, P11, and the current submission route were not authorized.

The independent audit reproduced these facts. Therefore the failed gate is not
currently attributable to a seed collision, omitted likelihood, self-normalization,
cost deletion, or aggregation error.

## 4. Strongest surviving result

For a finite Gaussian shift mixture, the implemented DCS estimator exactly integrates
one event-driving Gaussian coordinate under the actual proposal-conditional law. The
new moment-localized theorem turns abstract localization constants into mixture
weights, projected and residual controls, a threshold moment, and rarity radii:

\[
\operatorname{Var}(Y_{raw})-\operatorname{Var}(Y_{DCS})
\ge
\frac{\eta(A,R_0)}{M(R_0)}
\Phi(-A)^2\Phi(-A-B_\parallel)>0.
\]

This is an absolute, rarity-dependent finite-grid lower bound. It does not imply a
uniform variance ratio or bounded relative error.

For terminal rBergomi events, the internal proof supports the conservative candidate

\[
\alpha=r,\qquad\beta=2r,\qquad
\text{work}=O(\varepsilon^{-1/r}\log\varepsilon^{-1}),
\qquad r<H.
\]

Barriers remain outside this rate theorem.

## 5. Final verification

The final repository verification completed on the laptop environment:

- `python -m pytest -q`: **888 passed** in 111.39 seconds;
- focused T1 theory/novelty regression: 50 passed before closure additions;
- final closure-specific regression: 9 passed;
- Ruff on all newly changed Python files: passed;
- mypy on all newly changed source/experiment modules: passed;
- `git diff --check`: passed;
- T1 novelty update SHA equals the SHA recorded by its audit;
- final closure ledger SHA:
  `674ff92041637276720fb74cf9929bcfbd68f75d4a0af27a0a06cb2475533169`;
- final closure audit: all checks passed.

Canonical closure artifacts:

- `configs/g11_v8/completion_status_ledger_v3.yaml`;
- `results/g11_v8_completion_status_audit_v3_2026-08-02.json`;
- `experiments/g11_v8_completion_status_v3_audit.py`;
- `tests/test_g11_v8_completion_status_v3_audit.py`.

## 6. Current academic level

The repository now demonstrates doctoral-level research engineering:

- exact finite-dimensional stochastic identities;
- strong negative and corruption testing;
- independent seed and artifact audits;
- training-inclusive cost accounting;
- strong external baselines;
- falsification-first execution; and
- an honest negative gate outcome.

It is not yet a top-journal submission because the broad performance hypothesis
failed, the surviving theorem has not received independent mathematical review, the
barrier rate is open, and the final subscription-database novelty challenge is
missing.

## 7. Only valid next research branch

Any further computation must be a new V9 protocol, not a continuation that silently
changes V8. The most defensible candidate is terminal-only and regime-conditional:

1. define a new terminal-only estimand and explicit repeated-query amortization regime;
2. derive or externally validate the terminal threshold/intercept constants;
3. predeclare a hypothesis weaker and more scientifically meaningful than a uniform
   factor two;
4. use new seed namespaces and new reference bindings;
5. benchmark DCS against conditional lognormal MC, smoothing RQMC, and CEM under
   total work; and
6. seek independent proof and novelty review before qualification.

No current artifact may be relabelled as V9 confirmation data.
