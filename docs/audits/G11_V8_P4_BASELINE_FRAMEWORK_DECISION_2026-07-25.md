# G11 V8 P4 Strong-Baseline Framework Decision

Date: 2026-07-25

Decision: **PASS FOR FRAMEWORK ORACLES; PERFORMANCE QUALIFICATION OPEN**

The bound ledger
`configs/g11_v8/baseline_framework_ledger_v1.yaml` has SHA-256:

```text
ebd314dc1b6f9d6fc0eb0f7285a912491fd5f4b562ac81b1ea564fd57ded81f0
```

P4 implements one frozen train/plan/estimate/audit boundary for eight methods:
crude, antithetic, conditional rBergomi, pure CEM, defensive CEM, smoothing RQMC,
large-deviation subspace IS, and exact-likelihood nonlinear coupling-flow IS.

The main technical corrections are:

- antithetic pair means, rather than individual dependent paths, are inferential
  units;
- independent RQMC randomizations, rather than within-net points, are inferential
  units;
- pilot cost counts raw points, not only pair/randomization counts;
- every IS density is evaluated exactly without clipping or self-normalization;
- analytic conditioning and quadrature calls are charged explicitly;
- trained methods bind realized work to a predeclared budget;
- GPU work requires billed-cost or energy backing; and
- the entangling flow is baseline-only and cannot be relabelled as DCS.

The nonlinear flow has an exact triangular inverse and log-Jacobian. This is a valid
flow density oracle, not evidence that one layer is the strongest flow architecture.

P4 does not contain achieved-RMSE results. Fresh task-tuned LD and flow training,
budget parity, hyperparameter parity, the frozen comparison matrix, and actual
performance remain open. P5 reference/matrix construction is authorized, while
every superiority claim remains prohibited.

Verification:

- 36/36 P4-targeted tests pass;
- 680/680 repository regression tests pass;
- all 38 independent framework-ledger checks pass;
- Ruff passes on the complete CI scope;
- Mypy passes on 85 source files; and
- `git diff --check` passes.
