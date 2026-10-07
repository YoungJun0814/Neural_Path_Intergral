# Post-audit claim ledger: V16 and R0

Date: 2026-09-28. This document supersedes interpretation of the frozen historical `configs/g11_v15/theorem_ledger_v1.yaml`; it does not rewrite that file or any historical result. It records the current review status of each claim, not a formal proof assistant certificate.

## Status vocabulary

- **retained within scope:** the stated finite-grid identity or corrected argument has been checked against its assumptions and implementation. This is not a novelty assessment.
- **conditional:** the claim may be used only after the listed conditions are checked for the actual method and task.
- **review pending:** an identified proof step is missing or insufficiently justified. No counterexample is asserted.
- **not authorized:** current evidence does not establish the claim.

## Claim map

| Claim | Current status | Scope and required action |
|---|---|---|
| Conditional terminal Gaussian representation | Retained within scope | Finite BLP grid, adapted left-endpoint volatility, `|rho|<1`, positive integrated variance, exactly the independent price driver integrated out. |
| Frozen ordinary-IS unbiasedness | Retained within scope | Exact normalized `q`, full mixture denominator, integrable payoff, training/selection completed before independent final draws. Targets `mu_N`, not the continuous probability. |
| Defensive likelihood and moment bound | Retained within scope | For bounded `0<=g<=1`, `q>=delta p` gives `0<=g p/q<=1/delta` and `E_q[(g p/q)^2]<=mu_N/delta`. This need not be a useful relative rare-event bound. |
| T16-12A balance-mixture identity | Retained within scope | Positive normalized family weights; density is the sum over every sampled component. |
| T16-12B risk identity and final-sample exactness | Retained within scope | Candidate bank frozen before IID validation; selection finished before fresh final sample. This remains true even if the risk certificate is vacuous. |
| T16-12B simultaneous two-sided empirical-Bernstein/oracle bound | Conditional, correction implemented | The one-sided Maurer–Pontil form needs two deviation directions for all J fixed candidates. The implemented sufficient factor is `log(4J/gamma)`, with bounded observations and IID validation. Do not apply to dependent SMC particles or QMC points as IID. |
| T16-13 structural routing exactness | Conditional | Router depends only on predeclared task parameters, each route has exact likelihood, and claimed defensive bound requires positive natural mass. It implies no performance dominance. |
| T16-4 continuous Volterra probability exponent | Conditional scope review | Preserve distinction between noise scaling and Hurst index; confirm positive continuous volatility and the precise existing LDP hypotheses for the chosen payoff. |
| T16-5 trace-class Gaussian equivalence/density | Retained as a construction under its stated equivalence hypotheses | The Gaussian measure construction is separate from a rare-event second-moment efficiency claim. |
| T16-5 continuous rare-event Laplace/second-moment step | Review pending | Need a joint LDP or a valid extended-contraction/exponential-approximation argument for the epsilon-dependent conditional payoff and stochastic integral. Cameron–Martin path continuity alone is insufficient. |
| T16-9 joint mesh/noise efficiency | Review pending | Need the norm upgrade, Gaussian entropy or equivalent control, localization tail, stochastic quadratic-variation tail, left-grid integral approximation, and uniformity over the asserted `N(epsilon)` schedules. |
| T16-11 joint-grid work result | Review pending where dependent on T16-9 | State separately any fixed-grid or purely algebraic part that survives; do not quote the joint claim as established. |
| T16-10 local Newton correction | Conditional | Frozen prediction and exact fallback are separate from local Newton convergence. Local Newton assumptions do not prove global L-BFGS performance. |
| Finite-grid numerical dominance of V16 | Not authorized as a general statement | Historical work-proxy ratios concern seven development cells, one fitted proposal per variant, old CE/conditioning asymmetry, and no matched total-wall confirmation. |
| Continuous-target relative RMSE/complexity | Not authorized | Needs a quantified mesh bias and sampling/inference cost analysis. |

## Specific T16-12B correction

For validation law `G>=delta_G P`, candidates `Q_j>=delta_j P`, and bounded `0<=g<=1`, define

```text
Y_ij = g(X_i)^2 (dP/dQ_j)(X_i) (dP/dG)(X_i),  X_i iid~G,
0 <= Y_ij <= B_j = 1/(delta_j delta_G).
```

The one-sided empirical-Bernstein statement is invoked once for each sign and candidate. With a total error budget `gamma`, a union bound permits the conservative choice

```text
r_j = sqrt(2 S_j^2 log(4J/gamma)/n)
      + 7 B_j log(4J/gamma)/(3(n-1)).
```

This yields the simultaneous two-sided event used by the selected empirical-risk minimizer's oracle-excess inequality. The proof also requires `n>=2`, a fixed bank, correct empirical sample variance, and IID validation draws. If `r_j` exceeds meaningful risk differences, record the certificate as **vacuous**. The R0 code and [updated theorem note](G11_V16_BALANCE_SELECTION_AND_ROUTING_THEOREMS.md) now use the same factor. The V16 final routing experiment did not deploy this selector, so this correction does not itself change its historical estimates.

Source for the empirical-Bernstein form: [Maurer and Pontil, Theorem 4 and Corollary 5](https://arxiv.org/pdf/0907.3740).

## Conditions before any new headline claim

1. A paper theorem must be stated with its parameter range, payoff, discretization, sample role, and measure family.
2. A theorem implemented in code must have an oracle/property test for its testable algebraic part. Tests do not replace the proof.
3. A performance statement requires matched conditioning, an independently fitted strong comparator, fresh tasks, fit-level repetitions, reference uncertainty, and actual total time.
4. A continuous-time statement additionally needs a quantified numerical bias or a theorem proving the asserted limit.
5. If a dependency remains review pending, its dependent claim remains review pending. Changing a YAML status is not a proof.
