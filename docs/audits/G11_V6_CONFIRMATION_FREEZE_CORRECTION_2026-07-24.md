# G11 V6 confirmation-freeze correction audit

Date: 2026-07-24  
Scope: outcome-blind freeze transport and confirmatory authorization  
Decision: corrected before any confirmation outcome was generated

## 1. Qualification evidence available before this correction

The frozen 18-cell, 24-cluster qualification artifacts at source commit
`66e0d4a06f5a3b296d8571e85dd82a4551fce5c6` passed their operational,
accuracy, resource, and independent-audit gates.  The frozen power analysis
predeclared 64 confirmation clusters and reported
`freeze_power_ready=true`.

No confirmation baseline or policy outcome existed when the defects below
were found and corrected.  The primary comparator, 20% requested relative
sampling RMSE, 95% confidence level, 1.20 practical work ratio, accuracy
co-gates, cells, and planned cluster count were not changed.

## 2. Defect A: stale baseline schema support

The qualification baseline had advanced to
`npi.g11.v6-baseline-qualification.config.v6`, which carries the frozen rarity
certificate and aggregate-accuracy decision.  The protocol-freeze loader
still accepted only V1/V2 templates.  Falling back to the old V2 development
template would have silently transported a different protocol.

Correction:

- permit all baseline schemas already supported by the execution runner,
  including V6;
- add a V6 primary-only development template that preserves the qualified
  pure-CEM sampling, training, rarity-certificate, and aggregate-accuracy
  rules; and
- test that the frozen baseline remains V6.

## 3. Defect B: impossible power/source equality

The confirmatory analyzer required the power artifact's
`baseline_artifact_sha256` and `policy_artifact_sha256` to equal the hashes of
the new confirmation outcomes.  Those fields necessarily identify the
qualification artifacts used to estimate power, so the equality can never
hold for an untouched confirmation.

Correction:

- authenticate the exact frozen power artifact through
  `expected_sha256.power`;
- require `freeze_power_ready=true`;
- require well-formed qualification-source hashes in the power artifact;
- require the confirmation to contain exactly the predeclared
  `planned_clusters`; and
- retain the independent powered-cluster lower bound.

This does not weaken the design.  It replaces an impossible cross-phase
identity with the correct chain:

`qualification outcomes -> frozen power artifact -> exact planned sample size
-> untouched confirmation outcomes`.

## 4. V4 policy transport

The new unfrozen policy template is a phase-only transport of the qualified V4
policy:

- identical task-conditioned proposal bank and source hash;
- identical router and replicated selector rules;
- identical replicated direct-planning rule;
- identical rarity certificate;
- identical final estimator and work cap; and
- identical aggregate-accuracy role.

The freeze tool changes the phase, protocol ID, cluster count, and proposal
training amortization denominator.  The total training sample/work/time
charges remain fixed.

## 5. Validation

- both new templates pass their strict runtime config loaders;
- targeted freeze, confirmatory, and independent-audit tests pass;
- CI-scope Ruff passes;
- mypy passes all 81 checked source files; and
- the full suite passes: `516 passed`.

The confirmation protocol may be frozen only from a clean commit containing
these corrections.  Any later change to cells, thresholds, gates, cluster
count, estimator, proposal bank, or allocation rule requires a new version and
cannot be represented as the same untouched confirmation.
