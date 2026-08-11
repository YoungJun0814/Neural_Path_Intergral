# V14 Exact Conditional Local-Volterra Transport Report

Date: 2026-08-11
Status: development and independent-seed qualification passed

## Model

V14 conditions on the complete local Gaussian Volterra innovations and integrates the
entire independent price driver analytically for terminal events. An adaptive SMC
run at training power `beta=0.1` estimates a stable local-path shift. The final frozen
proposal is the exact defensive mixture

`q(z) = 0.2 p(z) + 0.8 p(z-mu)`.

For nominal probabilities `1e-3`, `1e-4`, and `1e-5`, respectively 1, 2, and 4
independent SMC estimates are averaged before freezing `mu`. The final likelihood is
an exact balance-mixture likelihood, bounded by five, and final samples are fresh IID
ordinary importance-sampling units.

## Iterative failures and corrections

1. V13 beta-one flow over-concentrated in 383 dimensions and missed the transition
   shell.
2. V14 tempered flow mixtures restored accuracy but had excessive density cost and
   unstable high-dimensional scale estimation.
3. Tempered identity-covariance shifts restored accuracy cheaply.
4. Integrating the complete price driver reduced the transported law from `3N-1` to
   the relevant `2N` local Volterra coordinates.
5. Independent SMC mean averaging stabilized the `1e-5` proposal without increasing
   final density cost.
6. Inaccurate rare-event comparator runs are retained but cannot be selected as the
   best primary method; each record must retain at least one reference-accurate strong
   comparator.

## Frozen development V4

| Metric | Result |
|---|---:|
| Records | 12 |
| Candidate accuracy | pass |
| Exactness / likelihood / pairing | pass |
| Cells favoring V14 | 3 / 3 |
| Best-primary / V14 geometric work ratio | 2.3574 |
| One-sided lower ratio | 1.6020 |
| Stage | pass |

## Independent qualification V1

| Metric | Result |
|---|---:|
| Records | 12 |
| Candidate accuracy | pass |
| Exactness / likelihood / pairing | pass |
| Cells favoring V14 | 3 / 3 |
| Best-primary / V14 geometric work ratio | 2.4159 |
| One-sided lower ratio | 1.8096 |
| Stage | pass |

The qualification used a disjoint namespace and was hash-bound to the development
configuration, result, and passing audit.

## Verification receipt

- V14 scoped Ruff: pass
- V14 scoped mypy: pass (8 source files)
- V14 focused tests: 20 passed
- repository-wide regression tests: 956 passed
- development audit: pass, zero failures
- qualification audit: pass, zero failures
- development result SHA-256:
  `ca33d39b474d8c10113a2a54bc03e02b2f0a19051fabba2fa57345c976508895`
- qualification result SHA-256:
  `bdb29c5d249bdd68d4d836965a5ed5eae46c8f6412d6afd236278cf09ea8c08d`

## Claim boundary

- The result supports an empirical, training-inclusive finite-grid terminal-event
  efficiency claim over the three frozen cells and 100-query amortization contract.
- It does not establish a uniform theorem over all Hurst parameters, thresholds,
  meshes, or barrier events.
- The conservative distribution-free bounded-range tail certificate remains resource
  censored. No distribution-free finite-sample tail claim is authorized.
- External hardware reproduction, current novelty review, and manuscript-level proof
  review remain necessary before a top-journal submission claim.

## Reproduction

```bash
python -m experiments.g11_v14_local_volterra_development \
  --config configs/g11_v14/local_volterra_development_v4.yaml \
  --output results/g11_v14_local_volterra_development_v4_2026-08-11.json

python -m experiments.g11_v14_local_volterra_qualification \
  --config configs/g11_v14/local_volterra_qualification_v1.yaml \
  --output results/g11_v14_local_volterra_qualification_v1_2026-08-11.json
```
