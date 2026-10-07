# V13 Structured ECRPT Development Report

Date: 2026-08-11
Status: implemented and independently audited; qualification blocked

## Outcome

V13 implemented the planned exact conditional residual path-space transport:

1. a positive price direction is selected on training-only data;
2. an ESS-adaptive annealed SMC sampler targets the residual density proportional
   to `g(r) p_R(r)` using pCN mutation;
3. the dependent SMC population is used only to fit an exact low-rank triangular
   coupling flow;
4. the frozen proposal is mixed with the natural residual law;
5. fresh IID residuals use the exact ordinary likelihood ratio `p_R/q_R`; and
6. the scalar Gaussian coordinate is integrated analytically.

The 3-cell by 4-cluster frozen development matrix completed. Its independent audit
passed every structural and arithmetic check. The development gate did **not** pass,
so no qualification run was executed.

## Verified correctness

The following checks passed in all 12 records:

- exact residual orthogonality and full-path reconstruction;
- exact hard-event/scalar-threshold agreement;
- frozen ordinary likelihood with no self-normalization;
- defensive likelihood bound;
- likelihood normalization diagnostic;
- paired raw/conditional mean identity;
- monotone adaptive SMC schedule ending exactly at beta one;
- SMC particles marked training-only, never IID inferential units;
- pathwise nonnegative Rao--Blackwell variance-gap integrand; and
- Bonferroni-valid Hoeffding/empirical-Bernstein tail certificates.

## Frozen performance result

| Metric | Result | Gate |
|---|---:|---:|
| Records | 12 | 12 |
| Exactness | pass | pass |
| Paired identity | pass | pass |
| Likelihood normalization | pass | pass |
| Rao--Blackwell mechanism | pass | pass |
| Candidate accuracy | fail | pass required |
| Primary comparator accuracy | fail | pass required |
| Tail-safe uncensored | fail | pass required |
| Best-primary / V13 geometric work ratio | 0.01789 | > 1.25 |
| One-sided lower work ratio | 0.00476 | > 1.0 |
| Cells favoring V13 | 0 / 3 | at least 2 |

The ratio is comparator work divided by V13 work; values below one mean V13 is more
expensive. This result therefore falsifies the current training-inclusive efficiency
claim by a wide margin.

## Failure diagnosis

The failure is not evidence of algebraic bias. The defensive mixture preserves
absolute continuity and exact ordinary importance sampling. It is a finite-budget
support-coverage failure:

- at nominal `1e-4`, the mean V13 estimate across clusters was approximately
  `3.91e-7`, versus reference `1.1445e-4`;
- at nominal `1e-5`, it was approximately `2.60e-10`, versus reference `8.85e-6`;
- the SMC schedules completed in 8, 10, and 13 stages and pCN acceptance remained
  roughly 0.68--0.78, so simple chain rejection is not the primary symptom;
- the learned flow did not allocate enough mass to all important residual modes;
- the 20% defensive component guarantees correctness asymptotically but yields too
  few natural draws in the extreme cells at 4,096 final samples; and
- empirical plug-in errors looked deceptively small after important regions were
  missed, while the distribution-free tail certificates correctly stayed censored.

The strong natural conditional comparator also failed some finite-budget accuracy
checks, confirming that 4,096 units are insufficient for universal accuracy at the
deepest cell. This does not rescue V13: its own accuracy and efficiency gates failed.

## Scientifically valid next redesign

Do not increase epochs or weaken gates. A V14 attempt is justified only if it changes
support coverage and amortized cost materially:

1. preserve several SMC modes with a deterministic mixture of independently trained
   flows rather than compressing the whole target into one flow;
2. train across a task family so the one-time cost is amortized across thresholds,
   Hurst values, or repeated risk queries;
3. add a held-out defensive-component stress test that estimates stratum-specific
   second moments before final evaluation;
4. select the defensive weight from a frozen tail-certificate objective on training
   data, never from final outcomes;
5. compare against direct conditional rBergomi before expensive V10R1/RQMC runs and
   stop early if the candidate cannot beat it; and
6. freeze a larger development budget only after the proposal passes a multi-mode
   coverage gate.

No uniform log-density approximation bound was empirically certified. The stability
theorem remains conditional and may not be advertised as an achieved guarantee.

## Verification receipt

- V13 scoped Ruff checks: pass
- V13 scoped mypy checks: pass (9 source files)
- V13 focused tests: 23 passed
- repository-wide regression tests: 947 passed
- result artifact audit: pass, zero failures
- repository-wide Ruff: not clean because of 49 pre-existing warnings in the
  unrelated root utilities `fix_viz.py`, `generate_concept_plots.py`, and
  `update_maxiter.py`; these files were not modified by V13

## Reproduction

```bash
python -m experiments.g11_v13_structured_ecrpt_development \
  --config configs/g11_v13/structured_ecrpt_development_v1.yaml \
  --output results/g11_v13_structured_ecrpt_development_v1_2026-08-11.json

python -m experiments.g11_v13_structured_ecrpt_audit \
  --config configs/g11_v13/structured_ecrpt_development_v1.yaml \
  --result results/g11_v13_structured_ecrpt_development_v1_2026-08-11.json \
  --output results/g11_v13_structured_ecrpt_development_v1_audit_2026-08-11.json
```
