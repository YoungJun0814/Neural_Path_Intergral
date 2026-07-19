# G11 Baseline Scope and Reproduction Contract

Date: 2026-07-19

This note prevents a strong result against a weak or mismatched baseline from being
reported as evidence for DCS-MGI-MLMC.

| ID | Baseline | Implementation | Target contract | Status before M7 |
|---|---|---|---|---|
| B1 | Crude target Monte Carlo | natural zero-control rBergomi sample and hard event | same finest finite grid | implemented in existing simulator; reproduction pending |
| B2 | Raw single-shift IS | one deterministic nonzero Gaussian shift, ordinary likelihood | same finest finite grid | existing likelihood core; dedicated reproduction pending |
| B3 | Raw natural/control defensive mixture | `RBergomiMLMCSampler(method="raw_defensive")` | same hierarchy and mixture | implemented and unit tested |
| B4 | Natural conditional smoothing | zero-control proposal plus fixed positive event direction | same finite grid | implemented and pathwise tested |
| B5 | G9 MGVS | frozen G9 experiment and artifacts | declared G9 grid only | preserved; cross-target comparison prohibited |
| B6 | G10 rank-one DCS-MGI | frozen G10 implementation and artifact | declared G10 grid only | reproduced pathwise by generic G11 adapter |
| B7 | Raw defensive MLMC | generic MLMC engine plus B3 sampler | same finest grid and RMSE target | implemented |
| B8 | DCS-MGI-MLMC | generic MLMC engine plus marginalized sampler | same finest grid and RMSE target | implemented |
| B9 | Published numerical preintegration/smoothing specialization | root-conditioned one-dimensional Gaussian integration | only cases satisfying its monotonicity assumptions | independent SciPy specialization implemented; pathwise reproduction passes |

## Non-negotiable comparison rules

1. B7 and B8 share proposal schedules, mixture weights, hierarchy, pilot size, task,
   and seed namespace. Their optimal allocations may differ because that is the
   method being compared.
2. All reported work includes discarded pilot samples and declared proposal
   calibration. Wall time and operation-scaled work are both retained.
3. G9/G10 frozen numbers may be used as historical evidence, not inserted into a
   new matched-target table unless their target and protocol are rerun.
4. In the declared monotone affine task, B9 reduces exactly to a Gaussian CDF or a
   signed CDF difference. The independent SciPy implementation is therefore a
   reproduction oracle, but not a distinct estimator. This equivalence must be
   stated instead of manufacturing a redundant performance comparison.
5. No baseline may receive task thresholds or oracle controls unavailable to the
   proposed method without an explicit `oracle` label.
