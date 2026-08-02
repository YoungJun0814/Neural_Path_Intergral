# G11 V9 Development V2 Execution Failure

Date: 2026-08-02

Status: **namespace invalid for inference; no result artifact was produced.**

Development V2 retained the minimum of four nonzero pilot inferential units and
raised the common pilot size to 256.  It still stopped atomically when a target-law
conditional-rBergomi pilot had only three nonzero units.  No aggregate or result
file was created.

A diagnostic replay of the already-invalid V2 pilot seeds showed that low support
was confined to `conditional_rbergomi`; the defensive CEM pilots had at least 20
nonzero paths in the diagnostic screen, while an RQMC unit is a positive averaged
conditional value.  The most difficult H=0.20, p-design=1e-5 conditional pilots had
only 1--2 nonzero values among 256.

V3 therefore freezes method-specific pilot sizes with new seeds:

- conditional rBergomi: 4,096 IID units;
- defensive CEM: 256 IID units; and
- smoothing RQMC: 64 independent scrambles of 4,096 points.

The support threshold, variance safety factor, final budgets, scientific gates,
cells, clusters, and target RMSE are unchanged.  V1/V2 seeds and partial in-memory
values are not reused.
