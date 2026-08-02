# G11 V8 D1 Stage B Full-Matrix Decision

Date: 2026-08-02

Decision: **D1 BROAD PERFORMANCE ROUTE STOPPED; STAGE C AND P8 NOT AUTHORIZED**

## Executive conclusion

The complete 24-cell, two-cluster Stage B development run is technically valid and
its independent audit passes. The predeclared scientific continuation gate does
not pass. This is a falsification result, not an execution failure.

The current V8 broad-performance paper must stop at D1 under the frozen plan. It
would be scientifically invalid to lower the mechanism threshold, delete difficult
cells, or rerun until the marginal external-accuracy record passes.

## Evidence integrity

- 24 Hurst/task/rarity cells were retained;
- 48 paired raw/DCS records were completed;
- 144 external-comparator records were completed;
- all 672 Stage B seeds are unique, contiguous in the frozen namespace, and
  disjoint from earlier stages;
- the result is strict standard JSON;
- exact-density and lifecycle checks pass;
- the new 24-cell DCS bank is hash-bound to every paired record;
- cell-specific bank training cost is included at `K = 1, 10, 100, 1000`;
- the independent auditor reconstructs every count, seed, aggregate, decision,
  bank binding, and amortized-work value with no failures.

## Gate results

| Gate | Boundary | Result | Decision |
|---|---:|---:|---|
| maximum exactness residual | `<= 1e-10` | `4.55e-13` | pass |
| DCS absolute accuracy | max combined z `<= 4` | `2.049` | pass |
| proposal-bank cost closure | required | closed | pass |
| external finite weights | all | all finite | pass |
| raw/DCS mechanism ratio | geometric ratio `>= 2.0` | `1.9086` | **fail** |
| primary external accuracy | max combined z `<= 4` | `4.4115` | **fail** |
| primary resource censoring | zero before P8 | 24 | **fail** |

The external accuracy failure is one defensive-CEM record at
`h0.20-discrete_lower_barrier-p1e-05`. Pure CEM, retained as a secondary method,
also fails that same cell more severely. Smoothing RQMC has maximum combined z
`2.819`, but 24 of its records are resource-censored under the frozen laptop
budget.

## Mechanism interpretation

The mathematical Rao--Blackwell identity remains exact:

`Var(raw) - Var(DCS) = E[Var(raw | residual state)] >= 0`.

Stage B does not contradict this identity. Small empirical clusters can reverse
sample-variance order. The scientific failure is instead that the frozen,
material-effect boundary of 2.0 was not achieved across the complete matrix.

The group geometric variance ratios were:

| Task/H | Ratio |
|---|---:|
| terminal, H=0.05 | 1.888 |
| terminal, H=0.12 | 2.762 |
| terminal, H=0.20 | 1.903 |
| barrier, H=0.05 | 1.642 |
| barrier, H=0.12 | 1.456 |
| barrier, H=0.20 | 2.037 |

Thus the effect is heterogeneous and weakest for several barrier regimes. A broad
uniform superiority statement is not supported.

## Cost interpretation

The replayable bank used 407,552 training paths, 199 completed CEM updates, and
`3.195e8` algorithmic work units. It required 10.63 seconds wall time on the
recorded multithreaded laptop environment; cumulative process CPU time was 168.84
seconds.

At paired-estimation time, DCS uses about 7.7% more algorithmic work than raw due to
the analytic conditional calculation. Including shared bank training, the median
raw/DCS work ratio ranges from about 0.962 at `K=1` to 0.929 at `K=1000`. The
observed variance reduction can still give favorable cost-normalized efficiency,
but the frozen mechanism gate was a variance-ratio gate and did not pass. No
headline total-work claim is authorized from two development clusters.

## Theory and implementation conclusions

What is established:

1. canonical log-spot event evaluation is consistent;
2. exact finite-dimensional likelihoods and ordinary means are implemented;
3. DCS is an exact proposal-conditional Rao--Blackwellization;
4. the new rank-one defensive bank is reproducible and fully costed;
5. DCS estimates agree with the independent large reference in every Stage B
   record under the frozen combined-z boundary.

What is not established:

1. a uniform material variance factor of two;
2. broad superiority over strong external comparators;
3. resource-feasible smoothing RQMC at the frozen precision target;
4. P8 qualification readiness;
5. a top-journal new theorem beyond the classical identity; or
6. submission readiness.

## Mandatory stop action

The following phases are not opened:

- Stage C robustness/mesh confirmation;
- Q1/P8 32-cluster qualification;
- P9 freeze;
- P10 confirmation;
- P11 stress submission evidence.

The independent T1 novelty/theory investigation may continue because it does not
reuse Stage B as confirmatory evidence. A future research program may formulate a
new, narrower hypothesis (for example, terminal-only or regime-conditional
benefit), but it must be versioned as a new estimand and start with a fresh
falsification design. It cannot be presented as completion of the stopped broad
V8 claim.
