# G11 V7 Linux reproduction V1 decision

Date: 2026-07-24
Decision: **PASS**
Role: disjoint-seed software-environment reproduction of the confirmed V7 DCS mechanism

## 1. Reproduced claim

This study reproduces the Windows V7 confirmation on Linux without changing the
scientific design:

> On the frozen 18-cell, 128-step rBergomi matrix, and under the same deterministic
> defensive proposal, DCS conditional Gaussian integration retains a practically
> significant variance and achieved-RMSE work advantage over raw defensive
> importance sampling while both methods satisfy simultaneous accuracy gates.

The Linux run uses new protocol namespaces and therefore new seeds. It does not
retrain, retune, or replace the proposal after seeing the Windows outcome.

## 2. Freeze and execution identity

The pre-outcome reproduction package:

- binds estimator source commit
  `80a5bc947c1ae6d4405bd4964cf268b88a5688d0`;
- requires a clean, self-contained checkout of that commit;
- permits only the four protocol IDs and the accuracy-bootstrap seed to differ
  from the Windows confirmation configs;
- binds the Windows aggregate audit and every canonical artifact hash;
- records immutable Docker image ID
  `sha256:912ad54b2a9d1e2435a691d9a912ff2d20a2107ad02d0c706d39b82b9313914f`;
  and
- has freeze-receipt SHA-256
  `9d7df83a8f3738e4a867255e931e2bfe200284dee0984404661a4af25a8d803c`.

The reproduction execution-freeze SHA-256 is
`cb9ea716e5b432828b6e5b85e9f1a14cc889174d64e79223d7db8a3b6e7b379f`.
The freeze tool parsed the canonical and reproduction configs and rejected any
scientific-field drift.

## 3. Runtime and scope

The Linux runtime was:

- Linux `6.6.87.2-microsoft-standard-WSL2`;
- Python 3.10.20;
- PyTorch 2.9.1 CPU;
- NumPy 1.26.4;
- SciPy 1.15.3;
- PyYAML 6.0.3;
- psutil 5.9.0;
- `torch.float64`; and
- 16 PyTorch threads.

The source tree reported `dirty_worktree=false`. The study contains:

- 18 finite-grid cells;
- 64 independent clusters;
- 1,152 paired mechanism records;
- 2,304 fixed-estimator records;
- 4,096 common-path samples per paired cell-cluster;
- 10% requested relative sampling RMSE;
- eight 512-sample planning replicates; and
- common final floor 512.

This is a cross-software-environment reproduction on the same physical laptop.
It is not an independent physical-hardware or external-laboratory replication.

## 4. Linux mechanism and efficiency result

All prespecified mechanism, integrity, floor, work, provenance, and accuracy gates
passed.

| Raw divided by DCS | Geometric ratio | One-sided 95% lower ratio | Frozen minimum |
|---|---:|---:|---:|
| Common-path probe variance | 3.3727 | 3.3244 | 2.0 |
| Production execution variance | 3.5172 | 3.4499 | 2.0 |
| Final sampling work | 3.3721 | 3.3021 | 1.5 |
| Non-training work | 1.9012 | 1.8714 | diagnostic |
| Training-inclusive total work | 1.8909 | 1.8613 | cross-environment gate |
| Isolated final wall time | 2.7634 | 2.7042 | diagnostic |

Neither method used the common final floor. The final-work advantage is therefore
not an artifact of both methods being pinned to the same arbitrary minimum.

The paired falsification diagnostics also passed:

- maximum absolute residual-mean z: 3.3901;
- maximum absolute covariance-product z: 3.3246; and
- frozen maximum for both: 4.5.

The independent paired auditor passed 1,152/1,152 records.

## 5. Absolute Linux computational totals

| Quantity | Fixed DCS | Fixed raw |
|---|---:|---:|
| Cell-cluster records | 1,152 | 1,152 |
| Final samples | 4,716,349 | 14,095,224 |
| Final work units | 4,225,848,704 | 12,629,320,704 |
| Allocation-pilot work units | 4,227,858,432 | 4,227,858,432 |
| Shared training charge | 88,080,384 | 88,080,384 |
| Training-inclusive work units | 8,541,787,520 | 16,945,259,520 |
| Isolated final wall seconds | 242.20 | 605.09 |
| Training-inclusive wall seconds | 581.02 | 862.09 |

Ratios of these grand totals are descriptive. The inferential headline ratios in
Section 4 are the frozen cluster-balanced geometric effects.

## 6. Simultaneous accuracy

The formal family contains 72 claims: 18 cells, two methods, and an attainment plus
an RMSE claim per method-cell. The family-wise alpha is 0.05.

- all 72 simultaneous claims passed;
- minimum exact Clopper--Pearson attainment lower bound: 0.8310, above 0.60;
- maximum bootstrap RMSE-upper/tolerance ratio: 0.7530, below 1;
- DCS individual empirical-target misses: 2/1,152; and
- raw individual empirical-target misses: 4/1,152.

The misses are retained. The pre-outcome decision rule is the method-cell
attainment-plus-RMSE co-gate, not an all-record Boolean conjunction.
Clopper--Pearson coverage is finite-sample exact. The fixed-seed percentile-bootstrap
RMSE bound is nominal rather than a finite-sample theorem.

## 7. Cross-environment reproduction audit

The host-side audit compared the frozen Windows package with the Linux package.
All six pairwise intersections among Windows/Linux probe/fixed seed sets were zero.
The seed counts were:

- Windows probe: 2,304;
- Windows fixed: 48,200;
- Linux probe: 2,304; and
- Linux fixed: 48,452.

The standardized Windows--Linux effect differences were:

| Effect | Absolute difference in combined SE units | Frozen maximum | Gate role |
|---|---:|---:|---|
| Common-path probe variance | 1.7061 | 3.0 | required |
| Production execution variance | 0.0397 | 3.0 | required |
| Final sampling work | 0.2030 | 3.0 | required |
| Training-inclusive total work | 0.3355 | 3.0 | required |
| Non-training work | 0.3332 | diagnostic | diagnostic |
| Isolated final wall time | 8.7979 | diagnostic | diagnostic |

Every required effect-stability gate passed. The large wall-time z difference is
reported, not suppressed. It is consistent with wall time being sensitive to OS,
container, scheduler, and serialization overhead, and confirms why wall time was
not made a cross-environment scientific gate.

The hardware-reproduction audit passed with zero failures. Its SHA-256 is
`cfe2b3d2d05077e9fe7984a35fd8c20ecfdbf91894ab20bf59cd155c9ebd599d`.

## 8. Independent Linux package audit

The independent aggregate auditor:

- verified the execution freeze and four config hashes;
- verified clean source identity and exact artifact links;
- checked record counts, completeness, and censoring;
- recomputed the five fixed-estimator effects and floor counts;
- recomputed all 36 method-cell accuracy groups; and
- repeated the frozen 50,000-draw bootstrap convention.

It passed with zero failures. Its SHA-256 is
`54d5928e3704f5c747cd2025ac1aef254b3abfbe36b7998f72550d45b7936fef`.

## 9. Artifact ledger

| Artifact | SHA-256 |
|---|---|
| Reproduction freeze receipt | `9d7df83a8f3738e4a867255e931e2bfe200284dee0984404661a4af25a8d803c` |
| Reproduction execution freeze | `cb9ea716e5b432828b6e5b85e9f1a14cc889174d64e79223d7db8a3b6e7b379f` |
| Linux paired mechanism probe | `fa5d42ec16bd9095d14003577f643156eda6d649ee414b33ad1ea0dafa80f996` |
| Independent Linux paired audit | `80ed4d8bbe011c1c055bd09aa08f0bd620d975ca93479d3da054a706d3458835` |
| Linux fixed estimators | `d08cf915101c316981192511f1343c342a1c6dc6c9e89b665f60be1d80979ba6` |
| Independent Linux fixed audit | `23785cd9190c36e15b9f2ba3fa8796c14e7b964a32149102b5de07f594288799` |
| Linux resource supplement | `d076938a2d7c168dcf37cf579510a6ac84fb4934b8e2116a02cd16a383bbfa3a` |
| Linux joint mechanism analysis | `46ac04ca1372034bfa210186565764f460b20be140f268359093b2b634eafaf8` |
| Linux simultaneous accuracy | `fb77125b0dda3a094f2f7e26fcb188927e661e535b6122b14aae101199f69064` |
| Independent Linux aggregate audit | `54d5928e3704f5c747cd2025ac1aef254b3abfbe36b7998f72550d45b7936fef` |
| Windows--Linux reproduction audit | `cfe2b3d2d05077e9fe7984a35fd8c20ecfdbf91894ab20bf59cd155c9ebd599d` |

## 10. Technical and theoretical error audit

### Supported exactly in the implemented scope

- the raw and DCS estimators use the same frozen proposal and exact balance-mixture
  likelihood;
- the raw fast path does not execute hidden DCS arithmetic;
- DCS is the finite-dimensional conditional expectation of the raw contribution
  along the integrated Gaussian coordinate;
- Rao--Blackwell gives variance non-increase under square integrability;
- the positive natural mixture component provides the declared pathwise likelihood
  bound;
- the ordinary, non-self-normalized sample mean is used;
- pilot, probe, and final seeds are separated;
- the implemented terminal and discrete-barrier tasks have exact scalar-threshold
  representations on the declared finite grid; and
- Windows and Linux execute the same clean estimator source.

### Statistical qualifications

- the 64-cluster t inference is parametric, not distribution-free exact;
- Clopper--Pearson attainment bounds are exact;
- percentile-bootstrap RMSE bounds are approximate;
- orthogonality z checks are falsification diagnostics rather than proofs; and
- cross-environment effect agreement does not remove shared model, reference,
  proposal, or implementation assumptions.

### Not established

- unbiased continuously monitored barrier probabilities;
- a universal rBergomi weak-error or complexity theorem;
- uniform superiority over unseen parameters and tasks;
- superiority over every task-tuned adaptive IS, CEM, flow, or transport method;
- independent physical-hardware or external-team replication;
- a neural-architecture contribution; or
- a quantum-mechanical path-integral result.

No technical or theoretical contradiction was found within the explicitly declared
finite-grid claim.

## 11. Publication decision

The Windows confirmation plus disjoint-seed Linux reproduction now form a credible
PhD-level computational core and materially strengthen a specialized
financial-mathematics or computational-probability submission. The result is no
longer only a development-machine effect.

It remains short of a defensible top-journal package. The next highest-value gates
are:

1. a matched, task-tuned adaptive-IS/CEM comparator and at least one strong
   flow/transport comparator with all training and tuning work charged;
2. a barrier mesh/weak-bias theorem, or a deliberately finite-grid theorem whose
   scope is central rather than hidden in limitations;
3. external mathematical proof and code review;
4. an up-to-date, reproducible literature and novelty audit; and
5. manuscript assembly around one claim:
   **mechanism-identified amortized DCS path-measure estimation**.

The cross-environment result should not be described as independent hardware
replication, universal optimality, a neural model, or a quantum path integral.
