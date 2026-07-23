# G11 V6 Baseline V4 Overconservative Defensive Design

Date: 2026-07-23

Decision: stop V4 after confirming the sentinel allocation; preserve all records;
replace it with the separately versioned V5 strong-baseline protocol

## 1. What V4 established

V4 corrected the crude-MC underallocation found in V3. Its crude rarity-band design
passed every executed final gate. V4 also made the defensive-CEM allocation
resource-feasible by replacing the unusable Hoeffding bound with the valid
second-moment envelope

`Var(1_A dP/dQ) <= B p_upper`.

The run from commit `53637bcaa6db47775acdf911a20299d44fd08ac9` was stopped
after 149 complete records. It had no accuracy miss, censoring, or CEM-convergence
failure.

The preserved external run is:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_formal\commit_53637bc\qualification\baseline_v4`

## 2. Why V4 was still not an acceptable strong baseline

For `h005-terminal-p1e-04 / cluster 0 / defensive_cem`:

- pilot variance: `2.1980980032040892e-7`;
- structural design variance: `1.0e-3`;
- final allocation: `2,500,000`;
- empirical sampling-variance/target ratio: `0.0003066817`; and
- total operation work: `2,262,020,096`.

Using the same fivefold pilot-variance safety factor already assigned to the primary
pure-CEM baseline would request fewer than the predeclared minimum 4,096 samples.
At 4,096 samples, the observed final variance scales to approximately 18.7% of the
target. Thus V4 oversampled this defensive baseline by roughly 610 times relative to
the minimum execution count and by more than three thousand times relative to the
empirical target.

The structural envelope is a legitimate upper bound, but using a worst-case bound as
the actual allocation unfairly weakens a method whose purpose is to exploit a
well-trained proposal. It would also spend several laptop-hours generating samples
that add no scientific discrimination.

## 3. V5 strong-baseline rule

V5 keeps the V4 crude rule because crude pilots can contain zero rare events:

`V_design,crude = max(V_pilot, p_upper(1-p_upper))`.

For both pure and defensive CEM, V5 uses the same predeclared plug-in rule:

`V_design,CEM = max(5 V_pilot, p_nominal^2)`.

The defensive structural value `B p_upper` remains serialized as a diagnostic bound,
but it no longer replaces the task-tuned variance estimate. This makes the comparison
algorithmically fair:

- both CEM methods train independently on the task;
- both use 4,096 independent allocation-pilot paths;
- both use a fivefold variance safety factor;
- both have the same minimum final count and work cap; and
- both must independently pass the final empirical-RMSE gate.

The defensive estimator remains bounded and exact-likelihood. V5 merely distinguishes
an estimator bound from an efficient allocation design.

## 4. Statistical interpretation

The V5 CEM allocation is a pre-final engineering design, not a simultaneous
finite-sample guarantee. This is intentional and is disclosed. Scientific acceptance
requires every final empirical target and the aggregate reference-based accuracy
co-gates to pass. A miss is retained as a failed run; final samples may not trigger an
allocation extension.

The V4 observation informed a development-stage baseline-design correction and is
not admissible performance evidence. V5 uses a new protocol ID and disjoint seed
namespace. No V2, V3, or V4 record is reused.

## 5. Claim-control consequence

The primary Route-A comparison remains V6 policy versus pure CEM, whose algorithm was
unchanged by V2--V5. Defensive CEM remains a mandatory robustness baseline. V5 makes
that secondary baseline stronger, not weaker, and therefore makes any later
computational claim harder to obtain but more defensible to reviewers.
