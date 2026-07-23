# G11 V6 Baseline V3 Empirical-RMSE Failure

Date: 2026-07-23

Decision: stop V3 after its first final-sample accuracy failure; preserve all
checkpoints; replace it with the separately versioned V4 protocol

## 1. Observed failure

The clean V3 qualification was run from commit
`4f1bd93903ff8666a9d50ba5fa54304b95adba92`. It was stopped with 134 complete
records after the first failure of the predeclared
`all_empirical_targets_attained` gate.

The failed record was:

- cell: `h005-terminal-p1e-03`;
- cluster: `15`;
- method: `crude`;
- frozen final allocation: `22,485`;
- design variance: `8.993895523278673e-4`;
- empirical final sampling variance: `4.347414779827172e-8`;
- required sampling-variance target: `4.0e-8`;
- empirical/target ratio: `1.086853694956793`; and
- reference z score: approximately `-0.061`.

The estimate agreed with the independent reference. The failure was specifically an
allocation/RMSE failure, not an estimator-bias failure.

The preserved external run is:

`C:\Users\Jun\Desktop\Thesis\Projects\NPI_formal\commit_4f1bd93\qualification\baseline_v3`

No V3 record is admitted to V4.

## 2. Root cause

Crude V3 still used a per-run 95% Clopper--Pearson pilot interval to upper-bound the
Bernoulli variance. The formal gate, however, requires every record in a
`18 cells x 24 clusters x 3 methods = 1,296` matrix to attain its empirical target.
A 95% pointwise design certificate is not a familywise certificate for that matrix.
Even a valid pointwise interval can underallocate at least one run with material
probability.

This is distinct from the V2 defensive-CEM failure. V3 fixed the defensive
second-moment design correctly, but it exposed the remaining pointwise crude design
weakness.

## 3. V4 correction

For a crude Bernoulli contribution `Y=1_A`,

`Var(Y)=p(1-p)`.

The frozen cell contract is `p <= p_upper = 2 p_nominal`. All configured upper
bounds are below `1/2`, so

`Var(Y) <= p_upper(1-p_upper)`.

V4 therefore sets

`V_design,crude = max(V_pilot, p_upper(1-p_upper))`.

It retains the V3 defensive identity

`Var(1_A dP/dQ) <= B p_upper`.

Before either bound is used, the independent reference must satisfy the
predeclared four-standard-error engineering certificate

`p_reference + 4 SE_reference <= p_upper`.

The reference number does not set the allocation. The globally frozen rarity-band
upper bound does. The final estimator remains an ordinary independent sample mean
with exact likelihood where applicable.

## 4. Why this correction is not outcome cherry-picking

- V4 changes only the crude secondary allocation design.
- The primary pure-CEM comparator, its proposal fitting, fivefold variance safety
  factor, work accounting, and primary endpoint are unchanged.
- The failure was mandatory under the predeclared all-record accuracy co-gate and
  could not be silently dropped.
- V4 uses a formula fixed for every cell and cluster, not a per-cell fitted margin.
- V4 receives a new protocol ID and disjoint seed namespace.
- All V3 checkpoints remain preserved and are not resumed.

## 5. Remaining statistical qualification

The rarity-band reference check is a high-confidence engineering certificate, not an
absolute deterministic statement about an unknown probability. Scientific
qualification still requires:

- every independent final empirical target to pass;
- no resource censoring;
- every CEM fit to converge;
- independent reference agreement;
- independent replay of the rarity-band arithmetic;
- all work and seed ledgers to reconcile; and
- untouched confirmation under a later frozen protocol.

V4 fails if any one of these conditions fails. It may not extend an allocation after
looking at final observations.
