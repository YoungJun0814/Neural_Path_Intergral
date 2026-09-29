# G11 V10R1 full-latent correction: final execution report

Date: 2026-08-11  
Branch: `codex/v10r1-full-latent`  
Conclusion: implementation/correctness repair succeeded; development performance
gate failed; qualification and submission remain unauthorized

## 1. Executive conclusion

V10R1 repaired the decisive law mismatch in exploratory V10.  CEM is now
trained, sampled, simulated, evaluated, serialized, and replayed in the exact
`3N=384` BLP latent coordinates.  No learned coordinate is discarded and the
frozen proposal is never changed after training.  Exact scalar DCS integrates
only one price-block coordinate and retains all other `3N-1` coordinates in the
residual likelihood.

The corrected implementation passed numerical exactness, likelihood,
ordinary-mean, paired-identity, clean-source, seed, proposal replay, work
reconstruction, aggregate recomputation, and mutation-detection checks.

It did **not** pass the frozen development gate.  The three blockers are:

1. external-reference accuracy failure;
2. 26 resource-censored external records;
3. zero Hurst groups passing the complete correctness-plus-efficiency gate.

The prespecified protocol therefore stops before qualification.  It would be a
protocol violation to run qualification seeds now in search of a better result.
This is a valid falsification result, not evidence of top-journal readiness.

## 2. What was corrected

| Item | Exploratory V10 | V10R1 |
|---|---|---|
| CEM training dimension | `3N` | `3N` |
| Evaluated proposal dimension | effectively projected; `N` local coordinates lost | exact `3N` |
| Post-training price mean | sign-modified | unchanged |
| Conditional integration | on altered/projected adapter | one basis direction, proposal unchanged |
| Residual likelihood | did not represent the complete trained law | exact `3N-1` marginal |
| Training randomness | one proposal per cell | four independent proposals per cell |
| Source provenance | dirty tree; named commit lacked executed V10 code/config | clean committed source and exact Git config blob |
| Bank audit | superficial seed check | exact roster/hash/cost plus full deterministic law replay |
| Performance audit | summary-level | seed order, bindings, work, aggregate, decisions independently recomputed |
| Theory claim | over-broad “orthogonal” language | exact Rao--Blackwell variance non-increase only |

## 3. Frozen mathematical result

For a frozen defensive mixture

\[
Q=\alpha N(0,I_{3N})+(1-\alpha)N(\mu,I_{3N}),
\]

V10R1 decomposes `X=Ae+R` using a fixed positive price-block direction `e`.
For the finite-grid terminal event there is an exact threshold `tau(R)`.  The
implemented contributions are

\[
Y_{raw}=1\{A\le\tau(R)\}\,dP/dQ(X),
\]

and

\[
Y_{dcs}=dP_R/dQ_R(R)\,\Phi(\tau(R)).
\]

Thus `Y_dcs=E_Q[Y_raw|R]`, both ordinary means are unbiased conditional on the
frozen training result, and `Var(Y_dcs) <= Var(Y_raw)`.  No uniform strict or
multiplicative improvement theorem is claimed.  Training samples are independent
of evaluation samples, so unconditional unbiasedness follows by iterated
expectation.

## 4. Verification evidence

### 4.1 Static and regression verification

- Ruff: pass.
- Mypy: pass for all 117 checked source files.
- Full regression before execution: 905 tests passed.
- New adversarial coverage: lost-coordinate mutation, proposal hash mutation,
  seed mutation, aggregate mutation, complete law replay, and raw/DCS paired
  identity.

### 4.2 Proposal bank

- Cells: 12.
- Independent proposal replicates per cell: 4.
- Total proposals: 48.
- Dimension of every proposal: 384.
- Unique training seeds: 48.
- Unique mathematical-law hashes: 48.
- Training samples: 335,872.
- Training algorithmic work: 300,941,312 units.
- Clean source: yes.
- Exact config present in recorded commit: yes.
- Full deterministic training replay: pass.
- Bank audit failures: none.

Runtime measurements are excluded from the replay-law hash because wall time and
peak memory are not deterministic.  They remain in the immutable full artifact
hash.  Mean, weights, dimension, family, likelihood contract, and conditional
role form the replayed mathematical-law hash.

### 4.3 Development artifact integrity

- Candidate records: 48.
- External records: 144.
- Evaluation seeds: 672, all unique, contiguous in execution order, and
  disjoint from reference/bank seeds.
- Clean source commit: `c9fd91a7aa0e899b60dabdca69fd1bee04faf9d8`.
- Config SHA-256:
  `670935de77f871ddc570f28aab83154601ef36933c158bf9d5f2e4e2ef84d063`.
- Full matrix elapsed process time: approximately 2,190 seconds on the local
  machine.
- Audit v2: all 18 checks passed.

The first audit file is intentionally retained as a failed diagnostic.  Its
auditor concatenated the stored candidate and external lists rather than
reconstructing the interleaved execution order.  The corrected auditor now
rebuilds `candidate -> external methods` per cell/cluster.  Unit tests show it
accepts the exact artifact and rejects seed or aggregate mutations.

## 5. Correctness results

| Gate/diagnostic | Result | Limit | Status |
|---|---:|---:|---|
| Maximum numerical exactness error | `5.684e-14` | `1e-10` | pass |
| Candidate DCS maximum combined reference z | `1.578` | `4.0` | pass |
| Raw/DCS paired-difference maximum z | `2.850` | `4.0` | pass |
| Likelihood-normalization pass fraction | `1.000` | `>=0.95` | pass |
| Maximum absolute likelihood-normalization z | `1.823` | `6.0` | pass |
| External maximum combined reference z | `9.122` | `4.0` | fail |
| Resource-censored external records | `26/144` | `0` | fail |

Maximum component and mixture density reconstruction errors are `5.684e-14`;
full-likelihood error and both defensive-bound violations are exactly zero at
recorded precision.  Local-latent, price-latent, scalar-coordinate, and path
reconstruction maxima are all below `1.5e-14`, except local reconstruction at
`4.39e-15` and component/mixture arithmetic at `5.68e-14`.

The candidate itself agrees with the independent reference.  The accuracy
failure is confined to external comparator records, primarily conditional
rBergomi at probability `10^-5`, plus one defensive-CEM record.

## 6. Mechanism and efficiency results

### 6.1 Paired mechanism

| Hurst | Geometric raw/DCS contribution-variance ratio | Training-inclusive raw/DCS work ratio at 100 queries | 95% one-sided lower bound |
|---:|---:|---:|---:|
| 0.05 | 4.168 | 2.863 | 1.245 |
| 0.12 | 2.176 | 1.492 | 0.948 |
| 0.20 | 3.339 | 2.265 | 1.566 |

A work ratio greater than one favors DCS.  The corrected DCS mechanism is
promising against its exact same-proposal raw control for `H=0.05` and `0.20`.
For `H=0.12`, the point estimate favors DCS but its cluster-level lower bound is
below one.  Four training clusters provide limited power.

### 6.2 Best-primary comparison

| Hurst | DCS / best-primary work-ratio geometric mean | 95% one-sided lower bound | Required development point/lower bound |
|---:|---:|---:|---:|
| 0.05 | 0.272 | 0.152 | 0.80 / 0.67 |
| 0.12 | 0.576 | 0.182 | 0.80 / 0.67 |
| 0.20 | 0.373 | 0.250 | 0.80 / 0.67 |

Here the ratio is comparator work divided by DCS work, so a value below one
means DCS requires more work than the best comparator selected per cell.  All
three Hurst groups fail this efficiency condition by a wide margin.

Across all 48 candidate records, the geometric work ratios at 100 queries are:

- raw same-proposal / DCS: `2.131`;
- conditional rBergomi / DCS: `1.225`;
- smoothing RQMC / DCS: `27.336`;
- defensive CEM / DCS: `1.252`;
- best primary per record / DCS: `0.388`.

The last value is the relevant competitive result: despite improving its raw
control, V10R1 does not beat the best available comparator per task.  Moreover,
external accuracy and censoring failures make all external efficiency ratios
descriptive only, not claim-bearing.

## 7. Diagnosed baseline-allocation failure

The external lifecycle currently allows as few as four final units.  In several
`10^-5` conditional-rBergomi clusters, a 4,096-unit pilot observed a highly
skewed set of tiny conditional contributions.  The plug-in variance then fell
below the absolute target variance and the planner allocated only four final
units.  The final estimate became zero with reported standard error zero, even
though the independent reference is approximately `9e-6`.  The reference z
gate correctly rejected this false “target attained” status.

This is not evidence that the conditional estimator is biased.  It is evidence
that a plug-in sample variance plus a four-unit floor is not a reliable stopping
rule for extremely skewed rare-event contributions.  It also means the current
external comparison cannot support a publication claim.

Resource censoring by method was:

- conditional rBergomi: 5/48;
- smoothing RQMC: 17/48;
- defensive CEM: 4/48.

## 8. Mandatory next actions

### Priority A — repair the external allocation protocol

1. Replace the four-unit floor with method-specific, prespecified minimum final
   units; RQMC must retain enough independent randomizations for a meaningful
   variance estimate.
2. Use a one-sided upper confidence bound for unit variance or a conservative
   tail-aware empirical-Bernstein/median-of-means planning rule.  A zero final
   variance must never certify target attainment for a positive reference task.
3. Require independent final nonzero/moment diagnostics and reference agreement
   before `target_attained=true`.
4. Implement batched/streaming final evaluation so multi-million-unit allocations
   do not require one giant tensor.
5. Freeze a new configuration and seed namespace.  Do not amend this result.

### Priority B — improve the candidate rather than only adding compute

1. Jointly learn a strictly positive deterministic integration direction and
   the full `3N` mixture law on training-only samples, optimizing the ordinary
   DCS second moment rather than the raw hard-event CEM objective.
2. Preserve exact likelihoods by parameterizing positivity, not by changing a
   learned proposal after training.
3. Test a defensive multi-component full-latent mixture for multimodal rare-event
   regions.  Component count must be chosen inside a training-only budget and
   its total cost charged.
4. Use a held-out mechanism screen.  Advance only if DCS beats both its raw
   control and the per-cell best strong baseline under total work.

### Priority C — obtain adequate compute and statistical power

1. Run the uncensored matrix on cloud hardware with streaming estimators.
2. Increase independent training/evaluation clusters beyond four; four gives
   weak lower bounds and unstable method rankings.
3. If development passes, freeze a completely disjoint qualification namespace
   before examining qualification outcomes.

### Priority D — requirements beyond this estimator study

1. Repeat across multiple time grids to distinguish finite-grid estimator
   correctness from discretization error.
2. Add another rough-volatility model or a genuinely different path-dependent
   application.
3. Derive theory beyond generic Rao--Blackwell non-increase: for example a
   verifiable strictness condition, stability bound, or complexity result for
   the jointly learned full-latent method.
4. Reproduce wall-clock results on independent hardware and compare against
   current peer-reviewed methods under identical estimands.

## 9. Research-level assessment

The corrected code, theorem boundary, provenance, replay, and falsification
protocol are doctoral-quality research infrastructure.  The current empirical
result is not a completed doctoral contribution and is not suitable for a top
journal performance claim: the candidate loses to the best per-cell comparator,
external gates fail, only one model/event family is covered, and no new strict
rate or complexity theorem has been established.

The defensible paper-level contribution at this point is a negative/corrective
study showing why full-latent law preservation, training-randomness replication,
and tail-safe allocation are necessary.  A positive top-journal submission
requires Priorities A–D and an independently passing qualification.

