# G11 V6 Policy V4 allocation and qualification-gate audit

Date: 2026-07-24  
Status: implementation complete; formal routed/secondary runs not yet executed  
Baseline source commit: `de7a00a5f899698d0e8564fe5a68610ef36957b3`

## 1. Why V4 was necessary

The frozen V1 secondary and V3 routed-policy configurations contained two
different problems.

1. A bounded Hoeffding variance upper endpoint was multiplied by the allocation
   safety factor and used as if it were an economical point design. The bound is
   mathematically valid, but for probabilities near \(10^{-4}\) it can require
   millions or billions of paths and therefore confounds estimator efficiency
   with worst-case certification.
2. `all_empirical_targets_attained` was included in the qualification decision.
   With \(m\) independent records, an all-record logical AND has probability
   \((1-q)^m\) of passing when each record has miss probability \(q\). Its failure
   probability therefore increases mechanically with the number of records,
   even if the estimator and its per-cell distribution are unchanged.

Neither defect changes the unbiasedness of a completed final estimate. They do
invalidate the intended resource comparison and the top-level qualification
interpretation.

## 2. Frozen V4/V2 allocation rule

For every direct DCS, fixed DCS, and fixed raw-defensive record, the planning
sample is split into predeclared independent replicates. If
\(\widehat V_r\) is the unbiased within-replicate variance, the frozen point
statistic is

\[
\widehat V_{\mathrm{plan}}
  = R^{-1}\sum_{r=1}^{R}\widehat V_r.
\]

The design variance is

\[
V_{\mathrm{design}}
  = \max\{s\widehat V_{\mathrm{plan}}, V_0\},
\]

where \(s=6\). The zero-pilot fallback is \(p_{\mathrm{nom}}^2\) for importance
estimators and \(p_{\mathrm{nom}}(1-p_{\mathrm{nom}})\) for crude Monte Carlo.

Qualification settings are:

- routed direct planning: 8 replicates × 256 paths;
- secondary fixed-method planning: 8 replicates × 512 paths;
- final minimum: 8,192 fresh paths;
- all planning, routing, selector, and final seed roles are disjoint.

This is an engineering plug-in rule. It is **not** claimed to be a
finite-sample upper confidence bound for the unknown variance. Its adequacy is
tested later by the frozen method-by-cell accuracy co-gates.

## 3. Exactness and boundedness review

### 3.1 Exact fixed-grid estimand

All final estimators target the probability on the frozen 128-step rBergomi
grid. Proposal labels and Brownian innovations use exact serialized mixture
likelihoods. No self-normalization, clipping, weight truncation, or reuse of
final samples in planning is permitted.

Conditional on every planning sigma-field \(\mathcal F_{\mathrm{plan}}\), the
fresh final observations \(Y_i\) satisfy

\[
\mathbb E_Q[Y_i\mid\mathcal F_{\mathrm{plan}}]=p_h.
\]

Consequently, a planning-dependent but final-sample-independent integer
allocation preserves unbiasedness:

\[
\mathbb E_Q[\widehat p_h]=p_h.
\]

### 3.2 Where the structural bound is valid

The proposal bank contains the natural component with weight \(w_0=0.15\).
For a single-grid raw or marginalized contribution,

\[
0\le Y\le B,\qquad B=1/w_0,
\]

and therefore

\[
\operatorname{Var}_Q(Y)
 \le \mathbb E_Q[Y^2]
 \le B\mathbb E_Q[Y]
 = Bp_h.
\]

The reference artifact and the frozen rarity-band contract certify
\(p_h\le 2p_{\mathrm{nom}}\) at four reference standard errors. V4 serializes
the resulting \(2Bp_{\mathrm{nom}}\) quantity only as a structural diagnostic.
`structural_bound_used_for_allocation` must be false.

This probability bound is **not** applied to multilevel fine-minus-coarse
corrections. Such corrections are signed, so the step
\(Y^2\le BY\) would be invalid. Hybrid allocation uses only independently
replicated correction variances and the valid absolute correction bounds.

## 4. Qualification decision

`all_empirical_targets_attained` remains serialized as a diagnostic. It is not
an operational qualification gate in V4/V2.

Operational gates cover:

- complete prespecified matrix;
- resolved routes;
- complete, uncensored execution;
- arithmetic design-target attainment;
- internal and offline audit success;
- selector-cost caps;
- absence of reference fields from the router decision;
- replayable planning certificates;
- strict final-seed separation.

Accuracy is deferred to frozen method-by-cell co-gates:

1. an exact one-sided binomial lower bound for the target-attainment rate must
   exceed 0.60;
2. a 2,000-replicate bootstrap upper bound for RMSE must not exceed
   \(\sqrt{(1.25\epsilon)^2+\mathrm{SE}_{\mathrm{ref}}^2}\).

The routed primary method is evaluated by
`g11-v6-power-analysis-qualification-v1`. The two fixed secondary methods are
evaluated by `g11-v6-secondary-accuracy-qualification-v1`.

No safety factor or threshold may be changed after observing qualification
outcomes. A failed aggregate co-gate is a negative result and blocks
confirmation freezing.

## 5. Independent replay added

The offline result auditor now independently recomputes:

- mean/median replicate planning variance;
- safety-factor and fallback design variance;
- pooled pilot count and unbiased variance convention;
- reference rarity-band certificate;
- structural diagnostic and its non-use in allocation;
- hybrid candidate integer point work;
- practical-equivalence selection;
- selected hybrid term identities and design variances;
- V4/V2 operational/diagnostic summary split;
- the serialized top-level qualification decision.

An initial smoke audit detected an \(n\) versus \(n-1\) variance serialization
mismatch. Direct design records and certificates were then standardized to the
unbiased \(1/(n-1)\) convention. The corrected direct smoke records pass every
record-level audit. Smoke artifacts remain formally unqualified by design.

## 6. Verification completed before formal execution

- Ruff over `src`, `experiments`, and `tests`: pass.
- Mypy over the 81 CI source files: pass.
- Pytest: 516 passed.
- V4 selector-off direct DCS smoke: all operational gates pass.
- V2 fixed DCS/raw smoke: all operational gates pass; an individual empirical
  miss remains diagnostic only.
- Independent direct and summary replay: all record audits pass.
- Synthetic hybrid certificate replay and tamper rejection: pass.
- Materialized V4/V2 qualification YAML files are byte-identical to the
  deterministic suite generator outputs.

## 7. Remaining falsification sequence

1. Commit, push, and require Linux CI on Python 3.10 and 3.11.
2. Run full routed V4 qualification (18 cells × 24 clusters).
3. Run selector-off V4 qualification on the identical matrix.
4. Run fixed DCS/raw V2 qualification (18 × 24 × 2).
5. Run offline audits and resource supplements for every artifact.
6. Run primary paired power/resource/accuracy analysis.
7. Run secondary method-by-cell accuracy analysis.
8. Run the frozen theory-diagnostic matrix.
9. Freeze the untouched 64-cluster confirmation protocol only if every
   prespecified gate passes.

The present implementation is suitable for formal falsification. It is not yet
evidence that the routed model beats pure CEM or that the work is journal-ready.
Those claims remain blocked on the full paired results, confirmation, and the
O1–O6 theoretical proof audit.
