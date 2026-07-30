# G11 V8 R2S — Method-role precision protocol

Date: 2026-07-31

## Statistical roles

The two estimators no longer receive an identical precision requirement merely
because they estimate the same fixed-grid probability.

- `dcs_reference` is the primary reference and retains a 2% relative standard
  error target.
- `raw_crosscheck` is an independent hard-indicator corroboration and receives
  a 5% relative standard error target.
- The raw estimate may not replace the DCS primary reference.
- Both estimates remain ordinary unbiased, non-self-normalized estimators.

For independent estimates \(\hat p_D,\hat p_R\), agreement is evaluated with

\[
Z={\hat p_D-\hat p_R\over
\sqrt{\widehat{\mathrm{SE}}_D^2+\widehat{\mathrm{SE}}_R^2}},
\qquad |Z|\le4.
\]

The 2%/5% design gives a nominal combined 4-standard-error scale of about
21.5% of the target probability.  This is materially stricter than the
initially considered 10% raw design while reducing raw variance-driven sample
allocation by a factor of 6.25 relative to the former 2% raw target.

## Why this is valid

Reference validity requires:

1. an accurate primary estimate;
2. an unbiased independent crosscheck;
3. uncertainty-aware agreement; and
4. no use of the crosscheck to replace or tune the final estimate.

It does not require the corroborating estimator to have exactly the same
standard error as the primary estimator.  The combined-SE test explicitly
accounts for unequal precision.

This is a statistical-design change informed by burned development outcomes,
not a change to event thresholds, rBergomi dynamics, grid, likelihood,
estimator expectation, or final performance metric.  It is preregistered
before a new validation namespace is opened.

## Candidate design

The four already passed development requirements are carried only as bound
inputs, not yet promoted.  The remaining four use:

- three raw full-rank proposal schedules from the audited V3 result, each
  evaluated with natural mass 0.08, 0.12, and 0.20; and
- H=0.05 terminal DCS profiles from two audited CEM fits, each evaluated under
  four positive rank-one amplitude mixtures.

Increasing raw natural mass tightens the exact likelihood bound from 12.5 to
8.33 or 5.0 and may reduce single-contribution concentration.  All nonnatural
weights remain positive and proportional to their frozen base values.

## Gates

The validation keeps:

- 8 held-out replicates × 32,768 paths;
- independent seeded permutation before eight diagnostic blocks;
- allocation safety factor 6;
- maximum projected-to-cap ratio 0.75;
- likelihood-normalization absolute z at most 4;
- block variance ratio at most 30;
- full/block contribution shares at most 0.05/0.25;
- raw coverage at least 1,024 per replicate and 64 per block; and
- exact DCS rank-one structure.

The 0.75 development margin remains below the formal resource cap of 1.0.  A
passing development result still requires an immutable manifest and a new
formal pilot; it cannot directly authorize final execution.

## Pre-execution review

- Method-specific SE targets reach the allocation calculation directly.
- Event estimands and likelihoods are unchanged.
- DCS/raw agreement uses both estimated standard errors.
- No self-normalization or outcome-dependent block permutation is used.
- Proposal, label, and permutation seeds are distinct and collision checked.
- Prior artifacts are bound by SHA-256.
- Smoke, lint, and changed-file type checks pass.

## Authorization

- Execute method-role development validation: authorized
- Promote partial inputs or build a manifest before audit: forbidden
- Open formal pilot/final namespaces: forbidden
- Make performance or submission claims: forbidden
