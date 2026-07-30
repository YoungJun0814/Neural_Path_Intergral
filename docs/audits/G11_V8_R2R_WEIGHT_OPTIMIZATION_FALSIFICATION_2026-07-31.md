# G11 V8 R2R — Weight-optimization partial falsification

Date: 2026-07-31

## Result

The exact raw mixture-weight training stage did not solve the complete
reference roster.  The complete selection increased from 3 to 4 of 8 because
the H=0.12 barrier 1e-3 DCS requirement passed with projected-to-cap ratio
0.339.  The remaining three raw requirements and H=0.05 terminal 1e-5 DCS
failed.

## Raw optimization finding

Training-only empirical second moments decreased:

- H=0.05 terminal 1e-4: approximately 19%;
- H=0.05 terminal 1e-5: approximately 19%; and
- H=0.12 barrier 1e-5: approximately 24%.

The held-out validation did not reproduce the same improvement.  Best
projected-to-cap ratios were approximately 1.669, 3.873, and 0.956,
respectively.  This is evidence of finite-sample weight-selection overfitting,
not an invalid off-policy identity.

The audit confirms:

- 17 candidates and 432 unique seeds;
- positive weights summing to one with natural mass 0.08;
- training objective nonincrease for all three fits;
- exact candidate-gate and minimum-ratio selection reconstruction;
- 4 of 8 complete requirements;
- clean source commit
  `f755f6d1850cccb41323df5c6e091f7798c42a6f`; and
- all downstream decisions fail closed.

## Method-role precision issue

The current allocation contract assigns both methods the same target standard
error:

\[
\mathrm{SE}_{DCS}=\mathrm{SE}_{raw}=0.02p.
\]

This is necessary for the primary DCS reference only if the final-method
comparison requires that precision.  `raw_crosscheck` has a different role:
it is an independent unbiased corroboration whose own standard error is
explicitly included in the DCS-versus-raw agreement statistic.

Requiring an equal 2% relative standard error for the hard-indicator raw
crosscheck is therefore not a mathematical condition for reference validity.
It forces the deliberately unsmoothed diagnostic to match the computational
precision of the smoothed primary estimator and obscures the practical value
of DCS.

## Required redesign

A new protocol may preregister method-specific precision before using new
seeds:

- DCS primary reference target: 2% relative SE;
- raw independent crosscheck target: 10% relative SE;
- DCS/raw agreement: combined standard error, fixed two-sided z gate;
- no substitution of the lower-precision raw estimate for the primary
  reference; and
- all estimates remain ordinary, non-self-normalized unbiased estimators.

The factor-five raw SE change reduces its variance-driven allocation by a
factor of 25.  This is a role-correct statistical design change, not an
estimator change.  It must be disclosed as informed by burned development
outcomes and validated in an entirely new namespace.

## Authorization

- Develop method-role precision protocol: authorized
- Promote partial V3/V4 development selections: forbidden
- Build proposal manifest or open formal pilot: forbidden
- Execute final references or make performance/submission claims: forbidden
