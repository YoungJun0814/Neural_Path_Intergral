# G11 V7 confirmation freeze V1 superseded

Date: 2026-07-24  
Status: superseded before any confirmation outcomes

The V1 confirmation freeze receipt bound source commit `1574964` and had SHA-256
`e61f4c3cc0442c9ef82c1f85d49926664839d9957d439564ff4ff789d618c392`.
No mechanism-probe or fixed-estimator confirmation outcome was generated from it.

The GitHub Actions Python 3.10 job exposed a cross-version numerical identity test
failure.  The production raw path evaluated the legacy exact balance likelihood,
whereas the paired mechanism path reconstructed the mathematically identical
likelihood through the generic DCS density routine.  Their values could differ by
roundoff even though the displayed tensors and mathematical estimator were the same.

The correction makes the paired diagnostic's raw member call the same hard-event
and legacy balance-likelihood expression as the isolated production raw path.  The
test is strengthened from an absolute tolerance comparison to bitwise
`torch.equal`.  This does not change DCS, the proposal, an outcome, a threshold, or
the experimental design.  It removes an avoidable implementation-path mismatch
before formal outcomes.

Because the source commit changed, the V1 receipt is not valid for execution.  A new
clean-commit V2 receipt is required after both Python 3.10 and 3.11 CI jobs pass.
