# G11 V8 T1 Theory and Novelty Decision

Date: 2026-08-02

Decision: **TERMINAL-ONLY INTERNAL THEORY CANDIDATE; SUBMISSION NOVELTY AND THE
TOP-JOURNAL ROUTE REMAIN BLOCKED**

## 1. What was completed

T1 completed the in-repository work that is independent of the falsified Stage B
broad-performance route:

1. refreshed the primary-source search through 2026-08-02;
2. hash-bound the update to the original P1 search ledger;
3. added fail-closed machine auditing for the update;
4. implemented a stable moment-localized population strictness certificate;
5. verified it against an independent numerical quadrature oracle;
6. expanded the terminal rBergomi proof obligations O1--O5; and
7. preserved the external-review, barrier-rate, and subscription-database blockers.

Machine-readable inputs and outputs:

- `configs/g11_v8/t1_novelty_update_v1.yaml`;
- `results/g11_v8_t1_novelty_audit_v1_2026-08-02.json`;
- update SHA-256
  `7a8f3fb0817161686727d61aaeaab9a8bbff037a88fd252762374e2bee6688f8`;
- base P1 ledger SHA-256
  `b3f31276194eee08333fd1d7c5eef68ea678a3e3c6840b2a6101316d4d820f19`.

The audit passes all 21 fail-closed checks over seven new primary sources and seven
recorded search families/queries.

## 2. Important literature correction

The earlier novelty matrix was incomplete on nonlinear rough-volatility weak error.
The following primary works materially narrow the claim:

| Primary work | Established prior art | Consequence |
|---|---|---|
| Friz, Salkeld, and Wagenhofer, [Weak error estimates for rough volatility models](https://arxiv.org/abs/2212.01591), [DOI](https://doi.org/10.1214/24-AAP2109) | weak rates for a model class including rough Bergomi, with an optimality lower bound in its smooth-test setting | no first nonlinear rough-volatility weak-rate claim |
| Gassiat, [Weak Error Rates of Numerical Schemes for Rough Volatility](https://arxiv.org/abs/2203.09298), [DOI](https://doi.org/10.1137/22M1485760) | hybrid-scheme weak rates; the analysis explicitly covers suitably weighted 3R-type schemes | hybrid weak rate and projection are prior art |
| Bayer, Fukasawa, and Nakahara, [On the weak convergence rate in the discretization of rough volatility models](https://arxiv.org/abs/2203.02943) | a general (2H) lower bound and a sharper linear-model result | a conservative (H)-scale statement is not novel alone |
| Fukasawa and Hirano, [3R hybrid refinement](https://doi.org/10.1080/14697688.2020.1866209) | orthogonal projection and random-number reuse for rough Bergomi simulation | orthogonal projection cannot be claimed as new |
| McCrickerd and Pakkanen, [Turbocharging rough Bergomi Monte Carlo](https://arxiv.org/abs/1708.02563) | conditional lognormal Monte Carlo and runtime-aware variance reduction | conditional rough Bergomi Monte Carlo is prior art |
| Bayer, Ben Hammouda, and Tempone, [Numerical smoothing](https://arxiv.org/abs/2111.01874) and [MLMC smoothing](https://arxiv.org/abs/2003.05708) | one-coordinate preintegration of nonsmooth events followed by QMC/MLMC | smoothing and smoothed MLMC are prior art and the closest computational boundary |
| Ahn and Zheng, [Conditional Importance Sampling for Convex Rare-Event Sets](https://doi.org/10.1109/WSC60868.2023.10408303) | conditional IS and bounded-relative-error analysis for convex events | neither the term nor generic conditional rare-event IS is new |
| Jacquier and Pannier, [Large and moderate deviations for stochastic Volterra systems](https://arxiv.org/abs/2004.10571) | pathwise LDP/MDP covering rough-volatility models | Volterra large-deviation controls are prior art |
| Burés, [Short-time up-and-in barrier analysis](https://arxiv.org/abs/2510.15423) | supremum concentration/density analysis with a rough Bergomi application | any future barrier anti-concentration proof must distinguish its mesh target and regime |

No full match to the exact defensive residual-likelihood construction was located in
this recorded search. That is an inference from a finite search, not proof of global
absence.

## 3. Surviving mathematical contribution

The strongest internally valid candidate is now:

> Exact proposal-conditional integration of a scalar rare-event direction under a
> defensive Gaussian balance mixture, together with an explicit finite-dimensional,
> rarity-dependent absolute variance-gap certificate.

The new certificate uses the exact component decomposition
(mu_j=b_ju+v_j), mixture weights (w_j), defensive mass, residual radius,
threshold moment (K_p), and rarity localization (A). It proves

\[
\operatorname{Var}(Y_{raw})-\operatorname{Var}(Y_{DCS})
\ge
\frac{\eta(A,R_0)}{M(R_0)}
\Phi(-A)^2\Phi(-A-B_\parallel)>0
\]

whenever the computable localization mass (eta) is positive. It makes the collapse
with rarity explicit and does not imply a factor-two variance ratio.

Relevant files:

- `docs/theory/G11_V8_MOMENT_LOCALIZED_STRICTNESS_THEOREM.md`;
- `src/path_integral/gaussian_mixture_strictness.py`;
- `tests/test_g11_v8_population_strictness.py`.

## 4. Terminal theory outcome

The internal terminal proof now explicitly covers:

- one common Brownian probability space and sigma-field;
- the exact BLP newest-cell/historical-cell convention;
- (L^p) Volterra and lognormal transfer at every (r<H);
- continuous/fine/coarse affine coefficients using one positive direction;
- uniform inverse terminal-slope moments;
- terminal threshold (L^2) convergence;
- weak terminal-indicator bias exponent (alpha=r);
- DCS correction second-moment exponent (eta=2r); and
- FFT cost exponent (gamma=1) with its logarithmic factor.

The resulting conservative terminal MLMC complexity is

\[
O(\varepsilon^{-1/r}\log\varepsilon^{-1}),\qquad r<H,
\]

which is unfavorable for very small (H). It is not an (O(\varepsilon^{-2}))
result.

The complete internal record is
`docs/theory/G11_V8_TERMINAL_RATE_INTERNAL_CLOSURE_2026-08-02.md`. It does not
replace the required independent specialist review.

## 5. What failed or remains open

### Scientific falsification already binding

Stage B remains binding:

- geometric DCS/raw variance ratio (1.9086<2.0);
- maximum primary-comparator combined z-score (4.4115>4);
- 24 primary resource-censored records; and
- no authorization for Stage C or P8--P11.

The T1 theorem cannot convert this failed broad empirical gate into a pass.

### Open mathematical obligations

1. no model-explicit sharp intercept moment constant (C_{A,2p}), hence no useful
   model-level rarity asymptotic for the new lower bound;
2. no discrete-barrier active-index and fine-only crossing rate;
3. no continuously monitored barrier theorem;
4. no independent stochastic-analysis proof review;
5. no MathSciNet/zbMATH/Scopus/Web of Science cited-reference completion; and
6. no external expert novelty challenge.

## 6. Gate decision

| Route | Decision | Reason |
|---|---|---|
| Current broad V8 computational claim | stop | frozen Stage B gates failed |
| Stage C and Q1/P8 | not authorized | D1 continuation precondition failed |
| P9--P11 submission evidence | not authorized | upstream gate failure |
| Terminal-only internal theorem | conditional pass as proof candidate | O1--O5 expanded; O6 external review missing |
| Barrier theorem | stop at finite-grid scope | rate and anti-concentration obligations open |
| Current top-journal submission | blocked | broad performance failed and surviving novelty is not externally established |
| Future terminal-only V9 program | permissible only as a new protocol | new estimand, thresholds, seeds, and falsification gates required |

## 7. Honest publication positioning

The current repository is a strong doctoral research artifact in reproducible
method development and falsification, but it is not a completed top-journal paper.
Three honest publication routes remain:

1. a rigorous terminal-only mathematical/numerical paper after external proof and
   novelty review;
2. a computational/reproducibility paper emphasizing exactness, cost accounting,
   and the negative broad-performance result; or
3. a new V9 terminal-only falsification program testing regime-conditional and
   repeated-query value without reusing the failed V8 claim.

The repository must not describe the present state as submission-ready, broadly
superior, or the first use of conditional Monte Carlo, orthogonal projection,
weak-rate analysis, Volterra large deviations, or Rao--Blackwellized importance
sampling.
