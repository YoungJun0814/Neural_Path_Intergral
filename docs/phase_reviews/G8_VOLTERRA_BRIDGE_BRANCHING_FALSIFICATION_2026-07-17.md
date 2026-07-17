# G8 Conditional Volterra Bridge Branching — Post-Implementation Audit

Date: 2026-07-17
Decision: **finite-grid mathematics retained; end-to-end model hypothesis rejected**

## 1. Executive decision

G8 implemented a new finite-grid principle for rough-volatility rare-event simulation:

> condition simultaneously on the coarse Brownian increment and the BLP
> singular local Volterra integral, then sample exact conditionally independent
> fine bridges and allocate extra branches using coarse-path information only.

The implementation is mathematically consistent at the declared finite grids.
It improves the work-normalized adjacent correction in the tested task, but it
does **not** make the complete estimator competitive with the frozen
single-level time-piecewise CEM baseline. The G8-2 hypothesis is therefore
falsified. A neural, attention, or quantum layer must not be added to this
branching design merely to continue it.

## 2. Mathematical object actually implemented

For one dyadic BLP block, define

\[
F=(X_0,X_1,X_2,Y_0,Y_1)^\top,
\qquad
C=AF=(X_0+Y_0, X_2+Y_1)^\top .
\]

With \(\Sigma_F=\operatorname{Cov}(F)\), the conditional gain is

\[
K=\Sigma_F A^\top(A\Sigma_F A^\top)^{-1}.
\]

For an independent \(F'\sim N(0,\Sigma_F)\), a conditional branch is sampled as

\[
F^{(m)}=KC + F'^{(m)}-KAF'^{(m)}.
\]

Consequently, \(AF^{(m)}=C\) pathwise and every branch has the exact BLP fine
marginal after integrating over \(C\). The second Brownian driver is refined by
the analogous Gaussian bridge conditioned on its coarse sum.

For deterministic fine-grid control \(u\), each branch uses its own exact
fine-space Girsanov likelihood

\[
L^{(m)}=\exp\!\left[-\sum_i u_i\cdot\Delta W_i^{Q,(m)}
-\frac12\sum_i\lVert u_i\rVert^2h\right].
\]

No coarse likelihood and no self-normalization are used. If the branch count
\(M(C)\) is chosen before observing fine residuals and is measurable with
respect to the coarse path, then

\[
\widehat Y=\frac1{M(C)}\sum_{m=1}^{M(C)}
(H_\ell^{(m)}-H_{\ell-1}^c)L^{(m)}
\]

is an unbiased finite-grid correction estimator. Its variance is

\[
\operatorname{Var}(\widehat Y)
=\operatorname{Var}(E[Y\mid C])
+E\!\left[\frac{\operatorname{Var}(Y\mid C)}{M(C)}\right].
\]

This is a finite-grid statement. It is not a continuous-monitoring theorem and
does not establish a positive weak rate for a discontinuous payoff.

## 3. Implementation map

- `src/path_integral/rbergomi_branching.py`: exact conditional Gaussian
  projection, coarse trunks, fine refinements, target reconstruction, and exact
  branch likelihoods.
- `src/evaluation/volterra_branching.py`: coarse-measurable fixed/adaptive
  allocation and unbiased variable-branch correction evaluation.
- `src/training/volterra_branching.py`: coarse-path feature map and development-
  only conditional-variance classifier.
- `experiments/g8_volterra_bridge_branching.py`: frozen development/validation
  split, checkpoints, total-work accounting, and stop gates.
- `configs/g8_volterra_bridge_branching.yaml`: original frozen G8 protocol.
- `configs/g8_volterra_bridge_branching_likelihood_audit.yaml`: explicitly
  post-hoc likelihood-diagnostic amendment.

## 4. Exactness and technical audit

The focused test suite checks:

1. Gaussian projection identity, rank, conditional mean, and conditional
   covariance;
2. conditional independence of branch residuals;
3. equality in law between one branch and the G7 adjacent-grid sampler;
4. exact reconstruction of deterministic-control likelihoods and target
   fine-to-coarse increments;
5. agreement of natural and controlled signed hard corrections;
6. the conditional variance decomposition;
7. coarse-only fixed and learned variable allocation without selection bias;
8. stable deterministic-Girsanov log-likelihood moment diagnostics.

The maximum conditional constraint error in the full audit was
\(3.55\times10^{-15}\). The aggregate standard-versus-branched correction
difference was \(z=0.679\), within the frozen \(|z|\le3\) requirement.

## 5. Likelihood gate correction

The original frozen run reported a maximum raw likelihood-normalization
studentized statistic of \(z=4.676\). The implementation audit found that the
control energies are approximately 5.97–6.59, implying likelihood coefficients
of variation of approximately 20–27. In this regime, a z statistic using the
sample variance of \(\exp(\log L)\) is itself unstable and can severely
underestimate its standard error.

The original result and protocol hash were preserved. A separate amendment was
created after observing the validation seeds and is therefore labelled
**post-hoc, not confirmatory**. It gates the known deterministic-control law

\[
\log L\sim N(-E/2,E)
\]

using its mean and variance. The maximum amended log-moment statistic was
2.407, so the implementation-level likelihood audit passed. The raw diagnostic
is still reported and was not deleted.

## 6. Frozen and amended results

| Quantity | Original frozen run | Likelihood audit run | Required |
|---|---:|---:|---:|
| Conditional constraint error | \(3.55\times10^{-15}\) | \(3.55\times10^{-15}\) | \(\le10^{-10}\) |
| Correction consistency \(|z|\) | 0.679 | 0.679 | \(\le3\) |
| Fixed branching levels improved by \(>1.10\) | 1/3 | 1/3 | at least 2/3 |
| Adaptive correction geometric work ratio | 1.893 | 1.938 | \(>1.10\) |
| End-to-end geometric work ratio | 0.325 | 0.299 | \(>1.25\) |
| Seeds with end-to-end improvement | 0/5 | 1/5 | at least 4/5 |
| Break-even | none | none | \(\le50\) queries |
| Correction variance log-slope | -0.372 | -0.372 | positive |
| Maximum raw likelihood z | 4.676 | 4.676 | original gate \(\le4\) |
| Maximum log-likelihood moment z | not recorded | 2.407 | audit gate \(\le4\) |

Ratios above one favor G8. Wall-clock ratios vary between executions, so they
must not be interpreted as hardware-independent complexity constants. The
failure margin is nevertheless large and the variance sequence itself is
unchanged between runs.

## 7. Why the complete estimator failed

The method can only remove
\(E[\operatorname{Var}(Y\mid C)]/M\). It cannot remove
\(\operatorname{Var}(E[Y\mid C])\), and every extra branch pays a full fine-path
refinement cost. The development estimates showed non-negligible removable
variance, but not enough at the two finest correction levels to pay for fixed
branching. The learned allocator improved the correction work proxy, yet the
correction variances did not decay across grids:

\[
(1.37,\ 2.39,\ 2.30)\times10^{-4}
\quad\text{at fine steps }(32,64,128).
\]

Thus the multilevel allocation continues to spend heavily on fine corrections.
The strong single-level CEM already targets the rare event effectively, and the
extra coarse trunk, classifier, and refinements cannot be amortized.

## 8. Research decision and next admissible pivot

Retain the exact conditional bridge as a verified library component and a
possible lemma in a future method. Stop the current `branch-more-near-boundary`
architecture.

The next mathematical direction must attack the irreducible conditional mean,
not merely average residual noise. A defensible next candidate is an exact-law
**conditional correction control variate**:

\[
Y-\widehat m_\theta(C)+E[\widehat m_\theta(C)],
\]

where the final term is obtained from an independent coarse-only estimator or
a provably integrable analytic surrogate. Cross-fitting is mandatory so that
the learned conditional mean cannot bias evaluation. The candidate advances
only if it demonstrates positive correction variance decay and end-to-end work
improvement before any operator or quantum extension.

## 9. Literature boundary

The BLP hybrid discretization originates in the rough Bergomi simulation work
at <https://arxiv.org/abs/1507.03004>. Repeated path branching for discontinuous
functionals has prior art in Markov diffusion settings at
<https://arxiv.org/abs/2209.03017>. Rough-Volterra weak-rate/PPDE results for
smooth test functions are discussed at <https://arxiv.org/abs/2304.03042>.
Therefore the exact bridge construction and its controlled finite-grid
combination may be research ingredients, but the present negative experiment
does not justify a top-journal performance or convergence claim.
