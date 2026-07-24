# G11 V8 P3 Mesh, Rate, and MLMC Complexity Audit

Date: 2026-07-25

Status: terminal rate chain remains conditional; barrier rate is open; finite-grid
identities and MLMC algebra are executable and audited

## 1. Decision

P3 does not promote the current model to an unconditional rough-Bergomi complexity
theorem.

- The terminal route has a coherent proof candidate at every \(r<H\), conditional
  on the continuous/discrete coupling and coefficient lemmas receiving a complete
  mathematical proof audit.
- The discretely monitored barrier has an exact finite-grid threshold
  decomposition, but its early-active-time and fine-only mesh moments do not yet
  have model-level rates.
- The standard MLMC implication is now implemented with explicit
  \((\alpha,\beta,\gamma)\) provenance and FFT logarithmic factors. Fitted slopes
  cannot authorize an asymptotic claim.
- The strongest currently authorized mathematical statement is therefore a
  conditional terminal corollary plus exact finite-grid barrier diagnostics.

This is the explicit P3 scope downgrade allowed by the V8 plan. It protects the
paper from turning a numerical regression into a theorem.

## 2. Exact adjacent threshold decomposition

For a terminal task there is one candidate, so the active-index and fine-only mesh
terms are zero. For a barrier, let

\[
f_k=\frac{\log B-A^h_{2k}}{B^h_{2k}},
\qquad
c_k=\frac{\log B-A^{2h}_k}{B^{2h}_k}
\]

denote the fine and coarse candidates on embedded common times. Let
\(k_c\in\arg\max_k c_k\), let

\[
F_c=\max_k f_k,\qquad C=\max_k c_k,
\]

and let \(F\) be the maximum over every fine monitoring time. Then the exact signed
identity is

\[
\begin{aligned}
F-C
&=
\underbrace{(f_{k_c}-c_{k_c})}_{D_{\mathrm{coefficient}}}
+
\underbrace{(F_c-f_{k_c})}_{D_{\mathrm{active}}}
+
\underbrace{(F-F_c)}_{D_{\mathrm{mesh}}}.
\end{aligned}
\]

The active and mesh terms are nonnegative; the coefficient term is signed. The
mesh term is exactly the improvement available from fine-only monitoring times.
It cannot be dropped when transferring a terminal argument to a barrier.

The implementation records:

- `coarse_active_coefficient_defect`;
- `common_active_switch_defect`;
- `mesh_enrichment_defect`; and
- `maximum_signed_decomposition_residual`.

The previous field named `maximum_exact_decomposition_violation` checked only the
deterministic absolute-error bound. P3 corrects this semantic overstatement: it now
also requires the signed three-term identity to hold. This was a diagnostic naming
and coverage defect, not a bias in the probability estimator.

## 3. From coefficient error to DCS correction variance

For the terminal task, write

\[
\log S_T^h=A_h+B_hZ,\qquad
a_h=\frac{\log K-A_h}{B_h}.
\]

The conditional proof candidate has the following chain for every \(r<H\):

\[
\|A_h-A_{2h}\|_{L^p}+\|B_h-B_{2h}\|_{L^p}
\leq C_{p,r}h^r,
\]

\[
\|a_h-a_{2h}\|_{L^2}\leq C_rh^r,
\]

and, using the global Lipschitz constant of \(\Phi\) and the defensive likelihood
bound,

\[
E_Q\!\left[
\bar L^2\{\Phi(a_h)-\Phi(a_{2h})\}^2
\right]
\leq C_rh^{2r}.
\]

Thus the conditional correction-variance exponent is

\[
\beta=2r.
\]

This implication is correct if its premises hold. The remaining submission-level
obligations are not hidden:

1. one common Brownian probability space and filtration for continuous, fine, and
   coarse models;
2. the implementation-specific BLP Volterra estimate with the exact singular cell
   and historical cell averages;
3. uniform lognormal moments and the BDG transfer to price integrals;
4. a code-level affine coefficient decomposition measurable with respect to the
   declared residual sigma-field; and
5. a continuous terminal limit with a common \(L^2([0,T])\) direction.

Until these are discharged line by line and externally reviewed, \(\beta=2r\) is a
conditional theorem premise, not a proved submission claim.

## 4. Weak bias

The desired terminal weak-bias statement is

\[
\left|P(S_T^h\leq K)-P(S_T\leq K)\right|
\leq C_rh^r,
\qquad r<H.
\]

The ratio-localization and Gaussian-CDF step are valid conditional on convergence
of the continuous and discrete affine coefficients. The latter is still part of
obligations 1--5 above. Therefore

\[
\alpha=r
\]

is recorded as conditional.

No continuously monitored barrier weak-bias claim is made. The current barrier
target is discretely monitored, and a continuous-monitoring limit would require a
separate crossing analysis.

## 5. Cost exponent

The FFT path construction has arithmetic work

\[
C_h=O\!\left(h^{-1}\log(h^{-1})\right).
\]

The abstract cost contract is therefore

\[
\gamma=1,\qquad \kappa=1,
\]

where \(\kappa\) is the logarithmic cost power. This is an algorithmic upper bound,
for a fixed finite mixture/control bank and fixed integration rank. If either grows
with mesh refinement, its cost must be added and \(\gamma\) rederived. This is not a
wall-clock superiority result. Training, proposal fitting, pilot allocation,
checkpointing, and heterogeneous hardware remain in the experimental total-work
ledger.

## 6. Correct MLMC algebra with logarithmic cost

Assume

\[
|\operatorname{bias}_h|=O(h^\alpha),\quad
V_h=O(h^\beta),\quad
C_h=O(h^{-\gamma}\log^\kappa(1/h)),
\]

with

\[
\alpha\geq\frac12\min(\beta,\gamma).
\]

The optimized allocation sum gives the following regimes:

| Regime | Polynomial work | Logarithmic power |
|---|---:|---:|
| \(\beta>\gamma\) | \(\epsilon^{-2}\) | \(0\), except a boundary finest-sample term |
| \(\beta=\gamma\) | \(\epsilon^{-2}\) | \(2+\kappa\) |
| \(\beta<\gamma\) | \(\epsilon^{-2-(\gamma-\beta)/\alpha}\) | \(\kappa\) |

The implementation also compares this allocation cost with the mandatory cost of
one finest-level sample,

\[
\epsilon^{-\gamma/\alpha}\log^\kappa(\epsilon^{-1}).
\]

This matters at the boundary \(\alpha=\gamma/2\) in the
\(\beta>\gamma\) regime, where the finest sample retains the \(\kappa\) log factor.

For the conditional terminal rBergomi triplet

\[
\alpha=r,\qquad \beta=2r,\qquad \gamma=1,\qquad \kappa=1,
\]

and \(r<H<1/2\), one has \(\beta<\gamma\). Hence

\[
\operatorname{Work}(\epsilon)
=
O\!\left(
\epsilon^{-1/r}\log(\epsilon^{-1})
\right).
\]

The conservative examples with \(\varepsilon_H=0.01\) and
\(r=H-\varepsilon_H\) are:

| \(H\) | \(r\) | conditional \(\beta\) | polynomial exponent | log power |
|---:|---:|---:|---:|---:|
| 0.05 | 0.04 | 0.08 | 25.000 | 1 |
| 0.12 | 0.11 | 0.22 | 9.091 | 1 |
| 0.30 | 0.29 | 0.58 | 3.448 | 1 |

These severe exponents are a negative theoretical limitation, not a result to
replace with favorable fitted slopes. DCS can still be practically useful on a
fixed finest grid without improving the continuous-target asymptotic exponent.

## 7. Evidence-provenance contract

Every input rate is labelled as one of:

- `proved_internal`;
- `proved_external`;
- `conditional`;
- `empirical`; or
- `open`.

The executable certificate applies these rules:

1. any `open` premise leaves the complexity result open;
2. any `empirical` premise makes the result diagnostic only;
3. a `conditional` premise permits only a conditional corollary;
4. an unconditional result requires all three premises to be proved and the MLMC
   compatibility condition to pass; and
5. failure of the compatibility condition returns no complexity exponent.

The specialized terminal rBergomi helper has no boolean promotion switch. Even
after an external review occurs, promotion requires a new hash-bound review
artifact, a revised rate ledger, and a new audited code path. A fitted result can
never trigger that promotion.

## 8. Barrier boundary

For a finite-grid barrier, P2 exactness and strict same-grid variance reduction
remain valid. A barrier model-rate theorem additionally needs bounds for:

- the coefficient defect at the coarse active index;
- probability and moments of an early active time with a small slope;
- the common-grid active-index switch;
- the fine-only mesh-enrichment term; and
- continuous-monitoring bias, if that target is later introduced.

P3 has implemented the exact decomposition and diagnostics but has not proved
these rates. Therefore the barrier remains a finite-grid experiment and cannot
inherit the terminal \((\alpha,\beta,\gamma)\) triplet.

## 9. P3 gate

Authorized after P3:

- exact finite-grid signed barrier decomposition;
- terminal conditional rate chain at each \(r<H\);
- correct conditional FFT-MLMC algebra including log factors;
- finite-grid barrier experiments; and
- P4 strong-baseline framework implementation.

Not authorized after P3:

- unconditional terminal weak-bias or complexity language;
- any barrier correction-variance or complexity rate;
- continuously monitored barrier exactness;
- \(O(\epsilon^{-2})\) rough-regime complexity;
- treating empirical slopes as proofs; or
- training-inclusive superiority from the asymptotic algebra.

The theory-led top mathematical-finance journal route remains open only if the
terminal proof obligations are independently discharged or a new barrier theorem
is proved. Otherwise the defensible route is a strong computational paper built on
the exact finite-grid mechanism and total-work comparison.
