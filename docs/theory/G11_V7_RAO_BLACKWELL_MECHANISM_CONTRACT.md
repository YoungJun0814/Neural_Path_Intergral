# G11 V7 Rao--Blackwell mechanism contract

Date: 2026-07-24  
Status: finite-grid identity proved; empirical effect size not yet qualified

## 1. Objects

Let \(P=N(0,I_d)\) be the target law of the standardized finite-grid Gaussian
input.  The frozen defensive proposal is

\[
Q=\sum_{j=0}^{J-1}\pi_j N(m_j,I_d),
\qquad \pi_j>0,\quad \sum_j\pi_j=1,
\]

and includes a zero-shift component of mass \(\delta>0\).

Let \(U\) be the deterministic orthonormal control-span basis and decompose

\[
X=UZ+R,\qquad U^\top R=0.
\]

For the implemented scalar finite-grid downside event, conditioning on \(R\)
gives an exactly computable target-law probability

\[
G(R)=P_P(A\mid R).
\]

Write

\[
L(X)=\frac{dP}{dQ}(X),\qquad
\bar L(R)=\frac{dP_R}{dQ_R}(R).
\]

The raw and DCS contributions are

\[
Y=L(X)1_A,\qquad D=\bar L(R)G(R).
\]

Neither estimator is self-normalized.

## 2. Exact conditional-expectation identity

For every \(R=r\) with positive proposal residual density,

\[
\begin{aligned}
E_Q[Y\mid R=r]
&=\int 1_A(z,r)\frac{p(z,r)}{q(z,r)}
                 \frac{q(z,r)}{q_R(r)}\,dz\\
&=\frac{p_R(r)}{q_R(r)}
  \int 1_A(z,r)p_P(z\mid r)\,dz\\
&=\bar L(r)G(r)=D.
\end{aligned}
\]

The proposal conditional law \(Z\mid R\) is generally a
residual-dependent mixture.  The proof does not replace it by a standard
normal.  The standard-normal CDF enters only through the *target*
conditional integral after the exact likelihood cancellation.

Consequently:

\[
E_Q[D]=E_Q[Y]=P_P(A),
\]

\[
\operatorname{Cov}_Q(D,Y-D)=0,
\]

and

\[
\operatorname{Var}_Q(Y)
=\operatorname{Var}_Q(D)
E_Q[\operatorname{Var}_Q(Y\mid R)]
\geq \operatorname{Var}_Q(D).
\]

This is the finite-grid Rao--Blackwell mechanism.  It is an exact identity,
not an empirical fitted-rate assertion.

## 3. Defensive moment contract

The zero component gives

\[
q(x)\geq\delta p(x),\qquad
0<L(x)\leq\delta^{-1}.
\]

Marginalizing the same inequality gives

\[
q_R(r)\geq\delta p_R(r),\qquad
0<\bar L(r)\leq\delta^{-1}.
\]

For bounded event contributions, raw and DCS second moments are finite.  The
implemented proposal uses \(\delta=0.15\), so both likelihoods have the
declared bound \(1/\delta=6.\overline 6\).

## 4. What the theorem does and does not imply

The theorem implies population variance non-increase under:

- the identical frozen proposal;
- deterministic controls;
- exact full and residual likelihoods;
- the declared finite-grid event;
- one common control coordinate; and
- no clipping or self-normalization.

It does not imply:

- a uniform variance ratio strictly greater than one;
- lower total work after proposal construction, planning, or routing;
- a wall-clock gain if conditional integration is expensive;
- a continuous-monitoring barrier result;
- an rBergomi weak-bias or complexity rate; or
- novelty of Rao--Blackwellization by itself.

The publishable novelty must therefore come from the rough-volatility
control-span construction, its exact path-functional threshold map, a
model-level theorem, and demonstrated end-to-end performance--not from
renaming the classical conditional-expectation inequality.

## 5. Executable contract

| Obligation | Implementation |
|---|---|
| raw \(Y\) without DCS arithmetic | `RBergomiMLMCSampler.__call__`, raw fast path |
| matched \(Y,D\) on identical paths | `RBergomiMLMCSampler.sample_raw_dcs_pair` |
| hybrid-profile forwarding | `RBergomiHybridTermSampler.sample_raw_dcs_pair` |
| residual mean and orthogonality diagnostics | `rao_blackwell_pair_diagnostics` |
| exact threshold and density reconstruction | `evaluate_rbergomi_dcs_level` |
| strict defensive component | sampler natural-component validation |
| no hidden normalization | raw and DCS contributions returned unnormalized |
| work separation | paired probe has `mechanism_probe` category |

The paired probe is a diagnostic expense and cannot be credited as production
work for either method.  Isolated production wall times come from separate raw
and DCS executions.  The common work-unit proxy measures simulated
path-step/FFT work and intentionally does not claim instruction-level equality.

## 6. Numerical falsification

For a common-random-number batch, V7 records:

- \(E_n[Y]\) and \(E_n[D]\);
- \(E_n[Y-D]\) and its standard error;
- sample variances of \(Y,D,Y-D\);
- sample covariance and correlation of \(D\) with \(Y-D\);
- raw/DCS sample-variance ratio; and
- the numerical variance-decomposition residual.

Finite-sample residual means or covariances need not be exactly zero.  They are
falsification diagnostics, not proof.  Exact code-level equality is separately
tested by comparing the raw-only and paired paths under identical seeds; the
current maximum observed difference is one float64 ULP.

## 7. Remaining theory obligation

The finite-grid identity above is already covered at greater generality by
T11-2 in `G11_THEOREMS.md`.  The unresolved top-journal theory is not this
identity.  It is the independent verification of the rough-Bergomi
coefficient/weak-rate proof and, if claimed, a separate barrier enrichment
theorem.  V7 must keep these layers distinct.
