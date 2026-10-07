# G11 V12 ECRPT Theorem and Proof Contract

Date: 2026-08-11
Status: finite-dimensional implementation contract; rough-path rates remain open

## 1. Probability space

Let `X` be standard Gaussian in `R^d`, let `e` be a deterministic unit vector, and set

\[
A=e^T X,\qquad R=(I-ee^T)X.
\]

Then `A` is standard normal, `R` is Gaussian on `e^perp`, and `A` and `R` are
independent.  All statements below are conditional on a frozen training result when
`e` or a proposal was trained randomly.

## 2. T12-1: exact conditional reduction

Assume an event has the representation

\[
F(X)=1\{A\leq\tau(R)\}
\]

for a measurable extended-real threshold.  Then

\[
g(R):=E_P[F\mid R]=\Phi(\tau(R)),
\qquad p=E_{P_R}[g(R)].
\]

The code must handle positive and negative infinite thresholds explicitly and reject
NaNs.

## 3. T12-2: residual-mixture likelihood

Let `m_j in e^perp`, `pi_j>0`, and `sum pi_j=1`.  Define a mixture on the residual
subspace by translations of the target residual Gaussian:

\[
Q_R=\sum_j\pi_j N_R(m_j,I_R).
\]

Then

\[
\frac{dQ_R}{dP_R}(r)
=\sum_j\pi_j\exp(m_j^Tr-\|m_j\|^2/2).
\]

If `m_0=0` and `pi_0=delta`, then

\[
0<\frac{dP_R}{dQ_R}(r)\leq 1/\delta.
\]

No ambient Lebesgue density is asserted for the singular residual Gaussian.  Every
density is with respect to the Gaussian measure on `e^perp`.

## 4. T12-3: unbiased ECRPT estimator

For `R_i` sampled independently from `Q_R`, define

\[
Y_i=g(R_i)\frac{dP_R}{dQ_R}(R_i).
\]

Then `E_Q[Y_i]=p`.  If `0<=g<=1` and the proposal is defensive, then
`0<=Y_i<=1/delta`.  Ordinary averaging is required; self-normalization changes the
finite-sample estimand and is prohibited.

## 5. T12-4: paired Rao--Blackwell identity

Under the joint proposal `Q_R x P_A`, define

\[
Y_{raw}=1\{A\leq\tau(R)\}\frac{dP_R}{dQ_R}(R).
\]

Then

\[
E[Y_{raw}\mid R]=Y,
\qquad
\operatorname{Var}(Y)\leq\operatorname{Var}(Y_{raw}).
\]

This comparison uses the same residual proposal.  It does not imply superiority over
a separately optimized baseline or lower wall-clock work.

## 6. T12-5: optimal residual density and divergence identity

Assume `p>0` and `g>=0`.  The zero-variance residual density is

\[
q^*(r)=g(r)p_R(r)/p.
\]

For any proposal density `q` positive wherever `q^*>0`,

\[
\frac{\operatorname{Var}_q(gp_R/q)}{p^2}
=\int\frac{q^{*2}}q-1
=\chi^2(q^*\Vert q).
\]

This is an optimal-importance-sampling identity, not by itself a novelty claim.  V12
uses it as the exact training and error-analysis target.

## 7. T12-6: weighted cross-entropy target

For a parametric residual proposal `q_theta`, minimizing

\[
-E_{P_R}[g(R)\log(q_\theta(R)/p_R(R))]/p
\]

is equivalent, up to a `theta`-independent constant, to minimizing
`KL(q^* || q_theta)`.  The unknown probability `p` only scales the objective and is
not needed for normalized empirical weights.

This KL objective does not guarantee a chi-square or variance improvement.  The
chi-square-sensitive second moment must still be evaluated on held-out data.

## 8. T12-7: signed correction

Let `G(R)` be an integrable signed conditional correction and
`mu=E_P[G(R)]`.  For `R~q`, the unbiased contribution is `G(R)p_R(R)/q(R)`.
Among unrestricted proposal densities, the second moment is minimized by

\[
q^*(r)=|G(r)|p_R(r)/E_P|G(R)|.
\]

The minimal second moment is `(E|G|)^2`; zero variance occurs only when the sign is
almost surely constant.  Implementations must train with `|G|` and retain the sign in
the estimator.

## 9. T12-8: tail-safe bounded certificate

Let `Y in [a,b]`, `M=max(|a|,|b|)`, and let `Z=Y^2/M^2 in [0,1]`.  For `n` independent
samples, Hoeffding's inequality gives, with probability at least `1-alpha`,

\[
E[Z]\leq \bar Z+\sqrt{\log(1/\alpha)/(2n)}.
\]

Therefore

\[
\operatorname{Var}(Y)
\leq\min\left\{M^2 E[Z],\frac{(b-a)^2}{4}\right\}
\]

with the same confidence.  This bound is conservative but cannot collapse to zero
from a zero-variance rare pilot.  Pilot and final certificates use independent data.

## 10. T12-9: training randomness

Let `T` be training data independent of evaluation data.  If `e(T)` and `q_T` satisfy
the exact contracts almost surely, then the estimator is unbiased conditional on `T`.
Iterated expectation gives unconditional unbiasedness.  Reusing evaluation data to
select the direction or proposal is outside this theorem.

## 11. T12-10: finite-grid ML telescoping

For a common fine probability space, define

\[
G_0(R)=E[F_0\mid R],\qquad
G_l(R)=E[F_l-F_{l-1}\circ C_l\mid R].
\]

Each level may have its own frozen residual proposal, but one residual likelihood must
weight the entire level correction.  Independent ordinary means then satisfy

\[
E[\widehat G_0]+\sum_{l=1}^L E[\widehat G_l]=E[F_L].
\]

This proves only the declared finest-grid target.  A continuous-time statement needs
an additional weak-bias theorem.

## 12. T12-11: adaptive and annealed training laws

Suppose round `k` samples from an exact residual proposal `q_k` and fits the next
proposal with normalized training weights proportional to

\[
g(r)^{\beta_k}\frac{p_R(r)}{q_k(r)},\qquad 0<\beta_k\leq1.
\]

This is an importance-sampling approximation to the cross entropy for the annealed
target proportional to `g^beta_k p_R`.  Normalizing these weights is valid only in
the optimization objective.  If the final proposal is frozen and fresh evaluation
uses `g p_R/q_final`, the probability estimator remains unbiased for every training
outcome.  Annealing supplies no theorem of variance improvement; V12 V2/V3 are
empirical negative results at the declared budgets.

## 13. T12-12: exact residual coupling flow

Let `B:R^(d-1)->e^perp` be the orthogonal Householder coordinate map and let `f` be a
bijective triangular affine-coupling flow.  The flow density is computed by change of
variables in residual coordinates.  For

\[
q_R=\delta p_R+(1-\delta)f_\#p_R,
\]

the exact mixture ratio is available and `dP_R/dQ_R <= 1/delta`.  The Householder
map has unit absolute determinant on the two residual coordinate spaces; it creates
no omitted Jacobian factor.  Dense coupling-flow cost is not a performance theorem.

## 14. T12-13: task-conditioned emission

A trained deterministic generator may emit direction, residual translations, and
mixture weights.  Once these numeric values are projected, validated, and frozen,
the emitted proposal has the same exact likelihood theorem as any directly trained
proposal.  Generator approximation error changes proposal efficiency but not
conditional unbiasedness.  Any held-out generalization or amortized-work advantage
requires separate evidence.

## 15. Implementation proof boundary

The V12 tests establish finite-dimensional identities, input-contract rejection,
likelihood normalization, stable signed corrections, and compatibility with the
existing finite-grid MLMC engine.  The V1--V3 development experiments do not establish
strict efficiency.  In particular, finite samples that miss raw events can show zero
empirical raw variance even though the Rao--Blackwell theorem is true; the gate treats
such observations as unresolved rather than as infinite improvement.

## 16. Open proof obligations

- strict efficiency within the implemented finite mixture family;
- chi-square control from the weighted-KL training objective;
- rBergomi threshold convergence uniform over a declared parameter set;
- barrier active-index and monitoring-mesh stability;
- task-conditioned proposal generalization;
- multilevel variance lower bounds and sharp complexity;
- continuous-monitoring bias.

These obligations may not be replaced by fitted numerical slopes.
