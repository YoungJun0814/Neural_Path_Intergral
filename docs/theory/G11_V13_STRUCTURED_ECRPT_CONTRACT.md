# V13 Structured ECRPT Mathematical Contract

Date: 2026-08-11
Scope: finite-dimensional Gaussian residual laws

## T13-1: low-rank parameterization does not alter change of variables

Low-rank factorization changes only how a triangular conditioner is evaluated. The
coupling Jacobian remains triangular and its log determinant is the sum of the emitted
log scales. Exact forward and inverse evaluation therefore gives the exact flow
density in `R^(d-1)`. The Householder residual coordinate map is an isometry and has
unit absolute determinant.

## T13-2: defensive flow likelihood

For `Q=delta P+(1-delta)F#P`, `dQ/dP` is evaluated by a balance mixture and
`0<dP/dQ<=1/delta`. This bound applies whether the flow approximates the ideal target
well or badly.

## T13-3: pCN invariant reference and acceptance ratio

The pCN kernel `y'=sqrt(1-gamma^2)y+gamma xi` is reversible with respect to the
standard Gaussian reference. For a tempered target with density proportional to
`g(By)^beta` relative to that reference, Metropolis--Hastings acceptance is
`min(1, exp(beta(log g'-log g)))`. Adding a Gaussian-density term would double-count
the reference and is prohibited.

## T13-4: SMC does not define the final estimand

Adaptive beta selection, normalized incremental weights, resampling, and dependent
pCN particles are permitted for proposal training. Conditional on the frozen output,
fresh ordinary importance sampling under the exact defensive flow remains unbiased.
The SMC particles are not independent final units and their empirical variance cannot
be reported as estimator uncertainty.

## T13-5: empirical-Bernstein square certificate

Apply the fixed-function empirical Bernstein inequality to `Z=Y^2/M^2`. After
clipping the upper confidence endpoint at one and rescaling, `E[Y^2]` and hence
`Var(Y)` are upper bounded. Taking the minimum with separately valid Hoeffding and
deterministic range bounds preserves coverage because the three bounds are evaluated
at the same confidence only when each individually has that coverage for the same
fixed variable; no data-dependent method selection outside this intersection is
allowed. To avoid an invalid multiple-bound minimum, V13 allocates alpha across the
two stochastic bounds by Bonferroni and uses the deterministic range bound freely.

## T13-6: log-density stability

Let `h=log(q/q*)` and suppose `|h|<=epsilon` on the support of `q*`. Since
`q*/q=exp(-h)<=exp(epsilon)`,

\[
\chi^2(q^*\Vert q)
=E_{q^*}[q^*/q]-1
\leq e^\epsilon-1.
\]

Combining this with the exact ECRPT chi-square identity proves the declared relative-
variance bound. It is a finite-dimensional conditional theorem, not a proof that the
implemented optimizer attains the uniform log-density premise.

## T13-7: exact Rao--Blackwell variance ordering

For a residual proposal likelihood `L=p_R/q_R` and `g(R)=P(A<=a(R)|R)`, the law
of total variance gives

\[
\operatorname{Var}_q(L\mathbf 1\{A\leq a(R)\})-
\operatorname{Var}_q(Lg(R))
=\mathbb E_q[L^2g(1-g)]\geq0.
\]

Thus exact conditionalization cannot increase variance under the same residual
proposal. Strictness requires positive probability of `0<g(R)<1`; an empirical
positive gap is a diagnostic, not by itself a population proof.

## Open obligations

- a verifiable nonvacuous uniform log-density approximation result for the structured
  flow family;
- dimension/mesh-uniform pCN mixing for the rBergomi conditional potential;
- a rough-Volterra approximation theorem controlling the finest-grid bias;
- training-inclusive strict efficiency under a nontrivial task class;
- independent external proof review and current novelty review before submission.
