# G11 V10R1 full-latent DCS: theorem and claim boundary

Date: 2026-08-11  
Status: implementation contract, outcome blind

## 1. Finite-dimensional setting

Fix a finite rBergomi grid with `N` time steps.  The BLP simulator consumes

\[
X=(X^{\mathrm{loc}},X^{\mathrm{price}})\in\mathbb R^{3N},
\qquad X\sim P=N(0,I_{3N}).
\]

The first `2N` coordinates generate the Volterra driver and local Gaussian
integrals.  The final `N` coordinates generate the independent price Brownian
driver.  V10R1 never projects, drops, reconstructs approximately, or changes
any of these proposal coordinates.

For one independently trained and then frozen CEM mean `mu`, the proposal is

\[
Q=\alpha N(0,I_{3N})+(1-\alpha)N(\mu,I_{3N}),\qquad 0<\alpha<1.
\]

The ordinary likelihood is `L=dP/dQ`; self-normalization is prohibited.

## 2. Frozen scalar direction

Let `v` be the normalized, strictly positive vector obtained from the absolute
price-block coordinates of the frozen `mu`, with a deterministic positive
numerical floor.  Embed it as

\[
e=(0_{2N},v)\in\mathbb R^{3N}.
\]

Write `A=e^T X` and `R=X-Ae`.  The direction changes neither `Q` nor `mu`.
Because it acts only on the independent price block, the simulated variance
path is measurable with respect to `R`.  Positivity makes terminal log spot
strictly increasing in `A` whenever `|rho|<1` and variance is positive.

For a terminal left-tail event there is therefore an exactly computed
threshold `tau(R)` such that

\[
1\{S_T\le K\}=1\{A\le\tau(R)\}.
\]

This statement is only for the implemented finite grid.  It is not a
continuous-time or barrier-event theorem.

## 3. Exact residual likelihood identity

Decompose each mixture mean as `mu_j=(e^T mu_j)e+mu_{j,perp}`.  The marginal
proposal density of `R` relative to its standard Gaussian target law is

\[
\frac{dQ_R}{dP_R}(r)=
\sum_j w_j\exp\{r^T\mu_{j,\perp}-\|\mu_{j,\perp}\|^2/2\}.
\]

Hence `L_R=dP_R/dQ_R` is evaluated from all `3N-1` retained residual degrees
of freedom.  This is the term that exploratory V10 failed to preserve.

## 4. Unbiasedness and Rao--Blackwell theorem

Define

\[
Y_{raw}=1\{A\le\tau(R)\}L(X),\qquad
Y_{dcs}=L_R(R)\Phi(\tau(R)).
\]

Then

\[
E_Q[Y_{raw}]=E_Q[Y_{dcs}]=P(S_T\le K)
\]

and, exactly,

\[
Y_{dcs}=E_Q[Y_{raw}\mid R].
\]

Consequently

\[
\operatorname{Var}_Q(Y_{dcs})\le
\operatorname{Var}_Q(Y_{raw}).
\]

Equality is possible.  The theorem does **not** imply a uniform strict,
multiplicative, asymptotic-rate, or wall-clock improvement.

## 5. Defensive bounds

Because `Q >= alpha P`,

\[
0\le L(X)\le 1/\alpha,
\qquad 0\le L_R(R)\le 1/\alpha.
\]

These are finite-grid density bounds, not evidence of efficiency superiority.

## 6. What is and is not learned

CEM learns a full `3N` deterministic mixture mean.  It is not a new stochastic
dynamics model and it does not learn rBergomi parameters.  DCS analytically
integrates one standard-normal price direction from that frozen proposal.  The
scientific object being tested is therefore a structure-preserving rare-event
estimator, not a replacement asset-price model.

## 7. Statistical unit and training randomness

Each inferential cluster receives its own independently trained CEM proposal
and its own evaluation seeds.  Training cost is charged once to that cluster
and amortized only over the declared query count.  Paths sharing one proposal
are not treated as independent training replicates.  Development and any later
qualification use disjoint namespaces.

## 8. Claim boundary

The strongest result allowed by this contract before qualification is:

> On the frozen finite terminal grid, the full-latent estimator was implemented
> exactly, satisfied its correctness gates, and showed (or did not show) the
> prespecified regime-conditional empirical work ratios.

The following remain unauthorized: uniform superiority, barrier performance,
continuous-time unbiasedness, a new convergence rate, broad practical
dominance, top-journal readiness, and submission readiness.

