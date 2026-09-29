# V13 Structured ECRPT: Implementation and Falsification Plan

Date: 2026-08-11
Status: development; qualification locked until the V13 gate passes
Parent: V12 ECRPT independently audited negative result

## 1. Decision

V12 proved that the finite-grid conditional estimator and its residual likelihood are
implemented correctly, but translated Gaussian mixtures trained by one-pass, adaptive,
or fixed annealed cross entropy do not cover the 384-dimensional rare residual region
reliably. V13 changes both the proposal family and the training sampler:

1. an exact defensive low-rank coupling flow with `O(L d r)` conditioner work;
2. an adaptive annealed SMC bridge with Gaussian-prior-preserving pCN moves;
3. a fresh evaluation stream using the original, untempered ordinary ECRPT mean.

No SMC normalizing-constant estimate is used as the reported probability. SMC and its
normalized weights are training machinery only. The final probability is always

\[
\widehat p=\frac1n\sum_{i=1}^n
g(R_i)\frac{p_R(R_i)}{q_R(R_i)},\qquad R_i\sim q_R.
\]

## 2. Prior-art boundary

- Sequential Monte Carlo samplers over a sequence of common-space targets are
  established by [Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
- Adaptive tempering using conditional ESS is established in
  [Zhou, Johansen, and Aston (2016)](https://doi.org/10.1080/10618600.2015.1060885).
- pCN proposals that preserve a Gaussian reference measure are established by
  [Cotter, Roberts, Stuart, and White (2013)](https://arxiv.org/abs/1202.0709).
- Empirical Bernstein confidence bounds are established by
  [Maurer and Pontil (2009)](https://arxiv.org/abs/0907.3740).
- Normalizing-flow rare-event importance sampling already exists; for example
  [NOFIS](https://arxiv.org/abs/2310.19167) uses a sequence of proposal distributions.

V13 therefore does **not** claim any of those ingredients individually. A potential
paper contribution must come from the exact Gaussian-Volterra conditional residual
law, exact defensive flow likelihood on its hyperplane, signed common-coordinate MLMC
extension, a new stability result, and demonstrated training-inclusive efficiency.

## 3. Exact structured proposal

Let `B:R^(d-1)->e^perp` be the Householder isometry already audited in V12 and let
`y=B^T R`. A coupling layer splits coordinates into active set `A` and transformed set
`C` and applies

\[
z_C=y_C\odot\exp s(y_A)+t(y_A),\qquad z_A=y_A.
\]

The conditioners are low rank:

\[
s(y_A)=s_{max}\tanh((y_AU_s)V_s+b_s),\qquad
t(y_A)=(y_AU_t)V_t+b_t,
\]

where the inner rank is `r << d`. Alternating parity or declared prefix/suffix blocks
give an exactly invertible triangular map. “Block causal” refers only to the declared
coordinate ordering; it is not a continuous-time adapted-control claim.

The reported proposal is

\[
q_R=\delta p_R+(1-\delta)f_\#p_R.
\]

The natural component implies `p_R/q_R <= 1/delta`. The complete balance-mixture
likelihood is recomputed from the frozen flow; no self-normalization is permitted.

## 4. Adaptive SMC training target

The bridge is

\[
\pi_\beta(dy)\propto g(By)^\beta\phi(y)dy,\qquad 0=\beta_0<\cdots<\beta_K=1.
\]

At a stage with equally weighted particles and cached `log g_i`, the next beta is the
largest value for which

\[
ESS(\Delta\beta)=
\frac{(\sum_i\exp(\Delta\beta\log g_i))^2}
{\sum_i\exp(2\Delta\beta\log g_i)}
\geq \rho N.
\]

It is found by deterministic bisection. The stage uses systematic resampling followed
by pCN rejuvenation

\[
y'=\sqrt{1-\gamma^2}y+\gamma\xi,qquad \xi\sim N(0,I),
\]

accepted with probability

\[
1\wedge\exp\{\beta[\log g(By')-\log g(By)]\}.
\]

The Gaussian density cancels because pCN preserves the reference law. Every beta,
ESS, resampling seed, pCN seed, acceptance rate, conditional evaluation, and work unit
is recorded. SMC particles may be dependent and are never treated as final inferential
units.

## 5. Fitting and selection

The final equally weighted SMC particles fit the low-rank flow by maximum likelihood
relative to the residual Gaussian. A small predeclared candidate set may vary rank,
number of layers, and partition style. Selection uses a disjoint screening stream and
charges:

- direction selection;
- every SMC conditional evaluation;
- every pCN proposal evaluation;
- all flow optimization trials;
- all screening evaluations;
- failed numerical restarts.

Final evaluation uses new Gaussian, mixture-label, and scalar-coordinate streams.

## 6. Tail-safe inference improvement

For a bounded unit `Y`, apply empirical Bernstein to
`Z=Y^2/M^2 in [0,1]`:

\[
E Z\leq \bar Z+
\sqrt{\frac{2\widehat V_Z\log(2/\alpha)}n}
+\frac{7\log(2/\alpha)}{3(n-1)}.
\]

The implemented upper bound is the minimum of empirical Bernstein, Hoeffding-on-
square, and the deterministic range bound. This remains a fixed-sample IID bound.
One RQMC randomization, not one Sobol point, is one inferential unit. If the bound is
still above the resource cap, the record remains censored.

## 7. Development matrix

Use the same three bound V12 cells and four independent training clusters so the new
architecture is compared against the retained negative result without changing the
scientific question. Primary methods are:

1. structured SMC-flow ECRPT;
2. exact conditional rBergomi;
3. smoothing RQMC;
4. V10R1 full-latent CEM-DCS.

Development may use a laptop-sized final budget, but target-work claims require an
uncensored forecast and accurate methods. All candidate and comparator estimates must
agree with the independent reference under the frozen combined-z threshold.

## 8. Gates

### G13-C correctness

- Householder round trip and residual projection `<=1e-11`;
- flow forward/inverse and log-Jacobian reconstruction `<=1e-10`;
- `E_q[p_R/q_R]=1` within `4 SE`;
- likelihood never exceeds `1/delta` beyond tolerance;
- SMC beta begins at zero, increases strictly, and ends exactly at one;
- every nonterminal adaptive stage meets the ESS target within bisection tolerance;
- pCN acceptance uses only the tempered conditional potential;
- paired raw-minus-ECRPT mean agrees with zero within `4 SE`;
- all training/evaluation streams are disjoint and hash-audited.

### G13-P performance

- all G13-C requirements pass;
- candidate and every primary comparator pass the combined-reference-z gate;
- no candidate or primary record is resource-censored;
- paired raw/ECRPT population mechanism is empirically resolved;
- best-primary/ECRPT geometric total-work ratio `>1.5`;
- one-sided 95% lower ratio `>1.0`;
- at least two of three cells favor ECRPT;
- all training and candidate-selection work is included at the declared query count.

Failure of G13-P blocks qualification. It does not invalidate G13-C.

## 9. Stability theorem target

For the ideal residual density `q*=g p_R/p`, if a frozen proposal satisfies

\[
\|\log(q/q^*)\|_\infty\leq\epsilon,
\]

then

\[
\frac{Var_q(gp_R/q)}{p^2}=\chi^2(q^*\Vert q)\leq e^\epsilon-1.
\]

This yields a strict-improvement certificate over natural residual sampling whenever
`e^epsilon-1` is smaller than its relative variance. The assumption is strong and is
not inferred from finite samples. V13 implements finite-oracle verification and keeps
rough-Volterra uniform control as an open theorem.

## 10. Qualification rule

Qualification is generated only if G13-P passes in the frozen development namespace.
Otherwise V13 must produce:

- the audited negative result;
- a resource forecast for a valid external run;
- the exact blocker list;
- no qualification result and no journal-performance claim.
