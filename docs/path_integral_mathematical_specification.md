# Path-Integral Mathematical and Implementation Specification

Status: PI0–PI1 implementation contract<br>
Last updated: 2026-07-13

This document fixes the measure, sign, Brownian-coordinate, objective, and
continuous-versus-discrete conventions used by the path-integral research
track. It is intentionally stricter than a narrative paper derivation: every
implemented quantity below has one orientation and one code location.

## 1. Scope

The project estimates a nonnegative path functional under a financial target
measure and learns a controlled Brownian proposal that reduces Monte Carlo
variance. The main future application is repeated rare-event and option queries
under rough volatility. This is not a market-crash prediction model and does
not infer a physical probability from risk-neutral option prices.

The implementation separates:

1. soft, strictly positive functionals used to train a path-law approximation;
2. hard indicators or contractual payoffs used in the final estimator;
3. a frozen proposal used only on fresh evaluation paths.

## 2. Measures and Brownian coordinates

Let \(\mathbb M\) be the target financial measure and \(\mathbb Q_u\) the
controlled simulation proposal. In an independent \(d\)-dimensional Brownian
basis,

$$
dB_t^{\mathbb M}=dB_t^{\mathbb Q_u}+u_t\,dt.
$$

The likelihood orientation is fixed throughout the repository as

$$
L_u=\frac{d\mathbb M}{d\mathbb Q_u},\qquad
\log L_u=-\int_0^T u_t^\top dB_t^{\mathbb Q_u}
          -\frac12\int_0^T\lVert u_t\rVert^2dt.
$$

The independent basis is mandatory for multidriver controls. A correlated
model Brownian motion is a derived linear combination, not another independent
likelihood channel. In the current one-driver Heston baseline, the controlled
coordinate is named `spot_brownian`; future full Heston and rBergomi code will
use `brownian_1` and `brownian_2` before applying correlation matrices.

## 3. Target functional and potential

For a strictly positive soft functional \(g_{\tau,\zeta}(X)\), define

$$
\Phi_{\tau,\zeta}(X)=-\log g_{\tau,\zeta}(X).
$$

The implemented terminal left-tail soft target is

$$
g_{\tau,K}(S_T)
=\operatorname{sigmoid}\left(\frac{K-S_T}{\tau K}\right),
$$

and its potential is evaluated stably as

$$
\Phi_{\tau,K}(S_T)
=\operatorname{softplus}\left(\frac{S_T-K}{\tau K}\right).
$$

`temperature=τ` is dimensionless and `barrier=K` is positive. The code returns
the potential rather than first forming \(g\), because the latter can underflow
far outside the target region.

The hard event \(1_{\{S_T\le K\}}\) is not passed to this potential interface.
It remains the terminal functional in the final likelihood-weighted estimator.

## 4. Tilted law and path-integral objective

Let

$$
Z=\mathbb E_{\mathbb M}[g(X)],\qquad
\frac{d\mathbb Q^\star}{d\mathbb M}=\frac{g(X)}{Z}.
$$

For admissible progressively measurable controls entering through the same
Brownian channels as the noise, with energy \(\frac12\int\lVert u\rVert^2dt\),
the formal Boué–Dupuis/path-integral objective is

$$
\mathcal J_{\mathrm{PI}}(u)
=\mathbb E_{\mathbb Q_u}\left[
  \Phi(X^u)+\frac12\int_0^T\lVert u_t\rVert^2dt
\right].
$$

Within the representable path-law class,

$$
\mathcal J_{\mathrm{PI}}(u)+\log Z
=\mathrm{KL}(\mathbb Q_u\Vert\mathbb Q^\star).
$$

Thus PI minimizes a forward KL. This identity requires the noise-control
matching condition and the stated quadratic cost. It must not be copied to a
model with a separate deterministic control channel or arbitrary penalty.

In a Markov or valid finite-dimensional lifted state, the desirability
\(\psi=e^{-V}\) satisfies a linear backward equation and the formal optimal
drift is \(u^\star=\sigma^\top\nabla\log\psi\). For rough non-Markovian models,
this is used only after specifying a causal lift or a path-space formulation.

## 5. Discrete path action

On a uniform grid with step \(h\), controls have shape
`(..., time_steps, brownian_drivers)` and are evaluated before their matching
proposal increments. The exact Gaussian mean-shift likelihood is

$$
\log L_u
=-\sum_k u_k^\top\Delta B_k^{\mathbb Q_u}
 -\frac12\sum_k\lVert u_k\rVert^2h.
$$

Define the path action

$$
\mathcal S_u
=\Phi(X^u)+\sum_k u_k^\top\Delta B_k^{\mathbb Q_u}
 +\frac12\sum_k\lVert u_k\rVert^2h.
$$

Then the unnormalized tilted trajectory weight is

$$
\omega_u=g(X^u)L_u=\exp[-\mathcal S_u].
$$

The equality `log_tilted_weight == -path_action == -potential + log_likelihood`
is a required unit-test invariant.

## 6. Three distinct divergences

For \(Y=gL_u\),

$$
\frac{Y}{Z}=\frac{d\mathbb Q^\star}{d\mathbb Q_u}.
$$

Therefore

$$
\frac{\operatorname{Var}_{\mathbb Q_u}(Y)}{Z^2}
=\chi^2(\mathbb Q^\star\Vert\mathbb Q_u),
$$

$$
\log\left(1+\frac{\operatorname{Var}_{\mathbb Q_u}(Y)}{Z^2}\right)
=D_2(\mathbb Q^\star\Vert\mathbb Q_u).
$$

Indeed,

$$
\chi^2(\mathbb Q^\star\Vert\mathbb Q_u)
=\mathbb E_{\mathbb Q_u}\left[
  \left(\frac{Y}{Z}-1\right)^2
\right]
=\frac{\mathbb E_{\mathbb Q_u}[Y^2]}{Z^2}-1
=\frac{\operatorname{Var}_{\mathbb Q_u}(Y)}{Z^2},
$$

and taking `log(1 + ·)` gives the order-two Rényi divergence. The equality
uses \(\mathbb E_{\mathbb Q_u}[Y]=Z\) and the absolute continuity required for
the displayed density ratio.

These are not the same objective as either PI or PICE:

| Stage | Mathematical target | Operational role |
|---|---|---|
| PI | \(\mathrm{KL}(\mathbb Q_u\Vert\mathbb Q^\star)\) | stable soft initialization |
| PICE | \(\mathrm{KL}(\mathbb Q^\star\Vert\mathbb Q_u)\) | target-law projection |
| \(J_2\) | \(D_2(\mathbb Q^\star\Vert\mathbb Q_u)\) | estimator-tail refinement |
| Final report | ordinary likelihood-weighted hard payoff | unbiased estimate on frozen fresh paths |

A decrease in PI loss alone is not evidence of variance reduction. The
relative second moment, contribution ESS, estimate, confidence interval, and
work-normalized efficiency must be reported separately.

The diagnostic implementation uses sample plug-in moments in the log domain.
Its empirical relative variance and Rényi-2 identity are exact for the supplied
sample, but the plug-in divergences are not unbiased population estimators.

## 7. PICE and off-policy coordinates

For behavior proposal \(\mathbb Q_{\bar u}\), the PICE target weight is

$$
\omega_{\bar u}=g(X)\frac{d\mathbb M}{d\mathbb Q_{\bar u}}.
$$

It must not be multiplied by an additional candidate likelihood. The candidate
appears in the weighted score or log-density only. First reconstruct the target
Brownian path,

$$
\Delta B_k^{\mathbb M}
=\Delta B_k^{\mathbb Q_{\bar u}}+\bar u_k h,
$$

then reconstruct candidate residuals causally,

$$
\Delta B_k^{\mathbb Q_u}
=\Delta B_k^{\mathbb M}-u_k h.
$$

For a constant independent-basis drift, the weighted score equation has the
closed-form projection

$$
\widehat u
=\frac{\sum_i\bar\omega_i B_T^{\mathbb M,(i)}}{T},\qquad
\bar\omega_i=\frac{\omega_i}{\sum_j\omega_j}.
$$

Self-normalization is allowed for this training projection. It is not used as
the final Monte Carlo estimator because it is biased at finite sample size.

## 8. Gaussian verification oracle

For \(B_T\sim N(0,T)\) under \(\mathbb M\) and
\(g(B_T)=\exp(aB_T)\),

$$
\log Z=\frac12a^2T,\qquad u^\star=a.
$$

For a constant proposal drift \(u\),

$$
\mathcal J_{\mathrm{PI}}(u)=-auT+\frac12u^2T,
$$

$$
\mathcal J_{\mathrm{PI}}(u)+\log Z
=\frac12(u-a)^2T,
$$

$$
\frac{\operatorname{Var}_{\mathbb Q_u}(gL_u)}{Z^2}
=\exp((a-u)^2T)-1.
$$

At \(u=a\), every simulated trajectory has the same log contribution
\(\frac12a^2T\). This simultaneously verifies the controlled-coordinate sign,
likelihood sign, action, normalizer, and zero-variance identity.

For the hard left tail \(B_T\le K\), with current state \(x=B_t\),

$$
h(t,x)=\Phi_N\left(\frac{K-x}{\sqrt{T-t}}\right),
$$

$$
u^\star(t,x)=\partial_x\log h(t,x)
=-\frac{\varphi((K-x)/\sqrt{T-t})}
        {\sqrt{T-t}\,\Phi_N((K-x)/\sqrt{T-t})}<0.
$$

This is a continuous conditional-law oracle. Its near-terminal singularity is
not evidence that a bounded finite-grid Gaussian control exactly realizes the
hard conditional law.

## 9. Continuous theory versus discrete claims

The following claims are allowed:

- a strictly positive Brownian functional admits a continuous path-space
  martingale/Föllmer representation under standard integrability conditions;
- the implemented simulator uses a left-adapted, piecewise-constant Gaussian
  mean-shift approximation;
- tests can identify the best control within a fixed proposal class and grid.

The following claims are prohibited without a separate proof:

- a finite-step Gaussian mean shift represents an arbitrary tilted discrete
  transition law;
- a bounded drift exactly realizes a hard event-conditioned law;
- low PI forward KL automatically implies low reverse chi-square divergence;
- a finite memory lift exactly makes rough volatility Markovian;
- self-normalized PICE weights give an unbiased final price or probability.

## 10. Code-to-mathematics map

| Mathematical object | Code |
|---|---|
| \(\Phi_{\tau,K}\) | `src.path_integral.terminal_left_tail_potential` |
| \(\log(d\mathbb M/d\mathbb Q_u)\) | `src.path_integral.brownian_log_likelihood` |
| \(\mathcal S_u\) | `src.path_integral.path_action` |
| \(\log(gL_u)\) | `src.path_integral.log_tilted_weight` |
| empirical \(\chi^2,D_2\) identity | `src.path_integral.tilted_divergence_diagnostics` |
| Gaussian analytic gate | `src.path_integral.gaussian_oracles` |
| constant PICE projection | `src.path_integral.fit_constant_pice` |
| candidate Brownian residual | `src.path_integral.reconstruct_candidate_increments` |

## 11. PI0–PI1 verification gates

- likelihood and path action agree in one and multiple Brownian dimensions;
- the soft potential equals a stable negative log-sigmoid;
- the exponential-tilt optimum recovers \(u=a\) and constant contribution;
- nonoptimal relative variance matches \(\exp((a-u)^2T)-1\);
- empirical relative variance equals chi-square and exponentiated Rényi-2;
- off-policy PICE recovers the analytic constant drift;
- the Gaussian left-tail drift matches a finite-difference log-probability
  gradient and has the correct negative sign;
- invalid shapes, nonfinite values, and nonpositive time steps fail loudly.

## 12. Related-work boundary for the implementation

| Existing line of work | Treated here as | Not claimed as novel | Candidate project contribution |
|---|---|---|---|
| Boué–Dupuis variational representation | foundation | free-energy/control identity | verified financial path-law formulation |
| Path-integral control / linearly solvable control | foundation | desirability transform | rough-volatility implementation with causal memory |
| PICE / adaptive importance sampling | baseline method | reverse-KL weighted projection | integration with PI initialization and \(J_2\) refinement |
| Neural importance sampling for options | strongest adjacent baseline | neural drift learning itself | rough-memory and amortized repeated-query study |
| Markovian approximations of Volterra processes | numerical representation | finite lift methodology | control-error and work-normalized lift ablation |
| Rough Bergomi simulation and martingale theory | model foundation | BLP scheme or martingale conditions | controlled two-driver BLP verification |

The publication claim must ultimately rest on a quantitatively verified rough
memory/amortization contribution, not on renaming Girsanov importance sampling
as path-integral control. See `PATH_INTEGRAL_RESEARCH_PLAN_V2.md` for the full
theory and experiment roadmap.
