# Exact Conditional Cameron–Martin Transport for Rare Events in Gaussian Volterra Models

Draft date: 2026-08-11

Submission status: **working-paper core; not authorized for top-journal submission**.

## Abstract

We study terminal rare-event probabilities in Gaussian Volterra stochastic
volatility models.  Conditioning on the volatility-correlated Gaussian driver
analytically removes the complete independent price driver and produces a smooth,
bounded path functional.  We define its finite-noise Cameron–Martin action, locate
multiple stationary modes with independently implemented L-BFGS and trust-region
Newton solvers, and construct a defensive mixture of finite-rank Gaussian Laplace
transports.  Every component has an exact density relative to the finite-grid
Gaussian reference; a fixed natural component gives the pathwise likelihood bound
`dP/dQ <= 1/delta`.  A structure-preserving neural operator may amortize mode
initialization, but deterministic correction and natural fallback keep estimator
validity independent of network accuracy.  In a frozen three-cell rBergomi
qualification, the estimator agrees with independent conditional references and
improves training-inclusive 100-query work-normalized variance over the strongest
accuracy-qualified comparator by factors 1.43, 4.69, and 3.96.  A declared
small-noise diagnostic reaches probability `1.34e-6`, relative variance 0.50, and a
second-moment exponent ratio 0.985.  These are finite-grid empirical results.  A
model-specific small-noise efficiency theorem and continuous-time mesh-rate theorem
remain open, so no asymptotic-optimality claim is made.

## 1. Problem and motivation

Rough-volatility models are non-Markovian: the current volatility retains a
singular-kernel memory of the driving Brownian path.  Direct Monte Carlo therefore
suffers both from rare terminal events and from a high-dimensional path input.  The
goal is not merely to attach a neural network to an existing importance sampler.
The goal is an exact, auditable path-space transport whose learned part changes cost
and variance, never the target probability.

For independent Brownian motions `W,B`, the rBergomi terminal log price is

```text
log S_T = log S_0 + rho int sqrt(V(W)) dW
          + sqrt(1-rho^2) int sqrt(V(W)) dB
          - 0.5 int V(W) dt.
```

Conditioning on `W` makes the final term driven by `B` Gaussian.  This removes the
entire independent price-driver path before transport learning.

## 2. Relation to prior work

The method deliberately does not claim the following ingredients individually as
new:

- conditional Monte Carlo for rBergomi, developed by McCrickerd and Pakkanen,
  [DOI 10.1080/14697688.2018.1459812](https://doi.org/10.1080/14697688.2018.1459812);
- most-important Gaussian paths and curvature adaptation, developed by Glasserman,
  Heidelberger, and Shahabuddin,
  [DOI 10.1111/1467-9965.00065](https://doi.org/10.1111/1467-9965.00065);
- the rough-Bergomi pathwise LDP of Jacquier, Pakkanen, and Stone,
  [DOI 10.1017/jpr.2018.72](https://doi.org/10.1017/jpr.2018.72);
- LDP-based adaptive importance sampling and likelihood-informed subspaces of Tong
  and Stadler,
  [DOI 10.1137/22M1524758](https://doi.org/10.1137/22M1524758);
- nonasymptotic suboptimal-IS bounds of Hartmann and Richter,
  [DOI 10.1137/21M1427760](https://doi.org/10.1137/21M1427760);
- neural approximation of Cameron–Martin importance-sampling drifts by Arandjelović,
  Rheinländer, and Shevchenko,
  [DOI 10.1007/s00780-024-00549-x](https://doi.org/10.1007/s00780-024-00549-x).

The candidate contribution is the complete price-driver contraction combined with
an exact defensive multimode finite-rank transport, explicit mesh and claim audits,
and an optional certified amortization layer.  Whether this combination is
sufficiently novel for a top journal still requires an external closest-work review
and closure of the open theorem.

## 3. Exact conditional target

For `|rho|<1`, conditionally on `W`,

```text
m(W) = log S_0 + rho int sqrt(V) dW - 0.5 int V dt,
c(W) = (1-rho^2) int V dt,
g_K(W) = Phi((log K-m(W))/sqrt(c(W))).
```

Thus `P(S_T<=K)=E[g_K(W)]`.  Digital, put, and call conditional formulas are
implemented in the log domain and checked against extreme-tail and put-call-parity
oracles.

For `P_K=E_P[g_K]`, the conditional zero-variance target is
`dPi*/dP=g_K/P_K`, and every proposal `Q` obeys

```text
Var_Q(g_K dP/dQ) / P_K^2 = chi_square(Pi* || Q).
```

This fixes the divergence direction; `chi_square(Q || Pi*)` would be incorrect.

## 4. Cameron–Martin action and transport

For the declared small-noise family, a finite-noise action in coefficient `h` is

```text
J_epsilon(h) = 0.5 ||h||^2 - epsilon log g_epsilon(h/sqrt(epsilon)).
```

The DCT Galerkin basis is orthonormal in the whitened BLP coordinates, so Euclidean
coefficient energy equals the finite-grid Cameron–Martin energy.  Deterministic
multistart uses two algorithmically independent solvers and rejects nonpositive
curvature when constructing a Laplace component.

For orthonormal `U` and positive spectrum `lambda`, each component is

```text
N(mu, I + U diag(lambda-1) U^T).
```

Woodbury inversion and the determinant lemma give the exact density.  The proposal

```text
Q = delta P + (1-delta) sum_j alpha_j Q_j
```

satisfies `dP/dQ<=1/delta`.  The final estimator is an ordinary mean; neither weight
clipping nor self-normalization is permitted.

## 5. Neural amortization without neural correctness assumptions

The operator maps task and kernel features to bounded mode coefficients, normalized
mode weights, and SPD precision matrices with clipped eigenvalues.  Its output is a
warm start.  A deterministic corrector checks the action gradient and curvature;
failure returns the natural conditional proposal.  Consequently the network cannot
introduce bias, although a poor prediction can remove any efficiency benefit.

Current evidence for this layer is architectural, oracle, and synthetic-teacher
validation.  A large in-domain/OOD teacher-bank experiment is still required before
the operator is a headline empirical contribution.

## 6. Proven statements and open theorem

The exact conditional representation, frozen-proposal unbiasedness and
Rao–Blackwell ordering, zero-variance-target identity, and defensive moment bound are
proved in `docs/theory/G11_V15_THEOREMS.md` and linked to executable oracles.

The small-noise efficiency statement is open.  It requires an LDP for the exact
declared family, continuity/localization of the stochastic integral map, existence
and coverage of every dominating minimizer, and a multimode second-moment upper
bound.  The published rough-Bergomi small-time LDP is not silently reused as a
fixed-strike small-noise theorem.

## 7. Frozen finite-grid qualification

All cells use 32 BLP steps, terminal left-tail events, independent training and
evaluation seeds, and a primary repeated-query count of 100.  Accuracy is compared
with a separate natural conditional stream.  Work includes proposal training.

| Cell | Reference probability | V15 estimate ± SE | Accuracy z | strongest-primary/V15 work ratio |
|---|---:|---:|---:|---:|
| `H=.05,K=70` | 0.0624653 | 0.0634565 ± 0.000750 | 0.994 | 1.430 |
| `H=.12,K=50` | 0.0133285 | 0.0128852 ± 0.000150 | 1.260 | 4.689 |
| `H=.20,K=35` | 0.00437530 | 0.00395005 ± 0.000051 | 2.220 | 3.964 |

All five primary comparators—natural conditional MC, V14 local Volterra transport,
defensive CEM, finite-grid LD importance sampling, and smoothing RQMC—are present and
accuracy-qualified in every cell.  Proposal hash checks and defensive likelihood
bounds pass.  The matrix is too small and not rare enough to be a final top-journal
experiment.

## 8. Small-noise and mesh diagnostics

At `H=.12,K=70`, 16 steps, the independent frozen-proposal diagnostic gives:

| epsilon | probability | relative variance | second-moment exponent ratio |
|---:|---:|---:|---:|
| 1.0 | 0.0631681 | 0.535 | 0.923 |
| 0.5 | 0.0115053 | 0.500 | 0.955 |
| 0.25 | 0.000508040 | 0.530 | 0.972 |
| 0.125 | 0.00000134426 | 0.502 | 0.985 |

The trend is consistent with, but does not prove, asymptotic efficiency.  At the
rarest point the natural conditional estimate has standard error comparable to its
mean, so the precise reference is a disjoint stream from the same frozen V15
proposal.

The 50,000-sample mesh v2 study passes a numerical RMSE bias budget through 512
steps.  Adjacent corrections have high correlation (`0.952` to `0.972`) but only
roughly one to two standard errors of signal, and the fitted rate is `0.124`.
Accordingly no continuous-time rate is claimed.

## 9. Limitations and required completion

Before a top-journal submission, the project needs:

1. a complete and independently reviewed T15-5 proof or a narrower theorem that is
   still materially new;
2. a model-specific weak-bias/mesh theorem, or a paper scope that explicitly remains
   finite-grid;
3. a much broader frozen matrix spanning truly rare probabilities, Hurst/vol-of-vol/
   correlation regimes, several meshes, and disjoint confirmation seeds;
4. an unrelated high-precision reference for the deepest small-noise points;
5. a teacher-bank/OOD study showing whether neural amortization reduces total work;
6. independent person/hardware reproduction and an external novelty/proof audit.

Until those items close, the honest target is a strong doctoral working paper or a
finite-grid computational submission, not a top mathematical-finance journal.
