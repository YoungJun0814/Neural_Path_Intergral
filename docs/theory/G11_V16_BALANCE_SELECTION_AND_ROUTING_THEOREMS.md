# G11 V16 exact balance, independent selection, and fail-closed routing

Date: 2026-08-12

## 1. Scope

This note proves the exactness statements used by the final V16 policy. It does
not assert that every route is faster than every comparator. Empirical dominance
is authorized only for cells labelled `dominance_candidate`; the remaining cells
are exact `correctness_fallback` routes.

Let `P` be the frozen finite-grid Gaussian Volterra reference law, let `0 <= f <= 1`
be the analytically conditioned terminal payoff, and let each proposal `Q_j` have
a known density `q_j` with respect to the same coordinate measure.

## 2. T16-12A: exact balance mixture and second-moment inheritance

For positive weights `alpha_j` summing to one, define

```text
Q = sum_j alpha_j Q_j,        q = sum_j alpha_j q_j.
```

If final samples are drawn from `Q`, ordinary importance sampling gives

```text
E_Q[f(X) p(X)/q(X)] = E_P[f(X)].
```

If `M2(Q_j) = integral f^2 p^2/q_j`, then `q >= alpha_j q_j` implies

```text
M2(Q) <= M2(Q_j)/alpha_j
```

for every component family `j`. If `Q_j >= delta_j P`, then

```text
Q >= (sum_j alpha_j delta_j) P,
dP/dQ <= 1/(sum_j alpha_j delta_j).
```

Thus combining independently trained residual and tempered proposals cannot
introduce bias or an unknown normalizing constant. The complete balance density,
including every natural and shifted component, must be used.

## 3. T16-12B: independent proposal selection

Let a finite frozen bank `{Q_1,...,Q_J}` be trained without final-evaluation data.
Let `G` be any validation law dominating `P` and every candidate. For an independent
validation sample `X_i ~ G`, define

```text
Y_ij = f(X_i)^2 (dP/dQ_j)(X_i) (dP/dG)(X_i).
```

Then

```text
E_G[Y_ij] = M2(Q_j).
```

If `Q_j >= delta_j P` and `G >= delta_G P`, then

```text
0 <= Y_ij <= 1/(delta_j delta_G).
```

Consequently a finite-bank simultaneous empirical-Bernstein certificate is
available. With probability at least `1-gamma`, all candidate risks lie within

```text
r_j = sqrt(2 S_j^2 log(4J/gamma)/n)
      + 7 B_j log(4J/gamma)/(3(n-1)),
B_j = 1/(delta_j delta_G)
```

of their empirical means. If the empirical-risk minimizer `s` is selected, then

```text
M2(Q_s) <= min_j M2(Q_j) + r_s + max_j r_j.
```

The factor `4J/gamma` allocates the error probability across both deviation
directions and all `J` frozen candidates when invoking the one-sided
Maurer--Pontil empirical-Bernstein form. A one-sided UCB alone has a different
claim. The bound requires independent validation draws and is often numerically
vacuous for rare events; it is not an observed speed guarantee.

Most importantly, if selection is completed before a fresh final sample is drawn,
then conditional on all training and validation data,

```text
E[ f(X) dP/dQ_s(X) | training, validation ] = E_P[f].
```

Selection therefore cannot bias final ordinary IS. In the rough/high-eta study the
certificate was numerically far too wide to justify selection; V16 records that
negative result and does not use the selector in its final route.

## 4. T16-13: fail-closed structural routing

Let `R(theta)` be a deterministic map from declared task/model parameters—Hurst
index, vol-of-vol, correlation, and strike ratio—to a frozen exact proposal family.
Assume `R` never reads reference estimates, evaluation contributions, or final
seeds. If every routed family uses its exact density and positive defensive mass,
then for every fixed `theta` the routed ordinary-IS estimator is unbiased and has
the routed defensive likelihood bound.

This theorem is an exactness result, not a uniform performance theorem. A route may
carry one of two claim roles:

- `dominance_candidate`: a fresh confirmation must beat every accuracy-qualified
  comparator under training-inclusive work;
- `correctness_fallback`: only accuracy, robust uncertainty, likelihood
  normalization, and likelihood-bound claims are authorized.

The final V5 policy uses a confirmed V14/tempered balance route for rough `K=2`,
confirmed tempered or CM routes in the other passing cells, and exact V14 residual
fallbacks in repeatedly falsified joint-extreme regimes.

## 5. Non-claims

These theorems do not prove bounded relative error, uniform dominance, neural
operator generalization, exact continuous-time simulation, or a quantitative
continuous-target relative-bias/complexity rate.
