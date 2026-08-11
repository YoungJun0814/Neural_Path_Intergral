# G11 V16 fixed-grid small-noise theorem closure

Date: 2026-08-11

Status: proof-complete for every **fixed finite BLP grid** under the assumptions
below. This document does not claim a mesh-uniform or infinite-dimensional result.

## 1. Why V15 small-noise evidence had to be withdrawn

The declared volatility family is

```text
V_t^epsilon = xi_0(t) exp(
    eta sqrt(epsilon) Y_t
    - 0.5 eta^2 epsilon Var(Y_t)
).
```

The V15 implementation scaled the Gaussian innovations but retained the unit-noise
Wick compensator `-0.5 eta^2 Var(Y_t)`. Consequently its zero-noise variance curve
was not `xi_0`. V16 scales both terms and adds pathwise regression oracles. The V15
small-noise v1/v2 JSON files are retained for provenance but are invalid evidence.

## 2. Frozen finite-dimensional law

Fix a BLP grid with `N` cells and let `Z in R^(2N)` denote all independent local
standard-normal coordinates. For `h=sqrt(epsilon) Z`, let

```text
I_epsilon(h) > 0
A_epsilon(h)
```

be respectively the left-point integrated variance and correlated terminal return
produced by the exact finite-grid map with the Wick compensator multiplied by
`epsilon`. At zero noise,

```text
I_epsilon -> I_0,
A_epsilon -> A_0
```

locally uniformly in `h`, because the grid map is a finite composition of linear
maps, exponentials, sums and square roots of strictly positive quantities.

For `k=log(K/S_0)<0` and `|rho|<1`, define

```text
C_N(h) = ((A_0(h)-k)_+)^2 / (2(1-rho^2) I_0(h)),
J_N(h) = 0.5 |h|^2 + C_N(h).
```

The code evaluates this object in `finite_grid_small_noise.py`. A basis of rank less
than `2N` defines only a Galerkin upper bound on `inf J_N`; the complete fixed-grid
rate problem uses all `2N` columns.

## 3. T16-1: fixed-grid probability exponent

**Theorem.** Let `p_epsilon=P(S_T^epsilon <= K)` for the frozen finite-grid law.
Then

```text
lim_{epsilon down to 0} epsilon log p_epsilon = -J_N^*,
J_N^* = min_{h in R^(2N)} J_N(h) > 0.
```

**Proof.** Conditional on the local coordinates at `h/sqrt(epsilon)`, the exact
left-tail probability is

```text
g_epsilon(h/sqrt(epsilon))
 = Phi((k-A_epsilon(h)+0.5 epsilon I_epsilon(h))
       / sqrt(epsilon(1-rho^2)I_epsilon(h))).
```

The Gaussian Mills bounds, together with local uniform convergence of `A_epsilon`
and `I_epsilon`, give

```text
-epsilon log g_epsilon(h/sqrt(epsilon)) -> C_N(h)
```

locally uniformly, including the active-set boundary `A_0(h)=k`. Changing variables
from `Z` to `h=sqrt(epsilon)Z` and applying the finite-dimensional Laplace bounds
gives the result. Compact tails are harmless because `0<=g_epsilon<=1` and the
Gaussian energy is coercive. Since `J_N>=|h|^2/2`, a minimizer exists. Continuity at
zero and `k<0` imply that the closed terminal event is separated from zero, hence
`J_N^*>0`.

This is a direct finite-dimensional proof. It needs neither an identification of BLP
auxiliary coordinates with a continuum Cameron--Martin control nor a stochastic-
integral continuity theorem.

## 4. T16-2: mode-omission-safe proposal

Let `Q_epsilon^wide` be the exactly normalized proposal

```text
Z ~ Normal(0, I_(2N)/epsilon).
```

Equivalently, `h=sqrt(epsilon)Z ~ Normal(0,I_(2N))`. Its exact likelihood is

```text
L_epsilon(Z) = epsilon^(-N)
               exp(-0.5(1-epsilon)|Z|^2).
```

**Theorem.** For `Y_epsilon=g_epsilon(Z)L_epsilon(Z)`,

```text
lim epsilon log E_Qwide[Y_epsilon^2] = -2 J_N^*.
```

**Proof.** In the `h` coordinate the second moment is

```text
epsilon^(-2N) integral phi(h) g_epsilon(h/sqrt(epsilon))^2
    exp(-(1/epsilon-1)|h|^2) dh.
```

Polynomial powers of `epsilon` vanish at speed `1/epsilon`. The same compact
Laplace argument as in T16-1 has rate

```text
|h|^2 + 2 C_N(h) = 2 J_N(h).
```

Its infimum is exactly `2J_N^*`. Thus the broad component is logarithmically
efficient without finding or enumerating any dominating mode.

This theorem does **not** assert bounded relative error. Its pre-asymptotic factor can
be severe in high dimension; the component is an asymptotic safety net, not the main
practical proposal.

## 5. T16-3: inheritance by the practical mixture

Let the implemented proposal be any exact mixture

```text
Q_epsilon = alpha Q_epsilon^wide + (1-alpha) R_epsilon,
0 < alpha < 1,
```

where `R_epsilon` may contain the natural law, multimode mean shifts, curvature
changes, or a frozen neural proposal. Since `q_epsilon >= alpha q_epsilon^wide`,

```text
E_Qepsilon[(g dP/dQepsilon)^2]
 <= alpha^(-1) E_Qwide[(g dP/dQwide)^2].
```

Jensen's inequality supplies the reverse exponential bound through
`E[Y^2]>=p_epsilon^2`. Therefore the full mixture has the same `-2J_N^*`
second-moment exponent. Missed modes can hurt finite-noise performance but cannot
destroy fixed-grid logarithmic efficiency.

The natural defensive mass remains useful for the pathwise likelihood bound
`dP/dQ <= 1/delta`; the broad mass and natural mass solve different problems.

## 6. Infinite-dimensional boundary

The covariance `I/epsilon` in every coordinate is not an equivalent covariance
change on an infinite-dimensional Wiener space. The fixed-grid proof therefore
cannot pass to `N -> infinity` by assertion. A continuous-time theorem needs either:

1. an equivalence-preserving finite-rank safety family plus a proof that its ranks
   exhaust all rate-minimizing directions uniformly; or
2. a different adapted drift/subsolution construction with a continuous-time
   second-moment proof.

This is the remaining T15-5 continuum obligation. T16 closes the finite-grid theorem
and provides a falsifiable route for studying the mesh limit, but it does not close
T15-6 or T15-7.

## 7. Implementation obligations

- `epsilon=1` must remain pathwise identical to V14.
- The Wick compensator must be multiplied by `epsilon`.
- The safety covariance must be exactly `I/epsilon` in all `2N` local coordinates.
- Safety mass must be fixed and positive on the asymptotic sequence.
- The balance-mixture density, not the sampled-component density, must be used.
- Reduced-rank action minima must be labelled Galerkin upper bounds.
- No continuous-time or bounded-relative-error claim is authorized by T16.

## 8. Primary-source boundary

The finite-grid proof is self-contained. Gulisashvili's Volterra small-noise LDP
establishes that a continuous Gaussian-Volterra route is plausible under explicit
kernel and volatility assumptions; it is not used as a substitute for matching the
present discretization. Guyader--Touchette gives a general joint-LDP framework for
testing IS efficiency; the explicit broad-component calculation above supplies the
model-specific second-moment exponent required here.

