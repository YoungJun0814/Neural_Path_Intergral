# G11 V16 continuous small-noise LDP bridge

Date: 2026-08-11

Status: proof-complete corollary for the constant-forward-variance Gaussian Volterra
model and fixed terminal left tails. This closes the continuous **probability exponent**,
not the continuous proposal second moment or the mesh limit.

## 1. Parameter disambiguation

Gulisashvili (2018), Theorem 13, studies

```text
dS_t^(epsilon,a)
 = sqrt(epsilon) S_t^(epsilon,a)
   sigma(epsilon^a Y_t) d(rho B_t + sqrt(1-rho^2) W_t)
```

and proves an LDP for

```text
epsilon^(a-1/2) (log S_T^(epsilon,a) - log S_0)
```

with speed `epsilon^(-2a)`. The paper denotes this scaling exponent by `H`; it
must not be confused with the roughness parameter of a fractional kernel.

Set the paper's scaling exponent to `a=1/2`. Then

```text
epsilon^a = sqrt(epsilon),
epsilon^(a-1/2) = 1,
epsilon^(-2a) = epsilon^(-1).
```

Thus Theorem 13 directly addresses an unscaled fixed terminal log return under the
same small-noise speed used by this project.

## 2. Assumption match

For constant `xi>0`, choose

```text
sigma(x) = sqrt(xi) exp(eta x / 2).
```

This function is positive and locally Lipschitz, hence locally omega-continuous as
required by the theorem. For rough Bergomi,

```text
Y_t = sqrt(2H_r) int_0^t (t-s)^(H_r-1/2) dB_s,
0 < H_r < 1/2.
```

This is a Riemann--Liouville fractional Brownian Volterra process. Its kernel is
causal, square integrable, and satisfies the theorem's `L2` increment-modulus
condition. The correlation convention is aligned by assigning `B` to the volatility
driver and the independent Brownian motion to the orthogonal price driver.

The corollary in this document is restricted to:

- constant positive `xi`;
- `eta>0` and `rho in (-1,1)`;
- a Volterra kernel satisfying Definition 3 of the cited theorem, including the
  rBergomi Riemann--Liouville kernel with `H_r in (0,1/2)`;
- a fixed terminal threshold `k=log(K/S_0)<0`.

A general time-dependent forward variance curve is not silently included.

## 3. Wick-corrected family

The project uses

```text
V_t^epsilon
 = xi exp(eta sqrt(epsilon) Y_t
          - 0.5 eta^2 epsilon R(t)),
R(t)=Var(Y_t),
```

whereas the cited theorem applied to `sigma` gives the same expression without the
deterministic Wick factor. Let

```text
a_epsilon(t)=exp(-0.25 eta^2 epsilon R(t)).
```

The project volatility coefficient equals `a_epsilon(t) sigma(sqrt(epsilon)Y_t)`.

### T16-4a: exponential equivalence

Couple the corrected and uncorrected log returns with the same Brownian drivers. On
the localization event

```text
sup_t |sqrt(epsilon)Y_t| <= M,
```

local boundedness of `sigma`, boundedness of `R`, and
`sup_t|a_epsilon(t)-1|=O(epsilon)` give:

- an `O(epsilon^2)` finite-variation difference;
- a martingale difference with quadratic variation `O(epsilon^3)`.

The exponential martingale inequality makes the localized probability of a fixed
difference superexponential at speed `1/epsilon`. Gaussian concentration for the
continuous Volterra process gives

```text
limsup epsilon log P(sup_t |sqrt(epsilon)Y_t| > M) <= -c M^2.
```

Letting `M` tend to infinity proves exponential equivalence. Therefore the Wick
correction does not change the terminal LDP or its rate function.

## 4. Continuous terminal rate

Let `u in L2[0,T]`, `(Ku)(t)=int_0^t K(t,s)u(s)ds`, and define

```text
sigma_u(t) = sqrt(xi) exp(eta (Ku)(t)/2),
A(u) = rho int_0^T sigma_u(t) u(t) dt,
I(u) = int_0^T sigma_u(t)^2 dt.
```

The cited terminal rate at log return `x` is

```text
I_T(x) = inf_u [
    0.5 ||u||_2^2
    + (x-A(u))^2 / (2(1-rho^2)I(u))
].
```

The theorem also establishes that `I_T` is continuous and non-increasing on the
negative half-line. Hence, for `k<0`, the LDP upper and lower bounds for
`(-infinity,k]` meet at `I_T(k)`:

```text
lim epsilon log P(log(S_T^epsilon/S_0) <= k) = -I_T(k).
```

Equivalently, minimizing over the independent price control at the **event** level
gives

```text
inf_u J_k(u),
J_k(u)=0.5||u||_2^2
       + ((A(u)-k)_+)^2/(2(1-rho^2)I(u)).
```

The pointwise functions in the last two displays differ: the positive part belongs to
the inequality event, while the square without a positive part belongs to the exact
endpoint `x=k`. Their infima agree by the negative-half-line monotonicity. Confusing
these two formulas is an error.

## 5. What is now proved and what remains open

Proved under the stated scope:

- the continuous Wick-corrected small-noise terminal LDP;
- speed `1/epsilon` for fixed terminal left tails;
- the continuous event-level contracted action and probability exponent.

Still open:

- logarithmic efficiency of an exact equivalent **continuous-time** proposal;
- convergence of the BLP finite-grid rate and optimizers to this continuum rate;
- a mesh-uniform second-moment/complexity theorem;
- time-dependent `xi_0(t)` outside a separately verified extension.

## 6. Primary source

Archil Gulisashvili, *Large deviation principle for Volterra type fractional
stochastic volatility models*, Theorem 13, Lemma 15, and Definition 3:
<https://arxiv.org/abs/1710.10711>.

