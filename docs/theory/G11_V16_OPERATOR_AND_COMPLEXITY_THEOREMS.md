# G11 V16 operator certification and scoped complexity theorems

Date: 2026-08-11

## 1. Scope and non-claims

This note closes two precise statements that were previously grouped too broadly
under T15-8/T15-9.  It does not prove a uniform neural approximation theorem, a
quantitative BLP relative-bias rate, bounded relative error, or polynomial
complexity for the continuous target.

Let `F_theta` denote the finite-rank conditional action, `p_N` the declared BLP
Gaussian reference density, and let a frozen predictor output an initial coefficient
vector.  Final estimation always uses ordinary importance sampling and the exact
balance-mixture likelihood.

## 2. T16-10: predictor-independent exactness and local correction bound

### Theorem 10A (exactness with correction or fallback)

Suppose a deterministic correction maps a frozen prediction either to a certified
proposal `R_theta` or to a fallback proposal.  In both cases form

```text
Q_theta = delta P_N + (1-delta) R_theta,  0 < delta < 1,
```

where the balance density includes every component.  If evaluation samples are
independent of training and correction, then

```text
E_Q[g(X) p_N(X)/q_theta(X)] = E_P[g(X)]
```

and `p_N/q_theta <= 1/delta` pointwise.  The result is independent of prediction
quality.  The natural-only fallback is the special case `Q_theta=P_N`.

Proof: `q_theta >= delta p_N`, so domination and the likelihood bound hold.
Change of measure gives the expectation identity.  Freezing before evaluation
prevents data-dependent reuse from changing the conditional law.  No neural
assumption enters the proof.

### Theorem 10B (local Newton correction)

Let `x*` be a stationary point and assume on a convex neighborhood containing every
iterate that

```text
m I <= Hess F_theta(x),
||Hess F_theta(x)-Hess F_theta(y)|| <= L ||x-y||,
```

with `m>0`.  For full Newton steps `x_{k+1}=x_k-H_k^{-1} grad F(x_k)`, Taylor's
formula gives

```text
e_{k+1} <= (L/(2m)) e_k^2,   e_k=||x_k-x*||.
```

Hence, writing `a=L/(2m)` and assuming `a e_0<1`,

```text
e_k <= a^{-1}(a e_0)^(2^k).
```

If the Hessian is also bounded above by `M I` on that neighborhood, then
`||grad F(x_k)|| <= M e_k`.  This gives an explicit double-logarithmic iteration
bound for any requested gradient tolerance.  Strong convexity also yields the
a-posteriori quality bound

```text
0 <= F(x)-F(x*) <= ||grad F(x)||^2/(2m).
```

The assumptions are local-basin assumptions, not facts inferred from one pointwise
Hessian.  The implemented strong-Wolfe L-BFGS corrector is empirically confirmed;
Theorem 10B describes the mathematically controlled Newton variant and does not
silently turn the L-BFGS timings into a theorem.

## 3. T16-11: log-scale work for the joint-grid target

Let `Y_epsilon` be the unbiased V16 contribution for the joint schedule
`N(epsilon)` and let

```text
R_epsilon = Var(Y_epsilon)/P_{epsilon,N(epsilon)}^2.
```

T16-9 implies logarithmic efficiency, equivalently

```text
limsup epsilon log(1+R_epsilon) <= 0.
```

For a requested stochastic relative RMSE `tau>0`, ordinary averaging of

```text
M_epsilon = ceil(R_epsilon/tau^2) vee 1
```

samples has relative variance at most `tau^2`, and

```text
limsup epsilon log M_epsilon <= 0.
```

If per-sample work `W_epsilon`, deterministic setup work `S_epsilon`, and any frozen
operator-training charge are subexponential, meaning

```text
epsilon log(W_epsilon + S_epsilon + 1) -> 0,
```

then total work `S_epsilon + M_epsilon W_epsilon` is also subexponential.  This is
the strongest end-to-end conclusion supplied by logarithmic efficiency alone.

The proof is direct from the variance of an IID average and
`log(a+b)<=log 2+max(log a,log b)`.

## 4. Why full T15-8 remains conditional

T16-11 controls stochastic relative error for the **joint finite-grid target**.  A
relative-RMSE statement for the continuous probability additionally needs a rate
such as

```text
|P_{epsilon,N}-P_epsilon| / P_epsilon <= B(epsilon,N)
```

with a computable schedule making `B` smaller than the bias budget.  T16-9 proves
the same exponential scale but not this relative prefactor bound.  Therefore it
would be an error to label the full continuous-target complexity problem solved.

## 5. Executable boundaries

`src/path_integral/complexity_certificates.py` implements only the algebraic bounds
proved above.  It deliberately requires the caller to supply the constants and
returns no break-even query count when amortized per-query work is not lower.  The
replicated operator experiment separately measures attraction-basin and correction
cost behavior; it is evidence, not a substitute for the local assumptions.
