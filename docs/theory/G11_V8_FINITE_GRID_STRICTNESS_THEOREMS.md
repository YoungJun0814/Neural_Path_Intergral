# G11 V8 Finite-Grid Exactness and Strictness Theorems

Date: 2026-07-25

Status: finite-dimensional proof ledger for P2; external proof review pending

Scope: deterministic identity-covariance Gaussian-shift mixtures, exact ordinary
importance sampling, rank-one scalar-threshold events, and the declared finite-grid
rough Bergomi adapter

## 1. Setup

Let \(P=N(0,I_d)\) and

\[
Q=\sum_{j=0}^{J-1}\pi_jN(m_j,I_d),
\qquad \pi_j>0,\quad \sum_j\pi_j=1.
\]

The natural component satisfies \(m_0=0\) and \(\pi_0=\delta>0\). Let \(U\) be a
deterministic orthonormal rank-one vector, complete it by \(V\), and write

\[
X=UZ+VR,\qquad b_j=U^\mathsf Tm_j,\qquad c_j=V^\mathsf Tm_j.
\]

Under \(P\), \(Z\sim N(0,1)\), \(R\sim N(0,I_{d-1})\), and they are independent.
Define

\[
\mathcal D(z,r)=\sum_j\pi_j
\exp\left(b_jz-\frac12b_j^2\right)
\exp\left(c_j^\mathsf Tr-\frac12\|c_j\|^2\right),
\]

\[
\bar D(r)=\sum_j\pi_j
\exp\left(c_j^\mathsf Tr-\frac12\|c_j\|^2\right),
\qquad
L(z,r)=\mathcal D(z,r)^{-1},
\qquad
\bar L(r)=\bar D(r)^{-1}.
\]

The first displayed expression is understood term by term: each summand is the
product of its parallel and residual exponentials. It is not a product of two
separate sums.

Let

\[
\alpha_j(r)=
\frac{\pi_j\exp(c_j^\mathsf Tr-\|c_j\|^2/2)}
{\bar D(r)}
\]

and

\[
M_r(z)=\sum_j\alpha_j(r)\exp(b_jz-b_j^2/2).
\]

Then

\[
q(z\mid r)=\phi(z)M_r(z),
\qquad
L(z,r)=\frac{\bar L(r)}{M_r(z)}.
\]

This is the proposal conditional mixture. It is generally not a standard Gaussian.

## 2. T-V8-1: exact defensive-mixture likelihood

The exact likelihood is

\[
\frac{dP}{dQ}(x)
=
\left[
\sum_j\pi_j
\exp\left(m_j^\mathsf Tx-\frac12\|m_j\|^2\right)
\right]^{-1}.
\]

Because the natural component contributes \(\delta\),

\[
0<L(x)\leq\delta^{-1}.
\]

The same argument after integrating \(Z\) gives

\[
0<\bar L(r)\leq\delta^{-1}.
\]

Consequently, for an indicator \(F\),

\[
E_Q[L(X)F(X)]=E_P[F(X)]
\]

and the raw contribution is square-integrable. This theorem is finite-dimensional;
it does not remove time-discretization bias.

## 3. T-V8-2: proposal-conditional DCS identity

Let \(A\) be the event and

\[
Y=L(X)\mathbf 1_A(X).
\]

For every residual point with \(q_R(r)>0\),

\[
\begin{aligned}
E_Q[Y\mid R=r]
&=\int\frac{p(z,r)}{q(z,r)}\mathbf1_A(z,r)
       \frac{q(z,r)}{q_R(r)}\,dz\\
&=\frac{p_R(r)}{q_R(r)}
  \int\mathbf1_A(z,r)p(z\mid r)\,dz\\
&=\bar L(r)G(r),
\end{aligned}
\]

where \(G(r)=P_P(A\mid R=r)\). The full proposal density cancels before the target
conditional Gaussian integral appears. No step treats \(Q(Z\mid R)\) as standard
normal.

Thus

\[
E_Q[\bar L(R)G(R)]=P_P(A).
\]

Both the raw and DCS estimators use an ordinary sample mean, without
self-normalization.

## 4. T-V8-3: exact variance decomposition

Let \(Y_{\mathrm{DCS}}=E_Q[Y\mid R]\). Orthogonal projection in \(L^2(Q)\)
gives

\[
E_Q[(Y-Y_{\mathrm{DCS}})Y_{\mathrm{DCS}}]=0
\]

and

\[
\operatorname{Var}_Q(Y)
=
\operatorname{Var}_Q(Y_{\mathrm{DCS}})
+E_Q[\operatorname{Var}_Q(Y\mid R)].
\]

Therefore

\[
\operatorname{Var}_Q(Y)-\operatorname{Var}_Q(Y_{\mathrm{DCS}})
=E_Q[\operatorname{Var}_Q(Y\mid R)]\geq0.
\]

The earlier V7 mechanism document contained a typographical omission of the plus
sign in this display. The implementation and empirical audit used the correct
decomposition; P2 corrects the written formula.

## 5. T-V8-4: scalar-threshold strict improvement

Assume

\[
A=\{Z\leq a(R)\}.
\]

For fixed \(r\), write

\[
p(r)=\Phi(a(r)),
\qquad
s(r)=Q(Z\leq a(r)\mid R=r)
=\sum_j\alpha_j(r)\Phi(a(r)-b_j).
\]

The DCS conditional value is

\[
Y_{\mathrm{DCS}}(r)=\bar L(r)p(r).
\]

The conditional raw second moment is

\[
E_Q[Y^2\mid R=r]
=
\bar L(r)^2
\int_{-\infty}^{a(r)}
\frac{\phi(z)}{M_r(z)}\,dz.
\]

Define the integral as \(J(r)\). Cauchy--Schwarz on the event section gives

\[
p(r)^2
=
\left[
\int_{-\infty}^{a(r)}
\sqrt{\frac{\phi(z)}{M_r(z)}}
\sqrt{\phi(z)M_r(z)}\,dz
\right]^2
\leq J(r)s(r).
\]

Hence

\[
\begin{aligned}
\operatorname{Var}_Q(Y\mid R=r)
&=\bar L(r)^2[J(r)-p(r)^2]\\
&\geq
\bar L(r)^2p(r)^2\frac{1-s(r)}{s(r)}.
\end{aligned}
\]

All mixture weights are positive and every component is a nondegenerate Gaussian.
If \(a(r)\) is finite, then

\[
0<p(r)<1,\qquad 0<s(r)<1,
\]

so the displayed lower bound is strictly positive.

### Population strictness

If \(a(R)\) is finite on a residual set of positive \(Q_R\)-probability, then

\[
\operatorname{Var}_Q(Y)>\operatorname{Var}_Q(Y_{\mathrm{DCS}}).
\]

For the primary terminal task, a finite intercept and strictly positive terminal
slope make the threshold finite. For a discretely monitored downside barrier, fixed
\(S_0>B\), finite intercepts, and strictly positive post-initial slopes make the
maximum of finitely many candidate thresholds finite. Therefore the current
finite-grid terminal and barrier adapters satisfy the strictness condition.

If the barrier is already hit at time zero, the event is deterministic and the
threshold is \(+\infty\); strict improvement is then not claimed.

### Localized quantitative population bound

Let

\[
\mathcal B_{A,M}
=
\{|a(R)|\leq A,\ \bar D(R)\leq M\},
\qquad
P_R(\mathcal B_{A,M})\geq\eta>0,
\]

and let \(B_{\parallel}=\max_j|b_j|\). On this set,

\[
\bar L\geq M^{-1},
\qquad
\Phi(a)\geq\Phi(-A),
\]

and

\[
1-s
=\sum_j\alpha_j\Phi(b_j-a)
\geq\Phi(-A-B_{\parallel}).
\]

Since \(s\leq1\),

\[
\operatorname{Var}_Q(Y)-\operatorname{Var}_Q(Y_{\mathrm{DCS}})
\geq
\frac{\eta}{M}
\Phi(-A)^2\Phi(-A-B_{\parallel})>0.
\]

This is a valid localized constant, not a uniform rare-event asymptotic rate. P3 must
derive model-specific behavior of \(A,M,\eta\) before making a rate claim.

## 6. T-V8-5: terminal and discrete-barrier threshold maps

Suppose the finite-grid log spot has the affine representation

\[
\log S_n(z)=A_n+B_nz,
\qquad B_0=0,\quad B_n>0\ (n\geq1).
\]

### Terminal event

\[
S_N\leq K
\quad\Longleftrightarrow\quad
z\leq\frac{\log K-A_N}{B_N}.
\]

The code uses `<=`, so a path exactly on the level is included.

### Discrete-barrier event

For \(S_0>B\),

\[
\min_{1\leq n\leq N}S_n\leq B
\quad\Longleftrightarrow\quad
z\leq
\max_{1\leq n\leq N}
\frac{\log B-A_n}{B_n}.
\]

If \(S_0\leq B\), the threshold is \(+\infty\). The current implementation rejects
zero or negative post-initial slopes rather than silently dividing by zero or
reversing an inequality. These are exact discrete-monitoring statements only.

## 7. Stable executable certificate

`scalar_threshold_strictness_certificate` computes

\[
\log p,\quad \log s,\quad\log(1-s),
\quad
\log\left[
\bar L^2p^2\frac{1-s}{s}
\right]
\]

using `log_ndtr`, posterior mixture weights, and `logsumexp`. This avoids ordinary
probability underflow for finite rare thresholds such as \(-40\).

The executable oracles verify:

- equality with Bernoulli variance when \(Q=P\);
- the lower bound against independent SciPy quadrature for nontrivial mixtures;
- exact reuse of the production residual mixture density;
- stable finite-threshold certificates;
- explicit non-strict handling of infinite thresholds;
- closed-event tie handling;
- initial-barrier behavior; and
- rejection of zero post-initial slopes.

## 8. Claim boundary after P2

P2 authorizes:

- exact finite-grid balance-mixture likelihood;
- exact proposal-conditional DCS identity;
- exact variance decomposition;
- strict variance reduction for nondegenerate finite scalar thresholds; and
- exact terminal and discretely monitored barrier threshold maps.

P2 does not authorize:

- a continuously monitored barrier result;
- a nonlinear rough Bergomi weak-bias rate;
- a barrier mesh-enrichment rate;
- a matching rare-event asymptotic variance ratio;
- end-to-end MLMC complexity; or
- superiority in training-inclusive work.

Those obligations remain in P3 and later experimental phases.
