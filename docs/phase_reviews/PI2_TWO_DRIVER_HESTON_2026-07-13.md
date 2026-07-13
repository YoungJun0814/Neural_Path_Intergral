# PI2 Intermediate Review: Two-Driver Heston

Date: 2026-07-13<br>
Status: simulator and measure-change gate complete; Heston oracle still pending

## Outcome

The Heston simulator now supports drift control of both independent Brownian
coordinates without changing the existing one-driver API. This closes the
proposal-class restriction that previously forced \(u^2=0\), while retaining
the old implementation as a tested baseline.

PI2 is not complete. The soft conditional desirability, its state gradients,
the resulting analytic oracle direction, and the PI/PICE/\(J_2\) objective
benchmark remain open.

## Implemented dynamics

The independent target coordinates satisfy

$$
dB_t^{i,\mathbb M}=dB_t^{i,\mathbb Q}+u_t^i dt,
\qquad i\in\{1,2\}.
$$

The controlled log-spot drift gains \(\sqrt v\,u^1\). The controlled variance
drift gains

$$
\xi\sqrt v\left(\rho u^1+\sqrt{1-\rho^2}u^2\right).
$$

The likelihood is

$$
\log\frac{d\mathbb M}{d\mathbb Q}
=-\sum_k\left(u_k^1\Delta B_k^{1,\mathbb Q}
              +u_k^2\Delta B_k^{2,\mathbb Q}\right)
 -\frac12\sum_k\left((u_k^1)^2+(u_k^2)^2\right)h.
$$

Controls are evaluated before their matching Brownian increments are sampled.
The simulator records Brownian coordinates only when requested, so large
frozen-control evaluations do not pay the training-memory cost.

## Verification gates

Six dedicated tests cover:

1. exact `target = proposal + control * dt` coordinate reconstruction;
2. agreement with the generic multidriver likelihood and energy;
3. pathwise reconstruction of controlled Heston states using target Brownian
   increments and the uncontrolled target recursion;
4. state, likelihood, running integral, and first Brownian-coordinate agreement
   with the legacy simulator when \(u^2=0\);
5. two-dimensional likelihood normalization under nonzero constant controls;
6. fixed-control likelihood-weighted Heston call estimation against natural
   Monte Carlo.

The first test uses float64. Stepwise likelihood accumulation and a single
tensor reduction differ only in reduction order, so the test uses a strict
\(10^{-15}\) numerical tolerance rather than bitwise equality. Coordinate
reconstruction itself is bitwise equal because it repeats the stored operation.

## Error audit

No unresolved measure-change error was found in this implementation unit:

- \(u^2\) enters variance drift but not spot drift;
- correlation is applied after defining the independent likelihood basis;
- the likelihood contains both independent coordinates and no duplicated
  correlated-coordinate density;
- the raw full-truncation Euler variance state is retained for recursion while
  the public path is nonnegative;
- the terminal grid is exactly \(T\) using `ceil(T/dt)` and `T/n_steps`;
- likelihood and energy use float64 accumulation;
- Brownian history recording is opt-in;
- the old one-driver simulator remains unchanged and is a regression oracle.

## Remaining PI2 work

1. implement the soft Heston conditional desirability
   \(h_\tau(t,x,v)=E[g_\tau(S_T)\mid x,v]\);
2. cross-check \(\partial_x\log h\) and \(\partial_v\log h\) using independent
   finite-difference resolutions;
3. construct the two oracle coordinates
   \(u_1^\star=\sqrt v\,\partial_x\log h
   +\rho\xi\sqrt v\,\partial_v\log h\) and
   \(u_2^\star=\xi\sqrt{1-\rho^2}\sqrt v\,\partial_v\log h\);
4. quantify near-maturity and low-variance numerical instability;
5. run fixed-control unbiasedness and continuous-reference comparisons across
   a preregistered time-step grid;
6. benchmark CEM, one-driver, full two-driver, PI, PICE, and \(J_2\) controls.
