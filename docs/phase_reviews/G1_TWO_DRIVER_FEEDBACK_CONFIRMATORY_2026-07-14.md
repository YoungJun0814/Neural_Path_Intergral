# G1 Two-Driver Heston Feedback Confirmatory Review

Date: 2026-07-14<br>
Protocol: `g1-two-driver-heston-feedback-confirmatory-v1`<br>
Protocol SHA-256: `6f58c7759446c4eb61649176f08646b4ee14667d77c802d32bf5d94c2cb468a0`<br>
Result: `results/g1_heston_feedback_confirmatory_v1_2026-07-14.json`<br>
Checkpoint SHA-256: `db37aebec96d914ee322e66192b2b63c0e189f3955b6b2f7081fa41dc7e21480`<br>
Decision: **G1 correctness/reference gate passed; freeze Heston development and proceed to G2**

## 1. Scope of the decision

This gate establishes that a trainable controller can recover the independent-basis
two-driver Heston oracle and can be trained sequentially with soft PI, feedback PICE,
and hard-event `J2` without violating the declared measure convention. It is a
correctness and objective-recovery result.

It does **not** establish that VFO works in rough volatility, that the method is already
training-cost competitive, or that a journal-level contribution is complete. The
historical sealed Heston roots `11101–11120` were not accessed.

## 2. Implemented G1 path

The new G1 module contains:

1. bounded Markov feedback `u(t,S,v) in R^2` in the independent Brownian basis;
2. deterministic soft-Heston oracle-grid construction and supervised distillation;
3. soft path-integral free-energy training;
4. off-policy feedback PICE on reconstructed target Brownian paths;
5. hard-indicator `log J2` refinement using only the likelihood-ratio score gradient;
6. one-driver ablation, CEM comparison, time-step refinement, and paired timing;
7. versioned checkpoint save/load with a stable state hash.

The stages remain sequential. Soft PI and PICE are proposal-shaping/projection
objectives; the final hard-event estimator is ordinary, non-self-normalized importance
sampling.

## 3. Mathematical audit

### 3.1 Measure convention

The simulator and all objectives use:

$$
\Delta W_i^P=\Delta W_i^Q+u_i\Delta t,
$$

$$
\log\frac{dP}{dQ}
=-\sum_i u_i^\top\Delta W_i^Q
-\frac12\sum_i\|u_i\|^2\Delta t.
$$

Candidate PICE density is evaluated on a canonical target path:

$$
\log\frac{dQ_\theta}{dP}
=\sum_i u_i^\top\Delta W_i^P
-\frac12\sum_i\|u_i\|^2\Delta t.
$$

The candidate control is replayed causally on the state prefix. Behavior proposal
increments are not reused as candidate residuals.

### 3.2 Soft PI

For `g=exp(-Phi)`, the implemented objective is:

$$
E_Q\left[\Phi(X_T)+\frac12\int_0^T\|u_t\|^2dt\right].
$$

This is the Gibbs variational stochastic-control objective. Its likelihood-weighted
soft estimate is reported separately and the objective is not called a hard-event
second moment.

### 3.3 Feedback PICE

Behavior trajectories are weighted during training by:

$$
g(X)\frac{dP}{dQ_b}.
$$

Self-normalization is used only to estimate the reverse-KL projection gradient. It is
never used in the final probability estimate.

### 3.4 Hard `J2`

For:

$$
J_2(\theta)=E_{Q_\theta}[1_A L_\theta^2],
$$

the code uses:

$$
\nabla_\theta J_2
=-E_{Q_\theta}
\left[1_A L_\theta^2\nabla_\theta\log q_\theta\right].
$$

Hard-path simulation is graph-free. A second causal replay on detached states supplies
the score. A two-dimensional Gaussian closed-form regression test independently checks
the sign and magnitude of this vector gradient.

## 4. Development audit and corrections before freeze

Two development-gate failures were investigated before the confirmatory protocol was
created.

### 4.1 Oracle derivative diagnostic

At `v=0.01` and remaining time `0.2`, the default Richardson variance step `8e-4`
produced an absolute analytic-versus-finite-difference discrepancy `3.95e-4`. The
analytic derivative was about `36.798`, and the mismatch was a coarse-stencil error.
Reducing the declared relative step so the stencil reached `1e-4` changed the local
discrepancy to about `1.9e-10`. Across the full oracle grid the confirmatory maximum was
`6.78e-7`.

The acceptance threshold was not relaxed.

### 4.2 Continuous reference versus coarse discrete grids

The first development gate incorrectly required every coarse Euler grid to lie within
three Monte Carlo standard errors of the continuous-time Heston CDF. This conflated
discretization bias with estimator bias. The frozen rule applies the reference bias-z
gate only at the finest declared grid and uses coarser grids to quantify refinement.

The finest-grid rule and the refinement threshold were frozen before confirmatory seeds
were used.

## 5. Frozen confirmatory results

Target probability was `1e-3`; the Fourier-inverted barrier was approximately
`36.76758`. Five new validation roots `15001–15005` were used, with 50,000 paths per
seed and grid `dt in {1/64, 1/128, 1/256}`.

### 5.1 Gate metrics

| Metric | Result | Gate |
|---|---:|---:|
| Oracle validation normalized RMSE | 0.0232 | <= 0.35 |
| Oracle mean cosine | 0.99995 | >= 0.90 |
| Oracle sign agreement | 1.000 | >= 0.85 |
| Maximum oracle gradient discrepancy | 6.78e-7 | <= 1e-4 |
| Finest-grid maximum absolute bias-z | 1.97 | <= 3.0 |
| `dt=1/128` to `dt=1/256` mean change | 0.498% | <= 10% |

All frozen gates passed.

### 5.2 Sequential objective behavior

At `dt=1/256`, aggregate online work-normalized results against CEM were:

| Method | Mean single-path variance | Online work-VRF vs CEM |
|---|---:|---:|
| CEM constant | 1.033e-5 | 1.00 |
| Distilled oracle | 2.263e-6 | 2.09 |
| PI | 1.839e-6 | 2.64 |
| PICE | 1.278e-6 | 3.73 |
| hard `J2` | 1.108e-6 | 4.23 |
| hard `J2`, second driver removed | 4.951e-6 | 0.94 |

Across five paired seeds, hard `J2` had geometric mean online work-VRF `4.25` with a
95% t-interval approximately `[3.61, 4.99]`. The one-driver ablation interval included
one: geometric mean `0.945`, interval `[0.806, 1.108]`. This supports the numerical
relevance of the second independent Brownian control coordinate in this Heston task.

### 5.3 Training-inclusive limitation

Feedback training took about `196.1 s`; CEM training took about `0.75 s`. At the
50,000-path CEM precision, online timing and variance imply an approximate repeated-query
break-even of `M*=286`, far above the Plan-v3 practical target `M*<=25`.

Therefore this Heston controller is **not** claimed to be training-inclusive practical.
Heston remains an oracle/correctness benchmark. The practical amortization hypothesis
must be tested later with task-conditioned VFO over many rough-volatility queries.

## 6. Verification

The focused pre-confirmatory suite passed 24 tests. It covers:

- exact two-driver coordinate and likelihood reconstruction;
- target-path state reconstruction;
- oracle Fourier/Richardson agreement;
- PI/PICE/`J2` finite gradients;
- candidate target-density/residual-likelihood identity;
- independent two-dimensional Gaussian `J2` score gradient;
- no-hard-event failure behavior;
- checkpoint state-hash round trip.

The checkpoint reload reproduced state hash
`db37aebec96d914ee322e66192b2b63c0e189f3955b6b2f7081fa41dc7e21480`.

## 7. Decision and next gate

G1 is closed as a passed correctness/reference gate. Per Plan v3, Heston architecture
development stops here. The next implementation unit is G2:

1. independent-basis controlled rBergomi hybrid/BLP simulator;
2. exact target/proposal Brownian reconstruction;
3. recent singular-cell and historical memory drift shifts;
4. likelihood normalization and fixed-control unbiasedness;
5. `rho<=0` discounted-spot and refinement diagnostics.

No VFO performance training is allowed until all G2 law tests pass.
