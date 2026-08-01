# G11 V8 B1 Strong-Baseline Implementation Audit

Date: 2026-08-02  
Status: B1 implementation gate passed; no performance or submission claim  
Namespace: `v8-b1-baseline-implementation-smoke-v1`

## Decision

The seven requested external baseline algorithms, covering eight declared method
roles, are now connected to the same 128-step finite-grid rBergomi law and the same
fail-closed train--pilot--freeze--final lifecycle. The 15 admissible representative
method--cell executions passed the executor audit and a separately implemented
artifact audit.

This result authorizes D1 falsification development only. It does not authorize a
claim that any baseline or the proposed method is accurate, efficient, superior, or
submission-ready. The smoke allocation deliberately used at most 32 inferential
units, so its numerical estimates are not performance evidence.

## Bound evidence

| Artifact | SHA-256 |
|---|---|
| `configs/g11_v8/b1_baseline_implementation_v1.yaml` | `74f1122d0f582f80715f85bd4799ed93cc0e490630fdba0b629c308db86866b7` |
| `results/g11_v8_b1_baseline_implementation_v1_2026-08-02.json` | `d5704016d6006374f5b3fe5f348556832600c1f791b2f16eef90c2d9b5bb47e9` |
| `results/g11_v8_b1_baseline_implementation_audit_v1_2026-08-02.json` | `34d52b6d44592744fae2ec5c9f5d74a657769b65fe1e73b27603cde8fb0b7962` |

The result records source commit
`71ccb773f42d0741f72b5d886b44824718eb7691` and discloses the dirty implementation
worktree. This is acceptable for a non-performance implementation smoke. D1 must
run from a clean B1 commit in its new reserved namespace.

## Common finite-grid probability law

For `N` time steps, the BLP hybrid/FFT discretization consumes two independent
standard normals for each local Volterra cell and one independent standard normal
for the orthogonal price driver. Thus every full-path baseline operates on

\[
Z\sim N(0,I_{3N}).
\]

The mapping from `Z` to variance and price paths is shared by every method. Events
are evaluated from the canonical finite `log_spot`, rather than exponentiated price,
so a mathematically positive price that underflows to IEEE-754 zero is not rejected
or misclassified. No method clips a likelihood, discards an extreme finite path, or
uses a self-normalized estimate.

## Method-by-method mathematical audit

### Crude and antithetic Monte Carlo

Crude MC samples the target standard Gaussian and uses the unweighted hard-event
indicator. Antithetic MC creates adjacent `(Z,-Z)` pairs and treats the pair mean,
not either member, as one independent inferential unit. Both paths are charged.
Odd sample counts fail before simulation.

### Conditional rBergomi Monte Carlo

This comparator is deliberately restricted to terminal events. Conditional on the
two local BLP normals in every cell, the terminal log-price equals a known
conditional mean plus an independent scalar Gaussian with variance

\[
(1-\rho^2)\,\Delta t\sum_{i=0}^{N-1}V_i.
\]

The conditional event probability is therefore evaluated by one analytic normal
CDF. Applying this terminal formula to a barrier or occupation event would be
incorrect; the implementation rejects that case. CDF calls and all local-path
simulations are charged.

### Pure and defensive CEM

The CEM implementation adapts a task-specific mean in the exact P4 proposal family
while retaining identity covariance. Elite fraction, exponential smoothing, and a
maximum mean norm are fixed before evaluation. The final pure proposal is
`N(mu,I)`, with exact

\[
\log(q/p)(z)=z^\top\mu-\tfrac12\|\mu\|^2.
\]

Defensive CEM mixes the target component with the learned shift and evaluates the
balance-mixture density by `logsumexp`. Elite samples affect only the frozen
proposal; final hard-event contributions use the ordinary mean under an independent
seed. Training paths, updates, CPU time, wall time, peak memory, and deterministic
work units are included.

The current CEM comparator is a mean-shift CEM because that is the exact family
predeclared by the P4 ledger. It must not be described as full-covariance CEM.
Before D1, the manuscript/comparator description must use this exact name and the
training-budget frontier must disclose the restriction. A future covariance-adapted
variant would require a new density schema and oracle; silently changing covariance
inside the present family is prohibited.

### Smoothing RQMC

Each Owen-scrambled Sobol net is an independent replicate. Points within a net are
never treated as iid error bars. The price normal is decomposed into a positive flat
unit direction and an orthogonal residual. The parallel coordinate is removed before
simulation and exactly integrated with the finite-grid scalar threshold identity.
Inverse-normal endpoints use adjacent representable interior values. Point count,
scramble count, transforms, path simulations, and CDF calls are charged.

This comparator may share a mathematical rank-one smoothing identity with DCS, but
it samples the target law by RQMC and has no learned defensive mixture. It is labelled
a numerical-smoothing baseline, not a proposed DCS extension.

### Large-deviation subspace IS

The trainer minimizes finite-grid Gaussian action plus a feasibility penalty. A
candidate is accepted only after the exact hard event succeeds. The optimizer's ray
is bracketed and bisected against that exact event; if all optimized rays fail, a
prespecified negative price-driver direction provides a fail-closed fallback. For an
occupation event, a smooth occupation count is used only to train; exact discrete
hit-plus-occupation feasibility remains mandatory.

The final sampler is a defensive mixture of the target and the event-feasible shift.
It therefore has full support, an exact Gaussian-mixture likelihood, and
`dP/dQ <= 1/defensive_weight`. Optimization, feasibility evaluations, failed
restarts, and measured resources are charged. This is a one-action subspace tilt,
not a claim that the global Freidlin--Wentzell minimizer has been proved.

### Exact-likelihood coupling-flow IS

The flow is a globally invertible triangular affine coupling. Its log-scale is
bounded by `max_log_scale*tanh(.)`; the second block has an analytic inverse and
Jacobian. A ridge-regularized elite regression fits location, conditional shift, and
residual scale. Final contributions use the exact inverse-Jacobian likelihood and an
ordinary mean. Screening and regression work are charged.

The one-layer flow has full Gaussian support but no defensive target component. It
can consequently have severe finite-sample weight variance despite exactness. It is
explicitly `baseline_only`, with `dcs_extension_eligible=false`.

## Independent audit coverage

The auditor independently recomputed every proposal SHA and checked:

- exact terminal and barrier method rosters;
- 45 globally disjoint training, pilot, and final seeds;
- proposal family, exact-likelihood, frozen, and no-self-normalization flags;
- proposal-to-plan-to-estimate hash binding;
- iid, antithetic-pair, and RQMC-randomization units;
- integer raw-sample allocation;
- training, planning, likelihood, CDF, final, wall-time, and memory charges;
- a natural component in defensive CEM and large-deviation proposals;
- flow bounded-scale and baseline-only declarations; and
- refusal of performance, P8, and submission authorization.

Five mutation tests additionally require rejection of likelihood corruption, seed
collision, omitted conditional cost, and premature performance authorization.

## Interpretation of the smoke numbers

The smoke uses a loose target estimator variance and a final cap of 32 units. Zero
crude events and unstable small-sample IS estimates are therefore expected and do
not falsify unbiasedness or demonstrate accuracy. Conversely, the visually closer
RQMC smoke values do not demonstrate superiority. D1 must use a frozen logarithmic
budget ladder, independent development clusters, reference uncertainty, likelihood
normalization, and total-work diagnostics before any method comparison is made.

## Remaining B1-to-D1 obligations

1. Commit the complete B1 implementation and rerun D1 from that clean commit.
2. Freeze D1 Stage A cells, budget ladder, cluster count, seed namespaces, and
   censoring rules before opening the D1 stream.
3. Include fixed raw DCS and proposed DCS through their existing audited evaluator;
   B1 covers external comparators only.
4. Add likelihood-normalization and tail diagnostics to D1, not to the probability
   estimate itself.
5. Preserve all unsuccessful budget points and charge their training work.
6. Do not authorize P8 unless the independent D1 auditor passes and the separate T1
   novelty/theorem gate is resolved.
