# G11 V8 R2 canonical log-spot remediation audit

Date: 2026-08-02  
Scope: frozen V6 P5 independent reference execution

## Outcome

The first authorized local final execution was stopped and invalidated after a
deterministic numerical failure. It produced 6,846 complete immutable shards,
but none may be mixed with a remediated execution because their source commit is
part of the authorization and every shard receipt.

The failure is preserved in
`results/g11_v8_p5_reference_final_execution_failure_v1_2026-08-02.json`.

## Reproduced failure

The failure was reproduced with the frozen identity and seed family:

- cell: `h0.20-discrete_lower_barrier-p1e-05`
- method: `dcs_reference`
- final shard index: `403`
- requested paths: `4,096`
- original source: `5968afb7f72ef622d286abd358ac26292f1c2650`

The rBergomi simulator evolved a finite log-price, but its materialized
`exp(log_spot)` reached IEEE-754 zero. The previous validation treated this
representation limit as an invalid mathematical path. Retrying the same shard
without changing the implementation necessarily reproduced the exception.

## Mathematical correction

The finite-grid scheme already updates

\[
  X_{n+1}=X_n-\tfrac12 V_n\Delta t+\sqrt{V_n}\,\Delta W_n^S,
  \qquad S_n=\exp(X_n).
\]

The correction makes finite `log_spot = X` the canonical simulated price state.
`spot = exp(log_spot)` remains an IEEE-754 projection for consumers that need
ordinary prices; it is not inverted to recover the canonical state.

For every positive threshold `K`, the event equivalence

\[
  S_n \le K \quad\Longleftrightarrow\quad X_n \le \log K
\]

holds exactly. Terminal, discrete-barrier, and hit-plus-occupation decisions are
therefore evaluated directly from `log_spot`. DCS affine reconstruction also
uses the stored log path, avoiding the nonlinear distortion that flooring or
clipping `spot` would introduce.

No path is discarded, no price floor is applied, and no non-finite contribution
is converted to zero. The likelihood, mixture weights, allocation, thresholds,
and seed namespaces are unchanged.

## Technical propagation

The canonical state is propagated through:

- the direct and FFT single-grid rBergomi simulators;
- exact adjacent fine/coarse coupling;
- mixture concatenation and causal replay adapters;
- generic DCS-MGI, control-span smoothing, and rank-two smoothing;
- raw and DCS reference event evaluation.

When both representations are supplied to DCS, the code verifies
`spot == exp(log_spot)` exactly, including legitimate zero or infinity in the
projection. Variance remains subject to the existing strict positive finite
policy; it is not floored.

## Verification

- The exact failed shard now returns 4,096 finite contributions and 4,096 finite
  likelihood-normalization values under the same identity and seeds.
- A regression test fixes that shard identity permanently.
- A unit test covers a finite log-price of `-1000`, whose float64 exponential is
  zero, and verifies exact terminal and barrier decisions.
- Focused rBergomi/reference tests: `82 passed`.
- Full repository suite: `818 passed`.
- Mypy on the 11 affected reference/rBergomi modules: no issues.
- Ruff on all changed Python files: passed.

## Remaining authorization state

This remediation does not authorize a result claim. A clean committed source,
fresh benchmark, fresh environment/source-bound authorization, and a complete
41,487-shard execution are still required. Only the final aggregate and its
independent audit may set `reference_complete=true` or permit downstream
performance claims.
