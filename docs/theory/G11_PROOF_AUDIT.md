# G11 Proof Audit and Claim Ledger

Date: 2026-07-19
Audited document: `docs/theory/G11_THEOREMS.md`

## 1. Audit result

**Decision: pass for finite-dimensional implementation; conditional for any rate or
continuous-time claim.**

The density identities, Rao--Blackwell relation, defensive bounds, and finite-grid
telescoping are mathematically consistent under deterministic Gaussian shifts. The
rank-one threshold correction and its second-moment upper bounds are also valid under
the explicit finite-threshold and threshold-rate assumptions.

No general rough-Volterra rate or continuous-time theorem has been proved. Those
claims remain blocked.

## 2. Independent derivation checks

### 2.1 Target/proposal orientation

The proposal component is `Q_j=N(m_j,I)`, so

`dQ_j/dP = exp(m_j^T x-||m_j||^2/2)`.

The estimator uses `dP/dQ`, hence the full balance likelihood is the reciprocal of
the mixture density ratio. This agrees with the repository's convention that target
increments equal proposal innovations plus `u h`.

Decision: pass.

### 2.2 Marginal density

After an orthogonal coordinate split, integrating the shifted Gaussian density over
the span coordinate removes `b_j` exactly. The residual mixture retains `c_j` and the
same mixture weights. No normalization constant is missing.

Decision: pass.

### 2.3 Proposal conditional expectation

The proposal conditional law `Z|R` is generally a residual-dependent mixture. The
proof does not treat it as standard normal. Instead, `q` cancels against `p/q` inside
the conditional integral and leaves the target conditional integral multiplied by the
residual likelihood.

Decision: pass. This is the most important identity to test independently in code.

### 2.4 Defensive bound

The target component contributes `delta*p` to the full mixture and `delta*p_R` to the
residual marginal. Both full and marginal likelihoods are bounded by `1/delta`.

Decision: pass.

### 2.5 Telescoping

The condition `C_l C_l^T=I` ensures the normalized coarse vector has the correct
target law. Each level may use a different proposal because each corrected mean is
individually changed back to its target law. One fine likelihood is required for the
coupled correction formula and its variance advantage.

Decision: pass.

### 2.6 Threshold correction

For two events driven monotonically by the same scalar standard normal, conditional
linearity gives the signed CDF difference. Threshold order may switch pathwise; the
signed formula remains valid. Squaring the raw difference gives the indicator of the
interval between the thresholds.

Decision: pass.

### 2.7 Second-moment bounds

Under `Q`,

`E[(Lbar G)^2] = E_(P_R)[Lbar G^2]`,

not `E_P[Lbar^2 G^2]`. Applying `Lbar<=1/delta` and the global normal-CDF Lipschitz
constant yields the stated `h^(2r)` upper bound. The raw bound uses the conditional
probability of the interval and Cauchy--Schwarz, giving `h^r`.

Decision: pass as upper bounds only.

## 3. Assumption audit

| Assumption | Enforced mathematically | Planned runtime enforcement | Status |
|---|---|---|---|
| deterministic expert means | yes | reject path-dependent controls | required |
| positive mixture weights | yes | config/dataclass validation | required |
| natural component for bounds | yes | minimum defensive weight | required |
| orthonormal integrated span | yes | QR/SVD validation | required |
| normalized coarse map | yes | covariance/unit test | required |
| one fine likelihood per correction | yes | API ownership and negative test | required |
| common scalar coordinate | yes | coordinate mismatch gate | required |
| monotone positive slopes | yes | pathwise slope gate | required |
| finite thresholds for rate theorem | yes | extended-real branch and rate exclusion | required |
| `L2` threshold rate | assumed | M4 estimates; model proof still open | unresolved |
| weak bias rate | assumed only in T11-7 | separate continuous-target mode | unresolved |

## 4. Claim ledger

### Supported once the Gaussian oracle passes

- Exact finite-dimensional span marginalization for deterministic Gaussian mixtures.
- Exact residual balance-mixture likelihood.
- Rao--Blackwell variance non-increase under the matched proposal.
- Full and marginal defensive bounds.
- Exact finite-grid MLMC telescoping with level-specific proposals.
- Exact scalar-threshold CDF correction.

### Supported only after additional evidence

- `O(h^(2r))` applied to rBergomi: requires a proved or explicitly assumed `L2`
  threshold rate.
- Practical full-MLMC improvement: requires frozen total-work experiments.
- Training-inclusive improvement: requires the complete work ledger.
- CPU/GPU portability: requires independent reproduction.

### Unsupported and prohibited

- Exact doubling of a variance exponent without lower bounds.
- Continuous-time barrier/occupation exactness.
- Canonical rough-Bergomi complexity from the current G10 slopes.
- Novelty of common-likelihood MLIS or conditional smoothing in isolation.
- Correctness for feedback/path-dependent controls.
- Improvement from rank-two event integration.

## 5. Known theory risks

### 5.1 Threshold convergence is the central unresolved issue

The threshold is a max/min/order-statistic functional of affine path coefficients.
These maps are Lipschitz in their candidate arrays only when denominators remain
controlled away from zero. Early-time rBergomi slopes can be small, and occupation
order statistics can switch identities between levels. A rough-Volterra threshold
rate cannot be inferred from the Volterra-path strong rate alone without controlling
these effects.

Required response: prove easier terminal/barrier corollaries first and treat
hit-plus-occupation rates empirically unless a rigorous denominator/order-statistic
argument is completed.

### 5.2 Full likelihood boundedness does not imply good variance

The defensive component guarantees finite moments but can still leave a probability
so rare that the estimator is impractical. No efficiency theorem follows solely from
`L<=1/delta`.

### 5.3 Different proposals per level are unbiased but affect rates

Telescoping survives level-specific proposals, but arbitrary changes in proposal
quality can destroy a regular variance sequence. The frozen protocol must specify how
controls are restricted/interpolated across the hierarchy.

### 5.4 Random allocation

Using independent pilot samples makes the final estimator conditionally unbiased for
the level means. Reusing pilot samples after a data-dependent allocation needs a more
careful argument. V11 avoids that issue by excluding pilot samples from final means
while retaining their cost.

## 6. Code-review obligations generated by the proof

1. Store standardized target coordinates or expose an unambiguous conversion.
2. Evaluate all component log densities before `logsumexp`.
3. Project both coordinates and proposal means with the same span.
4. Compute residual energy after projection, not by subtracting nearly equal scalar
   norms when a direct residual vector is available.
5. Use one sample object and one likelihood for a level correction.
6. Treat CDF differences in signed log space.
7. Test all threshold ordering cases.
8. Record maximum full and residual density reconstruction errors.
9. Reject any evidence artifact whose source/config/seed hashes do not match.

## 7. M0 gate decision

The proof portion of M0 passes for implementation of T11-1--T11-6. T11-7 remains a
conditional corollary. Implementation may proceed to the generic Gaussian oracle only
after the novelty matrix records the 2025 occupation-time common-likelihood MLIS work
and removes common likelihood itself from the novelty claim.
