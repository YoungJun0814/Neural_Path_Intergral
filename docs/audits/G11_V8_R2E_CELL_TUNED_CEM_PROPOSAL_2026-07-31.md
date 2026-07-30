# G11 V8 R2e cell-tuned CEM proposal decision

Date: 2026-07-31  
Gate: independently validated, cell-tuned reference proposals  
Decision: pass for proposal-manifest construction only

## Result

The V2 development protocol trained three constrained four-segment CEM profiles for
each of the two method-cell bottlenecks. Every fitted profile:

- used an exact ordinary \(dP/dQ\) likelihood in the weighted CEM update;
- converged in two iterations under the frozen stopping rule;
- retained a strictly negative price-driver shift in every time segment; and
- generated defensive mixtures using only the natural component and positive
  scalar multiples of one profile.

The latter property is essential: it preserves the common rank-one control span
required by the analytic DCS marginalization.

Six independent validation replicates of 8,192 paths were used for every candidate.
The selected development proposals achieved:

| Cell | Required method | Projected paths | Cap ratio |
|---|---|---:|---:|
| \(H=.20\), barrier, \(10^{-5}\) | raw | 1,101,327 | 0.1313 |
| \(H=.05\), barrier, \(10^{-5}\) | DCS | 1,228,410 | 0.1464 |

Both are below the predeclared 0.75 selection margin, not merely below the absolute
cap. All 222 training and validation seeds are distinct, the likelihood
normalization tests pass, and the result was produced from clean commit
`45db83ec72eaac0b21427cb4a3dd7d87f05f087d`.

## V1 numerical failure

The earlier V1 validation encountered a nonfinite path under an over-aggressive
amplitude family. It produced no result file and no performance outcome was
inspected. Its training and validation namespaces were burned, and the error,
source commit, configuration hash, and stderr were preserved before V2 was frozen.
V2 used fresh namespaces and bounded amplitude families.

## Claim boundary

Candidate selection used the V2 development validation outcomes, so these ratios
are not an unbiased qualification result and cannot support a paper performance
claim. The audit only authorizes construction of a hash-bound proposal manifest.
The manifest must be used with fresh pilot seeds; final allocation must be frozen
from those pilots; final DCS and raw estimates must use independent streams; and
the later paper comparison must use an untouched qualification namespace.
