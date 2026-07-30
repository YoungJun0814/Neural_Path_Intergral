# G11 V8 R2f all-failed-cells CEM decision

Date: 2026-07-31  
Gate: stable proposals for all eleven failed method-cell requirements  
Decision: fail; nine pass and two require redesign

The V3 protocol expanded constrained CEM development from two cells to all eight
unique cells behind the original eleven allocation failures. It used 24 independent
fits, 72 defensive rank-one candidates, eight validation replicates per candidate,
and 1,176 mutually distinct seeds.

Nine requirements passed all predeclared gates:

- projected final samples below half of the absolute cap;
- maximum-to-median replicate variance ratio at most 20;
- no single path contributing more than 10% of a replicate's absolute sum;
- at least 256 nonzero raw contributions per replicate; and
- ordinary likelihood normalization within four standard errors.

Two raw cross-check requirements failed:

| Cell | Best relevant symptom |
|---|---|
| \(H=.05\), terminal \(10^{-5}\) | best cap ratio 0.739 and contribution share 0.143 |
| \(H=.05\), barrier \(10^{-5}\) | cap ratio can pass, but best contribution share is 0.122 |

The concentration threshold was not relaxed after observing the result. The entire
V3 gate remains failed, no partial proposal manifest is authorized, and all final
and performance-claim gates remain closed.

The next development step is limited to the two missing raw requirements. It may
reuse the already learned profiles as development inputs, but must construct
denser amplitude mixtures under a new namespace and reapply the same cap,
replicate-stability, concentration, coverage, and normalization gates.
