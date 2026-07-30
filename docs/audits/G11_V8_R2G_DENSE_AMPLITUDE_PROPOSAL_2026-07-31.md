# G11 V8 R2g dense-amplitude proposal decision

Date: 2026-07-31  
Gate: two remaining raw cross-check proposals  
Decision: fail; barrier passes and terminal remains open

The dense-amplitude protocol evaluated 24 seven-expert defensive mixtures with
eight independent validation replicates and 384 unique seeds. All expert schedules
are positive scalar multiples of their source CEM profile, with a 5% natural
component, so ordinary likelihood validity and the rank-one structural invariant
are preserved.

The \(H=.05\), barrier \(10^{-5}\) raw requirement now passes with:

- projected cap ratio 0.193;
- maximum-to-median variance ratio 1.41;
- maximum single-contribution share 0.057; and
- at least 1,075 nonzero raw contributions per replicate.

The \(H=.05\), terminal \(10^{-5}\) raw requirement remains open. Its best
cap-feasible candidate has cap ratio 0.319 but contribution share 0.109. A second
candidate reaches share 0.103 but cap ratio 0.541. Neither strict threshold was
relaxed.

Diagnostic replay of the burned validation stream showed that the largest terminal
contributions occur mainly around the middle-to-high amplitude components. The
next protocol may densify and reweight that interval under a fresh namespace, but
must retain the same cap, variance-stability, contribution-concentration,
coverage, and normalization gates.
