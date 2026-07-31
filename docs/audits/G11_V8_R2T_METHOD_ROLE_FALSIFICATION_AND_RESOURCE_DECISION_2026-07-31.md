# G11 V8 R2T — Method-role falsification and resource decision

Date: 2026-07-31

## Result

The method-role development protocol remained incomplete at 4 of 8
requirements.  None of the 17 new candidates passed every frozen gate.

The raw candidates generally met their new allocation targets but failed the
single-contribution concentration gate.  The best H=0.05 terminal 1e-5 DCS
candidate had projected-to-cap ratio 1.181 under the unchanged 2% primary
precision target.

The audit independently confirms:

- 17 candidates and 408 unique seeds;
- exact DCS 2% and raw 5% standard-error targets;
- exact candidate gate recomputation;
- no new selection and a 4-of-8 complete roster;
- clean source commit
  `dc2dc0c0eaf734b7594d053fdd5b7355fb42f577`; and
- all downstream decisions fail closed.

## Rank-two review

The existing rank-two implementation is mathematically admissible but does not
solve the present proposal-span limitation:

- it first requires all expert price shifts to pass the existing rank-one
  control-span check;
- the second direction is an event-integration direction, not a second
  independent proposal-control profile; and
- it is an unbiased nested estimator with additional inner randomization.

The frozen G10 rank-two development result already failed its work gates:

- geometric rank-one-over-rank-two work ratio: about 0.634;
- geometric raw-over-rank-two work ratio: about 0.916; and
- improved-regime fraction: 0.

It must therefore not be presented as a successful solution to the current
reference bottleneck.

## Decision: stop proposal p-hacking

Repeatedly relaxing concentration thresholds or redesigning candidate grids
after each seed realization would weaken the research protocol.  Proposal
development stops here.

Accurate reference generation is infrastructure, not the claimed model
speedup.  The defensible next step is a reference-only resource escalation:

1. select a complete exact proposal roster using disclosed development data;
2. retain DCS 2% and raw 5% precision roles;
3. set larger per-entry reference caps before new pilot outcomes are observed;
4. bind every proposal, cap, seed namespace, and implementation hash;
5. run one new formal pilot on the laptop;
6. reconstruct required final samples from that pilot; and
7. use external CPU only for the final immutable shards if the laptop forecast
   is infeasible.

No model-performance claim may include the reference-generation cost as a
hidden advantage.  Final-method efficiency comparisons must still include all
training and inference work specified by the claim contract.

## Authorization

- Build a reference-only resource-escalated manifest: authorized
- Tune further proposals on these development outcomes: forbidden
- Reuse any burned namespace: forbidden
- Open a formal pilot before manifest audit: forbidden
- Final execution/performance/submission claims: forbidden
