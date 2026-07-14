# G3 VFO Matched-Ablation Development Review

Date: 2026-07-14<br>
Protocol: `g3-vfo-staged-development`<br>
Result: `results/g3_vfo_development_matched_2026-07-14.json`<br>
Decision: **G3 memory gate not passed; use the one permitted task/representation pivot**

## 1. Implemented architecture and contracts

The first task-specific VFO prototype now contains:

- an instantaneous two-driver branch;
- a fixed positive SOE bank fitted to `sqrt(2H) r^(H-1/2)`;
- a structural two-driver branch using the SOE state;
- a GRU residual-memory branch;
- additive `tanh` gates, with the residual gate exactly zero at initialization;
- B0/B1/B2/B3 freeze and unfreeze contracts;
- accumulated branch energy and residual takeover diagnostics;
- simulator-managed memory reset and post-control target-increment observation;
- soft PI, target-coordinate feedback PICE, and hard `J2` score replay.

The SOE bank is used only as a controller feature. The target law remains the verified
BLP simulator. The fitted eight-term bank on the development interval had relative L2
kernel error about `1.40e-4` and maximum relative error about `7.34e-4`.

## 2. Correctness results

Focused VFO tests verify:

- positive SOE rates and weights;
- causal state update;
- zero initial memory gates and null initial proposal;
- stage-specific parameter freezing;
- suffix perturbations cannot alter earlier controls;
- simulator resets memory between batches;
- SOE memory consumes target, not proposal, first-driver increments;
- target-path replay reproduces behavior controls and density;
- PI, PICE, and `J2` gradients are finite;
- behavior snapshots copy `state_dict` into a fresh controller rather than copying a live
  autograd memory graph.

## 3. Confounded pilot and correction

The first full pilot appeared to show structural work improvement of about `3.19x` over
an early instantaneous snapshot. That comparison was invalid: the structural controller
had received additional PI/PICE updates.

The result is retained as a development artifact but excluded from the memory claim:

`results/g3_vfo_development_2026-07-14.json`

The corrected experiment branches all architectures from the same B0 checkpoint. Each
then receives the same total number of updates and the same PI/PICE/`J2` seed and
objective sequence. Only the active architecture differs.

## 4. Matched result

The task was an rBergomi terminal event `S_T<=80` with `H=0.1`, `rho=-0.7`, maturity
`0.25`. Three development validation seeds used 20,000 paths each.

| Method | Single-path variance | Online work proxy | Work-VRF vs instant |
|---|---:|---:|---:|
| Natural MC | 3.197e-2 | 1.399e-6 | 0.359 |
| Instant-only matched | 3.823e-3 | 5.025e-7 | 1.000 |
| Structural-only matched | 3.797e-3 | 5.473e-7 | 0.918 |
| Full VFO | 3.904e-3 | 6.243e-7 | 0.805 |

Instant-only control is approximately `2.78x` more online-work efficient than natural
MC on this pilot. In contrast, structural and residual memory do not materially reduce
variance and their inference overhead makes total online work worse than instant-only.

The maximum residual energy fraction was only `7.50e-5`. This rules out residual
takeover but also shows that the residual branch did not learn a material contribution.

## 5. Interpretation

The negative result is plausible: for a terminal event at one fixed parameter set, the
current `(S_t,V_t,Y_t)` state and a flexible instantaneous controller may already carry
most useful proposal information. Additional memory training cannot be credited with an
effect when a matched instant-only controller reaches the same variance.

This result does not falsify the controlled BLP law or two-driver neural importance
sampling. It specifically fails the stronger hypothesis that SOE/residual memory adds
work-normalized value on this terminal task.

## 6. Plan-v3 pivot

G3 permits one task or representation change before the memory-superiority claim is
abandoned. The pivot is fixed as:

1. add explicit running barrier/drawdown progress to the common event state;
2. test a down-crossing or drawdown event whose target depends on the path;
3. keep training updates, objective sequence, seeds, model widths, and timing protocol
   matched across instant-only, structural-only, and full VFO;
4. retain the BLP target law and exact likelihood unchanged;
5. require structural/full improvement in paired work, not variance alone.

If the matched path-dependent pilot also fails, the VFO memory-superiority claim is
removed and the research pivots to a two-driver task-conditioned instantaneous/Markov
controller or another candidate whose failure mechanism is empirically justified.
