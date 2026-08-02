# G11 V9 Final Execution Closure

Date: 2026-08-02

Final status: **all work authorized by the frozen V9 gates is complete; V9 is
closed by development falsification, not by an implementation error.**

## 1. Scope and meaning of completion

V9 is a new terminal-only protocol informed by V8 but not a relabeling of it.  It
uses new references, a newly trained proposal bank, fresh seed namespaces, and a
new total-work estimand.  Completing a gated protocol means executing and auditing
every authorized phase and stopping when a prerequisite fails.  It does not mean
running qualification after development explicitly forbids it.

The fixed estimand is a 128-step finite-grid rBergomi terminal left-tail probability
over 12 Hurst/rarity cells.  Barrier, continuum-unbiasedness, uniform-superiority,
top-journal, and submission claims are excluded.

## 2. Phase outcomes

| Phase | Outcome |
|---|---|
| Claim/estimand contract | 19 semantic checks pass |
| Reference V1 | integrity passes; precision fails in 4/12 cells |
| Fresh reference V2 | 12/12 cells pass relative SE <=0.15; independent audit passes |
| Proposal bank V1 | never executed because its reference stop rule was not code-enforced |
| Reference-gated proposal bank V2 | 12 cells, 36 seeds, 585,728 paths; audit passes |
| Development V1 | atomic execution failure: 3/64 nonzero pilot units; no output |
| Development V2 | atomic execution failure: 3/256 nonzero conditional units; no output |
| Development V3 | complete 48 paired + 144 external records; independent audit passes |
| Qualification | not authorized by V3 and therefore not executed |
| Final closure audit | 12/12 checks pass |

V1/V2 execution failures were preserved and superseded only with fresh seeds.  The
minimum support boundary was never lowered.  Method-specific V3 pilots charge the
additional target-law conditional work instead of hiding it.

## 3. Theoretical and numerical correctness

The core finite-dimensional identity remains valid:

\[
\operatorname{Var}(Y_{raw})-\operatorname{Var}(Y_{DCS})
=\mathbb{E}[\operatorname{Var}(Y_{raw}\mid\mathcal G)]\ge0.
\]

V9 evaluates raw and DCS on the same exact defensive mixture and uses ordinary
means.  The DCS step integrates one event-driving Gaussian coordinate under the
actual proposal-conditional law.  Training, pilot, diagnostic, and final costs are
retained.  IID paths and independent RQMC scrambles are not conflated.

Development correctness diagnostics pass:

- maximum finite-grid reconstruction/density error: `4.5475e-13`;
- likelihood normalization pass fraction: `1.0`;
- maximum DCS combined-reference z: `2.5452`;
- maximum paired raw-minus-DCS z: `2.6916`; and
- exact likelihood plus non-self-normalized ordinary means for every comparator.

Thus the failed continuation gate is not currently attributable to seed reuse,
likelihood omission, self-normalization, cost deletion, or aggregate corruption.

## 4. Scientific result

At the primary repeated-query count `K=100`, exact conditional integration improves
the same proposal after all charged work:

| H | Geometric raw/DCS work ratio | One-sided 95% lower bound |
|---:|---:|---:|
| 0.05 | 2.0319 | 1.5726 |
| 0.12 | 1.5020 | 1.1015 |
| 0.20 | 1.5773 | 1.4212 |

This is the strongest surviving positive V9 result.

The practical-competitiveness claim fails.  Best-primary/DCS geometric work ratios
are 0.4014, 0.4226, and 0.2978 for H=0.05, 0.12, and 0.20, respectively.  Values
below one favor the comparator.  Defensive CEM is best in 37/48 positions.  The
maximum external combined-reference z is 9.0984 and 22 records are resource-
censored.  No Hurst group passes the complete selection rule.

Therefore V9 supports “DCS improves its own frozen proposal,” but not “the current
DCS model is competitive with the strongest available estimator.”

## 5. Verification

- full repository regression: **898 passed** in 123.32 seconds;
- Ruff over every Python file changed since V8 closure: passed;
- mypy over all changed source/experiment modules: passed;
- Python byte-code compilation: passed;
- strict JSON parsing: 10 V9 artifacts passed;
- `git diff --check`: passed;
- completion ledger SHA-256:
  `11ccb2c07a2a0f3e468e0bd2a4f07c24fc1d55447f12d0272f2a6d615abbcafa`;
- completion audit SHA-256:
  `c57cd344719e5aab59d59f5776252ea82cdfd6cc14846d478b3a2b32f4152dff`;
- final closure audit: all 12 checks pass from a clean worktree.

## 6. Academic interpretation and only valid next branch

V9 is doctoral-level research engineering and yields a defensible negative result
plus a clean mechanism result.  It is not a top-journal submission: practical
competitiveness, uncensored comparator accuracy, qualification, independent
hardware reproduction, external proof review, and final novelty review are absent.

The most evidence-aligned V10 hypothesis is a full-dimensional CEM proposal with
exact proposal-conditional smoothing.  It targets the actual V9 bottleneck: the
rank-one bank, not the conditional-integration identity.  V10 must be a new frozen
protocol with new seeds and cannot present V9 development records as confirmation.
