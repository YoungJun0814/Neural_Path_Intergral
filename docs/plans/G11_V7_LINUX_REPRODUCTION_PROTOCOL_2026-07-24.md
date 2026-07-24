# G11 V7 Linux reproduction protocol

Date: 2026-07-24  
Status: implementation before Linux outcomes  
Role: disjoint-seed cross-platform reproduction

## Locked principle

The Linux run executes the exact Windows estimator source commit `80a5bc9` from a
clean, self-contained clone.  Reproduction configs live outside that source tree.
They differ from the Windows confirmation only in:

1. four protocol IDs, which create disjoint deterministic seed namespaces; and
2. the accuracy bootstrap seed.

Every cell, proposal, threshold, cluster count, planning rule, final floor, work cap,
accuracy rule, and scientific gate is unchanged.  The freeze code parses both config
sets and fails on any other difference.

## Container

- Python 3.10 slim Linux base;
- CPU PyTorch 2.9.1;
- NumPy 1.26.4;
- SciPy 1.15.3;
- PyYAML 6.0.3; and
- psutil 5.9.0.

The immutable Docker image ID is recorded in the pre-outcome freeze receipt.

## Reproduction sequence

1. Build and hash the committed reproduction image.
2. Generate external frozen configs and receipt from the passing Windows package.
3. Clone and checkout execution commit `80a5bc9`; require a clean tree.
4. Run the 64-cluster paired probe in Linux.
5. Run the independent paired audit.
6. Run all 2,304 fixed-estimator records with durable external checkpoints.
7. Run independent fixed audit and resource supplement.
8. Run joint mechanism and 72-claim accuracy analyses.
9. Run the independent aggregate package audit inside Linux.
10. On the host, compare Windows and Linux artifacts and actual seed sets.

## Cross-environment gates

- both aggregate confirmation audits pass;
- same execution source, estimand, reference, and proposal source;
- Linux and Windows OS identifiers differ;
- every pair among Windows/Linux probe/fixed seed sets has empty intersection;
- both mechanism and accuracy co-gates pass; and
- combined-SE effect-difference z is at most 3.0 for paired-probe variance,
  production variance, final work, and training-inclusive work.

Non-training work and wall-time z differences are reported but are diagnostics,
because OS scheduling and serialization overhead can legitimately differ.

## Interpretation

A pass establishes implementation and effect stability across the two declared
software environments.  It does not provide statistical independence from shared
modeling choices, references, or proposal construction, and it is not a replacement
for external code or proof review.
