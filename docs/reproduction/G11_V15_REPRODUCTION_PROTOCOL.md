# G11 V15 independent reproduction protocol

Date: 2026-08-11

Status: package prepared; independent person/hardware reproduction not yet executed.

## 1. Scope

This protocol reproduces the finite-grid numerical claims for **Exact Conditional
Cameron–Martin Transport for Gaussian Volterra Rare Events**.  It does not certify
the open continuous-time or asymptotic theorems T15-5 through T15-8.

The reproducer must use a fresh clone, record the commit hash and environment, and
must not change thresholds, gates, sample counts, or seeds after seeing results.

## 2. Environment receipt

From the repository root:

```powershell
git status --short
git rev-parse HEAD
python --version
python -m pip install -e ".[dev]"
python -c "import torch, numpy, scipy; print(torch.__version__, numpy.__version__, scipy.__version__)"
```

The working tree must be clean before execution.  CPU float64 is the reference
contract.  GPU results are supplementary until a CPU/GPU cross-audit is supplied.

## 3. Correctness suite

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy src/path_integral src/models/volterra_transport_operator.py src/training/volterra_transport_operator.py
```

At minimum, the focused V15 suite must pass:

```powershell
python -m pytest tests/test_volterra_conditional_payoffs.py tests/test_v15_continuous_contract.py tests/test_cameron_martin_basis.py tests/test_cameron_martin_modes.py tests/test_finite_rank_gaussian_transport.py tests/test_v15_theorem_oracles.py tests/test_rbergomi_cm_mesh.py tests/test_volterra_transport_operator.py tests/test_rbergomi_cm_transport.py tests/test_v15_result_audit.py -q
```

## 4. Frozen numerical runs

```powershell
python -m experiments.g11_v15_development --config configs/g11_v15/development_v1.yaml
python -m experiments.g11_v15_qualification --config configs/g11_v15/qualification_v1.yaml
python -m experiments.g11_v15_mesh_development --config configs/g11_v15/mesh_development_v2.yaml
python -m experiments.g11_v15_small_noise_development --config configs/g11_v15/small_noise_development_v2.yaml
python -m experiments.g11_v15_audit --result results/g11_v15_qualification_v1_2026-08-11.json
```

For a genuinely disjoint-seed reproduction, use
`configs/g11_v15/external_reproduction_v1.yaml`.  The original qualification result
must remain immutable.

## 5. Required receipts

Return all of the following:

- commit hash and `git status --short` before and after execution;
- OS, CPU model, RAM, Python, PyTorch, NumPy, and SciPy versions;
- stdout/stderr and wall-clock time for every command;
- generated JSON files and their SHA-256 hashes;
- any failed or interrupted run, without deleting it;
- a statement of whether the reproducer is independent of model design and code
  authorship.

## 6. Pass/fail interpretation

The numerical package passes only if:

1. result integrity and config hashes pass;
2. all five primary comparators are present in every cell;
3. V15 estimates satisfy the frozen accuracy gate;
4. the defensive likelihood bound and proposal hash checks pass;
5. the frozen training-inclusive work ratio passes in every cell.

Even if all five pass, the top-journal gate remains false while theorem gate G5 is
false.  An external run cannot replace a missing proof.

## 7. Known limitations of this package

- The current qualification has three parameter cells and probabilities from roughly
  `4e-3` to `6e-2`; it is not a broad extreme-tail matrix.
- The small-noise diagnostic reaches roughly `1.3e-6`, but its precise reference is
  an independent stream from the same frozen V15 proposal, not an unrelated method.
- The mesh bias budget passes in v2, but the observed correction rate is noisy and
  does not prove T15-6 or T15-7.
- No independent hardware/person reproduction has yet been received.
