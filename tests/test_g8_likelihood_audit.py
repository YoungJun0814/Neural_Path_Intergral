import math

import pytest
import torch

from experiments.g8_volterra_bridge_branching import (
    _constant_control_energy,
    _log_likelihood_moment_z,
)


def test_log_likelihood_moment_audit_matches_deterministic_girsanov_law() -> None:
    energy = 6.0
    residual = math.sqrt(energy) * torch.tensor([-1.0, 1.0] * 5_000, dtype=torch.float64)
    log_likelihood = residual - 0.5 * energy

    mean, variance, mean_z, variance_z, maximum_z = _log_likelihood_moment_z(
        log_likelihood,
        torch.full((log_likelihood.numel(),), energy, dtype=torch.float64),
    )

    assert mean == pytest.approx(-0.5 * energy, abs=1e-14)
    assert variance == pytest.approx(energy * 10_000 / 9_999)
    assert mean_z <= 1e-12
    assert variance_z < 0.01
    assert maximum_z == variance_z


def test_log_likelihood_audit_rejects_path_dependent_energy() -> None:
    with pytest.raises(ValueError, match="deterministic control energy"):
        _constant_control_energy(torch.tensor([1.0, 1.1], dtype=torch.float64))
