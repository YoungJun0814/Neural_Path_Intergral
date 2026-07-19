"""End-to-end rBergomi sampler bridge tests for raw and DCS MLMC."""

from __future__ import annotations

import pytest
import torch

from src.path_integral import (
    FixedFinestGridTarget,
    MLMCHierarchy,
    RBergomiMLMCSampler,
    RBergomiMLMCSamplerConfig,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    execute_mlmc,
    prepare_mlmc,
)
from src.physics_engine import RBergomiSimulator


def _sampler(method: str, engine: str = "fft") -> RBergomiMLMCSampler:
    simulator = RBergomiSimulator(
        H=0.15, eta=0.8, xi=0.04, rho=-0.55, device="cpu"
    )
    controls = (
        TimePiecewiseTwoDriverControl(((0.0, 0.0),), maturity=0.25),
        TimePiecewiseTwoDriverControl(((-0.25, -0.8),), maturity=0.25),
    )
    return RBergomiMLMCSampler(
        simulator,
        controls,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
        TerminalThresholdTask(100.0),
        RBergomiMLMCSamplerConfig(
            spot=100.0,
            maturity=0.25,
            coarsest_steps=4,
            method=method,
            engine=engine,
        ),
    )


@pytest.mark.parametrize("method", ["raw_defensive", "dcs_mgi"])
def test_rbergomi_bridge_runs_complete_mlmc_with_disjoint_streams(method: str) -> None:
    sampler = _sampler(method)
    hierarchy = MLMCHierarchy(4, 2, FixedFinestGridTarget(2))
    prepared = prepare_mlmc(
        hierarchy,
        sampler,
        protocol=f"g11-rbergomi-{method}",
        regime="unit",
        task="terminal",
        sampling_variance_target=0.05,
        pilot_samples=128,
        minimum_final_samples=32,
        chunk_size=64,
    )
    result = execute_mlmc(prepared, sampler)
    assert result.complete
    assert result.estimate is not None
    assert result.empirical_sampling_variance is not None
    assert torch.isfinite(torch.tensor(result.estimate))
    keys = [record.key for record in prepared.ledger.records]
    assert all(key.role == "pilot" for key in keys)
    assert {key.stream for key in keys} == {"proposal", "labels"}


def test_reference_engine_obeys_same_exact_adapter_contract() -> None:
    sampler = _sampler("dcs_mgi", engine="reference")
    values = sampler(
        1,
        "pilot",
        128,
        {"proposal": 123_456, "labels": 789_012},
    ).values
    assert values.shape == (128,)
    assert torch.isfinite(values).all()


def test_sampler_rejects_float32_evidence_and_missing_streams() -> None:
    with pytest.raises(ValueError, match="float64"):
        RBergomiMLMCSamplerConfig(
            spot=100.0,
            maturity=0.25,
            coarsest_steps=4,
            method="dcs_mgi",
            dtype=torch.float32,
        )
    sampler = _sampler("dcs_mgi")
    with pytest.raises(ValueError, match="proposal and label"):
        sampler(0, "pilot", 16, {"proposal": 1})
