"""End-to-end rBergomi sampler bridge tests for raw and DCS MLMC."""

from __future__ import annotations

import pytest
import torch

import src.path_integral.rbergomi_mlmc_sampler as sampler_module
from src.path_integral import (
    FixedFinestGridTarget,
    MLMCHierarchy,
    RBergomiMLMCSampler,
    RBergomiMLMCSamplerConfig,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    execute_mlmc,
    prepare_mlmc,
    rao_blackwell_pair_diagnostics,
)
from src.physics_engine import RBergomiSimulator


def _sampler(method: str, engine: str = "fft") -> RBergomiMLMCSampler:
    simulator = RBergomiSimulator(H=0.15, eta=0.8, xi=0.04, rho=-0.55, device="cpu")
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


def test_raw_single_shift_baseline_can_explicitly_drop_defensive_component() -> None:
    simulator = RBergomiSimulator(H=0.15, eta=0.8, xi=0.04, rho=-0.55, device="cpu")
    shifted = TimePiecewiseTwoDriverControl(((-0.25, -0.8),), maturity=0.25)
    sampler = RBergomiMLMCSampler(
        simulator,
        (shifted,),
        torch.ones(1, dtype=torch.float64),
        TerminalThresholdTask(100.0),
        RBergomiMLMCSamplerConfig(
            spot=100.0,
            maturity=0.25,
            coarsest_steps=4,
            method="raw",
            require_natural_component=False,
        ),
    )
    values = sampler(
        0,
        "pilot",
        64,
        {"proposal": 123_456, "labels": 789_012},
    ).values
    assert values.shape == (64,)
    assert torch.isfinite(values).all()

    with pytest.raises(ValueError, match="defensive natural"):
        RBergomiMLMCSamplerConfig(
            spot=100.0,
            maturity=0.25,
            coarsest_steps=4,
            method="dcs_mgi",
            require_natural_component=False,
        )


@pytest.mark.parametrize("level", [0, 1])
def test_raw_fast_path_and_dcs_path_match_same_path_pair(level: int) -> None:
    seeds = {"proposal": 6_010_101 + level, "labels": 6_010_201 + level}
    raw = _sampler("raw_defensive")(
        level,
        "pilot",
        512,
        seeds,
    ).values
    dcs = _sampler("dcs_mgi")(
        level,
        "pilot",
        512,
        seeds,
    ).values
    pair = _sampler("dcs_mgi").sample_raw_dcs_pair(
        level,
        "pilot",
        512,
        seeds,
    )
    assert torch.allclose(raw, pair.raw_values, atol=3e-16, rtol=0.0)
    assert torch.equal(dcs, pair.dcs_values)
    diagnostics = rao_blackwell_pair_diagnostics(
        pair.raw_values,
        pair.dcs_values,
    )
    assert diagnostics.count == 512
    assert abs(diagnostics.variance_decomposition_error) <= 1e-13


def test_raw_fast_path_does_not_invoke_dcs_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("raw path called the DCS evaluator")

    monkeypatch.setattr(sampler_module, "evaluate_rbergomi_dcs_level", fail)
    monkeypatch.setattr(sampler_module, "evaluate_rbergomi_dcs_adjacent", fail)
    for level in (0, 1):
        seeds = {
            "proposal": 6_020_101 + level,
            "labels": 6_020_201 + level,
        }
        values = _sampler("raw_defensive")(
            level,
            "pilot",
            64,
            seeds,
        ).values
        assert values.shape == (64,)
        with pytest.raises(AssertionError, match="called the DCS"):
            _sampler("dcs_mgi")(
                level,
                "pilot",
                64,
                seeds,
            )
