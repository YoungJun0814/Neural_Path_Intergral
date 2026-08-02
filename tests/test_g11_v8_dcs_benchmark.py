from __future__ import annotations

import math

import pytest
import torch

from src.path_integral.controllers.markov import TimePiecewiseTwoDriverControl
from src.path_integral.dcs_benchmark import evaluate_paired_dcs_benchmark
from src.path_integral.path_functionals import TerminalThresholdTask


def _controls() -> tuple[TimePiecewiseTwoDriverControl, ...]:
    return (
        TimePiecewiseTwoDriverControl(((0.0, 0.0),), maturity=1.0),
        TimePiecewiseTwoDriverControl(((1.0, -0.5),), maturity=1.0),
    )


def _evaluate():
    return evaluate_paired_dcs_benchmark(
        task=TerminalThresholdTask(90.0),
        task_id="terminal-90",
        spot=100.0,
        maturity=1.0,
        steps=16,
        hurst=0.12,
        eta=1.0,
        xi=0.04,
        rho=-0.7,
        controls=_controls(),
        weights=torch.tensor((0.25, 0.75), dtype=torch.float64),
        sample_count=4096,
        path_seed=101,
        label_seed=102,
    )


def test_paired_dcs_is_exact_reproducible_and_costed() -> None:
    first = _evaluate()
    replay = _evaluate()
    assert torch.equal(first.raw_contribution, replay.raw_contribution)
    assert torch.equal(first.dcs_contribution, replay.dcs_contribution)
    assert torch.equal(first.likelihood_normalization, replay.likelihood_normalization)
    assert first.component_counts == replay.component_counts
    assert first.maximum_path_reconstruction_error <= 1e-10
    assert first.maximum_component_density_error <= 1e-10
    assert first.maximum_mixture_density_error <= 1e-10
    assert first.maximum_full_likelihood_error <= 1e-10
    assert first.raw_cost.final_samples == first.dcs_cost.final_samples == 4096
    assert first.raw_cost.likelihood_evaluations == 8192
    assert first.dcs_cost.cdf_calls == 4096
    assert first.dcs_cost.algorithmic_work_units > first.raw_cost.algorithmic_work_units
    assert first.raw_cost.wall_seconds > 0.0
    assert first.dcs_cost.wall_seconds > 0.0


def test_paired_dcs_agrees_with_raw_and_reduces_variance() -> None:
    batch = _evaluate()
    difference = batch.raw_contribution - batch.dcs_contribution
    difference_se = math.sqrt(float(torch.var(difference, unbiased=True)) / difference.numel())
    assert abs(float(torch.mean(difference))) <= 5.0 * difference_se
    normalization = batch.likelihood_normalization
    normalization_se = math.sqrt(
        float(torch.var(normalization, unbiased=True)) / normalization.numel()
    )
    assert abs(float(torch.mean(normalization)) - 1.0) <= 5.0 * normalization_se
    assert float(torch.var(batch.dcs_contribution, unbiased=True)) < float(
        torch.var(batch.raw_contribution, unbiased=True)
    )


def test_paired_dcs_rejects_seed_and_weight_contract_violations() -> None:
    kwargs = dict(
        task=TerminalThresholdTask(90.0),
        task_id="terminal-90",
        spot=100.0,
        maturity=1.0,
        steps=8,
        hurst=0.12,
        eta=1.0,
        xi=0.04,
        rho=-0.7,
        controls=_controls(),
        sample_count=16,
        path_seed=1,
        label_seed=1,
    )
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_paired_dcs_benchmark(
            weights=torch.tensor((0.25, 0.75), dtype=torch.float64), **kwargs
        )
    kwargs["label_seed"] = 2
    with pytest.raises(ValueError, match="normalized"):
        evaluate_paired_dcs_benchmark(
            weights=torch.tensor((0.25, 0.70), dtype=torch.float64), **kwargs
        )
