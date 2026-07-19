"""Independent SciPy specialization checks for the published smoothing baseline."""

from __future__ import annotations

import torch

from src.path_integral import (
    DownsideExcursionTask,
    TimePiecewiseTwoDriverControl,
    evaluate_rbergomi_dcs_adjacent,
    simulate_coupled_rbergomi_mixture,
)
from src.path_integral.numerical_smoothing_reference import (
    scipy_scaled_normal_cdf,
    scipy_scaled_normal_cdf_difference,
)
from src.path_integral.stable_gaussian import (
    scaled_normal_cdf,
    scaled_normal_cdf_difference,
)
from src.physics_engine import RBergomiSimulator


def test_independent_scipy_reference_matches_extreme_tail_production_code() -> None:
    log_scale = torch.tensor([0.0, 2.0, -3.0, 1.0], dtype=torch.float64)
    threshold = torch.tensor([-40.0, -12.0, 11.0, 40.0], dtype=torch.float64)
    actual = scaled_normal_cdf(log_scale, threshold)
    reference = scipy_scaled_normal_cdf(log_scale, threshold)
    assert torch.allclose(actual, reference, atol=1e-14, rtol=2e-13)

    fine = torch.tensor([-11.0, 12.0, 2.0, -2.0], dtype=torch.float64)
    coarse = torch.tensor([-12.0, 11.0, -1.0, 3.0], dtype=torch.float64)
    difference = scaled_normal_cdf_difference(log_scale, fine, coarse)
    reference_difference = scipy_scaled_normal_cdf_difference(
        log_scale, fine, coarse
    )
    assert torch.allclose(difference, reference_difference, atol=1e-14, rtol=2e-13)


def test_rbergomi_adjacent_correction_matches_independent_scipy_specialization() -> None:
    simulator = RBergomiSimulator(
        H=0.12, eta=1.1, xi=0.04, rho=-0.6, device="cpu"
    )
    controls = (
        TimePiecewiseTwoDriverControl(((0.0, 0.0),), maturity=0.25),
        TimePiecewiseTwoDriverControl(((-0.3, -0.9),), maturity=0.25),
    )
    torch.manual_seed(88_001)
    sample = simulate_coupled_rbergomi_mixture(
        simulator,
        controls,
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        fine_steps=32,
        num_paths=4096,
        label_generator=torch.Generator().manual_seed(88_002),
        engine="fft",
    )
    task = DownsideExcursionTask(92.0, 97.0, 1.0 / 64.0, 3.0, 0.02)
    evaluation = evaluate_rbergomi_dcs_adjacent(
        sample, task=task, rho=simulator.rho
    )
    reference = scipy_scaled_normal_cdf_difference(
        evaluation.fine.density.residual_log_likelihood,
        evaluation.fine.threshold,
        evaluation.coarse.threshold,
    )
    assert torch.allclose(
        evaluation.marginalized_correction,
        reference,
        atol=2e-14,
        rtol=2e-13,
    )
