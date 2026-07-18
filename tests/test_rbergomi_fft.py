"""Pathwise replay tests for the FFT BLP rBergomi engine."""

from __future__ import annotations

import torch

from src.path_integral.controllers import TimePiecewiseTwoDriverControl
from src.path_integral.rbergomi_fft import (
    AdjacentRBergomiFFTInnovations,
    RBergomiFFTInnovations,
    blp_fft_kernel,
    historical_volterra_convolution,
    simulate_coupled_rbergomi_adjacent_fft,
    simulate_rbergomi_fft,
)
from src.physics_engine import RBergomiSimulator


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.2, xi=0.04, rho=-0.7, device="cpu")


def _control() -> TimePiecewiseTwoDriverControl:
    return TimePiecewiseTwoDriverControl(((-0.3, -1.0), (-0.1, -0.7)), maturity=0.25)


def test_linear_memory_coefficients_equal_reference_blp_matrix() -> None:
    simulator = _simulator()
    steps = 32
    step_dt = 0.25 / steps
    fast = blp_fft_kernel(simulator, n_steps=steps, step_dt=step_dt)
    local, weights, variance, drift = simulator._hybrid_coefficients(
        steps, step_dt, H=simulator.H, dtype=torch.float64
    )
    expected_weights = torch.zeros_like(weights)
    for row in range(steps):
        if row > 0:
            expected_weights[row, :row] = torch.flip(fast.historical_kernel[1 : row + 1], dims=(0,))
    assert torch.allclose(fast.local_cholesky, local, atol=2e-15, rtol=0.0)
    assert torch.allclose(expected_weights, weights, atol=2e-15, rtol=0.0)
    assert torch.allclose(fast.volterra_variance, variance, atol=2e-15, rtol=0.0)
    assert fast.local_drift_integral == drift


def test_fft_normalization_matches_direct_linear_convolution() -> None:
    simulator = _simulator()
    torch.manual_seed(9301)
    increments = torch.randn(19, 65, dtype=torch.float64)
    kernel = blp_fft_kernel(simulator, n_steps=65, step_dt=0.25 / 65).historical_kernel
    direct = historical_volterra_convolution(increments, kernel, method="direct")
    fast = historical_volterra_convolution(increments, kernel, method="fft")
    assert torch.allclose(fast, direct, atol=3e-14, rtol=3e-14)


def test_single_grid_fft_replays_direct_paths_and_likelihood() -> None:
    simulator = _simulator()
    paths, steps = 128, 32
    torch.manual_seed(9302)
    innovations = RBergomiFFTInnovations(
        local_standard_normal=torch.randn(paths, steps, 2, dtype=torch.float64),
        price_standard_normal=torch.randn(paths, steps, dtype=torch.float64),
    )
    arguments = dict(
        S0=100.0,
        T=0.25,
        dt=0.25 / steps,
        num_paths=paths,
        control_fn=_control(),
        innovations=innovations,
    )
    direct = simulate_rbergomi_fft(simulator, method="direct", **arguments)
    fast = simulate_rbergomi_fft(simulator, method="fft", **arguments)
    for left, right in (
        (fast.spot, direct.spot),
        (fast.variance, direct.variance),
        (fast.volterra, direct.volterra),
        (fast.running_minimum, direct.running_minimum),
        (fast.log_likelihood, direct.log_likelihood),
        (fast.target_local_integrals, direct.target_local_integrals),
    ):
        assert left is not None and right is not None
        assert torch.allclose(left, right, atol=2e-13, rtol=2e-13)


def test_adjacent_fft_replays_both_direct_blp_marginals() -> None:
    simulator = _simulator()
    paths, fine_steps = 96, 32
    coarse_steps = fine_steps // 2
    torch.manual_seed(9303)
    innovations = AdjacentRBergomiFFTInnovations(
        first_cell_standard_normal=torch.randn(paths, coarse_steps, 3, dtype=torch.float64),
        second_cell_standard_normal=torch.randn(paths, coarse_steps, 2, dtype=torch.float64),
        price_standard_normal=torch.randn(paths, fine_steps, dtype=torch.float64),
    )
    arguments = dict(
        S0=100.0,
        T=0.25,
        fine_steps=fine_steps,
        num_paths=paths,
        control_fn=_control(),
        innovations=innovations,
    )
    direct = simulate_coupled_rbergomi_adjacent_fft(simulator, method="direct", **arguments)
    fast = simulate_coupled_rbergomi_adjacent_fft(simulator, method="fft", **arguments)
    for left, right in (
        (fast.fine.spot, direct.fine.spot),
        (fast.fine.variance, direct.fine.variance),
        (fast.fine.volterra, direct.fine.volterra),
        (fast.coarse.spot, direct.coarse.spot),
        (fast.coarse.variance, direct.coarse.variance),
        (fast.coarse.volterra, direct.coarse.volterra),
        (fast.log_likelihood, direct.log_likelihood),
        (fast.target_coarse_local_integrals, direct.target_coarse_local_integrals),
    ):
        assert left is not None and right is not None
        assert torch.allclose(left, right, atol=3e-13, rtol=3e-13)


def test_fft_results_are_invariant_to_batch_chunking_for_fixed_innovations() -> None:
    simulator = _simulator()
    paths, steps = 37, 32
    torch.manual_seed(9304)
    innovations = RBergomiFFTInnovations(
        local_standard_normal=torch.randn(paths, steps, 2, dtype=torch.float64),
        price_standard_normal=torch.randn(paths, steps, dtype=torch.float64),
    )
    whole = simulate_rbergomi_fft(
        simulator,
        S0=100.0,
        T=0.25,
        dt=0.25 / steps,
        num_paths=paths,
        control_fn=_control(),
        innovations=innovations,
    )
    chunks = []
    for start, stop in ((0, 11), (11, 29), (29, paths)):
        chunk_innovations = RBergomiFFTInnovations(
            local_standard_normal=innovations.local_standard_normal[start:stop],
            price_standard_normal=innovations.price_standard_normal[start:stop],
        )
        chunks.append(
            simulate_rbergomi_fft(
                simulator,
                S0=100.0,
                T=0.25,
                dt=0.25 / steps,
                num_paths=stop - start,
                control_fn=_control(),
                innovations=chunk_innovations,
            )
        )
    assert torch.allclose(
        torch.cat([chunk.spot for chunk in chunks]), whole.spot, atol=2e-13, rtol=0.0
    )
    assert torch.allclose(
        torch.cat([chunk.variance for chunk in chunks]),
        whole.variance,
        atol=2e-13,
        rtol=0.0,
    )
