"""FFT-accelerated deterministic-control simulation for the BLP rBergomi grid law."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch

from src.path_integral.rbergomi_coupling import (
    CoupledRBergomiPaths,
    RBergomiLevelPaths,
    adjacent_local_gaussian_coefficients,
)
from src.physics_engine import (
    RBergomiSimulator,
    TwoDriverRBergomiPaths,
    strict_lognormal_variance,
)

ConvolutionMethod = Literal["fft", "direct"]
RBergomiControl = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class BLPFFTKernel:
    """Linear-convolution kernel and exact marginal variance for kappa=1 BLP."""

    local_cholesky: torch.Tensor
    historical_kernel: torch.Tensor
    volterra_variance: torch.Tensor
    local_drift_integral: float


@dataclass(frozen=True)
class RBergomiFFTInnovations:
    """Independent Gaussian innovations for a single BLP grid."""

    local_standard_normal: torch.Tensor
    price_standard_normal: torch.Tensor


@dataclass(frozen=True)
class AdjacentRBergomiFFTInnovations:
    """Independent innovations for exact adjacent fine/coarse BLP grids."""

    first_cell_standard_normal: torch.Tensor
    second_cell_standard_normal: torch.Tensor
    price_standard_normal: torch.Tensor


def blp_fft_kernel(
    simulator: RBergomiSimulator,
    *,
    n_steps: int,
    step_dt: float,
    H: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> BLPFFTKernel:
    """Construct BLP coefficients in linear rather than quadratic memory."""
    if n_steps <= 0 or not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("n_steps and step_dt must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    resolved_H = simulator.H if H is None else float(H)
    if not 0.0 < resolved_H < 0.5:
        raise ValueError("rBergomi requires H in (0, 0.5)")
    alpha = resolved_H - 0.5
    c11 = step_dt
    c12 = step_dt ** (alpha + 1.0) / (alpha + 1.0)
    c22 = step_dt ** (2.0 * alpha + 1.0) / (2.0 * alpha + 1.0)
    covariance = torch.tensor(((c11, c12), (c12, c22)), device=simulator.device, dtype=dtype)
    local_cholesky = torch.linalg.cholesky(covariance)

    lag = torch.arange(1, n_steps + 1, device=simulator.device, dtype=dtype)
    kernel = torch.zeros(n_steps, device=simulator.device, dtype=dtype)
    if n_steps > 1:
        historical_lag = lag[1:]
        kernel[1:] = (
            step_dt**alpha
            * (historical_lag ** (alpha + 1.0) - (historical_lag - 1.0) ** (alpha + 1.0))
            / (alpha + 1.0)
        )
    scale_squared = 2.0 * resolved_H
    variance = torch.zeros(n_steps + 1, device=simulator.device, dtype=dtype)
    variance[1:] = scale_squared * (c22 + step_dt * torch.cumsum(kernel.square(), dim=0))
    return BLPFFTKernel(
        local_cholesky=local_cholesky,
        historical_kernel=kernel,
        volterra_variance=variance,
        local_drift_integral=c12,
    )


def historical_volterra_convolution(
    brownian_increments: torch.Tensor,
    historical_kernel: torch.Tensor,
    *,
    method: ConvolutionMethod = "fft",
) -> torch.Tensor:
    """Return the BLP historical term by exact linear convolution."""
    if (
        brownian_increments.ndim != 2
        or historical_kernel.ndim != 1
        or brownian_increments.shape[1] != historical_kernel.shape[0]
    ):
        raise ValueError("increments and kernel must have shapes (paths, steps) and (steps,)")
    if (
        not brownian_increments.is_floating_point()
        or brownian_increments.device != historical_kernel.device
        or brownian_increments.dtype != historical_kernel.dtype
        or not torch.isfinite(brownian_increments).all()
        or not torch.isfinite(historical_kernel).all()
    ):
        raise ValueError("increments and kernel must be finite matching floating tensors")
    steps = historical_kernel.shape[0]
    if method == "direct":
        result = torch.zeros_like(brownian_increments)
        for index in range(1, steps):
            result[:, index] = torch.sum(
                brownian_increments[:, :index]
                * torch.flip(historical_kernel[1 : index + 1], dims=(0,)),
                dim=1,
            )
        return result
    if method != "fft":
        raise ValueError("method must be 'fft' or 'direct'")
    linear_length = 2 * steps - 1
    fft_length = 1 << (linear_length - 1).bit_length()
    increment_transform = torch.fft.rfft(brownian_increments, n=fft_length, dim=1)
    kernel_transform = torch.fft.rfft(historical_kernel, n=fft_length)
    return torch.fft.irfft(
        increment_transform * kernel_transform.unsqueeze(0), n=fft_length, dim=1
    )[:, :steps]


def _deterministic_schedule(
    control_fn: RBergomiControl | None,
    *,
    steps: int,
    step_dt: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if control_fn is None:
        return torch.zeros((steps, 2), device=device, dtype=dtype)
    if not bool(getattr(control_fn, "is_deterministic_time_control", False)):
        raise ValueError("fast BLP simulation requires a deterministic time-only control")
    evaluator = getattr(control_fn, "deterministic_schedule", None)
    if not callable(evaluator):
        raise ValueError("deterministic control must implement deterministic_schedule(times)")
    times = torch.arange(steps, device=device, dtype=dtype) * step_dt
    schedule = cast(torch.Tensor, evaluator(times))
    if (
        schedule.shape != (steps, 2)
        or schedule.device != device
        or schedule.dtype != dtype
        or not torch.isfinite(schedule).all()
    ):
        raise ValueError("deterministic_schedule returned an invalid tensor")
    return schedule


def _validate_single_innovations(
    innovations: RBergomiFFTInnovations,
    *,
    paths: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if innovations.local_standard_normal.shape != (paths, steps, 2):
        raise ValueError("local_standard_normal has the wrong shape")
    if innovations.price_standard_normal.shape != (paths, steps):
        raise ValueError("price_standard_normal has the wrong shape")
    for value in (innovations.local_standard_normal, innovations.price_standard_normal):
        if (
            value.device != device
            or value.dtype != dtype
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError("innovations must be finite and match simulator device/dtype")


def _assemble_level_with_h(
    *,
    S0: float,
    mu: float,
    H: float,
    xi: float,
    eta: float,
    rho: float,
    step_dt: float,
    target_driver_one: torch.Tensor,
    target_driver_two: torch.Tensor,
    target_local_integral: torch.Tensor,
    kernel: BLPFFTKernel,
    method: ConvolutionMethod,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    historical = historical_volterra_convolution(
        target_driver_one, kernel.historical_kernel, method=method
    )
    volterra_after_zero = math.sqrt(2.0 * H) * (historical + target_local_integral)
    variance_after_zero = strict_lognormal_variance(
        eta * volterra_after_zero - 0.5 * eta**2 * kernel.volterra_variance[1:],
        xi=xi,
    )
    initial_variance = torch.full(
        (target_driver_one.shape[0], 1),
        xi,
        device=target_driver_one.device,
        dtype=target_driver_one.dtype,
    )
    variance = torch.cat((initial_variance, variance_after_zero), dim=1)
    price_increment = rho * target_driver_one + math.sqrt(1.0 - rho**2) * target_driver_two
    log_increment = (mu - 0.5 * variance[:, :-1]) * step_dt + torch.sqrt(
        variance[:, :-1]
    ) * price_increment
    initial_log_spot = torch.full_like(initial_variance, math.log(S0))
    log_spot = torch.cat(
        (initial_log_spot, initial_log_spot + torch.cumsum(log_increment, dim=1)), dim=1
    )
    spot = torch.exp(log_spot)
    volterra = torch.cat((torch.zeros_like(initial_variance), volterra_after_zero), dim=1)
    running_minimum = torch.cummin(spot, dim=1).values
    if not torch.isfinite(spot).all() or not torch.isfinite(variance).all():
        raise FloatingPointError("fast rBergomi path became nonfinite")
    return spot, variance, volterra, running_minimum


def simulate_rbergomi_fft(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    dt: float,
    num_paths: int,
    mu: float = 0.0,
    control_fn: RBergomiControl | None = None,
    override_params: dict | None = None,
    innovations: RBergomiFFTInnovations | None = None,
    method: ConvolutionMethod = "fft",
    dtype: torch.dtype = torch.float64,
) -> TwoDriverRBergomiPaths:
    """Simulate the exact finite-grid BLP marginal with FFT history sums."""
    if not math.isfinite(S0) or S0 <= 0.0 or not math.isfinite(T) or T <= 0.0:
        raise ValueError("S0 and T must be finite and positive")
    if not math.isfinite(dt) or dt <= 0.0 or num_paths <= 0 or not math.isfinite(mu):
        raise ValueError("dt/num_paths/mu are invalid")
    params = simulator._resolved(override_params)
    H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
    steps = max(1, int(math.ceil(T / dt)))
    step_dt = T / steps
    kernel = blp_fft_kernel(simulator, n_steps=steps, step_dt=step_dt, H=H, dtype=dtype)
    schedule = _deterministic_schedule(
        control_fn,
        steps=steps,
        step_dt=step_dt,
        device=simulator.device,
        dtype=dtype,
    )
    if innovations is None:
        innovations = RBergomiFFTInnovations(
            local_standard_normal=torch.randn(
                num_paths, steps, 2, device=simulator.device, dtype=dtype
            ),
            price_standard_normal=torch.randn(
                num_paths, steps, device=simulator.device, dtype=dtype
            ),
        )
    _validate_single_innovations(
        innovations,
        paths=num_paths,
        steps=steps,
        device=simulator.device,
        dtype=dtype,
    )
    proposal_local_pair = innovations.local_standard_normal @ kernel.local_cholesky.T
    proposal_driver_one = proposal_local_pair[:, :, 0]
    proposal_local_integral = proposal_local_pair[:, :, 1]
    proposal_driver_two = innovations.price_standard_normal * math.sqrt(step_dt)
    proposal_brownian = torch.stack((proposal_driver_one, proposal_driver_two), dim=2)
    target_brownian = proposal_brownian + schedule.unsqueeze(0) * step_dt
    target_local_integral = (
        proposal_local_integral + schedule[:, 0].unsqueeze(0) * kernel.local_drift_integral
    )
    spot, variance, volterra, running_minimum = _assemble_level_with_h(
        S0=S0,
        mu=mu,
        H=H,
        xi=xi,
        eta=eta,
        rho=rho,
        step_dt=step_dt,
        target_driver_one=target_brownian[:, :, 0],
        target_driver_two=target_brownian[:, :, 1],
        target_local_integral=target_local_integral,
        kernel=kernel,
        method=method,
    )
    schedule_64 = schedule.to(torch.float64)
    proposal_64 = proposal_brownian.to(torch.float64)
    stochastic = torch.sum(schedule_64.unsqueeze(0) * proposal_64, dim=(1, 2))
    energy = step_dt * torch.sum(schedule_64.square()).expand(num_paths)
    return TwoDriverRBergomiPaths(
        spot=spot,
        variance=variance,
        volterra=volterra,
        running_minimum=running_minimum,
        log_likelihood=-stochastic - 0.5 * energy,
        control_energy=energy,
        step_dt=step_dt,
        proposal_brownian_increments=proposal_brownian,
        target_brownian_increments=target_brownian,
        proposal_local_integrals=proposal_local_integral,
        target_local_integrals=target_local_integral,
        controls=schedule.unsqueeze(0).expand(num_paths, -1, -1),
    )


def _validate_adjacent_innovations(
    innovations: AdjacentRBergomiFFTInnovations,
    *,
    paths: int,
    fine_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    coarse_steps = fine_steps // 2
    expected = (
        (innovations.first_cell_standard_normal, (paths, coarse_steps, 3)),
        (innovations.second_cell_standard_normal, (paths, coarse_steps, 2)),
        (innovations.price_standard_normal, (paths, fine_steps)),
    )
    for value, shape in expected:
        if (
            value.shape != shape
            or value.device != device
            or value.dtype != dtype
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError("adjacent innovations have invalid shape/device/dtype")


def simulate_coupled_rbergomi_adjacent_fft(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    fine_steps: int,
    num_paths: int,
    mu: float = 0.0,
    control_fn: RBergomiControl | None = None,
    override_params: dict | None = None,
    innovations: AdjacentRBergomiFFTInnovations | None = None,
    method: ConvolutionMethod = "fft",
    dtype: torch.dtype = torch.float64,
) -> CoupledRBergomiPaths:
    """Simulate exact adjacent BLP marginals with FFT historical convolutions."""
    if not math.isfinite(S0) or S0 <= 0.0 or not math.isfinite(T) or T <= 0.0:
        raise ValueError("S0 and T must be finite and positive")
    if fine_steps < 2 or fine_steps % 2 != 0 or num_paths <= 0 or not math.isfinite(mu):
        raise ValueError("fine_steps/num_paths/mu are invalid")
    params = simulator._resolved(override_params)
    H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
    coarse_steps = fine_steps // 2
    fine_dt = T / fine_steps
    coarse_dt = 2.0 * fine_dt
    fine_kernel = blp_fft_kernel(simulator, n_steps=fine_steps, step_dt=fine_dt, H=H, dtype=dtype)
    coarse_kernel = blp_fft_kernel(
        simulator, n_steps=coarse_steps, step_dt=coarse_dt, H=H, dtype=dtype
    )
    local = adjacent_local_gaussian_coefficients(simulator, fine_dt=fine_dt, H=H, dtype=dtype)
    schedule = _deterministic_schedule(
        control_fn,
        steps=fine_steps,
        step_dt=fine_dt,
        device=simulator.device,
        dtype=dtype,
    )
    if innovations is None:
        innovations = AdjacentRBergomiFFTInnovations(
            first_cell_standard_normal=torch.randn(
                num_paths, coarse_steps, 3, device=simulator.device, dtype=dtype
            ),
            second_cell_standard_normal=torch.randn(
                num_paths, coarse_steps, 2, device=simulator.device, dtype=dtype
            ),
            price_standard_normal=torch.randn(
                num_paths, fine_steps, device=simulator.device, dtype=dtype
            ),
        )
    _validate_adjacent_innovations(
        innovations,
        paths=num_paths,
        fine_steps=fine_steps,
        device=simulator.device,
        dtype=dtype,
    )
    first_triplet = innovations.first_cell_standard_normal @ local.first_cell_cholesky.T
    second_pair = innovations.second_cell_standard_normal @ local.second_cell_cholesky.T
    proposal_driver_one = torch.empty((num_paths, fine_steps), device=simulator.device, dtype=dtype)
    proposal_fine_local = torch.empty_like(proposal_driver_one)
    proposal_driver_one[:, 0::2] = first_triplet[:, :, 0]
    proposal_driver_one[:, 1::2] = second_pair[:, :, 0]
    proposal_fine_local[:, 0::2] = first_triplet[:, :, 1]
    proposal_fine_local[:, 1::2] = second_pair[:, :, 1]
    proposal_driver_two = innovations.price_standard_normal * math.sqrt(fine_dt)
    proposal_brownian = torch.stack((proposal_driver_one, proposal_driver_two), dim=2)
    target_brownian = proposal_brownian + schedule.unsqueeze(0) * fine_dt
    target_fine_local = (
        proposal_fine_local + schedule[:, 0].unsqueeze(0) * local.fine_drift_integral
    )
    proposal_coarse_local = first_triplet[:, :, 2] + second_pair[:, :, 1]
    target_coarse_local = (
        proposal_coarse_local
        + schedule[0::2, 0].unsqueeze(0) * local.coarse_first_drift_integral
        + schedule[1::2, 0].unsqueeze(0) * local.fine_drift_integral
    )
    target_coarse_brownian = target_brownian.reshape(num_paths, coarse_steps, 2, 2).sum(dim=2)

    fine_spot, fine_variance, fine_volterra, fine_running = _assemble_level_with_h(
        S0=S0,
        mu=mu,
        H=H,
        xi=xi,
        eta=eta,
        rho=rho,
        step_dt=fine_dt,
        target_driver_one=target_brownian[:, :, 0],
        target_driver_two=target_brownian[:, :, 1],
        target_local_integral=target_fine_local,
        kernel=fine_kernel,
        method=method,
    )
    coarse_spot, coarse_variance, coarse_volterra, coarse_running = _assemble_level_with_h(
        S0=S0,
        mu=mu,
        H=H,
        xi=xi,
        eta=eta,
        rho=rho,
        step_dt=coarse_dt,
        target_driver_one=target_coarse_brownian[:, :, 0],
        target_driver_two=target_coarse_brownian[:, :, 1],
        target_local_integral=target_coarse_local,
        kernel=coarse_kernel,
        method=method,
    )
    schedule_64 = schedule.to(torch.float64)
    proposal_64 = proposal_brownian.to(torch.float64)
    stochastic = torch.sum(schedule_64.unsqueeze(0) * proposal_64, dim=(1, 2))
    energy = fine_dt * torch.sum(schedule_64.square()).expand(num_paths)
    fine = RBergomiLevelPaths(
        spot=fine_spot,
        variance=fine_variance,
        volterra=fine_volterra,
        running_minimum=fine_running,
        step_dt=fine_dt,
        target_brownian_increments=target_brownian,
        target_local_integrals=target_fine_local,
    )
    coarse = RBergomiLevelPaths(
        spot=coarse_spot,
        variance=coarse_variance,
        volterra=coarse_volterra,
        running_minimum=coarse_running,
        step_dt=coarse_dt,
        target_brownian_increments=target_coarse_brownian,
        target_local_integrals=target_coarse_local,
    )
    return CoupledRBergomiPaths(
        fine=fine,
        coarse=coarse,
        log_likelihood=-stochastic - 0.5 * energy,
        control_energy=energy,
        proposal_fine_brownian_increments=proposal_brownian,
        target_fine_brownian_increments=target_brownian,
        proposal_fine_local_integrals=proposal_fine_local,
        target_fine_local_integrals=target_fine_local,
        proposal_coarse_local_integrals=proposal_coarse_local,
        target_coarse_local_integrals=target_coarse_local,
        fine_controls=schedule.unsqueeze(0).expand(num_paths, -1, -1),
    )
