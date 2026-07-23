"""Exact coarse-conditioned Gaussian bridge branching for BLP rBergomi paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.controllers.markov import TimePiecewiseTwoDriverControl
from src.path_integral.rbergomi_coupling import (
    adjacent_local_gaussian_coefficients,
)
from src.physics_engine import RBergomiSimulator, strict_lognormal_variance


@dataclass(frozen=True)
class ConditionalVolterraBridgeCoefficients:
    """Static Gaussian projection for one adjacent BLP block."""

    fine_covariance: torch.Tensor
    coarse_projection: torch.Tensor
    coarse_covariance: torch.Tensor
    conditional_gain: torch.Tensor
    conditional_covariance: torch.Tensor
    first_cell_cholesky: torch.Tensor
    second_cell_cholesky: torch.Tensor
    coarse_cholesky: torch.Tensor
    fine_drift_integral: float
    coarse_first_drift_integral: float


@dataclass(frozen=True)
class RBergomiCoarseTrunks:
    """Shared coarse BLP paths and innovations before fine bridge refinement."""

    spot: torch.Tensor
    variance: torch.Tensor
    volterra: torch.Tensor
    running_minimum: torch.Tensor
    proposal_brownian_increments: torch.Tensor
    target_brownian_increments: torch.Tensor
    proposal_local_integrals: torch.Tensor
    target_local_integrals: torch.Tensor
    fine_control_schedule: torch.Tensor
    fine_steps: int
    fine_dt: float
    maturity: float
    initial_spot: float
    mu: float
    model_parameters: tuple[float, float, float, float]


@dataclass(frozen=True)
class BranchedCoupledRBergomiPaths:
    """Fine conditional branches paired with one shared coarse path per parent."""

    fine_spot: torch.Tensor
    fine_variance: torch.Tensor
    fine_volterra: torch.Tensor
    fine_running_minimum: torch.Tensor
    trunks: RBergomiCoarseTrunks
    log_likelihood: torch.Tensor
    control_energy: float
    conditional_constraint_error: float
    proposal_fine_brownian_increments: torch.Tensor | None
    target_fine_brownian_increments: torch.Tensor | None
    proposal_fine_local_integrals: torch.Tensor | None
    target_fine_local_integrals: torch.Tensor | None

    @property
    def branches(self) -> int:
        return self.fine_spot.shape[1]

    @property
    def parents(self) -> int:
        return self.fine_spot.shape[0]


def subset_rbergomi_coarse_trunks(
    trunks: RBergomiCoarseTrunks, selector: torch.Tensor
) -> RBergomiCoarseTrunks:
    """Select parent trunks without altering their shared control schedule."""
    parents = trunks.spot.shape[0]
    if selector.ndim != 1 or selector.device != trunks.spot.device:
        raise ValueError("selector must be one-dimensional on the trunk device")
    if selector.dtype == torch.bool:
        if selector.shape[0] != parents:
            raise ValueError("boolean selector must match the parent batch")
    elif selector.dtype == torch.long:
        if bool((selector < 0).any()) or bool((selector >= parents).any()):
            raise ValueError("index selector is outside the parent batch")
    else:
        raise TypeError("selector must have bool or long dtype")

    def selected(value: torch.Tensor) -> torch.Tensor:
        return value[selector]

    return RBergomiCoarseTrunks(
        spot=selected(trunks.spot),
        variance=selected(trunks.variance),
        volterra=selected(trunks.volterra),
        running_minimum=selected(trunks.running_minimum),
        proposal_brownian_increments=selected(trunks.proposal_brownian_increments),
        target_brownian_increments=selected(trunks.target_brownian_increments),
        proposal_local_integrals=selected(trunks.proposal_local_integrals),
        target_local_integrals=selected(trunks.target_local_integrals),
        fine_control_schedule=trunks.fine_control_schedule,
        fine_steps=trunks.fine_steps,
        fine_dt=trunks.fine_dt,
        maturity=trunks.maturity,
        initial_spot=trunks.initial_spot,
        mu=trunks.mu,
        model_parameters=trunks.model_parameters,
    )


def conditional_volterra_bridge_coefficients(
    simulator: RBergomiSimulator,
    *,
    fine_dt: float,
    H: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> ConditionalVolterraBridgeCoefficients:
    """Build the exact projection from fine BLP variables to coarse variables."""
    local = adjacent_local_gaussian_coefficients(simulator, fine_dt=fine_dt, H=H, dtype=dtype)
    first_covariance = local.first_cell_cholesky @ local.first_cell_cholesky.T
    second_covariance = local.second_cell_cholesky @ local.second_cell_cholesky.T
    fine_covariance = torch.zeros((5, 5), device=simulator.device, dtype=dtype)
    fine_covariance[:3, :3] = first_covariance
    fine_covariance[3:, 3:] = second_covariance
    projection = torch.tensor(
        ((1.0, 0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0, 1.0)),
        device=simulator.device,
        dtype=dtype,
    )
    coarse_covariance = projection @ fine_covariance @ projection.T
    fine_to_coarse = fine_covariance @ projection.T
    gain = torch.linalg.solve(coarse_covariance, fine_to_coarse.T).T
    conditional_covariance = fine_covariance - gain @ coarse_covariance @ gain.T
    conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
    coarse_cholesky = torch.linalg.cholesky(coarse_covariance)
    return ConditionalVolterraBridgeCoefficients(
        fine_covariance=fine_covariance,
        coarse_projection=projection,
        coarse_covariance=coarse_covariance,
        conditional_gain=gain,
        conditional_covariance=conditional_covariance,
        first_cell_cholesky=local.first_cell_cholesky,
        second_cell_cholesky=local.second_cell_cholesky,
        coarse_cholesky=coarse_cholesky,
        fine_drift_integral=local.fine_drift_integral,
        coarse_first_drift_integral=local.coarse_first_drift_integral,
    )


def sample_conditional_volterra_fine_innovations(
    coefficients: ConditionalVolterraBridgeCoefficients,
    coarse_innovations: torch.Tensor,
    *,
    branches: int,
) -> torch.Tensor:
    """Sample ``F | A F = coarse`` independently across branch residuals.

    ``coarse_innovations`` has shape ``(parents, blocks, 2)``.  The output has
    shape ``(parents, branches, blocks, 5)`` in the order
    ``(X0, X1, X2, Y0, Y1)``.
    """
    if coarse_innovations.ndim != 3 or coarse_innovations.shape[-1] != 2:
        raise ValueError("coarse_innovations must have shape (parents, blocks, 2)")
    if branches <= 0:
        raise ValueError("branches must be positive")
    if (
        coarse_innovations.device != coefficients.fine_covariance.device
        or coarse_innovations.dtype != coefficients.fine_covariance.dtype
    ):
        raise ValueError("coarse innovations must match coefficient device and dtype")
    if not torch.isfinite(coarse_innovations).all():
        raise ValueError("coarse innovations must be finite")
    parents, blocks, _ = coarse_innovations.shape
    first_standard = torch.randn(
        parents,
        branches,
        blocks,
        3,
        device=coarse_innovations.device,
        dtype=coarse_innovations.dtype,
    )
    second_standard = torch.randn(
        parents,
        branches,
        blocks,
        2,
        device=coarse_innovations.device,
        dtype=coarse_innovations.dtype,
    )
    first = first_standard @ coefficients.first_cell_cholesky.T
    second = second_standard @ coefficients.second_cell_cholesky.T
    unconditional = torch.cat((first, second), dim=-1)
    unconditional_coarse = unconditional @ coefficients.coarse_projection.T
    residual = unconditional - unconditional_coarse @ coefficients.conditional_gain.T
    conditional_mean = coarse_innovations @ coefficients.conditional_gain.T
    return conditional_mean[:, None, :, :] + residual


def sample_conditional_brownian_fine_pairs(
    coarse_increments: torch.Tensor,
    *,
    fine_dt: float,
    branches: int,
) -> torch.Tensor:
    """Sample two fine Brownian increments conditional on their coarse sum."""
    if coarse_increments.ndim != 2:
        raise ValueError("coarse_increments must have shape (parents, blocks)")
    if not coarse_increments.is_floating_point() or not torch.isfinite(coarse_increments).all():
        raise ValueError("coarse increments must be finite floating point")
    if not math.isfinite(fine_dt) or fine_dt <= 0.0 or branches <= 0:
        raise ValueError("fine_dt and branches must be positive")
    parents, blocks = coarse_increments.shape
    unconditional = torch.randn(
        parents,
        branches,
        blocks,
        2,
        device=coarse_increments.device,
        dtype=coarse_increments.dtype,
    ) * math.sqrt(fine_dt)
    residual = unconditional - 0.5 * unconditional.sum(dim=-1, keepdim=True)
    return 0.5 * coarse_increments[:, None, :, None] + residual


def _control_schedule(
    control: TimePiecewiseTwoDriverControl | None,
    *,
    fine_steps: int,
    maturity: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if control is None:
        return torch.zeros((fine_steps, 2), device=device, dtype=dtype)
    if not isinstance(control, TimePiecewiseTwoDriverControl):
        raise TypeError("conditional branching currently requires time-piecewise control")
    if not math.isclose(control.maturity, maturity, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("control maturity must match simulation maturity")
    step_index = torch.arange(fine_steps, device=device)
    segment_index = torch.clamp(
        step_index * control.segments // fine_steps, max=control.segments - 1
    )
    return control.values.to(device=device, dtype=dtype)[segment_index]


def _rbergomi_paths_from_innovations(
    simulator: RBergomiSimulator,
    *,
    initial_spot: float,
    maturity: float,
    mu: float,
    target_brownian: torch.Tensor,
    target_local: torch.Tensor,
    H: float,
    eta: float,
    xi: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized BLP state reconstruction for arbitrary leading batch axes."""
    if target_brownian.shape[:-1] != target_local.shape or target_brownian.shape[-1] != 2:
        raise ValueError("target Brownian and local tensors have incompatible shapes")
    steps = target_local.shape[-1]
    step_dt = maturity / steps
    dtype = target_local.dtype
    _cholesky, weights, volterra_variance, _drift = simulator._hybrid_coefficients(
        steps, step_dt, H=H, dtype=dtype
    )
    leading_shape = target_local.shape[:-1]
    flat_driver_one = target_brownian[..., 0].reshape(-1, steps)
    flat_local = target_local.reshape(-1, steps)
    historical = flat_driver_one @ weights.T
    flat_volterra = math.sqrt(2.0 * H) * (historical + flat_local)
    flat_variance_after = strict_lognormal_variance(
        eta * flat_volterra - 0.5 * eta**2 * volterra_variance[1:],
        xi=xi,
    )
    initial_variance = torch.full(
        (flat_variance_after.shape[0], 1), xi, device=target_local.device, dtype=dtype
    )
    flat_variance = torch.cat((initial_variance, flat_variance_after), dim=-1)
    rho_perpendicular = math.sqrt(max(1.0 - rho * rho, 0.0))
    flat_brownian = target_brownian.reshape(-1, steps, 2)
    spot_driver = rho * flat_brownian[..., 0] + rho_perpendicular * flat_brownian[..., 1]
    log_increments = (mu - 0.5 * flat_variance[:, :-1]) * step_dt + torch.sqrt(
        flat_variance[:, :-1]
    ) * spot_driver
    initial_log_spot = torch.full(
        (flat_variance.shape[0], 1),
        math.log(initial_spot),
        device=target_local.device,
        dtype=dtype,
    )
    flat_log_spot = torch.cat(
        (initial_log_spot, initial_log_spot + torch.cumsum(log_increments, dim=-1)),
        dim=-1,
    )
    flat_spot = torch.exp(flat_log_spot)
    if not torch.isfinite(flat_spot).all() or not torch.isfinite(flat_variance).all():
        raise FloatingPointError("branched rBergomi path became nonfinite")
    flat_volterra_with_zero = torch.cat(
        (
            torch.zeros(
                (flat_volterra.shape[0], 1),
                device=target_local.device,
                dtype=dtype,
            ),
            flat_volterra,
        ),
        dim=-1,
    )
    output_shape = (*leading_shape, steps + 1)
    spot = flat_spot.reshape(output_shape)
    variance = flat_variance.reshape(output_shape)
    volterra = flat_volterra_with_zero.reshape(output_shape)
    running_minimum = torch.cummin(spot, dim=-1).values
    return spot, variance, volterra, running_minimum


def sample_rbergomi_coarse_trunks(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    fine_steps: int,
    num_parents: int,
    control: TimePiecewiseTwoDriverControl | None = None,
    mu: float = 0.0,
    override_params: dict | None = None,
    dtype: torch.dtype = torch.float64,
) -> RBergomiCoarseTrunks:
    """Sample shared coarse proposal innovations and their target BLP paths."""
    if not math.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 must be finite and positive")
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive")
    if fine_steps < 2 or fine_steps % 2:
        raise ValueError("fine_steps must be a positive even integer")
    if num_parents <= 0:
        raise ValueError("num_parents must be positive")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    params = simulator._resolved(override_params)
    H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
    fine_dt = T / fine_steps
    coarse_steps = fine_steps // 2
    coarse_dt = 2.0 * fine_dt
    coefficients = conditional_volterra_bridge_coefficients(
        simulator, fine_dt=fine_dt, H=H, dtype=dtype
    )
    coarse_standard = torch.randn(
        num_parents, coarse_steps, 2, device=simulator.device, dtype=dtype
    )
    proposal_coarse_pair = coarse_standard @ coefficients.coarse_cholesky.T
    proposal_coarse_driver_one = proposal_coarse_pair[..., 0]
    proposal_coarse_local = proposal_coarse_pair[..., 1]
    proposal_coarse_driver_two = torch.randn(
        num_parents, coarse_steps, device=simulator.device, dtype=dtype
    ) * math.sqrt(coarse_dt)
    proposal_coarse_brownian = torch.stack(
        (proposal_coarse_driver_one, proposal_coarse_driver_two), dim=-1
    )
    schedule = _control_schedule(
        control,
        fine_steps=fine_steps,
        maturity=T,
        device=simulator.device,
        dtype=dtype,
    )
    paired_control = schedule.reshape(coarse_steps, 2, 2)
    coarse_brownian_shift = fine_dt * paired_control.sum(dim=1)
    coarse_local_shift = (
        paired_control[:, 0, 0] * coefficients.coarse_first_drift_integral
        + paired_control[:, 1, 0] * coefficients.fine_drift_integral
    )
    target_coarse_brownian = proposal_coarse_brownian + coarse_brownian_shift[None, :, :]
    target_coarse_local = proposal_coarse_local + coarse_local_shift[None, :]
    spot, variance, volterra, running_minimum = _rbergomi_paths_from_innovations(
        simulator,
        initial_spot=S0,
        maturity=T,
        mu=mu,
        target_brownian=target_coarse_brownian,
        target_local=target_coarse_local,
        H=H,
        eta=eta,
        xi=xi,
        rho=rho,
    )
    return RBergomiCoarseTrunks(
        spot=spot,
        variance=variance,
        volterra=volterra,
        running_minimum=running_minimum,
        proposal_brownian_increments=proposal_coarse_brownian,
        target_brownian_increments=target_coarse_brownian,
        proposal_local_integrals=proposal_coarse_local,
        target_local_integrals=target_coarse_local,
        fine_control_schedule=schedule,
        fine_steps=fine_steps,
        fine_dt=fine_dt,
        maturity=T,
        initial_spot=S0,
        mu=mu,
        model_parameters=(H, eta, xi, rho),
    )


def refine_rbergomi_coarse_trunks(
    simulator: RBergomiSimulator,
    trunks: RBergomiCoarseTrunks,
    *,
    branches: int,
    record_augmented: bool = False,
) -> BranchedCoupledRBergomiPaths:
    """Conditionally refine shared coarse innovations into iid fine branches."""
    if branches <= 0:
        raise ValueError("branches must be positive")
    H, eta, xi, rho = trunks.model_parameters
    coefficients = conditional_volterra_bridge_coefficients(
        simulator, fine_dt=trunks.fine_dt, H=H, dtype=trunks.spot.dtype
    )
    coarse_volterra_innovations = torch.stack(
        (
            trunks.proposal_brownian_increments[..., 0],
            trunks.proposal_local_integrals,
        ),
        dim=-1,
    )
    conditional = sample_conditional_volterra_fine_innovations(
        coefficients, coarse_volterra_innovations, branches=branches
    )
    parents, _, blocks, _ = conditional.shape
    proposal_driver_one = torch.empty(
        parents,
        branches,
        trunks.fine_steps,
        device=trunks.spot.device,
        dtype=trunks.spot.dtype,
    )
    proposal_fine_local = torch.empty_like(proposal_driver_one)
    proposal_driver_one[..., 0::2] = conditional[..., 0]
    proposal_driver_one[..., 1::2] = conditional[..., 3]
    proposal_fine_local[..., 0::2] = conditional[..., 1]
    proposal_fine_local[..., 1::2] = conditional[..., 4]
    proposal_driver_two_pairs = sample_conditional_brownian_fine_pairs(
        trunks.proposal_brownian_increments[..., 1],
        fine_dt=trunks.fine_dt,
        branches=branches,
    )
    proposal_driver_two = proposal_driver_two_pairs.reshape(parents, branches, trunks.fine_steps)
    proposal_fine_brownian = torch.stack((proposal_driver_one, proposal_driver_two), dim=-1)
    schedule = trunks.fine_control_schedule
    target_fine_brownian = proposal_fine_brownian + schedule[None, None, :, :] * trunks.fine_dt
    target_fine_local = (
        proposal_fine_local + schedule[None, None, :, 0] * coefficients.fine_drift_integral
    )
    fine_spot, fine_variance, fine_volterra, fine_running_minimum = (
        _rbergomi_paths_from_innovations(
            simulator,
            initial_spot=trunks.initial_spot,
            maturity=trunks.maturity,
            mu=trunks.mu,
            target_brownian=target_fine_brownian,
            target_local=target_fine_local,
            H=H,
            eta=eta,
            xi=xi,
            rho=rho,
        )
    )
    schedule_64 = schedule.to(torch.float64)
    stochastic = torch.sum(
        proposal_fine_brownian.to(torch.float64) * schedule_64[None, None, :, :],
        dim=(-2, -1),
    )
    control_energy = trunks.fine_dt * float(torch.sum(schedule_64.square()))
    log_likelihood = -stochastic - 0.5 * control_energy
    reconstructed_coarse = conditional @ coefficients.coarse_projection.T
    constraint_error = torch.max(
        torch.abs(reconstructed_coarse - coarse_volterra_innovations[:, None, :, :])
    )
    brownian_constraint = torch.max(
        torch.abs(
            proposal_driver_two_pairs.sum(dim=-1)
            - trunks.proposal_brownian_increments[:, None, :, 1]
        )
    )
    maximum_constraint_error = max(
        float(constraint_error.detach()), float(brownian_constraint.detach())
    )
    return BranchedCoupledRBergomiPaths(
        fine_spot=fine_spot,
        fine_variance=fine_variance,
        fine_volterra=fine_volterra,
        fine_running_minimum=fine_running_minimum,
        trunks=trunks,
        log_likelihood=log_likelihood,
        control_energy=control_energy,
        conditional_constraint_error=maximum_constraint_error,
        proposal_fine_brownian_increments=(proposal_fine_brownian if record_augmented else None),
        target_fine_brownian_increments=(target_fine_brownian if record_augmented else None),
        proposal_fine_local_integrals=(proposal_fine_local if record_augmented else None),
        target_fine_local_integrals=(target_fine_local if record_augmented else None),
    )


def simulate_branched_rbergomi_adjacent(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    fine_steps: int,
    num_parents: int,
    branches: int,
    control: TimePiecewiseTwoDriverControl | None = None,
    mu: float = 0.0,
    override_params: dict | None = None,
    record_augmented: bool = False,
    dtype: torch.dtype = torch.float64,
) -> BranchedCoupledRBergomiPaths:
    """Convenience entry point for trunk sampling followed by fine branching."""
    trunks = sample_rbergomi_coarse_trunks(
        simulator,
        S0=S0,
        T=T,
        fine_steps=fine_steps,
        num_parents=num_parents,
        control=control,
        mu=mu,
        override_params=override_params,
        dtype=dtype,
    )
    return refine_rbergomi_coarse_trunks(
        simulator,
        trunks,
        branches=branches,
        record_augmented=record_augmented,
    )
