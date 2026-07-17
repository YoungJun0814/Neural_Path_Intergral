"""Exact adjacent-grid coupling for the kappa=1 BLP rBergomi scheme.

The coupling is exact for the two declared finite-grid BLP marginals.  In
particular, the coarse recent-cell integral is *not* obtained by adding the
two fine recent-cell integrals.  The first fine cell carries an additional
integral evaluated against the kernel whose endpoint is the coarse endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from scipy.integrate import quad

from src.physics_engine import RBergomiSimulator

RBergomiControl = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class RBergomiLevelPaths:
    """One marginal of an adjacent-grid coupled simulation."""

    spot: torch.Tensor
    variance: torch.Tensor
    volterra: torch.Tensor
    running_minimum: torch.Tensor
    step_dt: float
    target_brownian_increments: torch.Tensor | None
    target_local_integrals: torch.Tensor | None


@dataclass(frozen=True)
class CoupledRBergomiPaths:
    """Fine/coarse BLP paths driven by one fine-grid proposal law."""

    fine: RBergomiLevelPaths
    coarse: RBergomiLevelPaths
    log_likelihood: torch.Tensor
    control_energy: torch.Tensor
    proposal_fine_brownian_increments: torch.Tensor | None
    target_fine_brownian_increments: torch.Tensor | None
    proposal_fine_local_integrals: torch.Tensor | None
    target_fine_local_integrals: torch.Tensor | None
    proposal_coarse_local_integrals: torch.Tensor | None
    target_coarse_local_integrals: torch.Tensor | None
    fine_controls: torch.Tensor | None


@dataclass(frozen=True)
class AdjacentLocalGaussianCoefficients:
    """Covariance data for one fine pair inside a coarse BLP cell."""

    first_cell_cholesky: torch.Tensor
    second_cell_cholesky: torch.Tensor
    fine_drift_integral: float
    coarse_first_drift_integral: float


def adjacent_local_gaussian_coefficients(
    simulator: RBergomiSimulator,
    *,
    fine_dt: float,
    H: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> AdjacentLocalGaussianCoefficients:
    r"""Return exact Gaussian factors for an adjacent fine/coarse cell pair.

    On the first fine interval this jointly samples

    ``(int dW, int (h-s)^alpha dW, int (2h-s)^alpha dW)``.

    The cross-kernel covariance has no simplification used by the simulator,
    so it is evaluated once by deterministic quadrature in double precision.
    """
    if not math.isfinite(fine_dt) or fine_dt <= 0.0:
        raise ValueError("fine_dt must be finite and positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    resolved_H = simulator.H if H is None else float(H)
    if not 0.0 < resolved_H < 0.5:
        raise ValueError("rBergomi requires H in (0, 0.5)")
    alpha = resolved_H - 0.5
    h = float(fine_dt)
    c00 = h
    c01 = h ** (alpha + 1.0) / (alpha + 1.0)
    c11 = h ** (2.0 * alpha + 1.0) / (2.0 * alpha + 1.0)
    c02 = ((2.0 * h) ** (alpha + 1.0) - h ** (alpha + 1.0)) / (
        alpha + 1.0
    )
    c22 = (
        (2.0 * h) ** (2.0 * alpha + 1.0)
        - h ** (2.0 * alpha + 1.0)
    ) / (2.0 * alpha + 1.0)
    c12, quadrature_error = quad(
        lambda r: (h - r) ** alpha * (2.0 * h - r) ** alpha,
        0.0,
        h,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=200,
    )
    if not math.isfinite(c12) or quadrature_error > 1e-10 * max(abs(c12), 1.0):
        raise FloatingPointError("cross-kernel covariance quadrature failed")
    covariance = torch.tensor(
        ((c00, c01, c02), (c01, c11, c12), (c02, c12, c22)),
        device=simulator.device,
        dtype=dtype,
    )
    first_cholesky = torch.linalg.cholesky(covariance)
    second_cholesky, _weights, _variance, _drift = simulator._hybrid_coefficients(
        1, h, H=resolved_H, dtype=dtype
    )
    return AdjacentLocalGaussianCoefficients(
        first_cell_cholesky=first_cholesky,
        second_cell_cholesky=second_cholesky,
        fine_drift_integral=c01,
        coarse_first_drift_integral=c02,
    )


def _validated_control(
    control_fn: RBergomiControl | None,
    *,
    time: float,
    spot: torch.Tensor,
    variance: torch.Tensor,
    volterra: torch.Tensor,
    running_minimum: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch = spot.shape[0]
    if control_fn is None:
        return torch.zeros((batch, 2), device=device, dtype=dtype)
    if bool(getattr(control_fn, "uses_running_minimum", False)):
        stateful = cast(Callable[..., torch.Tensor], control_fn)
        value = stateful(time, spot, variance, volterra, running_minimum)
    else:
        value = control_fn(time, spot, variance, volterra)
    if not isinstance(value, torch.Tensor):
        raise TypeError("rBergomi control_fn must return a torch.Tensor")
    if value.shape != (batch, 2):
        raise ValueError("rBergomi control_fn must return shape (num_paths, 2)")
    if value.device != device or value.dtype != dtype:
        raise ValueError("rBergomi control output must match simulator device and dtype")
    if not torch.isfinite(value).all():
        raise ValueError("rBergomi control output must be finite")
    return value


def simulate_coupled_rbergomi_adjacent(
    simulator: RBergomiSimulator,
    *,
    S0: float,
    T: float,
    fine_steps: int,
    num_paths: int,
    mu: float = 0.0,
    control_fn: RBergomiControl | None = None,
    override_params: dict | None = None,
    record_augmented: bool = False,
    dtype: torch.dtype = torch.float64,
) -> CoupledRBergomiPaths:
    """Simulate exact adjacent fine/coarse kappa=1 BLP marginals.

    A single causal control acts on the independent fine Brownian coordinates.
    Consequently the returned ``log_likelihood`` is the only likelihood that
    may multiply either the fine payoff or the signed fine-minus-coarse
    correction.
    """
    if not math.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 must be finite and positive")
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive")
    if fine_steps < 2 or fine_steps % 2 != 0:
        raise ValueError("fine_steps must be a positive even integer")
    if num_paths <= 0:
        raise ValueError("num_paths must be positive")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")

    params = simulator._resolved(override_params)
    H, eta, xi, rho = params["H"], params["eta"], params["xi"], params["rho"]
    coarse_steps = fine_steps // 2
    fine_dt = T / fine_steps
    coarse_dt = 2.0 * fine_dt
    rho_perpendicular = math.sqrt(max(1.0 - rho * rho, 0.0))
    fine_local = adjacent_local_gaussian_coefficients(
        simulator, fine_dt=fine_dt, H=H, dtype=dtype
    )
    _fine_chol, fine_weights, fine_volterra_variance, _fine_drift = (
        simulator._hybrid_coefficients(fine_steps, fine_dt, H=H, dtype=dtype)
    )
    _coarse_chol, coarse_weights, coarse_volterra_variance, _coarse_drift = (
        simulator._hybrid_coefficients(coarse_steps, coarse_dt, H=H, dtype=dtype)
    )
    volterra_scale = math.sqrt(2.0 * H)
    device = simulator.device

    fine_log_spot = torch.full((num_paths,), math.log(S0), device=device, dtype=dtype)
    coarse_log_spot = fine_log_spot.clone()
    fine_volterra = torch.zeros(num_paths, device=device, dtype=dtype)
    coarse_volterra = fine_volterra.clone()
    fine_variance = torch.full((num_paths,), xi, device=device, dtype=dtype)
    coarse_variance = fine_variance.clone()
    fine_running_minimum = torch.full((num_paths,), S0, device=device, dtype=dtype)
    coarse_running_minimum = fine_running_minimum.clone()

    fine_spot_history = [torch.exp(fine_log_spot)]
    coarse_spot_history = [torch.exp(coarse_log_spot)]
    fine_variance_history = [fine_variance]
    coarse_variance_history = [coarse_variance]
    fine_volterra_history = [fine_volterra]
    coarse_volterra_history = [coarse_volterra]
    fine_minimum_history = [fine_running_minimum]
    coarse_minimum_history = [coarse_running_minimum]
    target_driver_one_history: list[torch.Tensor] = []
    coarse_driver_one_history: list[torch.Tensor] = []

    reset_memory = getattr(control_fn, "reset_for_simulation", None)
    if callable(reset_memory):
        reset_memory(batch_size=num_paths, device=device, dtype=dtype)

    stochastic_log_term = torch.zeros(num_paths, device=device, dtype=torch.float64)
    control_energy = torch.zeros(num_paths, device=device, dtype=torch.float64)
    proposal_brownian_history: list[torch.Tensor] | None = [] if record_augmented else None
    target_brownian_history: list[torch.Tensor] | None = [] if record_augmented else None
    proposal_fine_local_history: list[torch.Tensor] | None = [] if record_augmented else None
    target_fine_local_history: list[torch.Tensor] | None = [] if record_augmented else None
    proposal_coarse_local_history: list[torch.Tensor] | None = [] if record_augmented else None
    target_coarse_local_history: list[torch.Tensor] | None = [] if record_augmented else None
    control_history: list[torch.Tensor] | None = [] if record_augmented else None
    target_coarse_brownian_history: list[torch.Tensor] | None = [] if record_augmented else None

    stored_proposal_first_coarse_local: torch.Tensor | None = None
    stored_target_first_coarse_local: torch.Tensor | None = None
    stored_target_brownian: torch.Tensor | None = None

    for step in range(fine_steps):
        time = step * fine_dt
        fine_spot = torch.exp(fine_log_spot)
        control = _validated_control(
            control_fn,
            time=time,
            spot=fine_spot,
            variance=fine_variance,
            volterra=fine_volterra,
            running_minimum=fine_running_minimum,
            device=device,
            dtype=dtype,
        )

        if step % 2 == 0:
            standard = torch.randn(num_paths, 3, device=device, dtype=dtype)
            first_triplet = standard @ fine_local.first_cell_cholesky.T
            proposal_driver_one = first_triplet[:, 0]
            proposal_fine_integral = first_triplet[:, 1]
            stored_proposal_first_coarse_local = first_triplet[:, 2]
            stored_target_first_coarse_local = (
                stored_proposal_first_coarse_local
                + control[:, 0] * fine_local.coarse_first_drift_integral
            )
        else:
            standard = torch.randn(num_paths, 2, device=device, dtype=dtype)
            second_pair = standard @ fine_local.second_cell_cholesky.T
            proposal_driver_one = second_pair[:, 0]
            proposal_fine_integral = second_pair[:, 1]

        proposal_driver_two = (
            torch.randn(num_paths, device=device, dtype=dtype) * math.sqrt(fine_dt)
        )
        proposal_brownian = torch.stack((proposal_driver_one, proposal_driver_two), dim=-1)
        target_brownian = proposal_brownian + control * fine_dt
        target_driver_one = target_brownian[:, 0]
        target_driver_two = target_brownian[:, 1]
        target_fine_integral = (
            proposal_fine_integral + control[:, 0] * fine_local.fine_drift_integral
        )

        observe_increment = getattr(control_fn, "observe_target_increment", None)
        if callable(observe_increment):
            observe_increment(target_driver_one, fine_dt)

        spot_increment = rho * target_driver_one + rho_perpendicular * target_driver_two
        fine_log_spot = (
            fine_log_spot
            + (mu - 0.5 * fine_variance) * fine_dt
            + torch.sqrt(fine_variance) * spot_increment
        )
        target_driver_one_history.append(target_driver_one)
        fine_driver_matrix = torch.stack(target_driver_one_history, dim=1)
        fine_historical = torch.sum(
            fine_driver_matrix * fine_weights[step, : step + 1], dim=1
        )
        fine_volterra = volterra_scale * (fine_historical + target_fine_integral)
        fine_variance = xi * torch.exp(
            eta * fine_volterra
            - 0.5 * eta**2 * fine_volterra_variance[step + 1]
        )
        fine_variance = torch.clamp(fine_variance, min=1e-10)
        if not torch.isfinite(fine_variance).all() or not torch.isfinite(fine_log_spot).all():
            raise FloatingPointError("fine coupled rBergomi path became nonfinite")
        fine_running_minimum = torch.minimum(fine_running_minimum, torch.exp(fine_log_spot))
        fine_spot_history.append(torch.exp(fine_log_spot))
        fine_variance_history.append(fine_variance)
        fine_volterra_history.append(fine_volterra)
        fine_minimum_history.append(fine_running_minimum)

        control_64 = control.to(torch.float64)
        proposal_64 = proposal_brownian.to(torch.float64)
        stochastic_log_term += torch.sum(control_64 * proposal_64, dim=-1)
        control_energy += fine_dt * torch.sum(control_64.square(), dim=-1)

        if step % 2 == 0:
            stored_target_brownian = target_brownian
        else:
            if (
                stored_target_brownian is None
                or stored_proposal_first_coarse_local is None
                or stored_target_first_coarse_local is None
            ):
                raise RuntimeError("incomplete adjacent coupling state")
            coarse_brownian = stored_target_brownian + target_brownian
            proposal_coarse_integral = (
                stored_proposal_first_coarse_local + proposal_fine_integral
            )
            target_coarse_integral = (
                stored_target_first_coarse_local + target_fine_integral
            )
            coarse_spot_increment = (
                rho * coarse_brownian[:, 0]
                + rho_perpendicular * coarse_brownian[:, 1]
            )
            coarse_log_spot = (
                coarse_log_spot
                + (mu - 0.5 * coarse_variance) * coarse_dt
                + torch.sqrt(coarse_variance) * coarse_spot_increment
            )
            coarse_driver_one_history.append(coarse_brownian[:, 0])
            coarse_driver_matrix = torch.stack(coarse_driver_one_history, dim=1)
            coarse_index = step // 2
            coarse_historical = torch.sum(
                coarse_driver_matrix
                * coarse_weights[coarse_index, : coarse_index + 1],
                dim=1,
            )
            coarse_volterra = volterra_scale * (
                coarse_historical + target_coarse_integral
            )
            coarse_variance = xi * torch.exp(
                eta * coarse_volterra
                - 0.5 * eta**2 * coarse_volterra_variance[coarse_index + 1]
            )
            coarse_variance = torch.clamp(coarse_variance, min=1e-10)
            if (
                not torch.isfinite(coarse_variance).all()
                or not torch.isfinite(coarse_log_spot).all()
            ):
                raise FloatingPointError("coarse coupled rBergomi path became nonfinite")
            coarse_running_minimum = torch.minimum(
                coarse_running_minimum, torch.exp(coarse_log_spot)
            )
            coarse_spot_history.append(torch.exp(coarse_log_spot))
            coarse_variance_history.append(coarse_variance)
            coarse_volterra_history.append(coarse_volterra)
            coarse_minimum_history.append(coarse_running_minimum)
            if proposal_coarse_local_history is not None:
                assert target_coarse_local_history is not None
                assert target_coarse_brownian_history is not None
                proposal_coarse_local_history.append(proposal_coarse_integral)
                target_coarse_local_history.append(target_coarse_integral)
                target_coarse_brownian_history.append(coarse_brownian)

        if proposal_brownian_history is not None:
            assert target_brownian_history is not None
            assert proposal_fine_local_history is not None
            assert target_fine_local_history is not None
            assert control_history is not None
            proposal_brownian_history.append(proposal_brownian)
            target_brownian_history.append(target_brownian)
            proposal_fine_local_history.append(proposal_fine_integral)
            target_fine_local_history.append(target_fine_integral)
            control_history.append(control)

    def stack_optional(values: list[torch.Tensor] | None) -> torch.Tensor | None:
        return torch.stack(values, dim=1) if values is not None else None

    fine_target_brownian = stack_optional(target_brownian_history)
    fine_target_local = stack_optional(target_fine_local_history)
    coarse_target_brownian = stack_optional(target_coarse_brownian_history)
    coarse_target_local = stack_optional(target_coarse_local_history)
    fine_paths = RBergomiLevelPaths(
        spot=torch.stack(fine_spot_history, dim=1),
        variance=torch.stack(fine_variance_history, dim=1),
        volterra=torch.stack(fine_volterra_history, dim=1),
        running_minimum=torch.stack(fine_minimum_history, dim=1),
        step_dt=fine_dt,
        target_brownian_increments=fine_target_brownian,
        target_local_integrals=fine_target_local,
    )
    coarse_paths = RBergomiLevelPaths(
        spot=torch.stack(coarse_spot_history, dim=1),
        variance=torch.stack(coarse_variance_history, dim=1),
        volterra=torch.stack(coarse_volterra_history, dim=1),
        running_minimum=torch.stack(coarse_minimum_history, dim=1),
        step_dt=coarse_dt,
        target_brownian_increments=coarse_target_brownian,
        target_local_integrals=coarse_target_local,
    )
    return CoupledRBergomiPaths(
        fine=fine_paths,
        coarse=coarse_paths,
        log_likelihood=-stochastic_log_term - 0.5 * control_energy,
        control_energy=control_energy,
        proposal_fine_brownian_increments=stack_optional(proposal_brownian_history),
        target_fine_brownian_increments=fine_target_brownian,
        proposal_fine_local_integrals=stack_optional(proposal_fine_local_history),
        target_fine_local_integrals=fine_target_local,
        proposal_coarse_local_integrals=stack_optional(proposal_coarse_local_history),
        target_coarse_local_integrals=coarse_target_local,
        fine_controls=stack_optional(control_history),
    )
