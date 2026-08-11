"""Exact Cameron--Martin interpretation of BLP local Gaussian coordinates.

Each BLP cell uses two whitened coordinates for two observables of the *same*
Brownian path: the cell increment and the singular local-kernel integral.  This
module exposes the corresponding orthonormal cell functions and prevents those
coordinates from being misidentified as two independent continuum Brownian drivers.
"""

from __future__ import annotations

import math

import torch

from src.path_integral.rbergomi_fft import blp_fft_kernel
from src.physics_engine import RBergomiSimulator


def blp_cell_cameron_martin_shapes(
    reverse_times: torch.Tensor,
    *,
    hurst: float,
    step_dt: float,
) -> torch.Tensor:
    """Evaluate the two orthonormal CM shapes on one BLP cell.

    ``reverse_times`` is the distance to the cell's right endpoint and must lie in
    ``(0, step_dt]``.  The open endpoint is required because the rough kernel is
    singular when ``H<1/2`` even though it is square integrable.
    """

    if reverse_times.ndim != 1 or reverse_times.numel() < 1:
        raise ValueError("reverse_times must be a nonempty vector")
    if reverse_times.device.type != "cpu" or reverse_times.dtype != torch.float64:
        raise ValueError("reverse_times must be CPU float64")
    if not torch.isfinite(reverse_times).all() or bool((reverse_times <= 0.0).any()):
        raise ValueError("reverse_times must be finite and strictly positive")
    if not math.isfinite(step_dt) or step_dt <= 0.0 or bool((reverse_times > step_dt).any()):
        raise ValueError("reverse_times must lie within the cell")
    if not 0.0 < hurst < 0.5:
        raise ValueError("hurst must lie in (0,0.5)")
    simulator = RBergomiSimulator(H=hurst, device="cpu")
    kernel = blp_fft_kernel(
        simulator,
        n_steps=1,
        step_dt=step_dt,
        H=hurst,
        dtype=torch.float64,
    )
    alpha = hurst - 0.5
    observables = torch.stack(
        (torch.ones_like(reverse_times), torch.pow(reverse_times, alpha)),
        dim=1,
    )
    # If f=L e for the observable functions f and orthonormal functions e,
    # row-wise evaluations satisfy e=f L^{-T}.
    return torch.linalg.solve_triangular(
        kernel.local_cholesky,
        observables.T,
        upper=False,
    ).T


def blp_local_observable_means_from_standard_shift(
    standard_shift: torch.Tensor,
    *,
    hurst: float,
    step_dt: float,
) -> torch.Tensor:
    """Map whitened shifts to means of ``(Delta W, local kernel integral)``."""

    if standard_shift.ndim != 2 or standard_shift.shape[1] != 2:
        raise ValueError("standard_shift must have shape (cells,2)")
    if standard_shift.device.type != "cpu" or standard_shift.dtype != torch.float64:
        raise ValueError("standard_shift must be CPU float64")
    if not torch.isfinite(standard_shift).all():
        raise ValueError("standard_shift must be finite")
    simulator = RBergomiSimulator(H=hurst, device="cpu")
    kernel = blp_fft_kernel(
        simulator,
        n_steps=1,
        step_dt=step_dt,
        H=hurst,
        dtype=torch.float64,
    )
    return standard_shift @ kernel.local_cholesky.T


def piecewise_constant_drift_to_blp_standard_shift(
    drift: torch.Tensor,
    *,
    hurst: float,
    step_dt: float,
) -> torch.Tensor:
    """Project a cellwise-constant Brownian drift exactly into BLP coordinates."""

    if drift.ndim != 1 or drift.numel() < 1:
        raise ValueError("drift must be a nonempty vector")
    if drift.device.type != "cpu" or drift.dtype != torch.float64:
        raise ValueError("drift must be CPU float64")
    if not torch.isfinite(drift).all():
        raise ValueError("drift must be finite")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    if not 0.0 < hurst < 0.5:
        raise ValueError("hurst must lie in (0,0.5)")
    alpha = hurst - 0.5
    local_integral = step_dt ** (alpha + 1.0) / (alpha + 1.0)
    observable_means = torch.stack(
        (drift * step_dt, drift * local_integral),
        dim=1,
    )
    simulator = RBergomiSimulator(H=hurst, device="cpu")
    kernel = blp_fft_kernel(
        simulator,
        n_steps=1,
        step_dt=step_dt,
        H=hurst,
        dtype=torch.float64,
    )
    return torch.linalg.solve_triangular(
        kernel.local_cholesky,
        observable_means.T,
        upper=False,
    ).T
