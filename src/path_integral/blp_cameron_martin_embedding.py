"""Exact Cameron--Martin interpretation of BLP local Gaussian coordinates.

Each BLP cell uses two whitened coordinates for two observables of the *same*
Brownian path: the cell increment and the singular local-kernel integral.  This
module exposes the corresponding orthonormal cell functions and prevents those
coordinates from being misidentified as two independent continuum Brownian drivers.
"""

from __future__ import annotations

import math

import torch

from src.path_integral.cameron_martin_basis import (
    CameronMartinBasis,
    build_blp_cameron_martin_basis,
)
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


def build_mesh_compatible_blp_trace_safety_geometry(
    *,
    steps: int,
    maturity: float,
    hurst: float,
    spectrum_decay: float,
    spectrum_scale: float,
    complement_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a full-grid safety eigensystem with a continuum cosine limit.

    The first ``steps`` directions are exact BLP embeddings of orthonormal
    piecewise-constant cosine drifts.  Their eigenvalues discretize a positive
    summable continuum spectrum.  The orthogonal within-cell bridge complement
    remains strictly positive on every frozen grid, but its total trace vanishes
    as the mesh is refined.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    if not 0.0 < hurst < 0.5:
        raise ValueError("hurst must lie in (0,0.5)")
    if not math.isfinite(spectrum_decay) or spectrum_decay <= 1.0:
        raise ValueError("spectrum_decay must exceed one")
    if not math.isfinite(spectrum_scale) or spectrum_scale <= 0.0:
        raise ValueError("spectrum_scale must be finite and positive")
    if not math.isfinite(complement_decay) or complement_decay <= 1.0:
        raise ValueError("complement_decay must exceed one")
    step_dt = maturity / steps
    temporal = build_blp_cameron_martin_basis(
        steps=steps,
        drivers=1,
    ).matrix
    unit_shift = piecewise_constant_drift_to_blp_standard_shift(
        torch.ones(1, dtype=torch.float64),
        hurst=hurst,
        step_dt=step_dt,
    )[0]
    drift_values = temporal / math.sqrt(step_dt)
    main = (drift_values[:, :, None] * unit_shift[None, None, :]).permute(0, 2, 1)
    main = main.reshape(2 * steps, steps)
    gram = main.T @ main
    if float(torch.amax(torch.abs(gram - torch.eye(steps, dtype=torch.float64)))) > 2e-11:
        raise RuntimeError("embedded continuum safety directions lost orthonormality")
    complete, _ = torch.linalg.qr(main, mode="complete")
    complement = complete[:, steps:]
    directions = torch.cat((main, complement), dim=1)
    full_gram = directions.T @ directions
    if float(torch.amax(torch.abs(full_gram - torch.eye(2 * steps, dtype=torch.float64)))) > 2e-10:
        raise RuntimeError("full mesh safety directions lost orthonormality")
    frequencies = torch.arange(steps, dtype=torch.float64)
    continuum_spectrum = spectrum_scale / torch.pow(1.0 + frequencies, spectrum_decay)
    bridge_eigenvalue = spectrum_scale * steps ** (-complement_decay)
    bridge_spectrum = torch.full((steps,), bridge_eigenvalue, dtype=torch.float64)
    return directions, torch.cat((continuum_spectrum, bridge_spectrum))


def build_mesh_compatible_blp_drift_basis(
    *,
    steps: int,
    maturity: float,
    hurst: float,
    modes: int,
) -> CameronMartinBasis:
    """Embed low-frequency continuum Brownian drifts into the BLP grid exactly.

    Unlike a channel-separated BLP DCT basis, every column here is one actual
    cellwise-constant drift of the single volatility Brownian motion.  Consequently
    a fixed column has a well-defined continuum limit as the grid is refined.
    """

    if isinstance(modes, bool) or not isinstance(modes, int) or not 1 <= modes <= steps:
        raise ValueError("modes must lie between one and steps")
    directions, _ = build_mesh_compatible_blp_trace_safety_geometry(
        steps=steps,
        maturity=maturity,
        hurst=hurst,
        spectrum_decay=2.0,
        spectrum_scale=1.0,
        complement_decay=2.0,
    )
    return CameronMartinBasis(
        matrix=directions[:, :modes],
        steps=steps,
        drivers=2,
        modes_per_driver=modes,
        channel_separated=False,
    )
