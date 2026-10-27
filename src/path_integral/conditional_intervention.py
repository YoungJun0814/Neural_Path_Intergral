"""Exact conditional-mixture interventions and a fixed N32 sign projection.

No fitted density, quadrature, clipping, or accuracy certificate is supplied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.special import log_ndtr, logsumexp

from .gaussian_span_marginalization import GaussianMixtureShiftSpec


def _coordinates(z: torch.Tensor, dimension: int) -> None:
    if (z.ndim != 2 or z.shape[1] != dimension or not len(z)
            or z.dtype != torch.float64 or z.device.type != "cpu" or z.requires_grad
            or not torch.isfinite(z).all()):
        raise ValueError("finite detached nonempty CPU float64 coordinates required")


def _group(group: Sequence[int], dimension: int) -> list[int]:
    indices = list(group)
    if (not indices or any(type(i) is not int or not 0 <= i < dimension for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("nonempty distinct in-range coordinate group required")
    return indices


def conditional_component_probabilities(spec: GaussianMixtureShiftSpec, z: torch.Tensor,
                                        group: Sequence[int]) -> torch.Tensor:
    """Component posterior using only the retained coordinates, NOT old labels."""
    _coordinates(z, spec.dimension)
    indices = _group(group, spec.dimension)
    retain = torch.ones(spec.dimension, dtype=torch.bool)
    retain[indices] = False
    means = spec.means[:, retain]
    logits = z[:, retain] @ means.T - .5*means.square().sum(1) + spec.weights.log()
    return torch.softmax(logits, dim=1)


def conditional_replace(spec: GaussianMixtureShiftSpec, z: torch.Tensor, group: Sequence[int], *,
                        label_generator: torch.Generator,
                        gaussian_generator: torch.Generator) -> torch.Tensor:
    probabilities = conditional_component_probabilities(spec, z, group)
    indices = _group(group, spec.dimension)
    labels = torch.multinomial(probabilities, 1, generator=label_generator).flatten()
    noise = torch.randn((len(z), len(indices)), dtype=torch.float64, generator=gaussian_generator)
    replaced = z.clone()
    replaced[:, indices] = spec.means[labels][:, indices] + noise
    return replaced


def log_half_squared_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if (a.shape != b.shape or a.dtype != torch.float64 or b.dtype != torch.float64
            or a.device.type != "cpu" or b.device.type != "cpu" or a.requires_grad or b.requires_grad
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError("matching finite detached CPU float64 log contributions required")
    # expm1 preserves small differences. Exact equality is a genuine log zero.
    maximum = torch.maximum(a, b)
    difference = -torch.expm1(-torch.abs(a-b))
    return 2*(maximum+difference.log())-math.log(2)


def require_blind_residuals(spec: GaussianMixtureShiftSpec) -> None:
    if spec.dimension != 64 or not bool((spec.means[:, [57, 61, 63]] == 0).all()):
        raise ValueError("fixed N32 identity shifts with zero means57/61/63 required")


def sign_orbit_log_probabilities(cache: Any, spec: GaussianMixtureShiftSpec,
                                 z: torch.Tensor) -> torch.Tensor:
    """Columns (++),(-+),(+-),(--), with magnitudes from the original z.

    One full cached extraction and four scalar CDFs per context; q is unchanged.
    """
    require_blind_residuals(spec)
    _coordinates(z, 64)
    p = cache.problem
    if p.steps != 32 or cache.kernel.local_cholesky[0, 1] != 0:
        raise ValueError("fixed triangular N32 hybrid law required")
    base = z.clone()
    base[:, [57, 61, 63]] = 0
    _, _, variance = cache.quantities(base)
    v = variance[:, :-1]
    dw = (base.reshape(-1, 32, 2) @ cache.kernel.local_cholesky.T)[:, :, 0]
    keep = torch.ones(32, dtype=torch.bool)
    keep[[29, 31]] = False
    terminal = math.log(p.spot)+(-.5*p.step_dt*v[:, keep]
                                +p.rho*v[:, keep].sqrt()*dw[:, keep]).sum(1)
    integrated = p.step_dt*v[:, keep].sum(1)
    beta = p.eta*math.sqrt(2*p.hurst)*float(cache.kernel.local_cholesky[1, 1])
    signs = np.array([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        altered = np.exp(v[:, [29, 31]].log().numpy()[:, None, :]
                         +beta*z[:, [57, 61]].numpy()[:, None, :]*signs[None, :, :])
        integrated_all = integrated.numpy()[:, None]+p.step_dt*altered.sum(2)
        terminal_all = terminal.numpy()[:, None] + (
            -.5*p.step_dt*altered+p.rho*np.sqrt(altered)*dw[:, [29, 31]].numpy()[:, None, :]).sum(2)
        logs = np.asarray(log_ndtr((math.log(p.task.level)-terminal_all)
                                   /np.sqrt((1-p.rho**2)*integrated_all)))
    if not np.isfinite(logs).all() or (logs > 1e-12).any() or (altered <= 0).any():
        raise FloatingPointError("unresolved sign-orbit numerical range")
    return torch.from_numpy(logs)


def orbit_endpoints(log_g: torch.Tensor, log_d: torch.Tensor, *, design: str) -> dict[str, torch.Tensor]:
    """Event columns raw/s57/s61/joint; risk uses complete orbit symmetrization."""
    _coordinates(log_g, 4)
    if (log_d.shape != (len(log_g),) or log_d.dtype != torch.float64
            or log_d.device.type != "cpu" or log_d.requires_grad or not torch.isfinite(log_d).all()
            or design not in ("p", "q")):
        raise ValueError("finite density and declared p/q design required")
    g = log_g.numpy()
    s57a, s57b = logsumexp(g[:, [0, 1]], 1)-math.log(2), logsumexp(g[:, [2, 3]], 1)-math.log(2)
    s61a, s61b = logsumexp(g[:, [0, 2]], 1)-math.log(2), logsumexp(g[:, [1, 3]], 1)-math.log(2)
    joint = logsumexp(g, 1)-math.log(4)
    event = np.stack([g[:, 0], s57a, s61a, joint], 1)
    risk = np.stack([logsumexp(2*g, 1)-math.log(4),
        np.logaddexp(2*s57a, 2*s57b)-math.log(2),
        np.logaddexp(2*s61a, 2*s61b)-math.log(2), 2*joint], 1)
    density = log_d.numpy()[:, None]
    return {"event": torch.from_numpy(event-density if design == "q" else event),
            "risk": torch.from_numpy(risk-(2 if design == "q" else 1)*density)}
