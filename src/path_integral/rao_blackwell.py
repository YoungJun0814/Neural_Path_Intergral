"""Finite-sample diagnostics for exact Rao--Blackwellized estimator pairs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RaoBlackwellPairDiagnostics:
    """Moment diagnostics for raw ``Y`` and conditional estimator ``D``."""

    count: int
    raw_mean: float
    dcs_mean: float
    residual_mean: float
    residual_standard_error: float
    raw_variance: float
    dcs_variance: float
    residual_variance: float
    dcs_residual_covariance: float
    dcs_residual_covariance_product_variance: float
    dcs_residual_covariance_standard_error: float
    dcs_residual_covariance_z_score: float
    dcs_residual_correlation: float | None
    raw_over_dcs_variance_ratio: float
    variance_decomposition_error: float


def rao_blackwell_pair_diagnostics(
    raw_values: torch.Tensor,
    dcs_values: torch.Tensor,
) -> RaoBlackwellPairDiagnostics:
    """Summarize a common-random-number raw/DCS diagnostic batch.

    If ``D = E[Y | R]`` exactly, then the population residual ``Y-D`` has zero
    mean, is orthogonal to ``D``, and ``Var(Y) = Var(D) + Var(Y-D)``.  This
    function measures those implications; it does not promote a finite sample
    agreement into a proof of the conditional-expectation identity.
    """

    if (
        raw_values.ndim != 1
        or dcs_values.shape != raw_values.shape
        or raw_values.numel() < 2
        or raw_values.dtype != torch.float64
        or dcs_values.dtype != torch.float64
        or raw_values.device != dcs_values.device
        or not torch.isfinite(raw_values).all()
        or not torch.isfinite(dcs_values).all()
    ):
        raise ValueError("raw and DCS values must be matching finite float64 vectors")
    residual = raw_values - dcs_values
    count = int(raw_values.numel())
    raw_mean = float(torch.mean(raw_values))
    dcs_mean = float(torch.mean(dcs_values))
    residual_mean = float(torch.mean(residual))
    raw_variance = float(torch.var(raw_values, unbiased=True))
    dcs_variance = float(torch.var(dcs_values, unbiased=True))
    residual_variance = float(torch.var(residual, unbiased=True))
    centered_dcs = dcs_values - dcs_mean
    centered_residual = residual - residual_mean
    covariance_products = centered_dcs * centered_residual
    covariance = float(torch.sum(covariance_products) / (count - 1))
    covariance_product_variance = float(
        torch.var(covariance_products, unbiased=True)
    )
    covariance_standard_error = float(
        math.sqrt(covariance_product_variance / count)
        * count
        / (count - 1)
    )
    covariance_z_score = (
        0.0
        if covariance_standard_error == 0.0 and covariance == 0.0
        else math.inf
        if covariance_standard_error == 0.0
        else covariance / covariance_standard_error
    )
    scale = math.sqrt(dcs_variance * residual_variance)
    correlation = None if scale == 0.0 else covariance / scale
    variance_ratio = (
        math.inf if dcs_variance == 0.0 and raw_variance > 0.0
        else 1.0 if dcs_variance == 0.0
        else raw_variance / dcs_variance
    )
    decomposition_error = (
        raw_variance
        - dcs_variance
        - residual_variance
        - 2.0 * covariance
    )
    return RaoBlackwellPairDiagnostics(
        count=count,
        raw_mean=raw_mean,
        dcs_mean=dcs_mean,
        residual_mean=residual_mean,
        residual_standard_error=math.sqrt(residual_variance / count),
        raw_variance=raw_variance,
        dcs_variance=dcs_variance,
        residual_variance=residual_variance,
        dcs_residual_covariance=covariance,
        dcs_residual_covariance_product_variance=(
            covariance_product_variance
        ),
        dcs_residual_covariance_standard_error=covariance_standard_error,
        dcs_residual_covariance_z_score=covariance_z_score,
        dcs_residual_correlation=correlation,
        raw_over_dcs_variance_ratio=variance_ratio,
        variance_decomposition_error=decomposition_error,
    )
