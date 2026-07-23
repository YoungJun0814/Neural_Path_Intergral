"""Falsification diagnostics for the conditional V6 rBergomi rate theorem.

These summaries test whether implemented arrays satisfy the theorem's geometry and
whether proposed moment assumptions are empirically plausible.  They deliberately do
not label empirical moments or fitted behavior as proofs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.rbergomi_threshold_diagnostics import (
    RBergomiThresholdCouplingDiagnostics,
)


@dataclass(frozen=True)
class DirectionRegularityDiagnostics:
    fine_steps: int
    l2_norm: float
    minimum_weight: float
    maximum_weight: float
    l1_mass: float
    sqrt_steps_maximum_weight: float
    inverse_sqrt_steps_l1_mass: float
    coarse_aggregation_error: float | None
    positive: bool
    unit_normalized: bool
    coarse_consistent: bool | None


@dataclass(frozen=True)
class SlopeLowerTailDiagnostics:
    sample_count: int
    minimum_slope: float
    quantiles: tuple[tuple[float, float], ...]
    inverse_moments: tuple[tuple[float, float], ...]
    lower_tail_probabilities: tuple[tuple[float, float], ...]
    finite: bool


@dataclass(frozen=True)
class CoefficientMomentDiagnostics:
    sample_count: int
    mesh_size: float
    intercept_lp: tuple[tuple[float, float], ...]
    slope_lp: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class BarrierObligationDiagnostics:
    sample_count: int
    active_before_cutoff_fraction: float
    mesh_enrichment_l1: float
    mesh_enrichment_l2: float
    good_event_fraction: float
    maximum_exact_decomposition_violation: float


@dataclass(frozen=True)
class TerminalSlopeInverseMomentBound:
    """Explicit target-law bound for the inverse terminal affine slope moment."""

    steps: int
    order: float
    grid_scaled_l1_mass: float
    deterministic_slope_scale: float
    maximum_volterra_variance: float
    upper_bound: float


@dataclass(frozen=True)
class TerminalRateContract:
    """Conservative proof-candidate exponents for the strict terminal model."""

    hurst: float
    epsilon_margin: float
    coefficient_lp_exponent: float
    threshold_lp_exponent: float
    dcs_second_moment_exponent: float
    weak_bias_exponent: float
    fft_polynomial_cost_exponent: float
    fft_has_log_factor: bool
    mlmc_epsilon_polynomial_exponent: float
    barrier_included: bool
    proof_status: str
    journal_claim_ready: bool
    proof_scope: str


def terminal_rate_contract(
    hurst: float, *, epsilon_margin: float
) -> TerminalRateContract:
    """Return the proved conservative exponent ledger, not a fitted rate.

    The proof supports every exponent strictly below ``H``.  The explicit epsilon
    margin prevents a numerical configuration from silently claiming an endpoint
    uniformity that is absent from the compact-domain proof.
    """

    if not math.isfinite(hurst) or not 0.0 < hurst < 0.5:
        raise ValueError("Hurst parameter must lie in (0, 0.5)")
    if (
        not math.isfinite(epsilon_margin)
        or epsilon_margin <= 0.0
        or epsilon_margin >= hurst
    ):
        raise ValueError("epsilon margin must lie in (0, H)")
    rate = hurst - epsilon_margin
    second = 2.0 * rate
    return TerminalRateContract(
        hurst=hurst,
        epsilon_margin=epsilon_margin,
        coefficient_lp_exponent=rate,
        threshold_lp_exponent=rate,
        dcs_second_moment_exponent=second,
        weak_bias_exponent=rate,
        fft_polynomial_cost_exponent=1.0,
        fft_has_log_factor=True,
        mlmc_epsilon_polynomial_exponent=1.0 / rate,
        barrier_included=False,
        proof_status=(
            "self-contained proof candidate; independent mathematical review and "
            "clean-artifact reproduction remain required"
        ),
        journal_claim_ready=False,
        proof_scope=(
            "terminal strict-lognormal rBergomi under the declared compact parameter, "
            "positive-direction, deterministic-control, and exact adjacent-coupling assumptions"
        ),
    )


def direction_regularity_diagnostics(
    fine_direction: torch.Tensor,
    *,
    declared_coarse_weights: torch.Tensor | None = None,
    tolerance: float = 1e-12,
) -> DirectionRegularityDiagnostics:
    """Check the exact direction convention used by adjacent DCS smoothing."""

    direction = torch.as_tensor(fine_direction, dtype=torch.float64, device="cpu").reshape(-1)
    if direction.numel() < 2 or direction.numel() % 2:
        raise ValueError("fine direction must have a positive even number of entries")
    if not torch.isfinite(direction).all():
        raise ValueError("fine direction must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    steps = int(direction.numel())
    norm = float(torch.linalg.vector_norm(direction))
    minimum = float(torch.amin(direction))
    maximum = float(torch.amax(direction))
    l1 = float(torch.sum(torch.abs(direction)))
    positive = minimum > 0.0
    unit = abs(norm - 1.0) <= tolerance
    aggregation_error = None
    coarse_consistent = None
    if declared_coarse_weights is not None:
        coarse = torch.as_tensor(
            declared_coarse_weights, dtype=torch.float64, device="cpu"
        ).reshape(-1)
        if coarse.shape != (steps // 2,) or not torch.isfinite(coarse).all():
            raise ValueError("declared coarse weights must match paired fine steps")
        expected = direction.reshape(-1, 2).sum(dim=1)
        aggregation_error = float(torch.amax(torch.abs(expected - coarse)))
        coarse_consistent = aggregation_error <= tolerance
    return DirectionRegularityDiagnostics(
        fine_steps=steps,
        l2_norm=norm,
        minimum_weight=minimum,
        maximum_weight=maximum,
        l1_mass=l1,
        sqrt_steps_maximum_weight=math.sqrt(steps) * maximum,
        inverse_sqrt_steps_l1_mass=l1 / math.sqrt(steps),
        coarse_aggregation_error=aggregation_error,
        positive=positive,
        unit_normalized=unit,
        coarse_consistent=coarse_consistent,
    )


def slope_lower_tail_diagnostics(
    slopes: torch.Tensor,
    *,
    inverse_orders: tuple[float, ...] = (1.0, 2.0, 4.0),
    quantile_levels: tuple[float, ...] = (0.001, 0.01, 0.05, 0.5),
    lower_tail_floors: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
) -> SlopeLowerTailDiagnostics:
    """Summarize positive slopes without claiming a uniform inverse-moment proof."""

    sample = torch.as_tensor(slopes, dtype=torch.float64, device="cpu").reshape(-1)
    if sample.numel() < 2 or not torch.isfinite(sample).all() or bool((sample <= 0.0).any()):
        raise ValueError("slope sample must contain at least two finite positive values")
    if any(not math.isfinite(order) or order <= 0.0 for order in inverse_orders):
        raise ValueError("inverse orders must be finite and positive")
    if any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in quantile_levels):
        raise ValueError("quantile levels must lie in (0, 1)")
    if any(not math.isfinite(floor) or floor <= 0.0 for floor in lower_tail_floors):
        raise ValueError("lower-tail floors must be finite and positive")
    inverse = []
    finite = True
    for order in inverse_orders:
        values = torch.exp(-order * torch.log(sample))
        value = float(torch.mean(values))
        finite = finite and math.isfinite(value)
        inverse.append((order, value))
    return SlopeLowerTailDiagnostics(
        sample_count=int(sample.numel()),
        minimum_slope=float(torch.amin(sample)),
        quantiles=tuple(
            (level, float(torch.quantile(sample, level))) for level in quantile_levels
        ),
        inverse_moments=tuple(inverse),
        lower_tail_probabilities=tuple(
            (floor, float(torch.mean((sample <= floor).to(torch.float64))))
            for floor in lower_tail_floors
        ),
        finite=finite,
    )


def terminal_slope_inverse_moment_bound(
    direction: torch.Tensor,
    *,
    step_dt: float,
    maturity: float,
    hurst: float,
    xi: float,
    eta: float,
    rho: float,
    order: float,
    minimum_grid_scaled_l1_mass: float | None = None,
    grid_tolerance: float = 1e-12,
) -> TerminalSlopeInverseMomentBound:
    """Bound ``E[B_n^{-q}]`` for the terminal W2 smoothing slope.

    For positive unit direction ``u`` and target-law rBergomi variance,

    ``B_n=sqrt(1-rho^2)*sqrt(dt)*sum_i u_i*sqrt(V_i)``.

    Weighted Jensen and the Gaussian MGF give the returned finite bound.  The BLP
    cell-average construction has Volterra variance no larger than ``T^(2H)`` by
    Jensen on every historical kernel cell.  A uniform-in-grid theorem additionally
    requires the supplied grid-scaled L1 mass to have a positive common lower bound.
    """

    weights = torch.as_tensor(direction, dtype=torch.float64, device="cpu").reshape(-1)
    if weights.numel() < 1 or not torch.isfinite(weights).all():
        raise ValueError("direction must be a nonempty finite vector")
    if bool((weights <= 0.0).any()):
        raise ValueError("terminal inverse-moment theorem requires a positive direction")
    norm = float(torch.linalg.vector_norm(weights))
    if abs(norm - 1.0) > grid_tolerance:
        raise ValueError("terminal inverse-moment theorem requires a unit direction")
    numeric = (step_dt, maturity, hurst, xi, eta, rho, order, grid_tolerance)
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("terminal inverse-moment inputs must be finite")
    if step_dt <= 0.0 or maturity <= 0.0:
        raise ValueError("step_dt and maturity must be positive")
    if not 0.0 < hurst < 0.5:
        raise ValueError("rough Bergomi Hurst parameter must lie in (0, 0.5)")
    if xi <= 0.0 or eta < 0.0 or not -1.0 < rho < 1.0 or order <= 0.0:
        raise ValueError("invalid rBergomi or inverse-moment parameter")
    if grid_tolerance <= 0.0:
        raise ValueError("grid_tolerance must be positive")
    implied_maturity = weights.numel() * step_dt
    if abs(implied_maturity - maturity) > grid_tolerance * max(1.0, maturity):
        raise ValueError("direction length and step_dt do not match maturity")

    grid_mass = math.sqrt(step_dt) * float(torch.sum(weights))
    if minimum_grid_scaled_l1_mass is not None:
        if (
            not math.isfinite(minimum_grid_scaled_l1_mass)
            or minimum_grid_scaled_l1_mass <= 0.0
        ):
            raise ValueError("minimum grid-scaled L1 mass must be finite and positive")
        if grid_mass + grid_tolerance < minimum_grid_scaled_l1_mass:
            raise ValueError("direction violates the declared uniform L1-mass lower bound")
    maximum_variance = maturity ** (2.0 * hurst)
    deterministic_scale = math.sqrt(1.0 - rho**2) * math.sqrt(xi) * grid_mass
    log_bound = (
        -order * math.log(deterministic_scale)
        + eta**2 * maximum_variance * (order / 4.0 + order**2 / 8.0)
    )
    upper = math.exp(log_bound) if log_bound <= math.log(torch.finfo(torch.float64).max) else math.inf
    return TerminalSlopeInverseMomentBound(
        steps=int(weights.numel()),
        order=order,
        grid_scaled_l1_mass=grid_mass,
        deterministic_slope_scale=deterministic_scale,
        maximum_volterra_variance=maximum_variance,
        upper_bound=upper,
    )


def coefficient_moment_diagnostics(
    fine_intercept: torch.Tensor,
    fine_slope: torch.Tensor,
    coarse_intercept: torch.Tensor,
    coarse_slope: torch.Tensor,
    *,
    mesh_size: float,
    orders: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> CoefficientMomentDiagnostics:
    """Return raw terminal Lp errors for clustered, multimesh rate analysis."""

    tensors = tuple(
        torch.as_tensor(value, dtype=torch.float64, device="cpu").reshape(-1)
        for value in (fine_intercept, fine_slope, coarse_intercept, coarse_slope)
    )
    if len({tensor.shape for tensor in tensors}) != 1 or tensors[0].numel() < 2:
        raise ValueError("terminal coefficient arrays must be matching vectors")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("terminal coefficient arrays must be finite")
    if not math.isfinite(mesh_size) or mesh_size <= 0.0:
        raise ValueError("mesh_size must be finite and positive")
    if any(not math.isfinite(order) or order <= 0.0 for order in orders):
        raise ValueError("Lp orders must be finite and positive")
    intercept_error = torch.abs(tensors[0] - tensors[2])
    slope_error = torch.abs(tensors[1] - tensors[3])

    def lp(error: torch.Tensor, order: float) -> float:
        return float(torch.mean(error**order) ** (1.0 / order))

    return CoefficientMomentDiagnostics(
        sample_count=int(intercept_error.numel()),
        mesh_size=mesh_size,
        intercept_lp=tuple((order, lp(intercept_error, order)) for order in orders),
        slope_lp=tuple((order, lp(slope_error, order)) for order in orders),
    )


def barrier_obligation_diagnostics(
    diagnostics: RBergomiThresholdCouplingDiagnostics,
    *,
    active_time_cutoff: float,
) -> BarrierObligationDiagnostics:
    """Summarize barrier-only bad events and the fine-grid enrichment term."""

    if diagnostics.task_kind != "discrete_barrier_hit":
        raise ValueError("barrier obligations require discrete-barrier diagnostics")
    if not math.isfinite(active_time_cutoff) or active_time_cutoff <= 0.0:
        raise ValueError("active_time_cutoff must be finite and positive")
    finite = diagnostics.finite_threshold
    count = int(torch.sum(finite))
    if count < 1:
        raise ValueError("barrier obligation sample has no finite thresholds")
    active_early = diagnostics.fine_active_time[finite] < active_time_cutoff
    defect = diagnostics.mesh_enrichment_defect[finite]
    good = diagnostics.good_event[finite]
    return BarrierObligationDiagnostics(
        sample_count=count,
        active_before_cutoff_fraction=float(torch.mean(active_early.to(torch.float64))),
        mesh_enrichment_l1=float(torch.mean(torch.abs(defect))),
        mesh_enrichment_l2=float(torch.mean(defect**2)),
        good_event_fraction=float(torch.mean(good.to(torch.float64))),
        maximum_exact_decomposition_violation=diagnostics.maximum_exact_decomposition_violation,
    )
