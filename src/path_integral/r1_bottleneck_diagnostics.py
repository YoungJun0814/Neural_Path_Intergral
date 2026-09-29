"""Development-only, exact-density R1 ablations on the conditional 2N law.

Fitting weights may be normalized; final importance-sampling contributions never are.
No estimated nested floor in this module is a lower confidence bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


def _check_bank(samples: torch.Tensor, log_g: torch.Tensor, log_p_over_bank: torch.Tensor) -> None:
    n = samples.shape[0]
    if (samples.ndim != 2 or n < 2 or samples.dtype != torch.float64
            or samples.device.type != "cpu" or not torch.isfinite(samples).all()
            or log_g.shape != (n,) or log_p_over_bank.shape != (n,)
            or log_g.dtype != torch.float64 or log_p_over_bank.dtype != torch.float64
            or torch.isnan(log_g).any() or torch.isposinf(log_g).any()
            or not torch.isfinite(log_p_over_bank).all()
            or bool(torch.isneginf(log_g).all())):
        raise ValueError("invalid conditional training bank")


def _top_eigenvectors(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    if not 1 <= rank <= matrix.shape[0]:
        raise ValueError("rank exceeds the Gaussian dimension")
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    return vectors[:, torch.argsort(values, descending=True)[:rank]].contiguous()


def weighted_target_directions(
    samples: torch.Tensor, log_g: torch.Tensor, log_p_over_bank: torch.Tensor,
    *, rank: int, kind: str, gradients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weighted target-PCA, FIS-matrix, or risk-contribution PCA directions.

    ``fis_matrix`` uses E_{p*g/mu}[grad log(g) grad log(g)^T], the
    whitened-prior matrix in Uribe et al. (3.7). It does not implement their
    complete sequential iCEred training algorithm. ``risk_pca`` is only an
    exploratory second-moment-contribution heuristic, not an optimal projector.
    """

    _check_bank(samples, log_g, log_p_over_bank)
    if kind == "fis_matrix":
        if (gradients is None or gradients.shape != samples.shape
                or gradients.dtype != torch.float64 or not torch.isfinite(gradients).all()):
            raise ValueError("FIS requires finite full-bank log-payoff gradients")
        features = gradients
        weights = torch.softmax(log_g + log_p_over_bank, dim=0)
        matrix = features.T @ (weights[:, None] * features)
    elif kind in ("target_pca", "risk_pca"):
        features = samples
        log_weights = (log_g + log_p_over_bank if kind == "target_pca"
                       else 2.0 * (log_g + log_p_over_bank))
        weights = torch.softmax(log_weights, dim=0)
        # Uncentered target moment retains the mean-shift direction.
        matrix = features.T @ (weights[:, None] * features)
    else:
        raise ValueError("unknown direction construction")
    return _top_eigenvectors(matrix, rank)


def conditional_log_payoff_gradients(
    problem: RBergomiBaselineProblem, samples: torch.Tensor, *, batch_size: int = 128,
) -> torch.Tensor:
    """Autodifferentiate the exact conditional log CDF; cost is part of fitting."""

    if samples.ndim != 2 or samples.shape[1] != problem.local_dimension:
        raise ValueError("gradient samples have the wrong shape")
    blocks = []
    for block in samples.split(batch_size):
        x = block.detach().clone().requires_grad_(True)
        log_g = evaluate_rbergomi_conditional_terminal(problem, x).payoffs.log_left_probability
        gradient = torch.autograd.grad(log_g.sum(), x)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite conditional log-payoff gradient")
        blocks.append(gradient.detach())
    return torch.cat(blocks, dim=0)


def mean_shift_proposal(
    directions: torch.Tensor, coefficients: torch.Tensor, *, defensive_mass: float,
) -> DefensiveFiniteRankGaussianMixture:
    """Natural/shifted mixture with identity covariance in every direction."""

    if (directions.ndim != 2 or directions.dtype != torch.float64
            or directions.device.type != "cpu" or coefficients.shape != (directions.shape[1],)
            or not 0.0 < defensive_mass < 1.0):
        raise ValueError("invalid mean-shift proposal parameters")
    mean = directions @ coefficients
    return DefensiveFiniteRankGaussianMixture(
        (FiniteRankGaussianComponent.natural(directions.shape[0]),
         FiniteRankGaussianComponent(
             mean, torch.empty((directions.shape[0], 0), dtype=torch.float64),
             torch.empty(0, dtype=torch.float64))),
        torch.tensor((defensive_mass, 1.0 - defensive_mass), dtype=torch.float64),
    )


def fit_projected_mean_shift(
    samples: torch.Tensor, log_g: torch.Tensor, log_p_over_bank: torch.Tensor,
    directions: torch.Tensor, *, objective: str, defensive_mass: float = 0.1,
    steps: int = 80, learning_rate: float = 0.08, maximum_norm: float = 20.0,
) -> tuple[DefensiveFiniteRankGaussianMixture, float]:
    """Fit the *same* defensive identity-covariance family by KL or M2.

    Both empirical objectives use the same IID bank ``X~q_bank``. The M2
    objective is E_qbank[g^2 p^2/(q_bank q_theta)], not the in-sample
    variance of self-normalized or fitted contributions. Independent held-out
    evaluation is essential because either empirical objective can overfit.
    """

    _check_bank(samples, log_g, log_p_over_bank)
    if (directions.shape[0] != samples.shape[1] or directions.ndim != 2
            or not 1 <= directions.shape[1] <= samples.shape[1]
            or directions.dtype != torch.float64
            or float(torch.amax(torch.abs(directions.T @ directions - torch.eye(
                directions.shape[1], dtype=torch.float64)))) > 2e-10
            or objective not in ("kl", "m2") or steps < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0.0
            or not math.isfinite(maximum_norm) or maximum_norm <= 0.0):
        raise ValueError("invalid projected fit configuration")
    coordinates = samples @ directions
    target_weights = torch.softmax(log_g + log_p_over_bank, dim=0).detach()
    initial = (target_weights @ coordinates).detach()
    norm = float(torch.linalg.vector_norm(initial))
    if norm > maximum_norm:
        initial = initial * (maximum_norm / norm)
    coefficients = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam((coefficients,), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        log_shift = coordinates @ coefficients - 0.5 * torch.sum(coefficients.square())
        log_q_over_p = torch.logaddexp(
            math.log(defensive_mass) + torch.zeros_like(log_shift),
            math.log1p(-defensive_mass) + log_shift,
        )
        if objective == "kl":
            loss = -torch.sum(target_weights * log_q_over_p)
        else:
            loss = torch.logsumexp(
                2.0 * log_g + log_p_over_bank - log_q_over_p,
                dim=0,
            ) - math.log(samples.shape[0])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = torch.linalg.vector_norm(coefficients)
            if norm > maximum_norm:
                coefficients.mul_(maximum_norm / norm)
    # Report the loss of the *returned* proposal, not the pre-update value
    # from the final optimizer iteration.
    with torch.no_grad():
        final_log_shift = coordinates @ coefficients - 0.5 * torch.sum(coefficients.square())
        final_log_q_over_p = torch.logaddexp(
            math.log(defensive_mass) + torch.zeros_like(final_log_shift),
            math.log1p(-defensive_mass) + final_log_shift,
        )
        if objective == "kl":
            final_loss = -torch.sum(target_weights * final_log_q_over_p)
        else:
            final_loss = torch.logsumexp(
                2.0 * log_g + log_p_over_bank - final_log_q_over_p,
                dim=0,
            ) - math.log(samples.shape[0])
    if not torch.isfinite(coefficients).all() or not torch.isfinite(final_loss):
        raise FloatingPointError("nonfinite projected fit")
    return mean_shift_proposal(
        directions, coefficients.detach(), defensive_mass=defensive_mass,
    ), float(final_loss)


@dataclass(frozen=True)
class LogMomentSummary:
    count: int
    log_mean: float | None
    log_second_moment: float | None
    relative_se: float | None
    contribution_ess: float
    maximum_fraction: float


def summarize_log_contributions(log_values: torch.Tensor) -> LogMomentSummary:
    """Raw IID contribution statistics in a stable log representation."""

    if (log_values.ndim != 1 or log_values.numel() < 2
            or log_values.dtype != torch.float64 or torch.isnan(log_values).any()
            or torch.isposinf(log_values).any()):
        raise ValueError("invalid log contributions")
    n = log_values.numel()
    log_sum = torch.logsumexp(log_values, dim=0)
    if not torch.isfinite(log_sum):
        return LogMomentSummary(n, None, None, None, 0.0, 0.0)
    log_mean = float(log_sum - math.log(n))
    log_m2 = float(torch.logsumexp(2.0 * log_values, dim=0) - math.log(n))
    log_ratio = log_m2 - 2.0 * log_mean
    squared_cv = math.expm1(log_ratio) if log_ratio < 700 else math.inf
    return LogMomentSummary(
        count=n, log_mean=log_mean, log_second_moment=log_m2,
        relative_se=math.sqrt(max(0.0, squared_cv) / (n - 1)),
        contribution_ess=(n / math.exp(log_ratio) if log_ratio < 700 else 0.0),
        maximum_fraction=float(torch.exp(torch.max(log_values) - log_sum)),
    )


def nested_reference_complement(
    problem: RBergomiBaselineProblem, directions: torch.Tensor, *,
    outer_count: int, inner_counts: tuple[int, ...], shift: torch.Tensor,
    outer_seed: int, inner_seed: int,
) -> dict[str, object]:
    """Nested plug-in diagnostic for q(U,V)=q_U(U) phi(V), not a certificate.

    All inner sizes use a common maximum-size bank but only independent
    *complement* Gaussian draws. U uses a known shifted normal density, with
    its exact phi_U/G_U weight. Distinct outer and inner seeds are required.
    """

    if directions.shape[0] != problem.local_dimension:
        raise ValueError("nested dimension does not match the problem")
    return nested_reference_complement_generic(
        directions, log_payoff=lambda x: evaluate_rbergomi_conditional_terminal(
            problem, x,
        ).payoffs.log_left_probability,
        outer_count=outer_count, inner_counts=inner_counts, shift=shift,
        outer_seed=outer_seed, inner_seed=inner_seed,
    )


def nested_reference_complement_generic(
    directions: torch.Tensor, *, log_payoff: Callable[[torch.Tensor], torch.Tensor],
    outer_count: int, inner_counts: tuple[int, ...], shift: torch.Tensor,
    outer_seed: int, inner_seed: int,
) -> dict[str, object]:
    """The Gaussian U/complement diagnostic with a testable generic payoff."""

    d, rank = directions.shape
    if (not 1 <= rank <= d
            or directions.dtype != torch.float64 or shift.shape != (rank,)
            or shift.dtype != torch.float64 or outer_count < 2
            or not inner_counts or any(x < 2 for x in inner_counts)
            or tuple(sorted(set(inner_counts))) != inner_counts
            or outer_seed == inner_seed
            or float(torch.amax(torch.abs(directions.T @ directions - torch.eye(
                rank, dtype=torch.float64)))) > 2e-10):
        raise ValueError("invalid nested conditional diagnostic")
    outer_gen = torch.Generator().manual_seed(outer_seed)
    inner_gen = torch.Generator().manual_seed(inner_seed)
    u = shift + torch.randn((outer_count, rank), dtype=torch.float64, generator=outer_gen)
    log_phi_over_g = -u @ shift + 0.5 * torch.sum(shift.square())
    max_inner = inner_counts[-1]
    noise = torch.randn(
        (outer_count, max_inner, d), dtype=torch.float64, generator=inner_gen,
    )
    noise = noise - (noise @ directions) @ directions.T
    samples = (u @ directions.T)[:, None, :] + noise
    log_g = log_payoff(samples.reshape(-1, d)).reshape(outer_count, max_inner)
    if torch.isnan(log_g).any() or torch.isposinf(log_g).any():
        raise FloatingPointError("invalid nested log payoff")
    results: dict[str, object] = {}
    for inner in inner_counts:
        selected = log_g[:, :inner]
        log_m1 = torch.logsumexp(selected, dim=1) - math.log(inner)
        log_m2 = torch.logsumexp(2.0 * selected, dim=1) - math.log(inner)
        log_mu = float(torch.logsumexp(log_phi_over_g + log_m1, dim=0) - math.log(outer_count))
        log_floor = float(2.0 * (torch.logsumexp(
            log_phi_over_g + 0.5 * log_m2, dim=0,
        ) - math.log(outer_count)))
        log_reference_m2 = float(torch.logsumexp(
            log_phi_over_g + log_m2, dim=0,
        ) - math.log(outer_count))
        results[str(inner)] = {
            "log_mean_plugin": log_mu,
            "log_floor_plugin": log_floor,
            "log_reference_m2_plugin": log_reference_m2,
            "outer_max_fraction_for_floor": float(torch.max(torch.softmax(
                log_phi_over_g + 0.5 * log_m2, dim=0,
            ))),
            "inner_count": inner,
        }
    return {
        "outer_count": outer_count,
        "rank": rank,
        "outer_shift_norm": float(torch.linalg.vector_norm(shift)),
        "inner_sweep": results,
        "interpretation": "biased_nested_plugin_not_a_lower_confidence_bound",
    }
