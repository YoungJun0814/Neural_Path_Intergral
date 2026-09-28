"""Weighted cross-entropy on the 2N conditional rBergomi Gaussian law."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.research_cost_accounting import measure_stage
from src.path_integral.research_result_contract import StageCost, canonical_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


@dataclass(frozen=True)
class WeightedConditionalCEConfig:
    iterations: int = 5
    samples_per_iteration: int = 2048
    elite_fraction: float = 0.1
    smoothing: float = 0.7
    defensive_mass: float = 0.1
    maximum_mean_norm: float = 20.0
    covariance_rank: int = 0
    covariance_shrinkage: float = 0.5
    minimum_eigenvalue: float = 0.25
    maximum_eigenvalue: float = 4.0
    minimum_target_ess: float = 20.0

    def __post_init__(self) -> None:
        if any(
            isinstance(x, bool) or not isinstance(x, int) or x < 1
            for x in (self.iterations, self.samples_per_iteration)
        ):
            raise ValueError("iteration and sample counts must be positive integers")
        if isinstance(self.covariance_rank, bool) or not isinstance(self.covariance_rank, int) or self.covariance_rank < 0:
            raise ValueError("covariance_rank must be a nonnegative integer")
        if not (0 < self.elite_fraction < 1 and 0 < self.smoothing <= 1):
            raise ValueError("invalid elite fraction or smoothing")
        if not 0 < self.defensive_mass < 1:
            raise ValueError("defensive_mass must be in (0, 1)")
        if not math.isfinite(self.maximum_mean_norm) or self.maximum_mean_norm <= 0:
            raise ValueError("maximum_mean_norm must be finite and positive")
        if not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if (
            not math.isfinite(self.minimum_eigenvalue)
            or not math.isfinite(self.maximum_eigenvalue)
            or not 0 < self.minimum_eigenvalue <= 1 <= self.maximum_eigenvalue
        ):
            raise ValueError("eigenvalue bounds must contain one")
        if not math.isfinite(self.minimum_target_ess) or self.minimum_target_ess <= 0:
            raise ValueError("minimum_target_ess must be finite and positive")


@dataclass(frozen=True)
class WeightedConditionalCEFit:
    proposal: DefensiveFiniteRankGaussianMixture
    proposal_digest: str
    cost: StageCost
    seed_ledger: dict[str, object]
    target_ess_history: tuple[float, ...]
    fitting_ess_history: tuple[float, ...]
    elite_threshold_history: tuple[float, ...]
    target_reached: bool


def _proposal(mean: torch.Tensor, directions: torch.Tensor, eigenvalues: torch.Tensor, mass: float) -> DefensiveFiniteRankGaussianMixture:
    learned = FiniteRankGaussianComponent(mean, directions, eigenvalues)
    return DefensiveFiniteRankGaussianMixture(
        components=(FiniteRankGaussianComponent.natural(mean.numel()), learned),
        weights=torch.tensor((mass, 1.0 - mass), dtype=torch.float64),
    )


def proposal_parameters(proposal: DefensiveFiniteRankGaussianMixture) -> dict[str, object]:
    return {
        "weights": proposal.weights.tolist(),
        "components": [
            {
                "mean": component.mean.tolist(),
                "directions": component.directions.tolist(),
                "eigenvalues": component.variance_eigenvalues.tolist(),
            }
            for component in proposal.components
        ],
    }


def weighted_gaussian_moments(
    samples: torch.Tensor, log_target_over_proposal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Weighted target fit. Normalization here is training, not final SNIS."""

    if (
        samples.ndim != 2
        or samples.shape[0] < 2
        or samples.dtype != torch.float64
        or samples.device.type != "cpu"
        or log_target_over_proposal.shape != (samples.shape[0],)
        or log_target_over_proposal.dtype != torch.float64
        or log_target_over_proposal.device.type != "cpu"
        or not torch.isfinite(samples).all()
        or torch.isnan(log_target_over_proposal).any()
        or torch.isposinf(log_target_over_proposal).any()
    ):
        raise ValueError("invalid samples or log CE weights")
    if bool(torch.isneginf(log_target_over_proposal).all()):
        raise ValueError("all CE weights vanish")
    weights = torch.softmax(log_target_over_proposal, dim=0)
    mean = weights @ samples
    centered = samples - mean
    covariance = centered.T @ (weights[:, None] * centered)
    ess = 1.0 / float(torch.sum(weights.square()))
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise FloatingPointError("weighted CE moments are nonfinite")
    return mean, covariance, ess


def train_weighted_conditional_ce(
    problem: RBergomiBaselineProblem,
    *,
    training_seed: int,
    config: WeightedConditionalCEConfig | None = None,
) -> WeightedConditionalCEFit:
    if not isinstance(problem.task, TerminalThresholdTask):
        raise ValueError("conditional CE requires a terminal threshold task")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training_seed must be a nonnegative integer")
    config = config or WeightedConditionalCEConfig()
    d = problem.local_dimension
    if config.covariance_rank > d:
        raise ValueError("covariance rank exceeds local dimension")

    ledger = SeedLedger()
    history_target_ess: list[float] = []
    history_fitting_ess: list[float] = []
    history_level: list[float] = []
    mean = torch.zeros(d, dtype=torch.float64)
    directions = torch.empty((d, 0), dtype=torch.float64)
    eigenvalues = torch.empty(0, dtype=torch.float64)
    reached = False

    def fit() -> DefensiveFiniteRankGaussianMixture:
        nonlocal mean, directions, eigenvalues, reached
        proposal = _proposal(mean, directions, eigenvalues, config.defensive_mass)
        for iteration in range(config.iterations):
            path_seed = ledger.allocate(
                SeedKey("post-audit-ce", "training", problem.task_id, problem.task_id, iteration, training_seed, "path")
            )
            label_seed = ledger.allocate(
                SeedKey("post-audit-ce", "training", problem.task_id, problem.task_id, iteration, training_seed, "label")
            )
            draw = proposal.sample(
                config.samples_per_iteration, path_seed=path_seed, label_seed=label_seed
            )
            conditional = evaluate_rbergomi_conditional_terminal(problem, draw.samples)
            log_g = conditional.payoffs.log_left_probability
            log_weight = log_g + draw.log_p_over_q
            _, _, target_ess = weighted_gaussian_moments(draw.samples, log_weight)
            score = conditional.payoffs.standardized_left_threshold
            elite_count = max(2, math.ceil(config.elite_fraction * config.samples_per_iteration))
            threshold = float(torch.topk(score, elite_count).values[-1])
            reached = target_ess >= config.minimum_target_ess
            if reached:
                fitted, covariance, ess = weighted_gaussian_moments(draw.samples, log_weight)
            else:
                selected = score >= threshold
                elite_weight = torch.where(
                    selected, draw.log_p_over_q, torch.full_like(draw.log_p_over_q, -torch.inf)
                )
                fitted, covariance, ess = weighted_gaussian_moments(draw.samples, elite_weight)
            history_target_ess.append(target_ess)
            history_fitting_ess.append(ess)
            history_level.append(threshold)
            mean = (1.0 - config.smoothing) * mean + config.smoothing * fitted
            norm = float(torch.linalg.vector_norm(mean))
            if norm > config.maximum_mean_norm:
                mean = mean * (config.maximum_mean_norm / norm)
            if config.covariance_rank:
                covariance = (1.0 - config.covariance_shrinkage) * torch.eye(d, dtype=torch.float64) + config.covariance_shrinkage * covariance
                spectrum, basis = torch.linalg.eigh(covariance)
                indices = torch.argsort(torch.abs(spectrum - 1.0), descending=True)[: config.covariance_rank]
                directions = basis[:, indices]
                eigenvalues = torch.clamp(
                    spectrum[indices], config.minimum_eigenvalue, config.maximum_eigenvalue
                )
            proposal = _proposal(mean, directions, eigenvalues, config.defensive_mass)
        return proposal

    proposal, cost = measure_stage("fit", fit)
    cost = StageCost(
        "fit", cost.wall_seconds, cost.cpu_seconds, cost.peak_memory_bytes,
        config.iterations * config.samples_per_iteration * (2 * d + problem.steps),
    )
    return WeightedConditionalCEFit(
        proposal, canonical_digest(proposal_parameters(proposal)), cost, ledger.to_dict(),
        tuple(history_target_ess), tuple(history_fitting_ess),
        tuple(history_level), reached,
    )


def evaluate_weighted_conditional_ce(
    problem: RBergomiBaselineProblem,
    proposal: DefensiveFiniteRankGaussianMixture,
    *,
    sample_count: int,
    path_seed: int,
    label_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, StageCost]:
    """Return ordinary IS contributions and p/q normalization diagnostics."""

    if not isinstance(problem.task, TerminalThresholdTask) or proposal.dimension != problem.local_dimension:
        raise ValueError("proposal does not match terminal conditional problem")

    def evaluate() -> tuple[torch.Tensor, torch.Tensor]:
        draw = proposal.sample(sample_count, path_seed=path_seed, label_seed=label_seed)
        conditional = evaluate_rbergomi_conditional_terminal(problem, draw.samples)
        log_contribution = conditional.payoffs.log_left_probability + draw.log_p_over_q
        contributions = torch.exp(log_contribution)
        normalization = torch.exp(draw.log_p_over_q)
        if not torch.isfinite(contributions).all() or not torch.isfinite(normalization).all():
            raise FloatingPointError("nonfinite conditional IS contribution")
        return contributions, normalization

    (contributions, normalization), cost = measure_stage("inference", evaluate)
    return contributions, normalization, StageCost(
        "inference", cost.wall_seconds, cost.cpu_seconds, cost.peak_memory_bytes,
        sample_count * (2 * problem.local_dimension + problem.steps),
    )
