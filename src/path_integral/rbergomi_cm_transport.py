"""End-to-end training and ordinary-IS evaluation of the V15 CM transport."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import (
    CameronMartinBasis,
    build_blp_cameron_martin_basis,
)
from src.path_integral.cameron_martin_modes import (
    ModeSearchConfig,
    RBergomiModeSearchResult,
    find_rbergomi_conditional_modes,
)
from src.path_integral.finite_rank_gaussian_transport import (
    CurvatureTransportConfig,
    DefensiveFiniteRankGaussianMixture,
    build_curvature_transport,
)
from src.path_integral.provenance import process_peak_resident_memory_bytes
from src.path_integral.volterra_action import RBergomiConditionalAction
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


@dataclass(frozen=True)
class RBergomiCMTransportTrainingResult:
    proposal: DefensiveFiniteRankGaussianMixture
    proposal_sha256: str
    basis: CameronMartinBasis
    action: RBergomiConditionalAction
    modes: RBergomiModeSearchResult
    training_cost: BaselineCostLedger


@dataclass(frozen=True)
class RBergomiCMTransportEvaluation:
    contribution: torch.Tensor
    conditional_probability: torch.Tensor
    likelihood: torch.Tensor
    component_labels: torch.Tensor
    log_q_over_p: torch.Tensor
    evaluation_cost: BaselineCostLedger
    maximum_likelihood_bound_violation: float
    likelihood_normalization_mean: float


def finite_rank_transport_sha256(proposal: DefensiveFiniteRankGaussianMixture) -> str:
    payload = {
        "schema": "npi.g11.v15-finite-rank-transport.v1",
        "weights": [float(value) for value in proposal.weights],
        "components": [
            {
                "mean": [float(value) for value in component.mean],
                "directions": component.directions.tolist(),
                "variance_eigenvalues": [
                    float(value) for value in component.variance_eigenvalues
                ],
            }
            for component in proposal.components
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def train_rbergomi_cm_transport(
    problem: RBergomiBaselineProblem,
    *,
    epsilon: float = 1.0,
    modes_per_driver: int = 8,
    basis: CameronMartinBasis | None = None,
    mode_search: ModeSearchConfig | None = None,
    transport_config: CurvatureTransportConfig | None = None,
) -> RBergomiCMTransportTrainingResult:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    if basis is None:
        basis = build_blp_cameron_martin_basis(
            steps=problem.steps,
            modes_per_driver=modes_per_driver,
        )
    elif basis.dimension != problem.local_dimension:
        raise ValueError("supplied basis dimension does not match the problem")
    action = RBergomiConditionalAction(
        problem=problem,
        basis=basis,
        epsilon=epsilon,
    )
    modes = find_rbergomi_conditional_modes(action, config=mode_search)
    if not modes.modes:
        raise RuntimeError("V15 action search found no converged mode")
    proposal = build_curvature_transport(action, modes, config=transport_config)
    attempts = len(modes.raw.attempts)
    iterations = sum(item.iterations for item in modes.raw.attempts)
    evaluations = sum(item.function_evaluations for item in modes.raw.attempts)
    work = evaluations * (problem.local_dimension + problem.steps) + iterations * basis.rank**2
    cost = BaselineCostLedger(
        optimizer_steps=iterations,
        hyperparameter_trials=attempts,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return RBergomiCMTransportTrainingResult(
        proposal=proposal,
        proposal_sha256=finite_rank_transport_sha256(proposal),
        basis=basis,
        action=action,
        modes=modes,
        training_cost=cost,
    )


def evaluate_rbergomi_cm_transport(
    problem: RBergomiBaselineProblem,
    proposal: DefensiveFiniteRankGaussianMixture,
    *,
    sample_count: int,
    path_seed: int,
    label_seed: int,
    epsilon: float = 1.0,
) -> RBergomiCMTransportEvaluation:
    if proposal.dimension != problem.local_dimension:
        raise ValueError("V15 proposal dimension does not match the problem")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    sample = proposal.sample(
        sample_count,
        path_seed=path_seed,
        label_seed=label_seed,
    )
    conditional = evaluate_rbergomi_conditional_terminal(
        problem,
        sample.samples,
        epsilon=epsilon,
    )
    likelihood = torch.exp(sample.log_p_over_q)
    contribution = conditional.payoffs.left_probability * likelihood
    if not torch.isfinite(contribution).all():
        raise FloatingPointError("V15 ordinary-IS contribution became nonfinite")
    bound = 1.0 / proposal.defensive_mass
    violation = max(0.0, float(torch.max(likelihood)) - bound)
    components = len(proposal.components)
    rank_work = sum(component.rank for component in proposal.components)
    work = sample_count * (
        problem.local_dimension
        + problem.steps
        + components * problem.local_dimension
        + rank_work
        + 1
    )
    cost = BaselineCostLedger(
        final_samples=sample_count,
        likelihood_evaluations=sample_count,
        cdf_calls=sample_count,
        algorithmic_work_units=float(work),
        wall_seconds=time.perf_counter() - started_wall,
        cpu_seconds=time.process_time() - started_cpu,
        peak_memory_bytes=process_peak_resident_memory_bytes(),
        measurement_mode="standardized_hardware_wall",
    )
    return RBergomiCMTransportEvaluation(
        contribution=contribution,
        conditional_probability=conditional.payoffs.left_probability,
        likelihood=likelihood,
        component_labels=sample.labels,
        log_q_over_p=sample.log_q_over_p,
        evaluation_cost=cost,
        maximum_likelihood_bound_violation=violation,
        likelihood_normalization_mean=float(torch.mean(likelihood)),
    )


def assert_transport_unchanged(
    proposal: DefensiveFiniteRankGaussianMixture,
    expected_sha256: str,
) -> None:
    actual = finite_rank_transport_sha256(proposal)
    if actual != expected_sha256:
        raise RuntimeError("V15 proposal mutated after freeze")
