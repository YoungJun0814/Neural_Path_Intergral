"""Exact balance mixtures of frozen defensive residual coupling flows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from src.path_integral.baseline_framework import BaselineCostLedger
from src.path_integral.low_rank_residual_flow import (
    FrozenLowRankResidualFlow,
    low_rank_flow_log_q_over_p,
    sample_low_rank_residual_flow,
)


def _derived_seed(root: int, role: str) -> int:
    if isinstance(root, bool) or not isinstance(root, int) or root < 0:
        raise ValueError("mixture root seed must be a nonnegative integer")
    digest = hashlib.sha256(f"NPI-V14-TEMPERED-MIXTURE\0{root}\0{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


@dataclass(frozen=True)
class TemperedFlowComponent:
    target_power: float
    mixture_weight: float
    flow: FrozenLowRankResidualFlow

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_power) or not 0.0 < self.target_power <= 1.0:
            raise ValueError("tempered target power must lie in (0, 1]")
        if not math.isfinite(self.mixture_weight) or self.mixture_weight <= 0.0:
            raise ValueError("tempered mixture weight must be finite and positive")
        if not self.flow.exact_likelihood or self.flow.self_normalized or not self.flow.frozen:
            raise ValueError("tempered component must be a frozen exact ordinary flow")


@dataclass(frozen=True)
class FrozenTemperedResidualFlowMixture:
    schema: str
    task_id: str
    direction: tuple[float, ...]
    components: tuple[TemperedFlowComponent, ...]
    training_seed: int
    training_cost: BaselineCostLedger
    exact_likelihood: bool
    self_normalized: bool
    frozen: bool
    sha256: str

    @property
    def effective_defensive_weight(self) -> float:
        return sum(
            component.mixture_weight * component.flow.defensive_weight
            for component in self.components
        )

    @property
    def likelihood_bound(self) -> float:
        return 1.0 / self.effective_defensive_weight


def freeze_tempered_residual_flow_mixture(
    *,
    task_id: str,
    components: tuple[TemperedFlowComponent, ...],
    training_seed: int,
    training_cost: BaselineCostLedger,
) -> FrozenTemperedResidualFlowMixture:
    if not task_id.strip() or not components:
        raise ValueError("tempered mixture identity is invalid")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("tempered mixture seed must be a nonnegative integer")
    direction = components[0].flow.direction
    if any(component.flow.task_id != task_id for component in components):
        raise ValueError("tempered component task IDs differ")
    if any(component.flow.direction != direction for component in components):
        raise ValueError("tempered components use different residual directions")
    powers = tuple(component.target_power for component in components)
    if any(left >= right for left, right in zip(powers, powers[1:], strict=False)):
        raise ValueError("tempered target powers must be strictly increasing")
    total = sum(component.mixture_weight for component in components)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("tempered mixture weights must sum to one")
    payload = {
        "schema": "npi.g11.v14-frozen-tempered-residual-flow-mixture.v1",
        "task_id": task_id,
        "direction": direction,
        "components": [asdict(component) for component in components],
        "training_seed": training_seed,
        "training_cost": asdict(training_cost),
        "exact_likelihood": True,
        "self_normalized": False,
        "frozen": True,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return FrozenTemperedResidualFlowMixture(
        schema="npi.g11.v14-frozen-tempered-residual-flow-mixture.v1",
        task_id=task_id,
        direction=direction,
        components=components,
        training_seed=training_seed,
        training_cost=training_cost,
        exact_likelihood=True,
        self_normalized=False,
        frozen=True,
        sha256=digest,
    )


def tempered_mixture_log_q_over_p(
    residual: torch.Tensor, proposal: FrozenTemperedResidualFlowMixture
) -> torch.Tensor:
    terms = [
        math.log(component.mixture_weight) + low_rank_flow_log_q_over_p(residual, component.flow)
        for component in proposal.components
    ]
    return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


@dataclass(frozen=True)
class TemperedResidualFlowMixtureSample:
    residual: torch.Tensor
    component_labels: torch.Tensor
    inner_flow_labels: torch.Tensor
    likelihood: torch.Tensor
    used_seeds: tuple[int, ...]
    maximum_projection_error: float
    maximum_likelihood_bound_violation: float


def sample_tempered_residual_flow_mixture(
    proposal: FrozenTemperedResidualFlowMixture,
    sample_count: int,
    *,
    root_seed: int,
) -> TemperedResidualFlowMixtureSample:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("tempered mixture sample count must be positive")
    if not proposal.exact_likelihood or proposal.self_normalized or not proposal.frozen:
        raise ValueError("tempered sampling requires a frozen exact ordinary proposal")
    label_seed = _derived_seed(root_seed, "component-labels")
    weights = torch.tensor(
        [component.mixture_weight for component in proposal.components],
        dtype=torch.float64,
    )
    labels = torch.multinomial(
        weights,
        sample_count,
        replacement=True,
        generator=torch.Generator().manual_seed(label_seed),
    )
    dimension = len(proposal.direction)
    residual = torch.empty((sample_count, dimension), dtype=torch.float64)
    inner = torch.empty(sample_count, dtype=torch.bool)
    used_seeds = [label_seed]
    for index, component in enumerate(proposal.components):
        mask = labels == index
        count = int(torch.count_nonzero(mask))
        if count == 0:
            continue
        gaussian_seed = _derived_seed(root_seed, f"component-{index}-gaussian")
        inner_seed = _derived_seed(root_seed, f"component-{index}-inner-label")
        if gaussian_seed in used_seeds or inner_seed in used_seeds or gaussian_seed == inner_seed:
            raise AssertionError("tempered mixture seed streams collided")
        used_seeds.extend((gaussian_seed, inner_seed))
        sampled = sample_low_rank_residual_flow(
            component.flow,
            count,
            gaussian_seed=gaussian_seed,
            label_seed=inner_seed,
        )
        residual[mask] = sampled.residual
        inner[mask] = sampled.labels
    log_q_over_p = tempered_mixture_log_q_over_p(residual, proposal)
    likelihood = torch.exp(-log_q_over_p)
    violation = max(0.0, float(torch.amax(likelihood)) - proposal.likelihood_bound)
    direction = torch.tensor(proposal.direction, dtype=torch.float64)
    projection = float(torch.amax(torch.abs(residual @ direction)))
    if not torch.isfinite(likelihood).all() or violation > 1e-9:
        raise FloatingPointError("tempered mixture likelihood is invalid")
    return TemperedResidualFlowMixtureSample(
        residual=residual,
        component_labels=labels,
        inner_flow_labels=inner,
        likelihood=likelihood,
        used_seeds=tuple(used_seeds),
        maximum_projection_error=projection,
        maximum_likelihood_bound_violation=violation,
    )
