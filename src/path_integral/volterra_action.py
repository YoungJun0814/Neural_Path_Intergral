"""Conditional finite-noise actions for Gaussian-Volterra terminal rare events."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import CameronMartinBasis
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)


@dataclass(frozen=True)
class ConditionalActionValue:
    value: torch.Tensor
    energy: torch.Tensor
    negative_scaled_log_probability: torch.Tensor
    log_probability: torch.Tensor
    control: torch.Tensor
    proposal_mean: torch.Tensor


@dataclass(frozen=True)
class RBergomiConditionalAction:
    """Callable coefficient-space action frozen to one problem and noise scale."""

    problem: RBergomiBaselineProblem
    basis: CameronMartinBasis
    epsilon: float = 1.0
    strike: float | None = None

    def __post_init__(self) -> None:
        if self.basis.dimension != self.problem.local_dimension:
            raise ValueError("action basis and rBergomi local dimension do not match")
        if not math.isfinite(self.epsilon) or not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in (0, 1]")
        if self.strike is not None and (
            not math.isfinite(self.strike) or self.strike <= 0.0
        ):
            raise ValueError("strike must be finite and positive")

    def evaluate(self, coefficients: torch.Tensor) -> ConditionalActionValue:
        if coefficients.ndim != 1 or coefficients.shape[0] != self.basis.rank:
            raise ValueError("action coefficients have the wrong shape")
        if coefficients.device.type != "cpu" or coefficients.dtype != torch.float64:
            raise ValueError("action coefficients must be CPU float64")
        control = self.basis.expand(coefficients)
        proposal_mean = control / math.sqrt(self.epsilon)
        batch = evaluate_rbergomi_conditional_terminal(
            self.problem,
            proposal_mean.unsqueeze(0),
            epsilon=self.epsilon,
            strike=self.strike,
        )
        log_probability = batch.payoffs.log_left_probability[0]
        energy = 0.5 * torch.dot(coefficients, coefficients)
        tail_cost = -self.epsilon * log_probability
        value = energy + tail_cost
        return ConditionalActionValue(
            value=value,
            energy=energy,
            negative_scaled_log_probability=tail_cost,
            log_probability=log_probability,
            control=control,
            proposal_mean=proposal_mean,
        )

    def __call__(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.evaluate(coefficients).value


@dataclass(frozen=True)
class ActionDerivativeAudit:
    value: float
    gradient: torch.Tensor
    gradient_norm: float
    hessian: torch.Tensor | None
    hessian_eigenvalues: torch.Tensor | None


def evaluate_action_derivatives(
    action: RBergomiConditionalAction,
    coefficients: torch.Tensor,
    *,
    include_hessian: bool = False,
) -> ActionDerivativeAudit:
    """Evaluate exact autodiff derivatives for solver and theorem-oracle audits."""

    point = coefficients.detach().clone().requires_grad_(True)
    value = action(point)
    (gradient,) = torch.autograd.grad(value, point, create_graph=include_hessian)
    hessian = None
    eigenvalues = None
    if include_hessian:
        hessian = torch.autograd.functional.hessian(action, point).detach()
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues = torch.linalg.eigvalsh(hessian)
    return ActionDerivativeAudit(
        value=float(value.detach()),
        gradient=gradient.detach(),
        gradient_norm=float(torch.linalg.vector_norm(gradient.detach())),
        hessian=hessian,
        hessian_eigenvalues=eigenvalues,
    )

