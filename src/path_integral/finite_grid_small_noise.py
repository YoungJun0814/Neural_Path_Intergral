"""Fixed-grid small-noise skeletons and contracted rate actions.

This module contains finite-dimensional objects only.  In particular, the BLP
auxiliary coordinates are not identified with a continuous-time
Cameron--Martin space here.  That separation prevents a fixed-grid large
deviation statement from being silently promoted to a Wiener-space theorem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.cameron_martin_basis import CameronMartinBasis
from src.path_integral.path_functionals import TerminalThresholdTask


@dataclass(frozen=True)
class FiniteGridRateValue:
    value: torch.Tensor
    energy: torch.Tensor
    conditional_cost: torch.Tensor
    correlated_return: torch.Tensor
    integrated_variance: torch.Tensor
    control: torch.Tensor


@dataclass(frozen=True)
class RBergomiFiniteGridRateAction:
    """The exact contracted rate candidate for one frozen BLP grid.

    When ``basis.rank == problem.local_dimension`` this is the full fixed-grid
    rate action.  A lower-rank basis is only a Galerkin upper bound on the true
    minimum and is never labelled as the fixed-grid rate itself.
    """

    problem: RBergomiBaselineProblem
    basis: CameronMartinBasis
    strike: float | None = None

    def __post_init__(self) -> None:
        if self.basis.dimension != self.problem.local_dimension:
            raise ValueError("rate basis and rBergomi local dimension do not match")
        if self.strike is None:
            if not isinstance(self.problem.task, TerminalThresholdTask):
                raise ValueError("a strike is required for a non-terminal task")
        elif not math.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("strike must be finite and positive")

    @property
    def is_full_grid(self) -> bool:
        return self.basis.rank == self.problem.local_dimension

    @property
    def resolved_strike(self) -> float:
        if self.strike is not None:
            return float(self.strike)
        task = self.problem.task
        if not isinstance(task, TerminalThresholdTask):
            raise RuntimeError("terminal task contract changed after construction")
        return float(task.level)

    def evaluate(self, coefficients: torch.Tensor) -> FiniteGridRateValue:
        if coefficients.shape != (self.basis.rank,):
            raise ValueError("rate coefficients have the wrong shape")
        if coefficients.device.type != "cpu" or coefficients.dtype != torch.float64:
            raise ValueError("rate coefficients must be CPU float64")
        if not torch.isfinite(coefficients).all():
            raise ValueError("rate coefficients must be finite")
        control = self.basis.expand(coefficients)
        skeleton = self.problem.simulate_local(
            control.unsqueeze(0),
            variance_compensator_scale=0.0,
        )
        integrated_variance = self.problem.step_dt * torch.sum(
            skeleton.variance[0, :-1]
        )
        # The local simulator has the orthogonal price driver fixed to zero.
        # Removing its unit-noise Itô drift therefore leaves precisely the
        # correlated skeleton return A_N(h).
        log_return = skeleton.log_spot[0, -1] - math.log(self.problem.spot)
        correlated_return = log_return + 0.5 * integrated_variance
        log_moneyness = math.log(self.resolved_strike / self.problem.spot)
        gap = torch.clamp(correlated_return - log_moneyness, min=0.0)
        conditional_cost = gap.square() / (
            2.0 * (1.0 - self.problem.rho**2) * integrated_variance
        )
        energy = 0.5 * torch.dot(coefficients, coefficients)
        return FiniteGridRateValue(
            value=energy + conditional_cost,
            energy=energy,
            conditional_cost=conditional_cost,
            correlated_return=correlated_return,
            integrated_variance=integrated_variance,
            control=control,
        )

    def __call__(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.evaluate(coefficients).value
