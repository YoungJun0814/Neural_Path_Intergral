"""Deterministic multistart solvers and audits for Cameron--Martin actions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from src.path_integral.volterra_action import RBergomiConditionalAction

Action = Callable[[torch.Tensor], torch.Tensor]
SolverMethod = Literal["lbfgs", "trust-ncg"]


def _validated_value(action: Action, point: torch.Tensor) -> torch.Tensor:
    value = action(point)
    if value.ndim != 0 or value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError("action must return a scalar CPU float64 tensor")
    if not torch.isfinite(value):
        raise FloatingPointError("action returned a nonfinite value")
    return value


@dataclass(frozen=True)
class ActionSolverConfig:
    method: SolverMethod = "lbfgs"
    maximum_iterations: int = 300
    gradient_tolerance: float = 1e-7
    function_tolerance: float = 1e-12
    maximum_line_search_steps: int = 40
    compute_hessian_eigenvalues: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"lbfgs", "trust-ncg"}:
            raise ValueError("unsupported action solver")
        integers = (self.maximum_iterations, self.maximum_line_search_steps)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in integers):
            raise ValueError("solver iteration limits must be positive integers")
        tolerances = (self.gradient_tolerance, self.function_tolerance)
        if any(not math.isfinite(x) or x <= 0.0 for x in tolerances):
            raise ValueError("solver tolerances must be finite and positive")


@dataclass(frozen=True)
class ActionOptimizationResult:
    method: SolverMethod
    initial_coefficients: torch.Tensor
    coefficients: torch.Tensor
    action_value: float
    gradient_norm: float
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    hessian_eigenvalues: torch.Tensor | None


def optimize_action(
    action: Action,
    initial_coefficients: torch.Tensor,
    *,
    config: ActionSolverConfig | None = None,
) -> ActionOptimizationResult:
    """Minimize a differentiable action with an independently selected solver."""

    config = config or ActionSolverConfig()
    initial = initial_coefficients.detach().clone()
    if initial.ndim != 1 or initial.device.type != "cpu" or initial.dtype != torch.float64:
        raise ValueError("initial coefficients must be a one-dimensional CPU float64 tensor")
    if not torch.isfinite(initial).all():
        raise ValueError("initial coefficients must be finite")

    function_evaluations = 0
    gradient_evaluations = 0
    iterations = 0
    status = 1
    message = "maximum iterations reached"

    if config.method == "lbfgs":
        optimum = initial.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [optimum],
            max_iter=config.maximum_iterations,
            tolerance_grad=config.gradient_tolerance,
            tolerance_change=config.function_tolerance,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal function_evaluations, gradient_evaluations
            optimizer.zero_grad()
            objective = _validated_value(action, optimum)
            objective.backward()
            if optimum.grad is None or not torch.isfinite(optimum.grad).all():
                raise FloatingPointError("action gradient is nonfinite")
            function_evaluations += 1
            gradient_evaluations += 1
            return objective

        optimizer.step(closure)
        state = optimizer.state[optimum]
        iterations = int(state.get("n_iter", 0))
        function_evaluations = int(state.get("func_evals", function_evaluations))
        optimum = optimum.detach()
    else:
        # A self-contained trust-region Newton method avoids platform-dependent
        # native SciPy solver failures and supplies an algorithmically independent
        # cross-check of L-BFGS.  The dense Hessian is intentional: V15 only uses
        # this audit solver in a low-rank Cameron--Martin subspace.
        optimum = initial.clone()
        radius = max(1.0, float(torch.linalg.vector_norm(initial)))
        maximum_radius = 100.0 * radius
        previous_value: float | None = None
        for iteration in range(1, config.maximum_iterations + 1):
            point = optimum.detach().requires_grad_(True)
            objective = _validated_value(action, point)
            (gradient,) = torch.autograd.grad(objective, point, create_graph=True)
            hessian = torch.autograd.functional.hessian(action, point)
            hessian = 0.5 * (hessian + hessian.T)
            function_evaluations += 1
            gradient_evaluations += 1
            iterations = iteration
            if not torch.isfinite(gradient).all() or not torch.isfinite(hessian).all():
                raise FloatingPointError("action derivatives are nonfinite")
            gradient_norm_now = float(torch.linalg.vector_norm(gradient.detach()))
            if gradient_norm_now <= config.gradient_tolerance:
                status = 0
                message = "gradient tolerance reached"
                break

            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            projected_gradient = eigenvectors.T @ gradient.detach()
            floor = max(1e-8, 1e-6 * float(torch.max(torch.abs(eigenvalues))))
            safe_eigenvalues = torch.clamp(eigenvalues, min=floor)
            step = -(eigenvectors @ (projected_gradient / safe_eigenvalues))
            step_norm = float(torch.linalg.vector_norm(step))
            if step_norm > radius:
                step = step * (radius / step_norm)

            predicted = -torch.dot(gradient.detach(), step) - 0.5 * torch.dot(
                step, hessian.detach() @ step
            )
            if float(predicted) <= 0.0:
                step = -radius * gradient.detach() / torch.linalg.vector_norm(gradient.detach())
                predicted = -torch.dot(gradient.detach(), step) - 0.5 * torch.dot(
                    step, hessian.detach() @ step
                )
            candidate = optimum + step
            candidate_value = _validated_value(action, candidate)
            function_evaluations += 1
            actual = objective.detach() - candidate_value.detach()
            ratio = float(actual / predicted) if float(predicted) > 0.0 else -math.inf
            if ratio < 0.25:
                radius *= 0.25
            elif ratio > 0.75 and abs(float(torch.linalg.vector_norm(step)) - radius) <= 1e-10 * (
                1.0 + radius
            ):
                radius = min(2.0 * radius, maximum_radius)
            if ratio > 0.1:
                optimum = candidate.detach()
            accepted_value = candidate_value if ratio > 0.1 else objective
            current_value = float(accepted_value.detach())
            if previous_value is not None and abs(previous_value - current_value) <= (
                config.function_tolerance * (1.0 + abs(previous_value))
            ):
                status = 0
                message = "function tolerance reached"
                break
            if radius <= 1e-14:
                status = 2
                message = "trust-region radius collapsed"
                break
            previous_value = current_value

    final_point = optimum.detach().requires_grad_(True)
    final_objective = _validated_value(action, final_point)
    (final_gradient,) = torch.autograd.grad(final_objective, final_point)
    function_evaluations += 1
    gradient_evaluations += 1
    value = float(final_objective.detach())
    gradient_norm = float(torch.linalg.vector_norm(final_gradient.detach()))
    hessian_eigenvalues = None
    if config.compute_hessian_eigenvalues:
        hessian = torch.autograd.functional.hessian(action, optimum.detach())
        hessian = 0.5 * (hessian + hessian.T)
        hessian_eigenvalues = torch.linalg.eigvalsh(hessian).detach()
    solver_success = gradient_norm <= 10.0 * config.gradient_tolerance
    if solver_success:
        status = 0
        message = "gradient tolerance reached"
    return ActionOptimizationResult(
        method=config.method,
        initial_coefficients=initial,
        coefficients=optimum.detach(),
        action_value=value,
        gradient_norm=gradient_norm,
        success=solver_success,
        status=status,
        message=message,
        iterations=iterations,
        function_evaluations=function_evaluations,
        gradient_evaluations=gradient_evaluations,
        hessian_eigenvalues=hessian_eigenvalues,
    )


@dataclass(frozen=True)
class ModeSearchConfig:
    methods: tuple[SolverMethod, ...] = ("lbfgs", "trust-ncg")
    random_starts: int = 8
    random_seed: int = 1729
    start_scale: float = 2.0
    merge_distance: float = 1e-4
    action_merge_tolerance: float = 1e-8
    maximum_modes: int = 8
    solver: ActionSolverConfig = ActionSolverConfig()

    def __post_init__(self) -> None:
        if not self.methods or any(x not in {"lbfgs", "trust-ncg"} for x in self.methods):
            raise ValueError("mode search requires supported solvers")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("mode-search solvers must be unique")
        integers = (self.random_starts, self.maximum_modes)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in integers):
            raise ValueError("mode-search counts must be positive integers")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("mode-search seed must be an integer")
        positives = (self.start_scale, self.merge_distance, self.action_merge_tolerance)
        if any(not math.isfinite(x) or x <= 0.0 for x in positives):
            raise ValueError("mode-search scales and tolerances must be positive")


@dataclass(frozen=True)
class ActionMode:
    coefficients: torch.Tensor
    action_value: float
    gradient_norm: float
    hessian_eigenvalues: torch.Tensor | None
    support_count: int
    methods: tuple[SolverMethod, ...]


@dataclass(frozen=True)
class ModeSearchResult:
    modes: tuple[ActionMode, ...]
    attempts: tuple[ActionOptimizationResult, ...]
    failed_attempts: int


def _deterministic_starts(
    dimension: int,
    *,
    count: int,
    scale: float,
    seed: int,
    supplied: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    starts = [torch.zeros(dimension, dtype=torch.float64)]
    for item in supplied:
        if item.shape != (dimension,) or item.dtype != torch.float64 or item.device.type != "cpu":
            raise ValueError("supplied mode start has the wrong contract")
        starts.append(item.detach().clone())
    generator = torch.Generator().manual_seed(seed)
    while len(starts) < count + 1 + len(supplied):
        direction = torch.randn(dimension, dtype=torch.float64, generator=generator)
        norm = torch.linalg.vector_norm(direction)
        radius = scale * (0.25 + 0.75 * torch.rand((), generator=generator).item())
        starts.append(radius * direction / norm)
    return tuple(starts)


def find_action_modes(
    action: Action,
    dimension: int,
    *,
    config: ModeSearchConfig | None = None,
    supplied_starts: Sequence[torch.Tensor] = (),
) -> ModeSearchResult:
    """Find and merge stationary modes using independent deterministic solvers."""

    config = config or ModeSearchConfig()
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("action dimension must be a positive integer")
    starts = _deterministic_starts(
        dimension,
        count=config.random_starts,
        scale=config.start_scale,
        seed=config.random_seed,
        supplied=supplied_starts,
    )
    attempts: list[ActionOptimizationResult] = []
    for method in config.methods:
        solver = ActionSolverConfig(
            method=method,
            maximum_iterations=config.solver.maximum_iterations,
            gradient_tolerance=config.solver.gradient_tolerance,
            function_tolerance=config.solver.function_tolerance,
            maximum_line_search_steps=config.solver.maximum_line_search_steps,
            compute_hessian_eigenvalues=config.solver.compute_hessian_eigenvalues,
        )
        for start in starts:
            attempts.append(optimize_action(action, start, config=solver))

    successful = sorted((x for x in attempts if x.success), key=lambda x: x.action_value)
    groups: list[list[ActionOptimizationResult]] = []
    for result in successful:
        destination = None
        for index, group in enumerate(groups):
            representative = group[0]
            distance = float(torch.linalg.vector_norm(result.coefficients - representative.coefficients))
            scale = 1.0 + float(torch.linalg.vector_norm(representative.coefficients))
            action_close = abs(result.action_value - representative.action_value)
            if (
                distance <= config.merge_distance * scale
                and action_close <= config.action_merge_tolerance * (1.0 + abs(result.action_value))
            ):
                destination = index
                break
        if destination is None:
            groups.append([result])
        else:
            groups[destination].append(result)
    modes = []
    for group in groups[: config.maximum_modes]:
        best = min(group, key=lambda x: (x.action_value, x.gradient_norm))
        methods = tuple(sorted({item.method for item in group}))
        modes.append(
            ActionMode(
                coefficients=best.coefficients,
                action_value=best.action_value,
                gradient_norm=best.gradient_norm,
                hessian_eigenvalues=best.hessian_eigenvalues,
                support_count=len(group),
                methods=methods,
            )
        )
    return ModeSearchResult(
        modes=tuple(modes),
        attempts=tuple(attempts),
        failed_attempts=sum(not x.success for x in attempts),
    )


@dataclass(frozen=True)
class RBergomiMode:
    coefficients: torch.Tensor
    control: torch.Tensor
    proposal_mean: torch.Tensor
    action_value: float
    gradient_norm: float
    hessian_eigenvalues: torch.Tensor | None
    support_count: int
    methods: tuple[SolverMethod, ...]


@dataclass(frozen=True)
class RBergomiModeSearchResult:
    modes: tuple[RBergomiMode, ...]
    raw: ModeSearchResult


def find_rbergomi_conditional_modes(
    action: RBergomiConditionalAction,
    *,
    config: ModeSearchConfig | None = None,
    supplied_starts: Sequence[torch.Tensor] = (),
) -> RBergomiModeSearchResult:
    raw = find_action_modes(
        action,
        action.basis.rank,
        config=config,
        supplied_starts=supplied_starts,
    )
    modes = []
    for mode in raw.modes:
        evaluated = action.evaluate(mode.coefficients)
        modes.append(
            RBergomiMode(
                coefficients=mode.coefficients,
                control=evaluated.control.detach(),
                proposal_mean=evaluated.proposal_mean.detach(),
                action_value=mode.action_value,
                gradient_norm=mode.gradient_norm,
                hessian_eigenvalues=mode.hessian_eigenvalues,
                support_count=mode.support_count,
                methods=mode.methods,
            )
        )
    return RBergomiModeSearchResult(modes=tuple(modes), raw=raw)
