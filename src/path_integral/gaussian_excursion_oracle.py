"""Finite-state dynamic-programming oracle for a Gaussian excursion event.

The dynamic program is an explicitly labelled spatial-grid approximation.  Its
bounded adaptive drift is nevertheless used with the exact discrete Gaussian
likelihood, so approximation error affects efficiency, not unbiasedness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GaussianExcursionSpec:
    """Finite-grid Gaussian random-walk event specification."""

    steps: int
    maturity: float
    hit_barrier: float
    stress_level: float
    minimum_occupation: float

    def __post_init__(self) -> None:
        values = (
            self.maturity,
            self.hit_barrier,
            self.stress_level,
            self.minimum_occupation,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gaussian excursion specification must be finite")
        if self.steps <= 0 or self.maturity <= 0.0:
            raise ValueError("steps and maturity must be positive")
        if self.hit_barrier >= self.stress_level:
            raise ValueError("hit_barrier must be strictly below stress_level")
        if not 0.0 < self.minimum_occupation <= self.maturity:
            raise ValueError("minimum_occupation must lie in (0, maturity]")

    @property
    def step_dt(self) -> float:
        return self.maturity / self.steps

    @property
    def required_occupation_steps(self) -> int:
        # Match ``count * dt >= minimum_occupation`` without a binary64
        # round-off accidentally adding an extra required step.
        return int(math.ceil(self.minimum_occupation / self.step_dt - 1e-12))


@dataclass(frozen=True)
class GaussianExcursionOracle:
    """Spatial-grid value function and its Gaussian information projection."""

    spec: GaussianExcursionSpec
    state_grid: FloatArray
    desirability: FloatArray
    projected_control: FloatArray
    reference_probability: float
    control_bound: float

    def _lookup_table(
        self,
        table: FloatArray,
        step: int,
        state: FloatArray,
        hit: BoolArray,
        occupation_steps: IntArray,
    ) -> FloatArray:
        if not 0 <= step < table.shape[0]:
            raise ValueError("step is outside the oracle table")
        state = np.asarray(state, dtype=np.float64)
        hit = np.asarray(hit, dtype=np.bool_)
        occupation_steps = np.asarray(occupation_steps, dtype=np.int64)
        if state.shape != hit.shape or state.shape != occupation_steps.shape:
            raise ValueError("state, hit, and occupation arrays must have equal shapes")
        capped = np.clip(occupation_steps, 0, self.spec.required_occupation_steps)
        result = np.empty_like(state)
        for hit_state in (False, True):
            for occupation in np.unique(capped):
                selected = (hit == hit_state) & (capped == occupation)
                if np.any(selected):
                    values = table[step, int(hit_state), int(occupation)]
                    result[selected] = np.interp(
                        state[selected],
                        self.state_grid,
                        values,
                        left=float(values[0]),
                        right=float(values[-1]),
                    )
        return result

    def control(
        self,
        step: int,
        state: FloatArray,
        hit: BoolArray,
        occupation_steps: IntArray,
    ) -> FloatArray:
        """Interpolate the bounded pre-increment adapted control."""
        return self._lookup_table(
            self.projected_control, step, state, hit, occupation_steps
        )


@dataclass(frozen=True)
class GaussianExcursionSample:
    """Exact-likelihood Monte Carlo sample for the declared Gaussian grid law."""

    event: BoolArray
    contribution: FloatArray
    likelihood: FloatArray
    terminal_state: FloatArray
    mean_control_energy: float

    @property
    def estimate(self) -> float:
        return float(np.mean(self.contribution))

    @property
    def standard_error(self) -> float:
        return float(np.std(self.contribution, ddof=1) / math.sqrt(len(self.contribution)))

    @property
    def second_moment(self) -> float:
        return float(np.mean(np.square(self.contribution)))


def _normal_transition_matrices(
    state_grid: FloatArray,
    step_dt: float,
    *,
    bin_edges: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return bin probabilities and unnormalized Brownian first moments."""
    if state_grid.ndim != 1 or len(state_grid) < 3 or not np.all(np.diff(state_grid) > 0.0):
        raise ValueError("state_grid must be a strictly increasing vector of length >= 3")
    if bin_edges is None:
        edges = np.empty(len(state_grid) + 1, dtype=np.float64)
        edges[0], edges[-1] = -np.inf, np.inf
        edges[1:-1] = 0.5 * (state_grid[:-1] + state_grid[1:])
    else:
        edges = np.asarray(bin_edges, dtype=np.float64)
        if (
            edges.shape != (len(state_grid) + 1,)
            or not np.isneginf(edges[0])
            or not np.isposinf(edges[-1])
            or not np.all(np.diff(edges) > 0.0)
        ):
            raise ValueError("bin_edges must be increasing with infinite outer edges")
    sqrt_dt = math.sqrt(step_dt)
    standardized = (edges[None, :] - state_grid[:, None]) / sqrt_dt
    cdf = ndtr(standardized)
    probability = np.diff(cdf, axis=1)
    density = np.exp(-0.5 * np.square(standardized)) / math.sqrt(2.0 * math.pi)
    density[:, 0] = 0.0
    density[:, -1] = 0.0
    brownian_moment = sqrt_dt * (density[:, :-1] - density[:, 1:])
    return probability, brownian_moment


def _event_aligned_grid(
    state_minimum: float,
    state_maximum: float,
    state_points: int,
    *,
    hit_barrier: float,
    stress_level: float,
) -> tuple[FloatArray, FloatArray]:
    """Create quantization bins whose edges contain both event thresholds."""
    if not state_minimum < hit_barrier < stress_level < state_maximum:
        raise ValueError("both event thresholds must lie strictly inside the state grid")
    finite_edges = np.unique(
        np.concatenate(
            (
                np.linspace(state_minimum, state_maximum, state_points - 1),
                np.asarray((hit_barrier, stress_level), dtype=np.float64),
            )
        )
    )
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf])).astype(np.float64)
    grid = np.empty(len(edges) - 1, dtype=np.float64)
    grid[1:-1] = 0.5 * (finite_edges[:-1] + finite_edges[1:])
    grid[0] = finite_edges[0] - 0.5 * (finite_edges[1] - finite_edges[0])
    grid[-1] = finite_edges[-1] + 0.5 * (finite_edges[-1] - finite_edges[-2])
    return grid, edges


def build_gaussian_excursion_oracle(
    spec: GaussianExcursionSpec,
    *,
    state_minimum: float = -4.0,
    state_maximum: float = 4.0,
    state_points: int = 401,
    control_bound: float = 8.0,
    desirability_floor: float = 1e-12,
) -> GaussianExcursionOracle:
    """Solve the quantized Gaussian path problem by backward dynamic programming."""
    if not all(
        math.isfinite(value)
        for value in (state_minimum, state_maximum, control_bound, desirability_floor)
    ):
        raise ValueError("oracle grid and regularization values must be finite")
    if state_minimum >= state_maximum or state_points < 3:
        raise ValueError("oracle state grid is invalid")
    if not state_minimum < 0.0 < state_maximum:
        raise ValueError("oracle state grid must contain the initial state zero")
    if control_bound <= 0.0 or desirability_floor <= 0.0:
        raise ValueError("control_bound and desirability_floor must be positive")

    grid, bin_edges = _event_aligned_grid(
        state_minimum,
        state_maximum,
        state_points,
        hit_barrier=spec.hit_barrier,
        stress_level=spec.stress_level,
    )
    transition, brownian_moment = _normal_transition_matrices(
        grid, spec.step_dt, bin_edges=bin_edges
    )
    required = spec.required_occupation_steps
    desirability = np.zeros(
        (spec.steps + 1, 2, required + 1, len(grid)), dtype=np.float64
    )
    projected = np.zeros((spec.steps, 2, required + 1, len(grid)), dtype=np.float64)
    desirability[-1, 1, required, :] = 1.0

    next_hit_from_state = grid <= spec.hit_barrier
    next_stress = grid <= spec.stress_level
    for step in range(spec.steps - 1, -1, -1):
        for hit_state in (0, 1):
            next_hit = next_hit_from_state | bool(hit_state)
            for occupation in range(required + 1):
                next_occupation = np.minimum(
                    required, occupation + next_stress.astype(np.int64)
                )
                future = desirability[
                    step + 1,
                    next_hit.astype(np.int64),
                    next_occupation,
                    np.arange(len(grid)),
                ]
                value = transition @ future
                moment = brownian_moment @ future
                desirability[step, hit_state, occupation] = value
                raw_control = moment / (spec.step_dt * np.maximum(value, desirability_floor))
                projected[step, hit_state, occupation] = np.clip(
                    raw_control, -control_bound, control_bound
                )

    reference = float(
        np.interp(0.0, grid, desirability[0, 0, 0], left=0.0, right=0.0)
    )
    return GaussianExcursionOracle(
        spec=spec,
        state_grid=grid,
        desirability=desirability,
        projected_control=projected,
        reference_probability=reference,
        control_bound=control_bound,
    )


def simulate_gaussian_excursion(
    spec: GaussianExcursionSpec,
    *,
    num_paths: int,
    seed: int,
    oracle: GaussianExcursionOracle | None = None,
    constant_control: float = 0.0,
) -> GaussianExcursionSample:
    """Simulate under a natural, constant, or oracle proposal with exact density."""
    if num_paths <= 1:
        raise ValueError("num_paths must exceed one")
    if not math.isfinite(constant_control):
        raise ValueError("constant_control must be finite")
    if oracle is not None and oracle.spec != spec:
        raise ValueError("oracle and simulation specifications differ")

    rng = np.random.default_rng(seed)
    state = np.zeros(num_paths, dtype=np.float64)
    hit = np.zeros(num_paths, dtype=np.bool_)
    occupation = np.zeros(num_paths, dtype=np.int64)
    log_likelihood = np.zeros(num_paths, dtype=np.float64)
    energy = np.zeros(num_paths, dtype=np.float64)
    sqrt_dt = math.sqrt(spec.step_dt)
    for step in range(spec.steps):
        if oracle is None:
            control = np.full(num_paths, constant_control, dtype=np.float64)
        else:
            control = oracle.control(step, state, hit, occupation)
        proposal_increment = sqrt_dt * rng.standard_normal(num_paths)
        target_increment = proposal_increment + control * spec.step_dt
        log_likelihood -= control * proposal_increment + 0.5 * np.square(control) * spec.step_dt
        energy += np.square(control) * spec.step_dt
        state += target_increment
        hit |= state <= spec.hit_barrier
        occupation = np.minimum(
            spec.required_occupation_steps,
            occupation + (state <= spec.stress_level).astype(np.int64),
        )

    event = hit & (occupation >= spec.required_occupation_steps)
    likelihood = np.exp(log_likelihood)
    contribution = event.astype(np.float64) * likelihood
    return GaussianExcursionSample(
        event=event,
        contribution=contribution,
        likelihood=likelihood,
        terminal_state=state,
        mean_control_energy=float(np.mean(energy)),
    )
