"""Shared latent-coordinate contract for faithful rBergomi baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import torch

from src.path_integral.path_functionals import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    TerminalThresholdTask,
)
from src.path_integral.rbergomi_fft import (
    RBergomiFFTInnovations,
    simulate_rbergomi_fft,
)
from src.physics_engine import RBergomiSimulator, TwoDriverRBergomiPaths

BaselineTask: TypeAlias = TerminalThresholdTask | DiscreteBarrierHitTask | DownsideExcursionTask


@dataclass(frozen=True)
class BaselineUnitBatch:
    """Independent inferential-unit contributions and exact work counters."""

    unit_contributions: torch.Tensor
    raw_sample_count: int
    likelihood_evaluations: int
    cdf_calls: int
    quadrature_calls: int

    def __post_init__(self) -> None:
        values = self.unit_contributions
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError("baseline unit contributions must be a nonempty vector")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("baseline unit contributions must be finite floating point")
        counts = (
            self.raw_sample_count,
            self.likelihood_evaluations,
            self.cdf_calls,
            self.quadrature_calls,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ValueError("baseline work counters must be nonnegative integers")


@dataclass(frozen=True)
class RBergomiBaselineProblem:
    """One frozen finite-grid rBergomi task in standard-normal coordinates.

    At each monitoring step the BLP FFT simulator consumes two independent
    standard normals for the local Volterra cell and one for the orthogonal
    price driver.  Thus the target path law is exactly the image of a
    ``3 * steps`` dimensional standard Gaussian.
    """

    task_id: str
    task: BaselineTask
    spot: float
    maturity: float
    steps: int
    hurst: float
    eta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be nonempty")
        if not math.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError("spot must be finite and positive")
        if not math.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not 0.0 < self.hurst < 0.5:
            raise ValueError("hurst must lie in (0, 0.5)")
        if not math.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(self.xi) or self.xi <= 0.0:
            raise ValueError("xi must be finite and positive")
        if not math.isfinite(self.rho) or not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")

    @property
    def latent_dimension(self) -> int:
        return 3 * self.steps

    @property
    def local_dimension(self) -> int:
        return 2 * self.steps

    @property
    def step_dt(self) -> float:
        return self.maturity / self.steps

    def simulator(self) -> RBergomiSimulator:
        return RBergomiSimulator(
            H=self.hurst,
            eta=self.eta,
            xi=self.xi,
            rho=self.rho,
            device="cpu",
        )

    def _validate_latent(self, latent: torch.Tensor, dimension: int) -> None:
        if latent.ndim != 2 or latent.shape[1] != dimension or latent.shape[0] < 1:
            raise ValueError("latent sample has the wrong shape")
        if (
            latent.device.type != "cpu"
            or latent.dtype != torch.float64
            or not torch.isfinite(latent).all()
        ):
            raise ValueError("latent sample must be finite CPU float64")

    def simulate_latent(self, latent: torch.Tensor) -> TwoDriverRBergomiPaths:
        self._validate_latent(latent, self.latent_dimension)
        local = latent[:, : self.local_dimension].reshape(-1, self.steps, 2)
        price = latent[:, self.local_dimension :]
        return simulate_rbergomi_fft(
            self.simulator(),
            S0=self.spot,
            T=self.maturity,
            dt=self.step_dt,
            num_paths=latent.shape[0],
            innovations=RBergomiFFTInnovations(
                local_standard_normal=local,
                price_standard_normal=price,
            ),
            dtype=torch.float64,
        )

    def simulate_local(self, local_latent: torch.Tensor) -> TwoDriverRBergomiPaths:
        self._validate_latent(local_latent, self.local_dimension)
        full = torch.cat(
            (
                local_latent,
                torch.zeros(local_latent.shape[0], self.steps, dtype=torch.float64),
            ),
            dim=1,
        )
        return self.simulate_latent(full)

    def hard_event(self, paths: TwoDriverRBergomiPaths) -> torch.Tensor:
        return self.task.hard_event_from_log_spot(paths.log_spot, paths.step_dt)

    def score(self, paths: TwoDriverRBergomiPaths) -> torch.Tensor:
        """Continuous CEM/optimization score whose nonnegative set is the event."""

        log_spot = paths.log_spot
        if isinstance(self.task, TerminalThresholdTask):
            return math.log(self.task.level) - log_spot[:, -1]
        if isinstance(self.task, DiscreteBarrierHitTask):
            return math.log(self.task.barrier) - torch.amin(log_spot, dim=1)
        if isinstance(self.task, DownsideExcursionTask):
            hit_margin = math.log(self.task.hit_barrier) - torch.amin(log_spot, dim=1)
            required = math.ceil((self.task.minimum_occupation - 1e-15) / paths.step_dt)
            occupation_margin = (
                torch.sum(log_spot[:, 1:] <= math.log(self.task.stress_level), dim=1).to(
                    torch.float64
                )
                - required
            )
            return torch.minimum(hit_margin, occupation_margin)
        raise TypeError("unsupported baseline task")
