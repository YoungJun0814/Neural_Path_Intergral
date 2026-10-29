"""Finite-grid research contracts, holdout guards and bounded work accounting.

These gates control research execution; none certifies continuum accuracy,
population mode coverage or scientific novelty.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .baselines.rbergomi_common import RBergomiBaselineProblem
from .path_functionals import TerminalThresholdTask
from .research_result_contract import canonical_digest
from .seed_ledger import SeedKey, derive_seed

Payoff = Literal["digital", "normalized_put"]
Role = Literal["reference_pilot", "training", "selection", "final", "heldout", "audit"]
LAW = "blp-hybrid-original-epsilon1-terminal-v1"
SCHEMA = "npi.finite-grid-geometry.protocol.v1"
ROLES = {"reference_pilot", "training", "selection", "final", "heldout", "audit"}


def positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GeometryTask:
    task_id: str
    split: Literal["development", "confirmation"]
    hurst: float
    eta: float
    rho: float
    maturity: float
    strike_ratio: float
    steps: int
    payoff: Payoff
    spot: float = 100.0
    xi: float = .04

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z][0-9]{2}", self.task_id):
            raise ValueError("task_id must be a letter and two digits")
        if self.split not in ("development", "confirmation"):
            raise ValueError("unknown task split")
        if self.payoff not in ("digital", "normalized_put"):
            raise ValueError("only declared bounded payoffs are supported")
        values = (self.hurst, self.eta, self.rho, self.maturity,
                  self.strike_ratio, self.spot, self.xi)
        if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("finite non-boolean task parameters required")
        if not self.strike_ratio > 0:
            raise ValueError("strike ratio must be positive")
        self.problem()  # Reuse the law's own parameter and dimension checks.

    def problem(self) -> RBergomiBaselineProblem:
        return RBergomiBaselineProblem(
            self.task_id, TerminalThresholdTask(self.spot*self.strike_ratio), self.spot,
            self.maturity, self.steps, self.hurst, self.eta, self.xi, self.rho,
        )

    @property
    def estimand_digest(self) -> str:
        return canonical_digest({"law": LAW, "task": asdict(self),
                                 "epsilon": 1.0, "rate": 0.0, "dividend": 0.0})


@dataclass(frozen=True)
class PhaseLimit:
    phase: str
    max_evaluations: int
    max_wall_seconds: float

    def __post_init__(self) -> None:
        if self.phase not in {"FR1", "FR2", "FR3", "FR4"}:
            raise ValueError("only first-cycle limits are declared")
        positive_integer(self.max_evaluations, "max evaluations")
        if (isinstance(self.max_wall_seconds, bool)
                or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0):
            raise ValueError("finite positive wall cap required")


@dataclass(frozen=True)
class GeometryProtocol:
    protocol_id: str
    tasks: tuple[GeometryTask, ...]
    limits: tuple[PhaseLimit, ...]
    pilot_units: int = 32
    pilot_power: int = 13
    reference_rse: float = .0125
    pilot_agreement_margin: float = .20
    family_alpha: float = .05
    max_redesigns: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"fg-geometry-[a-z0-9-]+", self.protocol_id):
            raise ValueError("invalid finite-grid protocol namespace")
        if not self.tasks or len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("nonempty distinct task IDs required")
        if not all(isinstance(t, GeometryTask) for t in self.tasks):
            raise ValueError("task contract objects required")
        if {t.split for t in self.tasks} != {"development", "confirmation"}:
            raise ValueError("development and sealed confirmation tasks required")
        if {v.phase for v in self.limits} != {"FR1", "FR2", "FR3", "FR4"} or len(self.limits) != 4:
            raise ValueError("exactly one limit per first-cycle phase required")
        positive_integer(self.pilot_units, "pilot units")
        if self.pilot_units < 2 or type(self.pilot_power) is not int or not 1 <= self.pilot_power <= 20:
            raise ValueError("at least two units and a bounded Sobol power required")
        for v in (self.reference_rse, self.pilot_agreement_margin, self.family_alpha):
            if isinstance(v, bool) or not math.isfinite(v) or not 0 < v < 1:
                raise ValueError("probability/tolerance must lie in (0,1)")
        if type(self.max_redesigns) is not int or self.max_redesigns != 1:
            raise ValueError("one redesign for the entire cycle, not one per phase")
        if self.planned_pilot_evaluations > self.limit("FR2").max_evaluations:
            raise ValueError("planned reference jobs exceed the FR2 cap")

    def limit(self, phase: str) -> PhaseLimit:
        return next(v for v in self.limits if v.phase == phase)

    @property
    def planned_pilot_evaluations(self) -> int:
        return sum(t.split == "development" for t in self.tasks)*2*self.pilot_units*2**self.pilot_power

    def authorize(self, task_id: str, phase: str, role: Role, *,
                  candidate_digest: str | None = None,
                  allocation_digest: str | None = None) -> GeometryTask:
        if phase not in {f"FR{i}" for i in range(1, 8)} or role not in ROLES:
            raise ValueError("unknown evaluation phase or role")
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task is None:
            raise ValueError("task is not registered")
        if task.split == "confirmation":
            if int(phase[2:]) < 5:
                raise ValueError("sealed confirmation task cannot be evaluated in FR1-FR4")
            if role == "reference_pilot":
                raise ValueError("holdout allocation needs a separate candidate-blind contract")
            if any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
                   for v in (candidate_digest, allocation_digest)):
                raise ValueError("frozen candidate and allocation digests required")
        if role == "reference_pilot" and phase != "FR2":
            raise ValueError("reference-only feasibility belongs to FR2")
        return task

    def seed(self, task_id: str, role: Role, replicate: int, stream: str) -> int:
        if role not in ROLES:
            raise ValueError("unknown seed role")
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task is None:
            raise ValueError("unknown task seed")
        return derive_seed(SeedKey(self.protocol_id, role, task.split, task_id,
                                   task.steps, replicate, stream))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = [asdict(t) for t in self.tasks]
        payload["limits"] = [asdict(v) for v in self.limits]
        return {"schema": SCHEMA, "law": LAW, **payload}

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GeometryProtocol:
        expected = set(cls.__dataclass_fields__) | {"schema", "law"}
        if set(raw) != expected or raw["schema"] != SCHEMA or raw["law"] != LAW:
            raise ValueError("strict finite-grid schema and original law required")
        fields = {k: v for k, v in raw.items() if k not in {"schema", "law", "tasks", "limits"}}
        return cls(**fields, tasks=tuple(GeometryTask(**v) for v in raw["tasks"]),
                   limits=tuple(PhaseLimit(**v) for v in raw["limits"]))


class EvaluationBudget:
    """Reserve before an operation, including an operation that subsequently fails."""

    def __init__(self, limit: PhaseLimit, *, already_spent: int = 0,
                 already_elapsed_seconds: float = 0.) -> None:
        if (type(already_spent) is not int or not 0 <= already_spent <= limit.max_evaluations
                or isinstance(already_elapsed_seconds, bool)
                or not math.isfinite(already_elapsed_seconds)
                or not 0 <= already_elapsed_seconds <= limit.max_wall_seconds):
            raise ValueError("valid cumulative phase expenditure required")
        self.limit = limit
        self.started = time.perf_counter()
        self.previous_wall = already_elapsed_seconds
        self.evaluations = already_spent

    def check_time(self) -> None:
        if self.previous_wall+time.perf_counter()-self.started > self.limit.max_wall_seconds:
            raise TimeoutError("phase wall cap reached; partial work is not a successful unit")

    def reserve(self, count: int) -> None:
        positive_integer(count, "reserved evaluations")
        self.check_time()
        if self.evaluations+count > self.limit.max_evaluations:
            raise RuntimeError("phase evaluation cap reached")
        self.evaluations += count
