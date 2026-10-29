"""Same-integrand 2N conditional MC/RQMC adapters, not a new pricing model."""

from __future__ import annotations

import math
from typing import Any

import torch
from scipy.stats import t as student_t

from .cached_volterra_terminal import CachedVolterraTerminal
from .finite_grid_geometry_protocol import EvaluationBudget, GeometryProtocol, GeometryTask
from .research_result_contract import Moments
from .volterra_conditional_payoffs import evaluate_lognormal_terminal_payoffs


def normal_coordinates(method: str, count: int, dimension: int, seed: int) -> torch.Tensor:
    if (type(count) is not int or count < 2 or type(dimension) is not int or dimension < 1
            or type(seed) is not int or not 0 < seed < 2**63):
        raise ValueError("positive count/dimension and registered seed required")
    if method == "conditional_mc":
        return torch.randn(count, dimension, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(seed))
    if method != "conditional_rqmc" or count & (count-1):
        raise ValueError("RQMC needs a power of two; unknown methods are rejected")
    u = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed).draw_base2(
        count.bit_length()-1, dtype=torch.float64)
    if not torch.isfinite(u).all() or bool(((u <= 0) | (u >= 1)).any()):
        raise FloatingPointError("inverse-CDF endpoint: no clipping or favorable rescrambling")
    return torch.special.ndtri(u)


def bounded_payoff(task: GeometryTask, cache: CachedVolterraTerminal,
                   z: torch.Tensor) -> torch.Tensor:
    if cache.problem != task.problem():
        raise ValueError("cached law/task/strike binding changed")
    if task.payoff == "digital":
        value = cache.log_probability(z).exp()
    else:
        mean, integrated, _ = cache.quantities(z)
        strike = task.spot*task.strike_ratio
        value = evaluate_lognormal_terminal_payoffs(
            mean, (1-task.rho**2)*integrated, strike=strike).put_value/strike
    if not torch.isfinite(value).all() or bool(((value < 0) | (value > 1)).any()):
        raise FloatingPointError("bounded payoff failed; no contribution clipping")
    return value


def reference_units(protocol: GeometryProtocol, task_id: str, method: str,
                    budget: EvaluationBudget) -> dict[str, Any]:
    task = protocol.authorize(task_id, "FR2", "reference_pilot")
    if method not in {"conditional_mc", "conditional_rqmc"}:
        raise ValueError("undeclared reference method")
    cache = CachedVolterraTerminal.create(task.problem())
    count = 2**protocol.pilot_power
    means, seeds = [], []
    pooled: Moments | None = None
    for unit in range(protocol.pilot_units):
        seed = protocol.seed(task_id, "reference_pilot", unit, method)
        budget.reserve(count)
        values = bounded_payoff(task, cache, normal_coordinates(method, count, 2*task.steps, seed))
        current = Moments.from_values(values)
        pooled = current if pooled is None else pooled.merge(current)
        means.append(current.mean)
        seeds.append(seed)
        budget.check_time()
    assert pooled is not None
    units = Moments.from_values(torch.tensor(means, dtype=torch.float64))
    se = pooled.standard_error if method == "conditional_mc" else units.standard_error
    return {"method": method, "estimand_digest": task.estimand_digest,
            "unit_type": "iid_path" if method == "conditional_mc" else "independent_scramble",
            "independent_unit_count": pooled.count if method == "conditional_mc" else units.count,
            "path_count": pooled.count, "mean": units.mean, "se": se,
            "rse": se/units.mean if units.mean > 0 else None,
            "block_se": units.standard_error, "unit_means": means, "seeds": seeds,
            "pointwise_moments": pooled.to_dict(),
            "warning": "empirical precision; shared simulator; not an exact oracle or continuum certificate"}


def feasibility(protocol: GeometryProtocol, mc: dict[str, Any],
                rqmc: dict[str, Any]) -> dict[str, Any]:
    if (mc["method"] != "conditional_mc" or rqmc["method"] != "conditional_rqmc"
            or mc["estimand_digest"] != rqmc["estimand_digest"]):
        raise ValueError("reference order or estimand binding changed")
    cells = sum(t.split == "development" for t in protocol.tasks)
    critical = float(student_t.ppf(1-protocol.family_alpha/(2*cells), protocol.pilot_units-1))
    difference = abs(mc["mean"]-rqmc["mean"])
    uncertainty = critical*math.hypot(mc["se"], rqmc["se"])
    scale = min(mc["mean"], rqmc["mean"])
    precision = all(v["rse"] is not None and v["rse"] <= protocol.reference_rse for v in (mc, rqmc))
    agreement = scale > 0 and difference+uncertainty <= protocol.pilot_agreement_margin*scale
    return {"precision_pass": precision, "agreement_diagnostic_pass": agreement,
            "absolute_difference": difference, "approximate_difference_radius": uncertainty,
            "family_adjusted_critical": critical,
            "status": "REFERENCE_FEASIBLE_DEVELOPMENT_ONLY" if precision and agreement
            else "UNRESOLVED_REFERENCE", "P2_qualified": False,
            "not_accuracy_or_mode_coverage_proof": True}


def verify_reference_record(protocol: GeometryProtocol, task_id: str,
                            row: dict[str, Any]) -> None:
    """Recompute saved summaries; this check does not replay all physical paths."""
    task = protocol.authorize(task_id, "FR2", "reference_pilot")
    method = row["method"]
    if method not in {"conditional_mc", "conditional_rqmc"}:
        raise ValueError("unknown stored reference method")
    expected_seeds = [protocol.seed(task_id, "reference_pilot", i, method)
                      for i in range(protocol.pilot_units)]
    if (row["seeds"] != expected_seeds or len(row["unit_means"]) != protocol.pilot_units
            or row["estimand_digest"] != task.estimand_digest
            or row["path_count"] != protocol.pilot_units*2**protocol.pilot_power):
        raise ValueError("stored reference role/count/law mismatch")
    units = Moments.from_values(torch.tensor(row["unit_means"], dtype=torch.float64))
    pooled = Moments(**row["pointwise_moments"])
    if pooled.count != row["path_count"] or not math.isclose(pooled.mean, units.mean, rel_tol=1e-13, abs_tol=1e-16):
        raise ValueError("pooled moment binding changed")
    expected_se = pooled.standard_error if method == "conditional_mc" else units.standard_error
    expected_count = pooled.count if method == "conditional_mc" else units.count
    expected_type = "iid_path" if method == "conditional_mc" else "independent_scramble"
    expected_rse = expected_se/units.mean if units.mean > 0 else None
    if (row["mean"] != units.mean or row["se"] != expected_se
            or row["rse"] != expected_rse or row["block_se"] != units.standard_error
            or row["independent_unit_count"] != expected_count or row["unit_type"] != expected_type):
        raise ValueError("stored inference statistics changed")
