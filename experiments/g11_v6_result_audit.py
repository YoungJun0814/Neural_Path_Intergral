"""Offline V6 audit using only serialized sufficient statistics.

This module deliberately does not import the policy preparation, execution, routing,
or in-memory audit helpers.  It independently repeats their arithmetic from JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import yaml

from src.path_integral import exact_binomial_probability_interval
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.seed_ledger import SeedLedger

_SCHEMA = "npi.g11.v6-independent-audit.config.v1"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 independent-audit config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "frozen",
        "accepted_source_schemas",
        "tolerances",
        "requirements",
    }:
        raise ValueError("malformed V6 independent-audit config fields")
    schemas = payload["accepted_source_schemas"]
    if not isinstance(schemas, list) or not schemas or len(set(schemas)) != len(schemas):
        raise ValueError("accepted source schemas must be a nonempty unique list")
    return payload, hashlib.sha256(raw).hexdigest()


def _close(left: float, right: float, *, relative: float, absolute: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(
        left, right, rel_tol=relative, abs_tol=absolute
    )


def _work_ledger_hash(payload: dict[str, Any]) -> str:
    if set(payload) != {"schema", "records"}:
        raise ValueError("malformed serialized V6 work ledger")
    if payload["schema"] != "npi.g11.v6-work-ledger.v1":
        raise ValueError("unsupported serialized V6 work ledger")
    return _canonical_hash(payload)


def _work_total(records: list[dict[str, Any]]) -> float:
    values = [float(record["work_units"]) for record in records]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("invalid work-unit value")
    return math.fsum(values)


def _recompute_integer_allocations(
    preparation: dict[str, Any], *, relative: float, absolute: float
) -> tuple[bool, float]:
    core = preparation["core"]
    target = core["target"]
    target_variance = (
        float(target["nominal_probability"]) * float(target["relative_sampling_rmse"])
    ) ** 2
    if not math.isfinite(target_variance) or target_variance <= 0.0:
        return False, math.nan
    allocations = core["allocations"]
    floor = int(preparation["minimum_final_samples"])
    if floor < 2 or not isinstance(allocations, list) or not allocations:
        return False, math.nan

    expected: list[tuple[float, int]] = []
    if preparation["execution_method"] != "hybrid":
        design = preparation["audit_design"]
        if not isinstance(design, dict) or len(allocations) != 1:
            return False, math.nan
        continuous = float(design["design_variance"]) / target_variance
        expected.append((continuous, max(floor, math.ceil(continuous))))
    else:
        if preparation["audit_design"] is not None:
            return False, math.nan
        variances = [float(item["design_variance"]) for item in allocations]
        costs = [float(item["cost_per_sample"]) for item in allocations]
        if any(value < 0.0 or not math.isfinite(value) for value in variances) or any(
            value <= 0.0 or not math.isfinite(value) for value in costs
        ):
            return False, math.nan
        root_sum = math.fsum(
            math.sqrt(variance * cost)
            for variance, cost in zip(variances, costs, strict=True)
        )
        counts: list[int] = []
        continuous_counts: list[float] = []
        for variance, cost in zip(variances, costs, strict=True):
            continuous = (
                root_sum * math.sqrt(variance / cost) / target_variance
                if variance > 0.0
                else 0.0
            )
            continuous_counts.append(continuous)
            counts.append(max(floor, math.ceil(continuous)))
        iterations = 0
        while (
            math.fsum(
                variance / count
                for variance, count in zip(variances, counts, strict=True)
            )
            > target_variance * (1.0 + 1e-14)
        ):
            best = max(
                range(len(counts)),
                key=lambda index: (
                    (
                        variances[index] / counts[index]
                        - variances[index] / (counts[index] + 1)
                    )
                    / costs[index],
                    -index,
                ),
            )
            counts[best] += 1
            iterations += 1
            if iterations > 1_000_000:
                return False, math.nan
        expected.extend(zip(continuous_counts, counts, strict=True))

    valid = True
    for item, (continuous, count) in zip(allocations, expected, strict=True):
        variance = float(item["design_variance"])
        valid = valid and _close(
            float(item["continuous_count"]),
            continuous,
            relative=relative,
            absolute=absolute,
        )
        valid = valid and int(item["final_count"]) == count
        valid = valid and _close(
            float(item["design_sampling_variance"]),
            variance / count,
            relative=relative,
            absolute=absolute,
        )
    expected_final_work = math.fsum(
        int(item["final_count"]) * float(item["cost_per_sample"])
        for item in allocations
    )
    valid = valid and _close(
        float(core["expected_final_work"]),
        expected_final_work,
        relative=relative,
        absolute=absolute,
    )
    return valid, target_variance


def _core_preparation_hash(preparation: dict[str, Any]) -> str:
    core = preparation["core"]
    seed_hash = _canonical_hash(core["seed_ledger"])
    common = {
        "protocol": core["protocol"],
        "regime": core["regime"],
        "task": core["task"],
        "target": core["target"],
        "expected_final_work": core["expected_final_work"],
        "operation_work_cap": core["operation_work_cap"],
        "resource_censored": core["resource_censored"],
        "chunk_size": core["chunk_size"],
        "streams": core["streams"],
        "preparation_seed_ledger_hash": seed_hash,
        "preprocessing_work": core["work_entries"],
    }
    if preparation["execution_method"] == "hybrid":
        payload = {
            "schema": "npi.g11.hybrid-preparation.v1",
            **common,
            "selected_candidate": core["selected_candidate"],
            "selection": core["selection"],
            "allocations": core["allocations"],
        }
    else:
        payload = {
            "schema": "npi.g11.single-term-preparation.v1",
            **common,
            "method": preparation["execution_method"],
            "design": preparation["audit_design"],
            "allocation": core["allocations"][0],
        }
    return _canonical_hash(payload)


def _audit_router(record: dict[str, Any]) -> bool:
    route = record.get("route")
    inputs = record.get("router_inputs")
    if route is None and inputs is None:
        return True
    if not isinstance(route, dict) or not isinstance(inputs, dict):
        return False
    config = inputs["config"]
    probability_interval = exact_binomial_probability_interval(
        int(inputs["successes"]),
        int(inputs["trials"]),
        confidence_level=float(config["confidence_level"]),
    )
    interval = {
        "successes": probability_interval.successes,
        "trials": probability_interval.trials,
        "confidence_level": probability_interval.confidence_level,
        "lower": probability_interval.lower,
        "upper": probability_interval.upper,
    }
    cutoff = float(config["probability_cutoff"])
    if probability_interval.lower > cutoff:
        rarity_class = "moderate"
    elif probability_interval.upper < cutoff:
        rarity_class = "rare"
    else:
        rarity_class = "ambiguous"

    crude = inputs["crude_work"]
    dcs = inputs["dcs_work"]
    available = [item for item in (crude, dcs) if item is not None]
    default = "crude" if rarity_class == "moderate" else "dcs_slis"
    if available:
        best = min(
            available,
            key=lambda item: (
                float(item["point"]),
                0 if item["method"] == default else 1,
                item["method"],
            ),
        )
        best_method = best["method"]
        best_point = float(best["point"])
    else:
        best_method = None
        best_point = None

    effective_cap = None
    trials = int(inputs["trials"])
    saving = float(config["minimum_certified_relative_saving"])
    hybrid = inputs["hybrid_opportunity"]
    if rarity_class == "ambiguous" and trials < int(config["maximum_screening_trials"]):
        action = "continue_screening"
        reason = "probability interval straddles cutoff before the screening cap"
    elif rarity_class == "ambiguous":
        action = config["ambiguous_fallback"]
        reason = "probability interval remains ambiguous at the screening cap"
    elif rarity_class == "moderate":
        certified = (
            crude is not None
            and dcs is not None
            and float(dcs["upper"]) <= (1.0 - saving) * float(crude["lower"])
        )
        if certified:
            action = "dcs_slis"
            reason = "DCS-SLIS is certified cheaper despite a moderate event"
        else:
            action = "crude"
            reason = "moderate event does not justify Hybrid profiling"
    else:
        fallback = best_method or "dcs_slis"
        if hybrid is None or best_point is None:
            action = fallback
            reason = "rare event lacks an economically admissible Hybrid profile"
        else:
            effective_cap = min(
                float(config["maximum_hybrid_profile_work"]),
                float(hybrid["external_profile_work_cap"]),
                float(config["maximum_profile_fraction"]) * best_point,
            )
            potential_saving = (
                best_point - float(hybrid["optimistic_total_work"])
            ) / best_point
            if float(hybrid["minimum_profile_work"]) > effective_cap:
                action = fallback
                reason = "minimum Hybrid profiling work exceeds the frozen cap"
            elif potential_saving < saving:
                action = fallback
                reason = "optimistic Hybrid saving cannot repay the required margin"
            else:
                action = "profile_hybrid"
                reason = "rare event has an economically admissible Hybrid profile"

    expected_route = {
        "action": action,
        "rarity_class": rarity_class,
        "reason": reason,
        "probability_interval": interval,
        "screening_work": float(inputs["screening_work"]),
        "effective_profile_work_cap": effective_cap,
        "current_best_method": best_method,
        "current_best_point_work": best_point,
    }
    decision_payload = {
        "schema": "npi.g11.v6-rarity-route.v1",
        **expected_route,
        "config": config,
        "crude_work": crude,
        "dcs_work": dcs,
        "hybrid_opportunity": hybrid,
    }
    return all(route.get(key) == value for key, value in expected_route.items()) and route.get(
        "decision_hash"
    ) == _canonical_hash(decision_payload)


def _audit_defensive_design_certificate(
    record: dict[str, Any],
    *,
    relative: float,
    absolute: float,
    required: bool,
) -> bool:
    """Independently replay the V3 defensive second-moment design contract."""

    if str(record.get("method")) != "defensive_cem":
        return True
    certificate = record.get("defensive_design_certificate")
    if certificate is None:
        return not required
    if not isinstance(certificate, dict):
        return False
    expected_fields = {
        "schema",
        "nominal_probability",
        "nominal_probability_upper_multiplier",
        "probability_upper_bound",
        "reference_certificate_z",
        "reference_upper_bound",
        "certified",
        "absolute_bound",
        "familywise_alpha",
        "pilot_count",
        "pilot_mean",
        "pilot_variance",
        "rigorous_bounded_variance_upper",
        "structural_variance_upper",
        "selected_design_variance",
    }
    if set(certificate) != expected_fields:
        return False
    try:
        nominal = float(certificate["nominal_probability"])
        multiplier = float(certificate["nominal_probability_upper_multiplier"])
        probability_upper = float(certificate["probability_upper_bound"])
        certificate_z = float(certificate["reference_certificate_z"])
        reference_upper = float(certificate["reference_upper_bound"])
        bound = float(certificate["absolute_bound"])
        alpha = float(certificate["familywise_alpha"])
        count = int(certificate["pilot_count"])
        mean = float(certificate["pilot_mean"])
        variance = float(certificate["pilot_variance"])
        rigorous_upper = float(certificate["rigorous_bounded_variance_upper"])
        structural_upper = float(certificate["structural_variance_upper"])
        selected_variance = float(certificate["selected_design_variance"])
        diagnostics = record["pilot_tail_diagnostics"]
        if (
            certificate["schema"]
            != "npi.g11.v6-defensive-design-certificate.v1"
            or count < 2
            or not 0.0 < alpha < 1.0
            or bound <= 0.0
            or multiplier < 1.0
            or certificate_z <= 0.0
            or not all(
                math.isfinite(value)
                for value in (
                    nominal,
                    multiplier,
                    probability_upper,
                    certificate_z,
                    reference_upper,
                    bound,
                    alpha,
                    mean,
                    variance,
                    rigorous_upper,
                    structural_upper,
                    selected_variance,
                )
            )
        ):
            return False

        alpha_per_moment = alpha / 2.0
        second = variance + mean**2
        mean_radius = bound * math.sqrt(
            2.0 * math.log(2.0 / alpha_per_moment) / count
        )
        second_radius = bound**2 * math.sqrt(
            math.log(2.0 / alpha_per_moment) / (2.0 * count)
        )
        mean_lower = max(-bound, mean - mean_radius)
        mean_upper = min(bound, mean + mean_radius)
        second_lower = max(0.0, second - second_radius)
        second_upper = min(bound**2, second + second_radius)
        maximum_abs_mean = max(abs(mean_lower), abs(mean_upper))
        minimum_abs_mean = (
            0.0
            if mean_lower <= 0.0 <= mean_upper
            else min(abs(mean_lower), abs(mean_upper))
        )
        variance_lower = max(0.0, second_lower - maximum_abs_mean**2)
        recomputed_rigorous = max(
            0.0, min(bound**2, second_upper - minimum_abs_mean**2)
        )
        if variance_lower > recomputed_rigorous:
            recomputed_rigorous = bound**2

        recomputed_probability_upper = min(1.0, multiplier * nominal)
        recomputed_reference_upper = float(
            record["reference_probability"]
        ) + certificate_z * float(record["reference_standard_error"])
        recomputed_structural = bound * recomputed_probability_upper
        recomputed_selected = max(
            variance, min(recomputed_rigorous, recomputed_structural)
        )
        diagnostics_population_variance = (
            float(diagnostics["variance"]) * (count - 1) / count
        )
        return all(
            (
                nominal == float(record["nominal_probability"]),
                int(diagnostics["count"]) == count,
                _close(
                    mean,
                    float(diagnostics["mean"]),
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    variance,
                    diagnostics_population_variance,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    probability_upper,
                    recomputed_probability_upper,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    reference_upper,
                    recomputed_reference_upper,
                    relative=relative,
                    absolute=absolute,
                ),
                bool(certificate["certified"])
                == (recomputed_reference_upper <= recomputed_probability_upper),
                bool(certificate["certified"]),
                _close(
                    rigorous_upper,
                    recomputed_rigorous,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    structural_upper,
                    recomputed_structural,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    selected_variance,
                    recomputed_selected,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    selected_variance,
                    float(record["design"]["design_variance"]),
                    relative=relative,
                    absolute=absolute,
                ),
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def _audit_record(
    record: dict[str, Any],
    *,
    relative: float,
    absolute: float,
    require_defensive_design_certificate: bool = False,
) -> dict[str, Any]:
    preparation = record["preparation"]
    result = record["result"]
    core = result["core"]
    prep_core = preparation["core"]
    allocation_valid, target_variance = _recompute_integer_allocations(
        preparation, relative=relative, absolute=absolute
    )

    prep_seed_ledger = SeedLedger.from_dict(prep_core["seed_ledger"])
    final_seed_ledger = SeedLedger.from_dict(core["seed_ledger_payload"])
    prep_keys = {item.key for item in prep_seed_ledger.records}
    final_keys = {item.key for item in final_seed_ledger.records}
    seed_valid = (
        _canonical_hash(prep_core["seed_ledger"]) == prep_seed_ledger.sha256
        and _canonical_hash(core["seed_ledger_payload"]) == core["seed_ledger_hash"]
        and prep_keys.issubset(final_keys)
        and all(item.key.role != "final" for item in prep_seed_ledger.records)
        and all(
            item.key.role == "final"
            for item in final_seed_ledger.records
            if item.key not in prep_keys
        )
    )

    preprocessing = preparation["preprocessing_work"]
    preprocessing_hash = _work_ledger_hash(preprocessing)
    recomputed_core_hash = _core_preparation_hash(preparation)
    policy_payload = {
        "schema": "npi.g11.v6-policy-preparation.v1",
        "policy_name": preparation["policy_name"],
        "cell_id": preparation["cell_id"],
        "execution_method": preparation["execution_method"],
        "route": preparation["route"],
        "audit_design": preparation["audit_design"],
        "minimum_final_samples": preparation["minimum_final_samples"],
        "core_preparation_hash": prep_core["preparation_hash"],
        "preprocessing_work_sha256": preprocessing_hash,
    }
    identity_valid = (
        preparation["schema"] == "npi.g11.v6-policy-preparation.v1"
        and preparation["policy_hash"] == _canonical_hash(policy_payload)
        and result["policy_hash"] == preparation["policy_hash"]
        and result["policy_name"] == preparation["policy_name"]
        and result["cell_id"] == preparation["cell_id"]
        and result["execution_method"] == preparation["execution_method"]
    )
    preparation_hash_valid = recomputed_core_hash == prep_core["preparation_hash"]

    expected_total = _work_total(result["total_work"]["records"])
    core_total = _work_total(core["work"]["entries"])
    total_work_hash = _canonical_hash(
        {
            "schema": "npi.g11.v6-work-ledger.v1",
            "records": result["total_work"]["records"],
        }
    )
    result_payload = {
        "schema": "npi.g11.v6-policy-result.v1",
        "policy_hash": result["policy_hash"],
        "core_preparation_hash": core["preparation_hash"],
        "complete": core["complete"],
        "resource_censored": core["resource_censored"],
        "estimate": core["estimate"],
        "empirical_sampling_variance": core["empirical_sampling_variance"],
        "seed_ledger_hash": core["seed_ledger_hash"],
        "total_work_sha256": total_work_hash,
    }
    result_hash_valid = result["result_hash"] == _canonical_hash(result_payload)
    work_valid = _close(
        expected_total, core_total, relative=relative, absolute=absolute
    )

    prep_entries = prep_core["work_entries"]
    prep_work_units = _work_total(prep_entries)
    expected_censored = (
        prep_work_units + float(prep_core["expected_final_work"])
        > float(prep_core["operation_work_cap"])
    )
    censoring_valid = bool(prep_core["resource_censored"]) == expected_censored
    if expected_censored:
        censoring_valid = censoring_valid and isinstance(
            prep_core["censoring_reason"], str
        )
    else:
        censoring_valid = censoring_valid and prep_core["censoring_reason"] is None

    result_allocations = list(core["allocations"])
    allocation_identity = result_allocations == list(prep_core["allocations"])
    design_variance = math.fsum(
        float(item["design_variance"]) / int(item["final_count"])
        for item in result_allocations
    )
    result_statistics_valid = _close(
        float(core["design_sampling_variance"]),
        design_variance,
        relative=relative,
        absolute=absolute,
    ) and bool(core["design_target_attained"]) == (design_variance <= target_variance)
    if core["complete"]:
        terms = core["terms"]
        estimate = math.fsum(float(item["mean"]) for item in terms)
        empirical = math.fsum(float(item["variance"]) / int(item["count"]) for item in terms)
        standard_error = math.sqrt(max(0.0, empirical))
        critical = NormalDist().inv_cdf(
            0.5 + float(prep_core["target"]["confidence_level"]) / 2.0
        )
        interval = (estimate - critical * standard_error, estimate + critical * standard_error)
        result_statistics_valid = result_statistics_valid and all(
            (
                len(terms) == len(result_allocations),
                [int(item["count"]) for item in terms]
                == [int(item["final_count"]) for item in result_allocations],
                _close(float(core["estimate"]), estimate, relative=relative, absolute=absolute),
                _close(
                    float(core["empirical_sampling_variance"]),
                    empirical,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(core["standard_error"]),
                    standard_error,
                    relative=relative,
                    absolute=absolute,
                ),
                all(
                    _close(float(actual), expected, relative=relative, absolute=absolute)
                    for actual, expected in zip(
                        core["asymptotic_confidence_interval"], interval, strict=True
                    )
                ),
                bool(core["empirical_target_attained"]) == (empirical <= target_variance),
            )
        )
    else:
        result_statistics_valid = result_statistics_valid and core["estimate"] is None

    checks = {
        "router_recomputed": _audit_router(record),
        "identity_and_policy_hash": identity_valid,
        "core_preparation_hash": preparation_hash_valid,
        "integer_allocation": allocation_valid,
        "allocation_identity": allocation_identity,
        "censoring_arithmetic": censoring_valid,
        "result_hash": result_hash_valid,
        "work_recomputed": work_valid,
        "seed_ledger_and_roles": seed_valid,
        "result_statistics": result_statistics_valid,
        "defensive_design_certificate": _audit_defensive_design_certificate(
            record,
            relative=relative,
            absolute=absolute,
            required=require_defensive_design_certificate,
        ),
    }
    return {
        "cell_id": str(record["cell_id"]),
        "cluster": int(record["cluster"]),
        "method": str(record.get("method", "v6_policy")),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(config_path: Path, source_path: Path) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    raw = source_path.read_bytes()
    source = json.loads(raw)
    if not isinstance(source, dict) or source.get("schema") not in set(
        config["accepted_source_schemas"]
    ):
        raise ValueError("unsupported source artifact for V6 independent audit")
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source artifact must contain records")
    tolerance = config["tolerances"]
    audits = [
        _audit_record(
            record,
            relative=float(tolerance["relative"]),
            absolute=float(tolerance["absolute"]),
            require_defensive_design_certificate=(
                source.get("protocol_id") == "g11-v6-baseline-qualification-v3"
            ),
        )
        for record in records
    ]
    requirements = config["requirements"]
    complete = all(bool(record["result"]["core"]["complete"]) for record in records)
    uncensored = all(
        not bool(record["result"]["core"]["resource_censored"]) for record in records
    )
    smoke = bool(source.get("smoke", False))
    gates = {
        "all_records_pass": all(record["passed"] for record in audits),
        "complete_if_required": complete or not bool(requirements["require_complete"]),
        "uncensored_if_required": uncensored
        or not bool(requirements["require_uncensored"]),
        "non_smoke_if_required": not smoke
        or not bool(requirements["require_non_smoke_for_qualification"]),
    }
    provenance = source_provenance()
    formal = {
        "frozen_audit_config": bool(config["frozen"]),
        "clean_source": not bool(provenance["dirty_worktree"]),
    }
    return {
        "schema": "npi.g11.v6-independent-audit.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "source_schema": source["schema"],
        "source_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "smoke": smoke,
        "records": audits,
        "gates": gates,
        "audit_passed": all(gates.values()),
        "formal_readiness": formal,
        "qualification_audit_passed": all(gates.values()) and all(formal.values()),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/result_audit_development.yaml"),
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.source)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"passed": result["audit_passed"], **result["gates"]}))


if __name__ == "__main__":
    main()
