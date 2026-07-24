"""Offline V6 audit using only serialized sufficient statistics.

This module deliberately does not import the policy preparation, execution, routing,
or in-memory audit helpers.  It independently repeats their arithmetic from JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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
    if (
        certificate.get("schema")
        == "npi.g11.v6-defensive-plugin-design-certificate.v1"
    ):
        plugin_fields = {
            "schema",
            "nominal_probability",
            "nominal_probability_upper_multiplier",
            "probability_upper_bound",
            "reference_certificate_z",
            "reference_upper_bound",
            "certified",
            "absolute_bound",
            "pilot_count",
            "pilot_mean",
            "pilot_variance",
            "variance_safety_factor",
            "zero_variance_fallback",
            "structural_variance_upper_diagnostic",
            "selected_design_variance",
        }
        if set(certificate) != plugin_fields:
            return False
        try:
            nominal = float(certificate["nominal_probability"])
            multiplier = float(
                certificate["nominal_probability_upper_multiplier"]
            )
            probability_upper = float(certificate["probability_upper_bound"])
            certificate_z = float(certificate["reference_certificate_z"])
            reference_upper = float(certificate["reference_upper_bound"])
            bound = float(certificate["absolute_bound"])
            count = int(certificate["pilot_count"])
            mean = float(certificate["pilot_mean"])
            variance = float(certificate["pilot_variance"])
            safety = float(certificate["variance_safety_factor"])
            fallback = float(certificate["zero_variance_fallback"])
            structural = float(
                certificate["structural_variance_upper_diagnostic"]
            )
            selected = float(certificate["selected_design_variance"])
            diagnostics = record["pilot_tail_diagnostics"]
            recomputed_probability_upper = min(1.0, multiplier * nominal)
            recomputed_reference_upper = float(
                record["reference_probability"]
            ) + certificate_z * float(record["reference_standard_error"])
            recomputed_structural = bound * recomputed_probability_upper
            recomputed_fallback = nominal**2
            recomputed_selected = max(safety * variance, recomputed_fallback)
            return all(
                (
                    count >= 2,
                    multiplier >= 1.0,
                    certificate_z > 0.0,
                    bound > 0.0,
                    safety >= 1.0,
                    all(
                        math.isfinite(value)
                        for value in (
                            nominal,
                            multiplier,
                            probability_upper,
                            certificate_z,
                            reference_upper,
                            bound,
                            mean,
                            variance,
                            safety,
                            fallback,
                            structural,
                            selected,
                        )
                    ),
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
                        float(diagnostics["variance"]),
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
                    == (
                        recomputed_reference_upper
                        <= recomputed_probability_upper
                    ),
                    bool(certificate["certified"]),
                    _close(
                        fallback,
                        recomputed_fallback,
                        relative=relative,
                        absolute=absolute,
                    ),
                    _close(
                        structural,
                        recomputed_structural,
                        relative=relative,
                        absolute=absolute,
                    ),
                    _close(
                        selected,
                        recomputed_selected,
                        relative=relative,
                        absolute=absolute,
                    ),
                    _close(
                        selected,
                        float(record["design"]["design_variance"]),
                        relative=relative,
                        absolute=absolute,
                    ),
                )
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
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


def _audit_crude_design_certificate(
    record: dict[str, Any],
    *,
    relative: float,
    absolute: float,
    required: bool,
) -> bool:
    """Independently replay the V4 crude rarity-band variance design."""

    if str(record.get("method")) != "crude":
        return True
    certificate = record.get("crude_design_certificate")
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
        "pilot_count",
        "pilot_mean",
        "pilot_variance",
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
        count = int(certificate["pilot_count"])
        mean = float(certificate["pilot_mean"])
        variance = float(certificate["pilot_variance"])
        structural_upper = float(certificate["structural_variance_upper"])
        selected_variance = float(certificate["selected_design_variance"])
        diagnostics = record["pilot_tail_diagnostics"]
        if (
            certificate["schema"] != "npi.g11.v6-crude-design-certificate.v1"
            or count < 2
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
                    mean,
                    variance,
                    structural_upper,
                    selected_variance,
                )
            )
        ):
            return False
        recomputed_probability_upper = min(1.0, multiplier * nominal)
        recomputed_reference_upper = float(
            record["reference_probability"]
        ) + certificate_z * float(record["reference_standard_error"])
        recomputed_structural = (
            0.25
            if recomputed_probability_upper >= 0.5
            else recomputed_probability_upper
            * (1.0 - recomputed_probability_upper)
        )
        recomputed_selected = max(variance, recomputed_structural)
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
                    float(diagnostics["variance"]),
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


def _audit_replicated_direct_certificate(
    record: dict[str, Any],
    *,
    relative: float,
    absolute: float,
    required: bool,
) -> bool:
    """Replay V2/V4 direct point-variance planning without production helpers."""

    certificate = record.get("planning_certificate")
    if certificate is None:
        certificate = record.get("direct_planning_certificate")
    if certificate is None:
        return not required
    if not isinstance(certificate, dict):
        return False
    expected_fields = {
        "schema",
        "method",
        "planning_replicates",
        "samples_per_replicate",
        "variance_statistic",
        "replicate_variances",
        "planning_variance",
        "pooled_count",
        "pooled_mean",
        "pooled_variance",
        "variance_safety_factor",
        "zero_variance_fallback",
        "selected_design_variance",
        "absolute_bound",
        "pilot_values_within_nonnegative_bound",
        "nominal_probability",
        "nominal_probability_upper_multiplier",
        "probability_upper_bound",
        "reference_certificate_z",
        "reference_upper_bound",
        "structural_variance_upper_diagnostic",
        "structural_bound_used_for_allocation",
        "certified",
    }
    if set(certificate) != expected_fields:
        return False
    try:
        if (
            certificate["schema"]
            != "npi.g11.v6-replicated-direct-plugin-certificate.v1"
        ):
            return False
        replicates = int(certificate["planning_replicates"])
        per_replicate = int(certificate["samples_per_replicate"])
        replicate_variances = tuple(
            float(value) for value in certificate["replicate_variances"]
        )
        statistic = str(certificate["variance_statistic"])
        if (
            replicates < 3
            or per_replicate < 2
            or len(replicate_variances) != replicates
            or any(
                not math.isfinite(value) or value < 0.0
                for value in replicate_variances
            )
        ):
            return False
        if statistic == "mean_replicate_variance":
            planning = statistics.fmean(replicate_variances)
        elif statistic == "median_replicate_variance":
            planning = statistics.median(replicate_variances)
        else:
            return False
        safety = float(certificate["variance_safety_factor"])
        fallback = float(certificate["zero_variance_fallback"])
        selected = max(safety * planning, fallback)
        nominal = float(certificate["nominal_probability"])
        multiplier = float(
            certificate["nominal_probability_upper_multiplier"]
        )
        probability_upper = min(1.0, multiplier * nominal)
        reference_z = float(certificate["reference_certificate_z"])
        reference_upper = (
            float(record["reference_probability"])
            + reference_z * float(record["reference_standard_error"])
        )
        bound = float(certificate["absolute_bound"])
        structural = bound * probability_upper
        method = str(certificate["method"])
        expected_fallback = (
            nominal * (1.0 - nominal)
            if method == "crude"
            else nominal**2
        )
        design = record.get("design")
        if design is None:
            design = record["preparation"]["audit_design"]
        if not isinstance(design, dict):
            return False
        bounded = bool(certificate["pilot_values_within_nonnegative_bound"])
        expected_certified = bounded and reference_upper <= probability_upper
        return all(
            (
                safety >= 1.0,
                fallback > 0.0,
                bound > 0.0,
                0.0 < nominal <= 1.0,
                multiplier >= 1.0,
                reference_z > 0.0,
                int(certificate["pooled_count"])
                == replicates * per_replicate,
                int(design["pilot_count"]) == replicates * per_replicate,
                _close(
                    float(certificate["planning_variance"]),
                    planning,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    fallback,
                    expected_fallback,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(certificate["selected_design_variance"]),
                    selected,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(design["design_variance"]),
                    selected,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(certificate["pooled_mean"]),
                    float(design["pilot_mean"]),
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(certificate["pooled_variance"]),
                    float(design["pilot_variance"]),
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(design["absolute_bound"]),
                    bound,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(certificate["probability_upper_bound"]),
                    probability_upper,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(certificate["reference_upper_bound"]),
                    reference_upper,
                    relative=relative,
                    absolute=absolute,
                ),
                _close(
                    float(
                        certificate[
                            "structural_variance_upper_diagnostic"
                        ]
                    ),
                    structural,
                    relative=relative,
                    absolute=absolute,
                ),
                certificate["structural_bound_used_for_allocation"] is False,
                bool(certificate["certified"]) == expected_certified,
                bool(certificate["certified"]),
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def _fixed_hybrid_work_audit(
    variances: tuple[float, ...],
    costs: tuple[float, ...],
    *,
    target: float,
    minimum_final_samples: int,
) -> float:
    """Independently replay the integer point-work calculation."""

    root_sum = math.fsum(
        math.sqrt(variance * cost)
        for variance, cost in zip(variances, costs, strict=True)
    )
    counts = [
        max(
            minimum_final_samples,
            math.ceil(root_sum * math.sqrt(variance / cost) / target)
            if variance > 0.0
            else minimum_final_samples,
        )
        for variance, cost in zip(variances, costs, strict=True)
    ]
    while (
        math.fsum(
            variance / count
            for variance, count in zip(variances, counts, strict=True)
        )
        > target * (1.0 + 1e-14)
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
    return math.fsum(
        count * cost for count, cost in zip(counts, costs, strict=True)
    )


def _audit_replicated_hybrid_selection(
    record: dict[str, Any],
    *,
    relative: float,
    absolute: float,
    required: bool,
) -> bool:
    """Replay V4 hybrid point-work selection and design variances."""

    if not required:
        return True
    selection = record.get("selection")
    if selection is None:
        return not required
    if not isinstance(selection, dict):
        return False
    try:
        if (
            selection.get("schema")
            != "npi.g11.v6-replicated-planning-selection.v1"
            or selection.get("finite_sample_work_certificate") is not False
        ):
            return False
        replicates = int(selection["planning_replicates"])
        samples = int(selection["samples_per_replicate"])
        statistic = str(selection["variance_statistic"])
        replicate_map = selection["profile_replicate_variances"]
        serialized_planning = selection["profile_planning_variances"]
        if (
            replicates < 3
            or samples < 2
            or int(selection["cumulative_sample_count"])
            != replicates * samples
            or not isinstance(replicate_map, dict)
            or set(replicate_map) != set(serialized_planning)
        ):
            return False
        planning: dict[str, float] = {}
        for profile_id, raw_values in replicate_map.items():
            values = tuple(float(value) for value in raw_values)
            if len(values) != replicates or any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                return False
            if statistic == "mean_replicate_variance":
                planning[profile_id] = statistics.fmean(values)
            elif statistic == "median_replicate_variance":
                planning[profile_id] = statistics.median(values)
            else:
                return False
        if not all(
            _close(
                planning[profile_id],
                float(serialized_planning[profile_id]),
                relative=relative,
                absolute=absolute,
            )
            for profile_id in planning
        ):
            return False
        profiles = selection["profiles"]
        costs = {
            str(profile["profile_id"]): float(profile["cost_per_sample"])
            for profile in profiles
        }
        if set(costs) != set(planning) or any(
            not math.isfinite(cost) or cost <= 0.0 for cost in costs.values()
        ):
            return False
        if any(
            int(profile["moments"]["sample_count"])
            != replicates * samples
            for profile in profiles
        ):
            return False
        safety = float(selection["allocation_safety_factor"])
        tolerance = float(
            selection["practical_equivalence_relative_tolerance"]
        )
        minimum_final = int(selection["minimum_final_samples"])
        if safety < 1.0 or tolerance < 0.0 or minimum_final < 2:
            return False
        target_payload = record["preparation"]["core"]["target"]
        target = (
            float(target_payload["nominal_probability"])
            * float(target_payload["relative_sampling_rmse"])
        ) ** 2
        selector_work = float(selection["cumulative_profile_work"])
        preprocessing_records = record["preparation"]["preprocessing_work"][
            "records"
        ]
        serialized_selector_work = math.fsum(
            float(item["work_units"])
            for item in preprocessing_records
            if item["category"] == "selector_profile"
        )
        if not _close(
            selector_work,
            serialized_selector_work,
            relative=relative,
            absolute=absolute,
        ):
            return False
        common_prework = _work_total(preprocessing_records) - selector_work
        candidate_work: dict[str, float] = {}
        for candidate in selection["candidate_point_work"]:
            start = int(str(candidate).split("_")[1])
            term_ids = (f"single_{start}",) + tuple(
                f"correction_{level}"
                for level in range(start + 1, max(
                    int(item.split("_")[1])
                    for item in planning
                    if item.startswith("single_")
                ) + 1)
            )
            if any(term_id not in planning for term_id in term_ids):
                return False
            candidate_work[candidate] = (
                common_prework
                + selector_work
                + _fixed_hybrid_work_audit(
                    tuple(safety * planning[item] for item in term_ids),
                    tuple(costs[item] for item in term_ids),
                    target=target,
                    minimum_final_samples=minimum_final,
                )
            )
        if not all(
            _close(
                candidate_work[candidate],
                float(selection["candidate_point_work"][candidate]),
                relative=relative,
                absolute=absolute,
            )
            for candidate in candidate_work
        ):
            return False
        best = min(candidate_work, key=lambda item: (candidate_work[item], item))
        simplest = f"start_{max(int(item.split('_')[1]) for item in candidate_work)}"
        if set(candidate_work) != {
            f"start_{level}"
            for level in range(int(simplest.split("_")[1]) + 1)
        }:
            return False
        selected = (
            simplest
            if candidate_work[simplest]
            <= (1.0 + tolerance) * candidate_work[best]
            else best
        )
        decision = selection["frozen_decision"]
        if (
            decision["selected_candidate"] != selected
            or int(decision["look_index"]) != replicates - 1
            or set(decision["surviving_candidates"]) != set(candidate_work)
            or not _close(
                float(decision["selected_point_work"]),
                candidate_work[selected],
                relative=relative,
                absolute=absolute,
            )
            or not all(
                _close(
                    float(value),
                    candidate_work[selected],
                    relative=relative,
                    absolute=absolute,
                )
                for value in decision["selected_work_interval"]
            )
            or not _close(
                float(decision["worst_case_interval_regret_bound"]),
                candidate_work[selected] / candidate_work[best],
                relative=relative,
                absolute=absolute,
            )
        ):
            return False
        start = int(selected.split("_")[1])
        finest = max(
            int(item.split("_")[1])
            for item in planning
            if item.startswith("single_")
        )
        selected_ids = (f"single_{start}",) + tuple(
            f"correction_{level}" for level in range(start + 1, finest + 1)
        )
        allocations = record["preparation"]["core"]["allocations"]
        if [item["profile_id"] for item in allocations] != list(selected_ids):
            return False
        return all(
            _close(
                float(allocation["design_variance"]),
                safety * planning[profile_id],
                relative=relative,
                absolute=absolute,
            )
            for profile_id, allocation in zip(
                selected_ids, allocations, strict=True
            )
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        IndexError,
    ):
        return False


_V6_BASELINE_OPERATIONAL_GATE_NAMES = (
    "complete_matrix",
    "all_runs_complete",
    "no_resource_censoring",
    "all_design_targets_attained",
    "all_cem_training_charged",
    "all_cem_fits_converged",
    "all_cem_controls_finite_and_bounded",
    "all_defensive_designs_certified",
    "all_crude_designs_certified",
    "all_final_seed_roles_separate",
)
_V6_BASELINE_ALL_GATE_NAMES = (
    *_V6_BASELINE_OPERATIONAL_GATE_NAMES[:4],
    "all_empirical_targets_attained",
    *_V6_BASELINE_OPERATIONAL_GATE_NAMES[4:],
)
_V6_AGGREGATE_ACCURACY_RULE = (
    "deferred_to_prespecified_method_cell_attainment_and_bootstrap_rmse_co_gates"
)
_V6_SECONDARY_OPERATIONAL_GATE_NAMES = (
    "complete_matrix",
    "all_runs_complete",
    "no_resource_censoring",
    "all_design_targets_attained",
    "all_independent_audits",
    "smoothing_pair_present",
    "all_planning_certificates_valid",
    "all_final_seed_roles_separate",
)
_V6_ROUTED_OPERATIONAL_GATE_NAMES = (
    "complete_matrix",
    "all_routes_resolved",
    "all_runs_complete",
    "all_design_targets_attained",
    "no_resource_censoring",
    "all_independent_audits",
    "median_selection_fraction",
    "p90_selection_fraction",
    "reference_not_used_by_router_schema",
    "all_planning_certificates_valid",
    "all_final_seed_roles_separate",
)


def _audit_v6_baseline_summary(source: dict[str, Any]) -> bool:
    """Replay the V6 operational/diagnostic gate split without production helpers."""

    if source.get("protocol_id") != "g11-v6-baseline-qualification-v6":
        return True
    try:
        records = source["records"]
        contract = source["qualification_contract"]
        if not isinstance(records, list) or not isinstance(contract, dict):
            return False
        expected_contract_fields = {
            "schema",
            "expected_cell_ids",
            "expected_clusters",
            "methods",
            "control_bound",
            "operational_gate_names",
            "per_record_empirical_target_role",
            "aggregate_accuracy_protocol_id",
        }
        if set(contract) != expected_contract_fields:
            return False
        cells = contract["expected_cell_ids"]
        clusters = int(contract["expected_clusters"])
        methods = contract["methods"]
        control_bound = float(contract["control_bound"])
        if (
            contract["schema"]
            != "npi.g11.v6-baseline-qualification-contract.v1"
            or not isinstance(cells, list)
            or not cells
            or len(cells) != len(set(cells))
            or not all(isinstance(cell, str) and cell for cell in cells)
            or clusters < 1
            or not isinstance(methods, list)
            or set(methods) != {"crude", "pure_cem", "defensive_cem"}
            or len(methods) != 3
            or not math.isfinite(control_bound)
            or control_bound <= 0.0
            or contract["operational_gate_names"]
            != list(_V6_BASELINE_OPERATIONAL_GATE_NAMES)
            or contract["per_record_empirical_target_role"]
            != _V6_AGGREGATE_ACCURACY_RULE
            or not isinstance(contract["aggregate_accuracy_protocol_id"], str)
            or not contract["aggregate_accuracy_protocol_id"]
            or source.get("methods") != methods
        ):
            return False
        expected_keys = {
            (cell, cluster, method)
            for cell in cells
            for cluster in range(clusters)
            for method in methods
        }
        actual_keys = [
            (
                str(record["cell_id"]),
                int(record["cluster"]),
                str(record["method"]),
            )
            for record in records
        ]
        matrix_complete = len(actual_keys) == len(set(actual_keys)) and set(
            actual_keys
        ) == expected_keys
        recomputed = {
            "complete_matrix": matrix_complete,
            "all_runs_complete": all(
                bool(record["result"]["core"]["complete"]) for record in records
            ),
            "no_resource_censoring": all(
                not bool(record["result"]["core"]["resource_censored"])
                for record in records
            ),
            "all_design_targets_attained": all(
                bool(record["result"]["core"]["design_target_attained"])
                for record in records
            ),
            "all_empirical_targets_attained": all(
                bool(record["result"]["core"]["empirical_target_attained"])
                for record in records
            ),
            "all_cem_training_charged": all(
                record["method"] == "crude"
                or record["result"]["total_work"]["records"][0]["category"]
                == "proposal_training"
                for record in records
            ),
            "all_cem_fits_converged": all(
                record["method"] == "crude"
                or (
                    isinstance(record["cem_fit"], dict)
                    and record["cem_fit"]["converged"] is True
                )
                for record in records
            ),
            "all_cem_controls_finite_and_bounded": all(
                record["method"] == "crude"
                or (
                    isinstance(record["cem_fit"], dict)
                    and all(
                        math.isfinite(float(value))
                        and abs(float(value)) <= control_bound
                        for segment in record["cem_fit"]["control"]
                        for value in segment
                    )
                )
                for record in records
            ),
            "all_defensive_designs_certified": all(
                record["method"] != "defensive_cem"
                or (
                    isinstance(record.get("defensive_design_certificate"), dict)
                    and record["defensive_design_certificate"].get("certified") is True
                )
                for record in records
            ),
            "all_crude_designs_certified": all(
                record["method"] != "crude"
                or (
                    isinstance(record.get("crude_design_certificate"), dict)
                    and record["crude_design_certificate"].get("certified") is True
                )
                for record in records
            ),
            "all_final_seed_roles_separate": all(
                all(
                    seed["key"]["role"] == "final"
                    for seed in record["result"]["core"]["seed_ledger_payload"][
                        "records"
                    ]
                    if seed["key"]["role"]
                    not in {"proposal-training", "allocation-pilot"}
                )
                for record in records
            ),
        }
        serialized_gates = source["gates"]
        qualification_gates = source["qualification_gates"]
        if (
            set(serialized_gates) != set(_V6_BASELINE_ALL_GATE_NAMES)
            or serialized_gates != recomputed
            or qualification_gates
            != {
                name: recomputed[name]
                for name in _V6_BASELINE_OPERATIONAL_GATE_NAMES
            }
        ):
            return False
        formal = source["formal_readiness"]
        expected_decision = all(qualification_gates.values()) and all(
            bool(value) for value in formal.values()
        )
        return bool(source["baseline_qualified"]) == expected_decision
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False


def _audit_v6_secondary_summary(source: dict[str, Any]) -> bool:
    """Replay the secondary V2 operational/diagnostic gate split."""

    if (
        source.get("config_schema")
        != "npi.g11.v6-secondary-baselines.config.v2"
    ):
        return True
    try:
        records = source["records"]
        contract = source["qualification_contract"]
        if not isinstance(records, list) or not isinstance(contract, dict):
            return False
        if set(contract) != {
            "schema",
            "expected_cell_ids",
            "expected_clusters",
            "methods",
            "operational_gate_names",
            "per_record_empirical_target_role",
            "aggregate_accuracy_protocol_id",
        }:
            return False
        cells = contract["expected_cell_ids"]
        clusters = int(contract["expected_clusters"])
        methods = contract["methods"]
        if (
            contract["schema"]
            != "npi.g11.v6-secondary-qualification-contract.v1"
            or not isinstance(cells, list)
            or not cells
            or len(cells) != len(set(cells))
            or clusters < 1
            or methods != ["fixed_dcs_slis", "fixed_raw_defensive"]
            or source.get("methods") != methods
            or contract["operational_gate_names"]
            != list(_V6_SECONDARY_OPERATIONAL_GATE_NAMES)
            or contract["per_record_empirical_target_role"]
            != _V6_AGGREGATE_ACCURACY_RULE
            or not isinstance(contract["aggregate_accuracy_protocol_id"], str)
            or not contract["aggregate_accuracy_protocol_id"]
        ):
            return False
        expected_keys = {
            (cell, cluster, method)
            for cell in cells
            for cluster in range(clusters)
            for method in methods
        }
        actual_keys = [
            (
                str(record["cell_id"]),
                int(record["cluster"]),
                str(record["method"]),
            )
            for record in records
        ]
        recomputed = {
            "complete_matrix": len(actual_keys) == len(set(actual_keys))
            and set(actual_keys) == expected_keys,
            "all_runs_complete": all(
                bool(record["result"]["core"]["complete"])
                for record in records
            ),
            "no_resource_censoring": all(
                not bool(record["result"]["core"]["resource_censored"])
                for record in records
            ),
            "all_design_targets_attained": all(
                bool(record["result"]["core"]["design_target_attained"])
                for record in records
            ),
            "all_empirical_targets_attained": all(
                bool(record["result"]["core"]["empirical_target_attained"])
                for record in records
            ),
            "all_independent_audits": all(
                bool(record["audit"]["passed"]) for record in records
            ),
            "smoothing_pair_present": set(methods)
            == {"fixed_dcs_slis", "fixed_raw_defensive"},
            "all_planning_certificates_valid": all(
                isinstance(record.get("planning_certificate"), dict)
                and record["planning_certificate"].get("certified") is True
                and record["planning_certificate"].get(
                    "structural_bound_used_for_allocation"
                )
                is False
                for record in records
            ),
            "all_final_seed_roles_separate": all(
                all(
                    seed["key"]["role"] == "final"
                    for seed in record["result"]["core"][
                        "seed_ledger_payload"
                    ]["records"]
                    if seed["key"]["role"]
                    not in {"proposal-training", "allocation-pilot"}
                )
                for record in records
            ),
        }
        gates = source["gates"]
        qualification = source["qualification_gates"]
        expected_gate_names = {
            *_V6_SECONDARY_OPERATIONAL_GATE_NAMES,
            "all_empirical_targets_attained",
        }
        if (
            set(gates) != expected_gate_names
            or gates != recomputed
            or qualification
            != {
                name: recomputed[name]
                for name in _V6_SECONDARY_OPERATIONAL_GATE_NAMES
            }
        ):
            return False
        expected_decision = all(qualification.values()) and all(
            bool(value) for value in source["formal_readiness"].values()
        )
        return (
            bool(source["secondary_baselines_qualified"])
            == expected_decision
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False


def _linear_quantile_audit(
    values: list[float], probability: float
) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _audit_v6_routed_summary(source: dict[str, Any]) -> bool:
    """Replay the routed V4 operational/diagnostic gate split."""

    if source.get("config_schema") != "npi.g11.v6-routed-policy.config.v4":
        return True
    try:
        records = source["records"]
        contract = source["qualification_contract"]
        if not isinstance(records, list) or not isinstance(contract, dict):
            return False
        if set(contract) != {
            "schema",
            "expected_cell_ids",
            "expected_clusters",
            "maximum_median_selection_fraction",
            "maximum_p90_selection_fraction",
            "operational_gate_names",
            "per_record_empirical_target_role",
            "aggregate_accuracy_protocol_id",
        }:
            return False
        cells = contract["expected_cell_ids"]
        clusters = int(contract["expected_clusters"])
        median_limit = float(
            contract["maximum_median_selection_fraction"]
        )
        p90_limit = float(contract["maximum_p90_selection_fraction"])
        if (
            contract["schema"]
            != "npi.g11.v6-routed-policy-qualification-contract.v1"
            or not isinstance(cells, list)
            or not cells
            or len(cells) != len(set(cells))
            or clusters < 1
            or not 0.0 <= median_limit <= 1.0
            or not 0.0 <= p90_limit <= 1.0
            or contract["operational_gate_names"]
            != list(_V6_ROUTED_OPERATIONAL_GATE_NAMES)
            or contract["per_record_empirical_target_role"]
            != _V6_AGGREGATE_ACCURACY_RULE
            or not isinstance(contract["aggregate_accuracy_protocol_id"], str)
            or not contract["aggregate_accuracy_protocol_id"]
        ):
            return False
        expected_keys = {
            (cell, cluster)
            for cell in cells
            for cluster in range(clusters)
        }
        actual_keys = [
            (str(record["cell_id"]), int(record["cluster"]))
            for record in records
        ]
        fractions = [
            float(record["selection_work_fraction"]) for record in records
        ]
        median = _linear_quantile_audit(fractions, 0.5)
        p90 = _linear_quantile_audit(fractions, 0.9)
        planning_valid = all(
            (
                record["route"]["action"] == "profile_hybrid"
                and isinstance(record.get("selection"), dict)
                and record["selection"].get("schema")
                == "npi.g11.v6-replicated-planning-selection.v1"
                and record["selection"].get("finite_sample_work_certificate")
                is False
            )
            or (
                record["route"]["action"] in {"dcs_slis", "crude"}
                and isinstance(
                    record.get("direct_planning_certificate"), dict
                )
                and record["direct_planning_certificate"].get("certified")
                is True
                and record["direct_planning_certificate"].get(
                    "structural_bound_used_for_allocation"
                )
                is False
            )
            for record in records
        )
        allowed_preparation_roles = {
            "proposal-training",
            "router-screening",
            "allocation-pilot",
            "selector-planning",
            "selector-profile",
        }
        recomputed = {
            "complete_matrix": len(actual_keys) == len(set(actual_keys))
            and set(actual_keys) == expected_keys,
            "all_routes_resolved": all(
                record["route"]["action"] != "continue_screening"
                for record in records
            ),
            "all_runs_complete": all(
                bool(record["result"]["core"]["complete"])
                for record in records
            ),
            "all_design_targets_attained": all(
                bool(record["result"]["core"]["design_target_attained"])
                for record in records
            ),
            "all_empirical_targets_attained": all(
                bool(record["result"]["core"]["empirical_target_attained"])
                for record in records
            ),
            "no_resource_censoring": all(
                not bool(record["result"]["core"]["resource_censored"])
                for record in records
            ),
            "all_independent_audits": all(
                bool(record["audit"]["passed"]) for record in records
            ),
            "median_selection_fraction": median is not None
            and median <= median_limit,
            "p90_selection_fraction": p90 is not None
            and p90 <= p90_limit,
            "reference_not_used_by_router_schema": all(
                all("reference" not in key for key in record["route"])
                for record in records
            ),
            "all_planning_certificates_valid": planning_valid,
            "all_final_seed_roles_separate": all(
                all(
                    seed["key"]["role"] == "final"
                    for seed in record["result"]["core"][
                        "seed_ledger_payload"
                    ]["records"]
                    if seed["key"]["role"]
                    not in allowed_preparation_roles
                )
                for record in records
            ),
        }
        summary = source["selection_fraction_summary"]
        if median is None or p90 is None:
            return False
        if not (
            _close(
                float(summary["median"]),
                median,
                relative=1e-12,
                absolute=1e-15,
            )
            and _close(
                float(summary["p90"]),
                p90,
                relative=1e-12,
                absolute=1e-15,
            )
        ):
            return False
        gates = source["gates"]
        qualification = source["qualification_gates"]
        expected_gate_names = {
            *_V6_ROUTED_OPERATIONAL_GATE_NAMES,
            "all_empirical_targets_attained",
        }
        if (
            set(gates) != expected_gate_names
            or gates != recomputed
            or qualification
            != {
                name: recomputed[name]
                for name in _V6_ROUTED_OPERATIONAL_GATE_NAMES
            }
        ):
            return False
        expected_decision = all(qualification.values()) and all(
            bool(value) for value in source["formal_readiness"].values()
        )
        return bool(source["policy_qualified"]) == expected_decision
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        OverflowError,
    ):
        return False


def _audit_v6_summary(source: dict[str, Any]) -> bool:
    return (
        _audit_v6_baseline_summary(source)
        and _audit_v6_secondary_summary(source)
        and _audit_v6_routed_summary(source)
    )


def _audit_record(
    record: dict[str, Any],
    *,
    relative: float,
    absolute: float,
    require_defensive_design_certificate: bool = False,
    require_crude_design_certificate: bool = False,
    require_direct_planning_certificate: bool = False,
    require_hybrid_selection_certificate: bool = False,
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
        "crude_design_certificate": _audit_crude_design_certificate(
            record,
            relative=relative,
            absolute=absolute,
            required=require_crude_design_certificate,
        ),
        "direct_planning_certificate": _audit_replicated_direct_certificate(
            record,
            relative=relative,
            absolute=absolute,
            required=require_direct_planning_certificate,
        ),
        "hybrid_selection_certificate": _audit_replicated_hybrid_selection(
            record,
            relative=relative,
            absolute=absolute,
            required=require_hybrid_selection_certificate,
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
                source.get("protocol_id")
                in {
                    "g11-v6-baseline-qualification-v3",
                    "g11-v6-baseline-qualification-v4",
                    "g11-v6-baseline-qualification-v5",
                    "g11-v6-baseline-qualification-v6",
                }
            ),
            require_crude_design_certificate=(
                source.get("protocol_id")
                in {
                    "g11-v6-baseline-qualification-v4",
                    "g11-v6-baseline-qualification-v5",
                    "g11-v6-baseline-qualification-v6",
                }
            ),
            require_direct_planning_certificate=(
                source.get("config_schema")
                == "npi.g11.v6-secondary-baselines.config.v2"
                or (
                    source.get("config_schema")
                    == "npi.g11.v6-routed-policy.config.v4"
                    and record.get("route", {}).get("action")
                    in {"dcs_slis", "crude"}
                )
            ),
            require_hybrid_selection_certificate=(
                source.get("config_schema")
                == "npi.g11.v6-routed-policy.config.v4"
                and record.get("route", {}).get("action")
                == "profile_hybrid"
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
        "summary_decision_recomputed": _audit_v6_summary(source),
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
