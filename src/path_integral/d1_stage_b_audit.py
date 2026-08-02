"""Independent audit for the complete-matrix D1 Stage B development result."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class D1StageBAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    recomputed_aggregate: dict[str, Any]
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        return math.inf
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _aggregate_matches(recorded: Any, recomputed: Any) -> bool:
    """Compare an aggregate without depending on platform-specific libm rounding.

    Derived floating-point fields (notably geometric means) can differ by a few
    ulps between Windows and Linux even when their inputs are byte-identical.
    Schema, discrete values, and decisions remain exact; only finite floats get
    a tolerance far below any scientific gate or serialized measurement error.
    """

    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        return recorded.keys() == recomputed.keys() and all(
            _aggregate_matches(recorded[key], recomputed[key]) for key in recorded
        )
    if isinstance(recorded, list) and isinstance(recomputed, list):
        return len(recorded) == len(recomputed) and all(
            _aggregate_matches(left, right)
            for left, right in zip(recorded, recomputed, strict=True)
        )
    if isinstance(recorded, bool) or isinstance(recomputed, bool):
        return type(recorded) is type(recomputed) and recorded == recomputed
    if isinstance(recorded, float) and isinstance(recomputed, float):
        return math.isclose(recorded, recomputed, rel_tol=1e-12, abs_tol=1e-15)
    return type(recorded) is type(recomputed) and recorded == recomputed


def _recompute(
    config: dict[str, Any], paired: list[dict[str, Any]], external: list[dict[str, Any]]
) -> dict[str, Any]:
    maximum_error = float(config["maximum_exactness_error"])
    normalization_limit = float(config["maximum_likelihood_normalization_absolute_z"])
    exactness = all(
        max(float(value) for value in record["exactness"].values()) <= maximum_error
        for record in paired
    ) and all(record["lifecycle_audit_passed"] is True for record in external)
    finite = all(
        record["likelihood_diagnostics"]["nonfinite_weight_count"] == 0
        for record in external
    )
    representable = all(
        record["likelihood_diagnostics"]["normalization_moments_representable"] is True
        for record in external
    )
    normalization_passes = sum(
        record["likelihood_diagnostics"]["normalization_moments_representable"] is True
        and abs(float(record["likelihood_diagnostics"]["normalization_z"]))
        <= normalization_limit
        for record in external
    )
    ratios = [float(record["mechanism"]["variance_ratio_raw_over_dcs"]) for record in paired]
    geometric_ratio = _geometric_mean(ratios)
    mechanism = geometric_ratio >= float(config["gate"]["minimum_mechanism_variance_ratio"])
    primary = set(config["external_methods"]["primary"])
    primary_records = [record for record in external if record["method"] in primary]
    expected_primary = len(config["cell_ids"]) * int(config["clusters"]) * len(primary)
    primary_evaluable = len(primary_records) == expected_primary and all(
        math.isfinite(float(record["estimate"]["value"]))
        and math.isfinite(float(record["estimate"]["standard_error"]))
        for record in primary_records
    )
    accuracy_budget = str(config["gate"]["primary_accuracy_budget"])
    accuracy_records = [
        record for record in primary_records if record["budget_id"] == accuracy_budget
    ]
    primary_accuracy_maximum = max(
        float(record["estimate"]["combined_reference_z"]) for record in accuracy_records
    )
    primary_accuracy = (
        len(accuracy_records) == expected_primary
        and primary_accuracy_maximum
        <= float(config["gate"]["primary_accuracy_combined_z"])
    )
    censoring = sum(bool(record["allocation"]["resource_censored"]) for record in primary_records)
    flow = all(
        record["flow_roundtrip"] is None
        or (
            float(record["flow_roundtrip"]["maximum_reconstruction_error"])
            <= maximum_error
            and float(
                record["flow_roundtrip"]["maximum_log_jacobian_cancellation_error"]
            )
            <= maximum_error
        )
        for record in external
    )
    stage_b_authorized = exactness and finite and mechanism and primary_evaluable
    stage_b_authorized = stage_b_authorized and primary_accuracy and flow
    dcs_accuracy_maximum = max(
        float(record["dcs"]["combined_reference_z"]) for record in paired
    )
    dcs_accuracy = dcs_accuracy_maximum <= float(
        config["gate"]["dcs_accuracy_combined_z"]
    )
    cost_closed = all(record["proposal_training_cost"] is not None for record in paired)
    stage_b_complete = stage_b_authorized and dcs_accuracy and cost_closed
    return {
        "paired_record_count": len(paired),
        "external_record_count": len(external),
        "maximum_exactness_error": max(
            max(float(value) for value in record["exactness"].values()) for record in paired
        ),
        "exactness_pass": exactness,
        "all_external_weights_finite": finite,
        "all_likelihood_moments_representable": representable,
        "likelihood_normalization_pass_fraction": normalization_passes / len(external),
        "mechanism_geometric_variance_ratio": geometric_ratio,
        "mechanism_pass": mechanism,
        "primary_external_evaluable": primary_evaluable,
        "primary_accuracy_budget": accuracy_budget,
        "primary_accuracy_pass": primary_accuracy,
        "primary_accuracy_maximum_combined_z": primary_accuracy_maximum,
        "primary_resource_censoring_count": censoring,
        "flow_roundtrip_pass": flow,
        "dcs_proposal_training_cost_closed": cost_closed,
        "stage_b_authorized": stage_b_authorized,
        "dcs_accuracy_maximum_combined_z": dcs_accuracy_maximum,
        "dcs_accuracy_pass": dcs_accuracy,
        "stage_b_complete": stage_b_complete,
        "p8_blockers": [
            *([] if exactness else ["exactness_failure"]),
            *([] if mechanism else ["mechanism_gate_failure"]),
            *([] if primary_accuracy else ["primary_accuracy_failure"]),
            *([] if dcs_accuracy else ["dcs_accuracy_failure"]),
            *([] if cost_closed else ["dcs_proposal_training_cost_not_closed"]),
            *([] if censoring == 0 else ["primary_resource_censoring"]),
            "stage_c_not_complete",
            "t1_theorem_and_novelty_not_closed",
        ],
    }


def audit_d1_stage_b(
    *, config_path: Path, result: dict[str, Any], root: Path
) -> D1StageBAudit:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage B config must be a mapping")
    paired = result.get("paired_records")
    external = result.get("external_records")
    if not isinstance(paired, list) or not isinstance(external, list):
        raise ValueError("Stage B result records must be lists")
    aggregate = _recompute(config, paired, external)

    bindings = True
    for binding in config.get("bindings", {}).values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            bindings = False
            break
        path = root / str(binding["path"])
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            bindings = False
            break

    seeds: list[int] = []
    for record in paired:
        seeds.extend((int(record["path_seed"]), int(record["label_seed"])))
    for record in external:
        seeds.extend(int(value) for value in record["seeds"].values())
    expected_seeds = list(range(int(config["base_seed"]), int(config["base_seed"]) + len(seeds)))
    seed_hash = hashlib.sha256(
        json.dumps(sorted(seeds), separators=(",", ":")).encode()
    ).hexdigest()

    bank = json.loads(
        (root / config["bindings"]["dcs_proposal_bank"]["path"]).read_text()
    )
    bank_entries = {entry["cell_id"]: entry for entry in bank["entries"]}
    bank_and_amortization = True
    for record in paired:
        entry = bank_entries[record["cell_id"]]
        if (
            record["proposal_bank_sha256"] != bank["bank_sha256"]
            or record["proposal_training_cost"] != entry["training_cost"]
        ):
            bank_and_amortization = False
            break
        for count in bank["amortization_query_counts"]:
            for method in ("raw", "dcs"):
                expected = float(record[method]["cost"]["algorithmic_work_units"])
                expected += float(entry["training_cost"]["algorithmic_work_units"]) / int(count)
                actual = float(record["amortized_total_work"][str(count)][method])
                if not math.isclose(actual, expected, rel_tol=1e-12):
                    bank_and_amortization = False
                    break

    expected_paired = len(config["cell_ids"]) * int(config["clusters"])
    expected_external = expected_paired * (
        len(config["external_methods"]["primary"])
        + len(config["external_methods"]["secondary"])
    )
    expected_decision = {
        "stage_b_complete": aggregate["stage_b_complete"],
        "stage_c_authorized": aggregate["stage_b_complete"],
        "p8_qualification_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
    }
    checks = (
        ("schema", result.get("schema") == "npi.g11.v8-d1-stage-b-result.v1"),
        ("config_hash", result.get("config_sha256") == hashlib.sha256(raw).hexdigest()),
        ("bindings", bindings),
        ("paired_record_count", len(paired) == expected_paired),
        ("external_record_count", len(external) == expected_external),
        ("seed_uniqueness", len(seeds) == len(set(seeds))),
        ("seed_contiguity", sorted(seeds) == expected_seeds),
        ("seed_count", result.get("seed_count") == len(seeds)),
        ("seed_hash", result.get("seed_set_sha256") == seed_hash),
        ("bank_and_amortization", bank_and_amortization),
        ("aggregate_recomputation", _aggregate_matches(result.get("aggregate"), aggregate)),
        ("decision_locks", result.get("decision") == expected_decision),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return D1StageBAudit(
        checks=checks,
        failures=failures,
        recomputed_aggregate=aggregate,
        passed=not failures,
    )
