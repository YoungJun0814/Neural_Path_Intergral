"""Fail-closed analysis of untouched V6 baseline and routed-policy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import scipy.stats
import yaml

from src.path_integral import exact_binomial_probability_interval
from src.path_integral.provenance import runtime_provenance, source_provenance

_SCHEMA = "npi.g11.v6-confirmatory.config.v1"
_HASH_FIELDS = (
    "baseline_config",
    "policy_config",
    "manifest",
    "reference",
    "power",
    "audit_config",
)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise ValueError("unsupported V6 confirmatory config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "expected_sha256",
        "statistics",
        "accuracy",
        "requirements",
    }:
        raise ValueError("malformed V6 confirmatory config fields")
    if payload["phase"] not in ("development", "confirmation"):
        raise ValueError("unsupported V6 confirmatory phase")
    hashes = payload["expected_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(_HASH_FIELDS):
        raise ValueError("confirmatory expected hashes are malformed")
    if payload["phase"] == "confirmation":
        if payload["frozen"] is not True:
            raise ValueError("confirmation config must be frozen")
        if any(
            not isinstance(hashes[field], str)
            or len(hashes[field]) != 64
            or any(character not in "0123456789abcdef" for character in hashes[field])
            for field in _HASH_FIELDS
        ):
            raise ValueError("confirmation requires every frozen artifact SHA-256")
    return payload, hashlib.sha256(raw).hexdigest()


def _load(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema} artifact")
    return payload, hashlib.sha256(raw).hexdigest()


def _artifact_canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _total_work(record: dict[str, Any]) -> float:
    value = math.fsum(
        float(item["work_units"]) for item in record["result"]["total_work"]["records"]
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("total work must be finite and positive")
    return value


def _quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid quantile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_rmse_upper(
    errors: list[float], *, confidence_level: float, repetitions: int, seed: int
) -> float:
    if len(errors) < 2:
        return math.inf
    generator = random.Random(seed)
    count = len(errors)
    values = []
    for _ in range(repetitions):
        resample = [errors[generator.randrange(count)] for _ in range(count)]
        values.append(math.sqrt(math.fsum(value * value for value in resample) / count))
    return _quantile(values, confidence_level)


def _one_sided_efficiency(
    cluster_effects: list[float], *, confidence_level: float
) -> dict[str, float | int | None]:
    count = len(cluster_effects)
    mean = math.fsum(cluster_effects) / count
    if count < 2:
        return {
            "cluster_count": count,
            "mean_log_ratio": mean,
            "standard_error": None,
            "one_sided_lower_log_ratio": None,
            "geometric_mean_ratio": math.exp(mean),
            "one_sided_lower_geometric_mean_ratio": None,
            "p_value_against_no_saving": None,
        }
    variance = math.fsum((value - mean) ** 2 for value in cluster_effects) / (count - 1)
    standard_error = math.sqrt(variance / count)
    if standard_error == 0.0:
        lower = mean
        p_value = 0.0 if mean > 0.0 else 1.0
    else:
        critical = float(scipy.stats.t.ppf(confidence_level, df=count - 1))
        lower = mean - critical * standard_error
        statistic = mean / standard_error
        p_value = float(scipy.stats.t.sf(statistic, df=count - 1))
    return {
        "cluster_count": count,
        "mean_log_ratio": mean,
        "standard_error": standard_error,
        "one_sided_lower_log_ratio": lower,
        "geometric_mean_ratio": math.exp(mean),
        "one_sided_lower_geometric_mean_ratio": math.exp(lower),
        "p_value_against_no_saving": p_value,
    }


def _accuracy_groups(
    records: list[dict[str, Any]],
    *,
    method_label: str,
    multiplier: float,
    minimum_attainment: float,
    confidence_level: float,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    cells = sorted({str(record["cell_id"]) for record in records})
    output = []
    one_sided_cp_level = 2.0 * confidence_level - 1.0
    for cell_index, cell_id in enumerate(cells):
        cell_records = [record for record in records if record["cell_id"] == cell_id]
        first = cell_records[0]
        reference = float(first["reference_probability"])
        reference_se = float(first["reference_standard_error"])
        nominal = float(first["nominal_probability"])
        requested = float(first["result"]["core"]["requested_relative_sampling_rmse"])
        if any(
            float(record["reference_probability"]) != reference
            or float(record["reference_standard_error"]) != reference_se
            or float(record["nominal_probability"]) != nominal
            for record in cell_records
        ):
            raise ValueError("reference or target drift within one cell")
        errors = [float(record["result"]["core"]["estimate"]) - reference for record in cell_records]
        successes = sum(
            bool(record["result"]["core"]["empirical_target_attained"])
            for record in cell_records
        )
        interval = exact_binomial_probability_interval(
            successes,
            len(cell_records),
            confidence_level=one_sided_cp_level,
        )
        empirical_rmse = math.sqrt(
            math.fsum(error * error for error in errors) / len(errors)
        )
        upper = _bootstrap_rmse_upper(
            errors,
            confidence_level=confidence_level,
            repetitions=repetitions,
            seed=seed + cell_index,
        )
        epsilon = nominal * requested
        tolerance = math.hypot(multiplier * epsilon, reference_se)
        output.append(
            {
                "method": method_label,
                "cell_id": cell_id,
                "cluster_count": len(cell_records),
                "target_attainment_count": successes,
                "target_attainment_rate": successes / len(cell_records),
                "one_sided_exact_attainment_lower": interval.lower,
                "minimum_target_attainment_rate": minimum_attainment,
                "empirical_rmse": empirical_rmse,
                "bootstrap_rmse_upper": upper,
                "rmse_tolerance_including_reference": tolerance,
                "attainment_gate": interval.lower >= minimum_attainment,
                "rmse_gate": upper <= tolerance,
            }
        )
    return output


def run(
    config_path: Path,
    baseline_path: Path,
    policy_path: Path,
    baseline_audit_path: Path,
    policy_audit_path: Path,
    power_path: Path,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    baseline, baseline_hash = _load(
        baseline_path, "npi.g11.v6-baseline-qualification.v1"
    )
    policy, policy_hash = _load(policy_path, "npi.g11.v6-routed-policy.v1")
    baseline_audit, baseline_audit_hash = _load(
        baseline_audit_path, "npi.g11.v6-independent-audit.v1"
    )
    policy_audit, policy_audit_hash = _load(
        policy_audit_path, "npi.g11.v6-independent-audit.v1"
    )
    power, power_hash = _load(power_path, "npi.g11.v6-power-analysis.v1")
    shared_manifest = baseline.get("manifest_sha256")
    shared_reference = baseline.get("reference_artifact_sha256")
    shared_protocol_identities = (
        isinstance(shared_manifest, str)
        and len(shared_manifest) == 64
        and policy.get("manifest_sha256") == shared_manifest
        and isinstance(shared_reference, str)
        and len(shared_reference) == 64
        and policy.get("reference_artifact_sha256") == shared_reference
        and baseline_audit.get("config_sha256") == policy_audit.get("config_sha256")
    )
    protocol_hashes = {
        "baseline_config": baseline.get("config_sha256"),
        "policy_config": policy.get("config_sha256"),
        "manifest": shared_manifest,
        "reference": shared_reference,
        "power": power_hash,
        "audit_config": baseline_audit.get("config_sha256"),
    }
    expected_hashes = config["expected_sha256"]
    hashes_match = all(
        expected_hashes[field] is None or expected_hashes[field] == protocol_hashes[field]
        for field in _HASH_FIELDS
    )

    comparator = str(config["statistics"]["primary_comparator"])
    baseline_records = {
        (str(record["cell_id"]), int(record["cluster"])): record
        for record in baseline["records"]
        if record["method"] == comparator
    }
    policy_records = {
        (str(record["cell_id"]), int(record["cluster"])): record
        for record in policy["records"]
    }
    if not baseline_records or set(baseline_records) != set(policy_records):
        raise ValueError("confirmation requires an identical complete paired matrix")
    cells = sorted({key[0] for key in policy_records})
    clusters = sorted({key[1] for key in policy_records})
    for cluster in clusters:
        if {cell for cell, item_cluster in policy_records if item_cluster == cluster} != set(
            cells
        ):
            raise ValueError("every paired cluster must contain every cell")

    paired = []
    cluster_effects = []
    for cluster in clusters:
        effects = []
        for cell_id in cells:
            baseline_record = baseline_records[(cell_id, cluster)]
            policy_record = policy_records[(cell_id, cluster)]
            ratio = _total_work(baseline_record) / _total_work(policy_record)
            effect = math.log(ratio)
            effects.append(effect)
            paired.append(
                {
                    "cell_id": cell_id,
                    "cluster": cluster,
                    "cem_over_policy_work_ratio": ratio,
                    "log_ratio": effect,
                }
            )
        cluster_effects.append(math.fsum(effects) / len(effects))

    statistics = config["statistics"]
    confidence = float(statistics["confidence_level"])
    efficiency = _one_sided_efficiency(cluster_effects, confidence_level=confidence)
    accuracy_config = config["accuracy"]
    repetitions = int(statistics["bootstrap_repetitions"])
    base_seed = int(statistics["bootstrap_seed"])
    accuracy = _accuracy_groups(
        list(baseline_records.values()),
        method_label=comparator,
        multiplier=float(accuracy_config["rmse_engineering_multiplier"]),
        minimum_attainment=float(accuracy_config["minimum_target_attainment_rate"]),
        confidence_level=confidence,
        repetitions=repetitions,
        seed=base_seed,
    ) + _accuracy_groups(
        list(policy_records.values()),
        method_label="v6_policy",
        multiplier=float(accuracy_config["rmse_engineering_multiplier"]),
        minimum_attainment=float(accuracy_config["minimum_target_attainment_rate"]),
        confidence_level=confidence,
        repetitions=repetitions,
        seed=base_seed + 100_000,
    )

    all_records = list(baseline_records.values()) + list(policy_records.values())
    requirements = config["requirements"]
    audits_valid = (
        baseline_audit["source_artifact_sha256"] == baseline_hash
        and policy_audit["source_artifact_sha256"] == policy_hash
        and baseline_audit["gates"]["all_records_pass"]
        and policy_audit["gates"]["all_records_pass"]
    )
    lower_ratio = efficiency["one_sided_lower_geometric_mean_ratio"]
    gates = {
        "frozen_artifact_hashes_match": hashes_match,
        "shared_protocol_identities": shared_protocol_identities,
        "identical_complete_pair_set": len(paired) == len(cells) * len(clusters),
        "minimum_cluster_count": len(clusters) >= int(requirements["minimum_clusters"]),
        "all_runs_complete": all(record["result"]["core"]["complete"] for record in all_records),
        "no_resource_censoring": all(
            not record["result"]["core"]["resource_censored"] for record in all_records
        ),
        "independent_audits": audits_valid,
        "all_accuracy_co_gates": all(
            record["attainment_gate"] and record["rmse_gate"] for record in accuracy
        ),
        "one_sided_efficiency_lower_exceeds_one": lower_ratio is not None
        and float(lower_ratio) > 1.0,
        "powered_cluster_count": power["forecast"] is not None
        and len(clusters)
        >= int(power["forecast"]["required_clusters_normal_approximation"]),
    }
    if not bool(requirements["require_all_complete"]):
        gates["all_runs_complete"] = True
    if not bool(requirements["require_no_censoring"]):
        gates["no_resource_censoring"] = True
    if not bool(requirements["require_independent_audits"]):
        gates["independent_audits"] = True
    provenance = source_provenance()
    formal = {
        "confirmation_phase": config["phase"] == "confirmation",
        "frozen_config": bool(config["frozen"]),
        "frozen_sources": bool(baseline.get("formal_readiness", {}).get("frozen_config"))
        and bool(policy.get("formal_readiness", {}).get("frozen_config"))
        and baseline.get("phase") == "confirmation"
        and policy.get("phase") == "confirmation",
        "qualified_audits": bool(baseline_audit.get("qualification_audit_passed"))
        and bool(policy_audit.get("qualification_audit_passed")),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not bool(baseline.get("smoke")) and not bool(policy.get("smoke")),
    }
    return {
        "schema": "npi.g11.v6-confirmatory.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "frozen_protocol_sha256": protocol_hashes,
        "result_artifact_sha256": {
            "baseline": baseline_hash,
            "policy": policy_hash,
            "baseline_audit": baseline_audit_hash,
            "policy_audit": policy_audit_hash,
            "power": power_hash,
        },
        "cell_count": len(cells),
        "cluster_count": len(clusters),
        "paired_records": paired,
        "equal_cell_cluster_log_ratios": cluster_effects,
        "primary_efficiency": efficiency,
        "practical_geometric_mean_ratio": float(
            statistics["practical_geometric_mean_ratio"]
        ),
        "accuracy": accuracy,
        "gates": gates,
        "formal_readiness": formal,
        "scientific_gates_passed": all(gates.values()),
        "confirmation_passed": all(gates.values()) and all(formal.values()),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v6/confirmatory_development.yaml"),
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--policy-audit", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.baseline,
        arguments.policy,
        arguments.baseline_audit,
        arguments.policy_audit,
        arguments.power,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"passed": result["confirmation_passed"], **result["gates"]}))


if __name__ == "__main__":
    main()
