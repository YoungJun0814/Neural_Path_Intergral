"""Independent JSON-only audit of V7 paired mechanism-probe artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import scipy.stats
import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance

_SOURCE_SCHEMA = "npi.g11.v7-mechanism-probe.v1"
_CONFIG_SCHEMA = "npi.g11.v7-mechanism-probe.config.v2"


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-13)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_inputs(
    config_path: Path,
    source_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    config_raw = config_path.read_bytes()
    config = yaml.safe_load(config_raw)
    if not isinstance(config, dict) or config.get("schema") != _CONFIG_SCHEMA:
        raise ValueError("independent audit requires a V7 mechanism config V2")
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw)
    if not isinstance(source, dict) or source.get("schema") != _SOURCE_SCHEMA:
        raise ValueError("independent audit requires a V7 mechanism-probe artifact")
    return (
        config,
        hashlib.sha256(config_raw).hexdigest(),
        source,
        hashlib.sha256(source_raw).hexdigest(),
    )


def _effect(
    cluster_effects: list[float],
    *,
    confidence_level: float,
) -> dict[str, float | int]:
    count = len(cluster_effects)
    if count < 2:
        raise ValueError("independent mechanism audit requires at least two clusters")
    mean = math.fsum(cluster_effects) / count
    variance = math.fsum((value - mean) ** 2 for value in cluster_effects) / (
        count - 1
    )
    standard_error = math.sqrt(variance / count)
    if standard_error == 0.0:
        lower = mean
        p_value = 0.0 if mean > 0.0 else 1.0
    else:
        critical = float(scipy.stats.t.ppf(confidence_level, df=count - 1))
        lower = mean - critical * standard_error
        p_value = float(scipy.stats.t.sf(mean / standard_error, df=count - 1))
    return {
        "cluster_count": count,
        "mean_log_raw_over_dcs_variance": mean,
        "standard_error": standard_error,
        "geometric_raw_over_dcs_variance_ratio": math.exp(mean),
        "one_sided_lower_raw_over_dcs_variance_ratio": math.exp(lower),
        "p_value_against_no_variance_reduction": p_value,
    }


def _record_failures(
    record: dict[str, Any],
    *,
    expected_samples: int,
) -> list[str]:
    failures: list[str] = []
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return ["missing diagnostics"]
    required = {
        "count",
        "raw_mean",
        "dcs_mean",
        "residual_mean",
        "residual_standard_error",
        "raw_variance",
        "dcs_variance",
        "residual_variance",
        "dcs_residual_covariance",
        "dcs_residual_covariance_product_variance",
        "dcs_residual_covariance_standard_error",
        "dcs_residual_covariance_z_score",
        "dcs_residual_correlation",
        "raw_over_dcs_variance_ratio",
        "variance_decomposition_error",
    }
    if set(diagnostics) != required:
        return ["malformed diagnostics fields"]
    numeric_fields = required - {"dcs_residual_correlation"}
    if any(
        isinstance(diagnostics[field], bool)
        or not isinstance(diagnostics[field], (int, float))
        or not math.isfinite(float(diagnostics[field]))
        for field in numeric_fields
    ):
        return ["nonfinite diagnostic"]
    count = diagnostics["count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != expected_samples
    ):
        failures.append("sample count mismatch")
    raw_mean = float(diagnostics["raw_mean"])
    dcs_mean = float(diagnostics["dcs_mean"])
    residual_mean = float(diagnostics["residual_mean"])
    raw_variance = float(diagnostics["raw_variance"])
    dcs_variance = float(diagnostics["dcs_variance"])
    residual_variance = float(diagnostics["residual_variance"])
    covariance = float(diagnostics["dcs_residual_covariance"])
    product_variance = float(
        diagnostics["dcs_residual_covariance_product_variance"]
    )
    if min(raw_variance, dcs_variance, residual_variance, product_variance) < 0.0:
        failures.append("negative variance")
    expected_residual_mean = raw_mean - dcs_mean
    expected_residual_se = math.sqrt(residual_variance / count)
    expected_raw_variance = (
        dcs_variance + residual_variance + 2.0 * covariance
    )
    expected_decomposition = (
        raw_variance - dcs_variance - residual_variance - 2.0 * covariance
    )
    expected_covariance_se = (
        math.sqrt(product_variance / count) * count / (count - 1)
    )
    expected_covariance_z = (
        0.0
        if expected_covariance_se == 0.0 and covariance == 0.0
        else math.inf
        if expected_covariance_se == 0.0
        else covariance / expected_covariance_se
    )
    expected_ratio = (
        math.inf
        if dcs_variance == 0.0 and raw_variance > 0.0
        else 1.0
        if dcs_variance == 0.0
        else raw_variance / dcs_variance
    )
    expected_correlation = (
        None
        if dcs_variance == 0.0 or residual_variance == 0.0
        else covariance / math.sqrt(dcs_variance * residual_variance)
    )
    comparisons = (
        ("residual mean", residual_mean, expected_residual_mean),
        (
            "residual standard error",
            float(diagnostics["residual_standard_error"]),
            expected_residual_se,
        ),
        ("raw variance identity", raw_variance, expected_raw_variance),
        (
            "variance decomposition",
            float(diagnostics["variance_decomposition_error"]),
            expected_decomposition,
        ),
        (
            "covariance standard error",
            float(diagnostics["dcs_residual_covariance_standard_error"]),
            expected_covariance_se,
        ),
        (
            "covariance z score",
            float(diagnostics["dcs_residual_covariance_z_score"]),
            expected_covariance_z,
        ),
        (
            "variance ratio",
            float(diagnostics["raw_over_dcs_variance_ratio"]),
            expected_ratio,
        ),
    )
    for label, observed, expected in comparisons:
        if not _close(observed, expected):
            failures.append(f"{label} mismatch")
    expected_residual_z = (
        0.0
        if expected_residual_se == 0.0 and residual_mean == 0.0
        else math.inf
        if expected_residual_se == 0.0
        else residual_mean / expected_residual_se
    )
    reference = record.get("reference_probability")
    reference_se = record.get("reference_standard_error")
    if (
        isinstance(reference, bool)
        or not isinstance(reference, (int, float))
        or not math.isfinite(float(reference))
        or isinstance(reference_se, bool)
        or not isinstance(reference_se, (int, float))
        or not math.isfinite(float(reference_se))
        or float(reference_se) < 0.0
    ):
        failures.append("reference fields malformed")
    else:
        expected_raw_reference_z = (
            raw_mean - float(reference)
        ) / math.hypot(math.sqrt(raw_variance / count), float(reference_se))
        expected_dcs_reference_z = (
            dcs_mean - float(reference)
        ) / math.hypot(math.sqrt(dcs_variance / count), float(reference_se))
        for label, field, expected in (
            ("residual z", "residual_z_score", expected_residual_z),
            (
                "raw reference z",
                "raw_reference_z_score",
                expected_raw_reference_z,
            ),
            (
                "DCS reference z",
                "dcs_reference_z_score",
                expected_dcs_reference_z,
            ),
        ):
            observed_field = record.get(field)
            if (
                isinstance(observed_field, bool)
                or not isinstance(observed_field, (int, float))
                or not math.isfinite(float(observed_field))
                or not _close(float(observed_field), expected)
            ):
                failures.append(f"{label} mismatch")
    observed_correlation = diagnostics["dcs_residual_correlation"]
    if expected_correlation is None:
        if observed_correlation is not None:
            failures.append("correlation nullability mismatch")
    elif (
        isinstance(observed_correlation, bool)
        or not isinstance(observed_correlation, (int, float))
        or not _close(float(observed_correlation), expected_correlation)
    ):
        failures.append("correlation mismatch")
    work = record.get("work_record")
    if (
        not isinstance(work, dict)
        or work.get("category") != "mechanism_probe"
        or work.get("method") != "paired_raw_dcs"
        or work.get("cell_id") != record.get("cell_id")
        or work.get("samples") != expected_samples
        or work.get("successful") is not True
    ):
        failures.append("work-record identity mismatch")
    return failures


def audit(
    config_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    config, config_hash, source, source_hash = _load_inputs(
        config_path,
        source_path,
    )
    if source.get("config_sha256") != config_hash:
        raise ValueError("source config hash does not match audit config")
    if source.get("protocol_id") != config.get("protocol_id"):
        raise ValueError("source protocol does not match audit config")
    sampling = config["sampling"]
    expected_clusters = int(sampling["clusters"])
    expected_samples = int(sampling["samples_per_cell_cluster"])
    expected_cells = int(config["requirements"]["expected_cells"])
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("source records must be a list")
    identities = [
        (str(record.get("cell_id")), int(record.get("cluster", -1)))
        for record in records
        if isinstance(record, dict)
    ]
    cells = sorted({identity[0] for identity in identities})
    expected_identity = {
        (cell, cluster)
        for cell in cells
        for cluster in range(expected_clusters)
    }
    failures: list[str] = []
    if (
        len(cells) != expected_cells
        or len(identities) != len(set(identities))
        or set(identities) != expected_identity
    ):
        failures.append("incomplete or duplicate record matrix")
    record_failure_count = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"record {index}: not an object")
            record_failure_count += 1
            continue
        item_failures = _record_failures(
            record,
            expected_samples=expected_samples,
        )
        if item_failures:
            record_failure_count += 1
            failures.extend(
                f"record {index}: {failure}" for failure in item_failures
            )
    ledger = source.get("seed_ledger")
    seed_records = ledger.get("records") if isinstance(ledger, dict) else None
    if (
        not isinstance(seed_records, list)
        or len(seed_records) != 2 * len(records)
        or len({item.get("seed") for item in seed_records}) != len(seed_records)
        or source.get("seed_ledger_sha256") != _canonical_hash(ledger)
    ):
        failures.append("seed-ledger integrity mismatch")
    else:
        expected_protocol = config["protocol_id"]
        for item in seed_records:
            key = item.get("key")
            if (
                not isinstance(key, dict)
                or key.get("protocol") != expected_protocol
                or key.get("role") != "mechanism-probe"
                or key.get("task") != "paired_raw_dcs"
                or key.get("stream") not in {"proposal", "labels"}
            ):
                failures.append("seed-key contract mismatch")
                break
    work_ledger = source.get("work_ledger")
    work_records = (
        work_ledger.get("records") if isinstance(work_ledger, dict) else None
    )
    source_work = [record.get("work_record") for record in records]
    if (
        not isinstance(work_records, list)
        or work_records != source_work
        or source.get("work_ledger_sha256") != _canonical_hash(work_ledger)
    ):
        failures.append("work-ledger integrity mismatch")
    cluster_effects = []
    for cluster in range(expected_clusters):
        values = [
            math.log(
                float(record["diagnostics"]["raw_over_dcs_variance_ratio"])
            )
            for record in records
            if int(record["cluster"]) == cluster
        ]
        if len(values) != expected_cells:
            failures.append("cluster effect lacks every cell")
            break
        cluster_effects.append(math.fsum(values) / len(values))
    recomputed_effect = (
        _effect(
            cluster_effects,
            confidence_level=float(sampling["confidence_level"]),
        )
        if len(cluster_effects) == expected_clusters
        else None
    )
    if recomputed_effect is not None:
        recorded_effect = source.get("paired_cluster_effect")
        if not isinstance(recorded_effect, dict) or any(
            not _close(
                float(recorded_effect[field]),
                float(recomputed_effect[field]),
            )
            for field in recomputed_effect
            if field != "cluster_count"
        ):
            failures.append("paired cluster effect mismatch")
    maximum_residual_z = max(
        abs(float(record["residual_z_score"])) for record in records
    )
    maximum_orthogonality_z = max(
        abs(
            float(
                record["diagnostics"][
                    "dcs_residual_covariance_z_score"
                ]
            )
        )
        for record in records
    )
    thresholds = config["development_thresholds"]
    recomputed_gates = {
        "complete_matrix": set(identities) == expected_identity
        and len(identities) == len(expected_identity),
        "all_finite_positive_variances": all(
            float(record["diagnostics"]["raw_variance"]) > 0.0
            and float(record["diagnostics"]["dcs_variance"]) > 0.0
            for record in records
        ),
        "all_variance_decompositions_numerical": all(
            abs(float(record["diagnostics"]["variance_decomposition_error"]))
            <= 1e-10
            * max(1.0, float(record["diagnostics"]["raw_variance"]))
            for record in records
        ),
        "aggregate_variance_ratio_lower": recomputed_effect is not None
        and float(
            recomputed_effect["one_sided_lower_raw_over_dcs_variance_ratio"]
        )
        >= float(thresholds["minimum_variance_ratio_lower"]),
        "residual_mean_diagnostic": maximum_residual_z
        <= float(thresholds["maximum_absolute_residual_z"]),
        "orthogonality_diagnostic": maximum_orthogonality_z
        <= float(thresholds["maximum_absolute_orthogonality_z"]),
    }
    if source.get("gates") != recomputed_gates:
        failures.append("source gate summary mismatch")
    if bool(source.get("development_mechanism_passed")) != all(
        recomputed_gates.values()
    ):
        failures.append("source development decision mismatch")
    training_audit = source.get("proposal_training_audit")
    expected_formal = {
        "frozen_config": bool(config["frozen"]),
        "clean_source": not bool(source.get("dirty_worktree")),
        "non_smoke": not bool(source.get("smoke")),
        "verified_training_source": bool(
            isinstance(training_audit, dict)
            and training_audit.get("verified")
        ),
        "formal_training_source": bool(
            isinstance(training_audit, dict)
            and training_audit.get("formal_training_source_readiness")
        ),
    }
    if source.get("formal_readiness") != expected_formal:
        failures.append("source formal-readiness mismatch")
    expected_formal_decision = all(recomputed_gates.values()) and all(
        expected_formal.values()
    )
    if bool(source.get("formal_mechanism_passed")) != expected_formal_decision:
        failures.append("source formal decision mismatch")
    auditor_provenance = source_provenance()
    return {
        "schema": "npi.g11.v7-mechanism-probe-audit.v1",
        "source_artifact_sha256": source_hash,
        "config_sha256": config_hash,
        "record_count": len(records),
        "record_failure_count": record_failure_count,
        "recomputed_effect": recomputed_effect,
        "maximum_absolute_residual_z": maximum_residual_z,
        "maximum_absolute_orthogonality_z": maximum_orthogonality_z,
        "recomputed_gates": recomputed_gates,
        "failures": failures,
        "passed": not failures,
        "environment": runtime_provenance(dtype="serialized-float64"),
        **auditor_provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.config, arguments.source)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "record_count": result["record_count"],
                "record_failure_count": result["record_failure_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
