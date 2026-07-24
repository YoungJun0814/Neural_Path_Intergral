"""Paired raw/DCS mechanism probe under one identical defensive proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import scipy.stats
import torch
import yaml

from experiments.g11_v6_baseline_qualification import (
    _load_references,
    _smoke_cells,
    _task,
)
from experiments.g11_v6_reference import _load_manifest
from experiments.g11_v6_routed_policy import (
    _task_conditioned_training_source_audit,
)
from src.path_integral import (
    RBergomiHybridTermSampler,
    SeedKey,
    SeedLedger,
    TimePiecewiseTwoDriverControl,
    V6WorkLedger,
    V6WorkRecord,
    rao_blackwell_pair_diagnostics,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

_SCHEMA_V1 = "npi.g11.v7-mechanism-probe.config.v1"
_SCHEMA_V2 = "npi.g11.v7-mechanism-probe.config.v2"
_PROPOSAL_FIELDS = {
    "weights",
    "task_controls",
    "training_source_artifact_sha256",
    "training_derivation",
    "training_source_record_count",
    "training_total_samples",
    "training_total_work_units",
    "training_total_wall_seconds",
    "training_total_cpu_seconds",
}


def _strict_positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _strict_probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{field} must be a probability")
    return result


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") not in {
        _SCHEMA_V1,
        _SCHEMA_V2,
    }:
        raise ValueError("unsupported V7 mechanism-probe config")
    if set(payload) != {
        "schema",
        "protocol_id",
        "phase",
        "frozen",
        "estimand",
        "hierarchy",
        "proposal",
        "sampling",
        "development_thresholds",
        "requirements",
    }:
        raise ValueError("malformed V7 mechanism-probe config fields")
    if payload["phase"] not in {"development", "qualification", "confirmation"}:
        raise ValueError("unsupported V7 mechanism-probe phase")
    if payload["phase"] != "development" and payload["frozen"] is not True:
        raise ValueError("formal V7 mechanism probes must be frozen")
    if payload["estimand"] != "fixed_finest_grid":
        raise ValueError("V7 mechanism probe requires a fixed-grid estimand")
    hierarchy = payload["hierarchy"]
    if not isinstance(hierarchy, dict) or set(hierarchy) != {
        "coarsest_steps",
        "finest_level",
    }:
        raise ValueError("malformed V7 mechanism hierarchy")
    coarsest_steps = _strict_positive_integer(
        hierarchy["coarsest_steps"], "coarsest_steps"
    )
    _strict_positive_integer(hierarchy["finest_level"], "finest_level")
    if coarsest_steps < 2 or coarsest_steps % 2:
        raise ValueError("coarsest_steps must be an even integer at least two")
    proposal = payload["proposal"]
    if not isinstance(proposal, dict) or set(proposal) != _PROPOSAL_FIELDS:
        raise ValueError("malformed V7 mechanism proposal")
    if (
        proposal["training_derivation"]
        != "componentwise_median_pure_cem_then_zero_half_full_bank"
    ):
        raise ValueError("unsupported V7 proposal derivation")
    digest = proposal["training_source_artifact_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("proposal source hash must be lowercase SHA-256")
    weights = proposal["weights"]
    controls = proposal["task_controls"]
    if (
        not isinstance(weights, list)
        or len(weights) < 2
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
            for weight in weights
        )
        or not math.isclose(
            math.fsum(float(weight) for weight in weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(controls, dict)
        or set(controls) != {"terminal_left_tail", "discrete_lower_barrier"}
    ):
        raise ValueError("invalid V7 task-conditioned proposal")
    for field in ("training_source_record_count", "training_total_samples"):
        _strict_positive_integer(proposal[field], field)
    for field in (
        "training_total_work_units",
        "training_total_wall_seconds",
        "training_total_cpu_seconds",
    ):
        value = proposal[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{field} must be finite and positive")
    sampling = payload["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != {
        "clusters",
        "samples_per_cell_cluster",
        "confidence_level",
        "engine",
    }:
        raise ValueError("malformed V7 mechanism sampling contract")
    _strict_positive_integer(sampling["clusters"], "clusters")
    if (
        _strict_positive_integer(
            sampling["samples_per_cell_cluster"],
            "samples_per_cell_cluster",
        )
        < 2
    ):
        raise ValueError("samples_per_cell_cluster must be at least two")
    confidence = _strict_probability(sampling["confidence_level"], "confidence_level")
    if confidence <= 0.5 or sampling["engine"] not in {"fft", "reference"}:
        raise ValueError("invalid V7 mechanism sampling contract")
    thresholds = payload["development_thresholds"]
    expected_thresholds = {
        "minimum_variance_ratio_lower",
        "maximum_absolute_residual_z",
        (
            "maximum_absolute_orthogonality_correlation"
            if payload["schema"] == _SCHEMA_V1
            else "maximum_absolute_orthogonality_z"
        ),
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_thresholds:
        raise ValueError("malformed V7 development thresholds")
    ratio = thresholds["minimum_variance_ratio_lower"]
    residual_z = thresholds["maximum_absolute_residual_z"]
    orthogonality = thresholds[
        "maximum_absolute_orthogonality_correlation"
        if payload["schema"] == _SCHEMA_V1
        else "maximum_absolute_orthogonality_z"
    ]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or float(ratio) <= 1.0
        or isinstance(residual_z, bool)
        or not isinstance(residual_z, (int, float))
        or float(residual_z) <= 0.0
        or isinstance(orthogonality, bool)
        or not isinstance(orthogonality, (int, float))
        or float(orthogonality) <= 0.0
        or (
            payload["schema"] == _SCHEMA_V1
            and float(orthogonality) >= 1.0
        )
    ):
        raise ValueError("invalid V7 development thresholds")
    requirements = payload["requirements"]
    if not isinstance(requirements, dict) or set(requirements) != {
        "expected_cells",
        "require_verified_training_source",
    }:
        raise ValueError("malformed V7 mechanism requirements")
    _strict_positive_integer(requirements["expected_cells"], "expected_cells")
    if requirements["require_verified_training_source"] is not True:
        raise ValueError("V7 mechanism probes require the verified proposal source")
    return payload, hashlib.sha256(raw).hexdigest()


def _one_sided_variance_effect(
    cluster_effects: list[float],
    *,
    confidence_level: float,
) -> dict[str, float | int]:
    count = len(cluster_effects)
    if count < 2:
        raise ValueError("paired mechanism inference requires at least two clusters")
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


def run(
    config_path: Path,
    manifest_path: Path,
    reference_path: Path,
    proposal_training_source_path: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    manifest = _load_manifest(manifest_path)
    references, reference_hash = _load_references(reference_path)
    proposal = config["proposal"]
    training_audit = _task_conditioned_training_source_audit(
        proposal,
        proposal_training_source_path,
    )
    if (
        not smoke
        and config["phase"] != "development"
        and (
            manifest.phase != config["phase"]
            or not manifest.frozen
            or manifest.smoke
        )
    ):
        raise ValueError(
            "formal V7 mechanism probes require a same-phase frozen manifest"
        )
    cells = _smoke_cells(manifest.cells) if smoke else manifest.cells
    requirements = config["requirements"]
    if not smoke and len(cells) != int(requirements["expected_cells"]):
        raise ValueError("manifest cell count differs from the V7 mechanism contract")
    hierarchy = config["hierarchy"]
    finest_level = int(hierarchy["finest_level"])
    if (
        int(hierarchy["coarsest_steps"]) * 2**finest_level
        != cells[0].finest_steps
    ):
        raise ValueError("V7 mechanism hierarchy does not match the manifest")
    sampling = config["sampling"]
    clusters = 1 if smoke else int(sampling["clusters"])
    sample_count = 128 if smoke else int(sampling["samples_per_cell_cluster"])
    weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    seed_ledger = SeedLedger()
    work_ledger = V6WorkLedger()
    records: list[dict[str, Any]] = []
    for cell in cells:
        reference_probability, reference_se, reference_cell = references[cell.cell_id]
        if reference_cell != cell.to_dict():
            raise ValueError(f"reference estimand drift for cell {cell.cell_id}")
        simulator = RBergomiSimulator(
            H=cell.hurst,
            eta=cell.eta,
            xi=cell.xi,
            rho=cell.rho,
            device="cpu",
        )
        controls = tuple(
            TimePiecewiseTwoDriverControl(
                tuple(
                    (float(segment[0]), float(segment[1]))
                    for segment in schedule
                ),
                maturity=cell.maturity,
            )
            for schedule in proposal["task_controls"][cell.task]
        )
        sampler = RBergomiHybridTermSampler(
            simulator,
            controls,
            weights,
            _task(cell),
            spot=cell.spot,
            maturity=cell.maturity,
            coarsest_steps=int(hierarchy["coarsest_steps"]),
            finest_level=finest_level,
            engine=sampling["engine"],
            correction_method="dcs_mgi",
        )
        for cluster in range(clusters):
            seeds = {
                stream: seed_ledger.allocate(
                    SeedKey(
                        str(config["protocol_id"]),
                        "mechanism-probe",
                        f"{cell.cell_id}:cluster-{cluster}",
                        "paired_raw_dcs",
                        finest_level,
                        0,
                        stream,
                    )
                )
                for stream in ("proposal", "labels")
            }
            cpu_started = time.process_time()
            pair = sampler.sample_raw_dcs_pair(
                f"single_{finest_level}",
                "pilot",
                sample_count,
                seeds,
            )
            cpu_seconds = time.process_time() - cpu_started
            diagnostics = rao_blackwell_pair_diagnostics(
                pair.raw_values,
                pair.dcs_values,
            )
            work_record = V6WorkRecord(
                category="mechanism_probe",
                method="paired_raw_dcs",
                cell_id=cell.cell_id,
                attempt=0,
                samples=sample_count,
                work_units=pair.work_units,
                wall_seconds=pair.wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_memory_bytes=0,
                successful=True,
            )
            work_ledger = work_ledger.append(work_record)
            residual_se = diagnostics.residual_standard_error
            residual_z = (
                0.0
                if residual_se == 0.0 and diagnostics.residual_mean == 0.0
                else math.inf
                if residual_se == 0.0
                else diagnostics.residual_mean / residual_se
            )
            raw_se = math.sqrt(diagnostics.raw_variance / sample_count)
            dcs_se = math.sqrt(diagnostics.dcs_variance / sample_count)
            records.append(
                {
                    "cell_id": cell.cell_id,
                    "task": cell.task,
                    "cluster": cluster,
                    "nominal_probability": cell.nominal_probability,
                    "reference_probability": reference_probability,
                    "reference_standard_error": reference_se,
                    "diagnostics": asdict(diagnostics),
                    "residual_z_score": residual_z,
                    "raw_reference_z_score": (
                        diagnostics.raw_mean - reference_probability
                    )
                    / math.hypot(raw_se, reference_se),
                    "dcs_reference_z_score": (
                        diagnostics.dcs_mean - reference_probability
                    )
                    / math.hypot(dcs_se, reference_se),
                    "work_record": asdict(work_record),
                }
            )
    cells_ids = sorted({str(record["cell_id"]) for record in records})
    cluster_effects = []
    for cluster in range(clusters):
        ratios = [
            math.log(
                float(record["diagnostics"]["raw_over_dcs_variance_ratio"])
            )
            for record in records
            if int(record["cluster"]) == cluster
        ]
        if len(ratios) != len(cells_ids):
            raise ValueError("every V7 cluster must contain every cell")
        cluster_effects.append(math.fsum(ratios) / len(ratios))
    effect = (
        {
            "cluster_count": 1,
            "mean_log_raw_over_dcs_variance": cluster_effects[0],
            "standard_error": None,
            "geometric_raw_over_dcs_variance_ratio": math.exp(
                cluster_effects[0]
            ),
            "one_sided_lower_raw_over_dcs_variance_ratio": None,
            "p_value_against_no_variance_reduction": None,
        }
        if smoke
        else _one_sided_variance_effect(
            cluster_effects,
            confidence_level=float(sampling["confidence_level"]),
        )
    )
    thresholds = config["development_thresholds"]
    maximum_residual_z = max(
        abs(float(record["residual_z_score"])) for record in records
    )
    maximum_orthogonality_correlation = max(
        abs(float(record["diagnostics"]["dcs_residual_correlation"]))
        if record["diagnostics"]["dcs_residual_correlation"] is not None
        else 0.0
        for record in records
    )
    maximum_orthogonality_z = max(
        abs(float(record["diagnostics"]["dcs_residual_covariance_z_score"]))
        for record in records
    )
    raw_lower_ratio = effect["one_sided_lower_raw_over_dcs_variance_ratio"]
    aggregate_variance_gate = smoke or (
        isinstance(raw_lower_ratio, (int, float))
        and not isinstance(raw_lower_ratio, bool)
        and float(raw_lower_ratio)
        >= float(thresholds["minimum_variance_ratio_lower"])
    )
    gates = {
        "complete_matrix": len(records) == len(cells) * clusters,
        "all_finite_positive_variances": all(
            math.isfinite(float(record["diagnostics"]["raw_variance"]))
            and float(record["diagnostics"]["raw_variance"]) > 0.0
            and math.isfinite(float(record["diagnostics"]["dcs_variance"]))
            and float(record["diagnostics"]["dcs_variance"]) > 0.0
            for record in records
        ),
        "all_variance_decompositions_numerical": all(
            abs(float(record["diagnostics"]["variance_decomposition_error"]))
            <= 1e-10
            * max(1.0, float(record["diagnostics"]["raw_variance"]))
            for record in records
        ),
        "aggregate_variance_ratio_lower": aggregate_variance_gate,
        "residual_mean_diagnostic": (
            True
            if smoke
            else maximum_residual_z
            <= float(thresholds["maximum_absolute_residual_z"])
        ),
        "orthogonality_diagnostic": True
        if smoke
        else (
            maximum_orthogonality_correlation
            <= float(thresholds["maximum_absolute_orthogonality_correlation"])
            if config["schema"] == _SCHEMA_V1
            else maximum_orthogonality_z
            <= float(thresholds["maximum_absolute_orthogonality_z"])
        ),
    }
    provenance = source_provenance()
    formal = {
        "frozen_config": bool(config["frozen"]),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
        "verified_training_source": bool(training_audit["verified"]),
        "formal_training_source": bool(
            training_audit["formal_training_source_readiness"]
        ),
    }
    return {
        "schema": "npi.g11.v7-mechanism-probe.v1",
        "protocol_id": config["protocol_id"],
        "config_schema": config["schema"],
        "phase": config["phase"],
        "smoke": smoke,
        "config_sha256": config_hash,
        "manifest_sha256": manifest.sha256,
        "reference_artifact_sha256": reference_hash,
        "proposal_training_audit": training_audit,
        "records": records,
        "paired_cluster_effect": effect,
        "maximum_absolute_residual_z": maximum_residual_z,
        "maximum_absolute_orthogonality_correlation": (
            maximum_orthogonality_correlation
        ),
        "maximum_absolute_orthogonality_z": maximum_orthogonality_z,
        "gates": gates,
        "formal_readiness": formal,
        "development_mechanism_passed": all(gates.values()),
        "formal_mechanism_passed": all(gates.values()) and all(formal.values()),
        "seed_ledger": seed_ledger.to_dict(),
        "seed_ledger_sha256": seed_ledger.sha256,
        "work_ledger": work_ledger.to_dict(),
        "work_ledger_sha256": work_ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--proposal-training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.manifest,
        arguments.reference,
        arguments.proposal_training_source,
        smoke=arguments.smoke,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "development_mechanism_passed": result[
                    "development_mechanism_passed"
                ],
                **result["gates"],
            }
        )
    )


if __name__ == "__main__":
    main()
