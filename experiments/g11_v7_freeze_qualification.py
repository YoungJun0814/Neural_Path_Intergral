"""Outcome-locked freeze receipt for the V7 qualification study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import scipy.stats

from experiments.g11_v6_secondary_baselines import (
    _load_config as load_fixed_config,
)
from experiments.g11_v7_accuracy_analysis import (
    _load_config as load_accuracy_config,
)
from experiments.g11_v7_mechanism_analysis import (
    _load_config as load_analysis_config,
)
from experiments.g11_v7_mechanism_probe import (
    _load_config as load_probe_config,
)
from src.path_integral.provenance import runtime_provenance, source_provenance


def _load_json(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    return payload, hashlib.sha256(raw).hexdigest()


def _planned_clusters(
    *,
    mean_log_ratio: float,
    standard_error: float,
    development_clusters: int,
    practical_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    values = (
        mean_log_ratio,
        standard_error,
        practical_ratio,
        alpha,
        power,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("power inputs must be finite")
    if (
        development_clusters < 2
        or standard_error < 0.0
        or practical_ratio <= 1.0
        or not 0.0 < alpha < 1.0
        or not 0.0 < power < 1.0
    ):
        raise ValueError("invalid V7 power inputs")
    margin = mean_log_ratio - math.log(practical_ratio)
    if margin <= 0.0:
        raise ValueError("development effect does not exceed the practical ratio")
    cluster_sd = standard_error * math.sqrt(development_clusters)
    z_alpha = float(scipy.stats.norm.ppf(1.0 - alpha))
    z_power = float(scipy.stats.norm.ppf(power))
    required = math.ceil(((z_alpha + z_power) * cluster_sd / margin) ** 2)
    return max(2, required)


def freeze(
    development_probe_path: Path,
    development_fixed_path: Path,
    development_analysis_path: Path,
    probe_config_path: Path,
    fixed_config_path: Path,
    analysis_config_path: Path,
    accuracy_config_path: Path,
) -> dict[str, Any]:
    probe_dev, probe_dev_hash = _load_json(
        development_probe_path,
        "npi.g11.v7-mechanism-probe.v1",
    )
    fixed_dev, fixed_dev_hash = _load_json(
        development_fixed_path,
        "npi.g11.v6-secondary-baselines.v1",
    )
    analysis_dev, analysis_dev_hash = _load_json(
        development_analysis_path,
        "npi.g11.v7-mechanism-analysis.v1",
    )
    if (
        not probe_dev.get("development_mechanism_passed")
        or not all(
            bool(value)
            for value in fixed_dev.get("qualification_gates", {}).values()
        )
        or not analysis_dev.get("development_mechanism_qualified")
    ):
        raise ValueError("V7 development evidence did not authorize qualification")
    if (
        probe_dev.get("source_commit") != fixed_dev.get("source_commit")
        or bool(probe_dev.get("dirty_worktree"))
        or bool(fixed_dev.get("dirty_worktree"))
    ):
        raise ValueError("V7 development execution-source identity drifted")
    if analysis_dev.get("source_artifact_sha256") != {
        "mechanism_probe": probe_dev_hash,
        "fixed_estimators": fixed_dev_hash,
    }:
        raise ValueError("V7 development analysis does not bind the supplied inputs")
    probe_config, probe_config_hash = load_probe_config(probe_config_path)
    fixed_config, fixed_config_hash = load_fixed_config(fixed_config_path)
    analysis_config, analysis_config_hash = load_analysis_config(
        analysis_config_path
    )
    accuracy_config, accuracy_config_hash = load_accuracy_config(
        accuracy_config_path
    )
    configs = (
        probe_config,
        fixed_config,
        analysis_config,
        accuracy_config,
    )
    if any(
        config.get("phase") != "qualification"
        or config.get("frozen") is not True
        for config in configs
    ):
        raise ValueError("every V7 qualification config must be frozen")
    planned_clusters = 24
    if (
        int(probe_config["sampling"]["clusters"]) != planned_clusters
        or int(fixed_config["sampling"]["clusters"]) != planned_clusters
        or int(analysis_config["matrix"]["expected_clusters"])
        != planned_clusters
        or int(accuracy_config["matrix"]["expected_clusters"])
        != planned_clusters
    ):
        raise ValueError("V7 qualification cluster counts are inconsistent")
    development_clusters = int(analysis_dev["cluster_count"])
    effects = analysis_dev["effects"]
    thresholds = analysis_config["development_thresholds"]
    power_requirements = {
        "probe_variance": _planned_clusters(
            mean_log_ratio=float(
                probe_dev["paired_cluster_effect"][
                    "mean_log_raw_over_dcs_variance"
                ]
            ),
            standard_error=float(
                probe_dev["paired_cluster_effect"]["standard_error"]
            ),
            development_clusters=development_clusters,
            practical_ratio=float(
                thresholds["minimum_probe_variance_ratio_lower"]
            ),
        ),
        "execution_variance": _planned_clusters(
            mean_log_ratio=float(
                effects["execution_variance"]["mean_log_ratio"]
            ),
            standard_error=float(
                effects["execution_variance"]["standard_error"]
            ),
            development_clusters=development_clusters,
            practical_ratio=float(
                thresholds["minimum_execution_variance_ratio_lower"]
            ),
        ),
        "final_work": _planned_clusters(
            mean_log_ratio=float(effects["final_work"]["mean_log_ratio"]),
            standard_error=float(effects["final_work"]["standard_error"]),
            development_clusters=development_clusters,
            practical_ratio=float(
                thresholds["minimum_final_work_ratio_lower"]
            ),
        ),
    }
    if planned_clusters < max(power_requirements.values()):
        raise ValueError("planned V7 clusters do not satisfy development power")
    proposal_hashes = {
        probe_config["proposal"]["training_source_artifact_sha256"],
        fixed_config["proposal"]["training_source_artifact_sha256"],
        probe_dev["proposal_training_audit"]["source_artifact_sha256"],
        fixed_dev["proposal_training_audit"]["source_artifact_sha256"],
    }
    if len(proposal_hashes) != 1:
        raise ValueError("V7 proposal training-source identity drifted")
    provenance = source_provenance()
    if provenance["dirty_worktree"]:
        raise ValueError("V7 qualification freeze requires a clean source tree")
    return {
        "schema": "npi.g11.v7-qualification-freeze.v1",
        "source_commit": provenance["source_commit"],
        "dirty_worktree": provenance["dirty_worktree"],
        "planned_clusters": planned_clusters,
        "minimum_planned_clusters": 24,
        "power_alpha_one_sided": 0.05,
        "power_target": 0.80,
        "power_required_clusters": power_requirements,
        "development_artifact_sha256": {
            "mechanism_probe": probe_dev_hash,
            "fixed_estimators": fixed_dev_hash,
            "joint_analysis": analysis_dev_hash,
        },
        "qualification_config_sha256": {
            "mechanism_probe": probe_config_hash,
            "fixed_estimators": fixed_config_hash,
            "joint_analysis": analysis_config_hash,
            "simultaneous_accuracy": accuracy_config_hash,
        },
        "input_identity": {
            "manifest": fixed_dev["manifest_sha256"],
            "reference": fixed_dev["reference_artifact_sha256"],
            "proposal_training_source": proposal_hashes.pop(),
        },
        "development_execution_source": fixed_dev["source_commit"],
        "qualification_authorized": True,
        "environment": runtime_provenance(dtype="serialized-float64"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-probe", type=Path, required=True)
    parser.add_argument("--development-fixed", type=Path, required=True)
    parser.add_argument("--development-analysis", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
    parser.add_argument("--fixed-config", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--accuracy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = freeze(
        arguments.development_probe,
        arguments.development_fixed,
        arguments.development_analysis,
        arguments.probe_config,
        arguments.fixed_config,
        arguments.analysis_config,
        arguments.accuracy_config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qualification_authorized": result[
                    "qualification_authorized"
                ],
                "planned_clusters": result["planned_clusters"],
                "power_required_clusters": result[
                    "power_required_clusters"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
