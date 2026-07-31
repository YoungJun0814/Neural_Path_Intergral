"""Representative R2 benchmark and conservative reference resource forecast."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import psutil

from experiments.g11_v8_p5_sharded_reference_common import (
    execute_actual_reference_shard,
    load_context,
)
from src.path_integral.provenance import source_provenance
from src.path_integral.reference_protocol import (
    ReferenceShardIdentity,
    canonical_json_bytes,
)
from src.path_integral.resource_planner import (
    ReferenceBenchmarkObservation,
    forecast_reference_resources,
)

RESULT_SCHEMA = "npi.g11.v8-p5-reference-benchmark.v1"


def _forecast(
    observations: list[ReferenceBenchmarkObservation],
    *,
    total_paths: int,
    steps: int,
    workers: int,
    parallel_efficiency: float,
    safety_factor: float,
    available_memory_bytes: int,
    maximum_wall_seconds: float,
) -> dict[str, Any]:
    return forecast_reference_resources(
        observations,
        total_paths=total_paths,
        steps=steps,
        workers=workers,
        parallel_efficiency=parallel_efficiency,
        safety_factor=safety_factor,
        available_memory_bytes=available_memory_bytes,
        maximum_wall_seconds=maximum_wall_seconds,
    )


def run_benchmark(config_path: Path) -> dict[str, Any]:
    context = load_context(config_path)
    config = context.config
    benchmark = config["benchmark"]
    sampling = config["sampling"]
    provenance = source_provenance()
    if (
        context.environment["torch_threads"]
        != benchmark["expected_torch_threads_per_worker"]
    ):
        raise RuntimeError("benchmark PyTorch thread count differs from its contract")
    sample_count = int(benchmark["samples_per_observation"])
    observations: list[dict[str, Any]] = []
    forecast_inputs: list[ReferenceBenchmarkObservation] = []
    for cell_id in benchmark["representative_cells"]:
        cell = context.cells_by_id[cell_id]
        for method in benchmark["methods"]:
            for repetition in range(int(benchmark["repetitions"])):
                identity = ReferenceShardIdentity(
                    protocol_id=config["protocol_id"],
                    namespace=benchmark["namespace"],
                    stage="pilot",
                    method=method,
                    cell_id=cell_id,
                    shard_index=repetition,
                )
                artifact = execute_actual_reference_shard(
                    context,
                    identity,
                    requested_samples=sample_count,
                    parent_sha256=context.reference_parent_sha256,
                    source_commit=provenance["source_commit"],
                    dirty_worktree=provenance["dirty_worktree"],
                    benchmark=True,
                )
                artifact_bytes = len(canonical_json_bytes(artifact)) + 1
                observation = ReferenceBenchmarkObservation(
                    paths=sample_count,
                    steps=int(cell["finest_steps"]),
                    wall_seconds=float(artifact["elapsed_wall_seconds"]),
                    cpu_seconds=float(artifact["elapsed_cpu_seconds"]),
                    peak_resident_memory_bytes=int(
                        artifact["peak_resident_memory_bytes"]
                    ),
                    artifact_bytes=artifact_bytes,
                )
                forecast_inputs.append(observation)
                observations.append(
                    {
                        "cell_id": cell_id,
                        "task": cell["task"],
                        "hurst": cell["hurst"],
                        "nominal_probability": cell["nominal_probability"],
                        "method": method,
                        "repetition": repetition,
                        "paths": observation.paths,
                        "steps": observation.steps,
                        "wall_seconds": observation.wall_seconds,
                        "cpu_seconds": observation.cpu_seconds,
                        "path_steps_per_wall_second": (
                            observation.path_steps_per_wall_second
                        ),
                        "peak_resident_memory_bytes": (
                            observation.peak_resident_memory_bytes
                        ),
                        "artifact_bytes": observation.artifact_bytes,
                        "seed_key_sha256": artifact["seed_key_sha256"],
                        "contribution_mean": artifact["contribution"]["mean"],
                        "contribution_variance": (
                            artifact["contribution"]["m2"]
                            / (observation.paths - 1)
                        ),
                        "likelihood_normalization_mean": artifact[
                            "likelihood_normalization"
                        ]["mean"],
                    }
                )

    methods = len(config["reference_contract"]["methods"])
    cells = len(context.cells_by_id)
    pilot_paths = (
        cells
        * methods
        * int(sampling["pilot_replicates"])
        * int(sampling["pilot_samples_per_replicate"])
    )
    method_caps = sampling.get("maximum_final_samples_by_method")
    maximum_final_paths = cells * (
        sum(int(method_caps[method]) for method in config["reference_contract"]["methods"])
        if isinstance(method_caps, dict)
        else methods * int(sampling["maximum_final_samples"])
    )
    steps = int(benchmark["forecast_steps"])
    safety_factor = float(benchmark["forecast_safety_factor"])
    local_memory = math.floor(
        psutil.virtual_memory().total
        * float(benchmark["local_total_memory_fraction"])
    )
    local_pilot = _forecast(
        forecast_inputs,
        total_paths=pilot_paths,
        steps=steps,
        workers=int(benchmark["local_workers"]),
        parallel_efficiency=float(benchmark["local_parallel_efficiency"]),
        safety_factor=safety_factor,
        available_memory_bytes=local_memory,
        maximum_wall_seconds=float(benchmark["local_maximum_wall_seconds"]),
    )
    external_pilot = _forecast(
        forecast_inputs,
        total_paths=pilot_paths,
        steps=steps,
        workers=int(benchmark["external_cpu_workers"]),
        parallel_efficiency=float(benchmark["external_parallel_efficiency"]),
        safety_factor=safety_factor,
        available_memory_bytes=int(benchmark["external_available_memory_bytes"]),
        maximum_wall_seconds=float(benchmark["external_maximum_wall_seconds"]),
    )
    local_worst_case_final = _forecast(
        forecast_inputs,
        total_paths=maximum_final_paths,
        steps=steps,
        workers=int(benchmark["local_workers"]),
        parallel_efficiency=float(benchmark["local_parallel_efficiency"]),
        safety_factor=safety_factor,
        available_memory_bytes=local_memory,
        maximum_wall_seconds=float(benchmark["local_maximum_wall_seconds"]),
    )
    external_worst_case_final = _forecast(
        forecast_inputs,
        total_paths=maximum_final_paths,
        steps=steps,
        workers=int(benchmark["external_cpu_workers"]),
        parallel_efficiency=float(benchmark["external_parallel_efficiency"]),
        safety_factor=safety_factor,
        available_memory_bytes=int(benchmark["external_available_memory_bytes"]),
        maximum_wall_seconds=float(benchmark["external_maximum_wall_seconds"]),
    )
    observation_roster_complete = len(observations) == (
        len(benchmark["representative_cells"])
        * len(benchmark["methods"])
        * int(benchmark["repetitions"])
    )
    finite_observations = all(
        math.isfinite(float(item["wall_seconds"]))
        and float(item["wall_seconds"]) > 0.0
        and math.isfinite(float(item["path_steps_per_wall_second"]))
        and float(item["path_steps_per_wall_second"]) > 0.0
        and math.isfinite(float(item["contribution_mean"]))
        and math.isfinite(float(item["contribution_variance"]))
        and math.isfinite(float(item["likelihood_normalization_mean"]))
        for item in observations
    )
    seed_keys = [str(item["seed_key_sha256"]) for item in observations]
    seed_roster_unique = len(seed_keys) == len(set(seed_keys))
    if local_pilot["launch_authorized"]:
        pilot_recommendation = "local_cpu"
    elif external_pilot["launch_authorized"]:
        pilot_recommendation = "external_32vcpu_cpu"
    else:
        pilot_recommendation = "redesign_or_larger_cpu"
    if (
        local_pilot["launch_authorized"]
        and local_worst_case_final["launch_authorized"]
    ):
        end_to_end_recommendation = "local_cpu"
    elif (
        external_pilot["launch_authorized"]
        and external_worst_case_final["launch_authorized"]
    ):
        end_to_end_recommendation = "external_32vcpu_cpu"
    else:
        end_to_end_recommendation = "redesign_or_larger_cpu"
    return {
        "schema": RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "config_sha256": context.config_sha256,
        "threshold_binding_sha256": context.binding_sha256,
        "threshold_manifest_sha256": context.binding["threshold_manifest_sha256"],
        "benchmark_namespace": benchmark["namespace"],
        "benchmark_is_formal_reference_evidence": False,
        "observations": observations,
        "workload": {
            "cell_count": cells,
            "method_count": methods,
            "pilot_paths": pilot_paths,
            "maximum_final_paths": maximum_final_paths,
            "steps": steps,
        },
        "forecasts": {
            "local_pilot": local_pilot,
            "external_32vcpu_pilot": external_pilot,
            "local_worst_case_final": local_worst_case_final,
            "external_32vcpu_worst_case_final": external_worst_case_final,
        },
        "gates": {
            "observation_roster_complete": observation_roster_complete,
            "finite_observations": finite_observations,
            "benchmark_seed_keys_unique": seed_roster_unique,
            "local_pilot_resource_feasible": local_pilot["launch_authorized"],
            "external_32vcpu_pilot_resource_feasible": external_pilot[
                "launch_authorized"
            ],
        },
        "decision": {
            "status": (
                "representative_benchmark_pass"
                if observation_roster_complete
                and finite_observations
                and seed_roster_unique
                else "representative_benchmark_fail"
            ),
            "recommended_pilot_hardware": pilot_recommendation,
            "recommended_end_to_end_hardware": end_to_end_recommendation,
            "external_forecast_is_planning_only": True,
            "external_hardware_benchmark_required_before_launch": True,
            "formal_pilot_may_start_only_from_clean_committed_source": True,
            "full_pilot_execution_authorized": False,
            "final_execution_authorized": False,
            "reference_complete": False,
            "performance_claim_authorized": False,
        },
        "environment": context.environment,
        "environment_sha256": context.environment_sha256,
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"refusing to overwrite reference benchmark: {arguments.output}"
        )
    result = run_benchmark(arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))
    if result["decision"]["status"] != "representative_benchmark_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
