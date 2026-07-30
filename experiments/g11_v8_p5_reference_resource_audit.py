"""Audit the production-size benchmark and formal-pilot authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.g11_v8_p5_reference_pilot import (
    AUTHORIZATION_SCHEMA,
    _verify_authorization,
)
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from src.path_integral.reference_protocol import REFERENCE_METHODS
from src.path_integral.resource_planner import (
    ReferenceBenchmarkObservation,
    forecast_reference_resources,
)

REPORT_SCHEMA = "npi.g11.v8-p5-reference-resource-audit.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("resource authorization binding is malformed")
    path = (ROOT / str(record["path"])).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("resource authorization binding path is invalid")
    if record["sha256"] != _sha256(path):
        raise ValueError("resource authorization binding hash mismatch")
    return path


def load_authorization(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != AUTHORIZATION_SCHEMA:
        raise ValueError("unexpected P5 pilot authorization")
    return value, hashlib.sha256(raw).hexdigest()


def _forecasts(
    benchmark: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    observations = [
        ReferenceBenchmarkObservation(
            int(item["paths"]),
            int(item["steps"]),
            float(item["wall_seconds"]),
            float(item["cpu_seconds"]),
            int(item["peak_resident_memory_bytes"]),
            int(item["artifact_bytes"]),
        )
        for item in benchmark["observations"]
    ]
    settings = config["benchmark"]
    workload = benchmark["workload"]
    recorded = benchmark["forecasts"]

    def calculate(
        total_paths: int,
        workers: int,
        efficiency: float,
        memory: int,
        wall: float,
    ) -> dict[str, Any]:
        return forecast_reference_resources(
            observations,
            total_paths=total_paths,
            steps=int(workload["steps"]),
            workers=workers,
            parallel_efficiency=efficiency,
            safety_factor=float(settings["forecast_safety_factor"]),
            available_memory_bytes=memory,
            maximum_wall_seconds=wall,
        )

    local_memory = int(recorded["local_pilot"]["available_memory_bytes"])
    external_memory = int(settings["external_available_memory_bytes"])
    pilot_paths = int(workload["pilot_paths"])
    final_paths = int(workload["maximum_final_paths"])
    return {
        "local_pilot": calculate(
            pilot_paths,
            int(settings["local_workers"]),
            float(settings["local_parallel_efficiency"]),
            local_memory,
            float(settings["local_maximum_wall_seconds"]),
        ),
        "external_32vcpu_pilot": calculate(
            pilot_paths,
            int(settings["external_cpu_workers"]),
            float(settings["external_parallel_efficiency"]),
            external_memory,
            float(settings["external_maximum_wall_seconds"]),
        ),
        "local_worst_case_final": calculate(
            final_paths,
            int(settings["local_workers"]),
            float(settings["local_parallel_efficiency"]),
            local_memory,
            float(settings["local_maximum_wall_seconds"]),
        ),
        "external_32vcpu_worst_case_final": calculate(
            final_paths,
            int(settings["external_cpu_workers"]),
            float(settings["external_parallel_efficiency"]),
            external_memory,
            float(settings["external_maximum_wall_seconds"]),
        ),
    }


def audit_resource_authorization(
    authorization: dict[str, Any], authorization_sha256: str
) -> dict[str, Any]:
    load_ok = True
    try:
        config_path = _bound_path(authorization["execution_config"])
        benchmark_path = _bound_path(authorization["representative_benchmark"])
        implementation_path = _bound_path(authorization["implementation_manifest"])
        context = load_context(config_path)
        _verify_authorization(
            ROOT / "configs/g11_v8/p5_reference_pilot_authorization_v1.yaml",
            context,
        )
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if not isinstance(benchmark, dict):
            raise ValueError("benchmark is not a mapping")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        load_ok = False
        context = None
        benchmark = {}
        implementation_path = None
    config = context.config if context is not None else {}
    settings = config.get("benchmark", {})
    observations = benchmark.get("observations", [])
    roster = {
        (item.get("cell_id"), item.get("method"), item.get("repetition"))
        for item in observations
        if isinstance(item, dict)
    }
    expected_roster = {
        (cell, method, repetition)
        for cell in settings.get("representative_cells", [])
        for method in REFERENCE_METHODS
        for repetition in range(int(settings.get("repetitions", 0)))
    }
    seeds = [
        item.get("seed_key_sha256")
        for item in observations
        if isinstance(item, dict)
    ]
    try:
        recomputed = _forecasts(benchmark, config)
    except (KeyError, TypeError, ValueError):
        recomputed = {}
    recorded = benchmark.get("forecasts", {})
    local_pilot = recorded.get("local_pilot", {})
    local_final = recorded.get("local_worst_case_final", {})
    external_final = recorded.get("external_32vcpu_worst_case_final", {})
    formal = authorization.get("formal_pilot", {})
    final = authorization.get("final_execution", {})
    decision = authorization.get("decision", {})
    checks = {
        "all_bound_inputs_and_implementation_pass": load_ok
        and implementation_path is not None
        and formal.get("required_implementation_manifest_sha256")
        == _sha256(implementation_path),
        "clean_production_size_benchmark": benchmark.get("dirty_worktree") is False
        and benchmark.get("source_commit")
        == "d778864952d4f33ca53a1b144e6709e08127ee50"
        and settings.get("samples_per_observation") == 32768
        and benchmark.get("benchmark_is_formal_reference_evidence") is False,
        "benchmark_roster_and_seeds_exact": roster == expected_roster
        and len(observations) == len(expected_roster)
        and len(seeds) == len(set(seeds))
        and all(isinstance(seed, str) and len(seed) == 64 for seed in seeds),
        "benchmark_gates_pass": benchmark.get("decision", {}).get("status")
        == "representative_benchmark_pass"
        and all(benchmark.get("gates", {}).values()),
        "forecasts_exactly_recomputed": recomputed == recorded,
        "only_local_pilot_resource_feasible": local_pilot.get(
            "launch_authorized"
        )
        is True
        and local_final.get("launch_authorized") is False
        and external_final.get("launch_authorized") is False,
        "formal_pilot_authorized_exactly": formal.get("namespace")
        == config.get("sampling", {}).get("pilot_namespace")
        and formal.get("total_paths") == 12582912
        and formal.get("benchmark_source_commit") == benchmark.get("source_commit")
        and formal.get("laptop_predicted_wall_seconds")
        == local_pilot.get("predicted_wall_seconds")
        and formal.get("laptop_execution_authorized") is True,
        "final_remains_closed": final.get("laptop_worst_case_predicted_wall_seconds")
        == local_final.get("predicted_wall_seconds")
        and final.get("external_32vcpu_worst_case_predicted_wall_seconds")
        == external_final.get("predicted_wall_seconds")
        and final.get("laptop_worst_case_authorized") is False
        and final.get("external_32vcpu_forecast_is_planning_only") is True
        and final.get("external_hardware_benchmark_required") is True
        and final.get("execution_authorized") is False,
        "decision_fail_closed": authorization.get(
            "formal_pilot_outcomes_inspected_before_authorization"
        )
        is False
        and decision.get("formal_pilot_execution_authorized") is True
        and decision.get("final_execution_authorized") is False
        and decision.get("reference_complete") is False
        and decision.get("performance_claim_authorized") is False
        and decision.get("submission_authorized") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": REPORT_SCHEMA,
        "authorization_sha256": authorization_sha256,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "decision": {
            "status": (
                "formal_pilot_authorization_pass"
                if not failures
                else "formal_pilot_authorization_fail"
            ),
            "formal_pilot_execution_authorized": not failures,
            "final_execution_authorized": False,
            "reference_complete": False,
            "performance_claim_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    authorization, digest = load_authorization(arguments.authorization)
    report = audit_resource_authorization(authorization, digest)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is not None:
        if arguments.output.exists():
            raise FileExistsError(
                f"refusing to overwrite resource audit: {arguments.output}"
            )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
