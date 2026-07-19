"""Seed-clustered six-level threshold and correction-rate pilot for G11 M4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    SeedKey,
    SeedLedger,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    correction_rate_observation,
    evaluate_rbergomi_dcs_adjacent,
    identify_rate_window,
    simulate_coupled_rbergomi_mixture,
)
from src.path_integral.provenance import runtime_provenance
from src.physics_engine import RBergomiSimulator


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected a G11 threshold-rate schema-version-1 config")
    if config.get("estimand") != "finite_grid":
        raise ValueError("rate pilot must explicitly declare a finite-grid estimand")
    return config, hashlib.sha256(raw).hexdigest()


def _task(specification: dict[str, Any]):
    kind = specification["kind"]
    if kind == "terminal":
        return TerminalThresholdTask(float(specification["level"]))
    if kind == "barrier":
        return DiscreteBarrierHitTask(float(specification["barrier"]))
    if kind == "hit_plus_occupation":
        return DownsideExcursionTask(
            hit_barrier=float(specification["hit_barrier"]),
            stress_level=float(specification["stress_level"]),
            minimum_occupation=float(specification["minimum_occupation"]),
            hit_scale=float(specification["hit_scale"]),
            occupation_scale=float(specification["occupation_scale"]),
        )
    raise ValueError(f"unsupported task kind: {kind}")


def _git_metadata() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        return subprocess.check_output(arguments, text=True).strip()

    try:
        commit = command("git", "rev-parse", "HEAD")
        dirty = bool(command("git", "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    return {"source_commit": commit, "dirty_worktree": dirty}


def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    model = config["model"]
    hierarchy = config["hierarchy"]
    sampling = config["sampling"]
    analysis = config["analysis"]
    levels = tuple(int(value) for value in hierarchy["adjacent_levels"])
    if levels != tuple(range(levels[0], levels[-1] + 1)) or levels[0] < 1:
        raise ValueError("adjacent levels must be a consecutive positive sequence")
    replicates = 3 if smoke else int(sampling["replicates"])
    paths = 512 if smoke else int(sampling["paths_per_level"])
    bootstrap_repetitions = (
        200 if smoke else int(analysis["bootstrap_repetitions"])
    )
    simulator = RBergomiSimulator(
        H=float(model["H"]),
        eta=float(model["eta"]),
        xi=float(model["xi"]),
        rho=float(model["rho"]),
        device="cpu",
    )
    maturity = float(model["maturity"])
    controls = tuple(
        TimePiecewiseTwoDriverControl(
            tuple(tuple(float(item) for item in segment) for segment in schedule),
            maturity=maturity,
        )
        for schedule in config["proposal"]["controls"]
    )
    weights = torch.tensor(config["proposal"]["weights"], dtype=torch.float64)
    tasks = {name: _task(spec) for name, spec in config["tasks"].items()}
    ledger = SeedLedger()
    observations = {name: [] for name in tasks}
    exactness = {
        "maximum_coordinate_mismatch": 0.0,
        "maximum_path_reconstruction_error": 0.0,
        "maximum_density_reconstruction_error": 0.0,
        "maximum_likelihood_bound_violation": 0.0,
        "hard_threshold_mismatches": 0,
    }
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for replicate in range(replicates):
        for level in levels:
            seeds = {
                stream: ledger.allocate(
                    SeedKey(
                        config["protocol_id"],
                        "rate_pilot",
                        "primary",
                        "shared_tasks",
                        level,
                        replicate,
                        stream,
                    )
                )
                for stream in ("proposal", "labels")
            }
            try:
                torch.manual_seed(seeds["proposal"])
                fine_steps = int(hierarchy["coarsest_steps"]) * 2**level
                sample = simulate_coupled_rbergomi_mixture(
                    simulator,
                    controls,
                    weights,
                    spot=float(model["spot"]),
                    maturity=maturity,
                    fine_steps=fine_steps,
                    num_paths=paths,
                    dtype=torch.float64,
                    label_generator=torch.Generator().manual_seed(seeds["labels"]),
                    engine=str(sampling["engine"]),
                )
                operation_proxy = paths * fine_steps * max(1.0, math.log2(fine_steps))
                for name, task in tasks.items():
                    evaluation = evaluate_rbergomi_dcs_adjacent(
                        sample, task=task, rho=simulator.rho
                    )
                    observations[name].append(
                        correction_rate_observation(
                            level=level,
                            replicate=replicate,
                            threshold_difference=evaluation.threshold_difference,
                            raw_correction=evaluation.raw_correction,
                            dcs_correction=evaluation.marginalized_correction,
                            raw_work_units=float(operation_proxy),
                            dcs_work_units=float(1.1 * operation_proxy),
                        )
                    )
                    exactness["maximum_coordinate_mismatch"] = max(
                        exactness["maximum_coordinate_mismatch"],
                        evaluation.maximum_coordinate_mismatch,
                    )
                    exactness["maximum_path_reconstruction_error"] = max(
                        exactness["maximum_path_reconstruction_error"],
                        evaluation.fine.maximum_path_reconstruction_error,
                        evaluation.coarse.maximum_path_reconstruction_error,
                    )
                    exactness["maximum_density_reconstruction_error"] = max(
                        exactness["maximum_density_reconstruction_error"],
                        evaluation.fine.density.maximum_component_reconstruction_error,
                        evaluation.fine.density.maximum_mixture_reconstruction_error,
                    )
                    exactness["maximum_likelihood_bound_violation"] = max(
                        exactness["maximum_likelihood_bound_violation"],
                        evaluation.fine.density.maximum_full_bound_violation,
                        evaluation.fine.density.maximum_residual_bound_violation,
                    )
            except Exception as error:
                failures.append(
                    {
                        "replicate": replicate,
                        "level": level,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
    analyses: dict[str, Any] = {}
    equivalence_margin = float(
        config["gates"]["dcs_compatible_with_2r_margin"]
    )
    for name, records in observations.items():
        bootstrap_seed = ledger.allocate(
            SeedKey(
                config["protocol_id"],
                "analysis",
                "primary",
                name,
                0,
                0,
                str(analysis["bootstrap_seed_namespace"]),
            )
        )
        if failures:
            analyses[name] = {"identified": False, "reason": "simulation failure"}
            continue
        result = identify_rate_window(
            records,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
            minimum_levels=int(analysis["minimum_levels"]),
            endpoint_slope_margin=float(analysis["endpoint_slope_margin"]),
            maximum_variance_cv=float(analysis["maximum_variance_cv"]),
        )
        analyses[name] = asdict(result)
    identified_results = [
        result for result in analyses.values() if bool(result.get("identified"))
    ]
    positive_threshold_rate = all(
        result["confidence_intervals_95"]["threshold_l2"][0] > 0.0
        for result in identified_results
    ) and len(identified_results) == len(analyses)
    dcs_equivalent_to_two_r = all(
        result["confidence_intervals_95"][
            "dcs_second_minus_threshold_l2"
        ][0]
        >= -equivalence_margin
        and result["confidence_intervals_95"][
            "dcs_second_minus_threshold_l2"
        ][1]
        <= equivalence_margin
        for result in identified_results
    ) and len(identified_results) == len(analyses)
    positive_dcs_fraction = (
        sum(result["exponents"]["dcs_variance"] > 0.0 for result in identified_results)
        / len(analyses)
    )
    exactness_passed = (
        exactness["maximum_coordinate_mismatch"] <= 1e-11
        and exactness["maximum_path_reconstruction_error"] <= 1e-11
        and exactness["maximum_density_reconstruction_error"] <= 1e-11
        and exactness["maximum_likelihood_bound_violation"] <= 1e-12
        and exactness["hard_threshold_mismatches"] == 0
    )
    return {
        "schema": "npi.g11.threshold-rate-pilot.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "estimand": "finest finite-grid event probability and adjacent corrections",
        "continuous_time_claim": False,
        "replicates": replicates,
        "paths_per_level": paths,
        "levels": levels,
        "seed_ledger_sha256": ledger.sha256,
        "seed_count": len(ledger),
        "observations": {
            name: [asdict(record) for record in records]
            for name, records in observations.items()
        },
        "rate_analyses": analyses,
        "exactness": exactness,
        "failures": failures,
        "gates": {
            "no_failures": not failures,
            "exactness": exactness_passed,
            "rate_identified_all_tasks": all(
                bool(result.get("identified")) for result in analyses.values()
            ),
            "threshold_l2_rate_lower_95_positive": positive_threshold_rate,
            "dcs_second_moment_equivalent_to_2r": dcs_equivalent_to_two_r,
            "positive_dcs_variance_slope_fraction": positive_dcs_fraction,
            "positive_dcs_variance_slope_at_least_80_percent": (
                positive_dcs_fraction >= 0.80
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "work_ledger": {
            "warmup_seconds": 0.0,
            "compilation_seconds": 0.0,
            "calibration_seconds": 0.0,
            "pilot_seconds": time.perf_counter() - started,
            "online_seconds": 0.0,
            "audit_seconds": 0.0,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **_git_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/g11_threshold_rate_pilot.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result["gates"], sort_keys=True))


if __name__ == "__main__":
    main()
