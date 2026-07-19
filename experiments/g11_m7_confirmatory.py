"""Frozen, resource-censored M7 confirmation for G11 DCS-MGI-MLMC.

The runner deliberately keeps the established MLMC mathematics unchanged.  A raw
baseline allocation that exceeds the frozen per-level ceiling is truncated only in
this orchestration layer and is reported as censored if its empirical target is then
missed.  DCS allocations are never truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import geometric_mean
from typing import Any, Literal, cast

import scipy.stats
import torch
import yaml

from src.path_integral import (
    DiscreteBarrierHitTask,
    DownsideExcursionTask,
    FixedFinestGridTarget,
    MLMCHierarchy,
    MLMCPreparedRun,
    RBergomiMLMCSampler,
    RBergomiMLMCSamplerConfig,
    TerminalThresholdTask,
    TimePiecewiseTwoDriverControl,
    execute_mlmc,
    prepare_mlmc,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.physics_engine import RBergomiSimulator

M7Method = Literal["raw_defensive", "dcs_mgi"]
M7Engine = Literal["reference", "fft"]
METHODS: tuple[M7Method, ...] = ("raw_defensive", "dcs_mgi")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("expected an M7 schema-version-1 config")
    run_class = config.get("run_class")
    if run_class not in {"qualification", "confirmatory"}:
        raise ValueError("run_class must be qualification or confirmatory")
    if config.get("estimand") != "finite_grid":
        raise ValueError("M7 is restricted to an explicit finite-grid estimand")
    if (run_class == "confirmatory") is not bool(config.get("frozen")):
        raise ValueError("only the confirmatory config may be frozen")
    return config, hashlib.sha256(raw).hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), text=True).strip()


def _verify_source(config: dict[str, Any]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for declaration in config["core_source_manifest"]:
        path = Path(declaration["path"])
        actual = _sha256(path)
        expected = str(declaration["sha256"])
        if actual != expected:
            raise ValueError(f"core source hash mismatch: {path}")
        manifest.append({"path": str(path), "sha256": actual})

    core_commit = str(config["core_source_commit"])
    head = _git_output("rev-parse", "HEAD")
    try:
        subprocess.check_call(
            ("git", "merge-base", "--is-ancestor", core_commit, head),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("core source commit is not an ancestor of HEAD") from error

    if config["run_class"] == "confirmatory":
        tag = str(config["required_git_tag"])
        tag_commit = _git_output("rev-list", "-n", "1", tag)
        if tag_commit != head:
            raise ValueError("confirmatory HEAD does not match the frozen Git tag")
        provenance = source_provenance()
        if provenance["dirty_worktree"]:
            raise ValueError("confirmatory execution requires a clean Git worktree")
    return manifest


def _verified_payload(declaration: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(declaration["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input artifact is not an object: {path}")
    actual = _canonical_sha256(payload)
    if actual != str(declaration["canonical_sha256"]):
        raise ValueError(f"input artifact canonical hash mismatch: {path}")
    return payload, actual


def _verified_yaml(declaration: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(declaration["path"])
    payload = yaml.safe_load(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"input config is not a mapping: {path}")
    actual = _canonical_sha256(payload)
    if actual != str(declaration["canonical_sha256"]):
        raise ValueError(f"input config canonical hash mismatch: {path}")
    return payload, actual


def _probability_suffix(probability: float) -> str:
    exponent = round(-math.log10(probability))
    if not math.isclose(probability, 10.0 ** (-exponent), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("M7 probabilities must be exact negative powers of ten")
    return f"1e{exponent}"


def _event_task(specification: dict[str, Any]):
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
    raise ValueError(f"unsupported M7 task kind: {kind}")


def _load_regimes(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    contexts: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    expected_weights = [float(value) for value in config["proposal"]["weights"]]
    expected_controls = config["proposal"]["controls"]
    minimum_exclusion = float(config["selection"]["minimum_excursion_exclusion_fraction"])

    for declaration in config["regimes"]:
        calibration_config, config_hash = _verified_yaml(
            declaration["calibration_config"]
        )
        calibration_result, result_hash = _verified_payload(
            declaration["calibration_result"]
        )
        inputs.extend(
            (
                {
                    "path": str(declaration["calibration_config"]["path"]),
                    "canonical_sha256": config_hash,
                    "generation_sha256": str(
                        declaration["calibration_config"]["generation_sha256"]
                    ),
                },
                {
                    "path": str(declaration["calibration_result"]["path"]),
                    "canonical_sha256": result_hash,
                },
            )
        )
        if calibration_result.get("config_sha256") != str(
            declaration["calibration_config"]["generation_sha256"]
        ):
            raise ValueError("calibration result does not match its config")
        if calibration_result.get("smoke") is not False:
            raise ValueError("M7 cannot use smoke calibration")
        if calibration_result.get("continuous_time_claim") is not False:
            raise ValueError("M7 calibration must reject a continuous-time claim")
        gates = calibration_result.get("gates", {})
        for gate in (
            "all_probability_bands",
            "all_relative_standard_errors",
            "calibration_validation_seed_roles_disjoint",
            "likelihood_normalization",
        ):
            if gates.get(gate) is not True:
                raise ValueError(f"required calibration gate failed: {gate}")
        if [float(value) for value in calibration_config["proposal"]["weights"]] != expected_weights:
            raise ValueError("regime proposal weights differ from the frozen proposal")
        if calibration_config["proposal"]["controls"] != expected_controls:
            raise ValueError("regime controls differ from the frozen proposal")
        if int(calibration_config["finest_steps"]) != int(
            config["hierarchy"]["coarsest_steps"]
        ) * int(config["hierarchy"]["refinement"]) ** int(
            config["hierarchy"]["finest_level"]
        ):
            raise ValueError("calibration grid differs from the frozen MLMC grid")

        cells = calibration_result.get("cells")
        if not isinstance(cells, list):
            raise ValueError("calibration result has no cell list")
        tasks: list[dict[str, Any]] = []
        for task_name in declaration["included_tasks"]:
            base = calibration_config["tasks"][task_name]
            for raw_probability in declaration["target_probabilities"]:
                probability = float(raw_probability)
                matches = [
                    cell
                    for cell in cells
                    if cell["task"] == task_name
                    and math.isclose(
                        float(cell["target_probability"]),
                        probability,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ]
                if len(matches) != 1:
                    raise ValueError("selected calibration cell is missing or duplicated")
                cell = matches[0]
                if cell.get("probability_band_passed") is not True or cell.get(
                    "precision_passed"
                ) is not True:
                    raise ValueError("selected calibration cell failed its gates")
                threshold = float(cell["calibrated_threshold"])
                if task_name == "terminal":
                    task_specification = {"kind": "terminal", "level": threshold}
                elif task_name == "barrier":
                    task_specification = {"kind": "barrier", "barrier": threshold}
                elif task_name == "excursion":
                    exclusion = float(cell["occupation_exclusion_fraction"])
                    if exclusion < minimum_exclusion:
                        raise ValueError("selected excursion cell is degenerate")
                    task_specification = {
                        "kind": "hit_plus_occupation",
                        "hit_barrier": threshold,
                        "stress_level": float(base["stress_level"]),
                        "minimum_occupation": float(base["minimum_occupation"]),
                        "hit_scale": float(base["hit_scale"]),
                        "occupation_scale": float(base["occupation_scale"]),
                    }
                else:
                    raise ValueError("unsupported task selected by M7")
                tasks.append(
                    {
                        "id": f"{task_name}_{_probability_suffix(probability)}",
                        "name": task_name,
                        "target_probability": probability,
                        "specification": task_specification,
                    }
                )
        contexts.append(
            {
                "name": str(declaration["name"]),
                "model": calibration_config["model"],
                "tasks": tasks,
            }
        )
    if len({context["name"] for context in contexts}) != len(contexts):
        raise ValueError("M7 regime names must be unique")
    return contexts, inputs


def expected_cell_count(config: dict[str, Any]) -> int:
    contexts, _ = _load_regimes(config)
    repetitions = int(config["sampling"]["repetitions"])
    return repetitions * sum(len(context["tasks"]) for context in contexts)


def cap_raw_allocation(
    prepared: MLMCPreparedRun, maximum_final_samples_per_level: int
) -> tuple[MLMCPreparedRun, tuple[int, ...], bool]:
    """Apply the frozen raw-only cap without changing MLMC core semantics."""

    if maximum_final_samples_per_level < 2:
        raise ValueError("raw final-sample ceiling must be at least two")
    uncapped = tuple(item.final_count for item in prepared.allocations)
    allocations = tuple(
        replace(
            item,
            final_count=min(item.final_count, maximum_final_samples_per_level),
        )
        for item in prepared.allocations
    )
    return (
        replace(prepared, allocations=allocations),
        uncapped,
        any(after.final_count < before for after, before in zip(allocations, uncapped, strict=True)),
    )


def _method_result(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    task_item: dict[str, Any],
    replicate: int,
    method: M7Method,
) -> dict[str, Any]:
    process_cpu_started = time.process_time()
    model = context["model"]
    proposal = config["proposal"]
    sampling = config["sampling"]
    hierarchy_specification = config["hierarchy"]
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
            tuple(
                (float(segment[0]), float(segment[1]))
                for segment in schedule
            ),
            maturity=maturity,
        )
        for schedule in proposal["controls"]
    )
    weights = torch.tensor(proposal["weights"], dtype=torch.float64)
    hierarchy = MLMCHierarchy(
        int(hierarchy_specification["coarsest_steps"]),
        int(hierarchy_specification["refinement"]),
        FixedFinestGridTarget(int(hierarchy_specification["finest_level"])),
    )
    task = _event_task(task_item["specification"])
    rmse = float(task_item["target_probability"]) * float(
        sampling["relative_rmse_target"]
    )
    protocol = (
        f"{config['protocol_id']}:regime={context['name']}:"
        f"task={task_item['id']}:rep={replicate}"
    )
    engine_text = str(sampling["engine"])
    if engine_text not in {"reference", "fft"}:
        raise ValueError("M7 engine must be reference or fft")
    engine = cast(M7Engine, engine_text)
    sampler = RBergomiMLMCSampler(
        simulator,
        controls,
        weights,
        task,
        RBergomiMLMCSamplerConfig(
            spot=float(model["spot"]),
            maturity=maturity,
            coarsest_steps=hierarchy.coarsest_steps,
            method=method,
            engine=engine,
        ),
    )
    preparation_started = time.perf_counter()
    prepared = prepare_mlmc(
        hierarchy,
        sampler,
        protocol=protocol,
        regime=str(context["name"]),
        task=str(task_item["id"]),
        sampling_variance_target=rmse**2,
        pilot_samples=int(sampling["pilot_samples"]),
        minimum_final_samples=int(sampling["minimum_final_samples"]),
        chunk_size=int(sampling["chunk_size"]),
        allocation_safety_factor=float(sampling["allocation_safety_factor"]),
        minimum_pilot_nonzero=int(sampling["minimum_pilot_nonzero"]),
        maximum_pilot_samples=int(sampling["maximum_pilot_samples"]),
    )
    preparation_seconds = time.perf_counter() - preparation_started
    uncapped_counts = tuple(item.final_count for item in prepared.allocations)
    allocation_capped = False
    if method == "raw_defensive":
        prepared, uncapped_counts, allocation_capped = cap_raw_allocation(
            prepared,
            int(config["resource_limits"]["raw_max_final_samples_per_level"]),
        )
    result = execute_mlmc(prepared, sampler)
    if not result.complete:
        raise AssertionError("uninterrupted M7 method execution must complete")
    target_attained = bool(
        result.empirical_sampling_variance is not None
        and result.empirical_sampling_variance <= rmse**2
    )
    return {
        "estimate": result.estimate,
        "empirical_sampling_variance": result.empirical_sampling_variance,
        "design_sampling_variance": result.design_sampling_variance,
        "standard_error": result.standard_error,
        "confidence_interval_95": result.confidence_interval_95,
        "target_attained": target_attained,
        "allocation_capped": allocation_capped,
        "resource_censored": allocation_capped and not target_attained,
        "total_work_units": result.work.total_work_units,
        "total_wall_seconds": result.work.total_wall_seconds,
        "process_cpu_seconds": time.process_time() - process_cpu_started,
        "preparation_orchestration_seconds": preparation_seconds,
        "pilot": [asdict(item) for item in result.pilot],
        "uncapped_final_counts": list(uncapped_counts),
        "allocations": [asdict(item) for item in result.allocations],
        "levels": [asdict(item) for item in result.levels],
        "seed_ledger_sha256": result.seed_ledger_hash,
    }


def _write_progress(
    path: Path,
    *,
    config_hash: str,
    cells: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    _atomic_json(
        path,
        {
            "schema": "npi.g11.m7-progress.v1",
            "config_sha256": config_hash,
            "cells": cells,
            "failures": failures,
        },
    )


def _cluster_lower_bound(cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_replicate: dict[int, list[float]] = {}
    for cell in cells:
        methods = cell.get("methods", {})
        if (
            methods.get("raw_defensive", {}).get("target_attained") is True
            and methods.get("dcs_mgi", {}).get("target_attained") is True
        ):
            by_replicate.setdefault(int(cell["replicate"]), []).append(
                float(cell["matched_work_ratio_raw_over_dcs"])
            )
    cluster_ratios = [
        geometric_mean(ratios)
        for _, ratios in sorted(by_replicate.items())
        if ratios
    ]
    if len(cluster_ratios) < 2:
        return {
            "cluster_count": len(cluster_ratios),
            "cluster_geometric_ratios": cluster_ratios,
            "one_sided_95_lower_bound": None,
        }
    logs = torch.log(torch.tensor(cluster_ratios, dtype=torch.float64))
    mean = float(torch.mean(logs))
    standard_error = float(torch.std(logs, correction=1)) / math.sqrt(len(cluster_ratios))
    critical = float(scipy.stats.t.ppf(0.95, len(cluster_ratios) - 1))
    return {
        "cluster_count": len(cluster_ratios),
        "cluster_geometric_ratios": cluster_ratios,
        "one_sided_95_lower_bound": math.exp(mean - critical * standard_error),
    }


def summarize(
    config: dict[str, Any], cells: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    complete_cells = [cell for cell in cells if set(cell.get("methods", {})) == set(METHODS)]
    matched = [
        cell
        for cell in complete_cells
        if cell["methods"]["raw_defensive"]["target_attained"]
        and cell["methods"]["dcs_mgi"]["target_attained"]
    ]
    censored = [
        cell
        for cell in complete_cells
        if cell["methods"]["dcs_mgi"]["target_attained"]
        and cell["methods"]["raw_defensive"]["resource_censored"]
    ]
    dcs_fraction = (
        sum(cell["methods"]["dcs_mgi"]["target_attained"] for cell in complete_cells)
        / len(complete_cells)
        if complete_cells
        else 0.0
    )
    raw_fraction = (
        sum(cell["methods"]["raw_defensive"]["target_attained"] for cell in complete_cells)
        / len(complete_cells)
        if complete_cells
        else 0.0
    )
    matched_ratio = (
        geometric_mean(
            float(cell["matched_work_ratio_raw_over_dcs"]) for cell in matched
        )
        if matched
        else None
    )
    censored_lower_bound = (
        geometric_mean(
            float(cell["censored_work_ratio_lower_bound"]) for cell in censored
        )
        if censored
        else None
    )
    cluster = _cluster_lower_bound(complete_cells)
    expected = expected_cell_count(config)
    limits = config["gates"]
    gates = {
        "protocol_complete": len(complete_cells) == expected,
        "no_unexpected_failures": not failures,
        "dcs_target_attainment_fraction": dcs_fraction,
        "dcs_target_attainment_at_least_threshold": dcs_fraction
        >= float(limits["minimum_dcs_target_attainment_fraction"]),
        "raw_target_attainment_fraction": raw_fraction,
        "matched_target_cell_count": len(matched),
        "matched_target_cells_at_least_minimum": len(matched)
        >= int(limits["minimum_matched_target_cells"]),
        "resource_censored_cell_count": len(censored),
        "matched_geometric_work_ratio": matched_ratio,
        "matched_geometric_work_ratio_above_threshold": matched_ratio is not None
        and matched_ratio > float(limits["minimum_geometric_work_ratio"]),
        "seed_cluster_count": cluster["cluster_count"],
        "seed_clusters_at_least_minimum": cluster["cluster_count"]
        >= int(limits["minimum_seed_clusters_for_uncertainty"]),
        "one_sided_95_cluster_lower_bound": cluster["one_sided_95_lower_bound"],
        "one_sided_95_cluster_lower_bound_above_one": cluster[
            "one_sided_95_lower_bound"
        ]
        is not None
        and cluster["one_sided_95_lower_bound"] > 1.0,
    }
    gates["performance_headline_passed"] = bool(
        gates["protocol_complete"]
        and gates["no_unexpected_failures"]
        and gates["dcs_target_attainment_at_least_threshold"]
        and gates["matched_target_cells_at_least_minimum"]
        and gates["matched_geometric_work_ratio_above_threshold"]
        and gates["seed_clusters_at_least_minimum"]
        and gates["one_sided_95_cluster_lower_bound_above_one"]
    )
    summary = {
        "expected_cell_count": expected,
        "complete_cell_count": len(complete_cells),
        "matched_cell_count": len(matched),
        "resource_censored_cell_count": len(censored),
        "dcs_target_attainment_fraction": dcs_fraction,
        "raw_target_attainment_fraction": raw_fraction,
        "matched_geometric_work_ratio": matched_ratio,
        "censored_geometric_work_ratio_lower_bound": censored_lower_bound,
        "cluster_uncertainty": cluster,
    }
    return summary, gates


def preflight(config_path: Path) -> dict[str, Any]:
    """Validate a protocol without constructing or consuming any random seed."""

    config, config_hash = _load_config(config_path)
    source_manifest = _verify_source(config)
    target_threads = int(config["resource_limits"]["torch_threads"])
    if torch.get_num_threads() != target_threads:
        raise ValueError(
            f"expected torch_threads={target_threads}, got {torch.get_num_threads()}"
        )
    contexts, inputs = _load_regimes(config)
    namespace_payload = {
        "protocol_id": config["protocol_id"],
        "roles": ["pilot", "final"],
        "streams": ["proposal", "labels"],
        "regimes": [context["name"] for context in contexts],
        "task_ids": {
            context["name"]: [task["id"] for task in context["tasks"]]
            for context in contexts
        },
        "repetitions": int(config["sampling"]["repetitions"]),
    }
    namespace_sha256 = hashlib.sha256(
        json.dumps(
            namespace_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": "npi.g11.m7-preflight.v1",
        "protocol_id": config["protocol_id"],
        "run_class": config["run_class"],
        "frozen": bool(config["frozen"]),
        "config_sha256": config_hash,
        "core_source_commit": config["core_source_commit"],
        "core_source_manifest": source_manifest,
        "input_artifacts": inputs,
        "expected_cell_count": expected_cell_count(config),
        "seed_namespace": namespace_payload,
        "seed_namespace_sha256": namespace_sha256,
        "random_seeds_allocated": 0,
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def run(config_path: Path, *, progress_path: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load_config(config_path)
    source_manifest = _verify_source(config)
    target_threads = int(config["resource_limits"]["torch_threads"])
    if torch.get_num_threads() != target_threads:
        raise ValueError(
            f"expected torch_threads={target_threads}, got {torch.get_num_threads()}"
        )
    contexts, inputs = _load_regimes(config)
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema") != "npi.g11.m7-progress.v1"
            or progress.get("config_sha256") != config_hash
        ):
            raise ValueError("M7 progress file does not match the config")
        cells = list(progress["cells"])
        failures = list(progress["failures"])

    cell_by_key = {
        (str(cell["regime"]), str(cell["task"]), int(cell["replicate"])): cell
        for cell in cells
    }
    budget_cpu_hours = float(config["resource_limits"]["total_cpu_hours"])
    budget_exhausted = False
    for replicate in range(int(config["sampling"]["repetitions"])):
        for context in contexts:
            for task_item in context["tasks"]:
                key = (str(context["name"]), str(task_item["id"]), replicate)
                cell = cell_by_key.get(key)
                if cell is None:
                    cell = {
                        "regime": context["name"],
                        "task": task_item["id"],
                        "target_probability": task_item["target_probability"],
                        "rmse_target": task_item["target_probability"]
                        * float(config["sampling"]["relative_rmse_target"]),
                        "replicate": replicate,
                        "methods": {},
                    }
                    cells.append(cell)
                    cell_by_key[key] = cell
                for method in METHODS:
                    if method in cell["methods"]:
                        continue
                    try:
                        cell["methods"][method] = _method_result(
                            config=config,
                            context=context,
                            task_item=task_item,
                            replicate=replicate,
                            method=method,
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "regime": context["name"],
                                "task": task_item["id"],
                                "replicate": replicate,
                                "method": method,
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                        if progress_path is not None:
                            _write_progress(
                                progress_path,
                                config_hash=config_hash,
                                cells=cells,
                                failures=failures,
                            )
                        raise
                    if progress_path is not None:
                        _write_progress(
                            progress_path,
                            config_hash=config_hash,
                            cells=cells,
                            failures=failures,
                        )
                raw = cell["methods"]["raw_defensive"]
                dcs = cell["methods"]["dcs_mgi"]
                allocated_ratio = (
                    float(raw["total_work_units"]) / float(dcs["total_work_units"])
                )
                cell["allocated_work_ratio_raw_over_dcs"] = allocated_ratio
                cell["matched_work_ratio_raw_over_dcs"] = (
                    allocated_ratio
                    if raw["target_attained"] and dcs["target_attained"]
                    else None
                )
                cell["censored_work_ratio_lower_bound"] = (
                    allocated_ratio
                    if raw["resource_censored"] and dcs["target_attained"]
                    else None
                )
                cell["wall_ratio_raw_over_dcs"] = (
                    float(raw["total_wall_seconds"]) / float(dcs["total_wall_seconds"])
                )
                cell["paired_estimate_difference"] = float(dcs["estimate"]) - float(
                    raw["estimate"]
                )
                process_cpu = math.fsum(
                    float(method_result["process_cpu_seconds"])
                    for completed in cells
                    for method_result in completed.get("methods", {}).values()
                )
                if process_cpu / 3600.0 >= budget_cpu_hours:
                    budget_exhausted = True
                    break
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    summary, gates = summarize(config, cells, failures)
    seed_manifest = [
        {
            "regime": cell["regime"],
            "task": cell["task"],
            "replicate": cell["replicate"],
            "method": method,
            "seed_ledger_sha256": result["seed_ledger_sha256"],
        }
        for cell in cells
        for method, result in sorted(cell.get("methods", {}).items())
    ]
    aggregate_seed_hash = hashlib.sha256(
        json.dumps(
            seed_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    method_wall_seconds = math.fsum(
        float(method["total_wall_seconds"])
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    process_cpu_seconds = math.fsum(
        float(method["process_cpu_seconds"])
        for cell in cells
        for method in cell.get("methods", {}).values()
    )
    return {
        "schema": "npi.g11.m7-confirmatory.v1",
        "protocol_id": config["protocol_id"],
        "run_class": config["run_class"],
        "frozen": bool(config["frozen"]),
        "config_sha256": config_hash,
        "core_source_commit": config["core_source_commit"],
        "core_source_manifest": source_manifest,
        "input_artifacts": inputs,
        "estimand": "fixed finest finite-grid probability",
        "continuous_time_claim": False,
        "self_normalized": False,
        "budget_exhausted": budget_exhausted,
        "resource_limits": config["resource_limits"],
        "summary": summary,
        "cells": cells,
        "failures": failures,
        "gates": gates,
        "seed_ledger_sha256": aggregate_seed_hash,
        "seed_manifest_entries": len(seed_manifest),
        "work_ledger": {
            "measured_method_wall_seconds": method_wall_seconds,
            "measured_process_cpu_seconds": process_cpu_seconds,
            "measured_process_cpu_hours": process_cpu_seconds / 3600.0,
            "current_process_orchestration_seconds": time.perf_counter() - started,
            "resumable_cell_progress_used": progress_path is not None,
        },
        "environment": runtime_provenance(dtype="torch.float64"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        if arguments.progress is not None:
            raise ValueError("preflight does not accept a progress path")
        result = preflight(arguments.config)
    else:
        progress = arguments.progress or arguments.output.with_suffix(
            arguments.output.suffix + ".progress.json"
        )
        result = run(arguments.config, progress_path=progress)
    _atomic_json(arguments.output, result)
    print(
        json.dumps(
            result.get(
                "gates",
                {
                    "expected_cell_count": result["expected_cell_count"],
                    "random_seeds_allocated": result["random_seeds_allocated"],
                },
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
