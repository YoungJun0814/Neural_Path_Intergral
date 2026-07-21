"""Independent audit of the frozen G11 V4 crossover qualification artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.seed_ledger import SeedKey, SeedLedger


def _strict_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    payload = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("V4 result root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported V4 crossover config")
    return config, hashlib.sha256(raw).hexdigest()


def _normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _close(left: Any, right: Any, *, rel_tol: float = 2e-10) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=1e-15)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, rel_tol=rel_tol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close(left[key], right[key], rel_tol=rel_tol) for key in left
        )
    return left == right


def _expected_cells(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for regime in config["regimes"]:
        for selection in regime["selections"]:
            probability = float(selection["target_probability"])
            task = f"{selection['task']}_{probability:.0e}"
            key = (str(regime["name"]), task)
            expected[key] = {
                "changed_parameter": str(regime["changed_parameter"]),
                "target_probability": probability,
                "cem_enabled": bool(selection.get("cem_enabled", False)),
            }
    return expected


def _reconstruct_seed_ledger(
    config: dict[str, Any], cell: dict[str, Any], *, cem_enabled: bool
) -> SeedLedger:
    ledger = SeedLedger()
    protocol = str(config["protocol_id"])
    regime = str(cell["regime"])
    task = str(cell["task"])
    finest = int(config["hierarchy"]["finest_level"])
    if cem_enabled:
        ledger.allocate(
            SeedKey(protocol, "training", regime, task, finest, 0, "proposal")
        )
    for replicate in range(int(config["sampling"]["repetitions"])):
        for level in range(finest + 1):
            for stream in ("proposal", "labels"):
                ledger.allocate(
                    SeedKey(
                        protocol,
                        "profile",
                        regime,
                        f"{task}:dcs_single",
                        level,
                        replicate,
                        stream,
                    )
                )
        for level in range(1, finest + 1):
            for stream in ("proposal", "labels"):
                ledger.allocate(
                    SeedKey(
                        protocol,
                        "profile",
                        regime,
                        f"{task}:dcs_correction",
                        level,
                        replicate,
                        stream,
                    )
                )
        for stream in ("proposal", "labels"):
            ledger.allocate(
                SeedKey(
                    protocol,
                    "profile",
                    regime,
                    f"{task}:crude_single",
                    finest,
                    replicate,
                    stream,
                )
            )
        if cem_enabled:
            for stream in ("proposal", "labels"):
                ledger.allocate(
                    SeedKey(
                        protocol,
                        "profile",
                        regime,
                        f"{task}:cem_slis",
                        finest,
                        replicate,
                        stream,
                    )
                )
    return ledger


def _profile_failures(
    profile: dict[str, Any], *, expected_count: int, identity: str
) -> list[str]:
    failures: list[str] = []
    if int(profile.get("count", -1)) != expected_count:
        failures.append(f"{identity}: wrong profile count")
    variance = profile.get("variance")
    if not isinstance(variance, (int, float)) or float(variance) < 0.0:
        failures.append(f"{identity}: invalid variance")
    for field in (
        "mean",
        "standard_error",
        "second_moment",
        "cost_per_sample",
        "wall_seconds",
        "work_units",
    ):
        value = profile.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{identity}: invalid {field}")
    if isinstance(variance, (int, float)) and isinstance(
        profile.get("standard_error"), (int, float)
    ):
        expected_se = math.sqrt(float(variance) / expected_count)
        if not _close(profile["standard_error"], expected_se):
            failures.append(f"{identity}: standard error mismatch")
    return failures


def _work_coefficients(
    single: list[dict[str, Any]], corrections: list[dict[str, Any]]
) -> list[float]:
    finest = len(single) - 1
    coefficients: list[float] = []
    for start in range(finest + 1):
        if start == finest:
            coefficients.append(
                float(single[start]["variance"])
                * float(single[start]["cost_per_sample"])
            )
            continue
        root = math.sqrt(
            float(single[start]["variance"])
            * float(single[start]["cost_per_sample"])
        ) + math.fsum(
            math.sqrt(float(item["variance"]) * float(item["cost_per_sample"]))
            for item in corrections[start:]
        )
        coefficients.append(root**2)
    return coefficients


def _decision(
    *,
    single: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    crude: dict[str, Any],
    cem_profile: dict[str, Any] | None,
    cem_training_work: float,
    variance_target: float,
) -> dict[str, Any]:
    coefficients = _work_coefficients(single, corrections)
    profile_work = math.fsum(
        float(item["work_units"]) for item in (*single, *corrections)
    )
    dcs_totals = [profile_work + value / variance_target for value in coefficients]
    optimal_start = min(
        range(len(dcs_totals)), key=lambda index: (dcs_totals[index], index)
    )
    candidates = {"dcs_mlmc_or_slis": dcs_totals[optimal_start]}
    crude_total = None
    if float(crude["variance"]) > 0.0:
        crude_total = float(crude["work_units"]) + (
            float(crude["variance"])
            * float(crude["cost_per_sample"])
            / variance_target
        )
        candidates["crude_single"] = crude_total
    cem_total = None
    if cem_profile is not None and float(cem_profile["variance"]) > 0.0:
        cem_total = (
            cem_training_work
            + float(cem_profile["work_units"])
            + float(cem_profile["variance"])
            * float(cem_profile["cost_per_sample"])
            / variance_target
        )
        candidates["cem_slis"] = cem_total
    selected = min(candidates, key=lambda method: (candidates[method], method))
    return {
        "coefficients": coefficients,
        "profile_work": profile_work,
        "dcs_totals": dcs_totals,
        "optimal_start": optimal_start,
        "optimal_total": dcs_totals[optimal_start],
        "crude_total": crude_total,
        "cem_total": cem_total,
        "selected_method": selected,
        "selected_total": candidates[selected],
    }


def _cell_failures(
    config: dict[str, Any], cell: dict[str, Any], contract: dict[str, Any]
) -> list[str]:
    identity = f"{cell.get('regime')}/{cell.get('task')}"
    failures: list[str] = []
    if cell.get("changed_parameter") != contract["changed_parameter"]:
        failures.append(f"{identity}: changed-parameter label mismatch")
    if not _close(cell.get("target_probability"), contract["target_probability"]):
        failures.append(f"{identity}: target probability mismatch")
    cem_enabled = bool(contract["cem_enabled"])
    if (cell.get("cem") is not None) is not cem_enabled:
        failures.append(f"{identity}: CEM presence mismatch")
    if cem_enabled:
        cem = cell["cem"]
        training_work = float(cem["training_work_units"])
        if not cem.get("history") or not math.isfinite(training_work) or training_work <= 0.0:
            failures.append(f"{identity}: invalid CEM training record")
    else:
        training_work = 0.0

    ledger = _reconstruct_seed_ledger(config, cell, cem_enabled=cem_enabled)
    if cell.get("seed_count") != len(ledger):
        failures.append(f"{identity}: seed count mismatch")
    if cell.get("seed_ledger_sha256") != ledger.sha256:
        failures.append(f"{identity}: seed-ledger hash mismatch")

    runs = cell.get("runs")
    repetitions = int(config["sampling"]["repetitions"])
    paths = int(config["sampling"]["paths_per_profile"])
    finest = int(config["hierarchy"]["finest_level"])
    if not isinstance(runs, list) or len(runs) != repetitions:
        return [*failures, f"{identity}: run count mismatch"]
    if {int(run.get("replicate", -1)) for run in runs} != set(range(repetitions)):
        failures.append(f"{identity}: replicate identities mismatch")
    for run in runs:
        replicate = int(run["replicate"])
        single = run.get("dcs_single_levels")
        corrections = run.get("dcs_corrections")
        if not isinstance(single, list) or len(single) != finest + 1:
            failures.append(f"{identity}/rep={replicate}: single-level profile mismatch")
            continue
        if not isinstance(corrections, list) or len(corrections) != finest:
            failures.append(f"{identity}/rep={replicate}: correction profile mismatch")
            continue
        for level, profile in enumerate(single):
            failures.extend(
                _profile_failures(
                    profile,
                    expected_count=paths,
                    identity=f"{identity}/rep={replicate}/single={level}",
                )
            )
        for level, profile in enumerate(corrections, start=1):
            failures.extend(
                _profile_failures(
                    profile,
                    expected_count=paths,
                    identity=f"{identity}/rep={replicate}/correction={level}",
                )
            )
        crude = run["crude_single"]
        failures.extend(
            _profile_failures(
                crude,
                expected_count=paths,
                identity=f"{identity}/rep={replicate}/crude",
            )
        )
        cem_profile = run.get("cem_slis")
        if (cem_profile is not None) is not cem_enabled:
            failures.append(f"{identity}/rep={replicate}: CEM profile presence mismatch")
        elif cem_profile is not None:
            failures.extend(
                _profile_failures(
                    cem_profile,
                    expected_count=paths,
                    identity=f"{identity}/rep={replicate}/cem",
                )
            )
        expected_estimates = [
            float(single[start]["mean"])
            + math.fsum(float(item["mean"]) for item in corrections[start:])
            for start in range(finest + 1)
        ]
        expected_standard_errors = [
            math.sqrt(
                float(single[start]["standard_error"]) ** 2
                + math.fsum(
                    float(item["standard_error"]) ** 2
                    for item in corrections[start:]
                )
            )
            for start in range(finest + 1)
        ]
        if not _close(run.get("dcs_telescoping_estimates_by_start"), expected_estimates):
            failures.append(f"{identity}/rep={replicate}: telescoping estimate mismatch")
        if not _close(
            run.get("dcs_telescoping_standard_errors_by_start"),
            expected_standard_errors,
        ):
            failures.append(f"{identity}/rep={replicate}: telescoping SE mismatch")
        for rmse_value in config["relative_rmse_targets"]:
            rmse = float(rmse_value)
            key = f"{rmse:.2f}"
            variance_target = (float(cell["target_probability"]) * rmse) ** 2
            expected = _decision(
                single=single,
                corrections=corrections,
                crude=crude,
                cem_profile=cem_profile,
                cem_training_work=training_work,
                variance_target=variance_target,
            )
            observed = run["decisions"].get(key)
            if not isinstance(observed, dict):
                failures.append(f"{identity}/rep={replicate}: missing RMSE {key}")
                continue
            dcs = observed["dcs"]
            comparisons = {
                "variance_target": variance_target,
                "dcs.multilevel_work_coefficients": expected["coefficients"],
                "dcs.total_work_by_start_level": expected["dcs_totals"],
                "dcs.optimal_start_level": expected["optimal_start"],
                "dcs.optimal_total_work": expected["optimal_total"],
                "crude_single_total_work": expected["crude_total"],
                "cem_slis_total_work": expected["cem_total"],
                "selected_method": expected["selected_method"],
                "selected_total_work": expected["selected_total"],
            }
            actual = {
                "variance_target": observed.get("variance_target"),
                "dcs.multilevel_work_coefficients": dcs["online"][
                    "multilevel_work_coefficients"
                ],
                "dcs.total_work_by_start_level": dcs["total_work_by_start_level"],
                "dcs.optimal_start_level": dcs["optimal_start_level"],
                "dcs.optimal_total_work": dcs["optimal_total_work"],
                "crude_single_total_work": observed.get("crude_single_total_work"),
                "cem_slis_total_work": observed.get("cem_slis_total_work"),
                "selected_method": observed.get("selected_method"),
                "selected_total_work": observed.get("selected_total_work"),
            }
            if not _close(actual, comparisons):
                failures.append(
                    f"{identity}/rep={replicate}: independent decision mismatch at {key}"
                )
    return failures


def _summary(config: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts: dict[str, dict[str, int]] = {}
    start_counts: dict[str, dict[str, int]] = {}
    reference_z: list[float] = []
    for rmse_value in config["relative_rmse_targets"]:
        key = f"{float(rmse_value):.2f}"
        decision_counts[key] = {}
        start_counts[key] = {}
        for cell in cells:
            for run in cell["runs"]:
                decision = run["decisions"][key]
                method = str(decision["selected_method"])
                decision_counts[key][method] = decision_counts[key].get(method, 0) + 1
                start = str(decision["dcs"]["optimal_start_level"])
                start_counts[key][start] = start_counts[key].get(start, 0) + 1
    for cell in cells:
        for run in cell["runs"]:
            estimate = float(run["dcs_telescoping_estimates_by_start"][0])
            standard_error = float(run["dcs_telescoping_standard_errors_by_start"][0])
            combined = math.sqrt(standard_error**2 + float(cell["reference_standard_error"]) ** 2)
            if combined > 0.0:
                reference_z.append((estimate - float(cell["reference_estimate"])) / combined)
    absolute_z = [abs(value) for value in reference_z]
    return {
        "cell_count": len(cells),
        "run_count": sum(len(cell["runs"]) for cell in cells),
        "decision_counts": decision_counts,
        "dcs_optimal_start_level_counts": start_counts,
        "reference_comparison_count": len(reference_z),
        "reference_within_4_se_fraction": statistics.fmean(
            float(value <= 4.0) for value in absolute_z
        ),
        "reference_median_absolute_z": statistics.median(absolute_z),
        "reference_maximum_absolute_z": max(absolute_z),
    }


def _diagnostics(config: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, Any] = {}
    for regime in sorted({str(cell["regime"]) for cell in cells}):
        selected = [cell for cell in cells if cell["regime"] == regime]
        starts = [
            int(run["decisions"]["0.20"]["dcs"]["optimal_start_level"])
            for cell in selected
            for run in cell["runs"]
        ]
        by_regime[regime] = {
            "run_count": len(starts),
            "dcs_slis_count": sum(
                value == int(config["hierarchy"]["finest_level"]) for value in starts
            ),
            "full_mlmc_count": sum(value == 0 for value in starts),
            "start_level_counts": {
                str(level): starts.count(level) for level in sorted(set(starts))
            },
        }
    cem_ratios: dict[str, Any] = {}
    for cell in cells:
        if cell.get("cem") is None:
            continue
        ratios: dict[str, list[float]] = {
            f"{float(value):.2f}": [] for value in config["relative_rmse_targets"]
        }
        for run in cell["runs"]:
            for key in ratios:
                decision = run["decisions"][key]
                cem = decision["cem_slis_total_work"]
                if cem is not None:
                    ratios[key].append(float(cem) / float(decision["dcs"]["optimal_total_work"]))
        cem_ratios[str(cell["task"])] = {
            key: statistics.geometric_mean(values) if values else None
            for key, values in ratios.items()
        }
    return {"by_regime_at_rmse_020": by_regime, "cem_over_best_dcs": cem_ratios}


def run(config_path: Path, result_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config, config_hash = _load_config(config_path)
    result, result_hash = _strict_json(result_path)
    failures: list[str] = []
    expected = _expected_cells(config)
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise ValueError("V4 result has no cells")
    observed = [(str(cell.get("regime")), str(cell.get("task"))) for cell in cells]
    if len(observed) != len(set(observed)):
        failures.append("duplicate cell identity")
    if set(observed) != set(expected):
        failures.append("cell matrix differs from the frozen config")
    for cell in cells:
        key = (str(cell.get("regime")), str(cell.get("task")))
        if key in expected:
            failures.extend(_cell_failures(config, cell, expected[key]))

    summary = _summary(config, cells) if set(observed) == set(expected) else {}
    if not _close(result.get("summary"), summary):
        failures.append("serialized summary differs from independent reconstruction")
    expected_gates = {
        "complete_cell_matrix": len(cells) == len(expected),
        "minimum_repetitions": int(config["sampling"]["repetitions"]) >= 5,
        "all_reference_comparisons_within_fraction": summary.get(
            "reference_within_4_se_fraction", 0.0
        )
        >= float(config["gates"]["minimum_reference_within_4_se_fraction"]),
        "multiple_rmse_targets": len(config["relative_rmse_targets"]) >= 3,
        "parameter_separated": {
            contract["changed_parameter"] for contract in expected.values()
        }
        >= {"base", "H", "eta", "rho"},
    }
    expected_gates["qualification_passed"] = all(expected_gates.values())
    if result.get("gates") != expected_gates:
        failures.append("serialized qualification gates differ from reconstruction")

    fixed = {
        "schema": "npi.g11.v4-crossover-qualification.v1",
        "protocol_id": config["protocol_id"],
        "run_class": "qualification",
        "config_sha256": config_hash,
        "estimand": "fixed finest finite-grid probability",
        "continuous_time_claim": False,
        "dirty_worktree": False,
    }
    for field, value in fixed.items():
        if result.get(field) != value:
            failures.append(f"invalid fixed field {field}")
    repository = config_path.resolve().parents[1]
    tag_commit = subprocess.check_output(
        ("git", "rev-list", "-n", "1", str(config["required_git_tag"])),
        cwd=repository,
        text=True,
    ).strip()
    if result.get("source_commit") != tag_commit:
        failures.append("result source commit differs from the frozen tag")
    if result.get("environment", {}).get("dtype") != "torch.float64":
        failures.append("result dtype is not torch.float64")
    expected_inputs = [
        {
            "path": str(Path(declaration[kind]["path"])),
            "sha256": str(declaration[kind]["sha256"]),
        }
        for declaration in config["regimes"]
        for kind in ("calibration_config", "calibration_result")
    ]
    if result.get("input_artifacts") != expected_inputs:
        failures.append("input artifact manifest mismatch")
    for item in expected_inputs:
        if _normalized_sha256(Path(item["path"])) != item["sha256"]:
            failures.append(f"audited input hash mismatch: {item['path']}")

    return {
        "schema": "npi.g11.v4-crossover-audit.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "result": str(result_path),
        "result_sha256": result_hash,
        "integrity_failures": failures,
        "integrity_passed": not failures,
        "qualification_gate_passed": expected_gates["qualification_passed"],
        "independent_summary": summary,
        "independent_gates": expected_gates,
        "diagnostics": _diagnostics(config, cells),
        "limitations": [
            "fixed-size profiles predict continuous-allocation work but do not execute the requested RMSE",
            "CEM is trained and compared only on four base cells",
            "operation work is an FFT proxy and must be accompanied by hardware wall-time evidence",
            "the estimand is the fixed 128-step grid rather than continuous monitoring",
        ],
        "work_ledger": {"audit_seconds": time.perf_counter() - started},
        "environment": runtime_provenance(dtype="not_applicable"),
        **source_provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit = run(arguments.config, arguments.result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "integrity_passed": audit["integrity_passed"],
                "qualification_gate_passed": audit["qualification_gate_passed"],
                "integrity_failures": audit["integrity_failures"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
