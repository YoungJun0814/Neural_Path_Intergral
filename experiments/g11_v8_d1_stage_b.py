"""Run D1 Stage B on all 24 primary rough-volatility cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from experiments.g11_v8_d1_p7_falsification_stage_a import (
    _aggregate_stage_a,
    _external_record,
    _paired_record,
)
from src.path_integral.provenance import runtime_provenance, source_provenance

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v8-d1-stage-b.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected D1 Stage B config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate_bindings_and_source(config: dict[str, Any]) -> None:
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid Stage B binding: {name}")
        path = ROOT / str(binding["path"])
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise ValueError(f"Stage B binding mismatch: {name}")
    source_commit = str(config["source_parent_commit"])
    current = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", source_commit, current),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("Stage B source commit is not an ancestor")
    paths = (
        "experiments/g11_v8_d1_stage_b.py",
        "experiments/g11_v8_d1_p7_falsification_stage_a.py",
        "src/path_integral/dcs_benchmark.py",
        "src/path_integral/benchmark_executor.py",
        "src/path_integral/baselines/cem.py",
    )
    if subprocess.run(
        ("git", "diff", "--quiet", source_commit, "--", *paths),
        cwd=ROOT,
        check=False,
    ).returncode or subprocess.check_output(
        ("git", "status", "--porcelain", "--", *paths), cwd=ROOT, text=True
    ).strip():
        raise ValueError("Stage B implementation differs from its bound source")


def _bound_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    calibration = json.loads(
        (ROOT / config["bindings"]["threshold_calibration"]["path"]).read_text()
    )
    reference = json.loads(
        (ROOT / config["bindings"]["reference"]["path"]).read_text()
    )
    references = {
        str(cell["cell_id"]): (float(cell["estimate"]), float(cell["standard_error"]))
        for cell in reference["cells"]
        if cell["method"] == "dcs_reference"
    }
    requested = list(config["cell_ids"])
    calibrated = {str(cell["cell_id"]): cell for cell in calibration["cells"]}
    if len(requested) != 24 or len(set(requested)) != 24 or set(requested) != set(references):
        raise ValueError("Stage B must bind the complete 24-cell reference roster")
    cells: list[dict[str, Any]] = []
    for cell_id in requested:
        cell = calibrated[cell_id]
        estimate, standard_error = references[cell_id]
        cells.append(
            {
                "cell_id": cell_id,
                "task": str(cell["task"]),
                "hurst": float(cell["hurst"]),
                "nominal_probability": float(cell["nominal_probability"]),
                "threshold": float(cell["calibrated_threshold"]),
                "reference_estimate": estimate,
                "reference_standard_error": standard_error,
            }
        )
    return cells


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    _validate_bindings_and_source(config)
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("Stage B namespace was not outcome-blind at freeze")
    stage_a = json.loads((ROOT / config["bindings"]["stage_a_audit"]["path"]).read_text())
    bank_audit = json.loads(
        (ROOT / config["bindings"]["dcs_proposal_bank_audit"]["path"]).read_text()
    )
    if stage_a.get("passed") is not True or bank_audit.get("passed") is not True:
        raise ValueError("Stage A and DCS proposal-bank audits must pass")
    if config["external_methods"]["primary"] != ["defensive_cem", "smoothing_rqmc"]:
        raise ValueError("Stage B primary comparator roster changed")
    if int(config["clusters"]) < 2 or len(config["budgets"]) != 1:
        raise ValueError("Stage B requires at least two clusters and one operating budget")
    result = dict(config)
    result["cells"] = _bound_cells(config)
    return result


def _add_amortized_work(config: dict[str, Any], paired: list[dict[str, Any]]) -> None:
    bank = json.loads((ROOT / config["bindings"]["dcs_proposal_bank"]["path"]).read_text())
    entry_cost = {
        entry["cell_id"]: entry["training_cost"] for entry in bank["entries"]
    }
    query_counts = [int(value) for value in bank["amortization_query_counts"]]
    for record in paired:
        training = entry_cost[record["cell_id"]]
        record["amortized_total_work"] = {
            str(count): {
                method: float(record[method]["cost"]["algorithmic_work_units"])
                + float(training["algorithmic_work_units"]) / count
                for method in ("raw", "dcs")
            }
            for count in query_counts
        }


def _aggregate(config: dict[str, Any], paired: list[dict[str, Any]], external: list[dict[str, Any]]):
    aggregate = _aggregate_stage_a(config, paired, external)
    accuracy_limit = float(config["gate"]["dcs_accuracy_combined_z"])
    dcs_accuracy_maximum = max(float(record["dcs"]["combined_reference_z"]) for record in paired)
    dcs_accuracy_pass = dcs_accuracy_maximum <= accuracy_limit
    cost_closed = all(record["proposal_training_cost"] is not None for record in paired)
    stage_b_complete = (
        bool(aggregate["stage_b_authorized"]) and dcs_accuracy_pass and cost_closed
    )
    aggregate.update(
        {
            "dcs_accuracy_maximum_combined_z": dcs_accuracy_maximum,
            "dcs_accuracy_pass": dcs_accuracy_pass,
            "dcs_proposal_training_cost_closed": cost_closed,
            "stage_b_complete": stage_b_complete,
            "p8_blockers": [
                *([] if aggregate["exactness_pass"] else ["exactness_failure"]),
                *([] if aggregate["mechanism_pass"] else ["mechanism_gate_failure"]),
                *([] if aggregate["primary_accuracy_pass"] else ["primary_accuracy_failure"]),
                *([] if dcs_accuracy_pass else ["dcs_accuracy_failure"]),
                *([] if cost_closed else ["dcs_proposal_training_cost_not_closed"]),
                *(
                    []
                    if aggregate["primary_resource_censoring_count"] == 0
                    else ["primary_resource_censoring"]
                ),
                "stage_c_not_complete",
                "t1_theorem_and_novelty_not_closed",
            ],
        }
    )
    return aggregate


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    config = _validate(config)
    paired: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    cursor = int(config["base_seed"])

    def allocate(count: int) -> tuple[int, ...]:
        nonlocal cursor
        values = tuple(range(cursor, cursor + count))
        cursor += count
        if used_seeds & set(values):
            raise AssertionError("Stage B seed collision")
        used_seeds.update(values)
        return values

    budget = config["budgets"][0]
    primary = list(config["external_methods"]["primary"])
    secondary = list(config["external_methods"]["secondary"])
    for cell in config["cells"]:
        for cluster in range(int(config["clusters"])):
            paired.append(
                _paired_record(
                    config,
                    cell,
                    budget,
                    cluster,
                    cast(tuple[int, int], allocate(2)),
                )
            )
            for method in primary + secondary:
                external.append(
                    _external_record(
                        config,
                        cell,
                        budget,
                        method,
                        cluster,
                        cast(tuple[int, int, int, int], allocate(4)),
                    )
                )
    _add_amortized_work(config, paired)
    aggregate = _aggregate(config, paired, external)
    seed_payload = json.dumps(sorted(used_seeds), separators=(",", ":")).encode()
    return {
        "schema": "npi.g11.v8-d1-stage-b-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "seed_count": len(used_seeds),
        "seed_set_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "paired_records": paired,
        "external_records": external,
        "aggregate": aggregate,
        "decision": {
            "stage_b_complete": aggregate["stage_b_complete"],
            "stage_c_authorized": aggregate["stage_b_complete"],
            "p8_qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
