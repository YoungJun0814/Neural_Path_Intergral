"""Execute V9 terminal development or conditional qualification benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from experiments.g11_v8_d1_p7_falsification_stage_a import (
    _external_record,
    _paired_record,
)
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.v9_terminal_protocol import (
    aggregate_terminal_benchmark,
    attach_work_to_target,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v9-terminal-benchmark.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V9 terminal benchmark config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate_bindings_and_source(config: dict[str, Any]) -> None:
    for name, binding in config.get("bindings", {}).items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid V9 benchmark binding: {name}")
        path = ROOT / str(binding["path"])
        if not path.is_file() or _sha256(path) != str(binding["sha256"]):
            raise ValueError(f"V9 benchmark binding mismatch: {name}")
        if str(binding["path"]).replace("\\", "/").startswith("results/g11_v8"):
            raise ValueError("V8 result artifacts cannot be V9 benchmark bindings")
    source_commit = str(config["source_parent_commit"])
    current = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", source_commit, current),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("V9 benchmark source commit is not an ancestor")
    paths = (
        "experiments/g11_v9_terminal_benchmark.py",
        "experiments/g11_v8_d1_p7_falsification_stage_a.py",
        "src/path_integral/v9_terminal_protocol.py",
        "src/path_integral/dcs_benchmark.py",
        "src/path_integral/benchmark_executor.py",
        "src/path_integral/baselines/cem.py",
        "src/path_integral/baselines/conditional_rbergomi.py",
        "src/path_integral/baselines/smoothing_rqmc.py",
    )
    if subprocess.run(
        ("git", "diff", "--quiet", source_commit, "--", *paths),
        cwd=ROOT,
        check=False,
    ).returncode or subprocess.check_output(
        ("git", "status", "--porcelain", "--", *paths), cwd=ROOT, text=True
    ).strip():
        raise ValueError("V9 benchmark implementation differs from its bound source")


def _bound_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    reference = json.loads(
        (ROOT / str(config["bindings"]["reference"]["path"])).read_text(encoding="utf-8")
    )
    references = {str(cell["cell_id"]): cell for cell in reference["cells"]}
    claim = yaml.safe_load(
        (ROOT / str(config["bindings"]["claim_contract"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    claim_cells = {str(cell["cell_id"]): cell for cell in claim["cells"]}
    if set(references) != set(claim_cells):
        raise ValueError("V9 reference does not cover the complete claim roster")
    if config["stage"] == "development":
        requested = list(claim_cells)
    else:
        development = json.loads(
            (ROOT / str(config["bindings"]["development_result"]["path"])).read_text(
                encoding="utf-8"
            )
        )
        selected = {float(value) for value in development["aggregate"]["selected_hurst_groups"]}
        requested = [
            cell_id
            for cell_id, cell in claim_cells.items()
            if float(cell["hurst"]) in selected
        ]
        if not selected or sorted(selected) != sorted(map(float, config["selected_hurst_groups"])):
            raise ValueError("qualification Hurst roster differs from development selection")
    cells: list[dict[str, Any]] = []
    for cell_id in requested:
        design = claim_cells[cell_id]
        reference_cell = references[cell_id]
        cells.append(
            {
                **design,
                "reference_estimate": float(reference_cell["estimate"]),
                "reference_standard_error": float(reference_cell["standard_error"]),
            }
        )
    return cells


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    _validate_bindings_and_source(config)
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("V9 benchmark namespace was not outcome-blind at freeze")
    if config["stage"] not in {"development", "qualification"}:
        raise ValueError("V9 benchmark stage must be development or qualification")
    claim = yaml.safe_load(
        (ROOT / str(config["bindings"]["claim_contract"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    required = (
        "model",
        "relative_rmse_target",
        "amortization_query_counts",
        "primary_query_count",
    )
    if any(config[key] != claim[key] for key in required):
        raise ValueError("V9 benchmark numerical contract differs from the claim contract")
    if config["gate"] != {
        "maximum_combined_reference_z": claim["gate"]["maximum_combined_reference_z"],
        "maximum_paired_difference_z": claim["gate"]["maximum_paired_difference_z"],
        "minimum_likelihood_normalization_pass_fraction": claim["gate"][
            "minimum_likelihood_normalization_pass_fraction"
        ],
        "development": claim["gate"]["development"],
        "qualification": claim["gate"]["qualification"],
    }:
        raise ValueError("V9 benchmark gates differ from the claim contract")
    if float(config["maximum_exactness_error"]) != float(
        claim["gate"]["maximum_exactness_error"]
    ):
        raise ValueError("V9 benchmark exactness gate differs from the claim contract")
    if config["external_methods"] != {
        "primary": ["conditional_rbergomi", "smoothing_rqmc", "defensive_cem"]
    }:
        raise ValueError("V9 primary comparator roster changed")
    reference_audit = json.loads(
        (ROOT / str(config["bindings"]["reference_audit"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    bank_audit = json.loads(
        (ROOT / str(config["bindings"]["proposal_bank_audit"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if reference_audit.get("passed") is not True or bank_audit.get("passed") is not True:
        raise ValueError("V9 reference and proposal-bank audits must pass")
    if int(config["clusters"]) < 2 or len(config["budgets"]) != 1:
        raise ValueError("V9 benchmark requires >=2 clusters and one operating budget")
    if config["stage"] == "qualification":
        development_audit = json.loads(
            (ROOT / str(config["bindings"]["development_audit"]["path"])).read_text(
                encoding="utf-8"
            )
        )
        if development_audit.get("passed") is not True or development_audit.get(
            "qualification_authorized"
        ) is not True:
            raise ValueError("V9 development audit did not authorize qualification")
    result = dict(config)
    result["cells"] = _bound_cells(config)
    bank = json.loads(
        (ROOT / str(config["bindings"]["dcs_proposal_bank"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if {entry["cell_id"] for entry in bank["entries"]} != {
        cell["cell_id"] for cell in claim["cells"]
    }:
        raise ValueError("V9 proposal bank does not cover the claim roster")
    return result


def _augment_record(record: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    record["hurst"] = float(cell["hurst"])
    record["nominal_probability"] = float(cell["nominal_probability"])
    record["reference_estimate"] = float(cell["reference_estimate"])
    record["reference_standard_error"] = float(cell["reference_standard_error"])
    return record


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
            raise AssertionError("V9 benchmark seed collision")
        used_seeds.update(values)
        return values

    budget = config["budgets"][0]
    methods = list(config["external_methods"]["primary"])
    for cell in config["cells"]:
        for cluster in range(int(config["clusters"])):
            paired.append(
                _augment_record(
                    _paired_record(
                        config,
                        cell,
                        budget,
                        cluster,
                        cast(tuple[int, int], allocate(2)),
                    ),
                    cell,
                )
            )
            for method in methods:
                external.append(
                    _augment_record(
                        _external_record(
                            config,
                            cell,
                            budget,
                            method,
                            cluster,
                            cast(tuple[int, int, int, int], allocate(4)),
                        ),
                        cell,
                    )
                )
    bank = json.loads(
        (ROOT / str(config["bindings"]["dcs_proposal_bank"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    paired, external = attach_work_to_target(
        config=config,
        paired_records=paired,
        external_records=external,
        bank=bank,
    )
    aggregate = aggregate_terminal_benchmark(
        config=config,
        paired_records=paired,
        external_records=external,
    )
    seed_payload = json.dumps(sorted(used_seeds), separators=(",", ":")).encode()
    development = config["stage"] == "development"
    decision = {
        "development_complete": development,
        "qualification_authorized": development and aggregate["stage_pass"],
        "qualification_complete": not development,
        "regime_conditional_empirical_claim_authorized": (
            not development and aggregate["stage_pass"]
        ),
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }
    return {
        "schema": "npi.g11.v9-terminal-benchmark-result.v1",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "stage": config["stage"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "seed_count": len(used_seeds),
        "seed_set_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "paired_records": paired,
        "external_records": external,
        "aggregate": aggregate,
        "decision": decision,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
