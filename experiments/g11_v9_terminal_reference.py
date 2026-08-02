"""Generate fresh V9 terminal references with independently scrambled RQMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.baselines import (
    evaluate_smoothing_rqmc_units,
    freeze_smoothing_rqmc_proposal,
)
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import (
    process_peak_resident_memory_bytes,
    runtime_provenance,
    source_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v9-terminal-reference.v1"
SCHEMA_V2 = "npi.g11.v9-terminal-reference.v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("unexpected V9 terminal-reference config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate(config: dict[str, Any]) -> None:
    binding = config.get("claim_contract")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("V9 reference requires one exact claim-contract binding")
    claim_path = ROOT / str(binding["path"])
    if not claim_path.is_file() or _sha256(claim_path) != str(binding["sha256"]):
        raise ValueError("V9 reference claim-contract binding mismatch")
    claim = yaml.safe_load(claim_path.read_text(encoding="utf-8"))
    if config["model"] != claim["model"] or config["cells"] != claim["cells"]:
        raise ValueError("V9 reference model/cells differ from the claim contract")
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("V9 reference namespace was not outcome-blind at freeze")
    if config["schema"] == SCHEMA_V2:
        pilot_bindings = config.get("pilot_bindings")
        if not isinstance(pilot_bindings, dict) or set(pilot_bindings) != {
            "reference_v1",
            "reference_v1_audit",
        }:
            raise ValueError("V9 reference V2 requires immutable V1 pilot bindings")
        for name, pilot_binding in pilot_bindings.items():
            if not isinstance(pilot_binding, dict) or set(pilot_binding) != {"path", "sha256"}:
                raise ValueError(f"invalid V9 reference V2 pilot binding: {name}")
            pilot_path = ROOT / str(pilot_binding["path"])
            if not pilot_path.is_file() or _sha256(pilot_path) != str(pilot_binding["sha256"]):
                raise ValueError(f"V9 reference V2 pilot binding mismatch: {name}")
        pilot = json.loads(
            (ROOT / str(pilot_bindings["reference_v1"]["path"])).read_text(
                encoding="utf-8"
            )
        )
        pilot_audit = json.loads(
            (ROOT / str(pilot_bindings["reference_v1_audit"]["path"])).read_text(
                encoding="utf-8"
            )
        )
        if pilot_audit.get("passed") is not True or pilot["decision"]["reference_complete"] is not False:
            raise ValueError("V9 reference V2 requires valid but precision-incomplete V1")
        rule = config["allocation_rule"]
        if rule != {
            "pilot_randomizations": 32,
            "variance_safety_factor": 2.0,
            "minimum_randomizations": 64,
            "round_to_power_of_two": True,
            "pilot_samples_pooled_into_v2": False,
        }:
            raise ValueError("V9 reference V2 allocation rule changed")
        declared = {str(key): int(value) for key, value in config["randomizations_by_cell"].items()}
        expected: dict[str, int] = {}
        for cell in pilot["cells"]:
            relative = float(cell["relative_standard_error"])
            required = max(
                int(rule["minimum_randomizations"]),
                math.ceil(
                    float(rule["variance_safety_factor"])
                    * int(rule["pilot_randomizations"])
                    * (relative / float(config["maximum_relative_standard_error"])) ** 2
                ),
            )
            expected[str(cell["cell_id"])] = 1 << (required - 1).bit_length()
        if declared != expected:
            raise ValueError("V9 reference V2 allocation is not the frozen pilot rule")
        randomization_counts = list(declared.values())
    else:
        randomization_counts = [int(config["randomizations"])] * len(config["cells"])
    randomizations = min(randomization_counts)
    points = int(config["points_per_randomization"])
    if randomizations < 8 or points < 2 or points & (points - 1):
        raise ValueError("V9 reference requires >=8 randomizations and power-of-two points")
    proposal_base = int(config["proposal_seed_base"])
    randomization_base = int(config["randomization_seed_base"])
    spans = sum(randomization_counts)
    if proposal_base <= randomization_base + spans and randomization_base <= proposal_base + len(
        config["cells"]
    ):
        raise ValueError("V9 reference proposal and RQMC seeds overlap")
    source_commit = str(config["source_parent_commit"])
    current = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", source_commit, current),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise ValueError("V9 reference source commit is not an ancestor")
    source_paths = (
        "experiments/g11_v9_terminal_reference.py",
        "src/path_integral/baselines/smoothing_rqmc.py",
        "src/path_integral/baselines/rbergomi_common.py",
        "src/path_integral/rbergomi_smoothing.py",
        "src/path_integral/rbergomi_fft.py",
    )
    if subprocess.run(
        ("git", "diff", "--quiet", source_commit, "--", *source_paths),
        cwd=ROOT,
        check=False,
    ).returncode or subprocess.check_output(
        ("git", "status", "--porcelain", "--", *source_paths), cwd=ROOT, text=True
    ).strip():
        raise ValueError("V9 reference implementation differs from its bound source")


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    _validate(config)
    model = config["model"]
    points = int(config["points_per_randomization"])
    maximum_relative_se = float(config["maximum_relative_standard_error"])
    cells: list[dict[str, Any]] = []
    all_seeds: list[int] = []
    seed_cursor = int(config["randomization_seed_base"])
    for index, cell in enumerate(config["cells"]):
        randomizations = int(
            config["randomizations_by_cell"][str(cell["cell_id"])]
            if config["schema"] == SCHEMA_V2
            else config["randomizations"]
        )
        problem = RBergomiBaselineProblem(
            task_id=str(cell["cell_id"]),
            task=TerminalThresholdTask(float(cell["threshold"])),
            spot=float(model["spot"]),
            maturity=float(model["maturity"]),
            steps=int(model["steps"]),
            hurst=float(cell["hurst"]),
            eta=float(model["eta"]),
            xi=float(model["xi"]),
            rho=float(model["rho"]),
        )
        proposal_seed = int(config["proposal_seed_base"]) + index
        seed = seed_cursor
        seed_cursor += randomizations
        seeds = list(range(seed, seed + randomizations))
        all_seeds.extend([proposal_seed, *seeds])
        proposal = freeze_smoothing_rqmc_proposal(problem, training_seed=proposal_seed)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        batch = evaluate_smoothing_rqmc_units(
            problem,
            proposal,
            randomizations=randomizations,
            points_per_randomization=points,
            seed=seed,
        )
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        values = batch.unit_contributions
        estimate = float(torch.mean(values))
        standard_error = math.sqrt(float(torch.var(values, unbiased=True)) / randomizations)
        relative_se = standard_error / estimate if estimate > 0.0 else math.inf
        raw_samples = randomizations * points
        work = raw_samples * (problem.latent_dimension + problem.steps + 1)
        cells.append(
            {
                **cell,
                "method": "independent_smoothing_rqmc_reference",
                "proposal_seed": proposal_seed,
                "randomization_seeds": seeds,
                "randomizations": randomizations,
                "points_per_randomization": points,
                "unit_estimates": [float(value) for value in values],
                "estimate": estimate,
                "standard_error": standard_error,
                "relative_standard_error": relative_se,
                "relative_standard_error_pass": relative_se <= maximum_relative_se,
                "cost": {
                    "raw_samples": raw_samples,
                    "cdf_calls": raw_samples,
                    "algorithmic_work_units": float(work),
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "peak_memory_bytes": process_peak_resident_memory_bytes(),
                },
            }
        )
    complete = all(cell["relative_standard_error_pass"] for cell in cells)
    seed_payload = json.dumps(sorted(all_seeds), separators=(",", ":")).encode()
    return {
        "schema": (
            "npi.g11.v9-terminal-reference-result.v2"
            if config["schema"] == SCHEMA_V2
            else "npi.g11.v9-terminal-reference-result.v1"
        ),
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "seed_count": len(all_seeds),
        "seed_set_sha256": hashlib.sha256(seed_payload).hexdigest(),
        "cells": cells,
        "decision": {
            "reference_complete": complete,
            "proposal_bank_authorized": complete,
            "benchmark_authorized": False,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
