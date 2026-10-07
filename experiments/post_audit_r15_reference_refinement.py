"""Fresh fixed-count SMC reference after a failed first precision gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from experiments.post_audit_r15_reference_design import temperatures
from src.path_integral.research_result_contract import source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    design_bytes = (ROOT / config["design_path"]).read_bytes()
    first_bytes = (ROOT / config["first_crosscheck_path"]).read_bytes()
    design = json.loads(design_bytes)
    first = json.loads(first_bytes)
    selected_id = design["selected_candidate_id"]
    if selected_id != first["selected_candidate_id"]:
        raise ValueError("SMC design artifacts disagree")
    candidate = next(x for x in design["config"]["design"]["candidates"]
                     if x["id"] == selected_id)
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    cells = []
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        count = int(config["fixed_new_replicates"][cell["id"]])
        seed = ledger.allocate(SeedKey(
            "post-audit-r15", "smc-reference-refinement", "fresh-fixed-count",
            problem.task_id, 0, 0, selected_id,
        ))
        smc_config = WeightedSMCConfig(
            particles=int(design["config"]["design"]["particles"]),
            temperatures=temperatures(int(design["config"]["design"]["levels"]),
                                      int(candidate["bridge_power"])),
            mutation_steps=int(candidate["mutation_steps"]),
            pcn_scale=float(candidate["pcn_scale"]),
            replicates=count, seed=seed,
            resample_every=int(candidate["resample_every"]),
            resampling_scheme=str(candidate["resampling_scheme"]),
        )
        started = time.perf_counter()
        result = estimate_weighted_tempered_normalizer(
            partial(_log_potential, problem),
            dimension=problem.local_dimension, config=smc_config,
        )
        wall = time.perf_counter() - started
        first_cell = next(x for x in first["cells"] if x["cell"]["id"] == cell["id"])
        previous = first_cell["smc_reference"]
        combined_z = abs(result.mean - previous["mean"]) / math.hypot(
            result.standard_error, previous["standard_error"],
        )
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "new_reference": {
                "mean": result.mean,
                "log_mean": math.log(result.mean),
                "standard_error": result.standard_error,
                "relative_se": result.standard_error / result.mean,
                "replicates": count,
                "seed": seed,
                "log_replicate_estimates": result.log_replicate_estimates.tolist(),
                "potential_evaluations": result.potential_evaluations,
                "wall_seconds": wall,
                "median_unique_initial_ancestors": statistics.median(
                    int(x["final_unique_initial_ancestors"])
                    for x in result.replicate_diagnostics
                ),
                "median_final_weight_ess_fraction": statistics.median(
                    float(x["final_weight_ess_fraction"])
                    for x in result.replicate_diagnostics
                ),
            },
            "first_reference": {
                "mean": previous["mean"],
                "standard_error": previous["standard_error"],
                "relative_se": previous["relative_se"],
            },
            "independent_reference_z": combined_z,
            "precision_gate_pass": result.standard_error / result.mean
            <= config["maximum_reference_relative_se"],
        })
        print(json.dumps({"finished_cell": cell["id"],
                          "relative_se": cells[-1]["new_reference"]["relative_se"]},
                         allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-reference-refinement.v1",
        "role": "fresh_fixed_count_reference_not_confirmation",
        "source": source,
        "config": config,
        "design_artifact_sha256": hashlib.sha256(design_bytes).hexdigest(),
        "first_reference_artifact_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "selected_candidate_id": selected_id,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "First reference was not pooled because its failure triggered the fixed independent refinement.",
            "Small RSE does not establish complete mode coverage; another estimator must agree.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_reference_refinement_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"reference refinement output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
