"""Frozen two-cell SMC bridge/resampling/mutation development design."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.research_result_contract import source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.volterra_conditional_payoffs import (
    evaluate_rbergomi_conditional_terminal,
)
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def temperatures(levels: int, power: int) -> tuple[float, ...]:
    if levels < 2 or power < 1:
        raise ValueError("invalid fixed bridge")
    return tuple((stage / levels) ** power for stage in range(levels + 1))


def _candidate_result(
    problem: Any, candidate: dict[str, Any], design: dict[str, Any],
    ledger: SeedLedger,
) -> dict[str, Any]:
    seed = ledger.allocate(SeedKey(
        "post-audit-r15", "design", "conditional-smc", problem.task_id,
        0, 0, str(candidate["id"]),
    ))
    config = WeightedSMCConfig(
        particles=int(design["particles"]),
        temperatures=temperatures(int(design["levels"]), int(candidate["bridge_power"])),
        mutation_steps=int(candidate["mutation_steps"]),
        pcn_scale=float(candidate["pcn_scale"]),
        replicates=int(design["independent_replicates"]), seed=seed,
        resample_every=int(candidate["resample_every"]),
        resampling_scheme=str(candidate["resampling_scheme"]),
    )
    started = time.perf_counter()
    result = estimate_weighted_tempered_normalizer(
        lambda z: evaluate_rbergomi_conditional_terminal(
            problem, z,
        ).payoffs.log_left_probability,
        dimension=problem.local_dimension, config=config,
    )
    wall = time.perf_counter() - started
    if not 0 < result.mean < 1 or not math.isfinite(result.standard_error):
        raise FloatingPointError("design SMC produced an invalid estimate")
    cv2 = (result.standard_error / result.mean) ** 2 * config.replicates
    work_per_rep = result.potential_evaluations / config.replicates
    ancestors = [
        int(x["final_unique_initial_ancestors"]) for x in result.replicate_diagnostics
    ]
    final_ess = [
        float(x["final_weight_ess_fraction"]) for x in result.replicate_diagnostics
    ]
    return {
        "candidate": candidate,
        "seed": seed,
        "mean": result.mean,
        "standard_error": result.standard_error,
        "relative_se": result.standard_error / result.mean,
        "log_replicate_estimates": result.log_replicate_estimates.tolist(),
        "median_unique_initial_ancestors": statistics.median(ancestors),
        "minimum_unique_initial_ancestors": min(ancestors),
        "median_final_weight_ess_fraction": statistics.median(final_ess),
        "median_resampling_stages": statistics.median(
            int(x["resampling_stages"]) for x in result.replicate_diagnostics
        ),
        "mutation_acceptance_rate": result.mutation_acceptance_rate,
        "potential_evaluations_per_rep": work_per_rep,
        "cost_normalized_cv2": cv2 * work_per_rep,
        "wall_seconds": wall,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    source = source_manifest(ROOT, config=config)
    design = config["design"]
    ledger = SeedLedger()
    cells = []
    for cell in config["cells"]:
        problem = _problem(config["model"], cell, int(design["steps"]))
        results = []
        for candidate in design["candidates"]:
            results.append(_candidate_result(problem, candidate, design, ledger))
        cells.append({"cell": cell, "task_id": problem.task_id, "candidates": results})
        print(json.dumps({"finished_cell": cell["id"]}, allow_nan=False), flush=True)
    eligible = []
    for candidate in design["candidates"]:
        identifier = candidate["id"]
        per_cell = [next(x for x in cell["candidates"] if x["candidate"]["id"] == identifier)
                    for cell in cells]
        qualifies = all(
            item["median_unique_initial_ancestors"]
            >= design["minimum_median_unique_initial_ancestors"]
            and item["median_final_weight_ess_fraction"]
            >= design["minimum_median_final_weight_ess_fraction"]
            for item in per_cell
        )
        eligible.append({
            "id": identifier,
            "qualifies_diversity_gate": qualifies,
            "worst_cell_cost_normalized_cv2": max(
                item["cost_normalized_cv2"] for item in per_cell
            ),
            "minimum_cell_median_ancestors": min(
                item["median_unique_initial_ancestors"] for item in per_cell
            ),
        })
    qualified = [item for item in eligible if item["qualifies_diversity_gate"]]
    if qualified:
        selected = min(qualified, key=lambda item: (
            item["worst_cell_cost_normalized_cv2"], item["id"],
        ))
        selection_status = "diversity_gate_pass"
    else:
        selected = min(eligible, key=lambda item: (
            -item["minimum_cell_median_ancestors"],
            item["worst_cell_cost_normalized_cv2"], item["id"],
        ))
        selection_status = "diversity_gate_failed_exploratory_fallback"
    reference_counts = {}
    for cell in cells:
        item = next(x for x in cell["candidates"] if x["candidate"]["id"] == selected["id"])
        estimated = math.ceil(
            (item["relative_se"] / design["reference_target_relative_se"]) ** 2
            * design["independent_replicates"]
        )
        reference_counts[cell["cell"]["id"]] = max(
            int(design["reference_minimum_replicates"]),
            min(int(design["reference_maximum_replicates"]), estimated),
        )
    return {
        "schema": "npi.post-audit.r15-reference-design.v1",
        "role": "development_selection_not_reference",
        "source": source,
        "config": config,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "selection_table": eligible,
        "selected_candidate_id": selected["id"],
        "selection_status": selection_status,
        "prespecified_reference_counts": reference_counts,
        "reference_rule": "counts_from_independent_pilot_only_not_production_outcomes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_reference_design_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"design output already exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output), "selection": payload["selected_candidate_id"]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
