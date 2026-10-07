"""Fresh equal-count CE/SMC exact-mixture bank pilot; development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from src.path_integral.finite_rank_gaussian_transport import combine_defensive_gaussian_mixtures
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import (
    assign_saved_cluster_geometry,
    proposal_from_parameters,
)

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, candidate: str, rep: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", "hybrid-bank-pilot", candidate, task, 0, rep, stream,
    ))


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    inputs = {}
    hashes = {}
    for name in ("smc_design", "first_crosscheck", "mixture_crosscheck", "mixture_design"):
        raw = (ROOT / config[f"{name}_path"]).read_bytes()
        inputs[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    design = inputs["smc_design"]
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    cells = []
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        first = next(x for x in inputs["first_crosscheck"]["cells"]
                     if x["cell"]["id"] == cell["id"])
        smc = next(x for x in inputs["mixture_crosscheck"]["cells"]
                   if x["cell"]["id"] == cell["id"])
        mode = next(x for x in inputs["mixture_design"]["cells"]
                    if x["cell"]["id"] == cell["id"])["fixed_mode_geometry"]
        ce_q = proposal_from_parameters(
            first["independent_defensive_is"]["proposal_parameters"]
        )
        smc_q = proposal_from_parameters(smc["proposal_parameters"])
        proposals = {}
        for ce_weight in config["ce_mixture_weights"]:
            if ce_weight == 1.0:
                proposal = ce_q
            elif ce_weight == 0.0:
                proposal = smc_q
            else:
                proposal = combine_defensive_gaussian_mixtures(
                    (ce_q, smc_q), (float(ce_weight), 1.0 - float(ce_weight)),
                )
            proposals[f"ce-{ce_weight:.2f}"] = proposal
        candidate_results = []
        for name, proposal in proposals.items():
            reps = []
            for rep in range(int(config["bank_repetitions"])):
                draw = proposal.sample(
                    int(config["bank_count"]),
                    path_seed=_seed(ledger, problem.task_id, name, rep, "path"),
                    label_seed=_seed(ledger, problem.task_id, name, rep, "label"),
                )
                log_w = _log_potential(problem, draw.samples) + draw.log_p_over_q
                summary = summarize_log_contributions(log_w)
                labels = assign_saved_cluster_geometry(draw.samples, mode)
                weights = torch.softmax(log_w, dim=0)
                mode_masses = [
                    float(weights[labels == index].sum())
                    for index in range(len(mode["centers"]))
                ]
                reps.append({
                    "rep": rep,
                    "target_ess": summary.contribution_ess,
                    "maximum_target_weight": summary.maximum_fraction,
                    "mode_target_masses": mode_masses,
                    "normalization": summarize_log_contributions(draw.log_p_over_q).__dict__,
                })
            ess = sorted(x["target_ess"] for x in reps)
            tenth_index = math.floor(0.1 * (len(ess) - 1))
            candidate_results.append({
                "name": name,
                "median_target_ess": statistics.median(ess),
                "tenth_percentile_target_ess": ess[tenth_index],
                "minimum_target_ess": ess[0],
                "mean_maximum_target_weight": statistics.mean(
                    x["maximum_target_weight"] for x in reps
                ),
                "repetitions": reps,
            })
        cells.append({"cell": cell, "task_id": problem.task_id,
                      "candidates": candidate_results})
        print(json.dumps({"finished_cell": cell["id"]}, allow_nan=False), flush=True)
    control_names = ("ce-1.00", "ce-0.00")
    viable = []
    for candidate in cells[0]["candidates"]:
        name = candidate["name"]
        if name in control_names:
            continue
        checks = []
        for cell in cells:
            current = next(x for x in cell["candidates"] if x["name"] == name)
            best_parent = max(
                x["median_target_ess"] for x in cell["candidates"]
                if x["name"] in control_names
            )
            checks.append(
                current["median_target_ess"] >=
                config["selection"]["minimum_median_ess_factor_over_best_parent"] * best_parent
                and current["tenth_percentile_target_ess"] >=
                config["selection"]["minimum_tenth_percentile_ess"]
            )
        if all(checks):
            viable.append(name)
    return {
        "schema": "npi.post-audit.r15-hybrid-bank-pilot.v1",
        "role": "development_bank_diagnostic_not_confirmation",
        "source": source,
        "config": config,
        "input_artifact_sha256": hashes,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "selected_candidate": viable[0] if viable else None,
        "selection_status": "material_gain_both_cells" if viable else "no_material_gain",
        "limitations": [
            "ESS is for the ordinary IID target weights g*p/q at fixed bank count.",
            "The fixed clustering geometry came from earlier SMC development banks.",
            "A better bank ESS alone does not establish lower final-estimator second moment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_hybrid_bank_pilot_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"hybrid bank output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
