"""Fresh fixed-count CE-trained IS cross-check independent of SMC-bank fitting.

The CE portfolio was fitted using its own earlier training streams and is
frozen here. This development run uses new final IID draws only.
"""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.post_audit_r15_reference_crosscheck import _log_potential
from src.path_integral.r1_bottleneck_diagnostics import summarize_log_contributions
from src.path_integral.research_result_contract import source_manifest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

ROOT = Path(__file__).resolve().parents[1]


def _seed(ledger: SeedLedger, task: str, cluster: int, stream: str) -> int:
    return ledger.allocate(SeedKey(
        "post-audit-r15", "ce-precision", "fresh-fixed-count",
        task, 0, cluster, stream,
    ))


def _equivalence(mean: float, rse: float, other_mean: float, other_rse: float,
                 policy: dict[str, Any]) -> dict[str, float | bool]:
    upper = abs(mean - other_mean) + float(policy["confidence_z"]) * math.hypot(
        mean * rse, other_mean * other_rse,
    )
    margin = float(policy["relative_equivalence_margin"]) * other_mean
    return {
        "upper_difference": upper,
        "margin": margin,
        "pass": bool(rse <= policy["maximum_relative_se"]
                     and other_rse <= policy["maximum_reference_relative_se"]
                     and upper <= margin),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    inputs = {}
    hashes = {}
    for name in ("smc_design", "ce_crosscheck", "refined_reference", "smc_is_precision"):
        raw = (ROOT / config[f"{name}_path"]).read_bytes()
        inputs[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    design = inputs["smc_design"]
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    cells = []
    clusters = int(config["evaluation"]["clusters"])
    count = int(config["evaluation"]["units_per_cluster"])
    if clusters < 2 or count < 2:
        raise ValueError("CE precision requires independent clusters and draws")
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        first = next(x for x in inputs["ce_crosscheck"]["cells"]
                     if x["cell"]["id"] == cell["id"])
        refined = next(x for x in inputs["refined_reference"]["cells"]
                       if x["cell"]["id"] == cell["id"])["new_reference"]
        smc_is = next(x for x in inputs["smc_is_precision"]["cells"]
                      if x["cell"]["id"] == cell["id"])
        ce_fit = first["independent_defensive_is"]
        proposal = proposal_from_parameters(ce_fit["proposal_parameters"])
        logs = []
        norm_logs = []
        cluster_records = []
        started = time.perf_counter()
        for cluster in range(clusters):
            draw = proposal.sample(
                count,
                path_seed=_seed(ledger, problem.task_id, cluster, "path"),
                label_seed=_seed(ledger, problem.task_id, cluster, "label"),
            )
            values = _log_potential(problem, draw.samples) + draw.log_p_over_q
            logs.append(values)
            norm_logs.append(draw.log_p_over_q)
            cluster_records.append({
                "cluster": cluster,
                "conditional": summarize_log_contributions(values).__dict__,
                "normalization": summarize_log_contributions(draw.log_p_over_q).__dict__,
            })
        wall = time.perf_counter() - started
        summary = summarize_log_contributions(torch.cat(logs))
        norm = summarize_log_contributions(torch.cat(norm_logs))
        if summary.log_mean is None or summary.relative_se is None:
            raise FloatingPointError("all CE IS contributions vanished")
        scaled = [
            math.exp(x["conditional"]["log_mean"] - summary.log_mean)
            if x["conditional"]["log_mean"] is not None else 0.0
            for x in cluster_records
        ]
        between_rse = statistics.stdev(scaled) / math.sqrt(clusters) / statistics.mean(scaled)
        rse = max(summary.relative_se, between_rse)
        mean = math.exp(summary.log_mean)
        ordered = torch.sort(torch.cat(logs), descending=True).values
        top_count = max(1, math.ceil(0.01 * ordered.numel()))
        top_share = float(torch.exp(
            torch.logsumexp(ordered[:top_count], dim=0) - torch.logsumexp(ordered, dim=0)
        ))
        vs_smc = _equivalence(
            mean, rse, refined["mean"], refined["relative_se"], config["qualification"],
        )
        vs_smc_is = _equivalence(
            mean, rse, smc_is["mean"], smc_is["relative_se"], config["qualification"],
        )
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "ce_proposal_digest": ce_fit["proposal_digest"],
            "ce_training_fit_count": ce_fit["training_fit_count"],
            "samples": ordered.numel(),
            "clusters": clusters,
            "mean": mean,
            "log_mean": summary.log_mean,
            "iid_relative_se": summary.relative_se,
            "between_cluster_relative_se": between_rse,
            "relative_se": rse,
            "smc_reference_mean": refined["mean"],
            "smc_reference_relative_se": refined["relative_se"],
            "smc_is_mean": smc_is["mean"],
            "smc_is_relative_se": smc_is["relative_se"],
            "equivalence_vs_smc": vs_smc,
            "equivalence_vs_smc_is": vs_smc_is,
            "qualified": bool(vs_smc["pass"] and vs_smc_is["pass"]),
            "normalization_log_mean": norm.log_mean,
            "normalization_relative_se": norm.relative_se,
            "top_one_percent_contribution_share": top_share,
            "maximum_contribution_fraction": summary.maximum_fraction,
            "inference_wall_seconds": wall,
            "cluster_records": cluster_records,
        })
        print(json.dumps({"finished_cell": cell["id"], "rse": rse,
                          "qualified": cells[-1]["qualified"]}, allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-ce-precision.v1",
        "role": "fresh_development_ce_crosscheck_not_confirmation",
        "source": source,
        "config": config,
        "input_artifact_sha256": hashes,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "CE fitting and CE final IS do not use SMC training particles.",
            "The CE proposal was fixed from earlier development, not trained in fresh confirmation repeats.",
            "Even an exact-density, bounded-contribution IS run can miss rare contributions at finite n.",
            "This is an independent-estimator cross-check, not a claim that CE training is stable.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_ce_precision_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"CE precision output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
