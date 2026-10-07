"""Fresh fixed-count, exact-density IS precision cross-check against SMC.

The proposal was frozen using development data. These 640 independent IID
clusters per cell are new, are not pooled with the earlier failed IS run, and
remain development rather than sealed confirmation evidence.
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
        "post-audit-r15", "is-precision", "fresh-fixed-count",
        task, 0, cluster, stream,
    ))


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["torch_threads"]))
    inputs = {}
    hashes = {}
    for name in ("smc_design", "mixture_crosscheck", "refined_reference"):
        raw = (ROOT / config[f"{name}_path"]).read_bytes()
        inputs[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    design = inputs["smc_design"]
    mixture = inputs["mixture_crosscheck"]
    reference = inputs["refined_reference"]
    ledger = SeedLedger()
    source = source_manifest(ROOT, config=config)
    cells = []
    spec = config["evaluation"]
    clusters = int(spec["clusters"])
    count = int(spec["units_per_cluster"])
    if clusters < 2 or count < 2:
        raise ValueError("at least two clusters with at least two draws required")
    for cell in design["config"]["cells"]:
        problem = _problem(design["config"]["model"], cell,
                           int(design["config"]["design"]["steps"]))
        prior = next(x for x in mixture["cells"] if x["cell"]["id"] == cell["id"])
        ref = next(x for x in reference["cells"] if x["cell"]["id"] == cell["id"])[
            "new_reference"
        ]
        proposal = proposal_from_parameters(prior["proposal_parameters"])
        logs = []
        normalization_logs = []
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
            normalization_logs.append(draw.log_p_over_q)
            cluster_records.append({
                "cluster": cluster,
                "conditional": summarize_log_contributions(values).__dict__,
                "normalization": summarize_log_contributions(draw.log_p_over_q).__dict__,
            })
        wall = time.perf_counter() - started
        summary = summarize_log_contributions(torch.cat(logs))
        norm = summarize_log_contributions(torch.cat(normalization_logs))
        if summary.log_mean is None or summary.relative_se is None:
            raise FloatingPointError("all IS contributions vanished")
        scaled = [
            math.exp(x["conditional"]["log_mean"] - summary.log_mean)
            if x["conditional"]["log_mean"] is not None else 0.0
            for x in cluster_records
        ]
        between_rse = statistics.stdev(scaled) / math.sqrt(clusters) / statistics.mean(scaled)
        rse = max(summary.relative_se, between_rse)
        mean = math.exp(summary.log_mean)
        ordered = torch.sort(torch.cat(logs), descending=True).values
        total = torch.logsumexp(ordered, dim=0)
        top_count = max(1, math.ceil(0.01 * ordered.numel()))
        top_share = float(torch.exp(torch.logsumexp(ordered[:top_count], dim=0) - total))
        upper = abs(mean - ref["mean"]) + float(config["qualification"]["confidence_z"]) * math.hypot(
            mean * rse, ref["standard_error"],
        )
        margin = float(config["qualification"]["relative_equivalence_margin"]) * ref["mean"]
        qualified = (
            rse <= config["qualification"]["maximum_relative_se"]
            and ref["relative_se"] <= config["qualification"]["maximum_reference_relative_se"]
            and upper <= margin
        )
        cells.append({
            "cell": cell,
            "task_id": problem.task_id,
            "proposal_digest": prior["proposal_digest"],
            "samples": ordered.numel(),
            "clusters": clusters,
            "log_mean": summary.log_mean,
            "mean": mean,
            "iid_relative_se": summary.relative_se,
            "between_cluster_relative_se": between_rse,
            "relative_se": rse,
            "reference_mean": ref["mean"],
            "reference_relative_se": ref["relative_se"],
            "equivalence_upper_difference": upper,
            "equivalence_margin": margin,
            "qualified": qualified,
            "normalization_log_mean": norm.log_mean,
            "normalization_relative_se": norm.relative_se,
            "top_one_percent_contribution_share": top_share,
            "maximum_contribution_fraction": summary.maximum_fraction,
            "inference_wall_seconds": wall,
            "cluster_records": cluster_records,
        })
        print(json.dumps({"finished_cell": cell["id"], "rse": rse,
                          "qualified": qualified}, allow_nan=False), flush=True)
    return {
        "schema": "npi.post-audit.r15-is-precision.v1",
        "role": "fresh_development_crosscheck_not_confirmation",
        "source": source,
        "config": config,
        "input_artifact_sha256": hashes,
        "seed_ledger": ledger.to_dict(),
        "cells": cells,
        "limitations": [
            "IS final draws are independent of all SMC reference and training banks.",
            "Frozen proposal was chosen using earlier development data; this is not a sealed holdout.",
            "Precision does not certify discovery of all rare contribution modes.",
            "Earlier failed IS runs were not pooled into this estimator.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/post_audit/r15_is_precision_v1.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    output = ROOT / config["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"IS precision output exists: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, output)
    print(json.dumps({"output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
