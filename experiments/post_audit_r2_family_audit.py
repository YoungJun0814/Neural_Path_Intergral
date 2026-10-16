"""Arithmetic, snapshot, proposal and random-role audit; no statistical blessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import torch

from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.seed_ledger import SeedLedger
from src.path_integral.weighted_bank_mixture import proposal_from_parameters

ROOT = Path(__file__).resolve().parents[1]


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["schema"] != "npi.post-audit.r2-family-diagnosis.v1":
        raise ValueError("unsupported result schema")
    config, source = payload["config"], payload["source"]
    if canonical_digest(config) != source["config_digest"]:
        raise ValueError("configuration digest mismatch")
    snapshot = ROOT / source["snapshot_path"]
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source["snapshot_sha256"]:
        raise ValueError("source archive hash mismatch")
    with zipfile.ZipFile(snapshot) as archive:
        for name, digest in source["snapshot_file_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("snapshot entry mismatch")
        if json.loads(archive.read("RUN_CONFIG.json")) != config:
            raise ValueError("archived configuration mismatch")
    reference_bytes = (ROOT / config["reference_path"]).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != payload["reference_sha256"]:
        raise ValueError("reference artifact changed")
    references = json.loads(reference_bytes)
    ledger = SeedLedger.from_dict(payload["seed_ledger"])
    if len(payload["records"]) != len(config["cells"]) * config["parent_training_replicates"]:
        raise ValueError("missing whole-parent run")
    identities = set()
    candidate_count = 0
    final_count = 0
    allocation = config["allocation"]
    for row in payload["records"]:
        key = (row["cell"]["id"], row["parent_training_rep"])
        if key in identities:
            raise ValueError("duplicate parent identity")
        identities.add(key)
        if row["status"] == "unresolved" and not row["failure_reasons"]:
            raise ValueError("unexplained failed run")
        for item in row["candidates"]:
            candidate_count += 1
            parameters = item["proposal_parameters"]
            if canonical_digest(parameters) != item["proposal_digest"]:
                raise ValueError("proposal binding mismatch")
            proposal = proposal_from_parameters(parameters)
            if proposal.defensive_mass < .1 - 1e-12:
                raise ValueError("lost defensive mass")
            if "allocation" in item:
                pilot = item["pilot"]
                raw = allocation["safety_factor"] * pilot["relative_se"]**2 * pilot["count"] / allocation["target_relative_se"]**2
                planned = max(allocation["batch_size"], math.ceil(raw / allocation["batch_size"]) * allocation["batch_size"])
                if planned != item["allocation"]["planned_count"]:
                    raise ValueError("allocation arithmetic mismatch")
                if planned > allocation["maximum_count_per_method"] and item["allocation"]["status"] == "allocated":
                    raise ValueError("budget truncation mislabelled as allocation")
            if "final" in item:
                batches = item["final_batches"]
                n = sum(x["count"] for x in batches)
                means = torch.tensor([x["log_mean"] + math.log(x["count"]) for x in batches], dtype=torch.float64)
                seconds = torch.tensor([x["log_second_moment"] + math.log(x["count"]) for x in batches], dtype=torch.float64)
                log_mean = float(torch.logsumexp(means, 0) - math.log(n))
                log_second = float(torch.logsumexp(seconds, 0) - math.log(n))
                rse = math.sqrt(max(0., math.expm1(log_second-2*log_mean)) / (n-1))
                final = item["final"]
                if (n != final["count"] or not math.isclose(log_mean, final["log_mean"], abs_tol=1e-12)
                        or not math.isclose(rse, final["relative_se"], rel_tol=1e-10, abs_tol=1e-12)):
                    raise ValueError("final moment arithmetic mismatch")
                final_count += 1
                if item["final_completed"] != (n == item["allocation"]["planned_count"]):
                    raise ValueError("partial final mislabelled as complete")
                if item["final_completed"]:
                    ref = next(x["new_reference"] for x in references["cells"]
                               if x["cell"]["id"] == row["cell"]["id"])
                    q = config["qualification"]
                    mean = math.exp(log_mean)
                    se = mean * rse
                    upper = abs(mean - ref["mean"]) + q["confidence_z"] * math.hypot(se, ref["standard_error"])
                    accuracy = item["accuracy"]
                    expected = {"mean": mean, "standard_error": se,
                                "reference_mean": ref["mean"], "reference_standard_error": ref["standard_error"],
                                "equivalence_upper_difference": upper}
                    if any(not math.isclose(accuracy[k], v, rel_tol=1e-10, abs_tol=1e-20)
                           for k, v in expected.items()):
                        raise ValueError("accuracy arithmetic mismatch")
                    flags = {"method_precision_ok": rse <= allocation["target_relative_se"],
                             "reference_precision_ok": ref["standard_error"] <= q["reference_se_fraction_of_method_se"] * se,
                             "equivalence_ok": upper <= q["relative_equivalence_margin"] * ref["mean"]}
                    if any(accuracy[k] != v for k, v in flags.items()):
                        raise ValueError("accuracy flag mismatch")
            if item["qualified"]:
                if (item["failure_reasons"] or not item.get("final_completed")
                        or not all(item["accuracy"][k] for k in ("reference_precision_ok", "method_precision_ok", "equivalence_ok"))):
                    raise ValueError("stale qualification")
            expected_cost = sum(item.get(k, 0) for k in ("offline_wall_seconds", "fit_wall_seconds",
                                                        "selection_wall_seconds", "inference_wall_seconds"))
            if not math.isclose(expected_cost, item["total_deployment_wall_seconds"], abs_tol=1e-12):
                raise ValueError("cost mismatch")
        if row["precision_cost_comparison_authorized"] != (sum(bool(x["qualified"]) for x in row["candidates"]) >= 2):
            raise ValueError("unauthorized cost comparison")
    if "potential_evaluations" in payload["records"][0]:
        if sum(x["potential_evaluations"] for x in payload["records"]) != payload["potential_evaluations"]:
            raise ValueError("lost failed-run work")
    if payload["potential_evaluations"] > config["budget"]["max_potential_evaluations"]:
        raise ValueError("potential budget exceeded")
    return {"status": "snapshot_and_arithmetic_pass_not_performance_validation",
            "whole_parent_runs": len(identities), "candidates": candidate_count,
            "final_summaries": final_count, "seed_streams": len(ledger.records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.artifact.read_text(encoding="utf-8"))), allow_nan=False))


if __name__ == "__main__":
    main()
