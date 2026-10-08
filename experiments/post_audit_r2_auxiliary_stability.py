"""Whole auxiliary-fit diagnostics; no oracle or model winner certification."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def assess(payload: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload["records"]:
        groups[(row["cell"]["id"], row["parent_training_rep"])].append(row)
    expected = payload["config"].get("training_repetitions", 1)
    parent = json.loads((Path(__file__).resolve().parents[1]
                         /payload["config"]["proposal_artifact_path"]).read_text(encoding="utf-8"))
    parent_keys = {(r["cell"]["id"], r["parent_training_rep"]) for r in parent["records"]}
    whole_grid_complete = (len(parent_keys) == len(parent["records"]) and set(groups) == parent_keys
        and all(sorted(r.get("auxiliary_training_rep", 0) for r in rows) == list(range(expected))
                for rows in groups.values()))
    threshold = payload["config"]["evaluation"]["maximum_relative_se"]
    static_only = bool(payload["config"].get("static_only"))
    names = ["direct-q"] if static_only else ["direct-q", "event-auxiliary", "risk-auxiliary"]
    if payload["config"].get("static_control"):
        names.append("volterra-only")
    records = []
    for (cell, parent), rows in groups.items():
        fits_complete = sorted(r.get("auxiliary_training_rep", 0) for r in rows) == list(range(expected))
        summaries = []
        for name in names:
            estimators = [e for row in rows for e in row["estimators"] if e["id"] == name]
            completed = [e for e in estimators if e["status"] == "completed_development"]
            means = np.array([math.exp(e["summary"]["log_mean"]) for e in completed])
            ses = np.array([math.exp(e["summary"]["log_mean"])*e["summary"]["relative_se"] for e in completed])
            if len(means) < 2:
                summaries.append({"id": name, "stable": False, "status": "incomplete"})
                continue
            mean = float(means.mean())
            between = float(means.var(ddof=1))
            within = float(np.mean(ses**2))
            spread = float(means.max()/means.min()-1)
            passes = sum(e["summary"]["relative_se"] <= threshold for e in completed)
            summaries.append({"id": name, "whole_fits": 0 if static_only else len(means),
                "independent_runs": len(means), "precision_passes": passes,
                "mean_of_independent_fit_estimates": mean,
                "whole_fit_mean_empirical_relative_se": math.sqrt(between/len(means))/mean,
                "observed_between_fit_cv": math.sqrt(between)/mean,
                "within_final_relative_noise_rms": math.sqrt(within)/mean,
                "excess_between_run_cv_not_training_mean_variance": math.sqrt(max(0., between-within))/mean,
                "max_min_relative_spread": spread,
                "median_final_relative_se": float(np.median([e["summary"]["relative_se"] for e in completed])),
                "stable": fits_complete and len(completed) == expected and passes == expected and spread <= .25,
                "status": "development_diagnostic_not_certificate"})
        records.append({"cell": cell, "parent_training_rep": parent, "estimators": summaries})
    return {"records": records, "all_risk_fits_stable": whole_grid_complete and not static_only and bool(records) and expected >= 5 and all(
        next(e for e in row["estimators"] if e["id"] == "risk-auxiliary")["stable"] for row in records),
        "conditional_expectation_is_same_m2_for_every_r": True,
        "minimum_five_whole_fits_required": True,
        "whole_fit_grid_complete": whole_grid_complete,
        "replication_unit": "fixed_r_iid_evaluation" if static_only else "whole_auxiliary_training_and_final",
        "excess_variance_is_not_training_bias_or_random_mean_effect": True,
        "oracle_authorized": False, "d2_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(json.loads(args.result.read_text(encoding="utf-8"))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
