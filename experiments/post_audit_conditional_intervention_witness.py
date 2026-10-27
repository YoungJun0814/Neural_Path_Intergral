"""Post-hoc full-law checks of the largest observed intervention contributions.

This is a selected-extreme numerical audit, never independent confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.post_audit_conditional_intervention import (
    DIRECTORY,
    PLAN,
    Work,
    draw,
    streams,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from experiments.post_audit_small_grid_model_restart import prepare
from src.path_integral.conditional_intervention import sign_orbit_log_probabilities
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.risk_aware_model_correction import full_mixture_log_q_over_p
from src.path_integral.structural_v2_inventory import audit_snapshot

OUTPUT = DIRECTORY+"/selected_extreme_audit"
EXPECTED = "d4d1215077eef40fc842f0f78668bd7ab742cc79379646b7c3779826d55b0619"


def select_extremes(records: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for cell in sorted({r["cell"] for r in records}):
        for design in ("q", "p"):
            best: dict[str, Any] = {}
            for row in records:
                if row["cell"] != cell or row["phase"] != design:
                    continue
                if sha(path/row["array_path"]) != row["array_sha256"]:
                    raise ValueError("selected-extreme input array changed")
                with np.load(path/row["array_path"], allow_pickle=False) as arrays:
                    risk = arrays["risk"][:, 0]
                    event = arrays["event"]
                    with np.errstate(divide="ignore"):
                        difference = np.maximum(event[:, 0], event[:, 3])+np.log(
                            -np.expm1(-np.abs(event[:, 0]-event[:, 3])))
                for role, values in (("largest_original_M2_orbit", risk),
                                     ("largest_absolute_event_change", difference)):
                    index = int(np.argmax(values))
                    value = float(values[index])
                    if role not in best or value > best[role]["selection_log_value"]:
                        best[role] = {"cell": cell, "design": design, "block": row["block"],
                            "index": index, "selection_role": role, "selection_log_value": value}
            chosen.extend(best.values())
    if len(chosen) != 8:
        raise ValueError("two selected-extreme slots per cell/design required")
    return chosen


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    parent_path = ROOT/DIRECTORY
    if sha(parent_path/"result.json") != EXPECTED:
        raise ValueError("immutable parent result changed")
    parent = json.loads((parent_path/"result.json").read_text())
    audit = json.loads((parent_path/"audit.json").read_text())
    if audit["result_sha256"] != EXPECTED:
        raise ValueError("parent representative audit mismatch")
    audit_snapshot(ROOT, parent["source"], parent["config"])
    selected = select_extremes(parent["records"], parent_path)
    output = ROOT/OUTPUT
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("selected-extreme audit is nonoverwriting")
    output.mkdir(parents=True, exist_ok=True)
    config = {"parent_sha256": EXPECTED, "selected": selected, "full_cap": 80,
        "orbit_scalar_cap": 32, "maximum_full_log_error": 1e-8,
        "post_hoc_selected_extremes_not_confirmation": True}
    source = freeze_source(output/"run.json", config,
        extra_snapshot_paths=(PLAN, DIRECTORY+"/result.json", DIRECTORY+"/audit.json"))
    write(output/"protocol.json", {"source": source, "config": config})
    prepared, _ = prepare(parent["config"])
    mapping, _ = streams(list(prepared))
    work = Work(80, 32, 180)
    rows = []
    for item in selected:
        cache, spec = prepared[item["cell"]]
        count = 8192 if item["design"] == "q" else 2048
        z = draw(spec, count, item["cell"], item["design"], item["block"], mapping)[item["index"]:item["index"]+1]
        work.charge(1, 4)
        fast = sign_orbit_log_probabilities(cache, spec, z)
        full = z.repeat(4, 1)
        full[:, [57, 61]] *= torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]], dtype=torch.float64)
        work.charge(4)
        direct = cache.log_probability(full)
        d = full_mixture_log_q_over_p(full, spec)
        error = float((fast[0]-direct).abs().max())
        if error > 1e-8 or not torch.equal(d, d[0].expand(4)):
            raise FloatingPointError("selected-extreme full-law/density mismatch")
        changed_unused = full[:1].clone()
        changed_unused[:, 63] += 10
        work.charge(1)
        if not torch.equal(cache.log_probability(changed_unused), direct[:1]):
            raise FloatingPointError("unused63 negative control failed")
        work.charge(4)
        _, integrated, variance = cache.quantities(full)
        rows.append({**item, "coordinates": z[0].tolist(), "log_g_orbit": direct.tolist(),
            "log_density_q_over_p": float(d[0]), "maximum_full_log_error": error,
            "unused63_exactly_unchanged": True, "residual57": float(z[0, 57]),
            "residual61": float(z[0, 61]), "orbit_integrated_variance": integrated.tolist(),
            "orbit_peak_used_variance": variance[:, :-1].max(1).values.tolist(),
            "orbit_peak_time_index": variance[:, :-1].argmax(1).tolist()})
    result = {"source": source, "config": config, "rows": rows, "work": work.result(),
        "feature_full_calls_included": 32, "total_full_calls": work.full,
        "status": "SELECTED_EXTREME_FULL_LAW_AND_UNUSED_CONTROL_PASSED",
        "not_independent_confirmation_or_coverage_certificate": True,
        "P2_qualified": False, "production_adoption": False}
    write(output/"result.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps({"status": run()["status"]}))
