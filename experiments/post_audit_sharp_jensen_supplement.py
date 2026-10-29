"""Deterministic source-bound supplement; no sampling or changed HMC protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import torch

from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting as write
from src.path_integral.sharp_jensen_continuum_bound import (
    average_driver_moments,
    sharp_jensen_bounds,
)
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/sharp_jensen_supplement_v1/session_20261028"
THEORY = "docs/theory/SHARP_JENSEN_INVERSE_MOMENT_SUPPLEMENT_2026-10-28_KO.md"
ANCHOR = "results/post_audit/p2_hamiltonian_tail_v1/session_20261028/continuum.json"


def config() -> dict[str, Any]:
    return {"eta": [1.5, 2.], "xi": .04, "rho": -.7, "maturity": 1., "hurst": .05,
        "steps": [4096, 8192, 16384], "holder_r": [1.05, 1.1, 1.25, 1.5, 2., 3., 4., 8.],
        "anchor": ANCHOR, "anchor_sha256": sha(ROOT/ANCHOR), "physical_samples": 0}


def calculate(c: dict[str, Any]) -> dict[str, Any]:
    moments = {n: average_driver_moments(hurst=c["hurst"], maturity=c["maturity"], steps=n) for n in c["steps"]}
    rows = [{"eta": eta, **sharp_jensen_bounds(eta=eta, xi=c["xi"], rho=c["rho"], maturity=c["maturity"],
        hurst=c["hurst"], steps=n, holder_r=r, moments=moments[n])}
        for eta in c["eta"] for n in c["steps"] for r in c["holder_r"]]
    return {"moments": {str(n): asdict(m) for n, m in moments.items()}, "rows": rows,
        "physical_samples": 0, "relative_bias_certificate_obtained": False,
        "not_independent_reference_or_statistical_confirmation": True}


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    path, c = ROOT/DIRECTORY, config()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("nonoverwriting deterministic supplement required")
    path.mkdir(parents=True, exist_ok=True)
    source = freeze_source(path/"run.json", c, extra_snapshot_paths=(THEORY, ANCHOR))
    write(path/"protocol.json", {"source": source, "config": c})
    result = {"source": source, "config": c, "summary": calculate(c)}
    write(path/"result.json", result)
    return result


def audit() -> dict[str, Any]:
    torch.set_num_threads(1)
    path = ROOT/DIRECTORY
    value, protocol = [json.loads((path/n).read_text()) for n in ("result.json", "protocol.json")]
    if protocol != {"source": value["source"], "config": value["config"]} or config() != value["config"]:
        raise ValueError("supplement protocol or anchor changed")
    snapshot = audit_snapshot(ROOT, value["source"], value["config"])
    for n, h in value["source"]["snapshot_file_hashes"].items():
        if sha(ROOT/n) != h:
            raise ValueError("frozen supplement source changed: "+n)
    if calculate(value["config"]) != value["summary"]:
        raise ValueError("deterministic formula replay failed")
    result = {"status": "DETERMINISTIC_BOUND_ALL_48_ROWS_AND_SOURCE_REPLAY_PASSED",
        "source": snapshot, "result_sha256": sha(path/"result.json"), "rows_recomputed": 48,
        "relative_bias_certificate_obtained": False}
    write(path/"audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "audit"))
    mode = parser.parse_args().mode
    value = run() if mode == "run" else audit()
    print(json.dumps({"mode": mode, "status": value.get("status", "DETERMINISTIC_48_ROW_SCAN_COMPLETED")}))
