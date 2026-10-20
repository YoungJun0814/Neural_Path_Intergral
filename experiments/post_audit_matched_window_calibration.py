"""Four fixed large-grid chunks: numeric equivalence/resource preflight only."""

from __future__ import annotations

import json
import time
from typing import Any

import psutil
import torch

from experiments.post_audit_bridge_mutation_allocation import DIRECTORY as ALLOCATION_DIRECTORY
from experiments.post_audit_bridge_mutation_allocation import load_anchors
from experiments.post_audit_matched_window_mechanism import PLAN, THEORY, prepare
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_sleep_recovery import ROOT, sha
from src.path_integral.gaussian_span_marginalization import sample_gaussian_mixture
from src.path_integral.matched_window_diagnostics import (
    contribution_diagnostics,
    paired_contributions_with_features,
)
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

DIRECTORY = "results/post_audit/matched_window_calibration_v1/session_20261019"


def run() -> dict[str, Any]:
    torch.set_num_threads(8)
    original, _ = load_anchors()
    ap, receipt_path = ALLOCATION_DIRECTORY+"/result.json", ALLOCATION_DIRECTORY+"/audit.json"
    allocation, receipt = json.loads((ROOT/ap).read_text()), json.loads((ROOT/receipt_path).read_text())
    audit_snapshot(ROOT, allocation["source"], allocation["config"])
    if receipt["replayed_wholes"] != 4 or receipt["result_sha256"] != sha(ROOT/ap):
        raise ValueError("allocation audit required before calibration")
    config: dict[str, Any] = {"original": original, "fixture_seeds": [26900, 26901, 26902, 26903],
              "anchors": {p: sha(ROOT/p) for p in (ap, receipt_path)}, "torch_threads": 8}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, THEORY, ap, receipt_path))
    started = time.perf_counter()
    prepared = prepare(config)
    prep_wall = time.perf_counter()-started
    guide, pair = prepared[(original["cells"][1]["id"], "B-price-only-window")]
    rows = []
    with torch.no_grad():
        for seed in config["fixture_seeds"]:
            sample = sample_gaussian_mixture(guide.spec, 512, gaussian_generator=torch.Generator().manual_seed(seed*10),
                                            label_generator=torch.Generator().manual_seed(seed*10+1))
            eps = torch.randn((512, 2048), dtype=torch.float64, generator=torch.Generator().manual_seed(seed*10+2))
            begin = time.perf_counter()
            values, features = paired_contributions_with_features(pair, guide, sample.target_coordinates, eps, sample.labels)
            contribution_diagnostics(values[:, 0], sample.labels, features, components=373)
            wall = time.perf_counter()-begin
            oracle = pair.log_components(torch.cat((sample.target_coordinates, eps), 1))-guide.log_q_over_p(sample.target_coordinates)[:, None]
            torch.testing.assert_close(values, oracle, rtol=1e-12, atol=1e-12)
            rows.append({"feature_chunk_wall": wall, "maximum_absolute_log_difference": float((values-oracle).abs().max()),
                         "rss": psutil.Process().memory_info().rss})
    passed = prep_wall < 60 and max(x["feature_chunk_wall"] for x in rows)*64 < 180 and max(x["rss"] for x in rows) < 4*1024**3
    output = {"config": config, "source": source, "rows": rows, "preparation_wall": prep_wall,
              "physical": 8192, "density": 4096, "timestep_forward": 25165824,
              "pass": passed, "interpretation": "four_fixed_chunks_with_2x_time_safety_not_a_whole_runtime_or_accuracy_guarantee"}
    write_json_atomic_nonoverwriting(directory/"result.json", output)
    if not passed:
        raise RuntimeError("fixed matched-window resource preflight failed")
    return output


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k in ("pass", "physical", "preparation_wall", "rows")}))
