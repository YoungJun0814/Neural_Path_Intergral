import copy
from pathlib import Path

import pytest
import yaml

from experiments.post_audit_v2_reference_coverage_joint import (
    comparisons,
    identifier,
    identities,
    seed,
    summarize,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sealed_config_and_seed_grid():
    c = yaml.safe_load((ROOT/"configs/post_audit/reference_coverage_joint_v1.yaml").read_text())
    validate_config(c)
    for key, value in (("variance", 9.), ("unit_count", 100), ("replications", 8.0), ("torch_threads", True)):
        changed = copy.deepcopy(c)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)
    rows = [{"cell": {"id": cell}, "parent_training_rep": rep} for cell in ("a", "b") for rep in range(5)]
    jobs = identities(rows)
    assert len(jobs) == 640 and len(set(jobs)) == 640
    seeds = [seed(identifier(j), stream) for j in jobs for stream in ("path", "label")]
    assert len(set(seeds)) == len(seeds)


def test_summary_and_cost_gate_do_not_qualify_production():
    records = [{"family": "volterra", "cell": "a", "method": method,
        "replication": rep, "logs": {"mu": [-5.]*512, "risk": [-10.]*512},
        "geometry_bin": [0, 1, 2, 3]*128, "evaluation_wall": .1}
        for method in ("guide", "local-scale") for rep in range(8)]
    summary = summarize(records)
    for row in summary:
        assert row["geometry_contribution_shares"] == pytest.approx([.25]*4)
        assert row["summary"]["count"] == 4096
    comparisons_ = comparisons(summary)
    assert len(comparisons_) == 2
    assert all(c["accuracy_eligible_for_development_cost_comparison"] for c in comparisons_)
    assert all(c["fixed_precision_winner"] is None and not c["production_reference_qualified"] for c in comparisons_)
    summary[0]["accuracy_development_pass"] = False
    assert not comparisons(summary)[0]["accuracy_eligible_for_development_cost_comparison"]


def test_audit_applies_sealed_thread_setting_before_reduction(tmp_path, monkeypatch):
    import json

    from experiments import post_audit_v2_reference_coverage_joint as runner

    c = yaml.safe_load((ROOT/"configs/post_audit/reference_coverage_joint_v1.yaml").read_text())
    path = tmp_path/"result.json"
    path.write_text(json.dumps({"config": c, "source": {}}))
    calls = []
    monkeypatch.setattr(runner.torch, "set_num_threads", calls.append)
    def stop_before_io(*args):
        assert calls == [1]
        raise RuntimeError("bounded test stop")
    monkeypatch.setattr(runner, "audit_snapshot", stop_before_io)
    with pytest.raises(RuntimeError, match="bounded test stop"):
        runner.audit(path)
