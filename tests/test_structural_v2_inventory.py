"""Bounded JSON reader and immutable snapshot contracts, with malformed inputs."""

import copy
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.structural_v2_inventory import audit_snapshot, file_sha256, stream_evidence


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 32])
def test_stream_records_with_escaped_strings_and_numeric_boundaries(tmp_path: Path, compressed: bool, chunk: int) -> None:
    payload = {"schema": "test", "records": [{"s": 'x\\\"{}[]', "n": 12345, "e": 1e-12},
                                           {"s": "한국어", "data": [True, None]}], "total": 1.23e12}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path = tmp_path/("test.json.gz" if compressed else "test.json")
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    events = list(stream_evidence(path, chunk_size=chunk))
    assert [x for name, x in events if name == "record"] == payload["records"]
    assert dict((k, v) for k, v in events if k != "record")["total"] == payload["total"]
    assert file_sha256(path, decompressed=compressed) == (hashlib.sha256(raw).hexdigest(), len(raw))


@pytest.mark.parametrize("bad", [
    '{"records": [{"x":1,"x":2}]}', '{"records":[],"records":[]}',
    '{"records":[NaN]}', '{"records":[1]}', '{"records": [{"x": 1}',
    '{"records":[]} garbage', '{"records":[{},]}',
])
def test_stream_rejects_malformed_or_duplicate_json(tmp_path: Path, bad: str) -> None:
    path = tmp_path/"bad.json"
    path.write_text(bad)
    with pytest.raises(ValueError):
        list(stream_evidence(path, chunk_size=3))


def test_bounded_value_rejects_unlimited_record(tmp_path: Path) -> None:
    path = tmp_path/"big.json"
    path.write_text(json.dumps({"records": [{"large": "x"*1000}]}))
    with pytest.raises(ValueError, match="bounded-memory"):
        list(stream_evidence(path, maximum_value_chars=100, chunk_size=10))


def test_snapshot_binding_rejects_extra_members_and_config_changes(tmp_path: Path) -> None:
    path = tmp_path/"snapshot.zip"
    config = {"example": 1}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("source.py", b"immutable")
        archive.writestr("RUN_CONFIG.json", json.dumps(config))
    source = {"config_digest": canonical_digest(config), "snapshot_path": "snapshot.zip",
              "snapshot_sha256": file_sha256(path)[0],
              "snapshot_file_hashes": {"source.py": hashlib.sha256(b"immutable").hexdigest()}}
    assert audit_snapshot(tmp_path, source, config)["entries"] == 1
    with pytest.raises(ValueError, match="configuration"):
        audit_snapshot(tmp_path, source, {"example": 2})
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("extra", b"undeclared")
    source["snapshot_sha256"] = file_sha256(path)[0]
    with pytest.raises(ValueError, match="completion"):
        audit_snapshot(tmp_path, source, config)


def test_p0_runner_zero_sampling_budget_and_stale_gate_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import post_audit_v2_p0_inventory as runner

    document = tmp_path/"plan.md"
    document.write_text("frozen plan")
    config = {"schema": "npi.structural-v2.p0-config.v1", "historical_artifacts": ["old.json"],
              "evidence_documents": ["plan.md"], "budget": {"max_wall_seconds": 60,
              "max_process_rss_bytes": 2**40, "max_new_target_samples": 0,
              "max_potential_evaluations": 0, "max_new_model_candidates": 0,
              "max_whole_training_runs": 0}}
    source = {"source_tree_digest": "frozen", "config_digest": canonical_digest(config)}
    monkeypatch.setattr(runner, "source_tree_digest", lambda root: "frozen")
    monkeypatch.setattr(runner, "audit_snapshot", lambda *args: {})
    monkeypatch.setattr(runner, "scan_legacy", lambda root, path, **kwargs:
                        {"path": path, "current_runtime_source_matches": False})
    result = runner.run(config, source, root=tmp_path)
    assert result["new_target_samples"] == 0
    assert runner.audit(result, root=tmp_path)["historical_artifacts"] == 1
    for change in ("gate", "sampling", "rules", "documents", "row", "mismatch_count"):
        altered = copy.deepcopy(result)
        if change == "gate":
            altered["scientific_gates"]["independent_reference"] = "pass"
        elif change == "sampling":
            altered["new_target_samples"] = 1
        elif change == "rules":
            altered["measurement_rules"][0]["se_unit"] = "smc_particle"
        elif change == "documents":
            altered["documents"] = []
        elif change == "row":
            altered["records"][0]["current_runtime_source_matches"] = True
        else:
            altered["historical_source_mismatches"] = 0
        with pytest.raises(ValueError):
            runner.audit(altered, root=tmp_path)
    config["budget"]["max_new_target_samples"] = 1
    source["config_digest"] = canonical_digest(config)
    with pytest.raises(ValueError, match="must not authorize"):
        runner.run(config, source, root=tmp_path)


def test_p0_inventory_failure_stays_unresolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from experiments import post_audit_v2_p0_inventory as runner

    config = {"schema": "npi.structural-v2.p0-config.v1", "historical_artifacts": ["missing.json"],
              "evidence_documents": [], "budget": {"max_wall_seconds": 60,
              "max_process_rss_bytes": 2**40, "max_new_target_samples": 0,
              "max_potential_evaluations": 0, "max_new_model_candidates": 0,
              "max_whole_training_runs": 0}}
    source = {"source_tree_digest": "frozen", "config_digest": canonical_digest(config)}
    monkeypatch.setattr(runner, "source_tree_digest", lambda root: "frozen")
    result = runner.run(config, source, root=tmp_path)
    assert not result["p0_binding_grid_complete"]
    assert result["status"] == "unresolved"
    assert result["records"][0]["failure_reason"]
