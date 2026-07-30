from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import experiments.g11_v8_p5_reference_freeze_allocation as freeze_module
import experiments.g11_v8_p5_reference_pilot as pilot_module
import experiments.g11_v8_p5_reference_shard as final_module
from experiments.g11_v8_p5_reference_aggregate import aggregate_reference
from experiments.g11_v8_p5_reference_result_audit import audit_reference_result
from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context
from src.path_integral.reference_protocol import (
    REFERENCE_METHODS,
    ReferenceShardIdentity,
    SufficientStatistics,
    build_shard_artifact,
    seed_key_sha256,
)
from src.path_integral.reference_shards import write_shard_atomic

CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v3.yaml"


def _head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()


def _fake_artifact(
    context: Any,
    identity: ReferenceShardIdentity,
    *,
    requested_samples: int,
    parent_sha256: str,
    source_commit: str,
    dirty_worktree: bool,
    benchmark: bool = False,
) -> dict[str, Any]:
    del benchmark
    nominal = float(context.cells_by_id[identity.cell_id]["nominal_probability"])
    variance = 1.0e-12
    return build_shard_artifact(
        identity=identity,
        config_sha256=context.config_sha256,
        threshold_manifest_sha256=context.binding["threshold_manifest_sha256"],
        parent_sha256=parent_sha256,
        source_commit=source_commit,
        dirty_worktree=dirty_worktree,
        environment_sha256=context.environment_sha256,
        estimand="fixed_finest_grid",
        dtype="float64",
        device="cpu",
        seed_key_sha256=seed_key_sha256(identity.to_dict()),
        requested_samples=requested_samples,
        contribution=SufficientStatistics(
            requested_samples,
            nominal,
            (requested_samples - 1) * variance,
        ),
        likelihood_normalization=SufficientStatistics(
            requested_samples,
            1.0,
            0.0,
        ),
        invalid_spot_count=0,
        invalid_variance_count=0,
        nonfinite_contribution_count=0,
        elapsed_wall_seconds=0.01,
        elapsed_cpu_seconds=0.01,
        peak_resident_memory_bytes=1024,
    )


@pytest.mark.slow
def test_complete_sharded_reference_orchestration_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {"source_commit": _head(), "dirty_worktree": False}
    monkeypatch.setattr(pilot_module, "source_provenance", lambda: source)
    monkeypatch.setattr(
        pilot_module, "_verify_authorization", lambda _path, _context: {}
    )
    monkeypatch.setattr(freeze_module, "source_provenance", lambda: source)
    monkeypatch.setattr(final_module, "source_provenance", lambda: source)
    monkeypatch.setattr(
        pilot_module, "execute_actual_reference_shard", _fake_artifact
    )
    monkeypatch.setattr(
        final_module, "execute_actual_reference_shard", _fake_artifact
    )

    pilot_directory = tmp_path / "pilots"
    first = pilot_module.run_pilots(
        CONFIG,
        tmp_path / "synthetic-authorization.yaml",
        pilot_directory,
        cell_id="h0.12-terminal_left_tail-p1e-02",
        method="dcs_reference",
        replicate=0,
    )
    second = pilot_module.run_pilots(
        CONFIG,
        tmp_path / "synthetic-authorization.yaml",
        pilot_directory,
        cell_id="h0.12-terminal_left_tail-p1e-02",
        method="dcs_reference",
        replicate=0,
    )
    assert first["executed"] == 1 and first["skipped"] == 0
    assert second["executed"] == 0 and second["skipped"] == 1

    context = load_context(CONFIG)
    sampling = context.config["sampling"]
    for cell_id in context.cells_by_id:
        for method in REFERENCE_METHODS:
            for replicate in range(int(sampling["pilot_replicates"])):
                identity = ReferenceShardIdentity(
                    context.config["protocol_id"],
                    sampling["pilot_namespace"],
                    "pilot",
                    method,
                    cell_id,
                    replicate,
                )
                if (
                    cell_id == "h0.12-terminal_left_tail-p1e-02"
                    and method == "dcs_reference"
                    and replicate == 0
                ):
                    continue
                artifact = _fake_artifact(
                    context,
                    identity,
                    requested_samples=int(
                        sampling["pilot_samples_per_replicate"]
                    ),
                    parent_sha256=context.binding_sha256,
                    source_commit=source["source_commit"],
                    dirty_worktree=False,
                )
                write_shard_atomic(pilot_directory, artifact)

    manifest_path = tmp_path / "allocation.json"
    manifest, manifest_sha256 = freeze_module.freeze_allocation(
        CONFIG, pilot_directory, manifest_path
    )
    assert manifest["final_execution_authorized"] is True
    assert len(manifest["entries"]) == 48

    final_directory = tmp_path / "final"
    first_entry = manifest["entries"][0]
    first_chunk = first_entry["chunks"][0]
    first_identity = ReferenceShardIdentity.from_dict(first_chunk["identity"])
    first_final = final_module.run_final_shards(
        CONFIG,
        manifest_path,
        final_directory,
        cell_id=first_identity.cell_id,
        method=first_identity.method,
        shard_index=first_identity.shard_index,
    )
    resumed_final = final_module.run_final_shards(
        CONFIG,
        manifest_path,
        final_directory,
        cell_id=first_identity.cell_id,
        method=first_identity.method,
        shard_index=first_identity.shard_index,
    )
    assert first_final["executed"] == 1
    assert resumed_final["skipped"] == 1

    for entry in manifest["entries"]:
        for chunk in entry["chunks"]:
            identity = ReferenceShardIdentity.from_dict(chunk["identity"])
            if identity.shard_id == first_identity.shard_id:
                continue
            artifact = _fake_artifact(
                context,
                identity,
                requested_samples=int(chunk["requested_samples"]),
                parent_sha256=manifest_sha256,
                source_commit=source["source_commit"],
                dirty_worktree=False,
            )
            write_shard_atomic(final_directory, artifact)

    aggregate_path = tmp_path / "aggregate.json"
    aggregate, _ = aggregate_reference(
        CONFIG,
        manifest_path,
        final_directory,
        aggregate_path,
    )
    assert aggregate["reference_acceptance_pass"] is True
    assert len(aggregate["cells"]) == 48
    assert len(aggregate["method_agreements"]) == 24

    report = audit_reference_result(
        CONFIG,
        manifest_path,
        final_directory,
        aggregate_path,
    )
    assert report["passed"] is True
    assert report["decision"]["development_reference_complete"] is True
    assert report["decision"]["fresh_qualification_reference_required"] is True
    assert report["decision"]["performance_claim_authorized"] is False
