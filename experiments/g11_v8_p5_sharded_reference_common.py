"""Shared validation and actual rBergomi draws for the V8 sharded reference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from experiments.g11_v8_p5_reference import _draw_values
from experiments.g11_v8_p5_threshold_binding_v2_audit import (
    ROOT,
    audit_binding_v2,
    load_binding_v2,
)
from experiments.g11_v8_p7_calibration import load_config
from src.path_integral.provenance import runtime_provenance
from src.path_integral.reference_execution import (
    ReferenceBatch,
    execute_reference_shard,
)
from src.path_integral.reference_protocol import (
    REFERENCE_METHODS,
    ReferenceMethod,
    ReferenceShardIdentity,
    canonical_sha256,
    seed_key_sha256,
)
from src.path_integral.seed_ledger import SeedKey, SeedLedger, derive_seed
from src.physics_engine import RBergomiSimulator

SCHEMA = "npi.g11.v8-p5-sharded-reference-execution.v3"
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "phase",
    "design_informed_by_prior_development_outcomes",
    "current_namespace_outcomes_inspected_before_freeze",
    "threshold_binding",
    "reference_contract",
    "sampling",
    "benchmark",
    "decision",
}


@dataclass(frozen=True)
class ShardedReferenceContext:
    config: dict[str, Any]
    config_sha256: str
    binding: dict[str, Any]
    binding_sha256: str
    threshold_config: dict[str, Any]
    threshold_result: dict[str, Any]
    cells_by_id: dict[str, dict[str, Any]]
    environment: dict[str, Any]
    environment_sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(record: Any) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("bound artifact record is malformed")
    relative = record.get("path")
    if not isinstance(relative, str):
        raise ValueError("bound artifact path must be a string")
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise ValueError("bound artifact path is outside the repository or absent")
    if record.get("sha256") != _sha256(path):
        raise ValueError("bound artifact SHA-256 mismatch")
    return path


def load_sharded_reference_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if (
        not isinstance(config, dict)
        or config.get("schema") != SCHEMA
        or set(config) != ROOT_KEYS
    ):
        raise ValueError("unexpected or malformed sharded-reference V3 config")
    if (
        config.get("protocol_id") != "g11-v8-p5-sharded-reference-development-v1"
        or config.get("phase") != "r2_reference_development"
        or config.get("design_informed_by_prior_development_outcomes") is not True
        or config.get("current_namespace_outcomes_inspected_before_freeze") is not False
    ):
        raise ValueError("sharded-reference provenance flags are invalid")
    contract = config.get("reference_contract")
    sampling = config.get("sampling")
    benchmark = config.get("benchmark")
    decision = config.get("decision")
    if not all(
        isinstance(value, dict)
        for value in (contract, sampling, benchmark, decision)
    ):
        raise ValueError("sharded-reference sections must be mappings")
    assert isinstance(contract, dict)
    assert isinstance(sampling, dict)
    assert isinstance(benchmark, dict)
    assert isinstance(decision, dict)
    if (
        contract.get("estimand") != "fixed_finest_grid"
        or contract.get("continuous_time_claim") is not False
        or contract.get("methods") != list(REFERENCE_METHODS)
        or contract.get("dtype") != "float64"
        or contract.get("device") != "cpu"
        or float(contract.get("final_relative_rmse_design_target", 0.0)) != 0.20
        or float(
            contract.get("maximum_reference_se_fraction_of_final_target", 0.0)
        )
        != 0.10
        or float(contract.get("maximum_combined_z_score", 0.0)) != 4.0
        or contract.get("independent_method_streams_required") is not True
        or contract.get("ordinary_mean_required") is not True
        or contract.get("self_normalization_allowed") is not False
    ):
        raise ValueError("sharded-reference statistical contract is invalid")
    if (
        sampling.get("pilot_namespace") != "v8-r2-reference-development"
        or sampling.get("final_namespace") != "v8-r2-reference-development-final"
        or int(sampling.get("pilot_replicates", 0)) != 8
        or int(sampling.get("pilot_samples_per_replicate", 0)) != 32768
        or int(sampling.get("minimum_final_samples", 0)) != 8192
        or int(sampling.get("maximum_final_samples", 0)) != 8388608
        or int(sampling.get("final_chunk_size", 0)) != 4096
        or float(sampling.get("allocation_safety_factor", 0.0)) != 6.0
        or sampling.get("allocation_variance_statistic")
        != "maximum_replicate_variance"
        or sampling.get("engine") != "fft"
    ):
        raise ValueError("sharded-reference sampling contract is invalid")
    representative_cells = benchmark.get("representative_cells")
    if (
        not isinstance(representative_cells, list)
        or len(representative_cells) != 3
        or len(set(representative_cells)) != 3
        or benchmark.get("methods") != list(REFERENCE_METHODS)
        or int(benchmark.get("samples_per_observation", 0)) != 32768
        or int(benchmark.get("repetitions", 0)) != 1
        or int(benchmark.get("local_workers", 0)) != 1
        or int(benchmark.get("external_cpu_workers", 0)) != 2
        or int(benchmark.get("expected_torch_threads_per_worker", 0)) != 16
        or float(benchmark.get("local_total_memory_fraction", 0.0)) != 0.60
        or float(benchmark.get("forecast_safety_factor", 0.0)) != 2.0
    ):
        raise ValueError("representative benchmark contract is invalid")
    if (
        decision.get("representative_benchmark_authorized") is not True
        or decision.get("full_pilot_execution_authorized") is not False
        or decision.get("final_execution_authorized") is not False
        or decision.get("reference_complete") is not False
        or decision.get("performance_claim_authorized") is not False
        or decision.get("submission_authorized") is not False
    ):
        raise ValueError("sharded-reference decision must fail closed")
    return config, hashlib.sha256(raw).hexdigest()


def load_context(config_path: Path) -> ShardedReferenceContext:
    config, config_sha256 = load_sharded_reference_config(config_path)
    binding_path = _bound_path(config["threshold_binding"])
    binding, binding_sha256 = load_binding_v2(binding_path)
    binding_audit = audit_binding_v2(binding, binding_sha256)
    if not binding_audit["passed"]:
        raise ValueError(f"R2 threshold binding failed: {binding_audit['failures']}")
    protocol = binding["reference_protocol"]
    sampling = config["sampling"]
    if (
        binding_sha256 != config["threshold_binding"]["sha256"]
        or protocol["id"] != config["protocol_id"]
        or protocol["pilot_namespace"] != sampling["pilot_namespace"]
        or protocol["final_namespace"] != sampling["final_namespace"]
        or protocol["methods"] != config["reference_contract"]["methods"]
        or protocol["estimand"] != config["reference_contract"]["estimand"]
        or protocol["dtype"] != config["reference_contract"]["dtype"]
        or protocol["device"] != config["reference_contract"]["device"]
    ):
        raise ValueError("execution config and threshold binding disagree")
    threshold_config_path = _bound_path(binding["threshold_calibration_config"])
    threshold_result_path = _bound_path(binding["threshold_calibration_result"])
    threshold_config, threshold_config_sha256 = load_config(threshold_config_path)
    if threshold_config_sha256 != binding["threshold_calibration_config"]["sha256"]:
        raise ValueError("threshold proposal config hash mismatch")
    threshold_result_value = json.loads(
        threshold_result_path.read_text(encoding="utf-8")
    )
    if not isinstance(threshold_result_value, dict):
        raise ValueError("threshold calibration result must be a mapping")
    threshold_result = threshold_result_value
    manifest = threshold_result.get("candidate_manifest")
    if (
        not isinstance(manifest, dict)
        or canonical_sha256(manifest) != binding["threshold_manifest_sha256"]
    ):
        raise ValueError("threshold candidate manifest hash mismatch")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 24:
        raise ValueError("threshold manifest must contain exactly 24 cells")
    cells_by_id = {
        str(cell["cell_id"]): cell for cell in cells if isinstance(cell, dict)
    }
    if len(cells_by_id) != 24:
        raise ValueError("threshold manifest cell IDs must be unique")
    representative_cells = config["benchmark"]["representative_cells"]
    if not set(representative_cells).issubset(cells_by_id):
        raise ValueError("representative benchmark references an unknown cell")
    environment = runtime_provenance(dtype="torch.float64")
    return ShardedReferenceContext(
        config=config,
        config_sha256=config_sha256,
        binding=binding,
        binding_sha256=binding_sha256,
        threshold_config=threshold_config,
        threshold_result=threshold_result,
        cells_by_id=cells_by_id,
        environment=environment,
        environment_sha256=canonical_sha256(environment),
    )


def reference_seed_material(
    identity: ReferenceShardIdentity,
) -> tuple[str, int, int]:
    role = f"p5-reference-{identity.method}-{identity.stage}"
    seed_protocol = f"{identity.protocol_id}/{identity.namespace}"
    proposal_key = SeedKey(
        seed_protocol,
        role,
        identity.cell_id,
        "fixed-grid",
        0,
        identity.shard_index,
        "proposal",
    )
    label_key = SeedKey(
        seed_protocol,
        role,
        identity.cell_id,
        "fixed-grid",
        0,
        identity.shard_index,
        "labels",
    )
    proposal_seed = derive_seed(proposal_key)
    label_seed = derive_seed(label_key)
    digest = seed_key_sha256(
        {
            "identity": identity.to_dict(),
            "proposal_key": asdict(proposal_key),
            "proposal_seed": proposal_seed,
            "label_key": asdict(label_key),
            "label_seed": label_seed,
        }
    )
    return digest, proposal_seed, label_seed


def draw_actual_reference_batch(
    context: ShardedReferenceContext,
    identity: ReferenceShardIdentity,
    requested_samples: int,
) -> ReferenceBatch:
    cell = context.cells_by_id.get(identity.cell_id)
    if cell is None:
        raise ValueError("reference shard requested an unknown cell")
    if identity.method not in REFERENCE_METHODS:
        raise ValueError("reference shard requested an unsupported method")
    simulator = RBergomiSimulator(
        H=float(cell["hurst"]),
        eta=float(cell["eta"]),
        xi=float(cell["xi"]),
        rho=float(cell["rho"]),
        device="cpu",
    )
    ledger = SeedLedger()
    values, normalization = _draw_values(
        threshold_config=context.threshold_config,
        simulator=simulator,
        cell=cell,
        method=cast(ReferenceMethod, identity.method),
        stage=identity.stage,
        replicate=identity.shard_index,
        count=requested_samples,
        ledger=ledger,
        protocol_id=f"{identity.protocol_id}/{identity.namespace}",
    )
    expected_digest, proposal_seed, label_seed = reference_seed_material(identity)
    actual_seeds = {record.seed for record in ledger.records}
    if actual_seeds != {proposal_seed, label_seed} or len(ledger) != 2:
        raise RuntimeError("actual reference draw used an unexpected seed family")
    if expected_digest != reference_seed_material(identity)[0]:
        raise RuntimeError("reference seed material is not deterministic")
    return ReferenceBatch(
        contribution=values.detach().to(device="cpu", dtype=torch.float64),
        likelihood_normalization=normalization.detach().to(
            device="cpu", dtype=torch.float64
        ),
    )


def execute_actual_reference_shard(
    context: ShardedReferenceContext,
    identity: ReferenceShardIdentity,
    *,
    requested_samples: int,
    parent_sha256: str,
    source_commit: str,
    dirty_worktree: bool,
    benchmark: bool = False,
) -> dict[str, Any]:
    sampling = context.config["sampling"]
    expected_namespace = (
        context.config["benchmark"]["namespace"]
        if benchmark
        else (
            sampling["pilot_namespace"]
            if identity.stage == "pilot"
            else sampling["final_namespace"]
        )
    )
    if identity.protocol_id != context.config["protocol_id"]:
        raise ValueError("reference shard protocol does not match execution config")
    if identity.namespace != expected_namespace:
        raise ValueError("reference shard namespace does not match its execution stage")
    seed_digest, _, _ = reference_seed_material(identity)
    return execute_reference_shard(
        identity=identity,
        config_sha256=context.config_sha256,
        threshold_manifest_sha256=context.binding["threshold_manifest_sha256"],
        parent_sha256=parent_sha256,
        source_commit=source_commit,
        dirty_worktree=dirty_worktree,
        environment_sha256=context.environment_sha256,
        seed_key_sha256=seed_digest,
        requested_samples=requested_samples,
        draw=lambda: draw_actual_reference_batch(
            context, identity, requested_samples
        ),
    )
