"""Independent audit of corrected V10R1 development/qualification artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.g11_v10r1_terminal_benchmark import load_config, validate_config
from src.path_integral.v10r1_protocol import (
    aggregate_v10r1,
    attach_v10r1_work,
    semantic_equal,
)


@dataclass(frozen=True)
class V10R1BenchmarkAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    recomputed_aggregate: dict[str, Any]
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(root: Path, commit: str, path: str) -> bytes | None:
    try:
        return subprocess.check_output(("git", "show", f"{commit}:{path}"), cwd=root)
    except (OSError, subprocess.CalledProcessError):
        return None


def _expected_decision(stage: str, stage_pass: bool) -> dict[str, bool]:
    development = stage == "development"
    return {
        "development_complete": development,
        "qualification_authorized": development and stage_pass,
        "qualification_complete": not development,
        "regime_conditional_empirical_claim_authorized": not development and stage_pass,
        "broad_performance_claim_authorized": False,
        "top_journal_claim_authorized": False,
        "submission_authorized": False,
    }


def audit_v10r1_benchmark(
    *, config_path: Path, result: dict[str, Any], root: Path
) -> V10R1BenchmarkAudit:
    config, config_hash = load_config(config_path)
    config["config_path"] = config_path.resolve().relative_to(root).as_posix()
    validate_config(config, root=root)
    paired = result.get("paired_records")
    external = result.get("external_records")
    if not isinstance(paired, list) or not isinstance(external, list):
        raise ValueError("V10R1 result records must be lists")
    bindings_ok = all(
        (root / str(binding["path"])).is_file()
        and _sha256(root / str(binding["path"])) == str(binding["sha256"])
        for binding in config["bindings"].values()
    )
    bank = json.loads(
        (root / str(config["bindings"]["proposal_bank"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    bank_index = {
        (str(entry["cell_id"]), int(entry["replicate"])): entry
        for entry in bank["entries"]
    }
    raw_paired = deepcopy(paired)
    raw_external = deepcopy(external)
    for record in raw_paired:
        record.pop("work_to_target", None)
        record.pop("comparison", None)
    for record in raw_external:
        record.pop("work_to_target", None)
    expected_paired, expected_external = attach_v10r1_work(
        config=config,
        paired_records=raw_paired,
        external_records=raw_external,
        bank=bank,
    )
    aggregate = aggregate_v10r1(
        config=config,
        paired_records=expected_paired,
        external_records=expected_external,
    )

    seeds: list[int] = []
    for record in paired:
        seeds.extend((int(record["gaussian_seed"]), int(record["label_seed"])))
    for record in external:
        seeds.extend(int(value) for value in record["seeds"].values())
    expected_seeds = list(
        range(int(config["base_seed"]), int(config["base_seed"]) + len(seeds))
    )
    seed_hash = hashlib.sha256(
        json.dumps(seeds, separators=(",", ":")).encode()
    ).hexdigest()
    bound_seeds = {int(value) for value in bank["training_seeds"]}
    reference = json.loads(
        (root / str(config["bindings"]["reference"]["path"])).read_text(encoding="utf-8")
    )
    for cell in reference["cells"]:
        bound_seeds.add(int(cell["proposal_seed"]))
        bound_seeds.update(int(value) for value in cell["randomization_seeds"])

    proposal_binding = all(
        (
            str(record["cell_id"]),
            int(record["proposal_replicate"]),
        )
        in bank_index
        and record["proposal_sha256"]
        == bank_index[
            (str(record["cell_id"]), int(record["proposal_replicate"]))
        ]["proposal"]["sha256"]
        and int(record["proposal_training_seed"])
        == int(
            bank_index[
                (str(record["cell_id"]), int(record["proposal_replicate"]))
            ]["training_seed"]
        )
        for record in paired
    )
    source_commit = str(result.get("source_commit", ""))
    source_config = _git_blob(root, source_commit, str(result.get("config_path", "")))
    source_hash = hashlib.sha256(source_config).hexdigest() if source_config else ""
    cells = {str(record["cell_id"]) for record in paired}
    expected_paired_count = len(cells) * int(config["clusters"])
    expected_external_count = expected_paired_count * len(
        config["external_methods"]["primary"]
    )
    expected_decision = _expected_decision(str(config["stage"]), bool(aggregate["stage_pass"]))
    checks = (
        (
            "schema",
            result.get("schema") == "npi.g11.v10r1-terminal-benchmark-result.v1",
        ),
        ("config_hash", result.get("config_sha256") == config_hash),
        ("bindings", bindings_ok),
        ("clean_source", result.get("dirty_worktree") is False),
        ("source_contains_exact_config", source_hash == config_hash),
        ("stage", result.get("stage") == config.get("stage")),
        ("paired_record_count", len(paired) == expected_paired_count),
        ("external_record_count", len(external) == expected_external_count),
        ("seed_uniqueness", len(seeds) == len(set(seeds))),
        ("seed_contiguity", seeds == expected_seeds),
        ("seed_disjoint_from_bound_artifacts", not (set(seeds) & bound_seeds)),
        ("seed_count", result.get("seed_count") == len(seeds)),
        ("seed_hash", result.get("seed_set_sha256") == seed_hash),
        ("proposal_bank_binding", proposal_binding),
        (
            "all_3n_exactness_fields_present",
            all(
                set(record["exactness"])
                == {
                    "maximum_local_latent_reconstruction_error",
                    "maximum_price_latent_reconstruction_error",
                    "maximum_path_reconstruction_error",
                    "maximum_coordinate_error",
                    "maximum_component_density_error",
                    "maximum_mixture_density_error",
                    "maximum_full_likelihood_error",
                    "maximum_full_bound_violation",
                    "maximum_residual_bound_violation",
                }
                for record in paired
            ),
        ),
        (
            "work_reconstruction",
            semantic_equal(paired, expected_paired)
            and semantic_equal(external, expected_external),
        ),
        ("aggregate_recomputation", semantic_equal(result.get("aggregate"), aggregate)),
        ("decision_locks", result.get("decision") == expected_decision),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return V10R1BenchmarkAudit(
        checks=checks,
        failures=failures,
        recomputed_aggregate=aggregate,
        passed=not failures,
    )
