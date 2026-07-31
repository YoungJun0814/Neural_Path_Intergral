"""Pilot allocation and exact aggregation for sharded independent references."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from .reference_protocol import (
    ALLOCATION_SCHEMA,
    REFERENCE_METHODS,
    ReferenceShardIdentity,
    SufficientStatistics,
    canonical_sha256,
    validate_shard_artifact,
)


def _merge_all(statistics: Iterable[SufficientStatistics]) -> SufficientStatistics:
    iterator = iter(statistics)
    try:
        merged = next(iterator)
    except StopIteration as error:
        raise ValueError("at least one sufficient-statistic record is required") from error
    for item in iterator:
        merged = merged.merge(item)
    return merged


def _chunk_counts(total: int, chunk_size: int) -> list[int]:
    if total < 2 or chunk_size < 2:
        raise ValueError("final count and chunk size must be at least two")
    full_chunks, remainder = divmod(total, chunk_size)
    counts = [chunk_size] * full_chunks
    if remainder == 1:
        if not counts or counts[-1] <= 2:
            raise ValueError("final allocation cannot avoid a singleton chunk")
        counts[-1] -= 1
        remainder = 2
    if remainder:
        counts.append(remainder)
    if sum(counts) != total or any(count < 2 for count in counts):
        raise RuntimeError("final chunk partition is internally inconsistent")
    return counts


def build_allocation_manifest(
    *,
    protocol_id: str,
    config_sha256: str,
    threshold_manifest_sha256: str,
    pilot_parent_sha256: str,
    pilot_namespace: str,
    final_namespace: str,
    expected_cells: Sequence[str],
    expected_methods: Sequence[str],
    pilot_replicates: int,
    pilot_shards: Sequence[tuple[dict[str, Any], str]],
    target_standard_errors: Mapping[str | tuple[str, str], float],
    allocation_safety_factor: float,
    minimum_final_samples: int,
    maximum_final_samples: int,
    final_chunk_size: int,
    source_commit: str,
    environment_sha256: str,
    estimand: str,
    dtype: str,
    device: str,
    design_informed_by_prior_development_outcomes: bool,
    current_namespace_outcomes_inspected_before_freeze: bool,
    maximum_final_samples_by_method: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Freeze final counts using only complete independent pilot shards."""

    if estimand != "fixed_finest_grid":
        raise ValueError("R1 supports only the fixed-finest-grid estimand")
    if dtype != "float64" or device != "cpu":
        raise ValueError("R1 supports only CPU/float64 reference execution")
    if pilot_namespace == final_namespace:
        raise ValueError("pilot and final namespaces must be distinct")
    if pilot_replicates < 3:
        raise ValueError("at least three pilot replicates are required")
    if (
        not math.isfinite(allocation_safety_factor)
        or allocation_safety_factor < 1.0
    ):
        raise ValueError("allocation safety factor must be finite and at least one")
    if minimum_final_samples < 2 or maximum_final_samples < minimum_final_samples:
        raise ValueError("invalid final-sample bounds")
    if set(expected_methods) != set(REFERENCE_METHODS):
        raise ValueError("reference method roster must remain fixed")
    if len(set(expected_cells)) != len(expected_cells) or not expected_cells:
        raise ValueError("expected reference cells must be unique and nonempty")
    expected_entry_keys = {
        (cell, method) for cell in expected_cells for method in expected_methods
    }
    target_keys = set(target_standard_errors)
    cell_targets = target_keys == set(expected_cells)
    entry_targets = target_keys == expected_entry_keys
    if not cell_targets and not entry_targets:
        raise ValueError(
            "target standard errors must cover either the exact cell set "
            "or the exact cell-method matrix"
        )
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in target_standard_errors.values()
    ):
        raise ValueError("target standard errors must be finite and positive")
    resolved_caps = (
        {method: maximum_final_samples for method in expected_methods}
        if maximum_final_samples_by_method is None
        else dict(maximum_final_samples_by_method)
    )
    if (
        set(resolved_caps) != set(expected_methods)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum_final_samples
            or value > maximum_final_samples
            for value in resolved_caps.values()
        )
    ):
        raise ValueError("method-specific final-sample caps are invalid")
    if current_namespace_outcomes_inspected_before_freeze:
        raise ValueError("current final namespace outcomes must remain unopened")

    by_key: dict[tuple[str, str, int], tuple[dict[str, Any], str]] = {}
    pilot_seed_keys: set[str] = set()
    for payload, digest in pilot_shards:
        validate_shard_artifact(payload)
        if canonical_sha256(payload) != digest:
            raise ValueError("pilot shard digest mismatch")
        identity = ReferenceShardIdentity.from_dict(payload["identity"])
        if identity.protocol_id != protocol_id or identity.namespace != pilot_namespace:
            raise ValueError("pilot shard protocol or namespace mismatch")
        if identity.stage != "pilot":
            raise ValueError("allocation input must contain only pilot shards")
        if payload["config_sha256"] != config_sha256:
            raise ValueError("pilot shard config hash mismatch")
        if payload["threshold_manifest_sha256"] != threshold_manifest_sha256:
            raise ValueError("pilot shard threshold hash mismatch")
        if payload["parent_sha256"] != pilot_parent_sha256:
            raise ValueError("pilot shard parent-binding mismatch")
        if payload["source_commit"] != source_commit:
            raise ValueError("pilot shard source commit mismatch")
        if payload["environment_sha256"] != environment_sha256:
            raise ValueError("pilot shard environment hash mismatch")
        if payload["estimand"] != estimand:
            raise ValueError("pilot shard estimand mismatch")
        if payload["dtype"] != dtype or payload["device"] != device:
            raise ValueError("pilot shard dtype or device mismatch")
        if payload["dirty_worktree"] is not False:
            raise ValueError("formal allocation requires clean-source pilot shards")
        if any(
            payload[field] != 0
            for field in (
                "invalid_spot_count",
                "invalid_variance_count",
                "nonfinite_contribution_count",
            )
        ):
            raise ValueError("pilot shard contains invalid paths or contributions")
        seed_key = payload["seed_key_sha256"]
        if seed_key in pilot_seed_keys:
            raise ValueError("duplicate pilot seed key")
        pilot_seed_keys.add(seed_key)
        key = (identity.cell_id, identity.method, identity.shard_index)
        if key in by_key:
            raise ValueError(f"duplicate pilot shard identity: {key}")
        by_key[key] = (payload, digest)

    expected_keys = {
        (cell, method, replicate)
        for cell in expected_cells
        for method in expected_methods
        for replicate in range(pilot_replicates)
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        unexpected = sorted(set(by_key) - expected_keys)
        raise ValueError(
            f"pilot shard set mismatch: missing={missing}, unexpected={unexpected}"
        )

    entries: list[dict[str, Any]] = []
    all_feasible = True
    for cell in expected_cells:
        for method in expected_methods:
            target_key: str | tuple[str, str] = (
                cell if cell_targets else (cell, method)
            )
            target = float(target_standard_errors[target_key])
            method_cap = resolved_caps[method]
            records = [
                by_key[(cell, method, replicate)]
                for replicate in range(pilot_replicates)
            ]
            variances = [
                SufficientStatistics.from_dict(payload["contribution"]).variance
                for payload, _ in records
            ]
            if any(not math.isfinite(value) or value < 0.0 for value in variances):
                raise ValueError("pilot contribution variance is invalid")
            design_variance = max(variances)
            requested = max(
                minimum_final_samples,
                math.ceil(
                    allocation_safety_factor
                    * design_variance
                    / (target * target)
                ),
            )
            feasible = requested <= method_cap
            all_feasible &= feasible
            chunk_counts = _chunk_counts(requested, final_chunk_size) if feasible else []
            chunks = [
                {
                    "identity": ReferenceShardIdentity(
                        protocol_id=protocol_id,
                        namespace=final_namespace,
                        stage="final",
                        method=method,  # type: ignore[arg-type]
                        cell_id=cell,
                        shard_index=index,
                    ).to_dict(),
                    "requested_samples": count,
                }
                for index, count in enumerate(chunk_counts)
            ]
            entries.append(
                {
                    "cell_id": cell,
                    "method": method,
                    "target_standard_error": target,
                    "pilot_variances": variances,
                    "pilot_shard_sha256s": [digest for _, digest in records],
                    "pilot_seed_key_sha256s": [
                        payload["seed_key_sha256"] for payload, _ in records
                    ],
                    "allocation_variance_statistic": "maximum_replicate_variance",
                    "allocation_design_variance": design_variance,
                    "requested_final_samples": requested,
                    "maximum_final_samples": method_cap,
                    "resource_feasible": feasible,
                    "authorized_final_samples": requested if feasible else None,
                    "chunks": chunks,
                }
            )

    manifest = {
        "schema": ALLOCATION_SCHEMA,
        "protocol_id": protocol_id,
        "config_sha256": config_sha256,
        "threshold_manifest_sha256": threshold_manifest_sha256,
        "pilot_parent_sha256": pilot_parent_sha256,
        "pilot_namespace": pilot_namespace,
        "final_namespace": final_namespace,
        "source_commit": source_commit,
        "environment_sha256": environment_sha256,
        "estimand": estimand,
        "dtype": dtype,
        "device": device,
        "design_informed_by_prior_development_outcomes": (
            design_informed_by_prior_development_outcomes
        ),
        "current_namespace_outcomes_inspected_before_freeze": (
            current_namespace_outcomes_inspected_before_freeze
        ),
        "pilot_replicates": pilot_replicates,
        "allocation_safety_factor": allocation_safety_factor,
        "minimum_final_samples": minimum_final_samples,
        "maximum_final_samples": maximum_final_samples,
        "final_chunk_size": final_chunk_size,
        "expected_cells": list(expected_cells),
        "expected_methods": list(expected_methods),
        "entries": entries,
        "all_resources_feasible": all_feasible,
        "final_execution_authorized": all_feasible,
        "performance_claim_authorized": False,
    }
    validate_allocation_manifest(manifest)
    return manifest


def validate_allocation_manifest(payload: Any) -> dict[str, Any]:
    expected = {
        "schema",
        "protocol_id",
        "config_sha256",
        "threshold_manifest_sha256",
        "pilot_parent_sha256",
        "pilot_namespace",
        "final_namespace",
        "source_commit",
        "environment_sha256",
        "estimand",
        "dtype",
        "device",
        "design_informed_by_prior_development_outcomes",
        "current_namespace_outcomes_inspected_before_freeze",
        "pilot_replicates",
        "allocation_safety_factor",
        "minimum_final_samples",
        "maximum_final_samples",
        "final_chunk_size",
        "expected_cells",
        "expected_methods",
        "entries",
        "all_resources_feasible",
        "final_execution_authorized",
        "performance_claim_authorized",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("malformed reference allocation manifest")
    if payload.get("schema") != ALLOCATION_SCHEMA:
        raise ValueError("unexpected allocation-manifest schema")
    if payload.get("pilot_namespace") == payload.get("final_namespace"):
        raise ValueError("allocation namespaces must be distinct")
    if payload.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("allocation manifest is not outcome-blind to its final namespace")
    if payload.get("performance_claim_authorized") is not False:
        raise ValueError("allocation manifest cannot authorize performance")
    for field in (
        "protocol_id",
        "pilot_namespace",
        "final_namespace",
        "estimand",
        "dtype",
        "device",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"allocation {field} must be nonempty and stripped")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("allocation source_commit must be a lowercase full Git commit")
    for field in (
        "config_sha256",
        "threshold_manifest_sha256",
        "pilot_parent_sha256",
        "environment_sha256",
    ):
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"allocation {field} must be a lowercase SHA-256")
    if not isinstance(
        payload.get("design_informed_by_prior_development_outcomes"), bool
    ):
        raise ValueError("allocation design-disclosure flag must be Boolean")
    pilot_replicates = payload.get("pilot_replicates")
    minimum = payload.get("minimum_final_samples")
    maximum = payload.get("maximum_final_samples")
    chunk_size = payload.get("final_chunk_size")
    if not isinstance(pilot_replicates, int) or pilot_replicates < 3:
        raise ValueError("allocation requires at least three pilot replicates")
    if (
        not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or not isinstance(chunk_size, int)
        or minimum < 2
        or maximum < minimum
        or chunk_size < 2
    ):
        raise ValueError("allocation sample bounds or chunk size are invalid")
    safety_factor = payload.get("allocation_safety_factor")
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or float(safety_factor) < 1.0
    ):
        raise ValueError("allocation safety factor is invalid")
    expected_cells_value = payload.get("expected_cells")
    expected_methods_value = payload.get("expected_methods")
    if (
        not isinstance(expected_cells_value, list)
        or not expected_cells_value
        or any(not isinstance(cell, str) or not cell for cell in expected_cells_value)
        or len(set(expected_cells_value)) != len(expected_cells_value)
    ):
        raise ValueError("allocation expected cells are invalid")
    if (
        not isinstance(expected_methods_value, list)
        or set(expected_methods_value) != set(REFERENCE_METHODS)
        or len(expected_methods_value) != len(REFERENCE_METHODS)
    ):
        raise ValueError("allocation reference method roster is invalid")
    expected_cells = cast(list[str], expected_cells_value)
    expected_methods = cast(list[str], expected_methods_value)
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("allocation manifest entries must be nonempty")
    expected_entry_keys = {
        (cell, method) for cell in expected_cells for method in expected_methods
    }
    actual_entry_keys: set[tuple[str, str]] = set()
    shard_ids: set[str] = set()
    pilot_shard_digests: set[str] = set()
    pilot_seed_keys: set[str] = set()
    entry_fields = {
        "cell_id",
        "method",
        "target_standard_error",
        "pilot_variances",
        "pilot_shard_sha256s",
        "pilot_seed_key_sha256s",
        "allocation_variance_statistic",
        "allocation_design_variance",
        "requested_final_samples",
        "maximum_final_samples",
        "resource_feasible",
        "authorized_final_samples",
        "chunks",
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise ValueError("allocation entry must be a mapping")
        cell = entry.get("cell_id")
        method = entry.get("method")
        if not isinstance(cell, str) or not isinstance(method, str):
            raise ValueError("allocation entry cell and method must be strings")
        key = (cell, method)
        if key not in expected_entry_keys or key in actual_entry_keys:
            raise ValueError("allocation entry matrix is duplicated or unexpected")
        actual_entry_keys.add(key)
        target = entry.get("target_standard_error")
        design_variance = entry.get("allocation_design_variance")
        requested = entry.get("requested_final_samples")
        cap = entry.get("maximum_final_samples")
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(float(target))
            or float(target) <= 0.0
            or isinstance(design_variance, bool)
            or not isinstance(design_variance, (int, float))
            or not math.isfinite(float(design_variance))
            or float(design_variance) < 0.0
            or not isinstance(requested, int)
            or requested < minimum
            or isinstance(cap, bool)
            or not isinstance(cap, int)
            or cap < minimum
            or cap > maximum
        ):
            raise ValueError("allocation entry statistics or sample count are invalid")
        if entry.get("allocation_variance_statistic") != "maximum_replicate_variance":
            raise ValueError("allocation variance statistic is not predeclared")
        variances = entry.get("pilot_variances")
        digests = entry.get("pilot_shard_sha256s")
        seed_keys = entry.get("pilot_seed_key_sha256s")
        if (
            not isinstance(variances, list)
            or len(variances) != pilot_replicates
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in variances
            )
            or float(design_variance) != max(float(value) for value in variances)
            or not isinstance(digests, list)
            or len(digests) != pilot_replicates
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digests
            )
            or len(set(digests)) != len(digests)
            or any(digest in pilot_shard_digests for digest in digests)
            or not isinstance(seed_keys, list)
            or len(seed_keys) != pilot_replicates
            or any(
                not isinstance(seed_key, str)
                or len(seed_key) != 64
                or any(
                    character not in "0123456789abcdef" for character in seed_key
                )
                for seed_key in seed_keys
            )
            or len(set(seed_keys)) != len(seed_keys)
            or any(seed_key in pilot_seed_keys for seed_key in seed_keys)
        ):
            raise ValueError("allocation pilot evidence is malformed")
        pilot_shard_digests.update(digests)
        pilot_seed_keys.update(seed_keys)
        calculated_requested = max(
            minimum,
            math.ceil(
                float(safety_factor)
                * float(design_variance)
                / (float(target) * float(target))
            ),
        )
        if requested != calculated_requested:
            raise ValueError("allocation requested count is inconsistent with formula")
        resource_feasible = requested <= cap
        if entry.get("resource_feasible") is not resource_feasible:
            raise ValueError("allocation entry feasibility mismatch")
        authorized = entry.get("authorized_final_samples")
        if authorized != (requested if resource_feasible else None):
            raise ValueError("allocation authorized count mismatch")
        chunks = entry.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("allocation chunks must be a list")
        if not resource_feasible:
            if chunks:
                raise ValueError("resource-infeasible entry cannot contain chunks")
            continue
        expected_counts = _chunk_counts(requested, chunk_size)
        if len(chunks) != len(expected_counts):
            raise ValueError("allocation chunk count mismatch")
        for index, (chunk, expected_count) in enumerate(
            zip(chunks, expected_counts, strict=True)
        ):
            if not isinstance(chunk, dict) or set(chunk) != {
                "identity",
                "requested_samples",
            }:
                raise ValueError("allocation chunk is malformed")
            identity = ReferenceShardIdentity.from_dict(chunk["identity"])
            if (
                identity.protocol_id != payload["protocol_id"]
                or identity.namespace != payload["final_namespace"]
                or identity.stage != "final"
                or identity.cell_id != cell
                or identity.method != method
                or identity.shard_index != index
                or chunk["requested_samples"] != expected_count
                or identity.shard_id in shard_ids
            ):
                raise ValueError("allocation chunk identity or count mismatch")
            shard_ids.add(identity.shard_id)
    if actual_entry_keys != expected_entry_keys:
        raise ValueError("allocation entry matrix is incomplete")
    feasible = all(
        isinstance(entry, dict) and entry.get("resource_feasible") is True
        for entry in entries
    )
    if payload.get("all_resources_feasible") is not feasible:
        raise ValueError("allocation feasibility aggregate mismatch")
    if payload.get("final_execution_authorized") is not feasible:
        raise ValueError("final execution authorization mismatch")
    if payload.get("estimand") != "fixed_finest_grid":
        raise ValueError("allocation estimand is unsupported")
    if payload.get("dtype") != "float64" or payload.get("device") != "cpu":
        raise ValueError("allocation backend is unsupported")
    return payload


def aggregate_final_shards(
    manifest: dict[str, Any],
    manifest_sha256: str,
    final_shards: Sequence[tuple[dict[str, Any], str]],
    *,
    final_source_commit: str | None = None,
    final_environment_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute the exact fixed-size aggregate from the expected final shard set."""

    validate_allocation_manifest(manifest)
    if canonical_sha256(manifest) != manifest_sha256:
        raise ValueError("allocation manifest digest mismatch")
    if not manifest["final_execution_authorized"]:
        raise ValueError("resource-infeasible allocation cannot be aggregated")

    expected_source_commit = final_source_commit or manifest["source_commit"]
    expected_environment_sha256 = (
        final_environment_sha256 or manifest["environment_sha256"]
    )
    expected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in manifest["entries"]:
        for chunk in entry["chunks"]:
            identity = ReferenceShardIdentity.from_dict(chunk["identity"])
            expected[identity.shard_id] = (entry, chunk)

    pilot_seed_keys = {
        seed_key
        for entry in manifest["entries"]
        for seed_key in entry["pilot_seed_key_sha256s"]
    }
    actual: dict[str, tuple[dict[str, Any], str]] = {}
    final_seed_keys: set[str] = set()
    for payload, digest in final_shards:
        validate_shard_artifact(payload)
        if canonical_sha256(payload) != digest:
            raise ValueError("final shard digest mismatch")
        identity = ReferenceShardIdentity.from_dict(payload["identity"])
        if identity.stage != "final":
            raise ValueError("final aggregate received a non-final shard")
        if payload["parent_sha256"] != manifest_sha256:
            raise ValueError("final shard parent-manifest mismatch")
        if payload["shard_id"] in actual:
            raise ValueError(f"duplicate final shard: {payload['shard_id']}")
        seed_key = payload["seed_key_sha256"]
        if seed_key in final_seed_keys or seed_key in pilot_seed_keys:
            raise ValueError("duplicate or pilot-overlapping final seed key")
        final_seed_keys.add(seed_key)
        actual[payload["shard_id"]] = (payload, digest)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"final shard set mismatch: missing={missing}, unexpected={unexpected}"
        )

    cells: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        chunk_ids = [
            ReferenceShardIdentity.from_dict(chunk["identity"]).shard_id
            for chunk in entry["chunks"]
        ]
        shards = [actual[shard_id] for shard_id in chunk_ids]
        for (payload, _), chunk in zip(shards, entry["chunks"], strict=True):
            if payload["requested_samples"] != chunk["requested_samples"]:
                raise ValueError("final shard count differs from frozen chunk count")
            if payload["config_sha256"] != manifest["config_sha256"]:
                raise ValueError("final shard config hash mismatch")
            if (
                payload["threshold_manifest_sha256"]
                != manifest["threshold_manifest_sha256"]
            ):
                raise ValueError("final shard threshold hash mismatch")
            if payload["source_commit"] != expected_source_commit:
                raise ValueError("final shard source commit mismatch")
            if payload["environment_sha256"] != expected_environment_sha256:
                raise ValueError("final shard environment hash mismatch")
            if payload["estimand"] != manifest["estimand"]:
                raise ValueError("final shard estimand mismatch")
            if (
                payload["dtype"] != manifest["dtype"]
                or payload["device"] != manifest["device"]
            ):
                raise ValueError("final shard dtype or device mismatch")
            if payload["dirty_worktree"] is not False:
                raise ValueError("final shard came from a dirty worktree")
            if any(
                payload[field] != 0
                for field in (
                    "invalid_spot_count",
                    "invalid_variance_count",
                    "nonfinite_contribution_count",
                )
            ):
                raise ValueError("final shard contains invalid paths or contributions")
        contribution = _merge_all(
            SufficientStatistics.from_dict(payload["contribution"])
            for payload, _ in shards
        )
        normalization = _merge_all(
            SufficientStatistics.from_dict(payload["likelihood_normalization"])
            for payload, _ in shards
        )
        if contribution.count != entry["authorized_final_samples"]:
            raise ValueError("aggregated final sample count mismatch")
        normalization_z = (
            (normalization.mean - 1.0) / normalization.standard_error
            if normalization.standard_error > 0.0
            else (0.0 if normalization.mean == 1.0 else math.inf)
        )
        target = float(entry["target_standard_error"])
        cells.append(
            {
                "cell_id": entry["cell_id"],
                "method": entry["method"],
                "estimate": contribution.mean,
                "variance": contribution.variance,
                "standard_error": contribution.standard_error,
                "target_standard_error": target,
                "target_attained": contribution.standard_error <= target,
                "normalization_mean": normalization.mean,
                "normalization_standard_error": normalization.standard_error,
                "normalization_z": normalization_z,
                "normalization_pass": abs(normalization_z) <= 4.0,
                "final_samples": contribution.count,
                "chunk_sha256s": [digest for _, digest in shards],
            }
        )
    cell_methods: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        cell_methods.setdefault(cell["cell_id"], {})[cell["method"]] = cell
    agreements: list[dict[str, Any]] = []
    for cell_id in manifest["expected_cells"]:
        method_cells = cell_methods[cell_id]
        dcs = method_cells["dcs_reference"]
        raw = method_cells["raw_crosscheck"]
        denominator = math.hypot(dcs["standard_error"], raw["standard_error"])
        agreement_z = (
            abs(dcs["estimate"] - raw["estimate"]) / denominator
            if denominator > 0.0
            else (0.0 if dcs["estimate"] == raw["estimate"] else math.inf)
        )
        agreements.append(
            {
                "cell_id": cell_id,
                "absolute_difference": abs(dcs["estimate"] - raw["estimate"]),
                "combined_standard_error": denominator,
                "agreement_z": agreement_z,
                "agreement_pass": agreement_z <= 4.0,
            }
        )
    all_targets = all(cell["target_attained"] for cell in cells)
    all_normalizations = all(cell["normalization_pass"] for cell in cells)
    all_agreements = all(item["agreement_pass"] for item in agreements)
    return {
        "schema": "npi.g11.v8-sharded-reference-aggregate.v1",
        "allocation_manifest_sha256": manifest_sha256,
        "config_sha256": manifest["config_sha256"],
        "threshold_manifest_sha256": manifest["threshold_manifest_sha256"],
        "source_commit": expected_source_commit,
        "environment_sha256": expected_environment_sha256,
        "pilot_source_commit": manifest["source_commit"],
        "pilot_environment_sha256": manifest["environment_sha256"],
        "cells": cells,
        "method_agreements": agreements,
        "complete_reference_matrix": len(cells) == len(manifest["entries"]),
        "all_target_standard_errors": all_targets,
        "all_likelihood_normalizations": all_normalizations,
        "all_independent_methods_agree": all_agreements,
        "reference_acceptance_pass": all_targets
        and all_normalizations
        and all_agreements,
        "performance_claim_authorized": False,
    }
