"""Independent semantic audit of the post-audit result schema."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral.comparator_qualification import (
    QualificationPolicy,
    qualify_estimate,
    work_normalized_variance,
)
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.research_result_contract import (
    SCHEMA,
    Moments,
    StageCost,
    canonical_digest,
)
from src.path_integral.seed_ledger import SeedKey, SeedLedger


def _fail(message: str) -> None:
    raise ValueError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be a list")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        _fail(f"{name} must be finite numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        _fail(f"{name} is below minimum")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{name} must be a nonempty trimmed string")
    return value


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        _fail(f"{name} must be a lowercase SHA256 digest")
    return text


def _moments(value: Any, name: str) -> Moments:
    obj = _mapping(value, name)
    if set(obj) != {"count", "mean", "m2"}:
        _fail(f"{name} has incorrect moment fields")
    return Moments(
        _integer(obj["count"], f"{name}.count", minimum=2),
        _finite(obj["mean"], f"{name}.mean", minimum=0),
        _finite(obj["m2"], f"{name}.m2", minimum=0),
    )


def _cost(value: Any, stage: str) -> StageCost:
    obj = _mapping(value, f"{stage} cost")
    if obj.get("stage") != stage:
        _fail(f"expected {stage} cost stage")
    return StageCost(
        stage,
        _finite(obj.get("wall_seconds"), "wall_seconds", minimum=0),
        _finite(obj.get("cpu_seconds"), "cpu_seconds", minimum=0),
        _integer(obj.get("peak_memory_bytes"), "peak_memory_bytes"),
        _finite(obj.get("proxy_work_units"), "proxy_work_units", minimum=0),
    )


def _matches(reported: Any, expected: float | None, name: str) -> None:
    if expected is None:
        if reported is not None:
            _fail(f"{name} must be null")
        return
    observed = _finite(reported, name)
    if not math.isclose(observed, expected, rel_tol=1e-8, abs_tol=1e-30):
        _fail(f"{name} differs from raw statistics")


def _seed(
    ledger: SeedLedger,
    used: set[SeedKey],
    item: dict[str, Any],
    *,
    role: str,
    task_id: str,
    stream: str,
    replicate: int,
    field_prefix: str = "",
) -> None:
    key_field = f"{field_prefix}seed_key"
    seed_field = f"{field_prefix}seed"
    key = SeedKey(**_mapping(item.get(key_field), key_field))
    if key.role != role or key.task != task_id or key.stream != stream or key.replicate != replicate:
        _fail("seed role, task, stream or replicate mismatch")
    if key in used:
        _fail("random stream reused")
    used.add(key)
    if _integer(item.get(seed_field), seed_field, minimum=1) != ledger.lookup(key):
        _fail("seed does not match manifest")


def _combine(values: list[Moments]) -> Moments:
    if not values:
        _fail("no independent clusters")
    result = values[0]
    for other in values[1:]:
        result = result.merge(other)
    return result


def _normalization_z(moments: Moments) -> float | None:
    se = moments.standard_error
    if se == 0:
        return 0.0 if moments.mean == 1.0 else None
    return abs(moments.mean - 1.0) / se


def audit_measurement(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute displays and decisions; never use a stored `passed` flag."""

    try:
        result = _audit_measurement_checked(payload)
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return {
            "schema": "npi.post-audit.semantic-audit.v1",
            "integrity": "fail",
            "semantic_validity": "fail",
            "statistical_evidence": "unresolved",
            "performance": "unresolved",
            "reasons": [str(error)],
        }
    return result


def _audit_measurement_checked(payload: dict[str, Any]) -> dict[str, Any]:
    if _mapping(payload, "payload").get("schema") != SCHEMA:
        _fail("unsupported measurement schema")
    manifest = _mapping(payload.get("manifest"), "manifest")
    source = _mapping(manifest.get("source"), "source")
    _text(source.get("source_commit"), "source_commit")
    if not isinstance(source.get("source_dirty"), bool):
        _fail("source_dirty must be boolean")
    _digest(source.get("source_patch_digest"), "source_patch_digest")
    _digest(source.get("source_tree_digest"), "source_tree_digest")
    _digest(source.get("config_digest"), "config_digest")
    _mapping(source.get("runtime"), "runtime")
    if canonical_digest(manifest.get("config")) != source["config_digest"]:
        _fail("config digest mismatch")
    timing = _mapping(manifest.get("timing_context"), "timing_context")
    _text(timing.get("warmup"), "warmup")
    _text(timing.get("power_mode"), "power_mode")
    if not isinstance(timing.get("concurrent_timed_jobs"), bool):
        _fail("concurrent_timed_jobs must be boolean")
    task_id = _text(manifest.get("task_id"), "task_id")
    payoff = _text(manifest.get("payoff_id"), "payoff_id")
    if payoff != "terminal_left_conditional_cdf":
        _fail("unsupported payoff contract")
    _text(manifest.get("convention"), "convention")
    steps = _integer(manifest.get("steps"), "steps", minimum=1)
    _integer(manifest.get("level"), "level")
    training_rep = _integer(manifest.get("training_rep"), "training_rep")
    ledger = SeedLedger.from_dict(_mapping(manifest.get("seed_ledger"), "seed_ledger"))
    used: set[SeedKey] = set()
    reference = _mapping(payload.get("reference"), "reference")
    if reference.get("independent") is not True:
        _fail("reference independence not declared")
    if reference.get("inferential_unit") not in {"iid_path", "smc_run", "rqmc_randomization"}:
        _fail("invalid reference inferential unit")
    reference_points = _integer(reference.get("points_per_unit"), "reference points_per_unit", minimum=1)
    if reference["inferential_unit"] == "iid_path" and reference_points != 1:
        _fail("iid reference must have one point per inferential unit")
    _seed(ledger, used, reference, role="reference", task_id=task_id,
          stream="reference", replicate=training_rep)
    reference_moments = _moments(reference.get("moments"), "reference moments")
    if _integer(reference.get("raw_sample_count"), "reference raw samples", minimum=2) != reference_moments.count * reference_points:
        _fail("reference unit and point count mismatch")
    _cost(reference.get("cost"), "reference")
    reference_reported = _mapping(reference.get("reported"), "reference reported")
    _matches(reference_reported.get("estimate"), reference_moments.mean, "reference estimate")
    _matches(reference_reported.get("standard_error"), reference_moments.standard_error, "reference SE")

    policy = QualificationPolicy(**_mapping(manifest.get("qualification_policy"), "qualification policy"))
    records = _list(payload.get("methods"), "methods")
    if len(records) < 2:
        _fail("at least two methods are required")
    seen_methods: set[str] = set()
    measurements: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for raw in records:
        method = _mapping(raw, "method")
        method_id = _text(method.get("method_id"), "method_id")
        if method_id in seen_methods:
            _fail("duplicate method key")
        seen_methods.add(method_id)
        _digest(method.get("proposal_digest"), "proposal_digest")
        parameters = _mapping(method.get("proposal_parameters"), "proposal parameters")
        if canonical_digest(parameters) != method["proposal_digest"]:
            _fail("proposal digest mismatch")
        components = tuple(
            FiniteRankGaussianComponent(
                mean=torch.tensor(_mapping(raw_component, "component")["mean"], dtype=torch.float64),
                directions=torch.tensor(raw_component["directions"], dtype=torch.float64),
                variance_eigenvalues=torch.tensor(raw_component["eigenvalues"], dtype=torch.float64),
            )
            for raw_component in _list(parameters.get("components"), "components")
        )
        proposal = DefensiveFiniteRankGaussianMixture(
            components,
            torch.tensor(_list(parameters.get("weights"), "weights"), dtype=torch.float64),
        )
        if method.get("payoff_id") != payoff or _integer(method.get("dimension"), "dimension", minimum=1) != 2 * steps:
            _fail("method target or proposal dimension mismatch")
        if proposal.dimension != method["dimension"]:
            _fail("proposal parameters have incorrect dimension")
        inferential_unit = method.get("inferential_unit")
        if inferential_unit not in {"iid_path", "rqmc_randomization", "smc_run"}:
            _fail("invalid method inferential unit")
        points_per_unit = _integer(method.get("points_per_unit"), "points_per_unit", minimum=1)
        if inferential_unit == "iid_path" and points_per_unit != 1:
            _fail("iid path must have one point per inferential unit")
        defensive_mass = _finite(method.get("defensive_mass"), "defensive_mass", minimum=0)
        if defensive_mass <= 0 or defensive_mass > 1:
            _fail("defensive mass outside (0,1]")
        _matches(defensive_mass, proposal.defensive_mass, "defensive mass")
        if not math.isfinite(1.0 / defensive_mass):
            _fail("defensive mass bound is unrepresentable")
        if method.get("fitted") is True:
            fit_seed = _mapping(method.get("fit_seed"), "fit_seed")
            _seed(ledger, used, fit_seed, role="training", task_id=task_id,
                  stream=method_id, replicate=training_rep)
            internal = SeedLedger.from_dict(
                _mapping(method.get("training_substream_ledger"), "training substreams")
            )
            if len(internal) == 0 or any(
                record.key.role != "training"
                or record.key.task != task_id
                or record.key.replicate != fit_seed["seed"]
                for record in internal.records
            ):
                _fail("training substream provenance mismatch")
        elif (
            method.get("fitted") is not False
            or method.get("fit_seed") is not None
            or method.get("training_substream_ledger") is not None
        ):
            _fail("fitted flag or fit seed inconsistent")
        offline_cost = _cost(method.get("offline_cost"), "offline")
        fit_cost = _cost(method.get("fit_cost"), "fit")
        selection_cost = _cost(method.get("selection_cost"), "selection")
        if method["fitted"] and fit_cost.proxy_work_units <= 0:
            _fail("fitted method has no charged fit work")
        clusters = _list(method.get("clusters"), "clusters")
        if not clusters:
            _fail("method has no final clusters")
        moment_list: list[Moments] = []
        weight_list: list[Moments] = []
        wall = offline_cost.wall_seconds + fit_cost.wall_seconds + selection_cost.wall_seconds
        cpu = offline_cost.cpu_seconds + fit_cost.cpu_seconds + selection_cost.cpu_seconds
        peak = max(offline_cost.peak_memory_bytes, fit_cost.peak_memory_bytes,
                   selection_cost.peak_memory_bytes)
        work = (offline_cost.proxy_work_units + fit_cost.proxy_work_units
                + selection_cost.proxy_work_units)
        cluster_ids: set[int] = set()
        for item in clusters:
            cluster = _mapping(item, "cluster")
            index = _integer(cluster.get("evaluation_rep"), "evaluation_rep")
            if index in cluster_ids:
                _fail("duplicate evaluation replicate")
            cluster_ids.add(index)
            _seed(ledger, used, cluster, role="final", task_id=task_id,
                  stream=f"{method_id}:path", replicate=index)
            _seed(ledger, used, cluster, role="final", task_id=task_id,
                  stream=f"{method_id}:label", replicate=index,
                  field_prefix="label_")
            moments = _moments(cluster.get("moments"), "cluster moments")
            weights = _moments(cluster.get("normalization_moments"), "normalization moments")
            if moments.count != weights.count:
                _fail("payoff and normalization sample counts differ")
            if _integer(cluster.get("raw_sample_count"), "raw_sample_count", minimum=2) != moments.count * points_per_unit:
                _fail("raw samples incorrectly counted as independent units")
            if _finite(cluster.get("maximum_contribution"), "maximum_contribution", minimum=0) > 1.0 / defensive_mass * (1 + 1e-10):
                _fail("defensive likelihood bound violated")
            if moments.mean > cluster["maximum_contribution"] * (1 + 1e-10):
                _fail("reported maximum below cluster mean")
            cost = _cost(cluster.get("cost"), "inference")
            wall += cost.wall_seconds
            cpu += cost.cpu_seconds
            peak = max(peak, cost.peak_memory_bytes)
            work += cost.proxy_work_units
            moment_list.append(moments)
            weight_list.append(weights)
        combined = _combine(moment_list)
        normal = _combine(weight_list)
        se = combined.standard_error
        rse = se / combined.mean if combined.mean > 0 else None
        norm_z = _normalization_z(normal)
        qualification = qualify_estimate(
            estimate=combined.mean, standard_error=se,
            reference_estimate=reference_moments.mean,
            reference_standard_error=reference_moments.standard_error,
            reference_independent=True, policy=policy,
        )
        reported = _mapping(method.get("reported"), "method reported")
        for key, expected in (
            ("estimate", combined.mean), ("sample_variance", combined.sample_variance),
            ("standard_error", se), ("relative_standard_error", rse),
            ("normalization_z", norm_z), ("accuracy_z", qualification.accuracy_z_diagnostic),
            ("total_wall_seconds", wall), ("total_cpu_seconds", cpu),
            ("peak_memory_bytes", float(peak)), ("total_proxy_work", work),
        ):
            _matches(reported.get(key), expected, f"{method_id}.{key}")
        if reported.get("qualification") != qualification.status:
            _fail("stored qualification disagrees with common rule")
        if norm_z is None or norm_z > 4:
            reasons.append(f"{method_id}: normalization unresolved")
        if qualification.status != "qualified":
            reasons.append(f"{method_id}: {qualification.reason}")
        measurements[method_id] = {
            "estimate": combined.mean,
            "standard_error": se,
            "relative_standard_error": rse,
            "normalization_z": norm_z,
            "qualification": qualification.status,
            "total_wall_seconds": wall,
            "total_cpu_seconds": cpu,
            "peak_memory_bytes": peak,
            "total_proxy_work": work,
            "proxy_wnv": work_normalized_variance(combined.sample_variance, work, combined.count),
        }
    comparisons = _list(payload.get("comparisons"), "comparisons")
    seen_pairs: set[tuple[str, str]] = set()
    for raw in comparisons:
        pair = _mapping(raw, "comparison")
        candidate = _text(pair.get("candidate"), "candidate")
        comparator = _text(pair.get("comparator"), "comparator")
        if candidate == comparator or candidate not in measurements or comparator not in measurements:
            _fail("invalid comparison method")
        if (candidate, comparator) in seen_pairs:
            _fail("duplicate comparison key")
        seen_pairs.add((candidate, comparator))
        denominator = measurements[candidate]["proxy_wnv"]
        expected_ratio = (
            measurements[comparator]["proxy_wnv"] / denominator
            if denominator > 0 else None
        )
        _matches(pair.get("reported_proxy_ratio"), expected_ratio, "comparison ratio")
        if any(measurements[name]["qualification"] != "qualified" for name in (candidate, comparator)):
            reasons.append(f"{candidate}/{comparator}: comparison unresolved")
    return {
        "schema": "npi.post-audit.semantic-audit.v1",
        "integrity": "pass",
        "semantic_validity": "pass",
        "statistical_evidence": "qualified" if not reasons else "unresolved",
        "performance": "unresolved",  # R0 supplies no fresh multi-fit time comparison.
        "methods": measurements,
        "reasons": reasons,
    }


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def _load_bound_file(path: Path, *, legacy_nonfinite: bool = False) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        obj = json.loads(
            content, object_pairs_hook=_pairs_no_duplicates,
            parse_constant=(
                (lambda value: math.inf if value == "Infinity" else _fail(f"invalid JSON constant: {value}"))
                if legacy_nonfinite else (lambda value: _fail(f"invalid JSON constant: {value}"))
            ),
        )
    else:
        obj = yaml.load(content, Loader=_UniqueSafeLoader)
    return _mapping(obj, str(path))


def verify_artifact_graph(
    root: Path, bindings: dict[str, Any], *, legacy_nonfinite: bool = False
) -> dict[str, dict[str, Any]]:
    """Verify recursive artifact edges and config back-references without trusting flags."""

    visited: dict[str, tuple[str, dict[str, Any]]] = {}
    active: set[str] = set()

    def visit(binding: dict[str, Any], *, descend: bool = True) -> dict[str, Any]:
        relative = _text(binding.get("path"), "binding path")
        expected = _digest(binding.get("sha256"), "binding hash")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            _fail(f"binding escapes repository or is absent: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            _fail(f"binding hash mismatch: {relative}")
        if relative in active:
            _fail(f"artifact dependency cycle: {relative}")
        if relative in visited:
            if visited[relative][0] != expected:
                _fail(f"conflicting binding: {relative}")
            return visited[relative][1]
        data = _load_bound_file(path, legacy_nonfinite=legacy_nonfinite)
        visited[relative] = (expected, data)
        if descend:
            active.add(relative)
            for child in _mapping(data.get("artifact_bindings", {}), "artifact_bindings").values():
                visit(_mapping(child, "child binding"))
            if "config_binding" in data:
                # A result's own config can point back to that result. Verify the
                # digest without interpreting this provenance edge as a dependency.
                visit(_mapping(data["config_binding"], "config binding"), descend=False)
            active.remove(relative)
        return data

    for binding in _mapping(bindings, "bindings").values():
        visit(_mapping(binding, "binding"))
    return {key: value[1] for key, value in visited.items()}
