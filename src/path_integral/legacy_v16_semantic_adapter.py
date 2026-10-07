"""Read-only V16 consistency check; legacy records cannot become new confirmation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.path_integral.research_result_audit import (
    _finite,
    _integer,
    _list,
    _load_bound_file,
    _mapping,
    _matches,
    _text,
    verify_artifact_graph,
)


def audit_legacy_v16(root: Path, config_path: Path) -> dict[str, Any]:
    try:
        return _audit_legacy_checked(root, _load_bound_file(config_path))
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as error:
        return {
            "schema": "npi.post-audit.legacy-v16-adapter.v1",
            "integrity": "fail",
            "semantic_validity": "fail",
            "statistical_evidence": "unresolved",
            "performance": "unresolved",
            "reasons": [str(error)],
        }


def audit_legacy_v16_config(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Review a caller-supplied policy config, not only the on-disk default."""

    try:
        return _audit_legacy_checked(root, config)
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as error:
        return {
            "schema": "npi.post-audit.legacy-v16-adapter.v1",
            "integrity": "fail",
            "semantic_validity": "fail",
            "statistical_evidence": "unresolved",
            "performance": "unresolved",
            "reasons": [str(error)],
        }


def _audit_legacy_checked(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    bindings = _mapping(config.get("artifacts"), "policy bindings")
    graph = verify_artifact_graph(root, bindings, legacy_nonfinite=True)
    canonical_binding = _mapping(bindings.get("canonical_confirmation"), "canonical binding")
    ood_binding = _mapping(bindings.get("ood_confirmation"), "OOD binding")
    records: dict[str, float] = {}
    for binding in (canonical_binding, ood_binding):
        result = graph[_text(binding.get("path"), "result path")]
        own_config = _mapping(result.get("config_binding"), "result config binding")
        run_config = graph[_text(own_config.get("path"), "run config path")]
        baseline_binding = _mapping(
            _mapping(result.get("artifact_bindings"), "result bindings").get("baseline"),
            "baseline binding",
        )
        baseline = graph[_text(baseline_binding.get("path"), "baseline path")]
        baseline_cells = {
            _text(item.get("cell_id"), "baseline cell"): item
            for item in _list(baseline.get("cells"), "baseline cells")
        }
        if len(baseline_cells) != len(baseline["cells"]):
            raise ValueError("duplicate baseline cell")
        declared = {
            _text(item.get("reference_cell"), "declared cell"): item
            for item in _list(run_config.get("variants"), "declared variants")
        }
        if len(declared) != len(run_config["variants"]):
            raise ValueError("duplicate declared variant")
        samples_per_cluster = _integer(
            _mapping(run_config.get("evaluation"), "evaluation").get("samples_per_cluster"),
            "samples_per_cluster", minimum=2,
        )
        variants = _list(result.get("variants"), "variants")
        if len(variants) != len(declared):
            raise ValueError("missing or extra variant")
        for raw in variants:
            variant = _mapping(raw, "variant")
            cell = _text(variant.get("reference_cell"), "reference cell")
            if cell in records or cell not in declared:
                raise ValueError("duplicate or undeclared result cell")
            if variant.get("variant_id") != declared[cell].get("id"):
                raise ValueError("variant identity mismatch")
            clusters = _list(variant.get("clusters"), "clusters")
            if len(clusters) != _integer(declared[cell].get("clusters"), "clusters", minimum=2):
                raise ValueError("cluster count mismatch")
            n = len(clusters) * samples_per_cluster
            cluster_means = []
            for index, item in enumerate(clusters):
                cluster = _mapping(item, "cluster")
                if _integer(cluster.get("cluster"), "cluster index") != index:
                    raise ValueError("cluster index mismatch")
                cluster_means.append(_finite(cluster.get("estimate"), "cluster estimate", minimum=0))
            mean = sum(cluster_means) / len(clusters)
            _matches(variant.get("estimate"), mean, "legacy estimate")
            variance = _finite(variant.get("sample_variance"), "sample variance", minimum=0)
            pooled_se = math.sqrt(variance / n)
            _matches(variant.get("standard_error"), pooled_se, "legacy pooled SE")
            between = math.sqrt(
                sum((x - mean) ** 2 for x in cluster_means)
                / (len(clusters) - 1) / len(clusters)
            )
            _matches(variant.get("between_cluster_standard_error"), between, "between-cluster SE")
            robust_se = max(pooled_se, between)
            _matches(variant.get("robust_standard_error"), robust_se, "robust SE")
            if mean <= 0:
                raise ValueError("zero-hit legacy result cannot establish precision")
            _matches(variant.get("robust_relative_standard_error"), robust_se / mean, "robust RSE")
            ref_mean = _finite(variant.get("external_reference_estimate"), "reference", minimum=0)
            ref_se = _finite(variant.get("external_reference_standard_error"), "reference SE", minimum=0)
            if ref_mean <= 0 or math.hypot(robust_se, ref_se) == 0:
                raise ValueError("reference not resolvable")
            _matches(variant.get("external_accuracy_z"),
                     abs(mean - ref_mean) / math.hypot(robust_se, ref_se), "accuracy z")
            _finite(variant.get("likelihood_normalization_z"), "normalization z", minimum=0)
            if _finite(variant.get("maximum_likelihood_bound_violation"), "bound violation", minimum=0) > 0:
                raise ValueError("likelihood bound violation")
            work = _finite(variant.get("total_work_at_primary_query_count"), "total work", minimum=0)
            if work <= 0:
                raise ValueError("zero total work")
            candidate_wnv = variance * work / n
            _matches(variant.get("work_normalized_variance"), candidate_wnv, "candidate WNV")
            baseline_cell = _mapping(baseline_cells.get(cell), "baseline cell")
            method_records = {
                _text(method.get("method"), "baseline method"): method
                for method in _list(baseline_cell.get("methods"), "baseline methods")
            }
            if len(method_records) != len(baseline_cell["methods"]):
                raise ValueError("duplicate baseline method")
            displayed_ratios = _mapping(
                variant.get("comparator_over_candidate_work_ratios"), "ratios"
            )
            if not displayed_ratios:
                raise ValueError("no comparator ratios")
            ratios = []
            for method_id, displayed in displayed_ratios.items():
                method = _mapping(method_records.get(method_id), "baseline method")
                units = _integer(method.get("inferential_units"), "baseline units", minimum=2)
                method_variance = _finite(method.get("sample_variance"), "baseline variance", minimum=0)
                method_work = _finite(method.get("total_work_at_primary_query_count"), "baseline work", minimum=0)
                if method_work <= 0 or candidate_wnv <= 0:
                    raise ValueError("nonpositive ratio denominator")
                ratio = (method_variance * method_work / units) / candidate_wnv
                _matches(displayed, ratio, f"{method_id} ratio")
                ratios.append(ratio)
            strongest = min(ratios)
            _matches(variant.get("strongest_comparator_over_candidate_work_ratio"), strongest, "strongest ratio")
            records[cell] = strongest
    required = _mapping(config.get("required_cells"), "required cells")
    expected = [item for value in required.values() for item in _list(value, "required group")]
    if len(set(expected)) != len(expected) or set(expected) != set(records):
        raise ValueError("required cell mismatch or duplicate")
    return {
        "schema": "npi.post-audit.legacy-v16-adapter.v1",
        "integrity": "pass",
        "semantic_validity": "pass",
        "statistical_evidence": "unresolved",
        "performance": "unresolved",
        "proxy_ratios": records,
        "reasons": [
            "legacy summary lacks raw likelihood moments and independent training repetitions",
            "historical CE/conditioning and comparator qualification are not matched",
            "historical work ratios do not establish total wall-time improvement",
        ],
    }
