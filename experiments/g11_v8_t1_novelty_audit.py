"""Fail-closed audit for the post-Stage-B G11 V8 T1 novelty update."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

EXPECTED_SCHEMA = "npi.g11.v8-t1-novelty-update.v1"
REPORT_SCHEMA = "npi.g11.v8-t1-novelty-audit.v1"
ACCESS_DATE = "2026-08-02"
EXPECTED_BASE_SHA256 = "b3f31276194eee08333fd1d7c5eef68ea678a3e3c6840b2a6101316d4d820f19"
REQUIRED_SOURCE_IDS = {
    "friz_salkeld_wagenhofer_2025",
    "gassiat_2023",
    "bayer_fukasawa_nakahara_2022",
    "fukasawa_hirano_2021",
    "ahn_zheng_2023",
    "jacquier_pannier_2021",
    "bures_2025",
}
REQUIRED_FAMILIES = {
    "rough_weak_error",
    "rough_simulation_projection",
    "conditional_importance_sampling",
    "marginalized_importance_sampling",
    "rough_barrier_analysis",
    "volterra_rare_event",
}
REQUIRED_FORBIDDEN_CLAIMS = {
    "first_nonlinear_rough_volatility_weak_rate",
    "first_orthogonal_projection_for_rough_bergomi",
    "first_conditional_importance_sampler",
    "first_rao_blackwellized_importance_sampler",
    "first_volterra_large_deviation_method",
    "universal_rarity_independent_variance_factor",
    "broad_training_inclusive_superiority",
    "barrier_mesh_rate",
    "unconditional_top_journal_novelty",
}
ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "search_cutoff_date",
    "base_ledger",
    "interfaces",
    "queries",
    "sources",
    "decision",
}
SOURCE_KEYS = {
    "id",
    "family",
    "title",
    "authors",
    "year",
    "primary_url",
    "version",
    "doi",
    "peer_review_status",
    "overlap",
    "nonoverlap",
    "consequence",
}


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _https(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected T1 novelty-update schema")
    return payload, hashlib.sha256(raw).hexdigest()


def audit_t1_novelty_update(payload: dict[str, Any], payload_sha256: str) -> dict[str, Any]:
    """Return a deterministic fail-closed audit of the T1 update."""

    checks: dict[str, bool] = {}

    def record(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    base = _mapping(payload.get("base_ledger"))
    interfaces = _records(payload.get("interfaces"))
    queries = _records(payload.get("queries"))
    sources = _records(payload.get("sources"))
    decision = _mapping(payload.get("decision"))
    source_ids = {item.get("id") for item in sources}
    source_families = {item.get("family") for item in sources}
    query_families = {item.get("family") for item in queries}
    interface_ids = {item.get("id") for item in interfaces}

    record("schema_exact", payload.get("schema") == EXPECTED_SCHEMA)
    record("root_keys_exact", set(payload) == ROOT_KEYS)
    record("protocol_exact", payload.get("protocol_id") == "g11-v8-t1-novelty-update-v1")
    record(
        "dates_exact",
        payload.get("date") == ACCESS_DATE and payload.get("search_cutoff_date") == ACCESS_DATE,
    )
    record(
        "base_ledger_hash_bound",
        base
        == {
            "path": "configs/g11_v8/novelty_search_ledger_v1.yaml",
            "sha256": EXPECTED_BASE_SHA256,
            "decision": "conditional_pass_for_p2_only",
        },
    )
    record(
        "interfaces_valid",
        bool(interfaces)
        and len(interface_ids) == len(interfaces)
        and all(
            set(item) == {"id", "name", "access_date"}
            and _nonempty(item.get("id"))
            and _nonempty(item.get("name"))
            and item.get("access_date") == ACCESS_DATE
            for item in interfaces
        ),
    )
    record(
        "queries_valid",
        len(queries) >= 7
        and len({item.get("id") for item in queries}) == len(queries)
        and all(
            set(item) == {"id", "interface_id", "executed_on", "text", "family"}
            and item.get("interface_id") in interface_ids
            and item.get("executed_on") == ACCESS_DATE
            and _nonempty(item.get("text"))
            and _nonempty(item.get("family"))
            for item in queries
        ),
    )
    record("required_query_families", REQUIRED_FAMILIES.issubset(query_families))
    record("source_ids_exact", source_ids == REQUIRED_SOURCE_IDS)
    record("source_keys_exact", bool(sources) and all(set(item) == SOURCE_KEYS for item in sources))
    record(
        "source_families_covered",
        REQUIRED_FAMILIES - {"marginalized_importance_sampling"} == source_families,
    )
    record(
        "sources_primary_and_complete",
        bool(sources)
        and all(
            all(
                _nonempty(item.get(field))
                for field in (
                    "id",
                    "family",
                    "title",
                    "version",
                    "peer_review_status",
                    "overlap",
                    "nonoverlap",
                    "consequence",
                )
            )
            and isinstance(item.get("authors"), list)
            and bool(item["authors"])
            and all(_nonempty(author) for author in item["authors"])
            and isinstance(item.get("year"), int)
            and 2000 <= item["year"] <= 2026
            and _https(item.get("primary_url"))
            and _https(item.get("doi"))
            and item.get("peer_review_status") in {"published", "preprint"}
            for item in sources
        ),
    )
    record("decision_status_narrowed", decision.get("status") == "narrowed_and_blocked")
    record("full_combination_not_found", decision.get("full_combination_found") is False)
    surviving = decision.get("surviving_candidate")
    surviving_text = surviving if isinstance(surviving, str) else ""
    record(
        "surviving_candidate_narrow",
        _nonempty(surviving_text)
        and all(
            token in surviving_text.lower()
            for token in ("proposal-conditional", "defensive", "rarity-dependent")
        )
        and "first" not in surviving_text.lower(),
    )
    record(
        "forbidden_claims_complete",
        set(decision.get("claims_forbidden", [])) == REQUIRED_FORBIDDEN_CLAIMS,
    )
    record("submission_blocked", decision.get("submission_novelty_authorized") is False)
    record("top_journal_blocked", decision.get("top_journal_route_authorized") is False)
    record("external_review_required", decision.get("external_expert_review_required") is True)
    record(
        "database_search_required", decision.get("subscription_database_search_required") is True
    )
    record("decision_reasoned", _nonempty(decision.get("reason")))

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "payload_sha256": payload_sha256,
        "base_ledger_sha256": base.get("sha256"),
        "counts": {
            "interfaces": len(interfaces),
            "queries": len(queries),
            "sources": len(sources),
        },
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    payload, digest = _load(args.update)
    report = audit_t1_novelty_update(payload, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
