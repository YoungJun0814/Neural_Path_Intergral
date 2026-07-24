"""Fail-closed audit for the G11 V8 reproducible novelty-search ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

EXPECTED_SCHEMA = "npi.g11.v8-novelty-search-ledger.v1"
REPORT_SCHEMA = "npi.g11.v8-novelty-audit.v1"
ACCESS_DATE = "2026-07-25"

REQUIRED_FAMILIES = {
    "rough_bergomi_conditional_mc",
    "numerical_smoothing_qmc_asgq_mlmc",
    "multilevel_importance_sampling",
    "balance_multiple_importance_sampling",
    "large_deviation_and_cem_is",
    "adaptive_mixture_importance_sampling",
    "exact_density_flow_importance_sampling",
    "rough_weak_error_and_mlmc",
    "learning_enhanced_control_variates",
}
REQUIRED_SOURCES = {
    "mccrickerd_pakkanen_2018",
    "bayer_benhammouda_tempone_rbergomi_qmc_2020",
    "bayer_benhammouda_tempone_smoothing_qmc_2023",
    "bayer_benhammouda_tempone_smoothing_mlmc_2024",
    "giles_2008",
    "sbert_elvira_2022",
    "cappe_douc_guillin_marin_robert_2008",
    "tong_stadler_2022",
    "ben_amar_ben_rached_tempone_2026",
    "gao_zhang_daniel_boning_2023",
    "kruse_tzikas_delecki_arief_kochenderfer_2025",
    "bayer_hall_tempone_2022",
    "bourgey_de_marco_2025",
    "jouravlev_2025",
}
EXPECTED_FORBIDDEN_CLAIMS = {
    "first_conditional_monte_carlo_under_rough_bergomi",
    "first_numerical_smoothing_of_discontinuous_payoffs",
    "first_qmc_under_rough_bergomi",
    "first_mlmc_under_rough_volatility",
    "first_balance_mixture_importance_sampler",
    "first_mixture_rao_blackwellization",
    "first_large_deviation_or_cem_rare_event_sampler",
    "first_flow_based_rare_event_importance_sampler",
    "first_learning_enhanced_rough_volatility_variance_reduction",
    "unconditional_top_journal_novelty",
}

ROOT_KEYS = {
    "schema",
    "protocol_id",
    "date",
    "search_cutoff_date",
    "phase",
    "outcome_data_used",
    "interfaces",
    "families",
    "queries",
    "sources",
    "close_exclusions",
    "decision",
}
INTERFACE_KEYS = {"id", "name", "access_date"}
FAMILY_KEYS = {"id", "required"}
QUERY_KEYS = {"id", "interface_id", "executed_on", "text", "family_ids"}
SOURCE_KEYS = {
    "id",
    "family_ids",
    "title",
    "authors",
    "year",
    "source_type",
    "primary_url",
    "version",
    "doi",
    "model_class",
    "event_or_payoff",
    "proposal",
    "density_contract",
    "conditional_integration",
    "theorem_or_analysis",
    "work_accounting",
    "code_availability",
    "material_overlap",
    "material_nonoverlap",
    "baseline_role",
    "peer_review_status",
    "source_primary",
}
EXCLUSION_KEYS = {"id", "title", "primary_url", "reason"}
DECISION_KEYS = {
    "status",
    "full_combination_found",
    "component_prior_art_acknowledged",
    "candidate_contribution",
    "claims_forbidden",
    "external_expert_review_required",
    "submission_novelty_authorized",
    "p2_theory_implementation_authorized",
}
TEXT_SOURCE_FIELDS = {
    "id",
    "title",
    "source_type",
    "primary_url",
    "version",
    "model_class",
    "event_or_payoff",
    "proposal",
    "density_contract",
    "conditional_integration",
    "theorem_or_analysis",
    "work_accounting",
    "code_availability",
    "material_overlap",
    "material_nonoverlap",
    "baseline_role",
    "peer_review_status",
}
ALLOWED_SOURCE_TYPES = {
    "conference_paper",
    "journal_article",
    "preprint",
    "working_paper",
}
ALLOWED_PEER_REVIEW = {"published", "preprint", "working_paper"}
MINIMUM_FAMILY_SOURCE_COUNTS = {
    "numerical_smoothing_qmc_asgq_mlmc": 3,
    "exact_density_flow_importance_sampling": 2,
    "rough_weak_error_and_mlmc": 4,
}


def _load_ledger(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("novelty ledger root must be a mapping")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected novelty schema: {payload.get('schema')!r}")
    return payload, hashlib.sha256(raw).hexdigest()


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _ids(records: list[dict[str, Any]]) -> list[str]:
    return [record["id"] for record in records if isinstance(record.get("id"), str)]


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _https_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _exact_keys(records: list[dict[str, Any]], expected: set[str]) -> bool:
    return bool(records) and all(set(record) == expected for record in records)


def audit_novelty_ledger(
    ledger: dict[str, Any], ledger_sha256: str
) -> dict[str, Any]:
    """Return a deterministic audit report for the P1 literature ledger."""

    checks: dict[str, bool] = {}

    def record(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    interfaces = _records(ledger.get("interfaces"))
    families = _records(ledger.get("families"))
    queries = _records(ledger.get("queries"))
    sources = _records(ledger.get("sources"))
    exclusions = _records(ledger.get("close_exclusions"))
    decision = _mapping(ledger.get("decision"))

    interface_ids = _ids(interfaces)
    family_ids = _ids(families)
    query_ids = _ids(queries)
    source_ids = _ids(sources)
    exclusion_ids = _ids(exclusions)

    record("schema_exact", ledger.get("schema") == EXPECTED_SCHEMA)
    record("root_keys_exact", set(ledger) == ROOT_KEYS)
    record(
        "protocol_exact",
        ledger.get("protocol_id") == "g11-v8-novelty-search-ledger-v1",
    )
    record("date_exact", ledger.get("date") == ACCESS_DATE)
    record("search_cutoff_exact", ledger.get("search_cutoff_date") == ACCESS_DATE)
    record("phase_exact", ledger.get("phase") == "p1_development")
    record("outcome_blind", ledger.get("outcome_data_used") is False)

    record("interface_keys_exact", _exact_keys(interfaces, INTERFACE_KEYS))
    record("interface_ids_unique", len(interface_ids) == len(set(interface_ids)))
    record(
        "interface_dates_exact",
        bool(interfaces)
        and all(interface.get("access_date") == ACCESS_DATE for interface in interfaces),
    )
    record(
        "interfaces_described",
        bool(interfaces)
        and all(
            _nonempty_text(interface.get("id"))
            and _nonempty_text(interface.get("name"))
            for interface in interfaces
        ),
    )

    record("family_keys_exact", _exact_keys(families, FAMILY_KEYS))
    record("required_families_exact", set(family_ids) == REQUIRED_FAMILIES)
    record("family_ids_unique", len(family_ids) == len(set(family_ids)))
    record(
        "all_families_required",
        len(families) == len(REQUIRED_FAMILIES)
        and all(family.get("required") is True for family in families),
    )

    record("query_keys_exact", _exact_keys(queries, QUERY_KEYS))
    record("minimum_query_count", len(queries) >= 12)
    record("query_ids_unique", len(query_ids) == len(set(query_ids)))
    record(
        "queries_dated_and_described",
        bool(queries)
        and all(
            query.get("executed_on") == ACCESS_DATE
            and _nonempty_text(query.get("text"))
            for query in queries
        ),
    )
    record(
        "query_interfaces_valid",
        bool(queries)
        and all(query.get("interface_id") in set(interface_ids) for query in queries),
    )
    record(
        "query_family_references_valid",
        bool(queries)
        and all(
            bool(_string_set(query.get("family_ids")))
            and _string_set(query.get("family_ids")).issubset(REQUIRED_FAMILIES)
            for query in queries
        ),
    )
    query_family_coverage = set().union(
        *(_string_set(query.get("family_ids")) for query in queries)
    )
    record("all_families_have_queries", query_family_coverage == REQUIRED_FAMILIES)

    record("source_keys_exact", _exact_keys(sources, SOURCE_KEYS))
    record("required_sources_exact", set(source_ids) == REQUIRED_SOURCES)
    record("source_ids_unique", len(source_ids) == len(set(source_ids)))
    record(
        "source_text_fields_complete",
        bool(sources)
        and all(
            all(_nonempty_text(source.get(field)) for field in TEXT_SOURCE_FIELDS)
            for source in sources
        ),
    )
    record(
        "source_authors_complete",
        bool(sources)
        and all(
            isinstance(source.get("authors"), list)
            and bool(source["authors"])
            and all(_nonempty_text(author) for author in source["authors"])
            for source in sources
        ),
    )
    record(
        "source_years_valid",
        bool(sources)
        and all(
            isinstance(source.get("year"), int)
            and not isinstance(source.get("year"), bool)
            and 1900 <= source["year"] <= 2026
            for source in sources
        ),
    )
    record(
        "source_types_valid",
        bool(sources)
        and all(source.get("source_type") in ALLOWED_SOURCE_TYPES for source in sources),
    )
    record(
        "peer_review_status_valid",
        bool(sources)
        and all(
            source.get("peer_review_status") in ALLOWED_PEER_REVIEW
            for source in sources
        ),
    )
    primary_urls = [source.get("primary_url") for source in sources]
    record(
        "primary_urls_https",
        bool(primary_urls) and all(_https_url(url) for url in primary_urls),
    )
    record("primary_urls_unique", len(primary_urls) == len(set(primary_urls)))
    record(
        "dois_valid_or_null",
        all(source.get("doi") is None or _https_url(source.get("doi")) for source in sources),
    )
    record(
        "all_sources_primary",
        bool(sources) and all(source.get("source_primary") is True for source in sources),
    )
    record(
        "source_family_references_valid",
        bool(sources)
        and all(
            bool(_string_set(source.get("family_ids")))
            and _string_set(source.get("family_ids")).issubset(REQUIRED_FAMILIES)
            for source in sources
        ),
    )
    source_family_counts = {
        family: sum(
            family in _string_set(source.get("family_ids")) for source in sources
        )
        for family in REQUIRED_FAMILIES
    }
    record(
        "all_families_have_sources",
        all(count >= 1 for count in source_family_counts.values()),
    )
    for family, minimum in MINIMUM_FAMILY_SOURCE_COUNTS.items():
        record(
            f"family_depth_{family}",
            source_family_counts.get(family, 0) >= minimum,
        )

    source_by_id = {source.get("id"): source for source in sources}
    record(
        "closest_method_predeclared",
        source_by_id.get(
            "bayer_benhammouda_tempone_smoothing_qmc_2023", {}
        ).get("baseline_role")
        == "primary_closest_method",
    )
    record(
        "large_deviation_baseline_required",
        source_by_id.get("tong_stadler_2022", {}).get("baseline_role")
        == "secondary_required",
    )
    record(
        "flow_baseline_required",
        source_by_id.get("gao_zhang_daniel_boning_2023", {}).get(
            "baseline_role"
        )
        == "secondary_required",
    )

    record("exclusion_keys_exact", _exact_keys(exclusions, EXCLUSION_KEYS))
    record("exclusion_ids_unique", len(exclusion_ids) == len(set(exclusion_ids)))
    record(
        "exclusions_reasoned",
        bool(exclusions)
        and all(
            _nonempty_text(exclusion.get("id"))
            and _nonempty_text(exclusion.get("title"))
            and _nonempty_text(exclusion.get("reason"))
            and (
                exclusion.get("primary_url") is None
                or _https_url(exclusion.get("primary_url"))
            )
            for exclusion in exclusions
        ),
    )

    record("decision_keys_exact", set(decision) == DECISION_KEYS)
    record("decision_is_conditional_pass", decision.get("status") == "conditional_pass")
    record("full_combination_not_claimed_found", decision.get("full_combination_found") is False)
    record(
        "component_prior_art_acknowledged",
        decision.get("component_prior_art_acknowledged") is True,
    )
    candidate = decision.get("candidate_contribution")
    record(
        "candidate_contribution_narrow",
        _nonempty_text(candidate)
        and all(
            token in candidate.lower()
            for token in ("defensive", "gaussian", "residual", "rough-volterra")
        )
        and "first" not in candidate.lower(),
    )
    record(
        "forbidden_claims_exact",
        _string_set(decision.get("claims_forbidden"))
        == EXPECTED_FORBIDDEN_CLAIMS,
    )
    record(
        "external_review_still_required",
        decision.get("external_expert_review_required") is True,
    )
    record(
        "submission_novelty_not_authorized",
        decision.get("submission_novelty_authorized") is False,
    )
    record(
        "p2_theory_only_authorized",
        decision.get("p2_theory_implementation_authorized") is True,
    )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": REPORT_SCHEMA,
        "ledger_schema": ledger.get("schema"),
        "ledger_sha256": ledger_sha256,
        "counts": {
            "interfaces": len(interfaces),
            "families": len(families),
            "queries": len(queries),
            "sources": len(sources),
            "close_exclusions": len(exclusions),
        },
        "family_source_counts": dict(sorted(source_family_counts.items())),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    ledger, ledger_sha256 = _load_ledger(args.ledger)
    report = audit_novelty_ledger(ledger, ledger_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
