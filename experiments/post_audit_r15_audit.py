"""Structural and arithmetic audit of R1.5 development artifacts.

This is not a replay, proof, or scientific qualification. It checks strict
JSON, seed ledgers, hash-bound dependencies, and independently recomputable
statistics. Repository-wide source-tree hashes are reported separately:
incremental development added files after some earlier runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from src.path_integral.research_result_contract import canonical_digest, source_tree_digest
from src.path_integral.seed_ledger import SeedLedger

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "reference_design", "reference_crosscheck", "mixture_design",
    "reference_refinement", "mixture_independent", "fresh_ablation",
    "is_precision", "fixed_precision_total_work", "hybrid_bank_pilot",
    "ce_precision",
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _close(actual: float, expected: float, label: str) -> None:
    if not (math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-30)):
        raise ValueError(f"{label} mismatch: {actual} versus {expected}")


def _load(name: str) -> tuple[dict[str, Any], str]:
    path = ROOT / f"results/post_audit/r15_{name}_v1.json"
    raw = path.read_bytes()
    payload = json.loads(
        raw, object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON constant: {value}")
        ),
    )
    if payload["source"]["config_digest"] != canonical_digest(payload["config"]):
        raise ValueError(f"{name}: configuration digest mismatch")
    if not payload["cells"]:
        raise ValueError(f"{name}: no cells")
    SeedLedger.from_dict(payload["seed_ledger"])
    return payload, hashlib.sha256(raw).hexdigest()


def _smc(record: dict[str, Any], label: str) -> None:
    values = [math.exp(x) for x in record["log_replicate_estimates"]]
    if len(values) < 2:
        raise ValueError(f"{label}: need independent SMC replicates")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    _close(record["mean"], mean, f"{label} mean")
    _close(record["standard_error"], se, f"{label} standard error")
    _close(record["relative_se"], se / mean, f"{label} relative SE")


def _cluster_estimate(record: dict[str, Any], label: str,
                      cluster_logs: list[float | None]) -> None:
    if len(cluster_logs) < 2:
        raise ValueError(f"{label}: need independent clusters")
    finite = [x for x in cluster_logs if x is not None]
    scale = max(finite)
    scaled = [math.exp(x - scale) if x is not None else 0.0 for x in cluster_logs]
    log_mean = scale + math.log(statistics.mean(scaled))
    rse = statistics.stdev(scaled) / math.sqrt(len(scaled)) / statistics.mean(scaled)
    _close(record["log_mean"], log_mean, f"{label} log mean")
    _close(record["between_cluster_relative_se"], rse, f"{label} cluster RSE")
    _close(record["relative_se"], max(rse, record["iid_relative_se"]),
           f"{label} robust RSE")


def _links(data: dict[str, dict[str, Any]], sha: dict[str, str]) -> None:
    direct = (
        ("reference_crosscheck", "design_artifact_sha256", "reference_design"),
        ("mixture_design", "design_artifact_sha256", "reference_design"),
        ("mixture_design", "reference_artifact_sha256", "reference_crosscheck"),
        ("reference_refinement", "design_artifact_sha256", "reference_design"),
        ("reference_refinement", "first_reference_artifact_sha256", "reference_crosscheck"),
        ("mixture_independent", "smc_design_artifact_sha256", "reference_design"),
        ("mixture_independent", "mixture_design_artifact_sha256", "mixture_design"),
        ("mixture_independent", "refined_reference_artifact_sha256", "reference_refinement"),
    )
    for child, field, parent in direct:
        if data[child][field] != sha[parent]:
            raise ValueError(f"{child}: {parent} artifact binding mismatch")
    bundles = {
        "fresh_ablation": {
            "smc_design": "reference_design",
            "first_crosscheck": "reference_crosscheck",
            "mixture_crosscheck": "mixture_independent",
            "mixture_design": "mixture_design",
            "refined_reference": "reference_refinement",
        },
        "is_precision": {
            "smc_design": "reference_design",
            "mixture_crosscheck": "mixture_independent",
            "refined_reference": "reference_refinement",
        },
        "fixed_precision_total_work": {
            "smc_design": "reference_design",
            "mixture_design": "mixture_design",
            "refined_reference": "reference_refinement",
            "is_precision": "is_precision",
        },
        "hybrid_bank_pilot": {
            "smc_design": "reference_design",
            "first_crosscheck": "reference_crosscheck",
            "mixture_crosscheck": "mixture_independent",
            "mixture_design": "mixture_design",
        },
        "ce_precision": {
            "smc_design": "reference_design",
            "ce_crosscheck": "reference_crosscheck",
            "refined_reference": "reference_refinement",
            "smc_is_precision": "is_precision",
        },
    }
    for child, bindings in bundles.items():
        for field, parent in bindings.items():
            if data[child]["input_artifact_sha256"][field] != sha[parent]:
                raise ValueError(f"{child}: {parent} artifact binding mismatch")


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if not key.endswith("seconds")
            and key not in ("cost_normalized_cv2", "fixed_precision_total_wall_ratio_smc_over_is")
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _replay(data: dict[str, dict[str, Any]], names: tuple[str, ...]) -> list[str]:
    from experiments.post_audit_r15_ce_precision import run as ce_precision
    from experiments.post_audit_r15_fixed_precision_total_work import run as fixed_work
    from experiments.post_audit_r15_fresh_ablation import run as fresh_ablation
    from experiments.post_audit_r15_hybrid_bank_pilot import run as hybrid_bank
    from experiments.post_audit_r15_is_precision import run as is_precision
    from experiments.post_audit_r15_mixture_design import run as mixture_design
    from experiments.post_audit_r15_mixture_independent import run as mixture_independent
    from experiments.post_audit_r15_reference_crosscheck import run as crosscheck
    from experiments.post_audit_r15_reference_design import run as design
    from experiments.post_audit_r15_reference_refinement import run as refinement

    runners = {
        "reference_design": design,
        "reference_crosscheck": crosscheck,
        "mixture_design": mixture_design,
        "reference_refinement": refinement,
        "mixture_independent": mixture_independent,
        "fresh_ablation": fresh_ablation,
        "is_precision": is_precision,
        "fixed_precision_total_work": fixed_work,
        "hybrid_bank_pilot": hybrid_bank,
        "ce_precision": ce_precision,
    }
    replayed = []
    for name in names:
        fresh = runners[name](data[name]["config"])
        if canonical_digest(fresh["seed_ledger"]) != canonical_digest(data[name]["seed_ledger"]):
            raise ValueError(f"{name}: seed ledger changed on replay")
        if canonical_digest(_without_timing(fresh["cells"])) != canonical_digest(
            _without_timing(data[name]["cells"])
        ):
            raise ValueError(f"{name}: numerical result changed on replay")
        for field in ("selected_candidate_id", "selected_candidate", "selection_status"):
            if field in data[name] and fresh[field] != data[name][field]:
                raise ValueError(f"{name}: {field} changed on replay")
        replayed.append(name)
        print(json.dumps({"replayed": name}, allow_nan=False), flush=True)
    return replayed


def audit(*, replay: bool = False, replay_only: tuple[str, ...] = ()) -> dict[str, Any]:
    loaded = {name: _load(name) for name in NAMES}
    data = {name: value[0] for name, value in loaded.items()}
    sha = {name: value[1] for name, value in loaded.items()}
    _links(data, sha)
    # Seed ledgers from separate runs must not reuse one random stream.
    records = [
        record for name in NAMES
        for record in SeedLedger.from_dict(data[name]["seed_ledger"]).records
    ]
    SeedLedger(records)
    cells = set()
    for name in NAMES:
        ids = [x["cell"]["id"] for x in data[name]["cells"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate cell")
        cells.update(ids)
    for cell in data["reference_crosscheck"]["cells"]:
        _smc(cell["smc_reference"], "first reference")
    for cell in data["reference_refinement"]["cells"]:
        _smc(cell["new_reference"], "refined reference")
    for cell in data["mixture_independent"]["cells"]:
        _cluster_estimate(
            {"log_mean": cell["is_log_mean"],
             "between_cluster_relative_se": cell["is_between_cluster_relative_se"],
             "relative_se": cell["is_relative_se"],
             "iid_relative_se": cell["is_iid_relative_se"]},
            "mixture IS", [x["conditional"]["log_mean"] for x in cell["cluster_records"]],
        )
        if cell["qualified"] != (
            cell["is_relative_se"] <= data["mixture_independent"]["config"]["qualification"]["maximum_relative_se"]
            and cell["reference_relative_se"] <=
            data["mixture_independent"]["config"]["qualification"]["maximum_reference_relative_se"]
            and cell["equivalence_upper_difference"] <= cell["equivalence_margin"]
        ):
            raise ValueError("mixture IS qualification mismatch")
    for cell in data["is_precision"]["cells"]:
        _cluster_estimate(
            cell, "precision IS",
            [x["conditional"]["log_mean"] for x in cell["cluster_records"]],
        )
        _close(cell["mean"], math.exp(cell["log_mean"]), "precision IS mean")
        policy = data["is_precision"]["config"]["qualification"]
        if cell["qualified"] != (
            cell["relative_se"] <= policy["maximum_relative_se"]
            and cell["reference_relative_se"] <= policy["maximum_reference_relative_se"]
            and cell["equivalence_upper_difference"] <= cell["equivalence_margin"]
        ):
            raise ValueError("precision IS qualification mismatch")
    for cell in data["ce_precision"]["cells"]:
        _cluster_estimate(
            cell, "CE precision IS",
            [x["conditional"]["log_mean"] for x in cell["cluster_records"]],
        )
        _close(cell["mean"], math.exp(cell["log_mean"]), "CE precision mean")
        if cell["qualified"] != (
            cell["equivalence_vs_smc"]["pass"]
            and cell["equivalence_vs_smc_is"]["pass"]
        ):
            raise ValueError("CE precision qualification mismatch")
    ablation = data["fresh_ablation"]
    for cell in ablation["cells"]:
        if len(cell["repetitions"]) != 2 * ablation["config"]["independent_training_repetitions"]:
            raise ValueError("fresh ablation repetition count mismatch")
        for rep in cell["repetitions"]:
            if len(rep["methods"]) != 8:
                raise ValueError("fresh ablation method count mismatch")
            for method in rep["methods"]:
                n = ablation["config"]["final_count"]
                log_ratio = method["log_second_moment"] - 2 * method["log_mean"]
                expected_rse = math.sqrt(max(0.0, math.expm1(log_ratio)) / (n - 1))
                _close(method["relative_se"], expected_rse, "ablation RSE")
                _close(sum(method["mode_contribution_shares"]), 1.0, "mode contribution mass")
                _close(sum(method["mode_sample_fractions"]), 1.0, "mode sample mass")
    for cell in data["fixed_precision_total_work"]["cells"]:
        _smc({
            "log_replicate_estimates": cell["smc"]["log_replicate_estimates"],
            "mean": cell["smc"]["mean"],
            "standard_error": cell["smc"]["mean"] * cell["smc"]["relative_se"],
            "relative_se": cell["smc"]["relative_se"],
        }, "timed SMC")
        _cluster_estimate(
            cell["is"], "timed IS", cell["is"]["cluster_logmeans"],
        )
        for method in ("smc", "is"):
            _close(
                cell[method]["total_wall_seconds"],
                sum(cell[method]["stages"].values()),
                f"{method} total wall",
            )
        expected_pair = (
            cell["smc"]["accuracy"]["pass"]
            and cell["is"]["accuracy"]["pass"]
            and cell["pair_accuracy"]["pass"]
        )
        if cell["both_qualified"] != expected_pair:
            raise ValueError("timing accuracy gate mismatch")
        if expected_pair:
            _close(
                cell["fixed_precision_total_wall_ratio_smc_over_is"],
                cell["smc"]["total_wall_seconds"] / cell["is"]["total_wall_seconds"],
                "fixed-precision ratio",
            )
        elif cell["fixed_precision_total_wall_ratio_smc_over_is"] is not None:
            raise ValueError("unqualified timing ratio not withheld")
    current = source_tree_digest(ROOT)
    mismatches = [
        name for name in NAMES
        if data[name]["source"]["source_tree_digest"] != current
    ]
    replayed = _replay(data, replay_only or NAMES) if replay else []
    return {
        "status": "arithmetic_and_binding_pass",
        "artifact_count": len(NAMES),
        "cell_ids": sorted(cells),
        "combined_seed_count": len(records),
        "source_tree_digest_matches": sorted(set(NAMES) - set(mismatches)),
        "source_tree_digest_mismatches": mismatches,
        "full_numerical_replay": replay and not replay_only,
        "replayed_artifacts": replayed,
        "scope": "development_integrity_not_independent_replication_or_confirmation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-only", choices=NAMES, nargs="+")
    args = parser.parse_args()
    print(json.dumps(audit(replay=args.replay or bool(args.replay_only),
                           replay_only=tuple(args.replay_only or ())),
                     sort_keys=True, allow_nan=False))
