"""Independent reconstruction audit for V9 terminal RQMC references."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V9TerminalReferenceAudit:
    checks: tuple[tuple[str, bool], ...]
    failures: tuple[str, ...]
    maximum_relative_standard_error: float
    passed: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_v9_terminal_reference(
    *, config_path: Path, result: dict[str, Any], root: Path
) -> V9TerminalReferenceAudit:
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V9 reference config must be a mapping")
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise ValueError("V9 reference cells must be a list")
    declared = {str(cell["cell_id"]): cell for cell in config["cells"]}
    structure = len(cells) == len(declared) and {str(cell["cell_id"]) for cell in cells} == set(
        declared
    )
    values_valid = True
    costs_valid = True
    seeds: list[int] = []
    maximum_relative = 0.0
    for cell in cells:
        cell_id = str(cell["cell_id"])
        expected = declared.get(cell_id)
        units = [float(value) for value in cell.get("unit_estimates", [])]
        randomizations = int(
            config["randomizations_by_cell"][cell_id]
            if config.get("schema") == "npi.g11.v9-terminal-reference.v2"
            else config["randomizations"]
        )
        points = int(config["points_per_randomization"])
        if expected is None or len(units) != randomizations or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0 for value in units
        ):
            values_valid = False
            continue
        estimate = statistics.fmean(units)
        standard_error = statistics.stdev(units) / math.sqrt(randomizations)
        relative = standard_error / estimate if estimate > 0.0 else math.inf
        maximum_relative = max(maximum_relative, relative)
        if (
            any(cell[key] != expected[key] for key in expected)
            or not math.isclose(float(cell["estimate"]), estimate, rel_tol=1e-12, abs_tol=1e-18)
            or not math.isclose(
                float(cell["standard_error"]), standard_error, rel_tol=1e-12, abs_tol=1e-18
            )
            or not math.isclose(
                float(cell["relative_standard_error"]), relative, rel_tol=1e-12
            )
            or bool(cell["relative_standard_error_pass"])
            != (relative <= float(config["maximum_relative_standard_error"]))
        ):
            values_valid = False
        cost = cell["cost"]
        raw_samples = randomizations * points
        steps = int(config["model"]["steps"])
        expected_work = raw_samples * (4 * steps + 1)
        if (
            int(cost["raw_samples"]) != raw_samples
            or int(cost["cdf_calls"]) != raw_samples
            or float(cost["algorithmic_work_units"]) != float(expected_work)
        ):
            costs_valid = False
        proposal_seed = int(cell["proposal_seed"])
        randomization_seeds = [int(value) for value in cell["randomization_seeds"]]
        if len(randomization_seeds) != randomizations or randomization_seeds != list(
            range(randomization_seeds[0], randomization_seeds[0] + randomizations)
        ):
            values_valid = False
        seeds.extend([proposal_seed, *randomization_seeds])

    bindings = True
    binding = config.get("claim_contract")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        bindings = False
    else:
        path = root / str(binding["path"])
        bindings = path.is_file() and _sha256(path) == str(binding["sha256"])
    allocation_reconstruction = True
    if config.get("schema") == "npi.g11.v9-terminal-reference.v2":
        pilot_bindings = config.get("pilot_bindings")
        if not isinstance(pilot_bindings, dict):
            bindings = False
            allocation_reconstruction = False
        else:
            for pilot_binding in pilot_bindings.values():
                if not isinstance(pilot_binding, dict):
                    bindings = False
                    break
                pilot_path = root / str(pilot_binding["path"])
                if not pilot_path.is_file() or _sha256(pilot_path) != str(
                    pilot_binding["sha256"]
                ):
                    bindings = False
                    break
            if bindings:
                pilot = json.loads(
                    (
                        root / str(pilot_bindings["reference_v1"]["path"])
                    ).read_text(encoding="utf-8")
                )
                rule = config["allocation_rule"]
                expected_allocation: dict[str, int] = {}
                for pilot_cell in pilot["cells"]:
                    relative = float(pilot_cell["relative_standard_error"])
                    required = max(
                        int(rule["minimum_randomizations"]),
                        math.ceil(
                            float(rule["variance_safety_factor"])
                            * int(rule["pilot_randomizations"])
                            * (
                                relative
                                / float(config["maximum_relative_standard_error"])
                            )
                            ** 2
                        ),
                    )
                    expected_allocation[str(pilot_cell["cell_id"])] = 1 << (
                        required - 1
                    ).bit_length()
                allocation_reconstruction = expected_allocation == {
                    str(key): int(value)
                    for key, value in config["randomizations_by_cell"].items()
                }
    seed_hash = hashlib.sha256(
        json.dumps(sorted(seeds), separators=(",", ":")).encode()
    ).hexdigest()
    complete = bool(cells) and all(
        bool(cell.get("relative_standard_error_pass")) for cell in cells
    )
    expected_decision = {
        "reference_complete": complete,
        "proposal_bank_authorized": complete,
        "benchmark_authorized": False,
        "performance_claim_authorized": False,
        "submission_authorized": False,
    }
    expected_schema = (
        "npi.g11.v9-terminal-reference-result.v2"
        if config.get("schema") == "npi.g11.v9-terminal-reference.v2"
        else "npi.g11.v9-terminal-reference-result.v1"
    )
    pilot_disjoint = True
    if config.get("schema") == "npi.g11.v9-terminal-reference.v2":
        pilot_binding = config.get("pilot_bindings", {}).get("reference_v1")
        if not isinstance(pilot_binding, dict):
            pilot_disjoint = False
        else:
            pilot_path = root / str(pilot_binding["path"])
            if not pilot_path.is_file() or _sha256(pilot_path) != str(pilot_binding["sha256"]):
                pilot_disjoint = False
            else:
                pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
                pilot_seeds = {
                    int(seed)
                    for cell in pilot["cells"]
                    for seed in [cell["proposal_seed"], *cell["randomization_seeds"]]
                }
                pilot_disjoint = not (set(seeds) & pilot_seeds)
    checks = (
        ("schema", result.get("schema") == expected_schema),
        ("config_hash", result.get("config_sha256") == hashlib.sha256(raw).hexdigest()),
        ("bindings", bindings),
        ("allocation_reconstruction", allocation_reconstruction),
        ("cell_roster", structure),
        ("randomization_reconstruction", values_valid),
        ("cost_reconstruction", costs_valid),
        ("seed_uniqueness", len(seeds) == len(set(seeds))),
        ("seed_count", result.get("seed_count") == len(seeds)),
        ("seed_hash", result.get("seed_set_sha256") == seed_hash),
        ("fresh_v2_seeds", pilot_disjoint),
        ("clean_source_generation", result.get("dirty_worktree") is False),
        ("decision_locks", result.get("decision") == expected_decision),
    )
    failures = tuple(name for name, passed in checks if not passed)
    return V9TerminalReferenceAudit(
        checks=checks,
        failures=failures,
        maximum_relative_standard_error=maximum_relative,
        passed=not failures,
    )
