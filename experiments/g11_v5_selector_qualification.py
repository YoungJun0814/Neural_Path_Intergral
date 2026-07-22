"""Bounded-oracle qualification of the V5 finite-look crossover selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from src.path_integral import (
    SeedKey,
    SeedLedger,
    advance_sequential_crossover,
)
from src.path_integral.provenance import runtime_provenance, source_provenance


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get("schema") != (
        "npi.g11.v5-selector-qualification.config.v1"
    ):
        raise ValueError("unsupported V5 selector qualification config")
    return payload, hashlib.sha256(raw).hexdigest()


def _draw(specification: dict[str, Any], count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    distribution = specification["distribution"]
    if distribution == "rademacher":
        scale = math.sqrt(float(specification["variance"]))
        return scale * (2 * torch.randint(0, 2, (count,), generator=generator) - 1).to(
            torch.float64
        )
    if distribution == "bernoulli":
        probability = float(specification["probability"])
        return (torch.rand(count, generator=generator, dtype=torch.float64) < probability).to(
            torch.float64
        )
    raise ValueError("unsupported selector oracle distribution")


def _true_variance(specification: dict[str, Any]) -> float:
    if specification["distribution"] == "rademacher":
        return float(specification["variance"])
    probability = float(specification["probability"])
    return probability * (1.0 - probability)


def _absolute_bound(specification: dict[str, Any]) -> float:
    declared = float(specification["absolute_bound"])
    if specification["distribution"] == "rademacher":
        return max(declared, math.sqrt(float(specification["variance"])))
    return declared


def _quantile(values: list[float], probability: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability))


def run(config_path: Path, *, smoke: bool = False) -> dict[str, Any]:
    config, config_hash = _load(config_path)
    selection = config["selection"]
    looks = tuple(int(value) for value in selection["looks"])
    repetitions = 30 if smoke else int(config["repetitions"])
    ledger = SeedLedger()
    records: list[dict[str, Any]] = []
    for case in config["cases"]:
        case_id = str(case["id"])
        profiles = case["profiles"]
        profile_ids = tuple(sorted(profiles))
        candidate_profiles = {
            str(name): tuple(str(item) for item in terms)
            for name, terms in case["candidate_profiles"].items()
        }
        preprocessing = {
            str(name): float(value) for name, value in case["preprocessing_work"].items()
        }
        target = float(case["sampling_variance_target"])
        true_profile_variances = {
            name: _true_variance(specification) for name, specification in profiles.items()
        }
        true_work = {
            candidate: preprocessing[candidate]
            + math.fsum(
                math.sqrt(true_profile_variances[profile_id] * float(profiles[profile_id]["cost"]))
                for profile_id in terms
            )
            ** 2
            / target
            for candidate, terms in candidate_profiles.items()
        }
        oracle = min(true_work, key=lambda name: (true_work[name], name))
        for repetition in range(repetitions):
            maximum_samples = {
                profile_id: _draw(
                    profiles[profile_id],
                    looks[-1],
                    ledger.allocate(
                        SeedKey(
                            config["protocol_id"],
                            "selector-oracle",
                            case_id,
                            profile_id,
                            0,
                            repetition,
                            "sample",
                        )
                    ),
                )
                for profile_id in profile_ids
            }
            state = None
            coverage = True
            invalid_elimination = False
            for look_index, count in enumerate(looks):
                observations = {
                    profile_id: maximum_samples[profile_id][:count] for profile_id in profile_ids
                }
                state = advance_sequential_crossover(
                    observations,
                    absolute_bounds={
                        profile_id: _absolute_bound(profiles[profile_id])
                        for profile_id in profile_ids
                    },
                    costs_per_sample={
                        profile_id: float(profiles[profile_id]["cost"])
                        for profile_id in profile_ids
                    },
                    candidate_profiles=candidate_profiles,
                    preprocessing_work=preprocessing,
                    sampling_variance_target=target,
                    predeclared_looks=looks,
                    look_index=look_index,
                    familywise_alpha=float(selection["familywise_alpha"]),
                    simpler_candidate=str(case["simpler_candidate"]),
                    previous_state=state,
                    elimination_relative_tolerance=float(
                        selection["elimination_relative_tolerance"]
                    ),
                    practical_equivalence_relative_tolerance=float(
                        selection["practical_equivalence_relative_tolerance"]
                    ),
                )
                profile_map = {item.profile_id: item for item in state.profiles}
                coverage = coverage and all(
                    profile_map[name].moments.variance_interval[0] - 1e-14
                    <= true_profile_variances[name]
                    <= profile_map[name].moments.variance_interval[1] + 1e-14
                    for name in profile_ids
                )
                work_map = {item.candidate_id: item for item in state.candidate_work}
                coverage = coverage and all(
                    work_map[name].total_work_interval[0] - 1e-12
                    <= true_work[name]
                    <= work_map[name].total_work_interval[1] + 1e-12
                    for name in candidate_profiles
                )
                if coverage and oracle not in state.surviving_candidates:
                    invalid_elimination = True
                if state.stopped:
                    break
            assert state is not None and state.frozen_decision is not None
            selected = state.frozen_decision.selected_candidate
            records.append(
                {
                    "case_id": case_id,
                    "repetition": repetition,
                    "oracle_candidate": oracle,
                    "selected_candidate": selected,
                    "simultaneous_coverage": coverage,
                    "invalid_elimination_on_coverage_event": invalid_elimination,
                    "regret": true_work[selected] / true_work[oracle],
                    "look_index": state.look_index,
                    "cumulative_samples_per_profile": state.cumulative_sample_count,
                    "selection_work_fraction_of_full_profile": (
                        state.cumulative_sample_count / looks[-1]
                    ),
                    "stop_reason": state.stop_reason,
                    "efficiency_gate_eligible": bool(case.get("efficiency_gate_eligible", True)),
                    "eliminations": [asdict(item) for item in state.elimination_history],
                }
            )
    coverage_rate = sum(item["simultaneous_coverage"] for item in records) / len(records)
    regrets = [float(item["regret"]) for item in records]
    efficiency_records = [item for item in records if item["efficiency_gate_eligible"]]
    early_fraction = sum(
        item["selection_work_fraction_of_full_profile"] < 1.0 for item in efficiency_records
    ) / len(efficiency_records)
    invalid_count = sum(item["invalid_elimination_on_coverage_event"] for item in records)
    gates = {
        "coverage_at_least_declared": coverage_rate >= 1.0 - float(selection["familywise_alpha"]),
        "no_invalid_elimination_on_coverage_event": invalid_count == 0,
        "median_regret_at_most_10_percent": _quantile(regrets, 0.5) <= 1.10,
        "p90_regret_at_most_25_percent": _quantile(regrets, 0.9) <= 1.25,
        "early_stop_fraction_at_least_75_percent": early_fraction >= 0.75,
    }
    provenance = source_provenance()
    formal_readiness = {
        "frozen_config": bool(config.get("frozen")),
        "clean_source": not bool(provenance["dirty_worktree"]),
        "non_smoke": not smoke,
    }
    return {
        "schema": "npi.g11.v5-selector-qualification.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "smoke": smoke,
        "records": records,
        "summary": {
            "simultaneous_coverage_rate": coverage_rate,
            "invalid_elimination_count_on_coverage_event": invalid_count,
            "median_regret": _quantile(regrets, 0.5),
            "p90_regret": _quantile(regrets, 0.9),
            "early_stop_fraction": early_fraction,
        },
        "gates": gates,
        "formal_readiness": formal_readiness,
        "qualification_passed": (all(gates.values()) and all(formal_readiness.values())),
        "seed_ledger": ledger.to_dict(),
        "seed_ledger_sha256": ledger.sha256,
        "environment": runtime_provenance(dtype="torch.float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/g11_v5_selector_qualification.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, smoke=arguments.smoke)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
