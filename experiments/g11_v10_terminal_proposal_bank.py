"""Train the fresh V10 full-dimensional defensive CEM proposal bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.path_integral.baselines.cem import CEMTrainingConfig
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.provenance import runtime_provenance, source_provenance
from src.path_integral.v10_proposal_bank import V10BankCell, train_v10_proposal_bank

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "npi.g11.v10-terminal-proposal-bank.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError("unexpected V10 terminal proposal-bank config schema")
    return config, hashlib.sha256(raw).hexdigest()


def _validate(config: dict[str, Any]) -> None:
    binding = config.get("claim_contract")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("V10 proposal bank requires one claim-contract binding")
    path = ROOT / str(binding["path"])
    if not path.is_file() or _sha256(path) != str(binding["sha256"]):
        raise ValueError("V10 proposal-bank claim binding mismatch")
    claim = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["model"] != claim["model"] or config["cells"] != claim["cells"]:
        raise ValueError("V10 proposal-bank model/cells differ from the claim contract")
    if len(config["cells"]) != 12 or any(
        cell["task"] != "terminal_left_tail" for cell in config["cells"]
    ):
        raise ValueError("V10 proposal bank is exactly the 12-cell terminal roster")
    if config.get("current_namespace_outcomes_inspected_before_freeze") is not False:
        raise ValueError("V10 proposal-bank namespace was not outcome-blind at freeze")


def _cells(config: dict[str, Any]) -> tuple[V10BankCell, ...]:
    return tuple(
        V10BankCell(
            cell_id=str(cell["cell_id"]),
            task=TerminalThresholdTask(float(cell["threshold"])),
            hurst=float(cell["hurst"]),
        )
        for cell in config["cells"]
    )


def run(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    _validate(config)
    training = config["training"]
    cem_config = CEMTrainingConfig(
        iterations=int(training["iterations"]),
        samples_per_iteration=int(training["samples_per_iteration"]),
        elite_fraction=float(training["elite_fraction"]),
        smoothing=float(training["smoothing"]),
        defensive_weight=float(training["defensive_weight"]),
        max_mean_norm=float(training["max_mean_norm"]),
    )
    bank = train_v10_proposal_bank(
        _cells(config),
        spot=float(config["model"]["spot"]),
        maturity=float(config["model"]["maturity"]),
        steps=int(config["model"]["steps"]),
        eta=float(config["model"]["eta"]),
        xi=float(config["model"]["xi"]),
        rho=float(config["model"]["rho"]),
        base_seed=int(config["base_seed"]),
        cem_config=cem_config,
    )
    return {
        "schema": SCHEMA + "-result",
        "protocol_id": config["protocol_id"],
        "namespace": config["namespace"],
        "config_sha256": config_sha256,
        **source_provenance(),
        "environment": runtime_provenance(dtype="torch.float64"),
        "entries": [asdict(entry) for entry in bank.entries],
        "entry_count": len(bank.entries),
        "seed_count": len(bank.entries),
        "bank_sha256": bank.bank_sha256,
        "total_training_cost": asdict(bank.total_training_cost),
        "amortization_query_counts": list(config["amortization_query_counts"]),
        "decision": {
            "bank_construction_complete": True,
            "development_benchmark_authorized": True,
            "qualification_authorized": False,
            "performance_claim_authorized": False,
            "submission_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        args.output.unlink()
    config, digest = load_config(args.config)
    result = run(config, digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
