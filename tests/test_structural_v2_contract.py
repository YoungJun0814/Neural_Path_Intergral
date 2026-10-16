"""P0 declarations reject target confusion and learning/final stream reuse."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger
from src.path_integral.structural_v2_contract import (
    ControlBinding,
    MeasurementContract,
    audit_sample_uses,
)

D = "a" * 64
Q = "b" * 64
R = "c" * 64


def control() -> ControlBinding:
    params = {"basis": [[1.]], "centers": [[0.]], "directions": [[1.]],
              "lengthscales": [1.], "coefficients": [.2]}
    return ControlBinding(params, canonical_digest(params), canonical_digest(params["basis"]),
                          .25, "analytic_gaussian_window")


def raw(name: str = "raw") -> MeasurementContract:
    return MeasurementContract(name, D, 32, "event_probability", "ordinary_is",
                               "final-probability", "iid_draw", Q, None, final_rep=0)


def cv() -> MeasurementContract:
    return replace(raw("cv"), estimator_kind="event_stein_cv", control=control(), cv_fit_rep=0)


def use(name: str, consumer: str, key: SeedKey, pair: str | None = None) -> dict:
    return {"use_id": name, "consumer_id": consumer, "role": key.role,
            "seed_key": asdict(key), "pair_group": pair}


def key(role: str, stream: str = "path") -> SeedKey:
    return SeedKey("structural-v2-test", role, "development", "toy", 32, 0, stream)


@pytest.mark.parametrize("contract", [raw(), cv(),
    MeasurementContract("risk", D, 32, "raw_event_second_moment", "auxiliary_is",
                        "final-raw-risk", "iid_draw", Q, R, final_rep=0),
    MeasurementContract("smc", D, 32, "raw_event_second_moment", "smc_normalizer",
                        "reference-whole-smc", "whole_smc_run", Q, None, reference_rep=0),
    MeasurementContract("mesh", D, 32, "adjacent_probability_difference", "adjacent_coupled_is",
                        "final-probability", "iid_coupled_draw", Q, None, final_rep=0,
                        mesh_pair=(16, 32)),
    MeasurementContract("corrected", D, 32, "corrected_event_second_moment", "corrected_risk_is",
                        "final-cv-risk", "iid_draw", Q, R, control(), cv_fit_rep=0, final_rep=0),
])
def test_roundtrip(contract: MeasurementContract) -> None:
    assert MeasurementContract.from_dict(json.loads(json.dumps(contract.to_dict()))) == contract


@pytest.mark.parametrize("change", [
    {"estimand_kind": "raw_event_second_moment"}, {"se_unit": "whole_smc_run"},
    {"se_unit": "smc_particle"}, {"sample_role": "cv-training"},
    {"frozen_before_final": False}, {"self_normalized": True}, {"q_digest": None},
    {"q_digest": "BAD"}, {"r_digest": R}, {"control": control()}, {"final_rep": None},
    {"parent_training_rep": True}, {"grid_steps": 32.0}, {"mesh_pair": (16, 32)},
])
def test_raw_rejects_bad_bindings(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(raw(), **change)


def test_corrected_cannot_be_recorded_as_raw_risk() -> None:
    corrected = MeasurementContract("corrected", D, 32, "corrected_event_second_moment", "corrected_risk_is",
                                    "final-cv-risk", "iid_draw", Q, R, control(), cv_fit_rep=0, final_rep=0)
    with pytest.raises(ValueError, match="mismatch"):
        replace(corrected, estimand_kind="raw_event_second_moment")
    with pytest.raises(ValueError, match="CV fit"):
        replace(cv(), cv_fit_rep=None)


def test_control_mutation_and_empirical_bound_rejected() -> None:
    binding = control()
    with pytest.raises(ValueError, match="global bound"):
        replace(binding, bound_method="sampled_maximum")
    binding.parameters["centers"][0][0] = 2.
    with pytest.raises(ValueError, match="digest"):
        ControlBinding(**asdict(binding))


def test_explicit_same_q_pairing_and_training_split() -> None:
    ledger = SeedLedger()
    train, final = key("cv-training"), key("final-probability")
    ledger.allocate(train)
    ledger.allocate(final)
    uses = [use("train", "cv", train), use("raw-final", "raw", final, "pair-0"),
            use("cv-final", "cv", final, "pair-0")]
    outcome = audit_sample_uses(uses, ledger, [raw(), cv()], expected_use_ids={u["use_id"] for u in uses})
    assert outcome["paired_groups"] == 1
    assert outcome["seed_streams"] == 2
    with pytest.raises(ValueError, match="same-target"):
        audit_sample_uses(uses, ledger, [raw(), replace(cv(), q_digest=R)],
                          expected_use_ids={u["use_id"] for u in uses})
    bad = copy.deepcopy(uses)
    bad[0]["role"] = "final-probability"
    with pytest.raises(ValueError, match="role"):
        audit_sample_uses(bad, ledger, [raw(), cv()], expected_use_ids={u["use_id"] for u in bad})


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unused_seed", "unpaired_reuse", "false_pair"])
def test_completion_and_pairing_fail_closed(kind: str) -> None:
    ledger = SeedLedger()
    final = key("final-probability")
    ledger.allocate(final)
    uses = [use("raw-final", "raw", final)]
    contracts = [raw()]
    expected = {"raw-final"}
    if kind == "missing":
        expected.add("missing")
    elif kind == "duplicate":
        uses.append(copy.deepcopy(uses[0]))
    elif kind == "unused_seed":
        ledger.allocate(key("selection"))
    elif kind == "unpaired_reuse":
        uses.append(use("cv-final", "cv", final))
        contracts.append(cv())
        expected.add("cv-final")
    else:
        uses[0]["pair_group"] = "false-pair"
    with pytest.raises(ValueError):
        audit_sample_uses(uses, ledger, contracts, expected_use_ids=expected)


@pytest.mark.parametrize("change", [{"level": True}, {"replicate": 1.5}, {"role": 1}])
def test_seed_key_type_hardening_preserves_old_valid_seeds(change: dict) -> None:
    payload = asdict(key("final-probability"))
    payload.update(change)
    with pytest.raises(ValueError):
        SeedKey(**payload)


def test_schema_unknown_fields_and_stale_controls_rejected() -> None:
    payload = raw().to_dict()
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="schema"):
        MeasurementContract.from_dict(payload)
    contract = cv()
    assert contract.control is not None
    contract.control.parameters["coefficients"][0] = 5.
    ledger = SeedLedger()
    stream = key("final-probability")
    ledger.allocate(stream)
    with pytest.raises(ValueError, match="digest"):
        audit_sample_uses([use("final", "cv", stream)], ledger, [contract], expected_use_ids={"final"})
