"""Additive V2 measurement/role contracts; metadata is not a proof certificate."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.path_integral.research_result_contract import canonical_digest
from src.path_integral.seed_ledger import SeedKey, SeedLedger

SCHEMA = "npi.structural-v2.measurement.v1"
ROLES = frozenset({
    "design-pilot", "parent-training", "bank-fitting", "cv-training",
    "allocation-pilot", "selection", "geometry-calibration", "reference-iid",
    "reference-whole-smc", "final-probability", "final-raw-risk",
    "final-cv-risk", "mesh-augmentation", "audit",
})
FINAL_ROLES = frozenset({"final-probability", "final-raw-risk", "final-cv-risk"})
# (estimand, estimator) -> (SE unit, allowed sample roles)
RULES = {
    ("event_probability", "ordinary_is"): ("iid_draw", {"final-probability", "reference-iid"}),
    ("event_probability", "conditional_mean_is"): ("iid_outer_draw", {"reference-iid"}),
    ("event_probability", "event_stein_cv"): ("iid_draw", {"final-probability"}),
    ("event_probability", "smc_normalizer"): ("whole_smc_run", {"reference-whole-smc"}),
    ("raw_event_second_moment", "auxiliary_is"): ("iid_draw", {"final-raw-risk", "reference-iid"}),
    ("raw_event_second_moment", "nested_auxiliary_is"): ("iid_outer_draw", {"reference-iid"}),
    ("raw_event_second_moment", "auxiliary_stein_cv"): ("iid_draw", {"final-raw-risk"}),
    ("raw_event_second_moment", "smc_normalizer"): ("whole_smc_run", {"reference-whole-smc"}),
    ("corrected_event_second_moment", "corrected_risk_is"): ("iid_draw", {"final-cv-risk"}),
    ("auxiliary_estimator_second_moment", "auxiliary_squared_is"): ("iid_draw", {"final-raw-risk"}),
    ("adjacent_probability_difference", "adjacent_coupled_is"): ("iid_coupled_draw", {"final-probability"}),
}
CV_ESTIMATORS = frozenset({"event_stein_cv", "auxiliary_stein_cv", "corrected_risk_is"})
AUX_ESTIMATORS = frozenset({"auxiliary_is", "nested_auxiliary_is", "auxiliary_stein_cv", "corrected_risk_is", "auxiliary_squared_is"})


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty stripped text")


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def _index(value: Any, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ControlBinding:
    """Bind saved parameters and a declared analytic bound, not certify its proof."""

    parameters: dict[str, Any]
    field_digest: str
    basis_digest: str
    absolute_bound: float
    bound_method: str

    def __post_init__(self) -> None:
        required = {"basis", "centers", "directions", "lengthscales", "coefficients"}
        if not isinstance(self.parameters, dict) or set(self.parameters) != required:
            raise ValueError("control parameters must include the complete analytic dictionary")
        _digest(self.field_digest, "field_digest")
        _digest(self.basis_digest, "basis_digest")
        if canonical_digest(self.parameters) != self.field_digest:
            raise ValueError("control parameter digest mismatch")
        if canonical_digest(self.parameters["basis"]) != self.basis_digest:
            raise ValueError("control basis digest mismatch")
        if (isinstance(self.absolute_bound, bool) or not isinstance(self.absolute_bound, (int, float))
                or not math.isfinite(self.absolute_bound) or self.absolute_bound < 0):
            raise ValueError("control bound must be finite and nonnegative")
        if self.bound_method != "analytic_gaussian_window":
            raise ValueError("a sampled/grid maximum is not an analytic global bound")


@dataclass(frozen=True)
class MeasurementContract:
    consumer_id: str
    target_digest: str
    grid_steps: int
    estimand_kind: str
    estimator_kind: str
    sample_role: str
    se_unit: str
    q_digest: str | None
    r_digest: str | None
    control: ControlBinding | None = None
    parent_training_rep: int | None = None
    auxiliary_fit_rep: int | None = None
    cv_fit_rep: int | None = None
    selection_rep: int | None = None
    final_rep: int | None = None
    reference_rep: int | None = None
    mesh_pair: tuple[int, int] | None = None
    frozen_before_final: bool = True
    self_normalized: bool = False

    def __post_init__(self) -> None:
        _text(self.consumer_id, "consumer_id")
        _digest(self.target_digest, "target_digest")
        _index(self.grid_steps, "grid_steps", minimum=1)
        for name in ("estimand_kind", "estimator_kind", "sample_role", "se_unit"):
            _text(getattr(self, name), name)
        rule = RULES.get((self.estimand_kind, self.estimator_kind))
        if rule is None or self.se_unit != rule[0] or self.sample_role not in rule[1]:
            raise ValueError("estimand/estimator/SE/sample-role mismatch")
        if self.frozen_before_final is not True or self.self_normalized is not False:
            raise ValueError("frozen ordinary, non-self-normalized sampling is required")
        for name in ("q_digest", "r_digest"):
            if (value := getattr(self, name)) is not None:
                _digest(value, name)
        if self.q_digest is None and not (
            self.estimator_kind == "smc_normalizer" and self.estimand_kind == "event_probability"
        ):
            raise ValueError("this estimand/estimator requires a frozen q binding")
        if (self.r_digest is not None) != (self.estimator_kind in AUX_ESTIMATORS):
            raise ValueError("auxiliary r binding mismatch")
        if (self.control is not None) != (self.estimator_kind in CV_ESTIMATORS):
            raise ValueError("raw/corrected control binding mismatch")
        if self.control is not None and not isinstance(self.control, ControlBinding):
            raise ValueError("control must be a validated binding")
        for name in ("parent_training_rep", "auxiliary_fit_rep", "cv_fit_rep",
                     "selection_rep", "final_rep", "reference_rep"):
            if (value := getattr(self, name)) is not None:
                _index(value, name)
        if self.control is not None and self.cv_fit_rep is None:
            raise ValueError("CV fit identity is required")
        if self.sample_role in FINAL_ROLES and self.final_rep is None:
            raise ValueError("final identity is required")
        if self.sample_role.startswith("reference-") and self.reference_rep is None:
            raise ValueError("reference identity is required")
        if self.estimator_kind == "adjacent_coupled_is":
            if not isinstance(self.mesh_pair, tuple) or len(self.mesh_pair) != 2:
                raise ValueError("adjacent mesh pair is required")
            coarse, fine = self.mesh_pair
            _index(coarse, "coarse steps", minimum=1)
            _index(fine, "fine steps", minimum=1)
            if fine != 2 * coarse or fine != self.grid_steps:
                raise ValueError("mesh pair must be adjacent and match the fine grid")
        elif self.mesh_pair is not None:
            raise ValueError("mesh pair on a non-coupled estimator")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasurementContract:
        fields = set(cls.__dataclass_fields__)
        if set(payload) != fields | {"schema"} or payload["schema"] != SCHEMA:
            raise ValueError("unsupported or incomplete V2 measurement schema")
        values = {k: v for k, v in payload.items() if k != "schema"}
        if values["control"] is not None:
            if not isinstance(values["control"], dict) or set(values["control"]) != set(ControlBinding.__dataclass_fields__):
                raise ValueError("invalid control-binding schema")
            values["control"] = ControlBinding(**values["control"])
        if values["mesh_pair"] is not None:
            if not isinstance(values["mesh_pair"], (tuple, list)):
                raise ValueError("invalid mesh pair")
            values["mesh_pair"] = tuple(values["mesh_pair"])
        return cls(**values)


def audit_sample_uses(
    uses: list[dict[str, Any]], ledger: SeedLedger,
    contracts: list[MeasurementContract], *, expected_use_ids: set[str],
) -> dict[str, Any]:
    """Require complete use binding; only declared same-target final pairing is allowed.

    Different seeds do not prove independence or absence of data leakage. This
    checks declarations, not execution chronology or mathematical field validity.
    """
    contracts = [MeasurementContract.from_dict(c.to_dict()) for c in contracts]
    by_consumer = {c.consumer_id: c for c in contracts}
    if len(by_consumer) != len(contracts) or not contracts:
        raise ValueError("duplicate or missing measurement consumer")
    seen: set[str] = set()
    groups: dict[int, list[dict[str, Any]]] = {}
    paired_members: dict[str, frozenset[str]] = {}
    used_consumers: set[str] = set()
    for use in uses:
        if set(use) != {"use_id", "consumer_id", "role", "seed_key", "pair_group"}:
            raise ValueError("invalid sample-use schema")
        for name in ("use_id", "consumer_id"):
            _text(use[name], name)
        if use["use_id"] in seen:
            raise ValueError("duplicate sample use")
        seen.add(use["use_id"])
        key = SeedKey(**use["seed_key"])
        if key.role not in ROLES or use["role"] != key.role:
            raise ValueError("undeclared V2 sample role")
        contract = by_consumer.get(use["consumer_id"])
        if contract is None:
            raise ValueError("undeclared measurement consumer")
        used_consumers.add(contract.consumer_id)
        # Training/calibration streams may be declared by a final consumer, but
        # a stream in an inferential role must match that consumer's role.
        if key.role in FINAL_ROLES | {"reference-iid", "reference-whole-smc"}:
            if key.role != contract.sample_role:
                raise ValueError("inferential role reused as a different role")
        if use["pair_group"] is not None:
            _text(use["pair_group"], "pair_group")
            if key.role not in FINAL_ROLES:
                raise ValueError("pairing is allowed only for final IID streams")
        groups.setdefault(ledger.lookup(key), []).append(use)
    if seen != expected_use_ids or used_consumers != set(by_consumer):
        raise ValueError("missing/undeclared sample uses or consumers")
    if set(groups) != {r.seed for r in ledger.records}:
        raise ValueError("unused or undeclared seed streams")
    for members in groups.values():
        group_names = {x["pair_group"] for x in members}
        if len(members) == 1:
            if group_names != {None}:
                raise ValueError("declared pairing has no paired consumer")
            continue
        if None in group_names or len(group_names) != 1:
            raise ValueError("random stream reuse without explicit pairing")
        cs = [by_consumer[x["consumer_id"]] for x in members]
        identities = {(c.target_digest, c.grid_steps, c.estimand_kind, c.se_unit, c.mesh_pair,
                       c.q_digest, c.r_digest, c.sample_role, c.final_rep)
                      for c in cs}
        if len(identities) != 1 or len({c.consumer_id for c in cs}) != len(cs):
            raise ValueError("paired streams must have distinct same-target consumers")
        if any(c.se_unit not in {"iid_draw", "iid_coupled_draw"} for c in cs):
            raise ValueError("SMC units cannot be paired as IID draws")
        name = members[0]["pair_group"]
        consumers = frozenset(c.consumer_id for c in cs)
        if name in paired_members and paired_members[name] != consumers:
            raise ValueError("inconsistent consumers across paired streams")
        paired_members[name] = consumers
    return {"status": "declaration_and_role_binding_pass_not_independence_certificate",
            "consumers": len(contracts), "seed_streams": len(ledger),
            "sample_uses": len(uses), "paired_groups": len(paired_members)}
