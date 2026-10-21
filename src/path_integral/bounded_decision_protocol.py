"""Immutable bounded-study counts, allocation and development-only gates."""

from __future__ import annotations

import math
from typing import Any

ARMS = ("U", "E-IID", "E-STRAT")
DESIGNS = ("iid", "stratified")
PRODUCTION_CAP = 13479520
AUDIT_CAP = 1000000


def allocate(rse: float, block_rse: float, *, kind: str) -> dict[str, Any]:
    if kind not in ("probability", "risk"):
        raise ValueError("declared allocation endpoint required")
    pilot, target, unit, low, high = ((16384, .05, 16384, 65536, 262144) if kind == "probability"
        else (19840, .2, 9920, 39680, 158720))
    safe = max(rse, block_rse)
    if any(not math.isfinite(v) or v < 0 for v in (rse, block_rse)) or safe == 0:
        return {"status": "unallocated_unresolved_variance", "count": None, "required": None}
    required = max(low, math.ceil((2*pilot*(safe/target)**2)/unit)*unit)
    return {"status": "allocated" if required <= high else "unallocated_precision_budget",
        "count": required if required <= high else None, "required": required,
        "cap": high, "target_rse": target, "safety_factor": 2., "pilot_not_guarantee": True}


def precision_gate(summary: dict[str, Any], sensitivities: list[dict[str, Any]], *, probability_first: bool) -> list[bool]:
    result = []
    for k, raw in enumerate(summary["relative_se"]):
        s = sensitivities[k]
        threshold = .05 if probability_first and k == 0 else .2
        result.append(bool(raw <= threshold and s["between_unit_relative_se"] <= 2*raw
            and s["maximum_leave_one_out_relative_shift"] <= .15 and summary["maximum_fraction"][k] <= .1))
    return result


def d2_decision(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cells) != 2 or len({c["cell"] for c in cells}) != 2:
        raise ValueError("complete two-cell decision required")
    passed = all(all(c["methods"]["stratified"]["gates"])
        and not any(x["contradiction"] for x in c["contrasts"]) for c in cells)
    return {"d2_gate": passed, "decision": "D3_AUTHORIZED_DEVELOPMENT_ONLY" if passed else "STOP_UNRESOLVED_BUDGET",
        "iid_qualified_in_both_cells": all(all(c["methods"]["iid"]["gates"]) for c in cells),
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}


def repeated_decision(rows: list[dict[str, Any]], cells: tuple[str, ...]) -> dict[str, Any]:
    if len(rows) != 20 or {(r["cell"], r["rep"], r["arm"]) for r in rows} != {
            (c, rep, a) for c in cells for rep in range(5) for a in ARMS[1:]}:
        raise ValueError("complete two-cell five-repetition two-candidate family required")
    eligible = [a for a in ARMS[1:] if all(sum(r["evaluable"] for r in rows
        if (r["cell"], r["arm"]) == (c, a)) >= 4 for c in cells)]
    successful = [a for a in eligible if all(sum(r["success"] for r in rows
        if (r["cell"], r["arm"]) == (c, a)) >= 4 for c in cells)]
    practical = [a for a in successful if all(sum(r["success"] and r["cost_ratio"] <= .8 for r in rows
        if (r["cell"], r["arm"]) == (c, a)) >= 4 for c in cells)]
    bank = "E-STRAT" in practical and all(sum(r["bank_effect"] for r in rows
        if (r["cell"], r["arm"]) == (c, "E-STRAT")) >= 4 for c in cells)
    decision = ("STOP_UNRESOLVED_BUDGET" if not eligible else "PIVOT_WEIGHT_ONLY" if not successful
        else "PIVOT_PRACTICAL_CLAIM" if not practical else "CONTINUE_CANDIDATE" if bank else "CONTINUE_MODEL_ONLY")
    return {"decision": decision, "evaluable_arms": eligible, "performance_candidates": successful,
        "practical_candidates": practical, "bank_effect_candidate": bank,
        "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False}
