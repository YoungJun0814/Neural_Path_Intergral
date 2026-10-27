import numpy as np
import pytest

from experiments.post_audit_conditional_intervention import store_array
from experiments.post_audit_conditional_intervention_witness import select_extremes


def test_selected_extremes_use_absolute_event_change_and_both_designs(tmp_path):
    records = []
    for cell in ("c", "h"):
        for design in ("p", "q"):
            name = f"{cell}-{design}.npz"
            risk = np.array([[0., 0., 0., 0.], [10., 0., 0., 0.]])
            event = np.array([[5., 0., 0., 4.], [-1000., 0., 0., -900.]])
            digest = store_array(tmp_path/name, {"risk": risk, "event": event})
            records.append({"cell": cell, "phase": design, "block": 0,
                            "array_path": name, "array_sha256": digest})
    rows = select_extremes(records, tmp_path)
    assert len(rows) == 8
    assert all(r["index"] == 1 for r in rows if r["selection_role"] == "largest_original_M2_orbit")
    assert all(r["index"] == 0 for r in rows if r["selection_role"] == "largest_absolute_event_change")
    records[0]["array_sha256"] = "incorrect"
    with pytest.raises(ValueError):
        select_extremes(records, tmp_path)
