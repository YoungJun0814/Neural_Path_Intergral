import json

import numpy as np
import pytest

from experiments.post_audit_analytic_residual_recovery import json_compatible
from src.path_integral.analytic_residual_batch import batched_moments
from src.path_integral.reference_protocol import canonical_json_bytes
from tests.test_analytic_residual_batch import profile


def test_complete_payload_with_numpy_critical_flags_roundtrips_canonical_json():
    payload = batched_moments([profile()])
    with pytest.raises(TypeError, match="bool_"):
        canonical_json_bytes(payload)
    fixed = json_compatible(payload)
    assert json.loads(canonical_json_bytes(fixed)) == fixed
    assert fixed["rows"][0]["moments"][0]["log_estimate"] == payload["rows"][0]["moments"][0]["log_estimate"]


def test_serialization_adapter_never_accepts_nonfinite_json_or_changes_numbers():
    value = {"b": np.bool_(False), "x": np.float64(.125), "i": np.int64(3)}
    fixed = json_compatible(value)
    assert fixed == {"b": False, "x": .125, "i": 3}
    with pytest.raises(ValueError):
        canonical_json_bytes(json_compatible({"invalid": np.float64(np.nan)}))
