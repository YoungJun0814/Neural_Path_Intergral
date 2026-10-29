"""A namespaced end-to-end correctness check, never a study replicate."""

from dataclasses import replace

import torch

from src.path_integral.finite_grid_geometry_controls import ARMS, PER_WHOLE, whole
from src.path_integral.finite_grid_geometry_protocol import EvaluationBudget, PhaseLimit
from tests.test_finite_grid_geometry_protocol import protocol


def test_complete_whole_preserves_roles_moments_families_and_exact_work():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        p = replace(protocol(), protocol_id="fg-geometry-integration-validation-only-v1")
        budget = EvaluationBudget(PhaseLimit("FR2", PER_WHOLE, 300.))
        result = whole(p, "D02", 0, budget)
        assert budget.evaluations == PER_WHOLE
        assert [v["arm"] for v in result["finals"]] == list(ARMS)
        assert [v["components"] for v in result["finals"]] == [3, 373, 3, 2, 3, 2]
        assert len(set(result["seeds"].values())) == len(result["seeds"]) == 28
        assert result["risk_H"]["counts"] == [32768, 32768]
        assert result["risk_H"]["known_stratum_masses"] == [.5, .5]
        assert result["risk_P"]["counts"] == [65536]
        assert result["event_reference_column"] == 6
        assert result["not_confirmation"]
        assert all(v["mean"] > 0 and v["se"] >= 0 for v in result["finals"])
        assert all(v["not_population_dominance_proof"] for v in result["contrasts"])
    finally:
        torch.set_num_threads(old)
