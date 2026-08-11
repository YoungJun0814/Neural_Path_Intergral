from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.v16_transport_policy import (
    route_v16_hybrid_v1,
    route_v16_hybrid_v2,
)


def _problem(
    *,
    hurst: float,
    rho: float,
    threshold: float = 40.0,
    eta: float = 1.5,
) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="route",
        task=TerminalThresholdTask(level=threshold),
        spot=100.0,
        maturity=1.0,
        steps=32,
        hurst=hurst,
        eta=eta,
        xi=0.04,
        rho=rho,
    )


def test_v16_policy_uses_only_structural_regime_boundaries() -> None:
    rough = route_v16_hybrid_v1(_problem(hurst=0.075, rho=-0.9))
    strong = route_v16_hybrid_v1(_problem(hurst=0.12, rho=-0.85))
    regular = route_v16_hybrid_v1(_problem(hurst=0.12, rho=-0.7))
    assert rough.regime == "rough"
    assert strong.regime == "strong_negative_correlation"
    assert regular.regime == "regular"
    assert rough.final_components == strong.final_components == 3
    assert regular.adapt_covariance is True
    assert all(route.defensive_mass > 0.0 for route in (rough, strong, regular))
    assert all(route.safety_mass > 0.0 for route in (rough, strong, regular))


def test_v16_v2_routes_confirmed_deep_tail_regimes_without_oracle_inputs() -> None:
    rough = route_v16_hybrid_v2(
        _problem(hurst=0.05, rho=-0.7, threshold=1.0)
    )
    high_eta = route_v16_hybrid_v2(
        _problem(hurst=0.12, rho=-0.7, threshold=1.0, eta=2.0)
    )
    strong = route_v16_hybrid_v2(
        _problem(hurst=0.12, rho=-0.9, threshold=1.0)
    )
    assert rough.regime == "deep_rough"
    assert rough.initializer == "tempered_smc"
    assert rough.final_components == 5
    assert rough.candidate_overrides()["tempered_target"]["particles"] == 4096
    assert high_eta.regime == "deep_high_vol_of_vol"
    assert high_eta.adaptation_iterations == 10
    assert strong.regime == "deep_strong_negative_correlation"
    assert strong.final_components == 3


def test_v16_v2_preserves_v1_route_above_the_frozen_deep_tail_boundary() -> None:
    problem = _problem(hurst=0.05, rho=-0.9, threshold=2.0001, eta=2.0)
    v1 = route_v16_hybrid_v1(problem)
    v2 = route_v16_hybrid_v2(problem)
    assert v2.policy_id == "v16_hybrid_routing_v2"
    assert {**v2.__dict__, "policy_id": v1.policy_id} == v1.__dict__
