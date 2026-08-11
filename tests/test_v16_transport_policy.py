from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.v16_transport_policy import route_v16_hybrid_v1


def _problem(*, hurst: float, rho: float) -> RBergomiBaselineProblem:
    return RBergomiBaselineProblem(
        task_id="route",
        task=TerminalThresholdTask(level=40.0),
        spot=100.0,
        maturity=1.0,
        steps=32,
        hurst=hurst,
        eta=1.5,
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
