from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.v16_transport_policy import (
    route_v16_hybrid_v1,
    route_v16_hybrid_v2,
    route_v16_hybrid_v3,
    route_v16_hybrid_v4,
    route_v16_hybrid_v5,
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


def test_v16_v3_routes_joint_extremes_to_confirmable_target_clusters() -> None:
    rough = route_v16_hybrid_v3(
        _problem(hurst=0.05, rho=-0.7, threshold=0.5)
    )
    rough_eta = route_v16_hybrid_v3(
        _problem(hurst=0.05, rho=-0.7, threshold=1.0, eta=2.0)
    )
    eta_rho = route_v16_hybrid_v3(
        _problem(hurst=0.12, rho=-0.9, threshold=1.0, eta=2.0)
    )
    assert rough.initializer == "tempered_smc"
    assert rough.tempered_clustering == "kmeans"
    assert rough.defensive_mass == 0.10
    assert rough_eta.regime == "deep_rough_high_vol_of_vol"
    assert rough_eta.tempered_particles == 8192
    assert eta_rho.regime == "deep_high_vol_of_vol_strong_negative_correlation"
    assert eta_rho.tempered_particles == 4096
    assert eta_rho.candidate_overrides()["tempered_target"]["clustering"] == "kmeans"


def test_v16_v4_bags_joint_extreme_training_replicates_only() -> None:
    ordinary_rough = route_v16_hybrid_v4(
        _problem(hurst=0.05, rho=-0.7, threshold=1.0)
    )
    rough_eta = route_v16_hybrid_v4(
        _problem(hurst=0.05, rho=-0.7, threshold=1.0, eta=2.0)
    )
    rough_rho = route_v16_hybrid_v4(
        _problem(hurst=0.05, rho=-0.9, threshold=1.0)
    )
    eta_rho = route_v16_hybrid_v4(
        _problem(hurst=0.12, rho=-0.9, threshold=1.0, eta=2.0)
    )
    assert ordinary_rough.tempered_clustering == "kmeans"
    for route in (rough_eta, rough_rho, eta_rho):
        assert route.policy_id == "v16_hybrid_routing_v4"
        assert route.tempered_clustering == "replicate_kmeans"
        assert route.final_components == route.tempered_replicates == 4
    assert rough_eta.tempered_particles == 8192
    assert rough_rho.tempered_particles == eta_rho.tempered_particles == 4096


def test_v16_v5_uses_confirmed_hybrid_and_fail_closed_joint_fallback() -> None:
    k2 = route_v16_hybrid_v5(
        _problem(hurst=0.05, rho=-0.7, threshold=2.0)
    )
    k1 = route_v16_hybrid_v5(
        _problem(hurst=0.05, rho=-0.7, threshold=1.0)
    )
    rough_eta = route_v16_hybrid_v5(
        _problem(hurst=0.05, rho=-0.7, threshold=1.0, eta=2.0)
    )
    eta_rho = route_v16_hybrid_v5(
        _problem(hurst=0.12, rho=-0.9, threshold=1.0, eta=2.0)
    )
    assert k2.initializer == "v14_tempered_hybrid"
    assert k2.hybrid_target_mass == 0.8
    assert k2.candidate_overrides()["target_mass"] == 0.8
    assert k1.initializer == "tempered_smc"
    for fallback in (rough_eta, eta_rho):
        assert fallback.initializer == "v14_only"
        assert fallback.comparator_dominance_claim is False
        assert fallback.candidate_overrides()["v14_initializer"][
            "defensive_mass"
        ] == 0.2
