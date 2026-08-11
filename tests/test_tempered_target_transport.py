import torch

from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.blp_cameron_martin_embedding import (
    build_mesh_compatible_blp_hybrid_basis,
)
from src.path_integral.path_functionals import TerminalThresholdTask
from src.path_integral.rbergomi_cm_transport import evaluate_rbergomi_cm_transport
from src.path_integral.tempered_conditional_smc import TemperedSMCConfig
from src.path_integral.tempered_target_transport import (
    TemperedTargetTransportConfig,
    _kmeans_labels,
    _replicate_kmeans_labels,
    fit_tempered_target_transport,
)


def test_tempered_target_fit_preserves_exact_defensive_mixture() -> None:
    problem = RBergomiBaselineProblem(
        task_id="tempered-fit-test",
        task=TerminalThresholdTask(level=50.0),
        spot=100.0,
        maturity=1.0,
        steps=4,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )
    basis = build_mesh_compatible_blp_hybrid_basis(
        steps=4,
        maturity=1.0,
        hurst=0.12,
        drift_modes=3,
        bridge_modes=1,
    )
    fitted = fit_tempered_target_transport(
        problem,
        basis,
        config=TemperedTargetTransportConfig(
            smc=TemperedSMCConfig(
                particles=256,
                temperatures=tuple((index / 8) ** 2 for index in range(9)),
                mutation_steps=2,
                pcn_scale=0.35,
                replicates=2,
                seed=445,
                retain_final_particles=True,
            ),
            components=2,
        ),
    )
    assert len(fitted.proposal.components) == 4
    assert fitted.proposal.defensive_mass == 0.15
    assert fitted.fitted_particle_count == 512
    assert fitted.training_cost.algorithmic_work_units > 0.0
    evaluated = evaluate_rbergomi_cm_transport(
        problem,
        fitted.proposal,
        sample_count=10_000,
        path_seed=446,
        label_seed=447,
    )
    assert evaluated.maximum_likelihood_bound_violation == 0.0
    assert torch.isfinite(evaluated.contribution).all()


def test_kmeans_clustering_separates_two_target_modes_deterministically() -> None:
    generator = torch.Generator().manual_seed(991)
    left = -3.0 + 0.1 * torch.randn((100, 2), generator=generator)
    right = 3.0 + 0.1 * torch.randn((100, 2), generator=generator)
    points = torch.cat((left, right)).to(torch.float64)
    labels = _kmeans_labels(points, components=2, iterations=20)
    repeated = _kmeans_labels(points, components=2, iterations=20)
    assert torch.equal(labels, repeated)
    assert int(torch.sum(labels[:100] == labels[0])) == 100
    assert int(torch.sum(labels[100:] == labels[100])) == 100
    assert labels[0] != labels[100]


def test_replicate_clustering_guarantees_components_per_smc_replicate() -> None:
    generator = torch.Generator().manual_seed(773)
    points = torch.randn((4 * 40, 3), generator=generator, dtype=torch.float64)
    labels = _replicate_kmeans_labels(
        points,
        particles_per_replicate=40,
        replicates=4,
        components=8,
        iterations=10,
    )
    for replicate in range(4):
        block = labels[40 * replicate : 40 * (replicate + 1)]
        assert set(int(value) for value in torch.unique(block)) == {
            2 * replicate,
            2 * replicate + 1,
        }


def test_multiscale_fit_adds_normalized_tail_covering_components() -> None:
    problem = RBergomiBaselineProblem(
        task_id="tempered-multiscale-test",
        task=TerminalThresholdTask(level=50.0),
        spot=100.0,
        maturity=1.0,
        steps=3,
        hurst=0.12,
        eta=1.5,
        xi=0.04,
        rho=-0.7,
    )
    basis = build_mesh_compatible_blp_hybrid_basis(
        steps=3,
        maturity=1.0,
        hurst=0.12,
        drift_modes=2,
        bridge_modes=1,
    )
    fitted = fit_tempered_target_transport(
        problem,
        basis,
        config=TemperedTargetTransportConfig(
            smc=TemperedSMCConfig(
                particles=64,
                temperatures=(0.0, 0.25, 1.0),
                mutation_steps=1,
                pcn_scale=0.3,
                replicates=2,
                seed=998,
                retain_final_particles=True,
            ),
            components=2,
            covariance_scales=(1.0, 2.0),
        ),
    )
    assert len(fitted.proposal.components) == 2 + 2 * 2
    assert torch.isclose(
        torch.sum(fitted.proposal.weights),
        torch.tensor(1.0, dtype=torch.float64),
    )
