"""G1 correctness tests for two-driver PI/PICE/J2 feedback training."""

from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.heston_reference import HestonReferenceParams
from src.path_integral import brownian_log_likelihood
from src.physics_engine import MarketSimulator
from src.training.heston_feedback import (
    HestonOracleDataset,
    TwoDriverHestonControl,
    build_heston_oracle_dataset,
    candidate_log_density_on_target_paths,
    feedback_pice_objective,
    fit_heston_oracle_distillation,
    hard_j2_objective,
    load_two_driver_control_checkpoint,
    oracle_alignment,
    save_two_driver_control_checkpoint,
    soft_pi_objective,
)


def _simulator() -> MarketSimulator:
    return MarketSimulator(
        mu=0.01,
        kappa=1.8,
        theta=0.04,
        xi=0.35,
        rho=-0.65,
        device="cpu",
    )


def _control(initial: tuple[float, float] = (-0.6, 0.2)) -> TwoDriverHestonControl:
    return TwoDriverHestonControl(
        barrier=90.0,
        maturity=0.25,
        variance_scale=0.04,
        architecture="mlp",
        hidden_dim=12,
        n_layers=2,
        control_bound=(6.0, 5.0),
        initial_control=initial,
    )


def test_two_driver_control_initializes_exactly_constant_and_can_leave_constant() -> None:
    torch.manual_seed(10)
    control = _control()
    spot = torch.tensor([70.0, 90.0, 120.0])
    variance = torch.tensor([0.01, 0.04, 0.09])
    expected = torch.tensor([[-0.6, 0.2]]).expand(3, 2)
    assert torch.allclose(control(0.1, spot, variance, None), expected, atol=1e-7)

    loss = control(0.1, spot, variance, None)[:, 0].mul(spot).mean()
    loss.backward()
    assert control.output.weight.grad is not None
    assert torch.linalg.vector_norm(control.output.weight.grad) > 0.0


def test_candidate_target_density_matches_residual_likelihood_identity() -> None:
    torch.manual_seed(11)
    control = _control((-0.45, 0.3))
    paths = _simulator().simulate_controlled_two_driver(
        S0=100.0,
        v0=0.04,
        T=0.25,
        dt=1.0 / 16.0,
        num_paths=64,
        control_fn=None,
        record_brownian=True,
        dtype=torch.float32,
    )
    assert paths.target_brownian_increments is not None
    candidate_log_density, controls = candidate_log_density_on_target_paths(control, paths)
    candidate_residual = paths.target_brownian_increments - controls.detach() * paths.step_dt
    candidate_log_p_over_q = brownian_log_likelihood(
        controls.detach(), candidate_residual, paths.step_dt
    )
    assert torch.allclose(candidate_log_density.detach(), -candidate_log_p_over_q, atol=2e-6)


def test_soft_pi_and_feedback_pice_have_finite_nonzero_gradients() -> None:
    simulator = _simulator()

    torch.manual_seed(12)
    pi_control = _control()
    pi = soft_pi_objective(
        simulator,
        pi_control,
        spot=100.0,
        variance=0.04,
        maturity=0.25,
        dt=1.0 / 16.0,
        barrier=90.0,
        temperature=0.08,
        num_paths=1_000,
    )
    pi.loss.backward()
    pi_norm = math.sqrt(
        sum(float(torch.sum(parameter.grad.square())) for parameter in pi_control.parameters())
    )
    assert torch.isfinite(pi.loss)
    assert 0.0 < float(pi.soft_estimate) < 1.0
    assert pi_norm > 0.0

    torch.manual_seed(13)
    pice_control = _control()
    behavior = pice_control.frozen_copy()
    pice = feedback_pice_objective(
        simulator,
        pice_control,
        behavior_control=behavior,
        spot=100.0,
        variance=0.04,
        maturity=0.25,
        dt=1.0 / 16.0,
        barrier=90.0,
        temperature=0.08,
        num_paths=1_000,
    )
    pice.loss.backward()
    pice_norm = math.sqrt(
        sum(float(torch.sum(parameter.grad.square())) for parameter in pice_control.parameters())
    )
    assert torch.isfinite(pice.loss)
    assert 1.0 <= float(pice.effective_sample_size) <= 1_000.0
    assert pice_norm > 0.0


def test_hard_j2_uses_score_gradient_and_reports_ordinary_estimate() -> None:
    torch.manual_seed(14)
    control = _control((-0.8, 0.25))
    result = hard_j2_objective(
        _simulator(),
        control,
        spot=100.0,
        variance=0.04,
        maturity=0.25,
        dt=1.0 / 16.0,
        barrier=90.0,
        num_paths=4_000,
    )
    result.loss.backward()
    gradient_norm = math.sqrt(
        sum(float(torch.sum(parameter.grad.square())) for parameter in control.parameters())
    )
    assert torch.isfinite(result.loss)
    assert 0.0 < float(result.estimate) < 1.0
    assert 0.0 < float(result.proposal_event_fraction) < 1.0
    assert 1.0 <= float(result.contribution_ess) <= 4_000.0
    assert gradient_norm > 0.0


def test_two_driver_gaussian_j2_score_matches_closed_form() -> None:
    """Check the vector score sign and magnitude independently of Heston."""
    torch.manual_seed(1401)
    paths = 1_000_000
    control = torch.tensor([-1.1, 0.35], dtype=torch.float64, requires_grad=True)
    event_direction = torch.tensor([1.0, -0.6], dtype=torch.float64)
    threshold = -2.5
    proposal_noise = torch.randn(paths, 2, dtype=torch.float64)
    target_noise = proposal_noise + control.detach()
    event = target_noise.mv(event_direction) <= threshold
    log_weight = -proposal_noise.mv(control.detach()) - 0.5 * control.detach().square().sum()
    squared_contribution = event.double() * torch.exp(2.0 * log_weight)
    score_log_q = proposal_noise.mv(control)
    surrogate = -torch.mean(squared_contribution.detach() * score_log_q)
    surrogate.backward()

    u = control.detach()
    direction_norm = float(torch.linalg.vector_norm(event_direction))
    x = (threshold + float(torch.dot(event_direction, u))) / direction_norm
    cdf = 0.5 * math.erfc(-x / math.sqrt(2.0))
    density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    expected = torch.exp(u.square().sum()) * (
        2.0 * u * cdf + density * event_direction / direction_norm
    )
    assert control.grad is not None
    assert torch.allclose(control.grad, expected, rtol=0.05, atol=2e-8)


def test_oracle_distillation_metrics_improve_on_a_finite_grid() -> None:
    time = torch.tensor([0.0, 0.0, 0.12, 0.12])
    spot = torch.tensor([85.0, 105.0, 85.0, 105.0])
    variance = torch.tensor([0.02, 0.06, 0.06, 0.02])
    target = torch.tensor([[-1.5, 0.4], [-0.3, 0.1], [-1.0, 0.5], [-0.1, 0.2]])
    dataset = HestonOracleDataset(
        time=time,
        spot=spot,
        variance=variance,
        control=target,
        maximum_gradient_discrepancy=1e-8,
    )
    torch.manual_seed(15)
    control = TwoDriverHestonControl(
        barrier=90.0,
        maturity=0.25,
        variance_scale=0.04,
        architecture="mlp",
        hidden_dim=16,
        n_layers=2,
        initial_control=(0.0, 0.0),
    )
    before = oracle_alignment(control, dataset)
    history = fit_heston_oracle_distillation(
        control, dataset, epochs=250, learning_rate=1e-2
    )
    after = oracle_alignment(control, dataset)
    assert history[-1] < history[0]
    assert after.normalized_rmse < 0.15 * before.normalized_rmse
    assert after.mean_cosine > 0.98
    assert after.sign_agreement == 1.0


def test_real_oracle_dataset_builder_preserves_two_driver_signs() -> None:
    params = HestonReferenceParams(
        v0=0.04,
        kappa=1.8,
        theta=0.04,
        xi=0.45,
        rho=-0.65,
        r=0.03,
        q=0.0,
    )
    dataset = build_heston_oracle_dataset(
        times=(0.0,),
        spots=(100.0,),
        variances=(0.04,),
        maturity=0.5,
        barrier=85.0,
        temperature=0.05,
        params=params,
    )
    assert dataset.control.shape == (1, 2)
    assert float(dataset.control[0, 0]) < 0.0
    assert float(dataset.control[0, 1]) > 0.0
    assert dataset.maximum_gradient_discrepancy < 1e-5


def test_two_driver_checkpoint_round_trip(tmp_path) -> None:
    control = _control((-1.25, 0.55))
    path = tmp_path / "two_driver.pt"
    state_hash = save_two_driver_control_checkpoint(
        path, control, metadata={"protocol": "g1-test"}
    )
    restored, metadata = load_two_driver_control_checkpoint(path)
    assert metadata == {"protocol": "g1-test"}
    assert state_hash
    spot = torch.tensor([80.0, 100.0])
    variance = torch.tensor([0.02, 0.05])
    assert torch.equal(control(0.1, spot, variance, None), restored(0.1, spot, variance, None))


def test_j2_rejects_batches_without_events() -> None:
    torch.manual_seed(16)
    control = _control((0.0, 0.0))
    with pytest.raises(RuntimeError, match="no hard events"):
        hard_j2_objective(
            _simulator(),
            control,
            spot=100.0,
            variance=0.04,
            maturity=0.05,
            dt=0.05,
            barrier=1.0,
            num_paths=128,
        )
