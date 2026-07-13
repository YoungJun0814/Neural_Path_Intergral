from __future__ import annotations

import math

import pytest
import torch

from src.physics_engine import MarketSimulator
from src.training.markov_control import (
    MarkovianHestonControl,
    load_markovian_control_checkpoint,
    markov_control_objective,
    markov_control_state_sha256,
    save_markovian_control_checkpoint,
    train_markovian_control,
)


def _simulator() -> MarketSimulator:
    return MarketSimulator(
        mu=0.0,
        kappa=1.5,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        device="cpu",
    )


def _control(initial: float = -2.0) -> MarkovianHestonControl:
    return MarkovianHestonControl(
        initial_spot=100.0,
        barrier=75.0,
        maturity=0.5,
        variance_scale=0.04,
        hidden_dim=8,
        n_layers=1,
        control_bound=8.0,
        initial_constant=initial,
    )


def test_control_warm_start_is_exactly_constant() -> None:
    control = _control(-2.25)
    spot = torch.tensor([70.0, 100.0, 130.0])
    variance = torch.tensor([0.01, 0.04, 0.12])
    output = control(0.2, spot, variance, None)
    assert torch.allclose(output, torch.full_like(output, -2.25), atol=1e-6)


@pytest.mark.parametrize(
    "objective", ["scaled_second_moment", "log_second_moment", "entropy_stress"]
)
def test_each_objective_is_finite_and_differentiable(objective: str) -> None:
    torch.manual_seed(441)
    control = _control()
    diagnostics = markov_control_objective(
        _simulator(),
        control,
        spot=100.0,
        variance=0.04,
        maturity=0.5,
        dt=1.0 / 32.0,
        barrier=75.0,
        num_paths=2_000,
        objective=objective,  # type: ignore[arg-type]
        reference_probability=0.02,
    )
    diagnostics.loss.backward()
    gradient_norm = math.sqrt(
        sum(float(torch.sum(parameter.grad**2)) for parameter in control.parameters())
    )
    assert torch.isfinite(diagnostics.loss)
    assert gradient_norm > 0.0


def test_trainer_uses_disjoint_validation_and_restores_best_checkpoint() -> None:
    control = _control()
    result = train_markovian_control(
        _simulator(),
        control,
        spot=100.0,
        variance=0.04,
        maturity=0.25,
        dt=1.0 / 16.0,
        barrier=80.0,
        reference_probability=0.03,
        objective="log_second_moment",
        train_seeds=(11, 12),
        validation_seeds=(21, 22),
        epochs=2,
        paths_per_batch=1_000,
        validation_paths=1_000,
        validate_every=1,
    )
    assert len(result.history) == 2
    assert math.isfinite(result.best_validation_log_second_moment)

    with pytest.raises(ValueError, match="disjoint"):
        train_markovian_control(
            _simulator(),
            _control(),
            spot=100.0,
            variance=0.04,
            maturity=0.25,
            dt=1.0 / 16.0,
            barrier=80.0,
            reference_probability=0.03,
            objective="log_second_moment",
            train_seeds=(11,),
            validation_seeds=(11,),
            epochs=1,
        )


def test_gaussian_hard_event_score_gradient_matches_closed_form() -> None:
    """Validate the score-gradient sign/magnitude independently of Heston."""
    torch.manual_seed(701)
    paths = 1_000_000
    control = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    noise_q = torch.randn(paths, dtype=torch.float64)
    base_coordinate = noise_q + control.detach()
    event = base_coordinate <= -3.0
    log_weight = -control.detach() * noise_q - 0.5 * control.detach() ** 2
    squared_contribution = event.double() * torch.exp(2.0 * log_weight)
    score_log_q = control * noise_q
    surrogate = -torch.mean(squared_contribution.detach() * score_log_q)
    surrogate.backward()

    # J(u)=exp(u²) Φ(a+u), so J'(u)=exp(u²)[2uΦ(a+u)+φ(a+u)].
    u = float(control.detach())
    a = -3.0
    x = a + u
    cdf = 0.5 * math.erfc(-x / math.sqrt(2.0))
    density = math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)
    expected_gradient = math.exp(u**2) * (2.0 * u * cdf + density)
    assert float(control.grad) == pytest.approx(expected_gradient, rel=0.04, abs=1e-10)


def test_versioned_checkpoint_round_trip(tmp_path) -> None:
    control = _control(-2.25)
    path = tmp_path / "control.pt"
    state_hash = save_markovian_control_checkpoint(path, control, metadata={"protocol": "test-v1"})
    restored, metadata = load_markovian_control_checkpoint(path)
    assert metadata == {"protocol": "test-v1"}
    assert state_hash == markov_control_state_sha256(restored)

    spot = torch.tensor([80.0, 100.0])
    variance = torch.tensor([0.02, 0.04])
    assert torch.equal(control(0.1, spot, variance, None), restored(0.1, spot, variance, None))
