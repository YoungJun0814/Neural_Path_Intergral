"""Training-objective tests for lean multimodal path-integral mixtures."""

from __future__ import annotations

import torch

from src.path_integral import LeanRBergomiControl, simulate_rbergomi_mixture
from src.physics_engine import RBergomiSimulator
from src.training.path_mixture import (
    lean_pice_objective,
    lean_soft_pi_objective,
    mixture_weight_j2_objective,
    mixture_weights_from_logits,
    terminal_mode_potential,
    train_lean_pi_pice,
)


class _ConstantControl:
    def __init__(self, first: float, second: float) -> None:
        self.value = (first, second)

    def __call__(
        self,
        _time: float,
        spot: torch.Tensor,
        _variance: torch.Tensor,
        _volterra: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(self.value, device=spot.device, dtype=spot.dtype).expand(
            spot.shape[0], -1
        )


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(H=0.1, eta=1.1, xi=0.04, rho=-0.7, device="cpu")


def _control(mode: str) -> LeanRBergomiControl:
    return LeanRBergomiControl(
        spot=100.0,
        xi=0.04,
        maturity=0.25,
        lower_threshold=85.0,
        upper_threshold=115.0,
        mode=mode,  # type: ignore[arg-type]
        hidden_dim=12,
        control_bound=(5.0, 5.0),
    ).double()


def _gradient_norm(control: LeanRBergomiControl) -> float:
    return float(
        torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in control.parameters()
                if parameter.grad is not None
            )
        )
    )


def test_union_potential_is_small_in_either_tail_and_large_in_center() -> None:
    spots = torch.tensor([70.0, 100.0, 140.0], dtype=torch.float64)
    potential = terminal_mode_potential(
        spots,
        lower_threshold=80.0,
        upper_threshold=125.0,
        scale=3.0,
        mode="union",
    )
    assert float(potential[0]) < float(potential[1])
    assert float(potential[2]) < float(potential[1])


def test_lean_pi_and_pice_objectives_produce_gradients() -> None:
    simulator = _simulator()
    pi_control = _control("left")
    pi = lean_soft_pi_objective(
        simulator,
        pi_control,
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=256,
        lower_threshold=85.0,
        upper_threshold=115.0,
        soft_scale=4.0,
        mode="left",
    )
    pi.loss.backward()
    assert _gradient_norm(pi_control) > 0.0

    candidate = _control("right")
    with torch.no_grad():
        candidate.network[-1].bias.copy_(torch.tensor([0.2, -0.1], dtype=torch.float64))
    pice = lean_pice_objective(
        simulator,
        candidate,
        behavior=candidate.frozen_copy(),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=256,
        lower_threshold=85.0,
        upper_threshold=115.0,
        soft_scale=4.0,
        mode="right",
    )
    pice.loss.backward()
    assert _gradient_norm(candidate) > 0.0


def test_weight_floor_maps_logits_to_strict_simplex() -> None:
    logits = torch.tensor([10.0, -4.0, 0.0], dtype=torch.float64)
    weights = mixture_weights_from_logits(logits, minimum_weight=0.05)
    assert torch.all(weights >= 0.05)
    assert torch.allclose(
        torch.sum(weights), torch.tensor(1.0, dtype=torch.float64), atol=1e-15, rtol=0.0
    )


def test_mixture_weight_j2_has_finite_nonzero_gradient() -> None:
    simulator = _simulator()
    torch.manual_seed(821)
    sample = simulate_rbergomi_mixture(
        simulator,
        [_ConstantControl(-1.0, 0.2), _ConstantControl(0.8, -0.1)],
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=2_000,
        label_generator=torch.Generator().manual_seed(822),
    )
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    objective = mixture_weight_j2_objective(
        sample,
        logits,
        lower_threshold=88.0,
        upper_threshold=112.0,
        minimum_weight=0.05,
    )
    objective.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(torch.linalg.vector_norm(logits.grad)) > 0.0


def test_sequential_trainer_updates_lean_controller() -> None:
    simulator = _simulator()
    control = _control("union")
    initial = {name: value.detach().clone() for name, value in control.state_dict().items()}
    records = train_lean_pi_pice(
        simulator,
        control,
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=128,
        lower_threshold=85.0,
        upper_threshold=115.0,
        soft_scale=4.0,
        mode="union",
        pi_updates=2,
        pice_updates=2,
        pi_learning_rate=1e-3,
        pice_learning_rate=1e-3,
        gradient_clip=5.0,
        seed=99,
        behavior_refresh=1,
    )
    assert [record.objective for record in records] == ["pi", "pi", "pice", "pice"]
    assert any(
        not torch.equal(initial[name], current)
        for name, current in control.state_dict().items()
    )
