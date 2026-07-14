"""Objective and replay tests for sequential VFO training."""

from __future__ import annotations

import math

import torch

from src.path_integral.controllers import VolterraFollmerOperator
from src.physics_engine import RBergomiSimulator
from src.training.vfo import (
    replay_vfo_on_target_paths,
    train_vfo_stage,
    vfo_hard_j2_objective,
    vfo_pice_objective,
    vfo_soft_pi_objective,
)


def _simulator() -> RBergomiSimulator:
    return RBergomiSimulator(
        H=0.1, eta=1.0, xi=0.04, rho=-0.7, device="cpu"
    )


def _control() -> VolterraFollmerOperator:
    torch.manual_seed(3201)
    return VolterraFollmerOperator(
        H=0.1,
        rho=-0.7,
        eta=1.0,
        xi=0.04,
        maturity=0.25,
        barrier=90.0,
        minimum_dt=1.0 / 32.0,
        soe_terms=5,
        hidden_dim=12,
        residual_dim=6,
        control_bound=(5.0, 5.0),
    ).double()


def _gradient_norm(control: VolterraFollmerOperator) -> float:
    return math.sqrt(
        sum(
            float(parameter.grad.square().sum())
            for parameter in control.parameters()
            if parameter.grad is not None
        )
    )


def test_stateful_target_replay_recovers_behavior_controls_and_density() -> None:
    simulator = _simulator()
    behavior = _control()
    behavior.set_stage("structural")
    with torch.no_grad():
        behavior.structural_gate_parameter.fill_(0.4)
        behavior.instantaneous[-1].bias.copy_(
            torch.tensor([-0.2, 0.1], dtype=torch.float64)
        )
    torch.manual_seed(3202)
    with torch.no_grad():
        paths = simulator.simulate_controlled_two_driver(
            S0=100.0,
            T=0.25,
            dt=1.0 / 16.0,
            num_paths=32,
            control_fn=behavior,
            record_augmented=True,
        )
    assert paths.controls is not None
    assert paths.target_brownian_increments is not None
    candidate = behavior.frozen_copy()
    replayed = replay_vfo_on_target_paths(candidate, paths)
    assert torch.allclose(replayed, paths.controls, atol=1e-14, rtol=1e-14)
    log_q_over_p = torch.sum(
        replayed * paths.target_brownian_increments, dim=(1, 2)
    ) - 0.5 * paths.step_dt * torch.sum(replayed.square(), dim=(1, 2))
    assert torch.allclose(log_q_over_p, -paths.log_likelihood, atol=2e-14, rtol=2e-14)


def test_vfo_pi_pice_and_j2_objectives_have_finite_score_paths() -> None:
    simulator = _simulator()

    pi_control = _control()
    torch.manual_seed(3203)
    pi = vfo_soft_pi_objective(
        simulator,
        pi_control,
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=512,
        barrier=90.0,
        soft_scale=5.0,
    )
    pi.loss.backward()
    assert torch.isfinite(pi.loss)
    assert _gradient_norm(pi_control) > 0.0

    pice_control = _control()
    pice_control.set_stage("structural")
    torch.manual_seed(3204)
    pice = vfo_pice_objective(
        simulator,
        pice_control,
        behavior=pice_control.frozen_copy(),
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=512,
        barrier=90.0,
        soft_scale=5.0,
    )
    pice.loss.backward()
    assert torch.isfinite(pice.loss)
    assert 0.0 < float(pice.effective_sample_fraction) <= 1.0
    assert _gradient_norm(pice_control) > 0.0

    j2_control = _control()
    j2_control.set_stage("joint")
    with torch.no_grad():
        j2_control.instantaneous[-1].bias.copy_(
            torch.tensor([-0.15, 0.08], dtype=torch.float64)
        )
    torch.manual_seed(3205)
    j2 = vfo_hard_j2_objective(
        simulator,
        j2_control,
        spot=100.0,
        maturity=0.25,
        dt=1.0 / 16.0,
        num_paths=2_000,
        barrier=90.0,
    )
    j2.loss.backward()
    assert torch.isfinite(j2.loss)
    assert 0.0 < float(j2.event_fraction) < 1.0
    assert _gradient_norm(j2_control) > 0.0


def test_sequential_stage_trainer_opens_only_declared_gate() -> None:
    simulator = _simulator()
    control = _control()
    common = {
        "spot": 100.0,
        "maturity": 0.25,
        "dt": 1.0 / 16.0,
        "num_paths": 256,
        "barrier": 90.0,
        "soft_scale": 5.0,
    }
    instant = train_vfo_stage(
        simulator,
        control,
        stage="instant",
        objective="pi",
        updates=2,
        learning_rate=1e-3,
        seed=3206,
        **common,
    )
    assert len(instant) == 2
    assert float(control.structural_gate) == 0.0
    assert float(control.residual_gate) == 0.0

    structural = train_vfo_stage(
        simulator,
        control,
        stage="structural",
        objective="pi",
        updates=2,
        learning_rate=1e-3,
        seed=3207,
        **common,
    )
    assert len(structural) == 2
    assert float(control.structural_gate.detach()) != 0.0
    assert float(control.residual_gate.detach()) == 0.0

    residual = train_vfo_stage(
        simulator,
        control,
        stage="residual",
        objective="pice",
        updates=2,
        learning_rate=1e-3,
        seed=3208,
        **common,
    )
    assert len(residual) == 2
    assert float(control.residual_gate.detach()) != 0.0
    assert not any(record.takeover_alarm for record in residual)
