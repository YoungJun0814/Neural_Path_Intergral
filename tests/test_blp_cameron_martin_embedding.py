import torch
from scipy.integrate import quad

from src.path_integral.blp_cameron_martin_embedding import (
    blp_cell_cameron_martin_shapes,
    blp_local_observable_means_from_standard_shift,
    piecewise_constant_drift_to_blp_standard_shift,
)


def test_blp_cell_shapes_are_orthonormal_under_independent_quadrature() -> None:
    hurst = 0.12
    step_dt = 0.2
    def product(reverse_time: float, left: int, right: int) -> float:
        shapes = blp_cell_cameron_martin_shapes(
            torch.tensor([reverse_time], dtype=torch.float64),
            hurst=hurst,
            step_dt=step_dt,
        )
        return float(shapes[0, left] * shapes[0, right])

    gram = torch.tensor(
        [
            [
                quad(
                    product,
                    0.0,
                    step_dt,
                    args=(left, right),
                    epsabs=2e-11,
                    epsrel=2e-11,
                )[0]
                for right in range(2)
            ]
            for left in range(2)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        gram,
        torch.eye(2, dtype=torch.float64),
        rtol=0.0,
        atol=2e-9,
    )


def test_piecewise_constant_projection_preserves_cm_energy_and_observables() -> None:
    hurst = 0.12
    step_dt = 0.125
    drift = torch.tensor([0.7, -0.2, 1.1, -0.4], dtype=torch.float64)
    shift = piecewise_constant_drift_to_blp_standard_shift(
        drift,
        hurst=hurst,
        step_dt=step_dt,
    )
    torch.testing.assert_close(
        torch.sum(shift.square()),
        step_dt * torch.sum(drift.square()),
        rtol=2e-13,
        atol=2e-14,
    )
    means = blp_local_observable_means_from_standard_shift(
        shift,
        hurst=hurst,
        step_dt=step_dt,
    )
    alpha = hurst - 0.5
    expected = torch.stack(
        (
            drift * step_dt,
            drift * step_dt ** (alpha + 1.0) / (alpha + 1.0),
        ),
        dim=1,
    )
    torch.testing.assert_close(means, expected, rtol=2e-13, atol=2e-14)
