import torch

from src.path_integral.cameron_martin_basis import build_blp_cameron_martin_basis


def test_blp_basis_is_orthonormal_and_preserves_energy() -> None:
    basis = build_blp_cameron_martin_basis(steps=16, modes_per_driver=7)
    coefficients = torch.randn(
        5, basis.rank, dtype=torch.float64, generator=torch.Generator().manual_seed(8)
    )
    control = basis.expand(coefficients)
    reconstructed = basis.project(control)
    assert torch.max(torch.abs(reconstructed - coefficients)) < 2e-12
    assert torch.max(
        torch.abs(torch.sum(control**2, dim=1) - torch.sum(coefficients**2, dim=1))
    ) < 2e-12


def test_full_basis_round_trips_arbitrary_blp_control() -> None:
    basis = build_blp_cameron_martin_basis(steps=8)
    control = torch.randn(
        4, basis.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(11)
    )
    assert torch.max(torch.abs(basis.expand(basis.project(control)) - control)) < 3e-12
    assert torch.max(torch.abs(basis.projection_residual(control))) < 3e-12

