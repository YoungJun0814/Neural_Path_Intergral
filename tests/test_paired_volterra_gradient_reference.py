from dataclasses import replace

import pytest
import torch

from src.path_integral.gradient_umbrella_smc import evaluate_family_gradient
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from tests.test_p1_recovery_p2_parallel import problem


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_joint_law_uses_exact_fine_and_coarse_marginals_and_adapted_price():
    paired = PairedVolterraProbability.create(replace(problem(32), eta=1e-12))
    z = torch.randn((512, 80), dtype=torch.float64, generator=torch.Generator().manual_seed(6401))
    values = paired.log_components(z)
    assert paired.dimension == 80
    torch.testing.assert_close(values[:, 0], values[:, 1], rtol=0, atol=1e-9)
    torch.testing.assert_close(values[:, 0], paired.fine.log_probability(z[:, :64]))
    coarse = paired.adapter.coarse(z[:, :64], z[:, 64:])
    torch.testing.assert_close(values[:, 1], paired.coarse.log_probability(coarse))
    with pytest.raises(ValueError):
        paired.log_components(z[:, :-1])


def test_fine_coarse_joint_gradient_through_augmentation_and_fft():
    paired = PairedVolterraProbability.create(problem(32))
    generator = torch.Generator().manual_seed(6402)
    z = .5*torch.randn((3, paired.dimension), dtype=torch.float64, generator=generator)
    w = torch.full((2,), .5, dtype=torch.float64).log()
    _, _, gradient = evaluate_family_gradient(paired.log_components, z, w, maximum_drift_norm=1e6)
    singleton = torch.cat([evaluate_family_gradient(paired.log_components, row[None], w, maximum_drift_norm=1e6)[2] for row in z])
    torch.testing.assert_close(gradient, singleton, rtol=1e-10, atol=1e-10)
    direction = torch.randn(z.shape, dtype=torch.float64, generator=generator)
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    eps = 1e-5
    hp = evaluate_family_gradient(paired.log_components, z+eps*direction, w, maximum_drift_norm=1e6)[1]
    hm = evaluate_family_gradient(paired.log_components, z-eps*direction, w, maximum_drift_norm=1e6)[1]
    torch.testing.assert_close((hp-hm)/(2*eps), (gradient*direction).sum(1), rtol=2e-6, atol=2e-6)
