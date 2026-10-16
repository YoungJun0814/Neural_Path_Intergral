import torch

from experiments.post_audit_canonical_static_stratification import BLOCK_SIZE, identity, jobs, seed
from src.path_integral.baselines.rbergomi_common import RBergomiBaselineProblem
from src.path_integral.volterra_excursion_guide import build_volterra_excursion_guide
from tests.test_p1_recovery_p2_parallel import problem


def test_static_fresh_streams_and_actual_geometry_count(monkeypatch):
    grid = jobs()
    streams = [seed(j, s) for j in grid for s in (("path", "label") if j["method"] == "iid" else ("path",))]
    assert len(grid) == len({identity(j) for j in grid}) == 16
    assert len(streams) == len(set(streams))
    assert 16*BLOCK_SIZE+64 == 476224
    original = RBergomiBaselineProblem.simulate_local
    calls = []
    def traced(self, latent):
        calls.append(len(latent))
        return original(self, latent)
    monkeypatch.setattr(RBergomiBaselineProblem, "simulate_local", traced)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        q = build_volterra_excursion_guide(problem(32), amplitudes=[2., 4., 6., 8.], price_amplitudes=[0., 2., 4.])
    finally:
        torch.set_num_threads(old)
    assert calls == [64]
    assert len(q.components) == 373
