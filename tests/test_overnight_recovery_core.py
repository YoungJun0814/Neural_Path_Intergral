"""Independent oracle, checkpoint and same-law cache contracts."""

import json
import math
from dataclasses import replace

import pytest
import torch
from scipy.integrate import dblquad

from experiments.post_audit_r1_diagnostics import _problem
from src.path_integral.cached_volterra_terminal import CachedShiftDensity, CachedVolterraTerminal
from src.path_integral.finite_rank_gaussian_transport import (
    DefensiveFiniteRankGaussianMixture,
    FiniteRankGaussianComponent,
)
from src.path_integral.gaussian_bump_oracle import GaussianBumpOracle
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.volterra_conditional_payoffs import evaluate_rbergomi_conditional_terminal
from src.path_integral.weighted_tempered_smc import (
    WeightedSMCConfig,
    estimate_weighted_tempered_normalizer,
)


def test_bump_endpoint_independent_quadrature():
    oracle = GaussianBumpOracle((.2, .7), ((.3, -.1), (2., .5)), (.8, .6))
    for power in (1, 2):
        def integrand(y, x, fixed_power=power):
            g = sum(a*math.exp(-((x-m[0])**2+(y-m[1])**2)/(2*s*s))
                    for a, m, s in zip(oracle.amplitudes, oracle.centers, oracle.scales, strict=True))
            return g**fixed_power*math.exp(-.5*(x*x+y*y))/(2*math.pi)
        actual, error = dblquad(integrand, -10, 10, lambda x: -10, lambda x: 10, epsabs=1e-10)
        assert error < 1e-7
        assert math.isclose(math.exp(oracle.log_endpoint(power)), actual, rel_tol=1e-8)
    with pytest.raises(ValueError):
        oracle.log_endpoint(0)


def test_journal_resume_failure_and_cumulative_budget(tmp_path):
    manifest = {"source": "fixed", "limits": {"N1": {"work": 10, "wall": 10.}}}
    journal = RecoveryJournal(tmp_path, manifest)
    calls = []
    def operation():
        calls.append(1)
        return {"answer": 42}
    assert journal.unit("one", phase="N1", work=6, maximum_wall=1., operation=operation) == {"answer": 42}
    other = RecoveryJournal(tmp_path, manifest)
    assert other.unit("one", phase="N1", work=6, maximum_wall=1., operation=operation) == {"answer": 42}
    assert len(calls) == 1
    with pytest.raises(TimeoutError):
        other.unit("two", phase="N1", work=6, maximum_wall=1., operation=operation)
    with pytest.raises(ValueError):
        RecoveryJournal(tmp_path, {**manifest, "source": "changed"})
    path = tmp_path / "one.finish.json"
    payload = json.loads(path.read_text())
    payload["payload"]["result"]["answer"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        other.unit("one", phase="N1", work=6, maximum_wall=1., operation=operation)


def test_failed_unit_is_not_replaced(tmp_path):
    journal = RecoveryJournal(tmp_path, {"limits": {"N1": {"work": 20, "wall": 20.}}})
    def fail():
        raise FloatingPointError("scientific failure")
    with pytest.raises(FloatingPointError):
        journal.unit("failed", phase="N1", work=5, maximum_wall=2., operation=fail)
    with pytest.raises(RuntimeError):
        journal.unit("failed", phase="N1", work=5, maximum_wall=2., operation=lambda: {})
    assert journal.accounting("N1")[0] == 5


@pytest.mark.parametrize("steps", [1, 4, 32])
@pytest.mark.parametrize("eta,rho", [(1.5, -.7), (3., 0.), (1.5, .7)])
def test_cached_terminal_original_law(steps, eta, rho):
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1, "eta": eta,
                        "xi": .04, "rho": rho}, {"id": "toy", "threshold": 80.}, steps)
    z = torch.randn((128, 2*steps), dtype=torch.float64, generator=torch.Generator().manual_seed(59))
    evaluator = CachedVolterraTerminal.create(problem)
    original = evaluate_rbergomi_conditional_terminal(problem, z)
    torch.testing.assert_close(evaluator.log_probability(z), original.payoffs.log_left_probability, rtol=2e-13, atol=2e-12)
    changed = CachedVolterraTerminal.create(replace(problem, hurst=.2))
    if steps > 1:
        assert not torch.equal(evaluator.log_probability(z), changed.log_probability(z))


def test_observer_does_not_change_smc_rng_or_weights():
    cfg = WeightedSMCConfig(64, (0., .3, .7, 1.), 1, .35, 1, 81, retain_final_particles=True)
    def potential(z):
        return -.5*z.square().sum(1)
    a = estimate_weighted_tempered_normalizer(potential, dimension=2, config=cfg)
    observations = []
    def observer(event, replicate, beta, particles, weights):
        observations.append((event, beta, float(weights.sum())))
        particles.fill_(999)
        weights.zero_()
    b = estimate_weighted_tempered_normalizer(potential, dimension=2, config=cfg, observer=observer)
    torch.testing.assert_close(a.log_replicate_estimates, b.log_replicate_estimates, rtol=0, atol=0)
    torch.testing.assert_close(a.final_particles, b.final_particles, rtol=0, atol=0)
    assert len(observations) == 10


def test_cached_density_and_last_pair_match_original():
    from src.path_integral.structural_v2_conditional_reference import last_pair_cache
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .1, "eta": 3.,
                        "xi": .04, "rho": -.7}, {"id": "toy", "threshold": 80.}, 4)
    natural = FiniteRankGaussianComponent.natural(8)
    shifted = FiniteRankGaussianComponent(torch.arange(8, dtype=torch.float64)/10,
                                          torch.empty((8, 0), dtype=torch.float64),
                                          torch.empty(0, dtype=torch.float64))
    q = DefensiveFiniteRankGaussianMixture((natural, shifted), torch.tensor((.1, .9), dtype=torch.float64))
    z = torch.randn((128, 8), dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(CachedShiftDensity.create(q).log_q_over_p(z), q.log_q_over_p(z), rtol=1e-13, atol=1e-13)
    a = CachedVolterraTerminal.create(problem).last_pair(z[:, :-2], q)
    b = last_pair_cache(problem, z[:, :-2], q)
    pair = z[:, -2:, None].transpose(1, 2)
    torch.testing.assert_close(a.log_inner_risk(pair), b.log_inner_risk(pair), rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(a.log_mu_mean(), b.log_mu_mean(), rtol=2e-13, atol=2e-12)


def test_unfinished_reservation_counts_and_cannot_be_replaced(tmp_path):
    from src.path_integral.reference_protocol import canonical_sha256
    from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
    journal = RecoveryJournal(tmp_path, {"limits": {"N1": {"work": 20, "wall": 20.}}})
    reservation = {"manifest": journal.digest, "phase": "N1", "reserved_work": 17, "reserved_wall": 8.}
    write_json_atomic_nonoverwriting(tmp_path/"interrupted.start.json", {"payload": reservation, "sha256": canonical_sha256(reservation)})
    assert journal.accounting("N1") == (17, 8.)
    with pytest.raises(RuntimeError):
        journal.unit("interrupted", phase="N1", work=17, maximum_wall=8., operation=lambda: {})
    with pytest.raises(TimeoutError):
        journal.unit("later", phase="N1", work=4, maximum_wall=1., operation=lambda: {})


def test_cached_history_matches_direct_convolution():
    from src.path_integral.rbergomi_fft import historical_volterra_convolution
    problem = _problem({"spot": 100., "maturity": 1., "hurst": .2, "eta": 1.5,
                        "xi": .04, "rho": -.7}, {"id": "toy", "threshold": 80.}, 32)
    cached = CachedVolterraTerminal.create(problem)
    dw = torch.randn((128, 32), dtype=torch.float64, generator=torch.Generator().manual_seed(991))
    direct = historical_volterra_convolution(dw, cached.kernel.historical_kernel, method="direct")
    fft = torch.fft.irfft(torch.fft.rfft(dw, n=cached.fft_length, dim=1)*cached.kernel_transform,
                          n=cached.fft_length, dim=1)[:, :32]
    torch.testing.assert_close(direct, fft, rtol=2e-12, atol=2e-12)


def test_phase_wall_includes_between_unit_overhead(tmp_path, monkeypatch):
    import src.path_integral.recovery_checkpoint as module
    now = [100.]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    journal = RecoveryJournal(tmp_path, {"limits": {"N1": {"work": 100, "wall": 10.}}})
    journal.unit("first", phase="N1", work=1, maximum_wall=1., operation=lambda: {})
    now[0] += 11.
    with pytest.raises(TimeoutError, match="whole phase"):
        journal.unit("next", phase="N1", work=1, maximum_wall=1., operation=lambda: {})
    assert not (tmp_path/"next.start.json").exists()


def test_budget_cache_invalidates_on_file_changes(tmp_path):
    journal = RecoveryJournal(tmp_path, {"limits": {"N1": {"work": 100, "wall": 10.}}})
    journal.unit("first", phase="N1", work=4, maximum_wall=1., operation=lambda: {})
    assert journal.accounting("N1")[0] == 4
    path = tmp_path/"first.finish.json"
    value = json.loads(path.read_text())
    value["payload"]["work"] = 7
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        journal.accounting("N1")


def test_checkpoint_io_overrun_blocks_reusing_complete_prefix(tmp_path, monkeypatch):
    import src.path_integral.recovery_checkpoint as module
    now = [100.]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    journal = RecoveryJournal(tmp_path, {"limits": {"N1": {"work": 100, "wall": 10.}}})
    original_write = journal._write
    def slow_write(path, value):
        original_write(path, value)
        if path.name.endswith(".finish.json"):
            now[0] += 11.
    monkeypatch.setattr(journal, "_write", slow_write)
    with pytest.raises(TimeoutError, match="checkpoint I/O"):
        journal.unit("first", phase="N1", work=4, maximum_wall=1., operation=lambda: {})
    with pytest.raises(RuntimeError, match="phase failed"):
        journal.unit("first", phase="N1", work=4, maximum_wall=1., operation=lambda: {})
