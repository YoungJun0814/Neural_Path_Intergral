import pytest
import torch

from experiments.post_audit_p2_hamiltonian_tail import (
    ARMS,
    WORK,
    ident,
    make_jobs,
    pilot_decision,
    precision_summary,
    work_records,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(noisy=False):
    cells = ["canonical-h005-k1", "high-eta200-k1"]
    records = []
    mapping = {f"bootstrap-{phase}-{cell}-{arm}": 100+i
        for i, (phase, cell, arm) in enumerate((p, c, a) for p in ("pilot", "precision") for c in cells for a in ARMS)}
    for job in make_jobs(cells, "pilot"):
        log_mu = -18.+(.8*job["rep"] if noisy else .001*job["rep"])
        records.append({**job, "log_mu": log_mu, "wall_seconds": 1.,
            "stages": [{"initial_ancestors": 5, "mean_acceptance": .5,
                        "mean_squared_displacement_per_coordinate": .7}]*63})
    return cells, records, mapping


def test_fixed_work_and_full_pilot_membership():
    assert WORK == 31872
    assert 32*WORK+4*912 == 1023552
    cells, records, mapping = fixture()
    assert len(make_jobs(cells, "pilot")) == 32
    assert len(make_jobs(cells, "precision", counts={c: 128 for c in cells})) == 256
    for bad in (records[:-1], records+[records[0]]):
        with pytest.raises(ValueError):
            pilot_decision(bad, mapping, cells)


def test_fresh_allocation_does_not_promote_pilot_or_open_p2():
    cells, records, mapping = fixture()
    decision = pilot_decision(records, mapping, cells)
    assert decision["fresh_precision_authorized"]
    assert all(a["required_count"] == 32 for a in decision["fresh_allocations"].values())
    assert not decision["P2_qualified"]
    assert decision["pilot_not_promoted_to_confirmation"]
    with pytest.raises(ValueError):
        precision_summary(records, mapping, cells, {}, decision)


def test_fixed_budget_failure_blocks_more_samples_and_ignores_better_control():
    cells, records, mapping = fixture(noisy=True)
    for row in records:
        if row["arm"] == "gradient-control":
            row["log_mu"] = -18.
    decision = pilot_decision(records, mapping, cells)
    assert not decision["fresh_precision_authorized"]
    assert decision["decision"] == "STOP_PRIMARY_PRECISION_FORECAST_BUDGET"
    summary = precision_summary([], mapping, cells, {}, decision)
    assert not summary["executed"] and not summary["qualified_independent_reference"]
    with pytest.raises(ValueError):
        precision_summary(records, mapping, cells, {}, decision)


def test_unrun_precision_clock_is_zero_and_namespace_separates_phases():
    assert work_records({})["precision"]["wall_seconds"] == 0.
    j = {"phase": "pilot", "cell": "a", "arm": "gaussian-hmc", "rep": 0}
    assert ident(j) != ident({**j, "phase": "precision"})
