import pytest

from experiments.post_audit_defensive_allocation import (
    ARMS,
    CELLS,
    CHUNK,
    FINAL,
    TRAIN,
    VALIDATION,
    identity,
    jobs,
    seed,
    summarize,
)


def test_disjoint_fixed_bank_and_final_seed_roles_and_complete_work():
    work = jobs()
    assert len(work) == 64
    assert len({identity(j) for j in work}) == 64
    seeds = []
    for c in CELLS:
        for rep in range(2):
            for stream, n in (("train", TRAIN), ("validation", VALIDATION)):
                seeds.extend(seed(c, rep, stream, 0, k, role) for k in range(n//CHUNK) for role in ("path", "label"))
            for arm in ARMS:
                for b in range(8):
                    seeds.extend(seed(c, rep, "final-"+arm, b, k, role) for k in range(FINAL//CHUNK) for role in ("path", "label"))
    seeds.extend(seed(c, 1000, "preflight", 0, 0, role) for c in CELLS for role in ("path", "label"))
    assert len(seeds) == len(set(seeds))
    assert 4*(TRAIN+VALIDATION+2*8*FINAL)+2*CHUNK == 287232
    with pytest.raises(ValueError, match="complete"):
        summarize([], [])
