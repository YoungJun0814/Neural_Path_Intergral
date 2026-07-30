from __future__ import annotations

import math

from experiments.g11_v8_p5_cell_tuned_cem_proposal import (
    ROOT,
    _scaled_schedules,
    _training_seed,
    _validation_seeds,
    load_cell_tuned_config,
)

CONFIG = ROOT / "configs/g11_v8/p5_cell_tuned_cem_proposal_v1.yaml"
CONFIG_V2 = ROOT / "configs/g11_v8/p5_cell_tuned_cem_proposal_v2.yaml"
CONFIG_V3 = ROOT / "configs/g11_v8/p5_all_failed_cells_cem_proposal_v3.yaml"


def test_cell_tuned_cem_contract_is_frozen_and_fail_closed() -> None:
    config, digest = load_cell_tuned_config(CONFIG)

    assert len(digest) == 64
    assert len(config["cells"]) == 2
    assert config["decision"]["new_full_pilot_authorized"] is False

    config_v2, digest_v2 = load_cell_tuned_config(CONFIG_V2)
    assert len(digest_v2) == 64
    assert config_v2["training_namespace"].endswith("-v2")
    assert config_v2["validation_namespace"].endswith("-v2")
    assert config_v2["prior_execution_failure"]["sha256"]

    config_v3, digest_v3 = load_cell_tuned_config(CONFIG_V3)
    assert len(digest_v3) == 64
    assert len(config_v3["cells"]) == 8
    assert sum(len(cell["target_methods"]) for cell in config_v3["cells"]) == 11
    assert config_v3["validation"]["replicates"] == 8
    assert (
        config_v3["validation"]["maximum_single_contribution_share"] == 0.10
    )


def test_cell_tuned_training_and_validation_seeds_are_disjoint() -> None:
    config, _ = load_cell_tuned_config(CONFIG)
    records = []
    for cell in config["cells"]:
        key, seed = _training_seed(config, cell["cell_id"], 0)
        records.append((key, seed))
        records.extend(
            _validation_seeds(config, cell["cell_id"], "candidate", 0)
        )

    assert len({key for key, _seed in records}) == len(records)
    assert len({seed for _key, seed in records}) == len(records)


def test_scaled_schedules_preserve_rank_one_and_negative_price_sign() -> None:
    profile = ((1.0, -0.5), (2.0, -1.0), (0.5, -0.25))
    scales = [0.0, 0.5, 1.0, 1.5]
    schedules = _scaled_schedules(profile, scales)

    for scale, schedule in zip(scales[1:], schedules[1:], strict=True):
        for source, pair in zip(profile, schedule, strict=True):
            assert math.isclose(pair[0], scale * source[0])
            assert math.isclose(pair[1], scale * source[1])
            assert pair[1] < 0.0
