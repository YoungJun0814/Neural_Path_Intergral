from __future__ import annotations

from experiments.g11_v8_p5_sharded_reference_common import ROOT, load_context

CONFIG = ROOT / "configs/g11_v8/p5_sharded_reference_execution_v4.yaml"


def test_v4_context_binds_complete_method_cell_proposals() -> None:
    context = load_context(CONFIG)

    assert len(context.cells_by_id) == 24
    assert len(context.proposal_entries_by_key) == 48
    assert context.reference_parent_sha256 == context.config[
        "proposal_manifest"
    ]["sha256"]
    assert context.config["sampling"]["pilot_namespace"].endswith("-pilot-v1")
