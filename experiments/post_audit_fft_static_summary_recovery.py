"""Recover all completed IID units after a non-sampling aggregation exception."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from experiments.post_audit_fft_static_fine_reference import (
    DIRECTORY,
    ROOT,
    identity,
    jobs,
    summarize,
)
from experiments.post_audit_r2_family_diagnosis import freeze_source
from src.path_integral.recovery_checkpoint import RecoveryJournal
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.structural_v2_inventory import audit_snapshot

NOTE = "docs/reviews/FFT_STATIC_SUMMARY_METADATA_RECOVERY_2026-10-16_KO.md"


def run() -> dict[str, Any]:
    directory = ROOT/DIRECTORY
    if (directory/"result.json").exists():
        raise ValueError("original successful result exists; do not replace it")
    manifest = json.loads((directory/"manifest.json").read_text())
    source = json.loads((directory/"source.json").read_text())
    audit_snapshot(ROOT, source, manifest["config"])
    journal, records = RecoveryJournal(directory, manifest), []
    relatives = [DIRECTORY+"/manifest.json", DIRECTORY+"/source.json"]
    for j in jobs():
        path = directory/(identity(j)+".finish.json")
        finish = journal._read(path)
        if finish["status"] != "complete":
            raise ValueError("only a complete nonselected set may be recovered")
        records.append(finish["result"])
        relatives.append(path.relative_to(ROOT).as_posix())
    if len(records) != 16 or journal.accounting("I")[0] != 1048576:
        raise ValueError("incomplete original work grid")
    config = {"reason": "two_level_merge_lost_log_maximum_metadata_no_new_samples",
              "immutable_files": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in relatives}}
    recovery_dir = directory/"postprocessing_recovery_v1"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    recovery_source = freeze_source(recovery_dir/"run.json", config, extra_snapshot_paths=(NOTE, *relatives))
    write_json_atomic_nonoverwriting(recovery_dir/"source.json", recovery_source)
    elapsed = time.time()-journal._read(directory/"phase-I.clock.json")["started_unix"]
    if elapsed > 7200 or (directory/"phase-I.violation.json").exists():
        raise RuntimeError("original operational phase including recovery exceeded its cap")
    output = {"config": manifest["config"], "source": source, "manifest": manifest,
              "recovery_metadata": {"config": config, "source": recovery_source,
                                    "new_physical_samples": 0, "original_execution_status": "aggregation_exception_after_all_units"},
              "summaries": summarize(records), "units": 16, "work": 1048576,
              "density_evaluations": 524288, "time_step_forward_evaluations": 3221225472,
              "p1_qualified": False, "p2_qualified": False, "large_training_authorized": False,
              "phase_wall": elapsed}
    write_json_atomic_nonoverwriting(directory/"recovered-result.json", output)
    return {"status": "complete_original_units_reaggregated_requires_numerical_replay_audit",
            "units": 16, "new_samples": 0, "phase_wall_including_recovery": elapsed}


if __name__ == "__main__":
    print(json.dumps(run()))
