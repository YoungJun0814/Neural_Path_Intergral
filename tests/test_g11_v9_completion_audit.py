from __future__ import annotations

import copy
import json
from pathlib import Path

from src.path_integral.v9_completion_audit import audit_v9_completion

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "configs/g11_v9/completion_status_ledger_v1.yaml"


def test_v9_completion_audit_accepts_valid_falsification_closure() -> None:
    audit = audit_v9_completion(ledger_path=LEDGER, root=ROOT)
    assert audit.passed, audit.failures
    assert audit.scientific_status == "closed_by_development_falsification"


def test_v9_completion_audit_detects_development_mutation(tmp_path: Path) -> None:
    ledger = __import__("yaml").safe_load(LEDGER.read_text(encoding="utf-8"))
    development_binding = ledger["bindings"]["development"]
    original = ROOT / development_binding["path"]
    mutated = copy.deepcopy(json.loads(original.read_text(encoding="utf-8")))
    mutated["aggregate"]["stage_pass"] = True
    target = tmp_path / "development.json"
    target.write_text(json.dumps(mutated), encoding="utf-8")
    development_binding["path"] = target.relative_to(tmp_path).as_posix()
    development_binding["sha256"] = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    for binding in ledger["bindings"].values():
        if binding is development_binding:
            continue
        binding["path"] = (ROOT / binding["path"]).as_posix()
    ledger_path = tmp_path / "ledger.yaml"
    ledger_path.write_text(__import__("yaml").safe_dump(ledger), encoding="utf-8")
    failed = audit_v9_completion(ledger_path=ledger_path, root=tmp_path)
    assert not failed.passed
    assert "falsification_decision" in failed.failures
