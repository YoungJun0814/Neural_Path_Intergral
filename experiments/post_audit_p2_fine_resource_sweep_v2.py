"""Separate fixed whole/phase budget restart after preserved operational failure."""

from __future__ import annotations

import argparse
import json

from experiments.post_audit_p2_fine_resource_sweep import ROOT, audit, run

PLAN = "docs/plans/P2_FINE_RESOURCE_OPERATIONAL_RESTART_2026-10-16_KO.md"
DIRECTORY = "results/post_audit/p2_fine_resource_v2/session_20261016"
NAMESPACE = "paired-fine-resources-v2"
FAILURE = "results/post_audit/p2_fine_resource_v1/session_20261016/canonical-h005-k1-more-temperatures-0.finish.json"
MU_RESULT = "results/post_audit/canonical_mu_revalidation_v1/session_20261016/result.json"
BACKEND = "results/post_audit/fine_backend_calibration_v1/session_20261016/result.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(ROOT/args.audit)))
    else:
        if not (ROOT/MU_RESULT).is_file():
            raise RuntimeError("complete canonical mu first; no competing large local experiment")
        backend = json.loads((ROOT/BACKEND).read_text())
        output = run(directory_name=DIRECTORY, namespace=NAMESPACE, phase_wall=21600., unit_wall=600.,
                     plan=PLAN, failure_anchor=FAILURE, torch_threads=backend["selected_threads"], backend_anchor=BACKEND)
        print(json.dumps({k: v for k, v in output.items() if k in ("units", "work", "phase_wall")}))
