"""Post-run wall/CPU/peak-memory supplement for V6 method artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.path_integral.provenance import runtime_provenance, source_provenance

_SCHEMAS = {
    "npi.g11.v6-baseline-qualification.v1",
    "npi.g11.v6-routed-policy.v1",
    "npi.g11.v6-secondary-baselines.v1",
}


def summarize_resources(source_path: Path) -> dict[str, Any]:
    raw = source_path.read_bytes()
    source = json.loads(raw)
    if not isinstance(source, dict) or source.get("schema") not in _SCHEMAS:
        raise ValueError("unsupported V6 resource-supplement source")
    source_records = source.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("resource-supplement source contains no records")
    records = []
    for record in source_records:
        method = str(record.get("method", "v6_policy"))
        work_records = record["result"]["total_work"]["records"]
        wall = math.fsum(float(item["wall_seconds"]) for item in work_records)
        cpu = math.fsum(float(item["cpu_seconds"]) for item in work_records)
        peak = max(int(item["peak_memory_bytes"]) for item in work_records)
        if (
            not math.isfinite(wall)
            or wall < 0.0
            or not math.isfinite(cpu)
            or cpu < 0.0
            or peak < 0
        ):
            raise ValueError("invalid serialized resource measurement")
        records.append(
            {
                "cell_id": str(record["cell_id"]),
                "cluster": int(record["cluster"]),
                "method": method,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "peak_resident_memory_bytes": peak,
                "operation_work_units": math.fsum(
                    float(item["work_units"]) for item in work_records
                ),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    summaries = {}
    for method, values in sorted(grouped.items()):
        summaries[method] = {
            "record_count": len(values),
            "total_wall_seconds": math.fsum(item["wall_seconds"] for item in values),
            "total_cpu_seconds": math.fsum(item["cpu_seconds"] for item in values),
            "maximum_peak_resident_memory_bytes": max(
                item["peak_resident_memory_bytes"] for item in values
            ),
            "total_operation_work_units": math.fsum(
                item["operation_work_units"] for item in values
            ),
        }
    source_environment = source.get("environment", {})
    gates = {
        "all_cpu_measurements_positive": all(
            record["cpu_seconds"] > 0.0 for record in records
        ),
        "all_peak_memory_measurements_positive": all(
            record["peak_resident_memory_bytes"] > 0 for record in records
        ),
        "thread_count_recorded": isinstance(source_environment, dict)
        and isinstance(source_environment.get("torch_threads"), int)
        and source_environment["torch_threads"] > 0,
        "complete_record_identity": len(
            {
                (record["cell_id"], record["cluster"], record["method"])
                for record in records
            }
        )
        == len(records),
    }
    provenance = source_provenance()
    return {
        "schema": "npi.g11.v6-resource-supplement.v1",
        "measurement_semantics": {
            "wall_and_cpu": (
                "sum of sequential ledger intervals attached to each result record"
            ),
            "peak_resident_memory": (
                "absolute process high-water mark observed by the worker process; "
                "not an additive or baseline-subtracted per-method allocation"
            ),
            "method_totals": (
                "standalone accounting totals; shared proposal-training cost is charged "
                "according to each method artifact's declared amortization contract"
            ),
        },
        "source_schema": source["schema"],
        "source_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_environment": source_environment,
        "records": records,
        "method_summaries": summaries,
        "gates": gates,
        "passed": all(gates.values()),
        "environment": runtime_provenance(dtype="serialized-float64"),
        **provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = summarize_resources(arguments.source)
    if arguments.output.exists():
        raise FileExistsError("resource supplement refuses to overwrite an artifact")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], **result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
