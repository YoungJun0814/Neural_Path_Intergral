"""Bounded fixed-input two-thread power and numerical calibration, not inference."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import psutil
import torch

from experiments.post_audit_fine_backend_calibration import ANCHOR, ROOT
from experiments.post_audit_r1_diagnostics import _problem
from experiments.post_audit_r2_family_diagnosis import freeze_source
from experiments.post_audit_resource_stable_recovery import read_power
from src.path_integral.gradient_umbrella_smc import (
    estimate_gradient_umbrella_normalizers,
    evaluate_family_gradient,
)
from src.path_integral.paired_volterra_gradient_reference import PairedVolterraProbability
from src.path_integral.reference_shards import write_json_atomic_nonoverwriting
from src.path_integral.weighted_tempered_smc import WeightedSMCConfig

DIRECTORY = "results/post_audit/low_power_calibration_v1/session_20261019"
PLAN = "docs/plans/LOW_POWER_BACKEND_CALIBRATION_2026-10-19_KO.md"


def wmi_power() -> dict[str, Any]:
    command = ("Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction Stop | "
               "Select-Object PowerOnline,Charging,Discharging,ChargeRate,DischargeRate,RemainingCapacity | ConvertTo-Json -Compress")
    value = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command", command], timeout=15).decode("utf-8-sig"))
    if not isinstance(value, dict) or not all(k in value for k in ("PowerOnline", "Discharging", "RemainingCapacity")):
        raise ValueError("single battery telemetry required")
    return {"native": read_power(), "wmi": value}


def validate_power_samples(samples: list[dict[str, Any]]) -> None:
    if len(samples) < 3:
        raise ValueError("insufficient continuous-stress telemetry")
    for s in samples:
        if (s["native"]["ac_connected"] != 1 or not 20 <= s["native"]["battery_percent"] <= 100
                or s["wmi"]["PowerOnline"] is not True or s["wmi"]["Discharging"] is not False
                or isinstance(s["wmi"]["RemainingCapacity"], bool)
                or not isinstance(s["wmi"]["RemainingCapacity"], (int, float))
                or not 0 < s["wmi"]["RemainingCapacity"] < 1000000):
            raise ValueError("AC/discharge/battery/capacity telemetry failed")
    if (samples[-1]["native"]["battery_percent"] < samples[0]["native"]["battery_percent"]-1
            or samples[-1]["wmi"]["RemainingCapacity"] < samples[0]["wmi"]["RemainingCapacity"]-100):
        raise ValueError("two-thread load still depletes battery")


def assert_tree_close(a: Any, b: Any) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError("stage keys differ")
        for key in a:
            assert_tree_close(a[key], b[key])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("stage count differs")
        for x, y in zip(a, b, strict=True):
            assert_tree_close(x, y)
    elif isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=5e-10, abs_tol=1e-9):
            raise ValueError("floating stage values differ")
    elif type(a) is not type(b) or a != b:
        raise ValueError("discrete stage values differ")


def run() -> dict[str, Any]:
    begin = time.perf_counter()
    anchor = json.loads((ROOT/ANCHOR).read_text())
    cell = next(c for c in anchor["config"]["cells"] if c["id"] == "high-eta200-k1")
    config = {"model": anchor["config"]["model"], "cell": cell,
              "seed": int(hashlib.sha256(b"fine-backend-calibration-v1/input").hexdigest()[:15], 16),
              "anchor_sha256": hashlib.sha256((ROOT/ANCHOR).read_bytes()).hexdigest(),
              "threads": 2, "stress_seconds": 60., "maximum_callbacks": 514, "phase_seconds": 180.}
    directory = ROOT/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    source = freeze_source(directory/"run.json", config, extra_snapshot_paths=(PLAN, ANCHOR))
    write_json_atomic_nonoverwriting(directory/"source.json", source)
    calls, samples = 0, []
    try:
        paired = PairedVolterraProbability.create(_problem(config["model"], cell, 4096))
        z = torch.randn((128, paired.dimension), dtype=torch.float64, generator=torch.Generator().manual_seed(config["seed"]))
        weights = torch.full((2,), .5, dtype=torch.float64).log()
        torch.set_num_threads(8)
        baseline = evaluate_family_gradient(paired.log_components, z, weights, maximum_drift_norm=20.)
        calls += 1
        torch.set_num_threads(2)
        warm = evaluate_family_gradient(paired.log_components, z, weights, maximum_drift_norm=20.)
        calls += 1
        for a, b in zip(baseline, warm, strict=True):
            torch.testing.assert_close(a, b, rtol=5e-10, atol=1e-8)
        samples.append(wmi_power())
        start, next_sample = time.perf_counter(), 20.
        # Query in a helper thread so telemetry is taken under continuous load,
        # rather than pausing the stress callback during every WMI query.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending: Future[dict[str, Any]] | None = None
            while time.perf_counter()-start < 60:
                evaluate_family_gradient(paired.log_components, z, weights, maximum_drift_norm=20.)
                calls += 1
                if calls >= 514 or time.perf_counter()-begin > 180 or psutil.Process().memory_info().rss > 4*1024**3:
                    raise RuntimeError("calibration resource cap")
                if pending is not None and pending.done():
                    samples.append(pending.result())
                    pending = None
                if pending is None and time.perf_counter()-start >= next_sample:
                    pending = pool.submit(wmi_power)
                    print(f"two-thread calibration stress {int(time.perf_counter()-start)}s", flush=True)
                    next_sample += 20
            if pending is not None:
                samples.append(pending.result())
        samples.append(wmi_power())
        validate_power_samples(samples)
        toy = []
        means = torch.tensor([.2, -.1, .3, .1], dtype=torch.float64)

        def components(x: torch.Tensor) -> torch.Tensor:
            return -.05*(x-means).square().sum(1, keepdim=True)

        for rep in range(4):
            outputs = []
            for threads in (8, 2):
                torch.set_num_threads(threads)
                cfg = WeightedSMCConfig(256, tuple((k/15)**2 for k in range(16)), 2, .3, 1, 8200+rep,
                                        resample_every=4, resampling_scheme="stratified")
                result = estimate_gradient_umbrella_normalizers(components, dimension=4,
                                                               log_mixture_weights=torch.zeros(1, dtype=torch.float64), config=cfg)
                if result.potential_evaluations != 7424 or result.gradient_evaluations != 7424:
                    raise ValueError("toy work changed")
                outputs.append(result)
            assert_tree_close(list(outputs[0].stages), list(outputs[1].stages))
            if not math.isclose(outputs[0].log_normalizer, outputs[1].log_normalizer, abs_tol=1e-9, rel_tol=5e-10):
                raise ValueError("toy normalizers differ")
            toy.append(math.exp(outputs[1].log_normalizer))
        exact = 1.1**-2*math.exp(-.05*float(means.square().sum())/1.1)
        if abs(sum(toy)/4/exact-1) > .03 or time.perf_counter()-begin > 180:
            raise ValueError("toy normalizer oracle or phase cap failed")
        output = {"config": config, "source": source, "selected_threads": 2, "power_samples": samples,
                  "fixed_input_equivalence_pass": True, "full_toy_backend_equivalence_pass": True,
                  "callbacks": calls, "physical_forward": calls*256, "joint_backward": calls*128,
                  "toy_potential_forward": 59392, "toy_backward": 59392, "wall": time.perf_counter()-begin,
                  "scope": "operational_calibration_not_estimator_efficiency_or_accuracy"}
        write_json_atomic_nonoverwriting(directory/"result.json", output)
        return output
    except Exception as error:
        write_json_atomic_nonoverwriting(directory/"failure.json", {"error": type(error).__name__+": "+str(error),
                                                                   "wall": time.perf_counter()-begin, "config": config,
                                                                   "source": source, "power_samples": samples,
                                                                   "completed_callbacks": calls,
                                                                   "completed_callback_physical_forward": calls*256,
                                                                   "reserved_physical_forward": 131584})
        raise


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run().items() if k in ("selected_threads", "callbacks", "physical_forward", "wall")}))
