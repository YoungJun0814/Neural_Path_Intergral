"""Fresh IID or exactly proportional known-component Gaussian banks."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from .gaussian_span_marginalization import GaussianMixtureShiftSpec
from .research_result_contract import canonical_digest


def balanced_counts(weights: torch.Tensor, count: int) -> tuple[int, ...]:
    if (weights.ndim != 1 or len(weights) < 2 or weights.dtype != torch.float64
            or weights.device.type != "cpu" or not torch.isfinite(weights).all()
            or type(count) is not int or count < 1):
        raise ValueError("CPU float64 static guide and positive integer count required")
    k = len(weights)-1
    expected = torch.full_like(weights, .9/k)
    expected[0] = .1
    tile = math.lcm(10, 10*k//math.gcd(9, 10*k))
    if not torch.allclose(weights, expected, rtol=0, atol=1e-13) or count % tile:
        raise ValueError("exact .1/uniform-.9 proportional balanced tile required")
    result = (count//10,)+(9*count//(10*k),)*k
    if min(result) < 2 or sum(result) != count:
        raise ValueError("at least two samples per stratum required")
    return result


def bank_seed(key: dict[str, Any], component: int, offset: int, role: str) -> int:
    value = {"namespace": "bounded-bank-decision-v1", "key": key,
        "component": component, "absolute_offset": offset, "role": role}
    return int(hashlib.sha256(canonical_digest(value).encode()).hexdigest()[:15], 16)


def stream_layout(spec: GaussianMixtureShiftSpec, count: int, design: str,
                  key: dict[str, Any], *, chunk: int = 2048) -> list[dict[str, int]]:
    if type(chunk) is not int or chunk < 1 or type(count) is not int or count < 1:
        raise ValueError("positive fixed count/chunk required")
    if design not in ("iid", "stratified"):
        raise ValueError("iid or stratified design required")
    counts = balanced_counts(spec.weights, count) if design == "stratified" else (count,)
    rows = []
    for c, size in enumerate(counts):
        component = c if design == "stratified" else -1
        for offset in range(0, size, chunk):
            row = {"component": component, "offset": offset, "count": min(chunk, size-offset),
                "path_seed": bank_seed(key, component, offset, "path")}
            if design == "iid":
                row["label_seed"] = bank_seed(key, component, offset, "label")
            rows.append(row)
    seeds = [r[k] for r in rows for k in ("path_seed", "label_seed") if k in r]
    if len(set(seeds)) != len(seeds):
        raise ValueError("random-role collision")
    return rows


def draw_bank(spec: GaussianMixtureShiftSpec, count: int, design: str,
              key: dict[str, Any], *, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor, str]:
    if spec.means.dtype != torch.float64 or spec.means.device.type != "cpu":
        raise ValueError("CPU float64 Gaussian geometry required")
    layout = stream_layout(spec, count, design, key, chunk=chunk)
    zparts, labels = [], []
    for row in layout:
        n, c = row["count"], row["component"]
        noise = torch.randn((n, spec.dimension), dtype=torch.float64,
            generator=torch.Generator().manual_seed(row["path_seed"]))
        label = (torch.multinomial(spec.weights, n, replacement=True,
            generator=torch.Generator().manual_seed(row["label_seed"])) if c == -1 else torch.full((n,), c, dtype=torch.int64))
        zparts.append(noise+spec.means[label])
        labels.append(label)
    return torch.cat(zparts), torch.cat(labels), canonical_digest(layout)
