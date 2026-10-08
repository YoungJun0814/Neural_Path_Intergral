"""Render descriptive README evidence from the immutable R2 development artifact."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_SHA256 = "1baf805ebdc3c222c86d9b8dfff115c14576bc0b5fa16095b5cb253b2ea2d852"
METHODS = ("direct-q", "event-auxiliary", "risk-auxiliary", "volterra-only")
LABELS = ("Original q\nrisk check", "Event-fit\n+ guide", "Risk-fit\n+ guide", "Static guide\nonly")


def main() -> None:
    """Check the source artifact and render sample-count-normalized risk variance."""
    artifact = ROOT / "results/post_audit/r2_auxiliary_volterra_safeguard_v7.json.gz"
    with gzip.open(artifact, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise ValueError("README figure source artifact has changed")
    payload = json.loads(raw)
    records = payload["records"]
    if payload["model_q_changed"] or len(records) != 50 or any(
            r["status"] != "completed_development" for r in records):
        raise ValueError("expected 50 completed whole auxiliary-fit jobs")
    cells = (("canonical-h005-k1", "Canonical: eta = 1.5"),
             ("high-eta200-k1", "High volatility-of-volatility: eta = 2.0"))
    matplotlib.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": RAW_SHA256})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6), sharey=True)
    for ax, (cell, title) in zip(axes, cells, strict=True):
        rows = [r for r in records if r["cell"]["id"] == cell]
        if len(rows) != 25:
            raise ValueError(f"expected 25 evaluations for {cell}")
        proposals = {r["q_digest"] for r in rows}
        if len(proposals) != 5 or any(
                sorted(r["auxiliary_training_rep"] for r in rows if r["q_digest"] == q)
                != list(range(5)) for q in proposals):
            raise ValueError(f"expected five fixed proposals with five fresh fits: {cell}")
        values = []
        for method in METHODS:
            estimates = [next(e for e in r["estimators"] if e["id"] == method) for r in rows]
            expected_count = 262144 if method == "risk-auxiliary" else 131072
            if any(e["status"] != "completed_development" or
                   e["summary"]["count"] != expected_count for e in estimates):
                raise ValueError(f"incomplete or mismatched final sample count: {cell}/{method}")
            values.append(float(np.median([e["summary"]["count"] *
                                           e["summary"]["relative_se"] ** 2 for e in estimates])))
        ax.bar(range(4), values, width=.62, color=("#65758b", "#b67c48", "#356bbb", "#26836a"))
        ax.set_yscale("log")
        ax.set_ylim(50, 100000)
        ax.set_xticks(range(4), LABELS)
        ax.set_title(title, fontsize=12, pad=14)
        ax.grid(axis="y", which="major", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            ax.text(i, value * 1.15, f"{value:,.1f}", ha="center", fontsize=10)
    axes[0].set_ylabel("Median N x RSE squared (lower is better)")
    fig.suptitle("Auxiliary M2 risk estimation: descriptive development evidence", fontsize=15, y=.97)
    fig.text(.5, .90, "25 evaluations per cell: 5 fixed model proposals x 5 fresh auxiliary fits", ha="center")
    fig.text(.5, .085, "Risk-fit N = 262,144; other methods N = 131,072. No confidence interval or speedup claim.",
             ha="center", fontsize=10)
    fig.text(.5, .045, "Unseen tails may affect sample variance. This figure does not measure event-probability performance.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.80, bottom=.22, left=.09, right=.98, wspace=.12)
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "r2_auxiliary_risk_summary.png", dpi=180, facecolor="white")
    svg = io.StringIO()
    fig.savefig(svg, format="svg", facecolor="white",
                metadata={"Description": f"Source raw SHA256: {RAW_SHA256}", "Date": "2026-10-07"})
    (output / "r2_auxiliary_risk_summary.svg").write_text(
        "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print(output / "r2_auxiliary_risk_summary.png")


if __name__ == "__main__":
    main()
