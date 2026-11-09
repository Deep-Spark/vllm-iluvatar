# SPDX-License-Identifier: Apache-2.0
"""Normalize and write e2e serving benchmark reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_METRICS: tuple[tuple[str, str, str], ...] = (
    ("request_throughput", "request_throughput", "req/s"),
    ("output_throughput", "output_throughput", "tok/s"),
    ("total_token_throughput", "total_token_throughput", "tok/s"),
    ("median_ttft_ms", "median_ttft_ms", "ms"),
    ("p99_ttft_ms", "p99_ttft_ms", "ms"),
    ("median_tpot_ms", "median_tpot_ms", "ms"),
    ("p99_tpot_ms", "p99_tpot_ms", "ms"),
    ("median_itl_ms", "median_itl_ms", "ms"),
    ("p99_itl_ms", "p99_itl_ms", "ms"),
)
_THROUGHPUT_METRICS = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
}


def _normalized_workload(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        output_name: {"value": raw.get(raw_name), "unit": unit}
        for output_name, raw_name, unit in _METRICS
        if raw.get(raw_name) is not None
    }
    return {
        "completed": raw.get("completed"),
        "failed": raw.get("failed"),
        "duration_s": raw.get("duration"),
        "metrics": metrics,
    }


def build_compare(
    *,
    model: str,
    model_path: Path,
    phases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a renderer-neutral comparison payload."""
    normalized_phases: dict[str, Any] = {}
    for provider, phase in phases.items():
        normalized_phases[provider] = {
            "workloads": {
                workload_id: _normalized_workload(raw)
                for workload_id, raw in phase["workloads"].items()
            },
        }

    speedups: dict[str, dict[str, float]] = {}
    if "baseline" in phases and "ixaite" in phases:
        baseline = normalized_phases["baseline"]["workloads"]
        ixaite = normalized_phases["ixaite"]["workloads"]
        for workload_id in sorted(baseline.keys() & ixaite.keys()):
            workload_speedups: dict[str, float] = {}
            baseline_metrics = baseline[workload_id]["metrics"]
            ixaite_metrics = ixaite[workload_id]["metrics"]
            for metric in sorted(baseline_metrics.keys() & ixaite_metrics.keys()):
                baseline_value = baseline_metrics[metric]["value"]
                ixaite_value = ixaite_metrics[metric]["value"]
                if not baseline_value or not ixaite_value:
                    continue
                if metric in _THROUGHPUT_METRICS:
                    speedup = ixaite_value / baseline_value
                else:
                    speedup = baseline_value / ixaite_value
                workload_speedups[metric] = speedup
            speedups[workload_id] = workload_speedups

    return {
        "schema_version": 2,
        "model": {"slug": model, "path": str(model_path)},
        "speedup_meaning": (
            "ixaite/baseline for throughput and baseline/ixaite for latency; "
            ">1 means ixaite is better"
        ),
        "phases": normalized_phases,
        "speedups": speedups,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_metrics_csv(path: Path, compare: dict[str, Any]) -> None:
    fieldnames = ["provider", "workload", "metric", "value", "unit", "speedup"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for provider, phase in compare["phases"].items():
            for workload_id, workload in phase["workloads"].items():
                speedups = compare["speedups"].get(workload_id, {})
                for metric, value in workload["metrics"].items():
                    writer.writerow(
                        {
                            "provider": provider,
                            "workload": workload_id,
                            "metric": metric,
                            "value": value["value"],
                            "unit": value["unit"],
                            "speedup": speedups.get(metric),
                        }
                    )


def summary(compare: dict[str, Any]) -> str:
    lines = [
        f"model: {compare['model']['slug']}",
        "speedup: >1 means ixaite is better",
    ]
    for workload_id, metrics in compare["speedups"].items():
        output = metrics.get("output_throughput")
        ttft = metrics.get("median_ttft_ms")
        parts = []
        if output is not None:
            parts.append(f"output throughput {output:.3f}x")
        if ttft is not None:
            parts.append(f"median TTFT {ttft:.3f}x")
        lines.append(f"{workload_id}: " + (", ".join(parts) or "no comparable metrics"))
    correctness = compare.get("correctness")
    if correctness is not None:
        verdict = "exact match" if correctness["matched"] else "mismatch"
        providers = correctness.get("providers")
        if providers is not None:
            checked = ", ".join(
                f"{provider} ({detail['repeats']}x replay"
                + (" + eager" if detail["eager_compared"] else "")
                + ")"
                for provider, detail in sorted(providers.items())
            )
            verdict = f"{verdict} -- {checked}"
        lines.append(f"greedy correctness: {verdict}")
    if not compare["speedups"] and correctness is None:
        lines.append("comparison unavailable: only one provider was run")
    return "\n".join(lines) + "\n"
