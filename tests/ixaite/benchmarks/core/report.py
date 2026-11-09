# SPDX-License-Identifier: Apache-2.0
"""JSON / CSV report writers for benchmark runs."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.ixaite.benchmarks.core.types import CaseResult


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _git_sha(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _timing_dict(result: CaseResult, side: str) -> dict[str, Any]:
    timing = result.baseline if side == "baseline" else result.flydsl
    return {
        "suite": result.suite,
        "case_id": result.case_id,
        "model": result.model,
        "mode": result.mode,
        "cuda_graph": result.cuda_graph,
        "arch": result.arch,
        "dtype": result.dtype,
        "backend": side,
        "backend_name": result.baseline_name
        if side == "baseline"
        else result.flydsl_name,
        "median_ms": (
            None if result.mode == "check_only" else _finite_or_none(timing.median_ms)
        ),
        "aux_metric": _finite_or_none(timing.aux_metric),
        "aux_unit": timing.aux_unit,
        "correct": timing.correct,
        "stable": (
            None
            if result.mode == "check_only" or result.gate == "skip"
            else timing.stable
        ),
        "speedup": _finite_or_none(result.speedup),
        "gate": result.gate,
        "reason": result.reason,
        "tags": ",".join(result.tags),
        "error": timing.error,
    }


def build_report_payload(
    results: Sequence[CaseResult],
    *,
    repo_root: Path,
    argv: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serializable report with run metadata."""
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "git_sha": _git_sha(repo_root),
        "argv": list(argv),
        "results": [
            {
                "suite": item.suite,
                "case_id": item.case_id,
                "model": item.model,
                "mode": item.mode,
                "cuda_graph": item.cuda_graph,
                "arch": item.arch,
                "dtype": item.dtype,
                "baseline_name": item.baseline_name,
                "flydsl_name": item.flydsl_name,
                "baseline": {
                    "median_ms": _finite_or_none(item.baseline.median_ms),
                    "stable": None if item.gate == "skip" else item.baseline.stable,
                    "correct": item.baseline.correct,
                    "aux_metric": _finite_or_none(item.baseline.aux_metric),
                    "aux_unit": item.baseline.aux_unit,
                    "error": item.baseline.error,
                },
                "flydsl": {
                    "median_ms": _finite_or_none(item.flydsl.median_ms),
                    "stable": None if item.gate == "skip" else item.flydsl.stable,
                    "correct": item.flydsl.correct,
                    "aux_metric": _finite_or_none(item.flydsl.aux_metric),
                    "aux_unit": item.flydsl.aux_unit,
                    "error": item.flydsl.error,
                },
                "speedup": _finite_or_none(item.speedup),
                "speedup_meaning": (
                    "median_ms(baseline) / median_ms(flydsl); >1 means FlyDSL faster"
                ),
                "speedup_summary": (
                    None
                    if item.speedup is None
                    else (
                        f"flydsl {item.speedup:.2f}x faster"
                        if item.speedup >= 1.0
                        else (
                            f"flydsl {1.0 / item.speedup:.2f}x slower"
                            if item.speedup > 0
                            else "invalid"
                        )
                    )
                ),
                "gate": item.gate,
                "reason": item.reason,
                "tags": list(item.tags),
            }
            for item in results
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, results: Sequence[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite",
        "case_id",
        "model",
        "mode",
        "cuda_graph",
        "arch",
        "dtype",
        "backend",
        "backend_name",
        "median_ms",
        "aux_metric",
        "aux_unit",
        "correct",
        "stable",
        "speedup",
        "gate",
        "reason",
        "tags",
        "error",
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(_timing_dict(result, "baseline"))
        rows.append(_timing_dict(result, "flydsl"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
