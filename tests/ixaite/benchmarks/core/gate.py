# SPDX-License-Identifier: Apache-2.0
"""Per-case and suite-level gate evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from tests.ixaite.benchmarks.core.types import CaseResult, TimingResult


def per_case_gate(
    baseline: TimingResult,
    flydsl: TimingResult,
    *,
    min_speedup: float,
    check_only: bool,
) -> tuple[str, str]:
    """Return ``(gate, reason)`` for one paired result."""
    if not (baseline.correct and flydsl.correct):
        err = baseline.error or flydsl.error or "incorrect"
        return "fail", f"correctness: {err}"
    if check_only:
        return "pass", "check-only ok"
    if not (baseline.stable and flydsl.stable):
        which = []
        if not baseline.stable:
            which.append("baseline")
        if not flydsl.stable:
            which.append("flydsl")
        return "fail", "unstable:" + "+".join(which)
    if flydsl.median_ms <= 0:
        return "fail", "non-positive flydsl median"
    speedup = baseline.median_ms / flydsl.median_ms
    if speedup < min_speedup:
        if speedup > 0:
            return (
                "fail",
                f"need >= {min_speedup} (flydsl {1.0 / speedup:.2f}x slower)",
            )
        return "fail", f"speedup {speedup:.3f} < {min_speedup}"
    return "pass", f"ok (>= {min_speedup})"


def evaluate_suite(results: Sequence[CaseResult], *, check_only: bool) -> int:
    """Return nonzero when a run fails or executes no selected case."""
    del check_only  # gate field already encodes check-only semantics
    if not results or all(result.gate == "skip" for result in results):
        return 1
    if any(result.gate == "fail" for result in results):
        return 1
    return 0
