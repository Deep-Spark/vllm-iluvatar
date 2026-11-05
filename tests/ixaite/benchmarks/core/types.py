# SPDX-License-Identifier: Apache-2.0
"""Shared types for the cross-provider benchmark harness."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class Providers:
    """Explicit baseline and FlyDSL callables (never the auto-selected wrapper)."""

    baseline_name: str
    baseline: Callable[..., Any]
    flydsl: Callable[..., Any]
    flydsl_name: str = "ixaite"


@dataclass
class Workload:
    """One concrete run: tensors shared within a single provider invocation."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingResult:
    """Correctness / stability / latency for one provider on one case."""

    median_ms: float
    samples_ms: tuple[float, ...]
    stable: bool
    correct: bool
    error: str | None = None
    aux_metric: float | None = None
    aux_unit: str | None = None


@dataclass(frozen=True)
class CaseResult:
    """Paired baseline vs FlyDSL result for one case."""

    suite: str
    case_id: str
    model: str
    mode: str  # "bench" | "check_only"
    baseline_name: str
    flydsl_name: str
    dtype: str
    arch: str
    baseline: TimingResult
    flydsl: TimingResult
    speedup: float | None
    gate: str  # "pass" | "fail" | "skip"
    reason: str = ""
    tags: tuple[str, ...] = ()
    cuda_graph: bool = False


class BenchCase(Protocol):
    """One measurable workload under a suite ABI."""

    case_id: str
    tags: tuple[str, ...]

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload: ...

    def run(self, fn: Callable[..., Any], wl: Workload) -> None: ...

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None: ...

    def aux_from_ms(
        self, median_ms: float, wl: Workload
    ) -> tuple[float, str] | None: ...


class Suite(Protocol):
    """One provider ABI with many call-site cases."""

    name: str

    def resolve_providers(self) -> Providers: ...

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[BenchCase]: ...

    def gate_min_speedup(self) -> float: ...
