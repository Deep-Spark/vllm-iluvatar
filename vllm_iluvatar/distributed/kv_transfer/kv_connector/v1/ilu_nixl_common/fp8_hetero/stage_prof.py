# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-stage timer for the FP8 host-wire staging pipeline."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.fp8_stage_prof")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


ENABLED: bool = _env_flag("VLLM_FP8_STAGE_PROF")
INTERVAL_S: float = float(os.environ.get("VLLM_FP8_STAGE_PROF_INTERVAL", "10"))
OUTPUT_DIR: str = os.environ.get(
    "VLLM_FP8_STAGE_PROF_DIR", "/workspace/scripts/logs/fp8_stage_prof"
)
STRICT: bool = ENABLED and _env_flag("VLLM_FP8_STAGE_PROF_STRICT")

_MAX_SAMPLES = 16384

_STAGE_ORDER = (
    "cpu.01.queue_wait",
    "cpu.02.blockids_h2d",
    "cpu.04.zero_scratch",
    "cpu.05.quant.QUANT_PACK",
    "cpu.05.quant.QUANT_PACK_COMPRESSED",
    "cpu.05.quant.QUANT_INDEXER",
    "cpu.05.quant.REPACK_ONLY",
    "cpu.05.quant.COPY_ONLY",
    "cpu.06.d2h_submit",
    "cpu.07.wait_ev_copy",
    "cpu.09.futures_wait",
    "cpu.10.stage_task_total",
    "gpu.quant_kernels",
    "gpu.d2h_copy",
)


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = int(q * (len(ordered) - 1) + 0.5)
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:8.2f}"


def _gpu_label(device_id: int | None) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    parts = [p.strip() for p in visible.split(",") if p.strip()]
    index = int(device_id) if device_id is not None else -1
    if 0 <= index < len(parts):
        return parts[index]
    return str(index) if index >= 0 else "unk"


class StageStats:
    """Thread-safe windowed accumulator. Written by the stager + completion threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, float] = defaultdict(float)
        self._window_start = time.perf_counter()
        self._context = ""
        self._io_lock = threading.Lock()
        self._gpu = "unk"
        self._file = None

    def set_context(self, context: str) -> None:
        with self._lock:
            self._context = context

    def bind_device(self, device_id: int | None) -> None:
        with self._io_lock:
            self._gpu = _gpu_label(device_id)

    def _write(self, lines: list[str]) -> None:
        with self._io_lock:
            if self._file is None:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                path = os.path.join(
                    OUTPUT_DIR,
                    f"fp8_stage_prof_gpu{self._gpu}_pid{os.getpid()}.log",
                )
                self._file = open(path, "a", encoding="utf-8")
                logger.info("[fp8-stage-prof] writing to %s", path)
            self._file.write("".join(f"{line}\n" for line in lines))
            self._file.flush()

    def record(self, key: str, seconds: float) -> None:
        with self._lock:
            bucket = self._samples[key]
            if len(bucket) < _MAX_SAMPLES:
                bucket.append(seconds)
            else:
                self._counters["dropped_samples"] += 1

    def add(self, key: str, value: float) -> None:
        with self._lock:
            self._counters[key] += value

    def _ordered_keys(self, keys) -> list[str]:
        known = [k for k in _STAGE_ORDER if k in keys]
        extra = sorted(k for k in keys if k not in _STAGE_ORDER)
        return known + extra

    def flush_if_due(self, force: bool = False) -> None:
        now = time.perf_counter()
        with self._lock:
            window = now - self._window_start
            if not force and window < INTERVAL_S:
                return
            if not self._samples and not self._counters:
                self._window_start = now
                return
            samples = self._samples
            counters = self._counters
            context = self._context
            self._samples = defaultdict(list)
            self._counters = defaultdict(float)
            self._window_start = now

        submits = int(counters.get("submits", 0))
        tasks = int(counters.get("stage_tasks", 0))
        d2h_bytes = counters.get("d2h_bytes", 0.0)
        rate = d2h_bytes / window / 1e6 if window > 0 else 0.0
        dropped = (
            f" dropped_samples={int(counters['dropped_samples'])}"
            if counters.get("dropped_samples")
            else ""
        )
        lines = [
            f"[{time.strftime('%m-%d %H:%M:%S')}] window={window:.1f}s "
            f"{context} stage_tasks={tasks} submits={submits} "
            f"d2h={d2h_bytes / 1e6:.1f}MB ({rate:.1f} MB/s){dropped}"
        ]
        counts = {k: v for k, v in counters.items() if k.startswith("n.")}
        if counts:
            lines.append(
                "  counts "
                + " ".join(f"{k[2:]}={int(v)}" for k, v in sorted(counts.items()))
            )
        lines.append(
            f"  {'stage':<26} {'n':>6} {'mean/ms':>8} {'p50/ms':>8} "
            f"{'p90/ms':>8} {'max/ms':>8} {'sum/ms':>9} {'%win':>6}"
        )
        for key in self._ordered_keys(samples.keys()):
            values = sorted(samples[key])
            total = sum(values)
            lines.append(
                f"  {key:<26} {len(values):>6} "
                f"{_fmt_ms(total / len(values))} "
                f"{_fmt_ms(_percentile(values, 0.50))} "
                f"{_fmt_ms(_percentile(values, 0.90))} "
                f"{_fmt_ms(values[-1])} "
                f"{total * 1000.0:>9.1f} "
                f"{(total / window * 100.0) if window > 0 else 0.0:>5.1f}%"
            )
        self._write(lines)


stats = StageStats()


class _NullStep:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info) -> bool:
        return False


_NULL_STEP = _NullStep()


@contextmanager
def _timed_step(key: str):
    if STRICT:
        torch.cuda.current_stream().synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if STRICT:
            torch.cuda.current_stream().synchronize()
        stats.record(key, time.perf_counter() - start)


def step(key: str):
    return _timed_step(key) if ENABLED else _NULL_STEP
