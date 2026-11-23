# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-request event trace for the FP8 host-wire push path."""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque

from vllm.logger import init_logger

from .stage_prof import _env_flag, _gpu_label

logger = init_logger("vllm.fp8_xfer_trace")

ENABLED: bool = _env_flag("VLLM_FP8_XFER_TRACE")
OUTPUT_DIR: str = os.environ.get(
    "VLLM_FP8_XFER_TRACE_DIR", "/workspace/scripts/logs/fp8_xfer_trace"
)
_MAX_QUEUED = 1 << 20
_DRAIN_INTERVAL_S = 0.25

_HEADERS = (
    "# T task      req chunk nlayers t_begin_ms queue_wait_ms "
    "futures_wait_ms total_ms",
    "# S submit    seq req chunk group kind nlayers blocks first_page bytes "
    "t_begin_ms blockids_h2d_ms zero_ms d2h_submit_ms",
    "# H submit'   seq wait_ev_ms gpu_quant_ms gpu_d2h_ms",
    "# Q quant     seq layer kind ms",
    "# W write     seq req chunk layers ndesc t_submit_ms",
    "# X write'    seq t_done_ms",
)


class _Trace:
    """Lock-guarded append buffer plus a daemon writer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[tuple] = deque()
        self._seq = 0
        self._dropped = 0
        self._gpu = "unk"
        self._file = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ctx = threading.local()
        self._t0 = time.perf_counter()

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def emit(self, record: tuple) -> None:
        with self._lock:
            if len(self._queue) >= _MAX_QUEUED:
                self._dropped += 1
                return
            self._queue.append(record)

    def set_task(self, request_id: str, chunk_index: int) -> None:
        self._ctx.task = (request_id, chunk_index)

    def task(self) -> tuple[str, int]:
        return getattr(self._ctx, "task", ("?", -1))

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def bind_device(self, device_id: int | None) -> None:
        self._gpu = _gpu_label(device_id)
        self._start()

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._drain_loop, daemon=True, name="fp8-xfer-trace"
            )
        self._thread.start()
        atexit.register(self._final_flush)

    def _open(self):
        if self._file is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            path = os.path.join(
                OUTPUT_DIR, f"fp8_xfer_trace_gpu{self._gpu}_pid{os.getpid()}.log"
            )
            self._file = open(path, "a", encoding="utf-8", buffering=1 << 20)
            self._file.write("\n".join(_HEADERS) + "\n")
            logger.info("[fp8-xfer-trace] writing to %s", path)
        return self._file

    def _drain_loop(self) -> None:
        while not self._stop.wait(_DRAIN_INTERVAL_S):
            self._flush()
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue
            self._queue = deque()
            dropped = self._dropped
            self._dropped = 0
        try:
            handle = self._open()
            handle.write("".join(" ".join(map(str, r)) + "\n" for r in batch))
            if dropped:
                handle.write(f"! dropped {dropped}\n")
            handle.flush()
        except Exception:
            logger.exception("[fp8-xfer-trace] flush failed, dropping batch")

    def _final_flush(self) -> None:
        self._stop.set()
        self._flush()


trace = _Trace()
