# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host-wire stager thread: transform + move, then hand off to the writer."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

from .host_pool import HostKVPool
from .stage_prof import ENABLED as PROF
from .stage_prof import stats as prof
from .staging import StagingPipeline
from .xfer_trace import ENABLED as TRACE
from .xfer_trace import trace

_TIMED = PROF or TRACE

logger = init_logger(__name__)


@dataclass
class StageTask:
    """One (request, chunk, write batch) unit of quantization work."""

    request_id: str
    chunk_index: int
    layer_names: tuple[str, ...]
    ready_event: Any
    quant_block_ids: tuple[tuple[int, ...], ...]
    quant_first_pages: tuple[int, ...]
    num_valid_tokens: int
    requantize_block_ids: tuple[tuple[int, ...], ...] = ()
    is_last_chunk: bool = False
    make_writer_task: Callable[[], Any] = field(default=lambda: None)
    enqueued_at: float = field(default_factory=time.perf_counter if _TIMED else float)


class HostWireStager(threading.Thread):
    """Drains ``stage_inbox`` and refills the writer's inbox."""

    def __init__(
        self,
        pipeline: StagingPipeline,
        pool: HostKVPool,
        stage_inbox: queue.Queue[StageTask | None],
        writer_inbox: queue.Queue[Any],
        writer_wake: threading.Event,
        device_id: int,
    ) -> None:
        super().__init__(name="nixl-host-wire-stager", daemon=True)
        self.pipeline = pipeline
        self.pool = pool
        self._stage_inbox = stage_inbox
        self._writer_inbox = writer_inbox
        self._writer_wake = writer_wake
        self._device_id = device_id
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self._stage_inbox.put(None)

    def run(self) -> None:
        current_platform.set_device(self._device_id)
        while not self._stop.is_set():
            task = self._stage_inbox.get()
            if task is None:
                return
            try:
                self._stage(task)
            except Exception:
                logger.exception(
                    "Host wire staging failed for %s chunk %d layers %s; "
                    "failing closed",
                    task.request_id,
                    task.chunk_index,
                    task.layer_names,
                )

    def _stage(self, task: StageTask) -> None:
        if _TIMED:
            task_start = time.perf_counter()
            queue_wait = task_start - task.enqueued_at
            if PROF:
                prof.record("cpu.01.queue_wait", queue_wait)
                prof.add("stage_tasks", 1)
        if TRACE:
            trace.set_task(task.request_id, task.chunk_index)
            trace_begin = trace.now_ms()
        futures = []
        num_groups = max(len(task.quant_block_ids), len(task.requantize_block_ids))
        for group_idx in range(num_groups):
            layers = self._group_layers(task.layer_names, group_idx)
            if not layers:
                continue
            block_ids = (
                task.quant_block_ids[group_idx]
                if group_idx < len(task.quant_block_ids)
                else ()
            )
            stale = (
                task.requantize_block_ids[group_idx]
                if group_idx < len(task.requantize_block_ids)
                else ()
            )
            if PROF:
                bucket = "c0" if task.chunk_index == 0 else "cN"
                prof.add(f"n.g{group_idx}.{bucket}.tasks", 1)
                prof.add(f"n.g{group_idx}.{bucket}.blocks", len(block_ids))
                prof.add(
                    f"n.g{group_idx}.{bucket}.first_page",
                    task.quant_first_pages[group_idx]
                    if group_idx < len(task.quant_first_pages)
                    else -1,
                )
            if block_ids:
                fresh = self.pipeline.submit(
                    layers,
                    list(block_ids),
                    task.quant_first_pages[group_idx],
                    task.num_valid_tokens,
                    task.ready_event,
                )
                futures += fresh
                if PROF:
                    prof.add(f"n.g{group_idx}.submits", len(fresh))
                    prof.add(f"n.g{group_idx}.blocks", len(block_ids))
                    prof.add(f"n.g{group_idx}.layers", len(layers))
            if stale:
                requant = self.pipeline.submit(
                    layers, list(stale), 0, None, task.ready_event
                )
                futures += requant
                if PROF:
                    prof.add(f"n.g{group_idx}.requant_submits", len(requant))
                    prof.add(f"n.g{group_idx}.requant_blocks", len(stale))
        if _TIMED:
            t0 = time.perf_counter()
        for future in futures:
            future.result()
        if _TIMED:
            futures_wait = time.perf_counter() - t0
            if PROF:
                prof.record("cpu.09.futures_wait", futures_wait)

        writer_task = task.make_writer_task()
        if writer_task is not None:
            self._writer_inbox.put(writer_task)
            self._writer_wake.set()
        if _TIMED:
            total = time.perf_counter() - task_start
            if PROF:
                prof.record("cpu.10.stage_task_total", total)
                prof.flush_if_due()
            if TRACE:
                trace.emit(
                    (
                        "T",
                        task.request_id,
                        task.chunk_index,
                        len(task.layer_names),
                        round(trace_begin, 3),
                        round(queue_wait * 1000.0, 3),
                        round(futures_wait * 1000.0, 3),
                        round(total * 1000.0, 3),
                    )
                )

    def _group_layers(
        self, layer_names: tuple[str, ...], group_idx: int
    ) -> tuple[str, ...]:
        layer_to_group = self.pipeline.plan.layer_to_group
        return tuple(
            name for name in layer_names if layer_to_group.get(name) == group_idx
        )
