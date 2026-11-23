# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HBM KV -> GPU scratch -> pinned HostKVPool pipeline (direct D2H only)."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .host_pool import HostKVPool
from .quant import quantize_layer
from .stage_prof import ENABLED as PROF
from .stage_prof import stats as prof
from .wire_layout import SCRATCH_BUFFER_COUNT, WirePlan
from .xfer_trace import ENABLED as TRACE
from .xfer_trace import trace

TIMED = PROF or TRACE

logger = init_logger(__name__)


@dataclass(frozen=True)
class StageLayout:
    """Dense scratch packing for one (write batch, compute group) pair."""

    layers: tuple[str, ...]
    layer_offsets: tuple[int, ...]
    unit_stride: int
    group_idx: int
    dst_offsets: torch.Tensor
    src_offsets: torch.Tensor
    lengths: torch.Tensor

    @property
    def num_spans(self) -> int:
        return int(self.lengths.numel())

    @property
    def min_span_bytes(self) -> int:
        return int(self.lengths.min()) if self.num_spans else 0


def build_stage_layout(plan: WirePlan, layer_names: tuple[str, ...]) -> StageLayout:
    """Pack ``layer_names`` densely in pool-offset order."""
    ordered = sorted(layer_names, key=lambda name: plan.byte_offset[name])
    groups = {plan.layer_to_group[name] for name in ordered}
    if len(groups) != 1:
        raise ValueError(
            "A staging write batch must belong to exactly one compute group: "
            f"layers={ordered[:8]} span groups {sorted(groups)}"
        )
    spans: list[list[int]] = []
    offsets: list[int] = []
    cursor = 0
    for name in ordered:
        dst = plan.byte_offset[name]
        length = plan.page_bytes[name]
        offsets.append(cursor)
        if spans and spans[-1][0] + spans[-1][2] == dst:
            spans[-1][2] += length
        else:
            spans.append([dst, cursor, length])
        cursor += length
    return StageLayout(
        layers=tuple(ordered),
        layer_offsets=tuple(offsets),
        unit_stride=cursor,
        group_idx=next(iter(groups)),
        dst_offsets=torch.tensor([s[0] for s in spans], dtype=torch.int64),
        src_offsets=torch.tensor([s[1] for s in spans], dtype=torch.int64),
        lengths=torch.tensor([s[2] for s in spans], dtype=torch.int64),
    )


class StagingPipeline:
    """Ping-ponged quant -> direct D2H into the pinned host pool."""

    def __init__(
        self,
        plan: WirePlan,
        pool: HostKVPool,
        compute_caches: dict[str, torch.Tensor],
        *,
        rows: Sequence[int],
        unit_bytes: Sequence[int],
        device: torch.device,
        compute_scales: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.plan = plan
        self.pool = pool
        self.compute_caches = compute_caches
        self.compute_scales = compute_scales or {}
        self.rows = tuple(int(value) for value in rows)
        self.unit_bytes = tuple(int(value) for value in unit_bytes)
        if len(self.rows) != len(self.unit_bytes):
            raise ValueError(
                "Per-group rows and unit_bytes must describe the same groups: "
                f"{len(self.rows)} != {len(self.unit_bytes)}"
            )
        if len(self.rows) < len(plan.group_layers):
            raise ValueError(
                "Host wire staging was sized for "
                f"{len(self.rows)} groups but the plan has "
                f"{len(plan.group_layers)}"
            )
        self._device_id = int(device.index) if device.index is not None else 0
        self._layouts: dict[tuple[str, ...], StageLayout] = {}
        self._counters = [0] * len(self.rows)

        buffer_bytes = [r * u for r, u in zip(self.rows, self.unit_bytes)]
        with torch.cuda.device(device):
            self.scratch = [
                [
                    torch.empty(nbytes, dtype=torch.uint8, device=device)
                    for _ in range(SCRATCH_BUFFER_COUNT)
                ]
                for nbytes in buffer_bytes
            ]
            self.quant_stream = torch.cuda.Stream(device=device)
            self.copy_stream = torch.cuda.Stream(device=device)
            self.ev_quant = self._per_group_events(len(buffer_bytes))
            self.ev_copy = self._per_group_events(len(buffer_bytes))
            self.ev_ids = self._per_group_events(len(buffer_bytes))
            self.prof_events = (
                {
                    name: self._per_group_events(len(buffer_bytes), timing=True)
                    for name in ("quant_start", "quant_end", "copy_start", "copy_end")
                }
                if PROF
                else None
            )
        self.ids_host = [
            [
                torch.empty(r, dtype=torch.int64, device="cpu", pin_memory=True)
                for _ in range(SCRATCH_BUFFER_COUNT)
            ]
            for r in self.rows
        ]
        self.ids_np = [[slot.numpy() for slot in group] for group in self.ids_host]
        # Completion waits for D2H on a side thread so the stager can keep
        # submitting the next quant/D2H without blocking on synchronize().
        self._completion_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fp8-d2h-wait",
            initializer=self._bind_completion_device,
        )
        total_bytes = SCRATCH_BUFFER_COUNT * sum(buffer_bytes)
        per_group = " ".join(
            f"g{idx}:rows={r},unit={u},bytes={nbytes}"
            for idx, (r, u, nbytes) in enumerate(
                zip(self.rows, self.unit_bytes, buffer_bytes)
            )
        )
        logger.info(
            "Host wire staging pipeline: mode=direct, buffers=%d, "
            "gpu_scratch_bytes=%d, per_group=[%s]",
            SCRATCH_BUFFER_COUNT,
            total_bytes,
            per_group,
        )
        if TRACE:
            trace.bind_device(self._device_id)
        if PROF:
            prof.bind_device(self._device_id)
            kinds = Counter(plan.quant_kind[name] for name in plan.layer_to_group)
            prof.set_context(
                f"mode=direct rows={list(self.rows)} "
                f"unit_bytes={list(self.unit_bytes)} "
                f"scratch_bytes={total_bytes} "
                f"buffers={SCRATCH_BUFFER_COUNT} "
                f"groups={len(plan.group_layers)} "
                f"layers={dict(sorted(kinds.items()))}"
            )

    @staticmethod
    def _per_group_events(
        num_groups: int, *, timing: bool = False
    ) -> list[list[torch.cuda.Event]]:
        return [
            [
                torch.cuda.Event(enable_timing=timing)
                for _ in range(SCRATCH_BUFFER_COUNT)
            ]
            for _ in range(num_groups)
        ]

    def _bind_completion_device(self) -> None:
        current_platform.set_device(self._device_id)

    def shutdown(self) -> None:
        self._completion_pool.shutdown(wait=True)

    def layout_for(self, layer_names: tuple[str, ...]) -> StageLayout:
        layout = self._layouts.get(layer_names)
        if layout is None:
            layout = build_stage_layout(self.plan, layer_names)
            have = self.unit_bytes[layout.group_idx]
            if layout.unit_stride > have:
                raise ValueError(
                    "Host wire scratch row is too small for write batch "
                    f"{layer_names} (group {layout.group_idx}): need "
                    f"{layout.unit_stride}, have {have}"
                )
            self._layouts[layer_names] = layout
        return layout

    def submit(
        self,
        layer_names: tuple[str, ...],
        block_ids: list[int],
        first_page: int,
        num_valid_tokens: int | None,
        ready_event,
    ) -> list[Future]:
        layout = self.layout_for(layer_names)
        rows = self.rows[layout.group_idx]
        futures: list[Future] = []
        for offset in range(0, len(block_ids), rows):
            futures.append(
                self._stage_chunk(
                    layout,
                    block_ids[offset : offset + rows],
                    first_page + offset,
                    num_valid_tokens,
                    ready_event if offset == 0 else None,
                )
            )
        return futures

    def _stage_chunk(
        self,
        layout: StageLayout,
        block_ids: list[int],
        first_page: int,
        num_valid_tokens: int | None,
        ready_event,
    ) -> Future:
        g = layout.group_idx
        counter = self._counters[g]
        self._counters[g] = counter + 1
        i = counter % SCRATCH_BUFFER_COUNT
        scratch = self.scratch[g][i]
        num_rows = len(block_ids)
        device = scratch.device
        d2h_bytes = num_rows * layout.unit_stride
        if PROF:
            prof.add("submits", 1)
            prof.add("d2h_bytes", d2h_bytes)
        seq = 0
        if TRACE:
            seq = trace.next_seq()
            trace_begin = trace.now_ms()
            tr_blockids = tr_zero = tr_d2h_submit = 0.0

        scratch_rows = torch.as_strided(
            scratch, (num_rows, layout.unit_stride), (layout.unit_stride, 1)
        )
        if TIMED:
            t0 = time.perf_counter()
        if counter >= SCRATCH_BUFFER_COUNT:
            self.ev_ids[g][i].synchronize()
        ids_np = self.ids_np[g][i][:num_rows]
        ids_np[:] = block_ids
        with torch.cuda.stream(self.quant_stream):
            source_ids = torch.empty(num_rows, dtype=torch.int64, device=device)
            source_ids.copy_(self.ids_host[g][i][:num_rows], non_blocking=True)
            self.ev_ids[g][i].record(self.quant_stream)
            if TIMED:
                elapsed = time.perf_counter() - t0
                if PROF:
                    prof.record("cpu.02.blockids_h2d", elapsed)
                tr_blockids = elapsed * 1000.0
            if counter >= SCRATCH_BUFFER_COUNT:
                self.quant_stream.wait_event(self.ev_copy[g][i])
            if ready_event is not None:
                self.quant_stream.wait_event(ready_event)
            if PROF:
                self.prof_events["quant_start"][g][i].record(self.quant_stream)
            if TIMED:
                t0 = time.perf_counter()
            scratch_rows.zero_()
            if TIMED:
                elapsed = time.perf_counter() - t0
                if PROF:
                    prof.record("cpu.04.zero_scratch", elapsed)
                tr_zero = elapsed * 1000.0
            for layer_name, layer_offset in zip(layout.layers, layout.layer_offsets):
                if TIMED:
                    t0 = time.perf_counter()
                quantize_layer(
                    self.plan,
                    layer_name,
                    self.compute_caches[layer_name],
                    torch.as_strided(
                        scratch,
                        (num_rows, self.plan.page_bytes[layer_name]),
                        (layout.unit_stride, 1),
                        storage_offset=layer_offset,
                    ),
                    source_ids,
                    first_page,
                    num_valid_tokens,
                    source_scales=self.compute_scales.get(layer_name),
                )
                if TIMED:
                    elapsed = time.perf_counter() - t0
                    kind = self.plan.quant_kind[layer_name]
                    if PROF:
                        prof.record(f"cpu.05.quant.{kind}", elapsed)
                    if TRACE:
                        trace.emit(
                            ("Q", seq, layer_name, kind, round(elapsed * 1000.0, 3))
                        )
            self.ev_quant[g][i].record(self.quant_stream)
            if PROF:
                self.prof_events["quant_end"][g][i].record(self.quant_stream)

        if TIMED:
            t0 = time.perf_counter()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.ev_quant[g][i])
            if PROF:
                self.prof_events["copy_start"][g][i].record(self.copy_stream)
            self._scatter_to_pool(scratch_rows, layout, block_ids)
            self.ev_copy[g][i].record(self.copy_stream)
            if PROF:
                self.prof_events["copy_end"][g][i].record(self.copy_stream)
        if TIMED:
            elapsed = time.perf_counter() - t0
            if PROF:
                prof.record("cpu.06.d2h_submit", elapsed)
            tr_d2h_submit = elapsed * 1000.0
        if TRACE:
            req, chunk = trace.task()
            trace.emit(
                (
                    "S",
                    seq,
                    req,
                    chunk,
                    g,
                    "requant" if num_valid_tokens is None else "fresh",
                    len(layout.layers),
                    num_rows,
                    first_page,
                    d2h_bytes,
                    round(trace_begin, 3),
                    round(tr_blockids, 3),
                    round(tr_zero, 3),
                    round(tr_d2h_submit, 3),
                )
            )

        return self._completion_pool.submit(self._finish_direct, g, i, seq)

    def _record_device_times(
        self, group_idx: int, slot: int
    ) -> tuple[float, float]:
        events = self.prof_events
        out = [-1.0, -1.0]
        for idx, (key, first, last) in enumerate(
            (
                ("gpu.quant_kernels", "quant_start", "quant_end"),
                ("gpu.d2h_copy", "copy_start", "copy_end"),
            )
        ):
            try:
                elapsed = (
                    events[first][group_idx][slot].elapsed_time(
                        events[last][group_idx][slot]
                    )
                    / 1000.0
                )
            except Exception:
                continue
            if 0.0 <= elapsed < 60.0:
                prof.record(key, elapsed)
                out[idx] = elapsed * 1000.0
        return out[0], out[1]

    def _scatter_to_pool(
        self, scratch_rows: torch.Tensor, layout: StageLayout, block_ids: list[int]
    ) -> None:
        pool_rows = self.pool.rows
        spans = zip(
            layout.dst_offsets.tolist(),
            layout.src_offsets.tolist(),
            layout.lengths.tolist(),
        )
        for dst, src, length in spans:
            for dense_id, block_id in enumerate(block_ids):
                pool_rows[block_id, dst : dst + length].copy_(
                    scratch_rows[dense_id, src : src + length], non_blocking=True
                )

    def _finish_direct(self, g: int, i: int, seq: int = 0) -> None:
        tr_wait_ev = gpu_quant = gpu_d2h = -1.0
        if TIMED:
            t0 = time.perf_counter()
        self.ev_copy[g][i].synchronize()
        if TIMED:
            elapsed = time.perf_counter() - t0
            if PROF:
                prof.record("cpu.07.wait_ev_copy", elapsed)
            tr_wait_ev = elapsed * 1000.0
        if PROF:
            gpu_quant, gpu_d2h = self._record_device_times(g, i)
        if TRACE:
            trace.emit(
                (
                    "H",
                    seq,
                    round(tr_wait_ev, 3),
                    round(gpu_quant, 3),
                    round(gpu_d2h, 3),
                )
            )
