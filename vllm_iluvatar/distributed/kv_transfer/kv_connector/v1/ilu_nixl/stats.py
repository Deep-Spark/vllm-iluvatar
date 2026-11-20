# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end observability stats & Prometheus metrics for IluNixl pull."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from vllm.v1.metrics.utils import create_metric_per_engine

_EXTRA_STAT_KEYS = (
    "handshake_duration",
    "io_queue_wait_duration",
    "wake_duration",
    "lease_hold_duration",
)


@dataclass
class IluNixlKVConnectorStats(NixlKVConnectorStats):
    """Extends ``NixlKVConnectorStats`` with 4 architectural attribution metrics."""

    def __post_init__(self):
        super().__post_init__()
        for key in _EXTRA_STAT_KEYS:
            self.data.setdefault(key, [])

    def reset(self):
        super().reset()
        for key in _EXTRA_STAT_KEYS:
            self.data[key] = []

    def record_handshake_duration(self, seconds: float) -> None:
        self.data.setdefault("handshake_duration", []).append(max(0.0, float(seconds)))

    def record_io_queue_wait(self, seconds: float) -> None:
        self.data.setdefault("io_queue_wait_duration", []).append(
            max(0.0, float(seconds))
        )

    def record_wake_duration(self, seconds: float) -> None:
        self.data.setdefault("wake_duration", []).append(max(0.0, float(seconds)))

    def record_lease_hold_duration(self, seconds: float) -> None:
        self.data.setdefault("lease_hold_duration", []).append(
            max(0.0, float(seconds))
        )

    def clone_and_reset(self) -> "IluNixlKVConnectorStats":
        old = copy.copy(self)
        old.data = {k: list(v) for k, v in self.data.items()}
        self.reset()
        return old

    def is_empty(self) -> bool:
        if not super().is_empty():
            return False
        return all(len(self.data.get(k, ())) == 0 for k in _EXTRA_STAT_KEYS)

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data.setdefault(k, [])
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        summary = super().reduce()
        hs = self.data.get("handshake_duration", ())
        if hs:
            arr = np.asarray(hs)
            summary["Avg handshake time (ms)"] = round(float(arr.mean() * 1e3), 3)
            summary["P90 handshake time (ms)"] = round(
                float(np.percentile(arr, 90) * 1e3), 3
            )
        qwait = self.data.get("io_queue_wait_duration", ())
        if qwait:
            arr = np.asarray(qwait)
            summary["Avg IO queue wait (ms)"] = round(float(arr.mean() * 1e3), 3)
            summary["P99 IO queue wait (ms)"] = round(
                float(np.percentile(arr, 99) * 1e3), 3
            )
        wake = self.data.get("wake_duration", ())
        if wake:
            arr = np.asarray(wake)
            summary["Avg wake time (ms)"] = round(float(arr.mean() * 1e3), 3)
            summary["P99 wake time (ms)"] = round(
                float(np.percentile(arr, 99) * 1e3), 3
            )
        lease = self.data.get("lease_hold_duration", ())
        if lease:
            arr = np.asarray(lease)
            summary["Avg lease hold time (ms)"] = round(float(arr.mean() * 1e3), 3)
            summary["P90 lease hold time (ms)"] = round(
                float(np.percentile(arr, 90) * 1e3), 3
            )
        return summary


class IluNixlPromMetrics(NixlPromMetrics):
    """Prometheus histograms for IluNixl architectural attribution metrics."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        sub_ms_buckets = [
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.5,
        ]
        sec_buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            15.0,
            30.0,
        ]
        self.ilu_histogram_handshake = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ilu_nixl_handshake_seconds",
                documentation="Cold-start ZMQ + add_remote_agent + prep_xfer_dlist duration.",
                buckets=sec_buckets,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.ilu_histogram_io_queue_wait = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ilu_nixl_io_queue_wait_seconds",
                documentation="Queue wait from start_load_kv enqueue to IO thread dequeue.",
                buckets=sub_ms_buckets,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.ilu_histogram_wake = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ilu_nixl_wake_seconds",
                documentation="Completion outbox enqueue to main-thread drain latency (t_wake).",
                buckets=sub_ms_buckets,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )
        self.ilu_histogram_lease_hold = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ilu_nixl_lease_hold_seconds",
                documentation="Producer KV block lease hold duration until consumer release.",
                buckets=sec_buckets,
                labelnames=labelnames,
            ),
            self.per_engine_labelvalues,
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        super().observe(transfer_stats_data, engine_idx)
        for hist_map, key in (
            (self.ilu_histogram_handshake, "handshake_duration"),
            (self.ilu_histogram_io_queue_wait, "io_queue_wait_duration"),
            (self.ilu_histogram_wake, "wake_duration"),
            (self.ilu_histogram_lease_hold, "lease_hold_duration"),
        ):
            for val in transfer_stats_data.get(key, ()):
                hist_map[engine_idx].observe(val)
