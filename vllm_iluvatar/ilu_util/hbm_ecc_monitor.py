# SPDX-License-Identifier: Apache-2.0
"""Worker-scoped HBM ECC monitoring through the CoreX ixML interface."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from vllm_iluvatar.ilu_util.onchip_check import WorkerHealthCheckError

logger = logging.getLogger(__name__)

_XID_CP_HUNG = 38


@dataclass(frozen=True)
class EccCounts:
    corrected: int
    uncorrected: int


class EccBackend(Protocol):
    pci_bdf: str
    uuid: str

    def initialize(self, physical_gpu: int) -> None: ...

    def query(self) -> EccCounts: ...

    def shutdown(self) -> None: ...


class NvmlEccBackend:
    """Read volatile ECC totals from the NVML-compatible ixML library."""

    def __init__(self) -> None:
        self._nvml: Any | None = None
        self._handle: Any | None = None
        self.pci_bdf = ""
        self.uuid = ""

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    def initialize(self, physical_gpu: int) -> None:
        from vllm.third_party import pynvml as nvml

        nvml.nvmlInit()
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(physical_gpu)
            pci = nvml.nvmlDeviceGetPciInfo(handle)
            self.pci_bdf = self._as_text(pci.busId)
            self.uuid = self._as_text(nvml.nvmlDeviceGetUUID(handle))
        except Exception:
            nvml.nvmlShutdown()
            raise
        self._nvml = nvml
        self._handle = handle

    def query(self) -> EccCounts:
        if self._nvml is None or self._handle is None:
            raise RuntimeError("ECC backend is not initialized")
        nvml = self._nvml
        corrected = nvml.nvmlDeviceGetTotalEccErrors(
            self._handle,
            nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
            nvml.NVML_VOLATILE_ECC,
        )
        uncorrected = nvml.nvmlDeviceGetTotalEccErrors(
            self._handle,
            nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
            nvml.NVML_VOLATILE_ECC,
        )
        return EccCounts(int(corrected), int(uncorrected))

    def shutdown(self) -> None:
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        finally:
            self._nvml = None
            self._handle = None


class HbmEccMonitor:
    """Poll ECC counters and fail closed when monitoring is unreliable."""

    def __init__(
        self,
        *,
        backend_factory: Callable[[], EccBackend] = NvmlEccBackend,
        xid_reporter: Callable[[int, int, str], bool] | None = None,
        terminator: Callable[[], None] | None = None,
    ) -> None:
        self._backend_factory = backend_factory
        self._xid_reporter = xid_reporter or self._report_xid
        self._terminator = terminator or self._terminate_process_group
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._backend: EccBackend | None = None
        self._baseline: EccCounts | None = None
        self._fault_message = ""
        self._degraded_reported = False
        self._query_started_at: float | None = None
        self._last_cycle_at = 0.0
        self._consecutive_failures = 0
        self._initialized = False
        self._physical_gpu: int | None = None
        self._rank: int | None = None
        self._local_rank: int | None = None
        self._logical_device: int | None = None
        self._pci_bdf = ""
        self._uuid = ""
        self._poll_interval = 1.0
        self._heartbeat_timeout = 5.0
        self._max_query_failures = 3
        self._kmsg = False
        self._kill_process_group = False
        self._xid_report_timeout = 0.25

    @staticmethod
    def enabled() -> bool:
        import vllm.envs as envs

        return bool(envs.VLLM_ILU_HBM_ECC_CHECK)

    def _device_context(self) -> str:
        parts = []
        if self._rank is not None:
            parts.append(f"rank={self._rank}")
        if self._local_rank is not None:
            parts.append(f"local_rank={self._local_rank}")
        if self._logical_device is not None:
            parts.append(f"logical_device={self._logical_device}")
        if self._physical_gpu is not None:
            parts.append(f"physical_gpu={self._physical_gpu}")
        if self._pci_bdf:
            parts.append(f"PCI={self._pci_bdf}")
        if self._uuid:
            parts.append(f"UUID={self._uuid}")
        return " ".join(parts) or "device=unknown"

    @staticmethod
    def _report_xid(physical_gpu: int, xid: int, message: str) -> bool:
        from vllm_iluvatar import _onchip_check

        return bool(
            _onchip_check.report_xid(
                physical_gpu=physical_gpu,
                xid=xid,
                message=message,
            )
        )

    @staticmethod
    def _terminate_process_group() -> None:
        os.killpg(os.getpgrp(), signal.SIGTERM)

    def start(
        self,
        *,
        rank: int | None = None,
        local_rank: int | None = None,
        logical_device: int | None = None,
        physical_gpu: int,
    ) -> None:
        if not self.enabled():
            return
        if self._initialized:
            return

        import vllm.envs as envs

        self._physical_gpu = physical_gpu
        self._rank = rank
        self._local_rank = local_rank
        self._logical_device = logical_device
        self._poll_interval = envs.VLLM_ILU_HBM_ECC_POLL_INTERVAL_MS / 1000.0
        self._heartbeat_timeout = envs.VLLM_ILU_HBM_ECC_HEARTBEAT_TIMEOUT_MS / 1000.0
        self._max_query_failures = envs.VLLM_ILU_HBM_ECC_MAX_QUERY_FAILURES
        self._kmsg = bool(envs.VLLM_ILU_HBM_ECC_KMSG)
        self._kill_process_group = bool(envs.VLLM_ILU_HBM_ECC_KILL_PROCESS_GROUP)
        if self._heartbeat_timeout <= self._poll_interval:
            raise WorkerHealthCheckError(
                "VLLM_ILU_HBM_ECC_HEARTBEAT_TIMEOUT_MS must be greater than "
                "VLLM_ILU_HBM_ECC_POLL_INTERVAL_MS"
            )

        self._stop_event.clear()
        self._ready_event.clear()
        now = time.monotonic()
        self._last_cycle_at = now
        self._query_started_at = now
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"hbm-ecc-poll-{physical_gpu}",
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name=f"hbm-ecc-watchdog-{physical_gpu}",
        )
        self._initialized = True
        self._poll_thread.start()
        self._watchdog_thread.start()

        wait_timeout = self._heartbeat_timeout + self._poll_interval
        if not self._ready_event.wait(wait_timeout):
            self._degraded(
                f"HBM ECC monitor startup is still pending after "
                f"{wait_timeout:.3f}s ({self._device_context()}); "
                "serving will continue"
            )
        self.check()

    def _poll_loop(self) -> None:
        backend = self._backend_factory()
        self._backend = backend
        try:
            if self._physical_gpu is None:
                raise RuntimeError("physical GPU is not configured")
            backend.initialize(self._physical_gpu)
            self._pci_bdf = backend.pci_bdf
            self._uuid = backend.uuid
            counts = None
            while counts is None and not self._stop_event.is_set():
                counts = self._query(backend)
                if counts is None:
                    if self._consecutive_failures >= self._max_query_failures:
                        self._ready_event.set()
                    self._stop_event.wait(self._poll_interval)
            if counts is None:
                self._ready_event.set()
                return
            self._baseline = counts
            if counts.uncorrected:
                self._fatal(
                    "Startup HBM ECC check found existing uncorrected errors "
                    f"({self._device_context()}): DBE={counts.uncorrected}",
                    terminate=False,
                )
                self._ready_event.set()
                return
            if counts.corrected:
                logger.warning(
                    "Startup HBM ECC check found corrected errors (%s): SBE=%d",
                    self._device_context(),
                    counts.corrected,
                )
            logger.info(
                "HBM ECC monitor ready (%s): SBE=%d DBE=%d",
                self._device_context(),
                counts.corrected,
                counts.uncorrected,
            )
            self._ready_event.set()

            while not self._stop_event.wait(self._poll_interval):
                counts = self._query(backend)
                if counts is None:
                    if self.faulted:
                        return
                    continue
                self._process_counts(counts)
                if self.faulted:
                    return
        except Exception as exc:
            self._degraded(
                f"HBM ECC monitor failed ({self._device_context()}): {exc}; "
                "serving will continue"
            )
            self._ready_event.set()
        finally:
            if not self._ready_event.is_set():
                self._ready_event.set()
            try:
                backend.shutdown()
            except Exception:
                logger.exception(
                    "Failed to shut down HBM ECC backend (%s)",
                    self._device_context(),
                )

    def _query(self, backend: EccBackend) -> EccCounts | None:
        with self._lock:
            self._query_started_at = time.monotonic()
        try:
            counts = backend.query()
        except Exception as exc:
            with self._lock:
                self._query_started_at = None
                self._last_cycle_at = time.monotonic()
                self._consecutive_failures += 1
                failures = self._consecutive_failures
            logger.error(
                "HBM ECC query failed (%s, count=%d/%d): %s",
                self._device_context(),
                failures,
                self._max_query_failures,
                exc,
            )
            if failures >= self._max_query_failures:
                self._degraded(
                    "HBM ECC monitoring became unavailable "
                    f"({self._device_context()}) after {failures} failures: "
                    f"{exc}; serving will continue",
                )
            return None
        with self._lock:
            self._query_started_at = None
            self._last_cycle_at = time.monotonic()
            self._consecutive_failures = 0
            recovered = self._degraded_reported
            self._degraded_reported = False
        if recovered:
            logger.warning(
                "HBM ECC monitoring recovered (%s)",
                self._device_context(),
            )
        return counts

    def _process_counts(self, counts: EccCounts) -> None:
        previous = self._baseline
        if previous is None:
            self._baseline = counts
            return
        if (
            counts.corrected < previous.corrected
            or counts.uncorrected < previous.uncorrected
        ):
            self._degraded(
                "HBM ECC counters moved backwards "
                f"({self._device_context()}): previous={previous} "
                f"current={counts}; resetting the baseline and continuing",
            )
            self._baseline = counts
            return

        corrected_delta = counts.corrected - previous.corrected
        uncorrected_delta = counts.uncorrected - previous.uncorrected
        self._baseline = counts
        if corrected_delta:
            logger.warning(
                "HBM ECC corrected error detected (%s): delta=%d total=%d",
                self._device_context(),
                corrected_delta,
                counts.corrected,
            )
        if uncorrected_delta:
            self._fatal(
                "HBM ECC uncorrected error detected "
                f"({self._device_context()}): delta={uncorrected_delta} "
                f"total={counts.uncorrected}",
                terminate=True,
            )

    @property
    def faulted(self) -> bool:
        with self._lock:
            return bool(self._fault_message)

    def _watchdog_loop(self) -> None:
        interval = min(self._poll_interval, self._heartbeat_timeout / 4)
        while not self._stop_event.wait(interval):
            if not self._watchdog_check(time.monotonic()):
                return

    def _watchdog_check(self, now: float) -> bool:
        with self._lock:
            query_started_at = self._query_started_at
            last_cycle_at = self._last_cycle_at
        if query_started_at is not None:
            stale_for = now - query_started_at
            reason = "query"
        else:
            stale_for = now - last_cycle_at
            reason = "heartbeat"
        limit = (
            self._heartbeat_timeout
            if reason == "query"
            else self._heartbeat_timeout + self._poll_interval
        )
        if stale_for > limit:
            self._degraded(
                f"HBM ECC monitor {reason} stalled "
                f"({self._device_context()}) for {stale_for:.3f}s; "
                "serving will continue",
            )
            return False
        poll_thread = self._poll_thread
        if (
            self._ready_event.is_set()
            and poll_thread is not None
            and not poll_thread.is_alive()
            and not self.faulted
        ):
            self._degraded(
                f"HBM ECC polling thread exited ({self._device_context()}); "
                "serving will continue"
            )
            return False
        return True

    def _degraded(self, message: str) -> None:
        with self._lock:
            if self._degraded_reported:
                return
            self._degraded_reported = True
        logger.critical(message)

    def _fatal(self, message: str, *, terminate: bool) -> None:
        with self._lock:
            if self._fault_message:
                return
            self._fault_message = message
        logger.critical(message)
        if self._kmsg and self._physical_gpu is not None:
            physical_gpu = self._physical_gpu
            context = f"({self._device_context()})"
            detail = (
                message.replace(context, "")
                .replace("  ", " ")
                .replace(" :", ":")
                .strip(" :")
            )
            xid_message = f"hbm_ecc_monitor gpu={physical_gpu} {detail}"
            result: list[bool] = []
            report_done = threading.Event()

            def report_xid() -> None:
                try:
                    result.append(
                        self._xid_reporter(
                            physical_gpu,
                            _XID_CP_HUNG,
                            xid_message[:119],
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to report HBM ECC fault through /dev/itrctl (%s)",
                        self._device_context(),
                    )
                finally:
                    report_done.set()

            threading.Thread(
                target=report_xid,
                daemon=True,
                name=f"hbm-ecc-xid-{physical_gpu}",
            ).start()
            if not report_done.wait(self._xid_report_timeout):
                logger.error(
                    "Timed out reporting HBM ECC fault through /dev/itrctl (%s)",
                    self._device_context(),
                )
            elif result and not result[0]:
                logger.error(
                    "Failed to report HBM ECC fault through /dev/itrctl (%s)",
                    self._device_context(),
                )
        if terminate and self._kill_process_group:
            self._terminator()

    def check(self) -> None:
        with self._lock:
            message = self._fault_message
        if message:
            raise WorkerHealthCheckError(message)

    def free(self) -> None:
        if not self._initialized:
            return
        self._stop_event.set()
        timeout = min(self._heartbeat_timeout, 1.0)
        current = threading.current_thread()
        for thread in (self._poll_thread, self._watchdog_thread):
            if thread is not None and thread is not current:
                thread.join(timeout)
        self._initialized = False
        self._poll_thread = None
        self._watchdog_thread = None
        self._backend = None
