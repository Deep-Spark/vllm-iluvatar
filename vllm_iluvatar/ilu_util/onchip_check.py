# SPDX-License-Identifier: Apache-2.0
"""Worker-scoped policy wrapper for the native on-chip storage checker."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class OnchipStatus(IntEnum):
    OK = 0
    FAULT = 1
    INCOMPLETE = 2
    NOT_READY = 3
    CUDA_ERROR = 4


@dataclass(frozen=True)
class OnchipResult:
    status: OnchipStatus
    errors: tuple[int, int, int, int]
    first_level: int
    first_level_name: str
    first_block: int
    first_thread: int
    first_word: int
    first_expected: int
    first_got: int
    first_phase: int
    coverage_incomplete: bool
    error: str

    @classmethod
    def from_native(cls, raw: dict[str, Any]) -> OnchipResult:
        errors = tuple(int(value) for value in raw.get("errors", (0, 0, 0, 0)))
        if len(errors) != 4:
            raise RuntimeError(f"Invalid worker health result vector: {errors!r}")
        return cls(
            status=OnchipStatus(int(raw["status"])),
            errors=errors,  # type: ignore[arg-type]
            first_level=int(raw.get("first_level", 0)),
            first_level_name=str(raw.get("first_level_name", "?")),
            first_block=int(raw.get("first_block", 0)),
            first_thread=int(raw.get("first_thread", 0)),
            first_word=int(raw.get("first_word", 0)),
            first_expected=int(raw.get("first_expected", 0)),
            first_got=int(raw.get("first_got", 0)),
            first_phase=int(raw.get("first_phase", 0)),
            coverage_incomplete=bool(raw.get("coverage_incomplete", False)),
            error=str(raw.get("error", "")),
        )


class WorkerHealthCheckError(RuntimeError):
    pass


class OnchipChecker:
    """Own one native checker for one GPU Worker process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._native: Any | None = None
        self._initialized = False
        self._device_index: int | None = None
        self._rank: int | None = None
        self._local_rank: int | None = None
        self._logical_device: int | None = None
        self._physical_gpu: int | None = None
        self._pci_bdf = ""
        self._uuid = ""
        self._seed = (os.getpid() ^ time.monotonic_ns()) & 0xFFFFFFFF
        self._incomplete_count = 0

    @staticmethod
    def enabled() -> bool:
        import vllm.envs as envs

        return bool(envs.VLLM_ILU_ONCHIP_CHECK)

    def _load_native(self) -> Any:
        if self._native is None:
            try:
                from vllm_iluvatar import _onchip_check
            except ImportError as exc:
                raise WorkerHealthCheckError(
                    "Worker health checking is enabled but "
                    "vllm_iluvatar._onchip_check could not be imported. "
                    "Rebuild the plugin and verify the CoreX runtime libraries."
                ) from exc
            self._native = _onchip_check
        return self._native

    @staticmethod
    def _resolve_device_index(device: Any) -> int:
        if isinstance(device, int):
            return device
        index = getattr(device, "index", None)
        if index is None:
            raise WorkerHealthCheckError(
                f"Cannot determine GPU index from {device!r}"
            )
        return int(index)

    @staticmethod
    def _format_result(result: OnchipResult) -> str:
        detail = (
            f"status={result.status.name}, errors="
            f"SLB/REG/L1/L2={result.errors}, first={result.first_level_name} "
            f"block={result.first_block} thread={result.first_thread} "
            f"word={result.first_word} expected=0x{result.first_expected:08x} "
            f"got=0x{result.first_got:08x}"
        )
        if result.error:
            detail += f", cuda_error={result.error}"
        return detail

    def _next_seed(self) -> int:
        self._seed = (self._seed + 0x9E3779B9) & 0xFFFFFFFF
        return self._seed

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
        if self._device_index is not None:
            parts.append(f"device=cuda:{self._device_index}")
        if self._pci_bdf:
            parts.append(f"PCI={self._pci_bdf}")
        if self._uuid:
            parts.append(f"UUID={self._uuid}")
        return " ".join(parts) or "device=unknown"

    def _run(self) -> OnchipResult:
        import torch
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        stream_handle = int(torch.cuda.current_stream(self._device_index).cuda_stream)
        with self._lock, gpu_sync_allowed():
            raw = self._load_native().run(
                seed=self._next_seed(), stream_handle=stream_handle
            )
        return OnchipResult.from_native(raw)

    def start(
        self,
        device: Any,
        *,
        rank: int | None = None,
        local_rank: int | None = None,
        logical_device: int | None = None,
        physical_gpu: int | None = None,
    ) -> None:
        if not self.enabled():
            return

        import vllm.envs as envs

        device_index = self._resolve_device_index(device)
        self._device_index = device_index
        self._rank = rank
        self._local_rank = local_rank
        self._logical_device = logical_device
        self._physical_gpu = physical_gpu
        level_mask = int(envs.VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK)
        if not 0 <= level_mask <= 0xF:
            raise WorkerHealthCheckError(
                f"VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK must be between 0 and 15, "
                f"got {level_mask}"
            )
        with self._lock:
            raw = self._load_native().init(
                device=device_index,
                level_mask=level_mask,
                kmsg=envs.VLLM_ILU_ONCHIP_CHECK_KMSG,
            )
            init_result = OnchipResult.from_native(raw)
            if init_result.status is not OnchipStatus.OK:
                raise WorkerHealthCheckError(
                    f"Failed to initialize worker health checker "
                    f"({self._device_context()}): "
                    + self._format_result(init_result)
                )
            self._initialized = True
            config = self._native.config()
            self._pci_bdf = str(config.get("pci_bdf", ""))
            self._uuid = str(config.get("uuid", ""))
            kmsg_ready = (
                not envs.VLLM_ILU_ONCHIP_CHECK_KMSG or self._native.kmsg_ready()
            )

        if not kmsg_ready:
            logger.warning(
                "Worker health XID reporting was requested for %s, but "
                "/dev/itrctl is unavailable",
                self._device_context(),
            )

        retries = envs.VLLM_ILU_ONCHIP_CHECK_STARTUP_RETRIES
        delay = envs.VLLM_ILU_ONCHIP_CHECK_RETRY_DELAY_MS / 1000.0
        result: OnchipResult | None = None
        for attempt in range(retries + 1):
            result = self._run()
            if result.status is not OnchipStatus.INCOMPLETE:
                break
            if attempt < retries:
                time.sleep(delay)

        assert result is not None
        if result.status is not OnchipStatus.OK:
            message = (
                f"Startup worker health check failed ({self._device_context()}) "
                f"after {retries + 1} attempt(s): {self._format_result(result)}"
            )
            self.free()
            raise WorkerHealthCheckError(message)
        logger.info(
            "Worker health checker ready (%s): CU=%s, "
            "SLB=%s bytes, REG=%s/%s bytes, L1=%s bytes, L2=%s bytes",
            self._device_context(),
            config.get("cu_count"),
            config.get("slb_bytes_total"),
            config.get("reg_bytes_total"),
            config.get("reg_bytes_capacity"),
            config.get("l1_bytes_total"),
            config.get("l2_bytes_total"),
        )

    def before_batch(self) -> None:
        if not self.enabled():
            return
        if not self._initialized:
            raise WorkerHealthCheckError(
                "Worker health checker is not initialized"
            )

        import vllm.envs as envs

        result = self._run()
        if result.status is OnchipStatus.OK:
            return
        if result.status is OnchipStatus.INCOMPLETE:
            self._incomplete_count += 1
            interval = envs.VLLM_ILU_ONCHIP_CHECK_INCOMPLETE_LOG_INTERVAL
            if self._incomplete_count == 1 or self._incomplete_count % interval == 0:
                logger.warning(
                    "Worker health check was inconclusive before batch "
                    "(%s, count=%d); continuing this batch: %s",
                    self._device_context(),
                    self._incomplete_count,
                    self._format_result(result),
                )
            return
        raise WorkerHealthCheckError(
            f"Worker health check failed before batch "
            f"({self._device_context()}): "
            f"{self._format_result(result)}"
        )

    def check_shutdown(self) -> None:
        if not self.enabled() or not self._initialized:
            return
        result = self._run()
        if result.status is not OnchipStatus.OK:
            logger.error(
                "Worker health shutdown check (%s): %s",
                self._device_context(),
                self._format_result(result),
            )

    def free(self) -> None:
        if self._native is None or not self._initialized:
            return
        with self._lock:
            self._native.free()
            self._initialized = False
            self._device_index = None
            self._rank = None
            self._local_rank = None
            self._logical_device = None
            self._physical_gpu = None
            self._pci_bdf = ""
            self._uuid = ""
