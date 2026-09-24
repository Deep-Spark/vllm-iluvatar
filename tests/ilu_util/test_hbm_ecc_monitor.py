# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import threading
import time

import pytest

from vllm_iluvatar.ilu_util.hbm_ecc_monitor import (
    EccCounts,
    HbmEccMonitor,
)
from vllm_iluvatar.ilu_util.onchip_check import WorkerHealthCheckError


class _FakeBackend:
    def __init__(self, results):
        self._results = iter(results)
        self.pci_bdf = "00000000:A1:00.0"
        self.uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        self.initialized_with = None
        self.shutdown_calls = 0

    def initialize(self, physical_gpu):
        self.initialized_with = physical_gpu

    def query(self):
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def configured_env(monkeypatch):
    import vllm.envs as envs

    values = {
        "VLLM_ILU_HBM_ECC_CHECK": True,
        "VLLM_ILU_HBM_ECC_POLL_INTERVAL_MS": 1000.0,
        "VLLM_ILU_HBM_ECC_HEARTBEAT_TIMEOUT_MS": 5000.0,
        "VLLM_ILU_HBM_ECC_MAX_QUERY_FAILURES": 3,
        "VLLM_ILU_HBM_ECC_KMSG": False,
        "VLLM_ILU_HBM_ECC_KILL_PROCESS_GROUP": False,
    }
    for name, value in values.items():
        monkeypatch.setattr(envs, name, value, raising=False)
    return envs


def _monitor(*, reporter=None, terminator=None):
    return HbmEccMonitor(
        xid_reporter=reporter or (lambda *_args: True),
        terminator=terminator or (lambda: None),
    )


def test_disabled_start_does_not_create_backend(monkeypatch):
    created = []
    monitor = HbmEccMonitor(backend_factory=lambda: created.append(True))
    monkeypatch.setattr(HbmEccMonitor, "enabled", staticmethod(lambda: False))

    monitor.start(physical_gpu=3)

    assert created == []


def test_start_binds_physical_gpu(configured_env):
    backend = _FakeBackend([EccCounts(0, 0)])
    monitor = HbmEccMonitor(backend_factory=lambda: backend)

    monitor.start(
        rank=5,
        local_rank=1,
        logical_device=1,
        physical_gpu=17,
    )
    try:
        assert backend.initialized_with == 17
        assert monitor._device_context() == (
            "rank=5 local_rank=1 logical_device=1 physical_gpu=17 "
            "PCI=00000000:A1:00.0 "
            "UUID=GPU-01234567-89ab-cdef-0123-456789abcdef"
        )
    finally:
        monitor.free()


def test_existing_dbe_rejects_startup(configured_env):
    backend = _FakeBackend([EccCounts(0, 1)])
    monitor = HbmEccMonitor(backend_factory=lambda: backend)

    with pytest.raises(WorkerHealthCheckError, match="existing uncorrected"):
        monitor.start(physical_gpu=17)

    monitor.free()


def test_sbe_warns_without_failing(caplog):
    monitor = _monitor()
    monitor._physical_gpu = 17
    monitor._baseline = EccCounts(2, 0)

    with caplog.at_level(logging.WARNING):
        monitor._process_counts(EccCounts(5, 0))

    monitor.check()
    assert "delta=3 total=5" in caplog.text


def test_dbe_reports_xid_and_terminates():
    reports = []
    terminations = []
    monitor = _monitor(
        reporter=lambda *args: reports.append(args) or True,
        terminator=lambda: terminations.append(True),
    )
    monitor._physical_gpu = 17
    monitor._baseline = EccCounts(0, 0)
    monitor._kmsg = True
    monitor._kill_process_group = True

    monitor._process_counts(EccCounts(0, 2))

    with pytest.raises(WorkerHealthCheckError, match="uncorrected"):
        monitor.check()
    assert reports == [
        (
            17,
            38,
            "hbm_ecc_monitor gpu=17 HBM ECC uncorrected error detected: "
            "delta=2 total=2",
        )
    ]
    assert terminations == [True]


def test_repeated_query_failures_alert_without_stopping(caplog):
    terminations = []
    monitor = _monitor(terminator=lambda: terminations.append(True))
    monitor._physical_gpu = 17
    monitor._baseline = EccCounts(0, 0)
    monitor._max_query_failures = 2
    monitor._kill_process_group = True
    backend = _FakeBackend([RuntimeError("read failed"), RuntimeError("read failed")])

    with caplog.at_level(logging.CRITICAL):
        assert monitor._query(backend) is None
        assert monitor._query(backend) is None

    monitor.check()
    assert "monitoring became unavailable" in caplog.text
    assert "serving will continue" in caplog.text
    assert terminations == []


def test_xid_timeout_does_not_block_termination():
    release_reporter = threading.Event()
    terminations = []
    monitor = _monitor(
        reporter=lambda *_args: release_reporter.wait() or True,
        terminator=lambda: terminations.append(True),
    )
    monitor._physical_gpu = 17
    monitor._baseline = EccCounts(0, 0)
    monitor._kmsg = True
    monitor._kill_process_group = True
    monitor._xid_report_timeout = 0.01

    monitor._process_counts(EccCounts(0, 1))

    assert terminations == [True]
    release_reporter.set()


def test_counter_rollback_resets_baseline_without_stopping(caplog):
    terminations = []
    monitor = _monitor(terminator=lambda: terminations.append(True))
    monitor._baseline = EccCounts(5, 0)
    monitor._kill_process_group = True

    with caplog.at_level(logging.CRITICAL):
        monitor._process_counts(EccCounts(4, 0))

    monitor.check()
    assert monitor._baseline == EccCounts(4, 0)
    assert "resetting the baseline and continuing" in caplog.text
    assert terminations == []


def test_stalled_query_alerts_without_stopping(caplog):
    terminations = []
    monitor = _monitor(terminator=lambda: terminations.append(True))
    monitor._physical_gpu = 17
    monitor._baseline = EccCounts(0, 0)
    monitor._heartbeat_timeout = 5.0
    monitor._query_started_at = time.monotonic() - 6.0
    monitor._kill_process_group = True

    with caplog.at_level(logging.CRITICAL):
        keep_running = monitor._watchdog_check(time.monotonic())

    assert not keep_running
    monitor.check()
    assert "query stalled" in caplog.text
    assert "serving will continue" in caplog.text
    assert terminations == []


def test_free_is_idempotent():
    monitor = _monitor()

    monitor.free()
    monitor.free()
