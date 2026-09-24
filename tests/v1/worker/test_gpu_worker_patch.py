# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the IluWorker subclass (dummy-batch optimisation)."""

from __future__ import annotations

import logging
import sys
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def _stub_gpu_worker(monkeypatch):
    """Stub vllm.v1.worker.gpu_worker.Worker so IluWorker can be imported."""

    def make_package(name):
        package = ModuleType(name)
        package.__path__ = []
        return package

    vllm_mod = make_package("vllm")
    vllm_v1_mod = make_package("vllm.v1")
    worker_package = make_package("vllm.v1.worker")

    gpu_worker_mod = ModuleType("vllm.v1.worker.gpu_worker")

    class _FakeWorker:
        def compile_or_warm_up_model(self):
            self.base_calls.append("compile")
            return "compilation-times"

        def execute_model(self, scheduler_output):
            self.base_calls.append(("execute", scheduler_output))
            return "model-output"

        def shutdown(self) -> None:
            self.base_calls.append("shutdown")
            if error := getattr(self, "base_shutdown_error", None):
                raise error

        def execute_dummy_batch(self) -> None:
            raise NotImplementedError

    gpu_worker_mod.Worker = _FakeWorker

    gpu_model_runner_mod = ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_model_runner_mod.GPUModelRunner = type("GPUModelRunner", (), {})

    kv_cache_interface_mod = ModuleType("vllm.v1.kv_cache_interface")
    kv_cache_interface_mod.KVCacheConfig = type("KVCacheConfig", (), {})

    logger_mod = ModuleType("vllm.logger")
    logger_mod.init_logger = logging.getLogger

    platforms_mod = ModuleType("vllm.platforms")
    platforms_mod.current_platform = type(
        "FakePlatform",
        (),
        {
            "device_id_to_physical_device_id": staticmethod(
                lambda device_id: device_id + 24
            )
        },
    )()

    vllm_mod.v1 = vllm_v1_mod
    vllm_v1_mod.worker = worker_package
    worker_package.gpu_worker = gpu_worker_mod
    worker_package.gpu_model_runner = gpu_model_runner_mod

    modules = {
        "vllm": vllm_mod,
        "vllm.v1": vllm_v1_mod,
        "vllm.v1.worker": worker_package,
        "vllm.v1.worker.gpu_model_runner": gpu_model_runner_mod,
        "vllm.v1.worker.gpu_worker": gpu_worker_mod,
        "vllm.v1.kv_cache_interface": kv_cache_interface_mod,
        "vllm.logger": logger_mod,
        "vllm.platforms": platforms_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    # Invalidate any cached import of the plugin module so the stub is used.
    monkeypatch.delitem(
        sys.modules, "vllm_iluvatar.v1.worker.gpu_worker", raising=False
    )

    yield gpu_worker_mod


def _make_worker(model_runner):
    """Import IluWorker (against the stub) and construct an instance."""
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

    w = IluWorker.__new__(IluWorker)
    w.model_runner = model_runner
    w.base_calls = []
    return w


def test_execute_dummy_batch_passes_profile_seq_lens():
    """IluWorker.execute_dummy_batch must forward profile_seq_lens=1."""
    calls: list[dict] = []

    class FakeModelRunner:
        uniform_decode_query_len = 4

        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    worker = _make_worker(FakeModelRunner())
    worker.execute_dummy_batch()

    assert len(calls) == 1
    assert calls[0] == {
        "num_tokens": 4,
        "uniform_decode": True,
        "profile_seq_lens": 1,
    }


def test_execute_dummy_batch_uses_fallback_query_len():
    """When model_runner lacks uniform_decode_query_len, default to 1."""
    calls: list[dict] = []

    class BareModelRunner:
        def _dummy_run(self, num_tokens, **kwargs):
            calls.append({"num_tokens": num_tokens, **kwargs})

    worker = _make_worker(BareModelRunner())
    worker.execute_dummy_batch()

    assert len(calls) == 1
    assert calls[0] == {
        "num_tokens": 1,
        "uniform_decode": True,
        "profile_seq_lens": 1,
    }


def test_ilu_worker_is_subclass():
    """IluWorker must inherit from the community Worker."""
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

    # The stub Worker is the base in the test environment.
    assert issubclass(IluWorker, sys.modules["vllm.v1.worker.gpu_worker"].Worker)


class _FakeChecker:
    def __init__(self):
        self.calls = []

    def start(
        self,
        device,
        *,
        rank=None,
        local_rank=None,
        logical_device=None,
        physical_gpu=None,
    ):
        self.calls.append(
            (
                "start",
                device,
                rank,
                local_rank,
                logical_device,
                physical_gpu,
            )
        )

    def before_batch(self):
        self.calls.append("before_batch")

    def check_shutdown(self):
        self.calls.append("check_shutdown")

    def free(self):
        self.calls.append("free")


class _FakeEccMonitor:
    def __init__(self):
        self.calls = []

    def start(
        self,
        *,
        rank=None,
        local_rank=None,
        logical_device=None,
        physical_gpu=None,
    ):
        self.calls.append(("start", rank, local_rank, logical_device, physical_gpu))

    def check(self):
        self.calls.append("check")

    def free(self):
        self.calls.append("free")


def test_compile_starts_checker_before_parent():
    worker = _make_worker(object())
    checker = _FakeChecker()
    monitor = _FakeEccMonitor()
    started_before_parent = []

    def start(
        device, *, rank=None, local_rank=None, logical_device=None, physical_gpu=None
    ):
        started_before_parent.append(not worker.base_calls)
        checker.calls.append(
            ("start", device, rank, local_rank, logical_device, physical_gpu)
        )

    checker.start = start
    worker._onchip_checker = checker
    worker._hbm_ecc_monitor = monitor
    worker.device = "cuda:3"
    worker.rank = 7
    worker.local_rank = 3

    result = worker.compile_or_warm_up_model()

    assert result == "compilation-times"
    assert started_before_parent == [True]
    assert worker.base_calls == ["compile"]
    assert checker.calls == [("start", "cuda:3", 7, 3, 3, 27)]
    assert monitor.calls == [("start", 7, 3, 3, 27)]


@pytest.mark.parametrize("num_tokens", [0, 4])
def test_execute_checks_only_nonempty_batch(num_tokens):
    worker = _make_worker(object())
    checker = _FakeChecker()
    monitor = _FakeEccMonitor()
    worker._onchip_checker = checker
    worker._hbm_ecc_monitor = monitor
    scheduler_output = type(
        "SchedulerOutput", (), {"total_num_scheduled_tokens": num_tokens}
    )()

    result = worker.execute_model(scheduler_output)

    assert result == "model-output"
    assert checker.calls == (["before_batch"] if num_tokens else [])
    assert monitor.calls == ["check"]
    assert worker.base_calls == [("execute", scheduler_output)]


def test_checker_failure_prevents_model_execution():
    worker = _make_worker(object())

    class FailingChecker(_FakeChecker):
        def before_batch(self):
            raise RuntimeError("fault")

    worker._onchip_checker = FailingChecker()
    worker._hbm_ecc_monitor = _FakeEccMonitor()
    scheduler_output = type("SchedulerOutput", (), {"total_num_scheduled_tokens": 1})()

    with pytest.raises(RuntimeError, match="fault"):
        worker.execute_model(scheduler_output)

    assert worker.base_calls == []


def test_hbm_ecc_failure_prevents_model_execution():
    worker = _make_worker(object())

    class FailingMonitor(_FakeEccMonitor):
        def check(self):
            raise RuntimeError("hbm ecc fault")

    worker._hbm_ecc_monitor = FailingMonitor()
    worker._onchip_checker = _FakeChecker()
    scheduler_output = type("SchedulerOutput", (), {"total_num_scheduled_tokens": 1})()

    with pytest.raises(RuntimeError, match="hbm ecc fault"):
        worker.execute_model(scheduler_output)

    assert worker.base_calls == []


def test_shutdown_checks_and_frees():
    model_runner = type("ModelRunner", (), {})()
    worker = _make_worker(model_runner)
    checker = _FakeChecker()
    monitor = _FakeEccMonitor()
    worker._onchip_checker = checker
    worker._hbm_ecc_monitor = monitor
    worker._try_replace_model_runner_class = lambda: model_runner.__class__

    worker.shutdown()

    assert checker.calls == ["check_shutdown", "free"]
    assert monitor.calls == ["check", "free"]
    assert worker.base_calls == ["shutdown"]


def test_shutdown_check_failure_does_not_block_cleanup():
    model_runner = type("ModelRunner", (), {})()
    worker = _make_worker(model_runner)

    class FailingChecker(_FakeChecker):
        def check_shutdown(self):
            self.calls.append("check_shutdown")
            raise RuntimeError("inconclusive")

    checker = FailingChecker()
    monitor = _FakeEccMonitor()
    worker._onchip_checker = checker
    worker._hbm_ecc_monitor = monitor
    worker._try_replace_model_runner_class = lambda: model_runner.__class__

    worker.shutdown()

    assert checker.calls == ["check_shutdown", "free"]
    assert monitor.calls == ["check", "free"]
    assert worker.base_calls == ["shutdown"]


def test_parent_shutdown_failure_still_frees_checker():
    model_runner = type("ModelRunner", (), {})()
    worker = _make_worker(model_runner)
    checker = _FakeChecker()
    monitor = _FakeEccMonitor()
    worker._onchip_checker = checker
    worker._hbm_ecc_monitor = monitor
    worker._try_replace_model_runner_class = lambda: model_runner.__class__
    worker.base_shutdown_error = RuntimeError("parent shutdown failed")

    with pytest.raises(RuntimeError, match="parent shutdown failed"):
        worker.shutdown()

    assert checker.calls == ["check_shutdown", "free"]
    assert monitor.calls == ["check", "free"]
    assert worker.base_calls == ["shutdown"]
