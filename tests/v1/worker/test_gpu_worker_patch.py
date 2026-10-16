# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the IluWorker subclass (dummy-batch optimisation)."""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

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
        def determine_available_memory(self):
            self.base_calls.append("profile")
            explicit = getattr(self.cache_config, "kv_cache_memory_bytes", None)
            return self.available_memory if explicit is None else explicit

        def initialize_from_config(self, config):
            self.base_calls.append(("initialize", config))

        def _maybe_get_memory_pool_context(self, *, tag):
            assert tag == "kv_cache"
            return nullcontext()

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
    if hasattr(model_runner, "__dict__") and not hasattr(
        model_runner, "profile_cudagraph_memory"
    ):
        model_runner.profile_cudagraph_memory = lambda: 0
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


@pytest.mark.parametrize("format_name", ["i8g64", "i4g32"])
@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("explicit_budget", [False, True])
def test_new_format_profiles_total_budget_without_replacing_v2_class(
    format_name, role, explicit_budget, monkeypatch
):
    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", format_name)
    legacy_envs = ModuleType("vllm.envs")
    legacy_envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE = ""
    legacy_envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE = ""
    monkeypatch.setitem(sys.modules, "vllm.envs", legacy_envs)
    runner = type("V2Runner", (), {})()
    worker = _make_worker(runner)
    worker.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None if role is None else SimpleNamespace(kv_role=role),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41", index_topk=512)
        ),
    )
    expected_workspace = (512 if role in ("kv_producer", "kv_both") else 128) * 1024**2
    original_budget = expected_workspace + 8 * 65536
    worker.cache_config = SimpleNamespace(
        cache_dtype="bfloat16",
        kv_cache_memory_bytes=original_budget if explicit_budget else None,
    )
    worker.available_memory = (
        2 * original_budget if explicit_budget else original_budget
    )
    payload = worker.determine_available_memory()
    budget = worker._compressed_kv_budget
    assert worker.base_calls == ["profile"]
    assert worker.model_runner.__class__ is runner.__class__
    assert budget.total_bytes == original_budget
    assert budget.workspace_bytes == expected_workspace
    assert budget.prefill_producer is (role in ("kv_producer", "kv_both"))
    assert budget.retained_mapper_bytes == 4202496
    assert budget.arena_bytes == expected_workspace - budget.retained_mapper_bytes
    assert payload < original_budget - expected_workspace

    # Initialization must use the profiled object even if the role is changed
    # between phases; a policy re-query must not create an unbudgeted arena.
    worker.vllm_config.kv_transfer_config = SimpleNamespace(
        kv_role="kv_consumer" if budget.prefill_producer else "kv_producer"
    )
    worker.vllm_config.scheduler_config.max_num_batched_tokens = 8192
    worker.vllm_config.model_config.hf_text_config.index_topk = 1024
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv

    allocations = []

    class State:
        scales = {}

        def __init__(
            self, layout, fmt, device, context, *, workspace_bytes, prefill_producer
        ):
            allocations.append(
                (layout, fmt, device, context, prefill_producer, workspace_bytes)
            )

    page_count = payload // 65536
    layout = SimpleNamespace(
        payload_bytes=payload, scale_bytes=page_count * budget.scale_page_bytes
    )
    monkeypatch.setattr(compressed_kv, "plan_scale_layout", lambda *args: layout)
    monkeypatch.setattr(compressed_kv, "CompressedKVState", State)
    runner.kv_caches = []
    worker.device = "meta"
    config = object()
    worker.initialize_from_config(config)
    assert worker._compressed_kv_budget is budget
    assert allocations[0][-1] == budget.arena_bytes
    assert allocations[0][-2] is budget.prefill_producer
    assert len(allocations) == 1
    assert worker.base_calls == ["profile", ("initialize", config)]


def test_none_keeps_baseline_budget_and_skips_role_policy(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv

    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "none")
    monkeypatch.setattr(
        compressed_kv,
        "workspace_bytes_for_config",
        lambda *_: pytest.fail("none must not select a compressed workspace"),
    )
    monkeypatch.setattr(
        compressed_kv,
        "retained_mapper_bytes_for_config",
        lambda *_: pytest.fail("none must not credit a compressed mapper"),
    )
    worker = _make_worker(type("V2Runner", (), {})())
    worker.available_memory = 8 * 1024**3
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    worker._try_replace_model_runner_class = lambda: worker.model_runner.__class__
    assert worker.determine_available_memory() == worker.available_memory
    config = object()
    worker.initialize_from_config(config)
    assert worker.base_calls == ["profile", ("initialize", config)]
    assert not hasattr(worker, "_compressed_kv_budget")


@pytest.mark.parametrize("format_name", ["i8g64", "i4g32"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_manual_block_override_is_rejected_before_compressed_allocation(
    format_name, role, monkeypatch
):
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv

    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", format_name)
    legacy_envs = ModuleType("vllm.envs")
    legacy_envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE = ""
    legacy_envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE = ""
    monkeypatch.setitem(sys.modules, "vllm.envs", legacy_envs)
    worker = _make_worker(type("V2Runner", (), {})())
    worker.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=SimpleNamespace(kv_role=role),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41", index_topk=512)
        ),
    )
    total = 8 * 1024**3
    worker.cache_config = SimpleNamespace(
        cache_dtype="bfloat16", kv_cache_memory_bytes=total
    )
    worker.available_memory = 2 * total
    payload = worker.determine_available_memory()
    budget = worker._compressed_kv_budget
    override_blocks = payload // 65536 + 1
    worker.cache_config.num_gpu_blocks_override = override_blocks
    layout = SimpleNamespace(
        payload_bytes=override_blocks * 65536,
        scale_bytes=override_blocks * budget.scale_page_bytes,
    )
    monkeypatch.setattr(compressed_kv, "plan_scale_layout", lambda *args: layout)
    monkeypatch.setattr(
        compressed_kv,
        "CompressedKVState",
        lambda *args, **kwargs: pytest.fail("over-budget state must not allocate"),
    )
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        worker.initialize_from_config(object())
    assert worker.base_calls == ["profile"]
    assert worker._compressed_kv_budget is budget
    assert not hasattr(worker, "_compressed_kv_init_attempted")


def test_new_format_rejects_legacy_v1_runner_before_profiling(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "i8g64")
    worker = _make_worker(type("V1Runner", (), {})())
    worker.vllm_config = SimpleNamespace(use_v2_model_runner=False)
    with pytest.raises(NotImplementedError, match="model runner V2"):
        worker.determine_available_memory()
    assert worker.base_calls == []


@pytest.mark.parametrize("format_name", ["i8g64", "i4g32"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_unaffordable_retained_mapper_credit_rejects_before_state_allocation(
    format_name, role, monkeypatch
):
    from vllm_iluvatar.models.deepseek_v41 import compressed_kv

    monkeypatch.setenv("VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", format_name)
    legacy_envs = ModuleType("vllm.envs")
    legacy_envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE = ""
    legacy_envs.VLLM_ILU_MLA_ATTN_QUANT_TYPE = ""
    monkeypatch.setitem(sys.modules, "vllm.envs", legacy_envs)
    worker = _make_worker(type("V2Runner", (), {})())
    worker.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=SimpleNamespace(kv_role=role),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1 << 20),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41", index_topk=512)
        ),
    )
    worker.cache_config = SimpleNamespace(
        cache_dtype="bfloat16", kv_cache_memory_bytes=None
    )
    worker.available_memory = 8 * 1024**3
    monkeypatch.setattr(
        compressed_kv,
        "CompressedKVState",
        lambda *args, **kwargs: pytest.fail(
            "unaffordable credit must not allocate State"
        ),
    )
    with pytest.raises(ValueError, match="mapper credit leaves no usable"):
        worker.determine_available_memory()
    assert worker.base_calls == []
    assert not hasattr(worker, "_compressed_kv_budget")
    assert not hasattr(worker, "_compressed_kv_init_attempted")


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


def test_native_prefix_context_observes_batch_and_releases_after_execution():
    from types import SimpleNamespace

    events = []
    worker = _make_worker(object())
    worker._onchip_checker = _FakeChecker()
    worker._hbm_ecc_monitor = _FakeEccMonitor()
    output = SimpleNamespace(total_num_scheduled_tokens=1)
    worker.prefix_state = SimpleNamespace(
        observe_scheduler_output=lambda out: events.append(("observe", out)),
        end_step=lambda: events.append("end"),
    )
    assert worker.execute_model(output) == "model-output"
    assert events == [("observe", output), "end"]


def test_native_prefix_context_does_not_survive_a_failed_dummy_step():
    from types import SimpleNamespace

    events = []

    def fail(*args, **kwargs):
        raise RuntimeError("dummy failed")

    worker = _make_worker(SimpleNamespace(_dummy_run=fail))
    worker.prefix_state = SimpleNamespace(end_step=lambda: events.append("end"))
    with pytest.raises(RuntimeError, match="dummy failed"):
        worker.execute_dummy_batch()
    assert events == ["end"]
