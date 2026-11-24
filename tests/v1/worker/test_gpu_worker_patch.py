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

    # IluWorker imports ``vllm.v1.worker.gpu.model_runner`` (the dotted
    # path used since v0.30.0), whose real module pulls ``vllm.v1.watermarking``
    # at import time.
    # Stub the dotted path -- plus the ``vllm.v1.worker.gpu`` package -- so the
    # import resolves here and never touches the real model_runner.
    gpu_package = make_package("vllm.v1.worker.gpu")
    gpu_model_runner_mod = ModuleType("vllm.v1.worker.gpu.model_runner")
    gpu_model_runner_mod.GPUModelRunner = type("GPUModelRunner", (), {})

    kv_cache_mod = ModuleType("vllm.v1.kv_cache_interface")
    kv_cache_mod.KVCacheConfig = type("KVCacheConfig", (), {})

    vllm_mod.v1 = vllm_v1_mod
    vllm_v1_mod.worker = worker_package
    worker_package.gpu_worker = gpu_worker_mod
    worker_package.gpu = gpu_package
    gpu_package.model_runner = gpu_model_runner_mod

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

    modules = {
        "vllm.logger": logger_mod,
        "vllm.platforms": platforms_mod,
        "vllm": vllm_mod,
        "vllm.v1": vllm_v1_mod,
        "vllm.v1.worker": worker_package,
        "vllm.v1.worker.gpu": gpu_package,
        "vllm.v1.kv_cache_interface": kv_cache_mod,
        "vllm.v1.worker.gpu.model_runner": gpu_model_runner_mod,
        "vllm.v1.worker.gpu_worker": gpu_worker_mod,
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
    w.local_rank = 0
    w.rank = 0
    w.device = None
    # This CPU-only fixture does not run platform/env registration or own a
    # GPU. Keep hardware monitors inert while testing warmup and PP routing.
    w._onchip_checker = SimpleNamespace(start=lambda *a, **kw: None, free=lambda: None)
    w._hbm_ecc_monitor = SimpleNamespace(start=lambda *a, **kw: None, free=lambda: None)
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


def test_pcp_profile_bound_covers_mixed_dual_chunk_batches():
    import random

    from vllm_iluvatar.v1.worker.gpu_worker import _pcp_profile_local_tokens

    rng = random.Random(42)
    for pcp in (2, 8, 16):
        for _ in range(100):
            sizes = [rng.randint(1, 1024) for _ in range(rng.randint(1, 64))]
            prefills = [n != 1 for n in sizes]
            bound = _pcp_profile_local_tokens(sum(sizes), pcp, len(sizes))
            for rank in range(pcp):
                local = 0
                for n, prefill in zip(sizes, prefills):
                    if not prefill:
                        local += n
                        continue
                    chunk = (n + 2 * pcp - 1) // (2 * pcp)
                    local += sum(
                        max(0, min(chunk, n - index * chunk))
                        for index in (rank, 2 * pcp - 1 - rank)
                    )
                assert local <= bound <= sum(sizes)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_pcp_profile_only_changes_profile_calls_and_restores(raise_inside):
    calls = []

    class Runner:
        def _dummy_run(self, n, **kwargs):
            calls.append((n, kwargs))

    runner = Runner()
    worker = _make_worker(runner)
    worker.use_v2_model_runner = True
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=16)
    worker.scheduler_config = SimpleNamespace(max_num_seqs=64)
    worker.vllm_config = SimpleNamespace(speculative_config=None)
    try:
        with worker._pcp_memory_profile():
            runner._dummy_run(10240, is_profile=True, skip_attn=True)
            runner._dummy_run(10240, is_profile=False)
            if raise_inside:
                raise RuntimeError("test cleanup")
    except RuntimeError:
        assert raise_inside
    assert calls == [
        (768, {"is_profile": True, "skip_attn": True}),
        (10240, {"is_profile": False}),
    ]
    assert "_dummy_run" not in vars(runner)
    runner._dummy_run(10240, is_profile=True)
    assert calls[-1] == (10240, {"is_profile": True})


@pytest.mark.parametrize(
    "v2,pcp,spec", [(False, 16, None), (True, 1, None), (True, 16, object())]
)
def test_pcp_profile_does_not_change_other_runner_modes(v2, pcp, spec):
    class Runner:
        def _dummy_run(self, n, **kwargs):
            return n

    runner = Runner()
    worker = _make_worker(runner)
    worker.use_v2_model_runner = v2
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=pcp)
    worker.vllm_config = SimpleNamespace(speculative_config=spec)
    with worker._pcp_memory_profile():
        assert runner._dummy_run(10240, is_profile=True) == 10240
        assert "_dummy_run" not in vars(runner)


def test_non_glm_warmup_delegates_to_upstream(monkeypatch):
    attention = ModuleType("vllm_iluvatar.models.glm_moe_dsa.attention")
    attention.IluGlmMoeDsaSparseMLA = type("GlmLayer", (), {})
    monkeypatch.setitem(sys.modules, attention.__name__, attention)

    base = sys.modules["vllm.v1.worker.gpu_worker"].Worker
    monkeypatch.setattr(
        base, "compile_or_warm_up_model", lambda self: "base", raising=False
    )

    class Model:
        def modules(self):
            return [object()]

    worker = _make_worker(SimpleNamespace(model=Model()))
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=1)
    assert worker.compile_or_warm_up_model() == "base"


@pytest.mark.parametrize("pcp_size", [1, 16])
def test_warmup_temporarily_disables_glm_calibration(monkeypatch, pcp_size):
    class GlmLayer:
        _fp8_kv_scale_calibration_enabled = True

    attention = ModuleType("vllm_iluvatar.models.glm_moe_dsa.attention")
    attention.IluGlmMoeDsaSparseMLA = GlmLayer
    monkeypatch.setitem(sys.modules, attention.__name__, attention)

    layer = GlmLayer()
    model = SimpleNamespace(modules=lambda: [layer])
    worker = _make_worker(SimpleNamespace(model=model))
    worker.parallel_config = SimpleNamespace(prefill_context_parallel_size=pcp_size)
    base = sys.modules["vllm.v1.worker.gpu_worker"].Worker

    def base_warmup(self):
        assert layer._fp8_kv_scale_calibration_enabled is False
        return "base"

    monkeypatch.setattr(base, "compile_or_warm_up_model", base_warmup, raising=False)
    assert worker.compile_or_warm_up_model() == "base"
    assert layer._fp8_kv_scale_calibration_enabled is True


@pytest.mark.parametrize(
    "limit,pp,eager,expected",
    [(20, 8, True, 21), (20, 1, True, 20), (20, 8, False, 20), (32, 8, True, 32)],
)
def test_eager_pp_allocator_limit(
    _stub_gpu_worker, monkeypatch, limit, pp, eager, expected
):
    from types import SimpleNamespace

    seen = []
    sentinel = object()

    def scoped(self, value):
        seen.append(value)
        return sentinel

    monkeypatch.setattr(
        _stub_gpu_worker.Worker, "_scoped_allocator_max_split", scoped, raising=False
    )
    worker = _make_worker(None)
    worker.parallel_config = SimpleNamespace(pipeline_parallel_size=pp)
    worker.model_config = SimpleNamespace(enforce_eager=eager)
    assert worker._scoped_allocator_max_split(limit) is sentinel
    assert seen == [expected]


def test_pp_exchange_schema_and_drain(monkeypatch):
    import torch

    from vllm_iluvatar.v1.worker import pp_exchange as pe

    schema = (
        ("hidden_states", (4096,), torch.bfloat16),
        ("residual", (4, 4096), torch.bfloat16),
    )

    class FakeWork:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

        def is_completed(self):
            return self.waited

    works = []

    def fake_isend(tensor, dst, group):
        w = FakeWork()
        works.append(w)
        return w

    monkeypatch.setattr(pe.dist, "isend", fake_isend)
    pp = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        ranks=[0, 1],
        device_group="g",
        isend_tensor_dict=None,
        irecv_tensor_dict=None,
    )

    class CudaMeta:
        def __init__(self, t):
            self._t = t
            self.shape = t.shape
            self.dtype = t.dtype
            self.device = t.device
            self.is_cuda = True
            self.recorded_streams = []

        def is_contiguous(self):
            return True

        def untyped_storage(self):
            return self._t.untyped_storage()

        def record_stream(self, stream):
            self.recorded_streams.append(stream)

    wrapped = {
        k: CudaMeta(torch.zeros((4, *shape), dtype=dtype, device="meta"))
        for k, shape, dtype in schema
    }
    handles = pe.send_schema_tensors(pp, wrapped, 4, schema, retain_send=True)
    assert handles[0] is None
    assert all(isinstance(h, pe.RetainedSend) for h in handles[1:])
    assert all(isinstance(h.tensor, CudaMeta) for h in handles[1:])

    worker = SimpleNamespace(_pp_send_work=handles[1:])
    pending = pe.defer_pp_send_wait(worker)
    assert len(pending) == 1
    assert worker._pp_send_work == []
    with (
        pe.schema_pp_exchange(
            pp, 4, "meta", send_schema=schema, pending=pending, retain_send=True
        ),
        pytest.raises(ValueError, match="reused"),
    ):
        pp.isend_tensor_dict(wrapped)

    pe.drain_pp_send_wait(worker)
    assert len(pending) == 0
    assert all(w.waited for w in works)

    # Non-retained mode returns raw work handles and records current stream.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: "stream0")
    handles_direct = pe.send_schema_tensors(pp, wrapped, 4, schema, retain_send=False)
    assert handles_direct[0] is None
    assert all(isinstance(h, FakeWork) for h in handles_direct[1:])
    assert all(w.recorded_streams == ["stream0"] for w in wrapped.values())


@pytest.mark.parametrize(
    "fresh_alloc,free_bytes,schema_match,expected_active,expected_defer",
    [
        (True, 2 << 30, True, True, True),
        (False, 2 << 30, True, True, False),
        (True, 64 << 20, True, True, False),
        (True, 2 << 30, False, False, False),
    ],
)
def test_configure_eager_pp_exchange_negotiation(
    monkeypatch, fresh_alloc, free_bytes, schema_match, expected_active, expected_defer
):
    import torch

    import vllm_iluvatar.v1.worker.gpu_worker as gw

    cg_mode = type("CUDAGraphMode", (), {"NONE": "NONE", "FULL": "FULL"})
    vllm_cfg_mod = ModuleType("vllm.config")
    vllm_comp_mod = ModuleType("vllm.config.compilation")
    vllm_comp_mod.CUDAGraphMode = cg_mode
    vllm_cfg_mod.compilation = vllm_comp_mod
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_cfg_mod)
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", vllm_comp_mod)

    monkeypatch.setattr(gw, "is_ivcore11", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _dev: (free_bytes, 32 << 30))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class FakeTensor:
        def __init__(self, shape, dtype, ptr):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = dtype
            self.is_cuda = True
            self._ptr = ptr

        def is_contiguous(self):
            return True

        def numel(self):
            n = 1
            for s in self.shape:
                n *= s
            return n

        def element_size(self):
            return self.dtype.itemsize

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self._ptr)

    class FakeIntermediate:
        def __init__(self, tensors):
            self.tensors = tensors

    vllm_seq = ModuleType("vllm.sequence")
    vllm_seq.IntermediateTensors = FakeIntermediate
    monkeypatch.setitem(sys.modules, "vllm.sequence", vllm_seq)

    persistent_buf = torch.empty((1,), dtype=torch.bfloat16)
    persistent_ptr = persistent_buf.untyped_storage().data_ptr()

    class FakeModel:
        def __init__(self):
            self._hooks = []

        def register_forward_hook(self, fn):
            self._hooks.append(fn)
            return SimpleNamespace(remove=lambda: self._hooks.remove(fn))

        def parameters(self):
            return []

        def buffers(self):
            return [persistent_buf]

        def make_empty_intermediate_tensors(self, batch_size, dtype, device):
            return FakeIntermediate({"h": FakeTensor((batch_size, 16), dtype, 999)})

        def __call__(self, *args, **kwargs):
            ptr = 12345 if fresh_alloc else persistent_ptr
            out = FakeIntermediate({"h": FakeTensor((2, 16), torch.bfloat16, ptr)})
            for fn in list(self._hooks):
                fn(self, args, out)
            return out

    pp = SimpleNamespace(
        rank_in_group=1,
        world_size=2,
        is_first_rank=False,
        is_last_rank=True,
        cpu_group="cpu_g",
    )
    vllm_dist = ModuleType("vllm.distributed")
    vllm_dist.get_pp_group = lambda: pp
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_dist)

    def fake_all_gather_object(dst, local_val, group=None):
        dst[0] = {
            "send_schema": (("h", (16,), torch.bfloat16),),
            "recv_schema": None,
            "fresh_alloc": fresh_alloc,
            "free_bytes": free_bytes,
        }
        dst[1] = (
            local_val
            if schema_match
            else {
                "send_schema": None,
                "recv_schema": (("other", (16,), torch.bfloat16),),
                "fresh_alloc": fresh_alloc,
                "free_bytes": free_bytes,
            }
        )

    monkeypatch.setattr(gw.dist, "all_gather_object", fake_all_gather_object)

    class Runner:
        decode_query_len = 1
        uniform_decode_query_len = 1
        max_num_tokens = 2560
        is_pooling_model = False
        is_encoder_decoder = False
        use_aux_hidden_state_outputs = False
        pp_handler = SimpleNamespace(aux_hidden_state_relay_keys=None)
        ubatch_runner = None
        execute_model_state = "stale"
        model = FakeModel()
        intermediate_tensors = FakeIntermediate(
            {"h": FakeTensor((2560, 16), torch.bfloat16, 777)}
        )

        def _dummy_run(self, n, **kwargs):
            self.model()

    worker = _make_worker(Runner())
    worker.device = SimpleNamespace(type="cuda")
    worker.use_v2_model_runner = True
    worker.model_config = SimpleNamespace(enforce_eager=True, dtype=torch.bfloat16)
    worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=2560)
    worker.parallel_config = SimpleNamespace(
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_sequence_parallel_moe=False,
        enable_dbo=False,
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=cg_mode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        )
    )

    worker._configure_eager_pp_exchange()
    assert (worker._pp_recv_schema is not None) is expected_active
    assert worker._pp_defer_send_wait is expected_defer
    assert worker.model_runner.intermediate_tensors.tensors["h"].shape[0] == 16
