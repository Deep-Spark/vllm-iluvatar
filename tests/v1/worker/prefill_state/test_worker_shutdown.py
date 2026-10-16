"""Prefix worker references must outlive drain, then release after shutdown."""

import gc
import threading
import weakref
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker import gpu_worker as upstream_worker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_iluvatar.v1.worker.prefill_state.decode_prefix import PrefixBuffers
from vllm_iluvatar.v1.worker.prefill_state.host_source_worker import HostSourceWorker
from vllm_iluvatar.v1.worker.prefill_state.live_d_worker import LiveDWorker


class ParentShutdownError(RuntimeError):
    pass


def make_worker(worker_cls, *, device="cpu", pinned=False):
    """Construct small owners and the real V2 runner without loading a model."""
    worker = worker_cls.__new__(worker_cls)
    model = torch.nn.Module()
    layer = torch.nn.Module()
    model.add_module("layer", layer)
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.ones(16, device=device))
    )
    backing = torch.ones(1024, dtype=torch.uint8, device=device)
    layer.kv_cache = backing[:512]
    host = torch.empty(32, dtype=torch.uint8, pin_memory=pinned)
    references = {
        "model": weakref.ref(model),
        "weight": weakref.ref(model.weight),
        "KV backing": weakref.ref(backing),
        "KV view": weakref.ref(layer.kv_cache),
        "host": weakref.ref(host),
    }
    config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = config
    runner.model = model
    runner.aux_output_connector = None
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.kv_caches = {"layer": layer.kv_cache}
    runner.attn_groups = []
    worker.model_runner = runner
    worker.vllm_config = config
    worker.profiler = None
    worker.elastic_ep_executor = SimpleNamespace(shutdown=lambda: None)
    if worker_cls is LiveDWorker:
        worker._p21_ready = True
        worker._p21_stop = threading.Event()
        worker._p21_thread = None
        worker._p21_target = model
        worker._p21_cache_sources = [{"cache": layer.kv_cache}]
        worker._p21_prefix = PrefixBuffers(2, 8, 4, torch.device(device))
        references["prefix"] = weakref.ref(worker._p21_prefix.owner)
        worker._p21_packets = {"request": {"host": host}}
        worker._p21_host_inflight = [{"packet": {"host": host}, "event": None}]
        worker._p21_states = {"request": {"host": host}}
        worker._p21_recv = {}
        worker._p21_recv_metadata = {}
        worker._p21_full_ready = {}
        worker._p21_batch = host
        worker._p21_raw_tables = worker._p21_raw_slots = layer.kv_cache
    else:
        worker._src_ready = True
        worker._src_stop = threading.Event()
        worker._src_thread = None
        worker._src_model = model
        worker._src_specs = [{"cache": layer.kv_cache}]
        worker._src_free = deque([{"host": host}])
        worker._src_export = deque([{"pool": {"host": host}}])
        worker._src_active = {"request": {"pool": {"host": host}}}
        worker._src_batch = host
        worker._src_publisher = None
    return worker, references


def isolate_cpu_accelerator(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", "", raising=False)


def owner_collections(worker):
    prefix = "_p21_" if isinstance(worker, LiveDWorker) else "_src_"
    return {
        name: value
        for name, value in vars(worker).items()
        if name.startswith(prefix) and isinstance(value, (list, dict, deque))
    }


@pytest.mark.parametrize("worker_cls", [LiveDWorker, HostSourceWorker])
@pytest.mark.parametrize("thread_state", ["none", "stopped", "running"])
def test_successful_shutdown_releases_prefix_owners_with_live_worker(
    monkeypatch, worker_cls, thread_state
):
    isolate_cpu_accelerator(monkeypatch)
    worker, references = make_worker(worker_cls)
    collections = owner_collections(worker)
    stop = worker._p21_stop if worker_cls is LiveDWorker else worker._src_stop
    stopped = threading.Event()

    def receive_or_export():
        stop.wait()
        stopped.set()

    thread = None
    if thread_state != "none":
        if thread_state == "stopped":
            stop.set()
        thread = threading.Thread(target=receive_or_export)
        thread.start()
        if thread_state == "stopped":
            thread.join()
        setattr(
            worker,
            "_p21_thread" if worker_cls is LiveDWorker else "_src_thread",
            thread,
        )
    parent_calls = 0

    def after_join():
        nonlocal parent_calls
        assert stop.is_set()
        if thread is not None:
            assert stopped.is_set() and not thread.is_alive()
        assert all(
            (ref() is not None) == (parent_calls == 0) for ref in references.values()
        )
        parent_calls += 1

    worker.elastic_ep_executor.shutdown = after_join
    worker.shutdown()
    # These assert the real upstream V2 runner's cleanup, not a mocked parent.
    assert not hasattr(worker.model_runner, "model")
    assert not worker.model_runner.kv_caches
    assert not worker.vllm_config.compilation_config.static_forward_context
    gc.collect()
    assert all(not value for value in collections.values())
    assert {name for name, ref in references.items() if ref() is not None} == set()
    assert worker.model_runner is not None
    worker.shutdown()
    assert parent_calls == 2


@pytest.mark.parametrize("worker_cls", [LiveDWorker, HostSourceWorker])
@pytest.mark.parametrize("failure_phase", ["drain", "after_runner_cleanup"])
def test_parent_shutdown_error_preserves_all_prefix_owners(
    monkeypatch, worker_cls, failure_phase
):
    isolate_cpu_accelerator(monkeypatch)
    worker, references = make_worker(worker_cls)
    collection_ids = {
        name: id(value) for name, value in owner_collections(worker).items()
    }

    def incomplete_drain():
        raise ParentShutdownError("native transfer has not drained")

    if failure_phase == "drain":
        monkeypatch.setattr(
            upstream_worker, "ensure_kv_transfer_shutdown", incomplete_drain
        )
    else:
        original = GPUModelRunner.shutdown

        def cleanup_then_raise(runner):
            original(runner)
            incomplete_drain()

        monkeypatch.setattr(GPUModelRunner, "shutdown", cleanup_then_raise)
    with pytest.raises(ParentShutdownError, match="not drained"):
        worker.shutdown()
    gc.collect()
    assert all(ref() is not None for ref in references.values())
    assert {
        name: id(value) for name, value in owner_collections(worker).items()
    } == collection_ids
    assert hasattr(worker.model_runner, "model") == (failure_phase == "drain")


@pytest.mark.parametrize("worker_cls", [LiveDWorker, HostSourceWorker])
def test_shutdown_before_prefix_initialization(monkeypatch, worker_cls):
    isolate_cpu_accelerator(monkeypatch)
    worker, references = make_worker(worker_cls)
    prefix = "_p21_" if worker_cls is LiveDWorker else "_src_"
    for name in tuple(vars(worker)):
        if name.startswith(prefix):
            delattr(worker, name)
    worker.shutdown()
    gc.collect()
    assert all(ref() is None for ref in references.values())


@pytest.mark.parametrize("worker_cls", [LiveDWorker, HostSourceWorker])
def test_thread_join_error_retains_owners_and_does_not_shutdown_runner(
    monkeypatch, worker_cls
):
    isolate_cpu_accelerator(monkeypatch)
    worker, references = make_worker(worker_cls)

    def failed_join():
        raise ParentShutdownError("receiver/exporter did not stop")

    setattr(
        worker,
        "_p21_thread" if worker_cls is LiveDWorker else "_src_thread",
        SimpleNamespace(join=failed_join),
    )
    with pytest.raises(ParentShutdownError, match="did not stop"):
        worker.shutdown()
    gc.collect()
    assert all(ref() is not None for ref in references.values())
    assert hasattr(worker.model_runner, "model")


@pytest.mark.parametrize("worker_cls", [LiveDWorker, HostSourceWorker])
def test_cuda_shutdown_waits_for_host_copy_before_releasing_owners(
    monkeypatch, worker_cls
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is needed for asynchronous pinned-host ownership")
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", "", raising=False)
    worker, references = make_worker(worker_cls, device="cuda", pinned=True)
    stream, completed = torch.cuda.Stream(), torch.cuda.Event()
    with torch.cuda.stream(stream):
        references["KV backing"]().fill_(37)
        references["host"]().copy_(references["KV backing"]()[:32], non_blocking=True)
        completed.record()
    if worker_cls is LiveDWorker:
        worker._p21_host_inflight[0]["event"] = completed
    else:
        worker._src_export[0]["events"] = [completed]

    def after_synchronize():
        assert completed.query()
        assert references["host"]().tolist() == [37] * 32
        assert all(ref() is not None for ref in references.values())

    worker.model_runner.aux_output_connector = SimpleNamespace(close=after_synchronize)
    worker.shutdown()
    gc.collect()
    assert all(ref() is None for ref in references.values())
