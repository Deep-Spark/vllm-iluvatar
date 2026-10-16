"""Real E25 load hooks must become inert and release their owner on shutdown."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed as distributed
import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.connector import (
    OptNixlConnector,
)
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41Model
from vllm_iluvatar.v1.worker.gpu_worker import IluWorker
from vllm_iluvatar.v1.worker.prefill_state import live_d_worker


@pytest.fixture(autouse=True)
def restore_global_hooks():
    """Restore exact class ownership even if a load or shutdown check fails."""
    missing = object()
    previous = {
        name: vars(OptNixlConnector).get(name, missing)
        for name in ("bind_connector_metadata", "get_transfer_results")
    }
    yield
    for name, value in previous.items():
        if value is missing:
            if name in vars(OptNixlConnector):
                delattr(OptNixlConnector, name)
        else:
            setattr(OptNixlConnector, name, value)


def load_live_worker(monkeypatch, tmp_path):
    """Isolate weights/groups/device calls; keep real load hooks and shutdown."""
    monkeypatch.setattr(IluWorker, "load_model", lambda *args, **kwargs: "loaded")
    for name in ("get_ep_group", "get_dp_group", "get_tp_group"):
        monkeypatch.setattr(
            distributed, name, lambda: SimpleNamespace(rank_in_group=0, world_size=1)
        )
    monkeypatch.setattr(
        live_d_worker, "local_paths", lambda engine: (tmp_path, "audit-run")
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_ILU_MLA_ATTN_QUANT_TYPE", "", raising=False)

    model = IluDeepseekV41Model.__new__(IluDeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.start_layer, model.end_layer = 0, 40
    model.config = SimpleNamespace(hidden_size=8)
    model.hc_mult = 4
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(16)))
    runner = GPUModelRunner.__new__(GPUModelRunner)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="audit-decode"),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
    )
    runner.vllm_config = config
    runner.model = model
    runner.aux_output_connector = None
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, prepare_attn=lambda *args, **kwargs: None
    )
    runner.kv_caches, runner.attn_groups = [], []
    runner.req_states = SimpleNamespace(
        add_request=lambda *args: None, remove_request=lambda *args: None
    )
    runner.prepare_inputs = lambda *args: None
    runner.prepare_attn = lambda *args: None
    runner.execute_model = lambda *args, **kwargs: None
    worker = live_d_worker.LiveDWorker.__new__(live_d_worker.LiveDWorker)
    worker.vllm_config, worker.model_runner = config, runner
    worker.profiler = None
    worker.elastic_ep_executor = SimpleNamespace(shutdown=lambda: None)
    worker.device = torch.device("cpu")
    assert worker.load_model() == "loaded"
    return worker


def test_successful_shutdown_restores_hooks_and_releases_loaded_worker(
    monkeypatch, tmp_path
):
    previous = (
        OptNixlConnector.bind_connector_metadata,
        OptNixlConnector.get_transfer_results,
    )
    local_methods = set(vars(OptNixlConnector))
    worker = load_live_worker(monkeypatch, tmp_path)
    reference = weakref.ref(worker)
    worker.shutdown()
    assert not hasattr(worker.model_runner, "model")
    assert worker._p21_target is None
    assert (
        OptNixlConnector.bind_connector_metadata,
        OptNixlConnector.get_transfer_results,
    ) == previous
    assert set(vars(OptNixlConnector)) == local_methods
    del worker
    gc.collect()
    assert reference() is None


def test_parent_shutdown_error_retains_hooks_and_owners_until_retry(
    monkeypatch, tmp_path
):
    worker = load_live_worker(monkeypatch, tmp_path)
    target = worker._p21_target
    installed = (
        OptNixlConnector.bind_connector_metadata,
        OptNixlConnector.get_transfer_results,
    )
    with monkeypatch.context() as failure:

        def incomplete(*args, **kwargs):
            raise RuntimeError("parent drain incomplete")

        failure.setattr(IluWorker, "shutdown", incomplete)
        with pytest.raises(RuntimeError, match="parent drain incomplete"):
            worker.shutdown()
    assert worker._p21_target is target
    assert worker._p21_connector_hooks
    assert (
        OptNixlConnector.bind_connector_metadata,
        OptNixlConnector.get_transfer_results,
    ) == installed
    worker.shutdown()
    assert not worker._p21_connector_hooks and worker._p21_target is None


def test_later_connector_wrapper_survives_without_retaining_shutdown_worker(
    monkeypatch, tmp_path
):
    worker = load_live_worker(monkeypatch, tmp_path)
    reference = weakref.ref(worker)
    installed = OptNixlConnector.get_transfer_results

    def later(connector, finished_req_ids):
        return installed(connector, finished_req_ids)

    monkeypatch.setattr(OptNixlConnector, "get_transfer_results", later)
    worker.shutdown()
    assert OptNixlConnector.get_transfer_results is later
    del worker
    gc.collect()
    assert reference() is None

    def forbidden():
        pytest.fail("A shutdown owner created a CUDA completion event")

    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    expected = KVConnectorTransferResults(finished_recving={"new-request"})
    connector = OptNixlConnector.__new__(OptNixlConnector)
    connector.connector_worker = SimpleNamespace(get_transfer_results=lambda: expected)
    assert connector.get_transfer_results(set()) is expected


def test_active_live_hook_records_success_and_clears_failed_receipt(
    monkeypatch, tmp_path
):
    worker = load_live_worker(monkeypatch, tmp_path)
    worker._p21_ready = True
    worker._p21_recv_metadata["ready"] = {"current": True}
    worker._p21_recv["failed"] = worker._p21_full_ready["failed"] = object()
    records = []
    event = SimpleNamespace(record=lambda: records.append("record"))
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    expected = KVConnectorTransferResults(
        finished_recving={"ready", "failed"}, failed_recving={"failed"}
    )
    connector = OptNixlConnector.__new__(OptNixlConnector)
    connector.connector_worker = SimpleNamespace(get_transfer_results=lambda: expected)
    assert connector.get_transfer_results(set()) is expected
    assert records == ["record"]
    assert "failed" not in worker._p21_recv and "failed" not in worker._p21_full_ready
    assert worker._p21_recv["ready"]["event"] is event
    assert worker._p21_recv["ready"]["metadata"] == {"current": True}
    worker.shutdown()
