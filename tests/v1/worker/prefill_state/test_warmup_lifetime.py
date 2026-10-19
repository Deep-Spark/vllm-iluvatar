import weakref

from vllm_iluvatar.v1.worker.prefill_state.consumer import ConsumerPrefixState


def test_consumer_does_not_retain_warmup_batch():
    class Batch:
        pass

    state = ConsumerPrefixState.__new__(ConsumerPrefixState)
    state._p21_ready = False
    state._p21_batch = None
    batch = Batch()
    reference = weakref.ref(batch)

    state.prepare_inputs(batch)
    del batch
    assert reference() is None

    state._p21_ready = True
    batch = Batch()
    reference = weakref.ref(batch)
    state.prepare_inputs(batch)
    del batch
    assert reference() is not None
    state.end_step()
    assert reference() is None


# Exercise the real peer-gate branch without loading a CUDA context.
def _peer_gate_state(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import threading
    import numpy as np
    import torch
    import vllm.distributed as distributed
    import vllm.forward_context as forward_context
    from vllm.config.compilation import CUDAGraphMode
    import vllm_iluvatar.v1.worker.prefill_state.consumer as consumer

    state = ConsumerPrefixState.__new__(ConsumerPrefixState)
    state._p21_ready = True
    state._p21_batch = SimpleNamespace(
        query_start_loc_np=np.array([0, 1]), num_reqs=1,
        is_prefilling_np=np.array([False]), req_ids=["r0"])
    state._p21_dummy = False
    state._p21_states = {"r0": {}}
    state._p21_max_reqs = 2
    state._p21_prefix = SimpleNamespace(max_tokens=12)
    state.model_runner = SimpleNamespace()
    state._p21_target = SimpleNamespace(aux_hidden_state_layers=(37, 38, 39))
    state.geometry = SimpleNamespace(prefix_depth=20)
    state.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=False))
    state._p21_raw_tables = ()
    state._p21_raw_slots = torch.empty(0, dtype=torch.int64)
    state._p21_counters = {"NONE": 0, "fallback": 0}
    state._p21_reasons = {}
    state._p21_eager_warmed = False
    state._p21_result = tmp_path
    state._p21_rank = 0
    state._p21_cache_sources = []
    state._p21_lock = threading.Lock()
    group = SimpleNamespace(world_size=16, cpu_group=None)
    monkeypatch.setattr(distributed, "get_ep_group", lambda: group)
    monkeypatch.setattr(distributed, "get_tp_group",
                        lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(forward_context, "get_forward_context", lambda:
        SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, attn_metadata={}))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1050*2**20, 32*2**30))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    def no_cuda():
        raise AssertionError("CPU gate test must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    def gather(votes, decision, **_):
        for vote in votes:
            vote.copy_(torch.tensor([1, 1, 0, 12, 1]))
        votes[0].copy_(decision)
        votes[1][2] = 1  # Another DP has a valid tail replay.
    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    class ReachedGo(Exception):
        pass
    def save_gate(path, value):
        if value["go"]:
            raise ReachedGo
    monkeypatch.setattr(consumer, "save", save_gate)
    return state, ReachedGo


def test_successful_eager_fallback_warms_common_gate(monkeypatch, tmp_path):
    import pytest
    state, reached_go = _peer_gate_state(monkeypatch, tmp_path)
    marker = object()
    assert state.forward(lambda *args: marker, None, None, None) is marker
    assert state._p21_eager_warmed
    with pytest.raises(reached_go):
        state.forward(lambda *args: marker, None, None, None)


def test_failed_eager_fallback_does_not_warm_gate(monkeypatch, tmp_path):
    import pytest
    state, _ = _peer_gate_state(monkeypatch, tmp_path)
    def fail(*args):
        raise RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        state.forward(fail, None, None, None)
    assert not state._p21_eager_warmed


def test_dummy_rank_warms_after_common_real_eager_work(monkeypatch, tmp_path):
    state, _ = _peer_gate_state(monkeypatch, tmp_path)
    state._p21_dummy = True
    state.forward(lambda *args: None, None, None, None)
    assert state._p21_eager_warmed


def test_all_dummy_fallback_stays_cold(monkeypatch, tmp_path):
    import torch
    state, _ = _peer_gate_state(monkeypatch, tmp_path)
    state._p21_dummy = True

    def all_dummy(votes, decision, **_):
        for vote in votes:
            vote.copy_(torch.tensor([1, 0, 0, 12, 0]))

    monkeypatch.setattr(torch.distributed, "all_gather", all_dummy)
    state.forward(lambda *args: None, None, None, None)
    assert not state._p21_eager_warmed
