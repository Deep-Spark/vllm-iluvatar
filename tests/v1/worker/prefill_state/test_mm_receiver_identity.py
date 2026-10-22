"""Verify the real async E25 pending snapshot forwards certified MM identity."""

import threading
from types import SimpleNamespace

import torch
import zmq

from vllm_iluvatar.v1.worker.prefill_state import consumer


def test_receiver_pending_snapshot_keeps_mm_proof(tmp_path, monkeypatch):
    state = consumer.ConsumerPrefixState.__new__(consumer.ConsumerPrefixState)
    state.device = "cpu"
    state._p21_lock = threading.Lock()
    state._p21_stop = threading.Event()
    state._p21_host_inflight = []
    state._p21_packets = {}
    state._p21_unavailable = set()
    state._p21_states = {
        "image": dict(
            idx=0,
            generation=1,
            N=129,
            eligible=True,
            token_sha256="a" * 64,
            mm_input_digest="b" * 64,
        )
    }
    state._p21_recv_metadata = {
        "image": dict(source_endpoint=dict(engine_id="P", addr="unused:1"))
    }
    state._p21_tp = 0
    state.geometry = None
    state._p21_ring_rows = {}
    state._p21_rank = 0
    state._p21_result = tmp_path
    state._p21_counters = {}
    state._p21_reasons = {}
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)

    class Socket:
        def setsockopt(self, *_):
            pass

        def connect(self, *_):
            pass

        def close(self, *_):
            pass

    monkeypatch.setattr(
        zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *_: Socket(), term=lambda: None),
    )
    seen = []

    def fetch(*args, **kwargs):
        seen.append(kwargs["mm_input_digest"])
        state._p21_stop.set()
        return dict(status="PENDING"), b""

    monkeypatch.setattr(consumer, "fetch_state", fetch)
    state._p21_receiver()
    assert seen == ["b" * 64]
