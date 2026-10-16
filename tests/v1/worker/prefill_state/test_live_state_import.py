"""Actual prefix importer lifetime and nonidentity current-page writes.

Eligibility and EP agreement precede this helper in production. These tests
exercise the import boundary, not model execution or a simulated EP decision.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.v1.worker.prefill_state.live_d_worker import LiveDWorker
from vllm_iluvatar.v1.worker.prefill_state.source_state import validate_fields


def import_fixture(device="cpu", *, n=4098, generation=1):
    fields, parts, expected = [], [], {}

    def add(name, kind, shape, dtype, layer=None):
        value = torch.arange(torch.tensor(shape).prod().item()) % 31 + generation
        value = value.reshape(shape).to(dtype)
        raw = value.contiguous().view(torch.uint8).flatten()
        field = dict(
            name=name,
            kind=kind,
            shape=list(shape),
            dtype=str(dtype),
            offset_bytes=sum(part.numel() for part in parts),
            byte_count=raw.numel(),
        )
        if layer is not None:
            field["layer"] = layer
        fields.append(field)
        parts.append(raw)
        expected[name] = value

    for layer in range(21):
        add(f"swa{layer}", "SWA", (128, 512), torch.bfloat16, layer)
    for layer in (2, 8, 14):
        add(f"ring{layer}", "ring", (16, 1024), torch.float32, layer)
    add("HC", "boundary21", (128, 4, 5120), torch.bfloat16)
    add("pre_mix", "boundary21", (128, 4), torch.float32)
    add("topk20", "boundary21", (128, 512), torch.int32)
    add("candidates20", "boundary21", (128, 2048), torch.int32)
    host = torch.cat(parts)
    validate_fields(fields, host.numel())
    if device == "cuda":
        host = host.pin_memory()
    packet = dict(metadata=dict(fields=fields), host=host)
    worker = LiveDWorker.__new__(LiveDWorker)
    worker._p21_lock = threading.Lock()
    worker._p21_host_inflight = []
    worker._p21_packets = {"request": packet}
    worker._p21_cache_sources = []
    blocks = []
    for layer in range(21):
        table = [0] * ((n + 63) // 64)
        lo, hi = (n - 128) // 64, (n - 1) // 64
        table[lo : hi + 1] = [4, 1, 3][: hi - lo + 1]
        blocks.append(table)
        worker._p21_cache_sources.append(
            dict(
                name=f"swa{layer}",
                group=len(blocks) - 1,
                layer=layer,
                kind="SWA",
                cache=torch.full((6, 64, 512), -9, dtype=torch.bfloat16, device=device),
            )
        )
    for layer in (2, 8, 14):
        blocks.append([2])
        worker._p21_cache_sources.append(
            dict(
                name=f"ring{layer}",
                group=len(blocks) - 1,
                layer=layer,
                kind="ring",
                cache=torch.full((4, 16, 1024), -9, dtype=torch.float32, device=device),
            )
        )
    worker._p21_target = SimpleNamespace(
        topk_indices_buffer=torch.full(
            (134, 512), -9, dtype=torch.int32, device=device
        ),
        candidate_idxs_buffer=torch.full(
            (134, 2048), -9, dtype=torch.int32, device=device
        ),
    )
    hidden = torch.full((134, 4, 5120), -9, dtype=torch.bfloat16, device=device)
    pre = torch.full((134, 4), -9, dtype=torch.float32, device=device)
    state = dict(id="request", N=n, blocks=blocks, generation=generation)
    return worker, state, packet, expected, hidden, pre


def check_import(worker, state, expected, hidden, pre):
    for source in worker._p21_cache_sources:
        cache = source["cache"].cpu()
        want = torch.full_like(cache, -9)
        table = state["blocks"][source["group"]]
        data = expected[source["name"]]
        if source["kind"] == "SWA":
            for offset in range(128):
                logical, row = divmod(state["N"] - 128 + offset, 64)
                want[table[logical], row] = data[offset]
        else:
            want[table[0]] = data
        assert torch.equal(cache, want)  # Includes every untouched guard row.
    for actual, name in (
        (hidden, "HC"),
        (pre, "pre_mix"),
        (worker._p21_target.topk_indices_buffer, "topk20"),
        (worker._p21_target.candidate_idxs_buffer, "candidates20"),
    ):
        assert torch.all(actual[:6].cpu() == -9)
        assert torch.equal(actual[6:].cpu(), expected[name])
    assert state["imported"] and not worker._p21_packets
    assert len(worker._p21_host_inflight) == 1
    assert worker._p21_host_inflight[0]["event"] is not None


def mock_copy_stream(monkeypatch, worker):
    calls = []

    def wait(event):
        assert len(worker._p21_host_inflight) == 1
        assert worker._p21_host_inflight[0]["event"] is None
        calls.append(("wait", event))

    class Event:
        def record(self):
            calls.append(("record", self))

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(wait_event=wait)
    )
    monkeypatch.setattr(torch.cuda, "Event", Event)
    return calls


@pytest.mark.parametrize("n", [4096, 4098])
def test_import_preserves_old_rows_and_writes_current_nonidentity_pages(monkeypatch, n):
    worker, state, packet, expected, hidden, pre = import_fixture(n=n)
    calls = mock_copy_stream(monkeypatch, worker)
    receive_event = object()
    worker._p21_import_states(
        [(1, state, packet, dict(event=receive_event))], [0, 6, 134], hidden, pre
    )
    check_import(worker, state, expected, hidden, pre)
    assert calls[0] == ("wait", receive_event)
    assert [call[0] for call in calls] == ["wait", "record"]


def test_partial_import_keeps_pinned_packet_alive(monkeypatch):
    worker, state, packet, _, hidden, pre = import_fixture()
    mock_copy_stream(monkeypatch, worker)
    # A host-side failure after some copies were submitted must retain ownership.
    packet["metadata"]["fields"] = [
        f for f in packet["metadata"]["fields"] if f["name"] != "HC"
    ]
    with pytest.raises(KeyError, match="HC"):
        worker._p21_import_states(
            [(1, state, packet, dict(event=object()))], [0, 6, 134], hidden, pre
        )
    assert not state.get("imported")
    assert worker._p21_packets["request"] is packet
    assert worker._p21_host_inflight == [dict(event=None, packet=packet)]


@pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1",
    reason="requires an explicit permitted GPU lease",
)
def test_cuda_import_waits_for_receive_and_preserves_current_pages():
    assert torch.cuda.device_count() == 1
    assert torch.cuda.mem_get_info()[0] >= (256 << 20) + (1 << 30)
    worker, state, packet, expected, hidden, pre = import_fixture("cuda")
    receive_stream = torch.cuda.Stream()
    with torch.cuda.stream(receive_stream):
        ready = torch.cuda.Event()
        ready.record()
    worker._p21_import_states(
        [(1, state, packet, dict(event=ready))], [0, 6, 134], hidden, pre
    )
    worker._p21_host_inflight[0]["event"].synchronize()
    check_import(worker, state, expected, hidden, pre)
    assert torch.cuda.mem_get_info()[0] >= (1 << 30)
