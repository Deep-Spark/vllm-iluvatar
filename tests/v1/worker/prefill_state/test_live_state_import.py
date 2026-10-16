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
from vllm_iluvatar.v1.worker.prefill_state.source_state import (
    validate_fields, replay_layout, main_tail_layout,
)
from vllm_iluvatar.v1.worker.prefill_state.cache_rows import main_boundary_sources, indexer_row_views
from vllm_iluvatar.v1.worker.prefill_state.host_source_worker import HostSourceWorker


def import_fixture(device="cpu", *, n=4098, generation=1, main_format="none"):
    hit, start, rows = replay_layout(n)
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
        add(f"swa{layer}", "SWA", (rows, 512), torch.bfloat16, layer)
    for layer in (2, 8, 14):
        add(f"ring{layer}", "ring", (16, 1024), torch.float32, layer)
    add("HC", "boundary21", (rows, 4, 5120), torch.bfloat16)
    add("pre_mix", "boundary21", (rows, 4), torch.float32)
    add("topk20", "boundary21", (rows, 512), torch.int32)
    add("candidates20", "boundary21", (rows, 2048), torch.int32)
    dtypes = {"BF16": torch.bfloat16, "U8": torch.uint8, "FP16": torch.float16}
    if n % 2:
        for kind, (shape, dtype) in main_tail_layout(main_format).items():
            add(kind, kind, shape, dtypes[dtype], 20)
        add("indexer_tail", "indexer_tail", (132,), torch.uint8, 20)
    host = torch.cat(parts)
    validate_fields(fields, host.numel(), n, main_format)
    if device == "cuda":
        host = host.pin_memory()
    packet = dict(metadata=dict(fields=fields, H=hit, S=start), host=host)
    worker = LiveDWorker.__new__(LiveDWorker)
    worker._p21_lock = threading.Lock()
    worker._p21_host_inflight = []
    worker._p21_packets = {"request": packet}
    worker._p21_cache_sources = []
    blocks = []
    for layer in range(21):
        table = [0] * ((n + 63) // 64)
        lo, hi = start // 64, (n - 1) // 64
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
    # Both paired main views use identical nonidentity page tables; the indexer
    # has independent64-row SoA pages, unlike main I8R128/I4R256 pages.
    layout = main_tail_layout(main_format)
    page_rows = {"none": 64, "i8g64": 128, "i4g32": 256}[main_format]
    main_group = len(blocks)
    blocks.append([3] * ((n + page_rows - 1) // page_rows))
    for kind, (shape, dtype) in layout.items():
        width=shape[-1]
        owner=torch.full((5,page_rows*width+64),9,dtype=dtypes[dtype],device=device)
        cache=owner.as_strided((5,page_rows,width),(page_rows*width+64,width,1))
        worker._p21_cache_sources.append(dict(
            name=kind, group=main_group, layer=20, kind=kind, cache=cache,
            padding=owner[:,page_rows*width:]))
    blocks.append([2] * ((n + 63) // 64))
    worker._p21_cache_sources.append(dict(
        name="indexer_tail",group=len(blocks)-1,layer=20,kind="indexer_tail",
        cache=torch.full((5,64,132),9,dtype=torch.uint8,device=device)))
    worker._p21_target = SimpleNamespace(
        topk_indices_buffer=torch.full(
            (rows + 6, 512), -9, dtype=torch.int32, device=device
        ),
        candidate_idxs_buffer=torch.full(
            (rows + 6, 2048), -9, dtype=torch.int32, device=device
        ),
    )
    hidden = torch.full((rows + 6, 4, 5120), -9, dtype=torch.bfloat16, device=device)
    pre = torch.full((rows + 6, 4), -9, dtype=torch.float32, device=device)
    state = dict(id="request", N=n, H=hit, S=start, blocks=blocks, generation=generation)
    return worker, state, packet, expected, hidden, pre


def check_import(worker, state, expected, hidden, pre):
    for source in worker._p21_cache_sources:
        cache = source["cache"].cpu()
        want = torch.full_like(cache, -9 if source["kind"] in ("SWA","ring") else 9)
        table = state["blocks"][source["group"]]
        data = expected.get(source["name"])
        if source["kind"] == "SWA":
            for offset in range(state["N"] - state["S"]):
                logical, row = divmod(state["S"] + offset, 64)
                want[table[logical], row] = data[offset]
        elif source["kind"] == "ring":
            want[table[0]] = data
        elif state["N"] % 2:
            logical, row = divmod(state["N"]-1, cache.shape[1])
            if source["kind"] == "indexer_tail":
                offset=0
                for part in indexer_row_views(want, table[logical], row):
                    part.copy_(data[offset:offset+part.numel()]);offset+=part.numel()
            else:
                want[table[logical],row]=data[0]
        assert torch.equal(cache, want)  # Includes every untouched guard row.
        if "padding" in source:
            assert torch.all(source["padding"].cpu()==9)
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


@pytest.mark.parametrize("main_format", ["none", "i8g64", "i4g32"])
@pytest.mark.parametrize("n", [128,129,255,256,257,4099,1048575,1048576])
def test_p_export_d_import_preserve_boundary_payload_scale_and_neighbors(monkeypatch, main_format, n):
    worker,state,packet,expected,hidden,pre=import_fixture(n=n,main_format=main_format)
    mock_copy_stream(monkeypatch,worker)
    rows=n-state['S']
    worker._p21_import_states([(1,state,packet,dict(event=object()))],[0,6,6+rows],hidden,pre)
    check_import(worker,state,expected,hidden,pre)
    # Feed the exact destination views back through the real P exporter. This
    # checks page addressing/SoA and packed main rows, with no mock copy kernel.
    producer=HostSourceWorker.__new__(HostSourceWorker)
    producer._src_specs=[dict(s,R=s['cache'].shape[1],ratio=1) for s in worker._p21_cache_sources]
    pool={}
    for source in producer._src_specs:
        kind=source['kind'];cache=source['cache']
        shape=((129,512) if kind=='SWA' else (16,1024) if kind=='ring'
               else (132,) if kind=='indexer_tail' else (1,cache.shape[-1]))
        dtype=torch.uint8 if kind=='indexer_tail' else cache.dtype
        pool[source['name']]=torch.empty(shape,dtype=dtype)
    records=producer._copy_caches(dict(state,pool=pool))
    for rec in records:
        value=pool[rec['name']]
        if rec['kind']=='SWA':value=value[:rows]
        assert torch.equal(value,expected[rec['name']])
    main=next(s['cache'] for s in producer._src_specs if s['kind']=='main_tail')
    scale=next((s['cache'] for s in producer._src_specs if s['kind']=='main_tail_scale'),None)
    layer=SimpleNamespace(kv_cache=main,kv_cache_scales=scale)
    specs=main_boundary_sources(layer,'model.layers.20.attn',24,main_format)
    assert len(specs)==1+(main_format!='none')
    if scale is not None:
        layer.kv_cache_scales=scale[:,:-1]
        with pytest.raises(ValueError,match='paired main'):
            main_boundary_sources(layer,'model.layers.20.attn',24,main_format)
