"""Bounded actual family readers/mappers on an explicitly leased GPU."""

import os
from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek import (
    quantized_sparse_mla as facade,
)
from ixformer.inference.functions.deepseek import (
    quantized_sparse_mla_prefill as prefill_facade,
)
from ixformer.inference.functions.deepseek.quantized_kv_cache import quantize_paged_kv
from ixformer.inference.functions.deepseek.quantized_kv_format import (
    I4G32,
    I8G64,
    dequantize_reference,
    quantize_reference,
)
from ixformer.inference.functions.deepseek.quantized_sparse_mla import (
    QuantizedSparseMlaAffineView,
)
from vllm.forward_context import override_forward_context

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    WORKSPACE_BYTES,
    WorkspacePool,
)
from vllm_iluvatar.models.deepseek_v41.compressed_kv_restore import (
    CompressedFamilySelection,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1",
    reason="requires an explicitly leased GPU and KV_COMPRESSED_KV_GPU_TESTS=1",
)


def oracle(q, window, rows, logical, valid, positions, ratio, page_rows):
    output = torch.empty_like(q, device="cpu")
    q, window = q.cpu().float(), window.cpu().float()
    valid, logical, positions = valid.cpu(), logical.cpu(), positions.cpu()
    for token in range(q.shape[0]):
        main = []
        if valid[token]:
            for slot in logical[token].tolist():
                if 0 <= slot < min(2 * page_rows, (int(positions[token]) + 1) // ratio):
                    main.append(
                        rows[{page_rows - 1: 0, page_rows: 1, page_rows + 1: 2}[slot]]
                    )
        main = torch.stack(main) if main else torch.empty(0, 512)
        kv = torch.cat((window if valid[token] else window[:0], main)).float()
        logits = q[token] @ kv.T * 512**-0.5
        maximum = (
            torch.maximum(logits.amax(-1), torch.zeros(32))
            if kv.shape[0]
            else torch.zeros(32)
        )
        probability = torch.exp(logits - maximum[:, None])
        denominator = probability.sum(-1) + torch.exp(-maximum)
        output[token] = ((probability @ kv) / denominator[:, None]).to(torch.bfloat16)
    return output


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize(
    "owner,ratio,forwarder",
    [(2, 2, False), (8, 2, False), (14, 2, False), (20, 1, False), (2, 2, True)],
)
def test_actual_family_slices_reuse_views_and_reprepare_across_graphs(
    fmt, owner, ratio, forwarder, monkeypatch, record_property
):
    assert torch.cuda.device_count() == 1
    free, _ = torch.cuda.mem_get_info()
    assert free - (WORKSPACE_BYTES + 64 * 1024**2) >= 1024**3
    torch.manual_seed(1014 + owner)
    rows, width, groups = (
        fmt.rows_per_page(512),
        fmt.payload_bytes(512),
        fmt.scale_groups(512),
    )
    raw = torch.full((6 * 65536,), 0xA5, dtype=torch.uint8, device="cuda")
    scale_raw = torch.full(
        (6 * rows * groups * 2,), 0xFF, dtype=torch.uint8, device="cuda"
    )
    cache = raw.as_strided((3, rows, width), (2 * rows * width, width, 1), rows * width)
    scales = scale_raw.view(torch.float16).as_strided(
        (3, rows, groups), (2 * rows * groups, groups, 1), rows * groups
    )
    source = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(source)
    source.compressed_kv_format, source.kv_cache, source.kv_cache_scales = (
        fmt,
        cache,
        scales,
    )
    prefix = "pp.forwarder.5" if forwarder else f"owner{owner}"
    pool = WorkspacePool(torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"))
    metadata_guard = pool.reserve_metadata(2048)
    metadata_guard.fill_(0xA5)
    slots = torch.tensor(
        [3 * rows - 1, rows, rows + 1], dtype=torch.int64, device="cuda"
    )
    semantic = torch.randn(3, 512, dtype=torch.bfloat16, device="cuda")
    logical = torch.tensor(
        [rows - 1, -1, rows, rows + 1, 2 * rows, -1], dtype=torch.int32, device="cuda"
    ).repeat(5, 1)
    # Like the actual indexer, selection values already obey causal bounds.
    # The frozen mapper caps the number of scanned columns, not their values.
    logical[[0, 2], 3] = -1
    valid = torch.tensor([True, False, True, True, True], device="cuda")
    reqs = torch.tensor([0, -999, 0, 0, 0], dtype=torch.int32, device="cuda")
    positions = torch.tensor(
        [
            ratio * (rows + 1) - 1,
            0,
            ratio * (rows + 1) - 1,
            ratio * (rows + 2) - 1,
            ratio * (rows + 2),
        ],
        dtype=torch.int64,
        device="cuda",
    )
    table = torch.tensor([[2, 1]], dtype=torch.int32, device="cuda")
    swa_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda").repeat(5, 1)
    swa_lens = torch.tensor([2, 0, 2, 2, 2], dtype=torch.int32, device="cuda")
    metadata = {prefix: SimpleNamespace(block_size=rows * ratio, block_table=table)}
    ctx = SimpleNamespace(attn_metadata=metadata)
    layers, queries, outputs = [], [], []
    for index in range(3):
        layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
        torch.nn.Module.__init__(layer)
        layer.compressed_kv_format, layer.compress_ratio = fmt, ratio
        layer.is_kv_source = False
        layer.is_kv_forwarder = forwarder and index == 0
        if layer.is_kv_forwarder:
            layer.kv_cache, layer.kv_cache_scales = (
                cache.view_as(cache),
                scales.view_as(scales),
            )
        layer.compressed_cache_prefix = prefix
        layer._static_forward_context = {prefix: source}
        layer.compressed_kv_workspace = pool
        layer.index_source_layer_id, layer.topk_indices_buffer = owner, logical
        layer.scale, layer.attn_sink = 512**-0.5, torch.zeros(32, device="cuda")
        swa_prefix = f"swa{index}"
        layer.swa_cache_layer = SimpleNamespace(
            prefix=swa_prefix,
            kv_cache=torch.randn(3, 64, 512, dtype=torch.bfloat16, device="cuda"),
        )
        metadata[swa_prefix] = SimpleNamespace(
            num_decodes=2,
            num_prefills=1,
            num_decode_tokens=2,
            num_prefill_tokens=3,
            is_valid_token=valid,
            token_to_req_indices=reqs,
            decode_swa_indices=swa_indices[:2],
            decode_swa_lens=swa_lens[:2],
            prefill_swa_indices=swa_indices[2:],
            prefill_swa_lens=swa_lens[2:],
        )
        layers.append(layer)
        queries.append(torch.randn(5, 32, 512, dtype=torch.bfloat16, device="cuda"))
        outputs.append(torch.empty_like(queries[-1]))
    empty_kv = torch.empty(0, device="cuda")
    mapper = torch.ops._C.v41_global_topk_indices_and_lens
    prefix_indices = torch.arange(6, dtype=torch.int32, device="cuda").repeat(5, 1)
    prefix_positions = torch.tensor(
        [ratio - 1, 0, 2 * ratio - 1, 5 * ratio - 1, 6 * ratio - 1],
        dtype=torch.int64,
        device="cuda",
    )
    prefix_full, prefix_lengths = mapper(
        prefix_indices, reqs, table, rows, valid, prefix_positions, ratio, 0, False
    )
    assert prefix_lengths.cpu().tolist() == [1, 0, 2, 5, 6]
    for token_slice in (slice(2, 5), slice(0, 2)):
        separate, lengths = mapper(
            prefix_indices[token_slice],
            reqs[token_slice],
            table,
            rows,
            valid[token_slice],
            prefix_positions[token_slice],
            ratio,
            0,
            False,
        )
        assert torch.equal(prefix_full[token_slice], separate)
        assert torch.equal(prefix_lengths[token_slice], lengths)
    native_prepare = facade.prepare_quantized_sparse_mla_restore_plan
    native_reader = facade.quantized_sparse_mla
    native_prefill = prefill_facade.quantized_sparse_mla_prefill
    assert prefill_facade.quantized_sparse_mla is native_reader
    mappings, plans, affine_views = [], [], []

    def mapped(*args):
        result = mapper(*args)
        mappings.append(result)
        return result

    def prepared(*args, **kwargs):
        plans.append(args[4])
        return native_prepare(*args, **kwargs)

    def guarded_prefill(*args, **kwargs):
        view = kwargs["workspace"]
        assert isinstance(view, QuantizedSparseMlaAffineView)
        assert view.rows is plans[-1].rows and view.scratch is plans[-1].scratch
        assert view.rows.untyped_storage().data_ptr() == pool.storage.data_ptr()
        affine_views.append(view)
        return native_prefill(*args, **kwargs)

    def guarded_reader(q, *args, **kwargs):
        # Only old decode enters this module directly; prefill has its own
        # facade. Its native scratch must leave the retained family intact.
        assert q.shape[0] == 2
        plan = plans[-1]
        assert isinstance(kwargs["workspace"], torch.Tensor)
        assert kwargs["workspace"].data_ptr() == plan.scratch.data_ptr()
        assert kwargs["workspace"].numel() == plan.scratch.numel()
        check = not torch.cuda.is_current_stream_capturing()
        if check:
            before_rows, before_map = plan.rows.clone(), plan.row_map.clone()
        result = native_reader(q, *args, **kwargs)
        if check:
            torch.testing.assert_close(plan.rows, before_rows, rtol=0, atol=0)
            torch.testing.assert_close(plan.row_map, before_map, rtol=0, atol=0)
        return result

    monkeypatch.setattr(torch.ops._C, "v41_global_topk_indices_and_lens", mapped)
    monkeypatch.setattr(facade, "prepare_quantized_sparse_mla_restore_plan", prepared)
    monkeypatch.setattr(facade, "quantized_sparse_mla", guarded_reader)
    monkeypatch.setattr(prefill_facade, "quantized_sparse_mla_prefill", guarded_prefill)

    def publish():
        quantize_paged_kv(semantic, slots, cache, scales, format=fmt)
        pool.main_cache_written()

    def chain():
        with override_forward_context(ctx), pool.forward_scope():
            for layer, q, output in zip(layers, queries, outputs):
                layer.forward_mqa(q, empty_kv, positions, output)

    def check():
        payload, scale = quantize_reference(semantic.cpu().float().numpy(), format=fmt)
        restored = (
            torch.from_numpy(
                dequantize_reference(payload, scale, head_dim=512, format=fmt)
            )
            .to(torch.bfloat16)
            .float()
        )
        for layer, q, output in zip(layers, queries, outputs):
            expected = oracle(
                q,
                layer.swa_cache_layer.kv_cache[0, :2],
                restored,
                logical,
                valid,
                positions,
                ratio,
                rows,
            )
            torch.testing.assert_close(
                output.cpu().float(),
                expected.float(),
                rtol=2 * torch.finfo(torch.bfloat16).eps,
                atol=2 * torch.finfo(torch.bfloat16).eps,
            )

    publish()
    chain()
    torch.cuda.synchronize()
    assert len(mappings) == len(plans) == 1
    assert len(affine_views) == 3 and all(
        view is affine_views[0] for view in affine_views
    )
    full_indices, full_lengths = mappings[0]
    # A fresh host consumption view must allocate no GPU storage.
    allocation_before = torch.cuda.memory_allocated()
    selection = CompressedFamilySelection(full_indices, full_lengths, plans[0], 0)
    view = selection.for_queries(slice(2, 5))[2]
    allocation_delta = torch.cuda.memory_allocated() - allocation_before
    assert isinstance(view, QuantizedSparseMlaAffineView) and allocation_delta == 0
    record_property("affine_view_allocation_bytes", allocation_delta)
    del selection, view
    for token_slice in (slice(2, 5), slice(0, 2)):
        separate, lengths = mapper(
            logical[token_slice],
            reqs[token_slice],
            table,
            rows,
            valid[token_slice],
            positions[token_slice],
            ratio,
            0,
            False,
        )
        assert torch.equal(full_indices[token_slice], separate)
        assert torch.equal(full_lengths[token_slice], lengths)
    assert (plans[0].row_map[:2] == -1).all().item()
    assert plans[0].budget_bytes == pool.tensor.numel() == WORKSPACE_BYTES - 2048
    check()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        chain()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    for graph in graphs:
        with torch.cuda.graph(graph):
            chain()
    assert len(plans) == 4
    assert len(affine_views) == 12
    assert all(affine_views[3 * i] is not affine_views[3 * i + 3] for i in range(3))
    assert len({plan.rows.untyped_storage().data_ptr() for plan in plans}) == 1
    # Changed payload/scales and selection must refresh the captured GPU
    # preparation despite unchanged host keys and zero Python replay calls.
    semantic.mul_(0.75)
    logical[:, 2:4] = logical[:, 2:4].flip(-1)
    for q in queries:
        q.mul_(0.5)
    publish()
    ready = torch.cuda.Event()
    ready.record()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    count_before = len(plans)
    for stream, graph in zip(streams, graphs):
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            with pool.acquire():
                graph.replay()
    streams[-1].synchronize()
    assert len(plans) == count_before
    assert (
        len(affine_views) == 12
    )  # Replay performs GPU preparation without host views.
    check()
    assert torch.all(metadata_guard == 0xA5).item()
    for page in (0, 2, 4):
        assert torch.all(raw[page * 65536 : (page + 1) * 65536] == 0xA5).item()
        assert torch.all(
            scale_raw[page * rows * groups * 2 : (page + 1) * rows * groups * 2] == 0xFF
        ).item()
    assert not pool._leased and pool.restore._selection is None
    record_property("affine_host_reader_calls", len(affine_views))
    record_property("families_prepared", len(plans))
    assert torch.cuda.mem_get_info()[0] >= 1024**3
