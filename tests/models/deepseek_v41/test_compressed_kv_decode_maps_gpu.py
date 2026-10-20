"""Decode map reuse and opt-in selected rows across changing owned graphs."""

import os
from types import SimpleNamespace

import pytest
import torch
from ixformer.inference.functions.deepseek import quantized_sparse_mla as facade
from ixformer.inference.functions.deepseek.quantized_kv_cache import quantize_paged_kv
from ixformer.inference.functions.deepseek.quantized_kv_format import (
    I4G32,
    I8G64,
    dequantize_reference,
    quantize_reference,
)
from test_compressed_kv_restore_gpu import oracle
from vllm.forward_context import override_forward_context

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention
from vllm_iluvatar.models.deepseek_v41.compressed_kv import (
    WORKSPACE_BYTES,
    WorkspacePool,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KV_COMPRESSED_KV_GPU_TESTS") != "1",
    reason="requires an explicitly leased GPU and KV_COMPRESSED_KV_GPU_TESTS=1",
)


@pytest.mark.parametrize("fmt", [I8G64, I4G32])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("forwarder", [False, True])
@pytest.mark.parametrize("restore_decode", [False, True])
def test_decode_capture_reuses_map_with_changed_inputs_pages_and_aliases(
    fmt, ratio, forwarder, restore_decode, monkeypatch, record_property
):
    assert torch.cuda.device_count() == 1
    assert torch.cuda.mem_get_info()[0] - (WORKSPACE_BYTES + 64 * 1024**2) >= 1024**3
    torch.manual_seed(1014 + ratio)
    rows, width, groups = (
        fmt.rows_per_page(512),
        fmt.payload_bytes(512),
        fmt.scale_groups(512),
    )
    raw = torch.full((8 * 65536,), 0xA5, dtype=torch.uint8, device="cuda")
    scale_raw = torch.full(
        (8 * rows * groups * 2,), 0xFF, dtype=torch.uint8, device="cuda"
    )
    cache = raw.as_strided((4, rows, width), (2 * rows * width, width, 1), rows * width)
    scales = scale_raw.view(torch.float16).as_strided(
        (4, rows, groups), (2 * rows * groups, groups, 1), rows * groups
    )
    source = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(source)
    source.compressed_kv_format, source.kv_cache, source.kv_cache_scales = (
        fmt,
        cache,
        scales,
    )
    prefix = "pp.forwarder.5" if forwarder else f"owner{20 if ratio == 1 else 2}"
    pool = WorkspacePool(torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"))
    metadata_guard = pool.reserve_metadata(2048)
    metadata_guard.fill_(0xA5)
    semantic = torch.randn(3, 512, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor(
        [3 * rows - 1, rows, rows + 1], dtype=torch.int64, device="cuda"
    )
    logical = torch.tensor(
        [rows - 1, -1, rows, rows + 1, 2 * rows, -1], dtype=torch.int32, device="cuda"
    ).repeat(5, 1)
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
        layer.compressed_decode_restore = restore_decode
        layer.is_kv_source, layer.is_kv_forwarder = False, forwarder and index == 0
        if layer.is_kv_forwarder:
            layer.kv_cache, layer.kv_cache_scales = (
                cache.view_as(cache),
                scales.view_as(scales),
            )
        layer.compressed_cache_prefix = prefix
        layer._static_forward_context = {prefix: source}
        layer.compressed_kv_workspace = pool
        layer.index_source_layer_id, layer.topk_indices_buffer = (
            20 if ratio == 1 else 2,
            logical,
        )
        layer.scale, layer.attn_sink = 512**-0.5, torch.zeros(32, device="cuda")
        swa_prefix = f"swa{index}"
        layer.swa_cache_layer = SimpleNamespace(
            prefix=swa_prefix,
            kv_cache=torch.randn(3, 64, 512, dtype=torch.bfloat16, device="cuda"),
        )
        metadata[swa_prefix] = SimpleNamespace(
            num_decodes=1,
            num_prefills=0,
            num_decode_tokens=5,
            num_prefill_tokens=0,
            is_valid_token=valid.clone() if restore_decode else valid,
            token_to_req_indices=reqs,
            decode_swa_indices=swa_indices,
            decode_swa_lens=swa_lens.clone() if restore_decode else swa_lens,
        )
        layers.append(layer)
        queries.append(torch.randn(5, 32, 512, dtype=torch.bfloat16, device="cuda"))
        outputs.append(torch.empty_like(queries[-1]))
    empty_kv = torch.empty(0, device="cuda")
    mapper = torch.ops._C.v41_global_topk_indices_and_lens
    mappings = []

    def mapped(*args):
        result = mapper(*args)
        mappings.append(result)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Pure decode capture must not size, create or prepare restored rows"
        )

    monkeypatch.setattr(torch.ops._C, "v41_global_topk_indices_and_lens", mapped)
    prepares = []
    for name in (
        "quantized_sparse_mla_scratch_bytes",
        "make_quantized_sparse_mla_restore_plan",
        "prepare_quantized_sparse_mla_restore_plan",
    ):
        if restore_decode:
            if name == "prepare_quantized_sparse_mla_restore_plan":
                original_prepare = getattr(facade, name)

                def prepared(*args, **kwargs):
                    prepares.append(args[4])
                    return original_prepare(*args, **kwargs)

                monkeypatch.setattr(facade, name, prepared)
        else:
            monkeypatch.setattr(facade, name, forbidden)

    def publish():
        quantize_paged_kv(semantic, slots, cache, scales, format=fmt)
        pool.main_cache_written()
        for index, layer in enumerate(layers):
            local = metadata[layer.swa_cache_layer.prefix]
            if restore_decode:
                local.is_valid_token.copy_(valid)
                # Distinct buffers model the real builders. The first
                # consumer masks a row that a later one must still consume.
                if index == 0:
                    local.is_valid_token[0] = False
                local.decode_swa_lens.copy_(swa_lens)
                local.decode_swa_lens.masked_fill_(~local.is_valid_token, 0)

    def chain():
        with override_forward_context(ctx), pool.forward_scope():
            for layer, q, output in zip(layers, queries, outputs):
                layer.forward_mqa(q, empty_kv, positions, output)

    oracle_errors = []

    def check():
        payload, scale = quantize_reference(semantic.cpu().float().numpy(), format=fmt)
        decoded = (
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
                decoded,
                logical,
                metadata[layer.swa_cache_layer.prefix].is_valid_token,
                positions,
                ratio,
                rows,
            )
            # Match the established norm-one reader oracle tolerance: native
            # QK/PV uses BF16 operands while this CPU softmax/PV is FP32.
            # The capture path additionally matches its eager packed output
            # bit for bit after changes below.
            observed = output.cpu().float()
            delta = observed.double() - expected.double()
            oracle_errors.append(
                (
                    delta.abs().max().item(),
                    (
                        delta.square().sum()
                        / expected.double().square().sum().clamp_min(1e-30)
                    )
                    .sqrt()
                    .item(),
                    observed.numel(),
                )
            )
            torch.testing.assert_close(
                observed,
                expected.float(),
                rtol=2 * torch.finfo(torch.bfloat16).eps,
                atol=2 * torch.finfo(torch.bfloat16).eps,
            )

    publish()
    chain()
    torch.cuda.synchronize()
    assert len(mappings) == (1 if restore_decode else 3) and not pool.capturing
    check()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        chain()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph in graphs:
        before = len(mappings)
        with torch.cuda.graph(graph):
            chain()
        assert len(mappings) - before == 1
    assert not pool.capturing
    semantic.mul_(0.75)
    logical[:, 2:4] = logical[:, 2:4].flip(-1)
    valid[3], reqs[3], swa_lens[3] = False, -999, 0
    for q in queries:
        q.mul_(0.5)
    publish()
    # Recycle a previously unused physical page and poison its old location.
    # The same block-table pointer and shape now refer to a different page.
    # Physical0 remains the mapper's reserved null block. Recycle nonzero3.
    cache[3].copy_(cache[2])
    scales[3].copy_(scales[2])
    table[0, 0] = 3
    cache[2].zero_()
    scales[2].fill_(2**-24)
    pool.main_cache_written()
    chain()
    expected_outputs = [output.clone() for output in outputs]
    ready = torch.cuda.Event()
    ready.record()
    count_before = len(mappings)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream, graph in zip(streams, graphs):
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            with pool.acquire():
                graph.replay()
        stream.synchronize()
        for output, expected in zip(outputs, expected_outputs):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        check()
    assert len(mappings) == count_before
    assert len(prepares) == (len(mappings) if restore_decode else 0)
    assert torch.all(metadata_guard == 0xA5).item()
    for page in (0, 1, 2, 4, 6):
        assert torch.all(raw[page * 65536 : (page + 1) * 65536] == 0xA5).item()
        assert torch.all(
            scale_raw[page * rows * groups * 2 : (page + 1) * rows * groups * 2] == 0xFF
        ).item()
    assert not pool._leased and not pool.capturing and pool.restore._selection is None
    assert pool.storage.numel() == WORKSPACE_BYTES
    assert torch.cuda.mem_get_info()[0] >= 1024**3
    record_property("oracle_max_abs", max(row[0] for row in oracle_errors))
    record_property("oracle_max_layer_nrmse", max(row[1] for row in oracle_errors))
    record_property("oracle_elements", sum(row[2] for row in oracle_errors))
