from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention


def test_large_kv_span_allows_both_a64_fp8_layouts():
    from vllm_iluvatar.v1.worker.kv_layout_report import _A64_SAFE_CACHE_DTYPES

    assert {"fp8_ds_mla", "fp8_e4m3", "bfloat16"} == _A64_SAFE_CACHE_DTYPES


def test_dspark_context_insert_supports_plain_e4m3():
    from ixformer.inference.functions import deepseekv41 as formal_ops

    from vllm_iluvatar.models.deepseek_v41.dspark import _insert_context_kv

    gen = torch.Generator(device="cuda").manual_seed(2026)
    kv = torch.randn((3, 512), generator=gen, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(3, device="cuda", dtype=torch.int64)
    slots = torch.tensor([4, 5, -1], device="cuda", dtype=torch.int64)
    phase = torch.randn((8, 32), generator=gen, device="cuda")
    cos_sin = torch.cat((phase.cos(), phase.sin()), dim=-1)
    cache = torch.zeros((2, 4, 512), device="cuda", dtype=torch.float8_e4m3fn)
    attn = SimpleNamespace(
        kv_cache_dtype="fp8_e4m3",
        kv_scale=1.0,
        swa_cache_layer=SimpleNamespace(kv_cache=cache),
        _attn_out=torch.zeros((3, 4, 512), device="cuda", dtype=torch.bfloat16),
        cos_sin=cos_sin,
    )
    ref_kv = kv.clone()
    ref_cache = cache.clone()
    ref_q = torch.zeros_like(attn._attn_out)

    _insert_context_kv(attn, kv, positions, slots)
    formal_ops.dsv41_qkv_rope_fp8_pt_insert(
        ref_q,
        ref_cache,
        ref_kv,
        slots,
        positions,
        cos_sin,
        kv_scale=1.0,
        backend="torch",
    )

    assert torch.equal(cache.view(torch.uint8), ref_cache.view(torch.uint8))
    assert torch.equal(kv, ref_kv)
    assert torch.all(cache[0] == 0)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_compressed_plain_insert_and_pipeline_copy_replay(ratio, dtype):
    """Compression boundaries, RoPE position, saturation and PP bytes stay correct."""
    from vllm_iluvatar.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        move_rows_to_paged_cache,
        rope_quant_insert,
    )

    latent = torch.full((6, 512), 2, dtype=torch.bfloat16, device="cuda")
    latent[:, 0] = 1024
    latent[:, 1] = -1024
    latent[:, 448::2] = 4
    pos = torch.arange(6, dtype=torch.int64, device="cuda")
    slots = torch.tensor([4, 5, -1, 6, 7, 8], device="cuda")
    # Even positions rotate 90 degrees, odd positions are identity. CR2 must
    # use the even start position of its group rather than its odd closing one.
    cs = torch.zeros((8, 64), device="cuda")
    cs[::2, 32:] = 1
    cs[1::2, :32] = 1
    cache = torch.zeros((3, 4, 512), dtype=dtype, device="cuda")
    replica = torch.zeros_like(cache)
    rows = torch.zeros((6, 512), dtype=dtype, device="cuda")

    def run():
        rope_quant_insert(latent, pos, cs, cache, slots, ratio, row_out=rows)
        move_rows_to_paged_cache(rows, replica, slots, pos, ratio)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for delta in (0, 2):
        latent.add_(delta)
        cache.zero_()
        replica.zero_()
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.zeros_like(cache).float()
        values = latent.float().clone()
        angles = (pos // ratio * ratio) % 2 == 0
        values[angles, 448::2] = -latent[angles, 449::2].float()
        values[angles, 449::2] = latent[angles, 448::2].float()
        if dtype == torch.float8_e4m3fn:
            values = values.clamp(-448, 448)
        for t in range(6):
            s = int(slots[t])
            if s >= 0 and (t + 1) % ratio == 0:
                expected[s // 4, s % 4] = values[t]
        torch.testing.assert_close(
            cache.float(), expected.to(dtype).float(), rtol=0, atol=0
        )
        assert torch.equal(cache.view(torch.uint8), replica.view(torch.uint8))


@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
def test_sparse_pt_plugin_dispatch_replays_dynamic_q_scale(heads):
    """The plugin combines two PT caches and recomputes device scale on replay."""
    from ixformer.inference.backend_registry import registry

    for name in ("dsv41_quant_mla_q_fp8_pt", "dsv41_sparse_mla_fp8_pt"):
        assert registry.resolve(name) == "ixformer_native", registry.describe(name)
    a = object.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(a)
    a.kv_cache_dtype = "fp8_e4m3"
    a.kv_scale = 1.0
    a.scale = 0.04
    a.attn_sink = torch.zeros(heads, dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(123)
    cache = (torch.randn((2, 4, 512), generator=gen, device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    extra = (torch.randn((2, 4, 512), generator=gen, device="cuda") * 0.1).to(
        torch.float8_e4m3fn
    )
    a.swa_cache_layer = SimpleNamespace(kv_cache=cache)
    q = torch.randn((2, heads, 512), generator=gen, device="cuda", dtype=torch.bfloat16)
    a._q_fp8_pt = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    a._q_scale_pt = torch.empty(1, device="cuda")
    a._q_quant_workspace_pt = torch.empty(256, device="cuda")
    a._window_idx = torch.full((2, 256), -1, dtype=torch.int32, device="cuda")
    idx = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32, device="cuda")
    lens = torch.full((2,), 2, dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)

    def run():
        a._sparse_mla(q, idx, lens, extra, idx, lens, output)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for multiplier in (1, 2):
        q.mul_(multiplier)
        a._q_quant_workspace_pt.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(a._q_scale_pt[0], q.float().abs().max() / 448)
        for t in range(2):
            keys = torch.cat(
                (
                    cache[1, t * 2 : t * 2 + 2].float(),
                    extra[1, t * 2 : t * 2 + 2].float(),
                )
            )
            query = a._q_fp8_pt[t].float() * a._q_scale_pt
            logits = query @ keys.T * a.scale
            probs = torch.softmax(
                torch.cat((logits, a.attn_sink[:, None]), dim=1), dim=1
            )
            expected = probs[:, :4] @ keys
            torch.testing.assert_close(
                output[t].float(), expected, atol=0.004, rtol=0.04
            )


@pytest.mark.parametrize("dtype, expected", [(None, 128), ("fp8", 128), ("bf16", 64)])
def test_main_backend_negotiates_physical_page_for_indexer(
    monkeypatch, dtype, expected
):
    from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
        IluDeepseekV4FlashMLABackend,
    )
    from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
        IluDeepseekV41IndexerBackend,
    )

    if dtype is None:
        monkeypatch.delenv("VLLM_ILU_V41_INDEXER_KV_DTYPE", raising=False)
    else:
        monkeypatch.setenv("VLLM_ILU_V41_INDEXER_KV_DTYPE", dtype)
    physical_page = IluDeepseekV4FlashMLABackend.get_preferred_block_size(16)
    assert physical_page == expected
    assert (
        physical_page in IluDeepseekV41IndexerBackend.get_supported_kernel_block_sizes()
    )
    if dtype != "bf16":
        assert (physical_page // 2) % 64 == 0


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])
def test_window_padding_replay_and_graph(n):
    a = object.__new__(IluDeepseekV41Attention)
    a.window_size = 128
    a._window_idx = torch.full((64, 256), -1, dtype=torch.int32, device="cuda")
    a._window_lens = torch.zeros(64, dtype=torch.int32, device="cuda")
    a._window_offsets = torch.arange(128, device="cuda")
    pos = torch.arange(n, device="cuda", dtype=torch.int64) + 65536
    table = torch.arange(1, 1027, device="cuda", dtype=torch.int32).view(1, -1)
    valid = torch.ones(n, device="cuda", dtype=torch.bool)
    valid[-1] = False
    md = SimpleNamespace(
        token_to_req_indices=torch.zeros(n, device="cuda", dtype=torch.int32),
        is_valid_token=valid,
        block_table=table,
        block_size=64,
        replay_start=torch.tensor([65520], device="cuda"),
    )

    def run():
        return a._window_indices(md, 0, n, pos)

    expected, lens = run()
    expected = expected.clone()
    lens = lens.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, length = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert torch.equal(length, lens)
    assert length[-1] == 0
    assert torch.all(out[-1] == -1)
    assert out.stride(0) == 256
    # Replay reads current positions, not capture-time host scalars.
    pos.add_(1)
    g.replay()
    torch.cuda.synchronize()
    assert length[0] == (18 if n > 1 else 0)


def test_compressor_metadata_never_maps_null_page_in_eager_and_graph():
    from vllm_iluvatar.models.deepseek_v41.compressor import (
        IluCompressorMetadataBuilder,
    )

    builder = object.__new__(IluCompressorMetadataBuilder)
    builder.capacity = 8
    builder.token_to_req_indices = torch.zeros(6, dtype=torch.int32, device="cuda")
    builder.slot_mapping_buffer = torch.full(
        (6,), 123, dtype=torch.int64, device="cuda"
    )
    req = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32, device="cuda")
    positions = torch.arange(6, dtype=torch.int64, device="cuda")
    block_table = torch.tensor([[0], [2], [7]], dtype=torch.int32, device="cuda")

    class CommonMetadata:
        slot_mapping = torch.full((6,), -1, dtype=torch.int64, device="cuda")
        num_actual_tokens = 4
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
        block_table_tensor = block_table

        @property
        def positions(self):
            return positions

        @staticmethod
        def token_to_req_indices(out):
            out.copy_(req)
            return out

    common = CommonMetadata()

    def run():
        return builder.build(0, common).slot_mapping

    eager = run()
    output_ptr = eager.data_ptr()
    assert eager.tolist() == [-1, -1, 18, 19, -1, -1]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    block_table.copy_(torch.tensor([[3], [0], [7]], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    assert captured.data_ptr() == output_ptr
    assert captured.tolist() == [24, 25, -1, -1, -1, -1]


@pytest.mark.parametrize("ratio", [1, 2])
def test_global_topk_graph_padding_uses_current_device_metadata(ratio):

    a = object.__new__(IluDeepseekV41Attention)
    a.compress_ratio = ratio
    a._global_topk = torch.full((64, 512), -1, dtype=torch.int32, device="cuda")
    a._global_topk_lens = torch.zeros(64, dtype=torch.int32, device="cuda")
    local = torch.full((64, 512), -1, dtype=torch.int32, device="cuda")
    local[:63, 0] = 0
    req = torch.zeros(64, dtype=torch.int32, device="cuda")
    req[-1] = 2**30
    table = torch.ones(1, 2048, dtype=torch.int32, device="cuda")
    valid = torch.ones(64, dtype=torch.bool, device="cuda")
    valid[-1] = False
    positions = torch.full((64,), 131071, dtype=torch.int64, device="cuda")

    def run():
        return a._global_indices(local, req, table, 64, valid, positions)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lens = run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out[:63, 0] == 64) and torch.all(lens[:63] == 1)
    assert lens[-1] == 0 and torch.all(out[-1] == -1)
    table.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out[:63, 0] == 128)


def test_swa_capture_dummy_never_writes_null_page(monkeypatch):
    from vllm_iluvatar.models.deepseek_v41.sparse_mla import (
        IluDeepseekV41SparseSWAMetadataBuilder,
    )
    from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
        IluDeepseekSparseSWAMetadataBuilder,
    )

    builder = object.__new__(IluDeepseekV41SparseSWAMetadataBuilder)
    builder._valid_slots = torch.full((64,), -1, dtype=torch.int64, device="cuda")
    builder.block_size = 64
    raw = torch.tensor([0, 63, 64, -1], dtype=torch.int64, device="cuda")
    cm = SimpleNamespace(slot_mapping=raw, causal=True)
    monkeypatch.setattr(
        IluDeepseekSparseSWAMetadataBuilder,
        "build",
        lambda self, prefix, cm, fast, replay_start=None: cm,
    )
    result = builder.build(0, cm)
    assert result.slot_mapping.tolist() == [-1, -1, 64, -1]
    assert raw.tolist() == [0, 63, 64, -1]


@pytest.mark.parametrize("n", [1, 6, 64, 384, 2048])
def test_fp8_indexer_query_out_replays_current_positions_without_overwrite(n):
    import vllm_iluvatar._C  # noqa: F401

    q = torch.randn(n, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(n, 32, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(n, device="cuda", dtype=torch.int64)
    phase = torch.randn(n + 2, 32, device="cuda")
    cs = torch.cat((phase.cos(), phase.sin()), dim=-1)
    q_storage = torch.full(
        (n + 2, 32, 128), 1, device="cuda", dtype=torch.float8_e4m3fn
    )
    w_storage = torch.full((n + 2, 32), 123, device="cuda", dtype=torch.float32)
    qo, wo = q_storage[1:-1], w_storage[1:-1]

    def run():
        formal_ops.dsv41_indexer_q_rope_fp8_out(
            q, positions, cs, weights, 0.015625, qo, wo
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    positions.add_(1)
    graph.replay()
    expected_q, expected_w = formal_ops.dsv41_indexer_q_rope_fp8(
        q, positions, cs, weights, 0.015625
    )
    assert torch.equal(qo.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(wo, expected_w)
    assert torch.all(q_storage[[0, -1]].float() == 1)
    assert torch.all(w_storage[[0, -1]] == 123)


@pytest.mark.parametrize("ratio,bs", [(1, 128), (2, 64)])
def test_fp8_indexer_key_store_matches_bf16_quantization_and_pp_mirror(ratio, bs):
    n = 513
    k = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(n, device="cuda", dtype=torch.int64) + 65536
    slots = torch.arange(n, device="cuda", dtype=torch.int64) + bs
    slots[-1] = -1
    phase = torch.randn(65536 + n, 32, device="cuda")
    cs = torch.cat((phase.cos(), phase.sin()), -1)
    pages = (n + bs - 1) // bs + 1
    bf16 = torch.zeros(pages, bs, 128, device="cuda", dtype=torch.bfloat16)
    fp8 = torch.zeros(pages, bs, 132, device="cuda", dtype=torch.uint8)
    mirror = torch.zeros(n, 132, device="cuda", dtype=torch.uint8)
    formal_ops.dsv41_indexer_k_norm_rope_store(
        k, pos, cs, weight, 1e-20, bf16, slots, ratio, False, None
    )
    formal_ops.dsv41_indexer_k_norm_rope_store(
        k, pos, cs, weight, 1e-20, fp8, slots, ratio, True, mirror
    )
    valid = (slots >= 0) & ((pos + 1) % ratio == 0)
    live_slots = slots[valid]
    source = bf16[live_slots // bs, live_slots % bs].float()
    scale = torch.exp2(
        torch.ceil(torch.log2(source.abs().amax(-1).clamp(min=1e-4) / 448))
    )
    expected = (
        (source / scale[:, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    page = fp8.reshape(pages, -1)
    values = page[:, : bs * 128].reshape(pages, bs, 128)
    scales = page[:, bs * 128 :].contiguous().view(torch.float32)
    assert torch.equal(values[live_slots // bs, live_slots % bs], expected)
    assert torch.equal(scales[live_slots // bs, live_slots % bs], scale)
    assert torch.equal(mirror[valid, :128], expected)
    assert torch.equal(
        mirror[:, 128:].contiguous().view(torch.float32).flatten()[valid], scale
    )
    assert torch.all(fp8[0] == 0) and torch.all(mirror[~valid] == 0)
