# SPDX-License-Identifier: Apache-2.0
"""Framework metadata, cache layout and padding contract for the FP8 route."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheTensor,
    SparseCacheRole,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_iluvatar.models.minimax_m3.mxfp8 import fp8_attention as fp8


def test_padded_metadata_keeps_query_end_and_independent_page_tables():
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    builders = [
        fp8.FP8MetadataBuilder(None, [name], config, torch.device("cpu"))
        for name in ("main", "index")
    ]
    cu = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32)
    seq = torch.tensor([132, 17, 0, 0], dtype=torch.int32)
    tables = [
        torch.tensor([[7, 8], [9, 0], [0, 0], [0, 0]], dtype=torch.int32),
        torch.tensor([[2, 3], [4, 0], [0, 0], [0, 0]], dtype=torch.int32),
    ]
    results = []
    for builder, table in zip(builders, tables):
        common = CommonAttentionMetadata(
            num_reqs=4,
            query_start_loc_cpu=cu,
            slot_mapping=torch.full((8,), -1, dtype=torch.int64),
            num_actual_tokens=8,
            query_start_loc=cu,
            seq_lens=seq,
            block_table_tensor=table,
            max_query_len=2,
            max_seq_len=132,
        )
        results.append(builder.build(0, common))
    for metadata in results:
        assert metadata.num_actual_tokens == 8 and metadata.cu[-1] == 3
        assert metadata.cu.data_ptr() == cu.data_ptr()
        assert torch.equal(metadata.prefix, torch.tensor([130, 16, 0, 0]))
        assert torch.equal(metadata.seq, seq)
    assert results[0].block_table.data_ptr() != results[1].block_table.data_ptr()
    assert torch.equal(results[0].block_table, tables[0])
    assert torch.equal(results[1].block_table, tables[1])


def test_text_and_outer_configs_share_status_lifetime(monkeypatch):
    layers = {}
    outer = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers)
    )
    text = SimpleNamespace(compilation_config=outer.compilation_config)
    unrelated = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    state = object()
    monkeypatch.setitem(fp8._STATES, id(layers), state)
    assert outer is not text
    assert fp8.get_state(outer) is fp8.get_state(text) is state
    assert fp8.get_state(unrelated) is None


def test_dense_metadata_preserves_mixed_request_order():
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=6))
    builder = fp8.FP8MetadataBuilder(None, ["main"], config, torch.device("cpu"))
    cu = torch.tensor([0, 1, 5, 6, 7, 7, 9], dtype=torch.int32)
    seq = torch.tensor([8, 13, 21, 34, 0, 40], dtype=torch.int32)
    table = torch.arange(12, dtype=torch.int32).reshape(6, 2)
    common = CommonAttentionMetadata(
        slot_mapping=torch.full((11,), -1, dtype=torch.int64),
        num_reqs=6,
        num_actual_tokens=11,
        query_start_loc=cu,
        query_start_loc_cpu=cu,
        seq_lens=seq,
        block_table_tensor=table,
        max_query_len=4,
        max_seq_len=40,
    )
    metadata = builder.build(0, common)
    assert (metadata.num_prefills, metadata.num_decodes) == (2, 3)
    assert metadata.dense_valid_tokens == 9
    assert metadata.seq.data_ptr() == seq.data_ptr()
    assert metadata.block_table.data_ptr() == table.data_ptr()
    segments = metadata.dense_segments
    assert [
        (s.request_start, s.request_end, s.token_start, s.token_end, s.decode_query_len)
        for s in segments
    ] == [(0, 1, 0, 1, 1), (1, 2, 1, 5, 0), (2, 4, 5, 7, 1), (4, 6, 7, 9, 0)]
    assert segments[0].cu.tolist() == [0, 1] and segments[2].cu.tolist() == [0, 1, 2]
    assert segments[1].cu.tolist() == [0, 4]
    assert segments[3].cu.tolist() == [0, 0, 2]


@pytest.mark.parametrize(
    "lengths,expected", [([1, 1, 1], True), ([1, 4], False), ([0, 0], False)]
)
def test_fp8_metadata_drives_graph_safe_moe_decode(lengths, expected):
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_decode_metadata import (
        _is_only_decode,
    )

    builder = fp8.FP8MetadataBuilder(
        None,
        ["main"],
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=3)),
        "cpu",
    )
    cu = torch.tensor(
        [0] + list(__import__("itertools").accumulate(lengths)), dtype=torch.int32
    )
    seq = torch.tensor([128 + n for n in lengths], dtype=torch.int32)
    common = CommonAttentionMetadata(
        slot_mapping=torch.full((sum(lengths),), -1, dtype=torch.int64),
        num_reqs=len(lengths),
        num_actual_tokens=sum(lengths),
        query_start_loc=cu,
        query_start_loc_cpu=cu,
        seq_lens=seq,
        block_table_tensor=torch.zeros(len(lengths), 2, dtype=torch.int32),
        max_query_len=max(lengths),
        max_seq_len=int(seq.max()),
    )
    metadata = builder.build(0, common)
    assert _is_only_decode({"main": metadata, "index": metadata}) is expected
    from vllm.v1.attention.backend import AttentionCGSupport

    assert (
        builder.get_cudagraph_support(None, None)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


@pytest.mark.parametrize("kind,kv_heads", [("main", 4), ("main", 2), ("index", 1)])
@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.BLHNC])
def test_official_allocator_views_bind_without_copy_and_preserve_padding(
    kind, kv_heads, layout
):
    # TP1/TP2 production KV heads, two official layer views, padded pages and
    # nonzero backing offset. The index spec keeps its own cache role/table.
    cls = fp8.MiniMaxM3FP8Attention if kind == "main" else fp8.MiniMaxM3FP8IndexerCache
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.num_kv_heads = kv_heads
    layer.kv_cache_dtype = "fp8_e4m3"
    layer.head_dim = 128
    layer.dtype = torch.float8_e4m3fn
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    spec = layer.get_kv_cache_spec(config)
    if kind == "index":
        assert spec.cache_role == SparseCacheRole.INDEXER
    page = spec.page_size_bytes + 64
    blocks, layers, offset = 3, ["first", "second"], 64
    layer_stride, block_stride = (
        (blocks * page, page) if layout == KVCacheLayout.LBHNC else (page, 2 * page)
    )
    raw = torch.full((offset + 2 * blocks * page,), 23, dtype=torch.int8)
    placement = KVCacheTensor(raw.numel(), layers, layer_stride, block_stride, offset)
    cache, sibling = create_kv_cache_views(raw, spec, blocks, layout, placement)
    layer.bind_kv_cache(cache)
    expected = cache if kind == "main" else cache.squeeze(1)
    assert (
        layer.kv_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    )
    assert layer.kv_cache.data_ptr() == expected.data_ptr()
    assert layer.kv_cache.stride() == expected.stride()
    assert layer.kv_cache.storage_offset() == expected.storage_offset() == offset
    layer.kv_cache.zero_()
    # Read raw bytes for an exact untouched sibling/guard check (FP8 bit pattern).
    assert torch.all(sibling.view(torch.int8) == 23)
    assert torch.all(raw[:offset] == 23)
    for block in range(blocks):
        begin = offset + block * block_stride + spec.page_size_bytes
        assert torch.all(raw[begin : begin + 64] == 23)
    previous = layer.kv_cache
    bad_shape = cache[..., :127] if kind == "index" else cache[..., :255]
    for bad in (torch.empty(cache.shape, dtype=torch.bfloat16), bad_shape):
        with pytest.raises(ValueError, match="allocator must bind"):
            layer.bind_kv_cache(bad)
        assert layer.kv_cache is previous
    if kind == "index":
        with pytest.raises(ValueError, match="allocator must bind"):
            layer.bind_kv_cache(cache.squeeze(1))
    else:
        row_padded = torch.empty((blocks, kv_heads, 128, 264), dtype=cache.dtype)[
            ..., :256
        ]
        with pytest.raises(ValueError, match="allocator must bind"):
            layer.bind_kv_cache(row_padded)


@pytest.mark.parametrize(
    "sparse,attention_capacity,partitioned",
    [
        (False, 5, False),
        (False, 8, False),
        (True, 5, False),
        (True, 8, False),
        (False, 5, True),
        (False, 8, True),
        (True, 5, True),
        (True, 8, True),
    ],
)
def test_forward_separates_model_and_attention_padding(
    monkeypatch, sparse, attention_capacity, partitioned
):
    # Eager DP prefill can have hidden/positions T=8 but only 5 slots.
    # FULL decode can instead pad the metadata/slots to 8 with cu[-1]=5.
    T, n = 8, attention_capacity
    layer = fp8.MiniMaxM3FP8Attention.__new__(fp8.MiniMaxM3FP8Attention)
    torch.nn.Module.__init__(layer)
    layer.max_tokens = T
    layer.layer_name = "main"
    layer.sparse = sparse
    layer.layer_id = 0
    layer.num_heads = layer.num_kv_heads = 1
    layer.score_capacity = 16
    layer.attn_workspace = torch.empty(64 * T * 130)
    layer._k_scale = layer._v_scale = 1.0
    norm = SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6)
    layer.q_norm = layer.k_norm = norm
    layer.index_q_norm = layer.index_k_norm = norm
    layer.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.ones(16, 128), rotary_dim=128
    )
    layer.kv_cache = torch.zeros(2, 1, 128, 256)
    layer.index_cache = SimpleNamespace(
        prefix="index", kv_cache=torch.zeros(2, 128, 128)
    )
    layer.qkv_proj, layer.o_proj = "qkv", "o"
    for name, shape in {
        "x8": (T, 6144),
        "xs": (T,),
        "packed": (T, 640 if sparse else 384),
        "q8": (T, 1, 128),
        "qs": (T, 1),
        "iq": (T, 1, 128),
        "scores": (T, 1, 16),
        "top": (T, 1, 16),
        "attn_out": (T, 1, 128),
        "y8": (T, 128),
        "ys": (T,),
        "result": (T, 6144),
        "empty": (0,),
    }.items():
        setattr(layer, name, torch.full(shape, float("nan")))
    cu = torch.tensor([0, 5], dtype=torch.int32)
    metadata = fp8.FP8Metadata(
        n,
        cu,
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        5,
        5,
    )
    if partitioned:
        metadata.cu = torch.tensor([0, 1, 4, 5], dtype=torch.int32)
        metadata.seq = torch.tensor([1, 3, 1], dtype=torch.int32)
        metadata.prefix = torch.tensor([0, 0, 0], dtype=torch.int32)
        metadata.block_table = torch.tensor([[1], [0], [1]], dtype=torch.int32)
        metadata.dense_segments = (
            fp8.DenseQuerySegment(
                0, 1, 0, 1, 1, torch.tensor([0, 1], dtype=torch.int32)
            ),
            fp8.DenseQuerySegment(
                1, 2, 1, 4, 0, torch.tensor([0, 3], dtype=torch.int32)
            ),
            fp8.DenseQuerySegment(
                2, 3, 4, 5, 1, torch.tensor([0, 1], dtype=torch.int32)
            ),
        )
        metadata.dense_valid_tokens = 5
    main_slots = torch.tensor(list(range(5)) + [-1] * (n - 5), dtype=torch.int64)
    index_slots = main_slots.clone()
    context = SimpleNamespace(
        attn_metadata={"main": metadata, "index": metadata},
        slot_mapping={"main": main_slots, "index": index_slots},
    )
    monkeypatch.setattr(fp8, "get_forward_context", lambda: context)
    calls = []

    def quant(**kw):
        assert kw["x"].shape[0] == n
        assert torch.isfinite(kw["x"]).all()
        kw["fp8_output"].fill_(1)
        kw["output_scales"].fill_(1)

    def project(which, A, As, out):
        assert A.shape[0] == As.shape[0] == out.shape[0] == n
        assert torch.isfinite(A).all() and torch.isfinite(As).all()
        out.fill_(3)
        calls.append(which)
        return out

    def producer(**kw):
        for key in (
            "packed",
            "positions",
            "slot_mapping",
            "q_out",
            "q_scale_out",
        ):
            assert kw[key].shape[0] == n
        assert kw["slot_mapping"] is main_slots
        if sparse:
            assert kw["index_slot_mapping"] is index_slots
            assert kw["index_q_out"].shape[0] == n

    dense_calls = []

    def reader(**kw):
        if partitioned:
            dense_calls.append(kw["q"].shape[0])
            assert kw["q"].shape[0] == kw["out"].shape[0] == kw["q_scale"].shape[0]
            if sparse:
                assert kw["cu_seqlens_q"].tolist() == (
                    [0, 3] if kw["phase"] == "prefill" else [0, 1]
                )
                assert kw["prefix_lens"].tolist() == [0]
                assert kw["workspace"] is layer.attn_workspace
                assert "num_splits" not in kw
            elif "decode_query_len" in kw:
                assert kw["decode_query_len"] == 1 and kw["seq_lens"].tolist() == [1]
            else:
                assert kw["cu_seqlens_q"].tolist() == [0, 3]
                assert kw["seq_lens"].tolist() == [3]
            if not sparse:
                assert kw["workspace"] is layer.attn_workspace
                assert (
                    kw["k_cache"].untyped_storage().data_ptr()
                    == layer.kv_cache.untyped_storage().data_ptr()
                )
                assert kw["v_cache"].storage_offset() == 128
        else:
            assert kw["q"].shape[0] == kw["out"].shape[0] == n
            assert kw["cu_seqlens_q"] is cu
        kw["out"].fill_(2)

    def output_quant(**kw):
        assert kw["input"].shape[0] == kw["out_fp8"].shape[0] == n
        if partitioned:
            assert torch.all(kw["input"][:5] == 2) and torch.all(kw["input"][5:] == 0)
        kw["out_fp8"].fill_(2)
        kw["out_scale"].fill_(1)

    monkeypatch.setattr(fp8.ixf, "fp8_per_token_dynamic_scaled", quant)
    monkeypatch.setattr(layer, "_project", project)
    monkeypatch.setattr(fp8, "qknorm_rope_fp8_kv_insert", producer)
    index_calls = []

    def score(index_q, index_cache, block_table, cu_seqlens_q, seq_lens, out, **kwargs):
        assert index_q.shape[0] == out.shape[0] == n
        index_calls.append("score")
        out.zero_()
        return out

    def topk(**kw):
        assert index_calls == ["score"]
        index_calls.append("topk")

    monkeypatch.setattr(fp8, "fp8_index_score", score)
    monkeypatch.setattr(fp8, "index_topk", topk)
    monkeypatch.setattr(fp8, "fp8_dsa", reader)
    monkeypatch.setattr(fp8.ixf, "paged_attention_prefill_fp8", reader)
    monkeypatch.setattr(fp8.ixf, "paged_attention_decode_fp8", reader)
    monkeypatch.setattr(fp8, "_ixf_fp8_attention_output_quant", output_quant)
    hidden = torch.ones(T, 6144)
    hidden[n:].fill_(float("nan"))
    out = layer(torch.arange(T, dtype=torch.int64), hidden)
    if sparse:
        assert index_calls == ["score", "topk"]
    assert calls == ["qkv", "o"] and out.shape == hidden.shape
    assert torch.all(out[:n] == 3) and torch.all(out[n:] == 0)
    if partitioned:
        assert dense_calls == [1, 3, 1]


@pytest.mark.parametrize("dtype", ["auto", "fp8", "bf16", "mxfp4", "nvfp4"])
def test_fp8_index_dtype_uses_official_auto_resolution(dtype):
    from vllm.config.attention import AttentionConfig

    config = AttentionConfig(indexer_kv_dtype=dtype)
    if dtype in ("auto", "fp8"):
        assert fp8._resolve_fp8_indexer_dtype(config) == "fp8"
    else:
        with pytest.raises(NotImplementedError, match="E4M3 index cache"):
            fp8._resolve_fp8_indexer_dtype(config)
    assert config.indexer_kv_dtype == dtype


@pytest.mark.parametrize(
    "suffix", ["k_scale", "v_scale", "kv_scale", "attn.k_scale", "qkv_proj.kv_scale"]
)
def test_fixed_kv_scales_load_into_attention_buffers_and_skip_missing_pp_layers(
    monkeypatch, suffix
):
    from vllm.model_executor.models.utils import PPMissingLayer

    from vllm_iluvatar.models.minimax_m3.mxfp8.model import MiniMaxM3Model

    model = MiniMaxM3Model.__new__(MiniMaxM3Model)
    torch.nn.Module.__init__(model)
    attention = fp8.MiniMaxM3FP8Attention.__new__(fp8.MiniMaxM3FP8Attention)
    torch.nn.Module.__init__(attention)
    for name in ("_k_scale", "_v_scale"):
        attention.register_buffer(name, torch.ones(1), persistent=False)
    layer = torch.nn.Module()
    layer.self_attn = attention
    model.layers = torch.nn.ModuleList([layer, PPMissingLayer()])
    monkeypatch.setattr(model, "get_expert_mapping", lambda: [])
    name = "layers.0.self_attn." + suffix
    loaded = model.load_weights(
        [
            (name, torch.tensor(0.25)),
            ("layers.1.self_attn.kv_scale", torch.tensor(0.75)),
        ]
    )
    assert loaded == {name}
    assert attention._k_scale.item() == (1.0 if suffix == "v_scale" else 0.25)
    assert attention._v_scale.item() == (
        0.25 if suffix in ("v_scale", "kv_scale", "qkv_proj.kv_scale") else 1.0
    )
    for value in (
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        torch.ones(2),
    ):
        with pytest.raises(ValueError, match="Invalid fixed FP8 KV scale"):
            model.load_weights([(name, value)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CoreX GPU")
@pytest.mark.parametrize(
    "hq,hk,sparse,layout",
    [
        (64, 4, False, KVCacheLayout.LBHNC),
        (64, 4, True, KVCacheLayout.BLHNC),
        (32, 2, False, KVCacheLayout.BLHNC),
        (32, 2, True, KVCacheLayout.LBHNC),
    ],
)
def test_native_producer_writes_official_views_with_independent_slots(
    hq, hk, sparse, layout
):
    """Known-pattern K/V/index writes expose axis, page, scale and guard errors."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        get_kv_quant_mode,
    )

    def allocate(spec, fill):
        page = spec.page_size_bytes + 64
        layers = ["first", "second"]
        ls, bs = (4 * page, page) if layout == KVCacheLayout.LBHNC else (page, 2 * page)
        raw = torch.full((64 + 8 * page,), fill, dtype=torch.int8, device="cuda")
        placement = KVCacheTensor(raw.numel(), layers, ls, bs, 64)
        views = create_kv_cache_views(raw, spec, 4, layout, placement)
        return raw, views[0], views[1], page, bs

    dtype = torch.float8_e4m3fn
    main_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=hk,
        head_size=128,
        head_size_v=128,
        dtype=dtype,
        kv_quant_mode=get_kv_quant_mode("fp8_e4m3"),
    )
    raw, cache, sibling, page, page_stride = allocate(main_spec, 17)
    before = raw.clone()
    expected = cache.float()
    slots = torch.tensor([383, -1, 0], device="cuda", dtype=torch.int64)
    heads = hq + 2 * hk + (hk + 1 if sparse else 0)
    packed = torch.full((3, heads, 128), 0.25, dtype=torch.bfloat16, device="cuda")
    signs = torch.tensor([1 if h % 2 == 0 else -1 for h in range(hk)], device="cuda")
    packed[:, hq : hq + hk] *= signs[None, :, None]
    values = (torch.arange(3, device="cuda")[:, None, None] + 1) * 0.25
    values = values + torch.arange(hk, device="cuda")[None, :, None] * 0.5
    values = values + (torch.arange(128, device="cuda") % 4)[None, None, :] * 0.125
    packed[:, hq + hk : hq + 2 * hk] = values
    weight = torch.zeros(128, dtype=torch.bfloat16, device="cuda")
    rope = torch.cat(
        [torch.ones((8, 32), device="cuda"), torch.zeros((8, 32), device="cuda")], 1
    )
    q = torch.empty((3, hq, 128), dtype=dtype, device="cuda")
    qs = torch.empty((3, hq), device="cuda")
    arguments = dict(
        packed=packed.flatten(1),
        q_norm_weight=weight,
        k_norm_weight=weight,
        rope_cache=rope,
        positions=torch.tensor([1, 0, 2], device="cuda"),
        slot_mapping=slots,
        kv_cache=cache,
        q_out=q,
        q_scale_out=qs,
        k_scale=torch.tensor([0.5], device="cuda"),
        v_scale=torch.tensor([2.0], device="cuda"),
        mode="dsa" if sparse else "full",
        backend="ixformer_native",
    )
    if sparse:
        spec = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=dtype,
            cache_role=SparseCacheRole.INDEXER,
        )
        index_raw, index_view, index_sibling, _, _ = allocate(spec, 19)
        index_before = index_raw.clone()
        index_expected = index_view.squeeze(1).float()
        arguments.update(
            index_norm_q_weight=weight,
            index_norm_k_weight=weight,
            index_slot_mapping=torch.tensor([255, -7, 384], device="cuda"),
            index_cache=index_view.squeeze(1),
            index_q_out=torch.empty((3, hk, 128), dtype=dtype, device="cuda"),
        )
    fp8.qknorm_rope_fp8_kv_insert(**arguments)
    for token, slot in [(0, 383), (2, 0)]:
        expected[slot // 128, :, slot % 128, :128] = (signs * 2)[:, None]
        expected[slot // 128, :, slot % 128, 128:] = (
            (values[token] / 2).to(dtype).float()
        )
    torch.testing.assert_close(cache.float(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        q.float() * qs[:, :, None],
        torch.ones_like(q, dtype=torch.float32),
        rtol=0.01,
        atol=0.01,
    )
    assert torch.all(sibling.view(torch.int8) == 17)
    assert torch.equal(raw[:64], before[:64])
    for block in range(4):
        start = 64 + block * page_stride + main_spec.page_size_bytes
        assert torch.equal(raw[start : start + 64], before[start : start + 64])
    if sparse:
        for slot in [255, 384]:
            index_expected[slot // 128, slot % 128] = 1
        torch.testing.assert_close(
            arguments["index_cache"].float(), index_expected, rtol=0, atol=0
        )
        assert torch.all(index_sibling.view(torch.int8) == 19)
        assert torch.equal(index_raw[:64], index_before[:64])
