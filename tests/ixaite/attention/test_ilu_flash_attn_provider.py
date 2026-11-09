# SPDX-License-Identifier: Apache-2.0
"""Fixed attention support and independent ixaite state/execution contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.worker.utils import select_common_block_size

from tests.ixaite.backend_isolation.ixformer_guard import forbid_ixformer_calls
from vllm_iluvatar.backends.ixaite.attention import flash_attn


# Compute bottom-right causal attention directly from the written paged cache.
def _attention_reference(query, cache, context_len):
    key, value = cache.unbind(0)
    key = key.permute(1, 0, 2, 3).flatten(1, 2)[:, :context_len].float()
    value = value.permute(1, 0, 2, 3).flatten(1, 2)[:, :context_len].float()
    ratio = query.shape[1] // key.shape[0]
    key = key.repeat_interleave(ratio, dim=0)
    value = value.repeat_interleave(ratio, dim=0)
    q = query.transpose(0, 1).float()
    scores = q @ key.transpose(1, 2) / query.shape[-1] ** 0.5
    positions = (
        torch.arange(query.shape[0], device=query.device) + context_len - query.shape[0]
    )
    mask = torch.arange(context_len, device=query.device)[None, :] > positions[:, None]
    scores.masked_fill_(mask, float("-inf"))
    return (scores.softmax(-1) @ value).transpose(0, 1).to(query.dtype)


# Exercise writer + prefill/decode and replay through the actual selected layer.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize(
    "head_dim,query_len,prefill,use_graph",
    [
        (128, 4, True, False),
        (256, 4, True, False),
        (128, 1, False, False),
        (256, 4, False, False),
        (256, 1, False, True),
    ],
)
def test_gpu_attention_lifecycle(monkeypatch, head_dim, query_len, prefill, use_graph):
    config = _supported_config(num_heads=8, num_kv_heads=2)
    monkeypatch.setattr(flash_attn, "get_current_vllm_config_or_none", lambda: config)
    torch.manual_seed(17)
    with forbid_ixformer_calls():
        flash_attn.configure_flash_attn_provider()
        impl = flash_attn.IxaiteFlashAttentionImpl(
            8, head_dim, head_dim**-0.5, 2, None, None, "auto"
        )
        q = torch.randn(query_len, 8, head_dim, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(query_len, 2, head_dim, device="cuda", dtype=q.dtype)
        value = torch.randn_like(key)
        cache = torch.randn(2, 4, 2, 16, head_dim, device="cuda", dtype=q.dtype)
        out = torch.empty_like(q)
        context_len = 19
        slots = torch.arange(
            context_len - query_len, context_len, device="cuda", dtype=torch.int64
        )
        seq_lens = torch.tensor([context_len], device="cuda", dtype=torch.int32)
        query_start = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
        key_start = torch.tensor([0, context_len], device="cuda", dtype=torch.int32)
        table = torch.arange(4, device="cuda", dtype=torch.int32).view(1, 4)
        metadata = SimpleNamespace(
            causal=True,
            use_cascade=False,
            num_actual_tokens=query_len,
            num_decode_tokens=0 if prefill else query_len,
            slot_mapping=slots,
            max_seq_len=context_len,
            num_prefills=int(prefill),
            num_decodes=int(not prefill),
            prefill=SimpleNamespace(
                query_start_loc=query_start,
                key_start_loc=key_start,
                max_query_len=query_len,
                block_table=table,
            )
            if prefill
            else None,
            decode=None
            if prefill
            else SimpleNamespace(
                query_start_loc=query_start,
                seq_lens=seq_lens,
                max_query_len=query_len,
                max_decode_seq_len=64,
                block_table=table,
                use_graph=use_graph,
            ),
        )
        layer = SimpleNamespace(
            _k_scale=torch.ones((), device="cuda"),
            _v_scale=torch.ones((), device="cuda"),
        )
        impl.forward(layer, q, key, value, cache, metadata, out)
        torch.testing.assert_close(
            out, _attention_reference(q, cache, context_len), rtol=3e-2, atol=3e-2
        )
        if use_graph:
            # Stable tensor addresses must permit a longer live context at replay.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                impl.forward(layer, q, key, value, cache, metadata, out)
            context_len += 7
            seq_lens.fill_(context_len)
            slots.copy_(
                torch.arange(
                    context_len - query_len,
                    context_len,
                    device="cuda",
                    dtype=slots.dtype,
                )
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out, _attention_reference(q, cache, context_len), rtol=3e-2, atol=3e-2
            )


# Isolate startup bindings and calibration controls for each constructor test.
@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    for name in (
        "VLLM_ILU_FLASH_ATTN_QUANT_TYPE",
        "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH",
        "VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH",
    ):
        monkeypatch.setattr(flash_attn.envs, name, "", raising=False)
    monkeypatch.setattr(flash_attn, "_SELECTED_FLASH_ATTN_OPS", None)
    monkeypatch.setattr(
        flash_attn, "get_current_vllm_config_or_none", _supported_config
    )


# All three functions must bind before a layer can own an attention plan.
def test_complete_runtime_pair(monkeypatch):
    functions = (Mock(), Mock(), Mock())
    monkeypatch.setattr(
        flash_attn,
        "load_ixaite_python_op",
        Mock(side_effect=[(f, "supported") for f in functions]),
    )
    flash_attn.configure_flash_attn_provider()
    assert functions == flash_attn._SELECTED_FLASH_ATTN_OPS


# An incomplete runtime cannot become an initialized layer.
def test_missing_runtime_fails(monkeypatch):
    monkeypatch.setattr(
        flash_attn, "load_ixaite_python_op", lambda *_: (None, "missing decode")
    )
    with pytest.raises(RuntimeError, match="missing decode"):
        flash_attn.configure_flash_attn_provider()


# Selector failures must not silently select a baseline implementation.
@pytest.mark.parametrize(
    "override,value",
    [
        ("dtype", torch.float16),
        ("block_size", 32),
        ("has_sliding_window", True),
        ("use_non_causal", True),
        ("use_kv_connector", True),
    ],
)
def test_unsupported_selector_raises(override, value):
    selector = _supported_selector()._replace(**{override: value})
    with pytest.raises(RuntimeError, match="requested flash_attn"):
        flash_attn.get_flash_attn_backend_class_path(selector)


# The entry selects the independent backend before constructing metadata.
def test_supported_selector_selects_ixaite():
    assert flash_attn.get_flash_attn_backend_class_path(_supported_selector()).endswith(
        "flash_attn.IxaiteFlashAttentionBackend"
    )


# Reject invalid dynamic queries before cache or output can be modified.
@pytest.mark.parametrize(
    "dtype,causal", [(torch.float16, True), (torch.bfloat16, False)]
)
def test_invalid_runtime_does_not_write_cache(monkeypatch, dtype, causal):
    monkeypatch.setattr(
        flash_attn, "_SELECTED_FLASH_ATTN_OPS", (Mock(), Mock(), Mock())
    )
    impl = flash_attn.IxaiteFlashAttentionImpl(16, 256, 1 / 16, 4, None, None, "auto")
    q = torch.zeros(1, 16, 256, dtype=dtype)
    cache = torch.zeros(2, 1, 4, 16, 256, dtype=torch.bfloat16)
    output = torch.ones_like(q)
    writer = Mock(side_effect=AssertionError("cache written"))
    monkeypatch.setattr(flash_attn.ops, "reshape_and_cache_flash", writer)
    with forbid_ixformer_calls(), pytest.raises(RuntimeError, match="requires"):
        impl.forward(Mock(), q, q, q, cache, SimpleNamespace(causal=causal), output)
    writer.assert_not_called()
    assert output.eq(1).all() and cache.eq(0).all()


# Numerical dependency gaps must fail before writes, including direct layer use.
def test_spec_graph_fails_before_cache_write(monkeypatch):
    monkeypatch.setattr(
        flash_attn, "_SELECTED_FLASH_ATTN_OPS", (Mock(), Mock(), Mock())
    )
    impl = flash_attn.IxaiteFlashAttentionImpl(16, 256, 1 / 16, 4, None, None, "auto")
    q = torch.zeros(4, 16, 256, dtype=torch.bfloat16)
    cache = torch.zeros(2, 1, 4, 16, 256, dtype=q.dtype)
    out = torch.ones_like(q)
    metadata = SimpleNamespace(
        causal=True, decode=SimpleNamespace(use_graph=True, max_query_len=4)
    )
    writer = Mock()
    monkeypatch.setattr(flash_attn.ops, "reshape_and_cache_flash", writer)
    with (
        forbid_ixformer_calls(),
        pytest.raises(RuntimeError, match="multi-token graph"),
    ):
        impl.forward(Mock(), q, q, q, cache, metadata, out)
    writer.assert_not_called()
    assert cache.eq(0).all() and out.eq(1).all()


# Known full-graph speculative configurations fail before kernel initialization.
def test_spec_graph_fails_at_construction(monkeypatch):
    config = _supported_config()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True)
    )
    monkeypatch.setattr(flash_attn, "get_current_vllm_config_or_none", lambda: config)
    loader = Mock(side_effect=AssertionError("runtime loaded"))
    monkeypatch.setattr(flash_attn, "load_ixaite_python_op", loader)
    with pytest.raises(RuntimeError, match="numerical failure"):
        flash_attn.IxaiteFlashAttentionImpl(16, 256, 1 / 16, 4, None, None, "auto")
    loader.assert_not_called()


def _supported_config(
    *, tensor_parallel_size: int = 1, num_heads: int = 16, num_kv_heads: int = 4
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            enforce_eager=True,
            get_num_attention_heads=lambda parallel_config: num_heads,
            get_num_kv_heads=lambda parallel_config: num_kv_heads,
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )


def _supported_selector():
    return AttentionSelectorConfig(
        head_size=256,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        attn_type=flash_attn.AttentionType.DECODER,
        has_sliding_window=False,
        use_non_causal=False,
        use_batch_invariant=False,
        use_kv_connector=False,
        use_pcp=False,
    )


def _config_reason(config=None, **overrides):
    arguments = dict(
        vllm_config=config or _supported_config(),
        num_heads=16,
        num_kv_heads=4,
        head_size=256,
        scale=1.0 / 16.0,
        attn_type=flash_attn.AttentionType.DECODER,
        kv_cache_dtype="auto",
        quant_type="",
        post_quant_kv_int8=False,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        sinks=None,
        kv_sharing_target_layer_name=None,
    )
    arguments.update(overrides)
    return flash_attn._ixaite_flash_attn_config_rejection_reason(**arguments)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_heads", "num_kv_heads"),
    [
        (1, 32, 8),
        (2, 16, 4),
        (4, 8, 2),
    ],
)
def test_supported_eager_qwen3_8b_d128_geometry(
    tensor_parallel_size: int, num_heads: int, num_kv_heads: int
) -> None:
    config = _supported_config(
        tensor_parallel_size=tensor_parallel_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    assert (
        _config_reason(
            config,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=128,
            scale=1.0 / (128**0.5),
        )
        is None
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_heads", "num_kv_heads"),
    [
        (1, 16, 4),
        (1, 16, 2),
        (2, 8, 2),
        (2, 8, 1),
        (4, 4, 1),
        (8, 2, 1),
        (2, 12, 2),
        (4, 6, 1),
    ],
)
def test_supported_eager_unquantized_geometry(
    tensor_parallel_size: int, num_heads: int, num_kv_heads: int
) -> None:
    config = _supported_config(
        tensor_parallel_size=tensor_parallel_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    assert (
        _config_reason(config, num_heads=num_heads, num_kv_heads=num_kv_heads) is None
    )


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_heads", "num_kv_heads"),
    [(1, 8, 2), (2, 16, 4), (2, 4, 1), (4, 8, 2), (8, 4, 1)],
)
def test_geometry_support_does_not_depend_on_tp_size(
    tensor_parallel_size: int, num_heads: int, num_kv_heads: int
) -> None:
    config = _supported_config(
        tensor_parallel_size=tensor_parallel_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    assert (
        _config_reason(config, num_heads=num_heads, num_kv_heads=num_kv_heads) is None
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_heads": 7}, "TP-local attention geometry"),
        ({"scale": 0.1}, "attention scale"),
        ({"quant_type": "FP8"}, "attention quant type"),
        ({"post_quant_kv_int8": True}, "post-quantization"),
        ({"sliding_window": (4095, 0)}, "sliding-window"),
        ({"logits_soft_cap": 30.0}, "soft cap"),
        ({"sinks": object()}, "sinks"),
    ],
)
def test_unsupported_attention_options_are_rejected(overrides, match) -> None:
    assert match in _config_reason(**overrides)


@pytest.mark.parametrize("quant_type", ["KINT8", "KVINT8"])
def test_int8_quant_types_are_accepted(quant_type: str) -> None:
    assert _config_reason(quant_type=quant_type) is None


def test_model_name_does_not_gate_supported_attention_geometry() -> None:
    config = _supported_config()
    config.model_config.architectures = ["FutureDecoderForCausalLM"]
    assert _config_reason(config) is None


def test_context_parallel_config_is_rejected() -> None:
    config = _supported_config()
    config.parallel_config.decode_context_parallel_size = 2
    assert "context parallelism" in _config_reason(config)

    config = _supported_config()
    config.quant_config = object()
    assert _config_reason(config) is None


def test_ixaite_backend_declares_layout_and_uniform_graph_support() -> None:
    assert flash_attn.IxaiteFlashAttentionBackend.get_required_kv_cache_layout() == (
        "HND"
    )
    assert (
        flash_attn.IxaiteFlashAttentionMetadataBuilder.query_len_support
        == flash_attn.QueryLenSupport.UNIFORM
    )
    assert (
        flash_attn.IxaiteFlashAttentionMetadataBuilder._cudagraph_support
        == flash_attn.AttentionCGSupport.UNIFORM_BATCH
    )


def test_ixaite_backend_splits_hybrid_physical_blocks_to_page_16() -> None:
    """The GDN-aligned physical block does not become the kernel cache page."""
    physical_block_size = 2112
    assert (
        select_common_block_size(
            physical_block_size, [flash_attn.IxaiteFlashAttentionBackend]
        )
        == 16
    )
    assert (
        select_common_block_size(
            physical_block_size, [flash_attn.IluFlashAttentionBackend]
        )
        == 64
    )


@pytest.mark.parametrize(
    ("num_prefills", "max_query_len", "expected"),
    [(0, 1, True), (1, 1, False), (0, 2, True), (0, 4, True)],
)
def test_ixaite_metadata_rejects_unverified_spec_graph_decode(
    monkeypatch, num_prefills, max_query_len, expected
) -> None:
    metadata = SimpleNamespace(
        num_prefills=num_prefills,
        decode=SimpleNamespace(max_query_len=max_query_len, use_graph=True),
    )
    monkeypatch.setattr(
        flash_attn.IluFlashAttentionMetadataBuilder,
        "build",
        lambda self, common_prefix_len, common_attn_metadata, fast_build=False: (
            metadata
        ),
    )
    builder = object.__new__(flash_attn.IxaiteFlashAttentionMetadataBuilder)

    if max_query_len > 1:
        with pytest.raises(RuntimeError, match="multi-token graph"):
            builder.build(0, Mock())
        return
    result = builder.build(0, Mock())

    assert result.decode.use_graph is expected


def test_cascade_writes_new_kv_before_reading_segments(monkeypatch):
    """A continuation's suffix includes newly projected tokens in the same forward."""
    impl = object.__new__(flash_attn.IxaiteFlashAttentionImpl)
    impl.kv_cache_dtype = "auto"
    impl.kv_sharing_target_layer_name = None
    events = []
    monkeypatch.setattr(
        flash_attn.ops, "reshape_and_cache_flash", lambda *args: events.append("write")
    )
    impl._forward_cascade = lambda *args: events.append("cascade") or args[-1]
    metadata = SimpleNamespace(
        use_cascade=True,
        num_actual_tokens=1,
        num_decode_tokens=0,
        slot_mapping=torch.tensor([0]),
    )
    q = torch.zeros(1, 12, 256, dtype=torch.bfloat16)
    kv = torch.zeros(2, 1, 2, 16, 256, dtype=q.dtype)
    layer = SimpleNamespace(_k_scale=1.0, _v_scale=1.0)
    impl._forward_unquantized(layer, q, q, q, kv, metadata, torch.empty_like(q))
    assert events == ["write", "cascade"]


def test_cascade_routes_both_segments_and_merge_to_selected_provider(monkeypatch):
    """Provider selection applies to all three cascade calls, including noncausal prefix."""
    varlen = Mock(return_value=(torch.zeros(2, 12, 256), torch.zeros(12, 2)))
    merge = Mock()
    q = torch.zeros(2, 12, 256)
    cache = torch.zeros(4, 2, 16, 256)
    table = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int32)
    flash_attn.cascade_attention(
        torch.empty_like(q),
        q,
        cache,
        cache,
        torch.tensor([0, 1, 2]),
        1,
        torch.tensor([0, 2]),
        torch.tensor([0, 32]),
        torch.tensor([0, 1, 2]),
        33,
        1 / 16,
        None,
        (-1, -1),
        0,
        table,
        32,
        varlen_op=varlen,
        merge_op=merge,
    )
    assert varlen.call_count == 2 and merge.call_count == 1
    assert not varlen.call_args_list[0].kwargs["causal"]
    assert varlen.call_args_list[1].kwargs["causal"]
    assert torch.equal(varlen.call_args_list[1].kwargs["block_table"], table[:, 2:])
