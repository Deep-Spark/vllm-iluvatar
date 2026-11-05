# SPDX-License-Identifier: Apache-2.0
"""KINT8/KVINT8 ixaite flash-attn: gates, binding, and baseline alignment."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_iluvatar.backends.ixaite.attention import flash_attn, kv_cache


@pytest.fixture(autouse=True)
def reset_int8_provider(monkeypatch, tmp_path):
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
    kv_cache._RESHAPE_WITH_SCALE_IMPL = None
    kv_cache._RESHAPE_MIX_IMPL = None
    kv_cache._QUANT_IMPL = None
    yield
    kv_cache._RESHAPE_WITH_SCALE_IMPL = None
    kv_cache._RESHAPE_MIX_IMPL = None
    kv_cache._QUANT_IMPL = None


def _supported_config(
    *, tensor_parallel_size: int = 1, num_heads: int = 16, num_kv_heads: int = 4
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            enforce_eager=True,
            get_num_attention_heads=lambda parallel_config: num_heads,
            get_num_kv_heads=lambda parallel_config: num_kv_heads,
            hf_config=SimpleNamespace(),
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        speculative_config=None,
        compilation_config=None,
    )


def _bind_int8_mocks(monkeypatch):
    quant = Mock(side_effect=lambda *a, **k: (Mock(), Mock()))
    with_scale = Mock()
    mix = Mock()

    def loader(name, func):
        mapping = {
            "scaled_int8_quant_for_attn": quant,
            "reshape_and_cache_with_scale": with_scale,
            "reshape_and_cache_mix": mix,
        }
        impl = mapping.get(name)
        if impl is None:
            return None, f"unexpected {name}"
        return impl, "supported"

    monkeypatch.setattr(kv_cache, "load_ixaite_python_op", loader)
    return quant, with_scale, mix


def test_kint8_init_binds_int8_providers_not_bf16_ops(monkeypatch):
    monkeypatch.setattr(flash_attn.envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KINT8")
    _bind_int8_mocks(monkeypatch)
    bf16_loader = Mock(side_effect=AssertionError("bf16 ops must not load"))
    monkeypatch.setattr(flash_attn, "load_ixaite_python_op", bf16_loader)
    impl = flash_attn.IxaiteFlashAttentionImpl(
        16, 256, 1 / 16, 4, None, None, "auto"
    )
    assert impl.quant_type == "KINT8"
    assert impl._ops is None
    assert kv_cache._QUANT_IMPL is not None
    bf16_loader.assert_not_called()


def test_kvint8_requires_v_scale_file(monkeypatch, tmp_path):
    monkeypatch.setattr(flash_attn.envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KVINT8")
    monkeypatch.setattr(flash_attn.envs, "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH", "")
    _bind_int8_mocks(monkeypatch)
    with pytest.raises(RuntimeError, match="V_SCALE_PATH"):
        flash_attn.IxaiteFlashAttentionImpl(16, 256, 1 / 16, 4, None, None, "auto")

    path = tmp_path / "v.pt"
    torch.save({"0": torch.ones(4, 256)}, path)
    monkeypatch.setattr(flash_attn.envs, "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH", str(path))
    impl = flash_attn.IxaiteFlashAttentionImpl(16, 256, 1 / 16, 4, None, None, "auto")
    assert impl.quant_type == "KVINT8"


def test_selector_allows_kint8(monkeypatch):
    monkeypatch.setattr(flash_attn.envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KINT8")
    from vllm.v1.attention.selector import AttentionSelectorConfig

    selector = AttentionSelectorConfig(
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
    assert flash_attn.get_flash_attn_backend_class_path(selector).endswith(
        "flash_attn.IxaiteFlashAttentionBackend"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
def test_quant_matches_ixformer_abi():
    """ixaite quant must match ixformer on the published ABI (amax/127)."""
    import ixformer.inference.functions as ixf_ops
    import ixaite

    arch = os.environ.get("IXAITE_ARCH", "ivcore11")
    ixaite.initialize(arch)
    kv_cache.configure_int8_attention_quant_provider()

    torch.manual_seed(0)
    x = torch.randn(8, 4, 256, device="cuda", dtype=torch.bfloat16)
    i8_ref, scale_ref = ixf_ops.scaled_int8_quant_for_attn(x, 2, transpose_scale=False)
    i8_act, scale_act = kv_cache.scaled_int8_quant_for_attn(
        x, 2, transpose_scale=False
    )
    # ABI is amax/127 with round-clamp; bf16 amax can leave +/-1 LSB on int8.
    assert int((i8_ref.to(torch.int16) - i8_act.to(torch.int16)).abs().max()) <= 1
    torch.testing.assert_close(scale_ref, scale_act, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("use_graph", [False, True])
def test_kint8_decode_aligns_with_baseline(monkeypatch, use_graph):
    """Decode (optional cudagraph) output matches baseline INT8 forward."""
    import ixaite

    from vllm_iluvatar.v1.attention.backends import ilu_flash_attn as baseline

    arch = os.environ.get("IXAITE_ARCH", "ivcore11")
    ixaite.initialize(arch)
    monkeypatch.setattr(flash_attn.envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KINT8")
    monkeypatch.setattr(baseline.envs, "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", "KINT8")

    num_heads, num_kv_heads, head_size = 8, 2, 256
    query_len = 1
    context_len = 17
    block_size = 16
    num_blocks = 4
    scale = head_size**-0.5

    torch.manual_seed(7)
    q = torch.randn(
        query_len, num_heads, head_size, device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn(
        query_len, num_kv_heads, head_size, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    # KINT8 layout: 3 planes (i8 key + 2 bf16-as-bytes value planes).
    cache_base = torch.zeros(
        3,
        num_blocks,
        num_kv_heads,
        block_size,
        head_size,
        device="cuda",
        dtype=torch.int8,
    )
    cache_ixaite = cache_base.clone()
    cache_base_line = cache_base.clone()
    key_scale = torch.zeros(
        num_blocks, num_kv_heads, block_size, device="cuda", dtype=torch.float32
    )
    layer = SimpleNamespace(kv_cache_scale=[key_scale.clone(), None], layer_name="0")
    layer_b = SimpleNamespace(
        kv_cache_scale=[key_scale.clone(), None], layer_name="0"
    )

    slots = torch.arange(
        context_len - query_len, context_len, device="cuda", dtype=torch.int64
    )
    seq_lens = torch.tensor([context_len], device="cuda", dtype=torch.int32)
    query_start = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
    table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1)
    metadata = SimpleNamespace(
        causal=True,
        use_cascade=False,
        num_actual_tokens=query_len,
        num_decode_tokens=query_len,
        slot_mapping=slots,
        max_seq_len=context_len,
        num_prefills=0,
        num_decodes=1,
        prefill=None,
        decode=SimpleNamespace(
            query_start_loc=query_start,
            seq_lens=seq_lens,
            max_query_len=query_len,
            max_decode_seq_len=64,
            block_table=table,
            use_graph=use_graph,
        ),
    )

    kv_cache.configure_int8_attention_providers()
    prefill_tokens = context_len - query_len
    if prefill_tokens > 0:
        k_hist = torch.randn(
            prefill_tokens,
            num_kv_heads,
            head_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_hist = torch.randn_like(k_hist)
        i8_k, k_s = kv_cache.scaled_int8_quant_for_attn(k_hist, 2, False)
        hist_slots = torch.arange(prefill_tokens, device="cuda", dtype=torch.int64)
        i8_key_view, value_view = baseline._get_k_int8_cache_views(
            cache_ixaite, torch.bfloat16
        )
        kv_cache.reshape_and_cache_mix(
            i8_k,
            v_hist,
            k_s,
            i8_key_view,
            value_view,
            layer.kv_cache_scale[0],
            hist_slots,
        )
        cache_base_line.copy_(cache_ixaite)
        layer_b.kv_cache_scale[0].copy_(layer.kv_cache_scale[0])

    out_i = torch.empty_like(q)
    out_b = torch.empty_like(q)

    impl_i = flash_attn.IxaiteFlashAttentionImpl(
        num_heads, head_size, scale, num_kv_heads, None, None, "auto"
    )
    impl_b = object.__new__(baseline.IluFlashAttentionImpl)
    impl_b.num_heads = num_heads
    impl_b.num_kv_heads = num_kv_heads
    impl_b.head_size = head_size
    impl_b.scale = scale
    impl_b.sliding_window = (-1, -1)
    impl_b.alibi_slopes = None
    impl_b.logits_soft_cap = 0
    impl_b.kv_sharing_target_layer_name = None
    impl_b.attn_type = flash_attn.AttentionType.DECODER
    impl_b.dcp_world_size = 1
    impl_b.quant_type = "KINT8"

    impl_i.forward(layer, q, key, value, cache_ixaite, metadata, out_i)
    impl_b._forward_k_int8(
        layer_b, q, key, value, cache_base_line, metadata, out_b
    )
    # Same ixformer paged kernel; residual comes from quant LSB differences.
    torch.testing.assert_close(out_i, out_b, rtol=3e-2, atol=3e-2)

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            impl_i.forward(layer, q, key, value, cache_ixaite, metadata, out_i)
        seq_lens.fill_(context_len + 1)
        slots.copy_(
            torch.arange(
                context_len,
                context_len + query_len,
                device="cuda",
                dtype=slots.dtype,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out_i).all()
