# SPDX-License-Identifier: Apache-2.0
"""Native readers must consume shared cache writes before model benchmarking."""

from types import SimpleNamespace

import pytest
import torch

from tests.ixaite.benchmarks.e2e.int8_cache_bridge import CacheReadAttention
from vllm_iluvatar.v1.attention.backends import ilu_flash_attn as baseline


def _metadata(tokens, length, table, *, prefill):
    """Use permuted physical pages for prefill and the following decode append."""
    start = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    seq = torch.tensor([length], device="cuda", dtype=torch.int32)
    positions = torch.arange(length - tokens, length, device="cuda")
    slots = (table[0, positions // 16].long() * 16 + positions % 16).contiguous()
    return SimpleNamespace(
        use_cascade=False,
        causal=True,
        num_actual_tokens=tokens,
        num_decode_tokens=0 if prefill else tokens,
        num_prefills=int(prefill),
        num_decodes=int(not prefill),
        slot_mapping=slots,
        max_seq_len=length,
        prefill=SimpleNamespace(
            query_start_loc=start,
            key_start_loc=torch.tensor([0, length], device="cuda", dtype=torch.int32),
            max_query_len=tokens,
            block_table=table,
        )
        if prefill
        else None,
        decode=None
        if prefill
        else SimpleNamespace(
            query_start_loc=start,
            seq_lens=seq,
            max_query_len=tokens,
            block_table=table,
            use_graph=True,
            max_decode_seq_len=64,
        ),
    )


def _reference(query, key, value, key_scale, value_scale, table, length):
    """Compute FP32 attention from each reader's actual quantized operands."""
    quant_q, query_scale = baseline.ixf_ops.scaled_int8_quant_for_attn(
        query, 2, transpose_scale=False
    )
    q = (quant_q.float() * query_scale[..., None]).transpose(0, 1)
    positions = torch.arange(length, device=query.device)
    pages, offsets = table[0, positions // 16].long(), positions % 16
    k = (key[pages, :, offsets].float() * key_scale[pages, :, offsets, None]).transpose(
        0, 1
    )
    v = value[pages, :, offsets].float()
    if value_scale is not None:
        v = v * value_scale
    ratio = query.shape[1] // key.shape[1]
    k = k.repeat_interleave(ratio, dim=0)
    v = v.transpose(0, 1).repeat_interleave(ratio, dim=0)
    scores = (q @ k.transpose(1, 2)) * query.shape[-1] ** -0.5
    query_positions = (
        torch.arange(query.shape[0], device=query.device) + length - query.shape[0]
    )
    scores.masked_fill_(positions[None, :] > query_positions[:, None], float("-inf"))
    return (scores.softmax(-1) @ v).transpose(0, 1).to(query.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Iluvatar GPU required")
@pytest.mark.parametrize("mode", ["KINT8", "KVINT8"])
@pytest.mark.parametrize("dim,qheads", [(128, 28), (256, 16)])
@pytest.mark.parametrize("packed_qkv", [False, True])
def test_native_readers_consume_shared_cache(
    monkeypatch, mode, dim, qheads, packed_qkv
):
    """Compare native fused writes with independent shared writes and graph replay."""
    import ixaite

    ixaite.initialize("ivcore11")
    from ixaite.ops.attention.int8_kv_cache import Int8KVCache

    torch.manual_seed(3910)
    heads = 4
    native_data = torch.zeros(
        3 if mode == "KINT8" else 2, 4, heads, 16, dim, dtype=torch.int8, device="cuda"
    )
    shared_data = native_data.clone()
    native_scale = torch.zeros(4, heads, 16, device="cuda")
    shared_scale = torch.zeros_like(native_scale)
    static = torch.full((heads, dim), 0.03, device="cuda") if mode == "KVINT8" else None
    if mode == "KINT8":
        k, v = baseline._get_k_int8_cache_views(shared_data, torch.bfloat16)
    else:
        k, v = shared_data.unbind(0)
    binding = Int8KVCache(k, v, shared_scale, static)
    native_k, native_v = (
        baseline._get_k_int8_cache_views(native_data, torch.bfloat16)
        if mode == "KINT8"
        else native_data.unbind(0)
    )
    native_layer = SimpleNamespace(kv_cache_scale=[native_scale, static])
    shared_layer = SimpleNamespace(_ixaite_int8_cache=binding)
    implementation = SimpleNamespace(
        _maybe_forward_encoder=lambda *args: None,
        dcp_world_size=1,
        kv_sharing_target_layer_name=None,
        scale=dim**-0.5,
        num_kv_heads=heads,
        sliding_window=(-1, -1),
        alibi_slopes=None,
        logits_soft_cap=0.0,
    )
    original = (
        baseline.IluFlashAttentionImpl._forward_k_int8
        if mode == "KINT8"
        else baseline.IluFlashAttentionImpl._forward_kv_int8
    )
    table = torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32)
    q = torch.randn(17, qheads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(17, heads, dim, device="cuda", dtype=q.dtype)
    value = torch.randn_like(key)
    if packed_qkv:
        packed = torch.randn(17, qheads + 2 * heads, dim, device="cuda", dtype=q.dtype)
        q, key, value = packed.split((qheads, heads, heads), dim=1)
        assert not key.is_contiguous() and not value.is_contiguous()
    metadata = _metadata(17, 17, table, prefill=True)
    want, actual = torch.empty_like(q), torch.empty_like(q)
    original(implementation, native_layer, q, key, value, native_data, metadata, want)

    def forbidden_writer(*args, **kwargs):
        raise AssertionError("shared-cache path must not call a native cache writer")

    with monkeypatch.context() as guard:
        guard.setattr(torch.ops._C, "reshape_and_cache_mix", forbidden_writer)
        guard.setattr(torch.ops._C, "reshape_and_cache_with_scale", forbidden_writer)
        CacheReadAttention._forward_cache(
            implementation, shared_layer, q, key, value, shared_data, metadata, actual
        )

    def check(length):
        # Native and FlyDSL quantizers can differ by one level at rounding ties.
        torch.testing.assert_close(binding.key, native_k, atol=1, rtol=0)
        torch.testing.assert_close(
            binding.value, native_v, atol=0 if static is None else 1, rtol=0
        )
        torch.testing.assert_close(shared_scale, native_scale, atol=1e-7, rtol=1e-5)
        for output, kbuf, vbuf, scales in (
            (actual, binding.key, binding.value, shared_scale),
            (want, native_k, native_v, native_scale),
        ):
            reference = _reference(q, kbuf, vbuf, scales, static, table, length)
            torch.testing.assert_close(output, reference, atol=0.03, rtol=0.03)

    check(17)

    # Replay changes the newest token at stable addresses; old pages stay valid.
    q, key, value = q[:1].clone(), key[:1].clone(), value[:1].clone()
    actual, want = torch.empty_like(q), torch.empty_like(q)
    metadata = _metadata(1, 18, table, prefill=False)

    def native_launch():
        original(
            implementation, native_layer, q, key, value, native_data, metadata, want
        )

    def shared_launch():
        CacheReadAttention._forward_cache(
            implementation, shared_layer, q, key, value, shared_data, metadata, actual
        )

    graphs = []
    for launch in (native_launch, shared_launch):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graphs.append(graph)
    for _ in range(2):
        q.normal_()
        key.normal_()
        value.normal_()
        for graph in graphs:
            graph.replay()
        check(18)
