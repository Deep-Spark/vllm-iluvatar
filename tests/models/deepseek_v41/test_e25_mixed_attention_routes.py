"""Actual frozen prefill-as-decode branch counts for the E25 suffix."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import attention
from vllm_iluvatar.models.deepseek_v41.attention import IluDeepseekV41Attention


@pytest.mark.parametrize("old_tokens", [0, 2, 10])
def test_e25_suffix_uses_separate_prefill_and_old_decode_calls(old_tokens, monkeypatch):
    new_tokens = 128
    tokens = old_tokens + new_tokens
    owner = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(owner)
    owner.kv_cache = torch.empty(3, 64, 512, dtype=torch.bfloat16)
    index_sources = (20, 24, 28, 32, 36)
    layers = []
    metadata = {
        "owner20": SimpleNamespace(
            block_size=64, block_table=torch.zeros(1, 1, dtype=torch.int32)
        )
    }
    calls, maps = [], []
    module = __import__(
        IluDeepseekV41Attention.__module__, fromlist=["get_forward_context"]
    )
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS4_PREFILL_OPT", False)
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS4_PREFILL_AS_DECODE", True)

    def mapper(
        topk,
        reqs,
        block_table,
        page_rows,
        valid,
        positions,
        ratio,
        slot_limit,
        pad_walk,
    ):
        maps.append(topk.shape[0])
        assert page_rows == 64 and ratio == 1 and not pad_walk
        assert reqs.numel() == valid.numel() == positions.numel() == topk.shape[0]
        return topk.clone(), torch.full(
            (topk.shape[0],), topk.shape[1], dtype=torch.int32
        )

    monkeypatch.setattr(module.ixfops, "global_topk_indices_and_lens", mapper)
    for layer_id in range(21, 40):
        layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio, layer.compressed_kv_format = 1, None
        layer.is_kv_source = layer.is_kv_forwarder = False
        layer.compressed_cache_prefix = "owner20"
        layer._static_forward_context = {"owner20": owner}
        layer.index_source_layer_id = max(
            source for source in index_sources if source <= layer_id
        )
        layer.topk_indices_buffer = torch.zeros(tokens, 4, dtype=torch.int32)
        prefix = f"swa{layer_id}"
        layer.swa_cache_layer = SimpleNamespace(prefix=prefix, kv_cache=torch.empty(0))
        metadata[prefix] = SimpleNamespace(
            num_decodes=bool(old_tokens),
            num_prefills=1,
            num_decode_tokens=old_tokens,
            num_prefill_tokens=new_tokens,
            decode_swa_indices=torch.zeros(old_tokens, 1, 2, dtype=torch.int32),
            decode_swa_lens=torch.full((old_tokens,), 2, dtype=torch.int32),
            prefill_swa_indices=torch.zeros(new_tokens, 1, 2, dtype=torch.int32),
            prefill_swa_lens=torch.full((new_tokens,), 2, dtype=torch.int32),
            is_valid_token=torch.ones(tokens, dtype=torch.bool),
            token_to_req_indices=torch.zeros(tokens, dtype=torch.int32),
        )

        def reader(
            q,
            swa_indices,
            swa_lens,
            cache,
            main_indices,
            main_lens,
            output,
            *,
            prefill=False,
            layer_id=layer_id,
            **kwargs,
        ):
            calls.append((layer_id, "prefill" if prefill else "decode", q.shape[0]))
            assert cache is owner.kv_cache
            assert swa_indices.shape[0] == swa_lens.numel() == q.shape[0]
            assert main_indices.shape[0] == main_lens.numel() == q.shape[0]
            output.fill_(1 if prefill else 2)

        layer._sparse_mla = reader
        layers.append(layer)
    monkeypatch.setattr(
        module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    q = torch.zeros(tokens, 1, 512, dtype=torch.bfloat16)
    positions = torch.arange(tokens)
    for layer in layers:
        output = torch.empty_like(q)
        layer.forward_mqa(q, torch.empty(0), positions, output)
        assert (output[:old_tokens] == 2).all()
        assert (output[old_tokens:] == 1).all()
    expected_order = []
    for layer_id in range(21, 40):
        expected_order.append((layer_id, "prefill", new_tokens))
        if old_tokens:
            expected_order.append((layer_id, "decode", old_tokens))
    assert calls == expected_order
    assert len(calls) == (38 if old_tokens else 19)
    assert maps == [call[2] for call in calls]
    assert Counter(layer.index_source_layer_id for layer in layers) == {
        20: 3,
        24: 4,
        28: 4,
        32: 4,
        36: 4,
    }


@pytest.mark.parametrize(
    ("decode_applied", "prefill_applied", "expected_rows"),
    [
        (False, False, [0, 1, 2, 3]),
        (True, False, [2, 3]),
        (False, True, [0, 1]),
        (True, True, []),
    ],
)
def test_output_projection_applies_inverse_rope_only_to_unrotated_rows(
    monkeypatch, decode_applied, prefill_applied, expected_rows
):
    layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer._uses_fused_inverse_rope = lambda: False
    layer._decode_inv_rope_applied = decode_applied
    layer._prefill_inv_rope_applied = prefill_applied
    layer._num_decode_tokens = lambda rows: 2
    layer._o_proj = lambda values: values
    layer.wo_a = None
    positions = torch.arange(4)
    values = torch.zeros(4, 2)
    applied = []

    def apply_rope(q, positions, cos_sin, *, inverse):
        assert inverse
        applied.extend(positions.tolist())
        q.add_(1)

    monkeypatch.setattr(attention, "apply_rope", apply_rope)
    assert layer._project_attention_output(values, positions) is values
    assert applied == expected_rows


def test_global_indices_cache_tracks_shared_scratch_contents(monkeypatch):
    context = SimpleNamespace()
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention, "is_forward_context_available", lambda: True)
    scratch_indices = torch.empty((5, 1), dtype=torch.int32)
    scratch_lens = torch.empty(5, dtype=torch.int32)
    consumer = SimpleNamespace(
        index_source_layer_id=20,
        kv_source_layer_id=20,
        compress_ratio=1,
        _global_topk=scratch_indices,
        _global_topk_lens=scratch_lens,
    )
    uncached = SimpleNamespace(
        compress_ratio=1,
        _global_topk=scratch_indices,
        _global_topk_lens=scratch_lens,
    )
    topk = torch.tensor([[10], [20], [30], [40], [50]], dtype=torch.int32)
    reqs = torch.zeros(5, dtype=torch.int32)
    table = torch.tensor([[100]], dtype=torch.int32)
    valid = torch.ones(5, dtype=torch.bool)
    positions = torch.arange(5)
    calls = []

    def mapper(
        local, req, table, block_size, valid, positions, ratio, *, output, output_lens
    ):
        calls.append(len(local))
        output.copy_(local + table[0, 0])
        output_lens.copy_(positions)

    monkeypatch.setattr(
        attention.ixfops, "compute_global_topk_indices_and_lens", mapper
    )

    def check(layer, start, end, block_table=table):
        indices, lengths = IluDeepseekV41Attention._global_indices(
            layer,
            topk[start:end],
            reqs[start:end],
            block_table,
            64,
            valid[start:end],
            positions[start:end],
        )
        assert torch.equal(indices, topk[start:end] + block_table[0, 0])
        assert torch.equal(lengths, positions[start:end].to(lengths.dtype))

    check(consumer, 2, 5)
    check(consumer, 2, 5)
    assert calls == [3]
    check(consumer, 0, 2)
    check(consumer, 2, 5)
    check(uncached, 0, 2)
    check(consumer, 2, 5)
    check(consumer, 2, 5, table + 100)
    assert calls == [3, 2, 3, 2, 3, 3]


@pytest.mark.parametrize("fused", [False, True])
def test_fp8_reader_reports_inverse_rope_applied_even_on_fallback(monkeypatch, fused):
    layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer.compressed_kv_format = None
    layer.kv_cache_dtype = "fp8_e4m3"
    layer.compress_ratio = 0
    layer._decode_inv_rope_applied = False
    layer._prefill_inv_rope_applied = False
    layer.scale = layer.kv_scale = 1.0
    layer.attn_sink = None
    layer.wo_a = None
    layer.swa_cache_layer = SimpleNamespace(kv_cache=torch.empty(0))
    layer._q_fp8_pt = torch.empty((2, 1, 2))
    layer._q_scale_pt = torch.empty(2)
    layer._q_quant_workspace_pt = None
    layer._tile_window = lambda rows: rows
    q = torch.empty_like(layer._q_fp8_pt)
    output = torch.zeros_like(q)
    indices = torch.zeros((2, 1), dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    positions = torch.arange(2)
    cos_sin = torch.empty(0)
    monkeypatch.setattr(attention, "_PT_MLA_HEADS", (1,))
    monkeypatch.setattr(attention.ixfops, "quant_mla_q_fp8_pt", lambda *a, **kw: None)

    def reader(*args, positions, cos_sin_cache):
        args[4].add_(1)
        return args[4], fused

    monkeypatch.setattr(attention.ixfops, "sparse_mla_fp8_pt_inv_rope", reader)
    layer._sparse_mla(
        q,
        indices,
        lengths,
        None,
        None,
        None,
        output,
        positions=positions,
        cos_sin_cache=cos_sin,
        prefill=True,
    )
    assert layer._prefill_inv_rope_applied
    assert not layer._decode_inv_rope_applied
    assert torch.equal(output, torch.ones_like(output))


@pytest.mark.parametrize("supports_rope", [False, True])
@pytest.mark.parametrize("prefill_projection", [False, True])
def test_bf16_reader_leaves_inverse_rope_to_projection_when_needed(
    monkeypatch, supports_rope, prefill_projection
):
    from ixformer.inference.functions.ops.attn.flash_mla_sparse.float.flash_mla_sparse_with_kvcache import (
        flash_mla_sparse_with_kvcache_supports_inv_rope,
    )

    layer = IluDeepseekV41Attention.__new__(IluDeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer.compressed_kv_format = None
    layer.kv_cache_dtype = "bfloat16"
    layer.compress_ratio = 0
    layer._prefill_inv_rope_applied = False
    layer.scale = 1.0
    layer.attn_sink = None
    layer.wo_a = SimpleNamespace(prefill_bmm_b_fp8=torch.empty(0))
    layer.swa_cache_layer = SimpleNamespace(kv_cache=torch.empty(2, 64, 512))
    monkeypatch.setattr(
        attention.envs, "VLLM_ILU_V41_PREFILL_FP8_O_PROJ", prefill_projection
    )

    def supports(**kwargs):
        assert kwargs["q"].shape == (129, 1, 1, 2)
        return supports_rope

    monkeypatch.setattr(
        flash_mla_sparse_with_kvcache_supports_inv_rope.__module__
        + ".flash_mla_sparse_with_kvcache_supports_inv_rope",
        supports,
    )
    forwarded = []
    monkeypatch.setattr(
        attention,
        "flash_mla_sparse_with_kvcache",
        lambda **kwargs: forwarded.append(kwargs),
    )
    q = torch.empty(129, 1, 2)
    indices = torch.zeros((129, 1), dtype=torch.int32)
    lengths = torch.ones(129, dtype=torch.int32)
    layer._sparse_mla(
        q,
        indices,
        lengths,
        None,
        None,
        None,
        torch.empty_like(q),
        positions=torch.arange(129),
        cos_sin_cache=torch.empty(0),
        prefill=True,
    )
    expected = supports_rope and not prefill_projection
    assert len(forwarded) == 1
    assert ("positions" in forwarded[0]) == expected
    assert layer._prefill_inv_rope_applied == expected
