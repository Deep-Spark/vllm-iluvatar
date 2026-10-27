# SPDX-License-Identifier: Apache-2.0
"""Rotation ownership and mixed-slice dispatch without a GPU context.

Native math is replaced explicitly by recorders here. GPU arithmetic and real
projection-chain tests are separate; these tests catch missing/double rotation
and positional slicing errors in the actual model control flow.
"""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_iluvatar.models.deepseek_v41 import attention as module

Attention = module.IluDeepseekV41Attention


def attention(format, ratio):
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.compressed_kv_format = format
    layer.compress_ratio = ratio
    layer.n_local_heads, layer.head_dim = 2, 512
    layer.rotary_emb = NS(cos_sin_cache=torch.empty(8, 64))
    return layer


@pytest.mark.parametrize("format", [None, "i4g32", "i8g64"])
@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("rows", [0, 6, 129])
def test_forward_has_exactly_one_inverse_owner(format, ratio, rows, monkeypatch):
    layer = attention(format, ratio)
    states = torch.zeros(rows, 2, 512, dtype=torch.bfloat16)
    positions = torch.arange(rows)
    events = []
    layer._run_parallel_input_projections = lambda x: (x, None, None)
    layer._split_qkv_and_norm = lambda x: (x, x)
    layer._o_proj = lambda x: x

    def prepare(*args):
        args[-1].zero_()
        if layer._uses_fused_inverse_rope():
            events.append("reader")
            args[-1].add_(3)

    def rotate(x, pos, table, inverse):
        assert inverse and pos is positions and table is layer.cos_sin
        events.append("external")
        x.add_(3)

    layer._prepare_and_attn = prepare
    monkeypatch.setattr(module, "apply_rope", rotate)
    result = layer(positions, states)
    assert events == ["reader" if format is not None and ratio > 0 else "external"]
    assert torch.all(result == 3)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("format", ["i4g32", "i8g64"])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    "old,new", [(6, 0), (12, 0), (18, 0), (0, 128), (0, 129), (6, 128), (12, 129)]
)
@pytest.mark.parametrize("restore_decode", [False, True])
def test_actual_mqa_routes_pass_exact_query_positions(
    format, ratio, old, new, restore_decode, monkeypatch
):
    from ixformer.inference.functions.deepseek import (
        quantized_sparse_mla as decode_module,
    )
    from ixformer.inference.functions.deepseek import (
        quantized_sparse_mla_prefill as prefill_module,
    )

    layer = attention(format, ratio)
    count = old + new
    payload = torch.empty(1, 64, 256, dtype=torch.uint8)
    scales = torch.empty(1, 64, 16, dtype=torch.float16)
    scratch = torch.empty(1, dtype=torch.uint8)
    layer.kv_cache_dtype = "auto"
    layer.scale = 512**-0.5
    layer.attn_sink = torch.zeros(2)
    layer.compressed_cache_prefix = "main"
    layer.compressed_decode_restore = restore_decode
    layer.topk_indices_buffer = torch.zeros(count, 512, dtype=torch.int32)
    layer.swa_cache_layer = NS(
        prefix="swa", kv_cache=torch.empty(1, 64, 512, dtype=torch.bfloat16)
    )
    layer.compressed_kv_workspace = NS(
        capturing=False,
        read_scope=lambda: nullcontext(None),
        acquire=lambda **kwargs: nullcontext(scratch),
    )
    layer._compressed_kv_cache = lambda: payload
    layer._compressed_kv_scales = lambda: scales
    swa = NS(
        num_decodes=int(old > 0),
        num_prefills=int(new > 0),
        num_decode_tokens=old,
        num_prefill_tokens=new,
        decode_swa_indices=torch.zeros(old, 1, 128, dtype=torch.int32),
        decode_swa_lens=torch.full((old,), 128, dtype=torch.int32),
        prefill_swa_indices=torch.zeros(new, 1, 128, dtype=torch.int32),
        prefill_swa_lens=torch.full((new,), 128, dtype=torch.int32),
        is_valid_token=torch.ones(count, dtype=torch.bool),
        token_to_req_indices=torch.zeros(count, dtype=torch.int32),
    )
    metadata = {
        "swa": swa,
        "main": NS(block_size=64, block_table=torch.zeros(1, 1, dtype=torch.int32)),
    }
    monkeypatch.setattr(
        module, "get_forward_context", lambda: NS(attn_metadata=metadata)
    )
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS4_PREFILL_OPT", False)
    monkeypatch.setattr(module.envs, "VLLM_ILU_DS4_PREFILL_AS_DECODE", True)
    indices = torch.zeros(count, 1, 512, dtype=torch.int32)
    lengths = torch.full((count,), 512, dtype=torch.int32)
    selection = NS(
        compact_plan=None, for_queries=lambda s: (indices[s], lengths[s], None)
    )
    layer._prepare_compressed_family = lambda *args: selection
    monkeypatch.setattr(
        torch.ops._C,
        "v41_global_topk_indices_and_lens",
        lambda topk, *args: (topk, lengths[: topk.shape[0]]),
    )
    events = []

    def reader(label):
        def call(*args, positions, cos_sin_cache, output, **kwargs):
            assert cos_sin_cache is layer.cos_sin
            assert kwargs["workspace"] is scratch
            assert positions.shape[0] == args[0].shape[0] == output.shape[0]
            events.append((label, positions.clone()))
            output.fill_(1)

        return call

    monkeypatch.setattr(decode_module, "quantized_sparse_mla", reader("decode"))
    monkeypatch.setattr(
        prefill_module, "quantized_sparse_mla_prefill", reader("prefill")
    )
    positions = torch.arange(1048576 - count, 1048576, dtype=torch.int64)
    q = torch.zeros(count, 2, 512, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    layer.forward_mqa(q, torch.empty(0), positions, output)
    expected = []
    if new:
        expected.append(("prefill", positions[old:]))
    if old:
        expected.append(("decode", positions[:old]))
    assert len(events) == len(expected)
    for (label, actual), (expected_label, pos) in zip(events, expected):
        assert label == expected_label
        torch.testing.assert_close(actual, pos)
    assert torch.all(output == 1)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("format", ["i4g32", "i8g64"])
def test_missing_quantized_payload_fails_before_unrotated_output(format):
    layer = attention(format, 1)
    q = torch.empty(1, 2, 512, dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, 2, dtype=torch.int32)
    lengths = torch.ones(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="main payload and scale planes"):
        layer._sparse_mla(
            q,
            indices,
            lengths,
            None,
            indices,
            lengths,
            q,
            positions=torch.zeros(1, dtype=torch.int64),
        )
    assert not torch.cuda.is_initialized()
