"""Unit tests for Step5 fusion gates used by prefill + decode / CUDA-graph / MTP."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_iluvatar.models.step5 import decode_fusions as df


class _Layer:
    def __init__(self, width=12288, accepts=True, parallel=True):
        self.input_is_parallel = parallel
        self.quant_method = SimpleNamespace(
            supports_prequantized_input=lambda _self: accepts
        )


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 2048, 3072, 4096])
def test_fusion_actual_tokens_prefill_all_rows_valid(rows):
    device = torch.device("cpu")
    actual = df.fusion_actual_tokens(None, rows, device=device)
    assert actual is not None
    assert actual.dtype == torch.int32 and actual.shape == (1,)
    assert int(actual.item()) == rows
    # Concurrent prefills must not share a mutable counter.
    other = df.fusion_actual_tokens(None, rows + 1, device=device)
    assert int(actual.item()) == rows
    assert int(other.item()) == rows + 1


@pytest.mark.parametrize(
    "rows,actual_n", [(8, 3), (16, 7), (32, 32), (64, 39), (128, 65)]
)
def test_fusion_actual_tokens_decode_metadata(rows, actual_n):
    """Decode CUDA-graph path must read query_start_loc[-1:], not a Python int."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for decode query_start_loc gate")
    attention = SimpleNamespace(layer_name="layers.0.self_attn.attn")
    starts = torch.zeros(actual_n + 1, dtype=torch.int32, device="cuda")
    starts[-1] = actual_n
    meta = SimpleNamespace(
        num_prefills=0,
        num_decodes=max(actual_n, 1),
        query_start_loc=starts,
    )
    ctx = SimpleNamespace(attn_metadata={attention.layer_name: meta})
    real_avail = df.is_forward_context_available
    real_get = df.get_forward_context
    df.is_forward_context_available = lambda: True
    df.get_forward_context = lambda: ctx
    try:
        got = df.fusion_actual_tokens(attention, rows, device=torch.device("cuda"))
        # Must be a view into query_start_loc (CUDA-graph safe), not the cache.
        assert got.data_ptr() == starts[-1:].data_ptr()
        assert int(got.item()) == actual_n
    finally:
        df.is_forward_context_available = real_avail
        df.get_forward_context = real_get


@pytest.mark.parametrize(
    "rows,actual_n",
    [
        (4, 4),  # MTP n=3, 1 seq → query_len 4
        (8, 8),
        (12, 12),
        (16, 7),  # padded capture size vs live actual
        (64, 39),  # DP4/TP2 MTP3 at concurrency 64
        (128, 65),
    ],
)
def test_fusion_actual_tokens_sparse_fp8_metadata_omits_counts(rows, actual_n):
    """Step5SparseFP8Metadata has query_start_loc only — must still be CG-safe."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for sparse-FP8 query_start_loc gate")
    attention = SimpleNamespace(layer_name="layers.0.self_attn.attn")
    starts = torch.zeros(actual_n + 1, dtype=torch.int32, device="cuda")
    starts[-1] = actual_n
    # Mirror fp8_attention.Step5SparseFP8Metadata: no num_prefills/num_decodes.
    meta = SimpleNamespace(
        query_start_loc=starts,
        num_actual_tokens=actual_n,
    )
    ctx = SimpleNamespace(attn_metadata={attention.layer_name: meta})
    real_avail = df.is_forward_context_available
    real_get = df.get_forward_context
    df.is_forward_context_available = lambda: True
    df.get_forward_context = lambda: ctx
    try:
        got = df.fusion_actual_tokens(attention, rows, device=torch.device("cuda"))
        assert got is not None
        assert got.data_ptr() == starts[-1:].data_ptr()
        assert int(got.item()) == actual_n
    finally:
        df.is_forward_context_available = real_avail
        df.get_forward_context = real_get


def test_fusion_actual_tokens_rejects_prefill_counts():
    """Explicit num_prefills>0 must not take the query_start_loc decode view."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    attention = SimpleNamespace(layer_name="layers.0.self_attn.attn")
    starts = torch.zeros(9, dtype=torch.int32, device="cuda")
    starts[-1] = 8
    meta = SimpleNamespace(num_prefills=1, num_decodes=0, query_start_loc=starts)
    ctx = SimpleNamespace(attn_metadata={attention.layer_name: meta})
    real_avail = df.is_forward_context_available
    real_get = df.get_forward_context
    df.is_forward_context_available = lambda: True
    df.get_forward_context = lambda: ctx
    try:
        got = df.fusion_actual_tokens(attention, 8, device=torch.device("cuda"))
        # Fallback scalar, not a view into starts.
        assert got is not None
        assert got.data_ptr() != starts[-1:].data_ptr()
        assert int(got.item()) == 8
    finally:
        df.is_forward_context_available = real_avail
        df.get_forward_context = real_get


def test_fusion_actual_tokens_skips_host_copy_while_capturing():
    """CUDA graph capture must not fall back to a host→device scalar alloc."""
    df._warned_capture_without_actual = False
    with mock.patch.object(df, "_capturing", return_value=True):
        assert df.fusion_actual_tokens(None, 4, device=torch.device("cpu")) is None


def test_fusion_actual_tokens_rejects_empty():
    assert df.fusion_actual_tokens(None, 0) is None


def test_can_fuse_gate_covers_tp1_width():
    layer = _Layer(width=12288)
    x = torch.empty(4096, 12288, dtype=torch.bfloat16)
    actual = torch.tensor([4096], dtype=torch.int32)
    assert df.can_fuse_gate(layer, x, actual)
    assert not df.can_fuse_gate(layer, x, None)
    assert not df.can_fuse_gate(layer, x.to(torch.float16), actual)


@pytest.mark.parametrize("rows", [4, 8, 12, 16, 512])
def test_can_fuse_swiglu_fp8_covers_mtp_and_prefill_sizes(rows):
    down = _Layer()
    gate_up = torch.empty(rows, 3072, dtype=torch.bfloat16)
    mask = torch.zeros(rows, dtype=torch.int32)
    assert df.can_fuse_swiglu_fp8(down, gate_up, mask)
    assert not df.can_fuse_swiglu_fp8(down, gate_up, None)


def test_can_fuse_swiglu_fp8_requires_row_mask():
    down = _Layer()
    gate_up = torch.empty(512, 3072, dtype=torch.bfloat16)
    mask = torch.arange(512, dtype=torch.int32)
    assert df.can_fuse_swiglu_fp8(down, gate_up, mask)
    assert not df.can_fuse_swiglu_fp8(down, gate_up, None)
    assert not df.can_fuse_swiglu_fp8(down, gate_up, mask.to(torch.int64))


def test_can_share_norm_quant_prefill_shape():
    attn = SimpleNamespace(
        use_head_wise_attn_gate=True,
        qkv_proj=_Layer(),
        g_proj=_Layer(),
    )
    x = torch.empty(4096, 4096, dtype=torch.bfloat16)
    actual = torch.tensor([4096], dtype=torch.int32)
    assert df.can_share_norm_quant(attn, x, actual)
    assert not df.can_share_norm_quant(attn, x, None)


def test_can_fuse_qk_norm_rope_requires_head_dim_192():
    qkv = torch.empty(64, 13824, dtype=torch.bfloat16)
    rope = torch.empty(8192, 96, dtype=torch.bfloat16)
    actual = torch.tensor([64], dtype=torch.int32)
    assert df.can_fuse_qk_norm_rope(qkv, rope, actual, head_dim=192, rotary_dim=96)
    assert not df.can_fuse_qk_norm_rope(qkv, rope, actual, head_dim=128, rotary_dim=96)
    assert not df.can_fuse_qk_norm_rope(qkv, rope, None, head_dim=192, rotary_dim=96)
    assert not df.can_fuse_qk_norm_rope(qkv, rope, actual, head_dim=192, rotary_dim=95)


def test_dense_valid_row_mask_reuses_prefill_capacity():
    buf = torch.zeros(4096, dtype=torch.int32)
    original = buf.data_ptr()
    for rows in (16, 4096, 4):
        same, mask = df.dense_valid_row_mask(buf, rows, torch.device("cpu"))
        assert same is buf and same.data_ptr() == original
        assert mask.shape == (rows,) and mask.data_ptr() == original
        assert torch.count_nonzero(mask) == 0


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("rows,device", [(64, "cpu"), (16, "meta"), (0, "cpu")])
def test_dense_valid_row_mask_never_relocates(capturing, rows, device):
    """Eager work after capture must retain storage referenced by existing graphs."""
    buf = torch.zeros(32, dtype=torch.int32)
    with mock.patch.object(df, "_capturing", return_value=capturing):
        same, mask = df.dense_valid_row_mask(buf, rows, torch.device(device))
        assert mask is None and same is buf


@torch.inference_mode()
def test_dense_row_mask_graph_replay_after_eager_prefill():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for row-mask capture/replay")
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[16, 64]),
    )
    with mock.patch.object(df, "get_current_vllm_config_or_none", return_value=config):
        buf = torch.zeros(
            df.dense_row_mask_capacity(), dtype=torch.int32, device="cuda"
        )
    original = buf.data_ptr()
    inputs = torch.arange(16, device="cuda", dtype=torch.float32)

    def run_decode():
        nonlocal buf
        buf, mask = df.dense_valid_row_mask(buf, inputs.shape[0], inputs.device)
        assert mask is not None
        return inputs + mask

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            run_decode()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_decode()

    # Both a normal large prefill and an unexpected overflow must preserve the
    # mask address already captured by decode, even though neither is capturing.
    for rows in (4096, 4097):
        buf, mask = df.dense_valid_row_mask(buf, rows, inputs.device)
        assert buf.data_ptr() == original
        assert (mask is not None) == (rows <= 4096)
        inputs.fill_(rows)
        graph.replay()
        torch.testing.assert_close(output, inputs)


@pytest.mark.parametrize(
    "seqs,drafts,captures,prefill,expected",
    [
        (16, 3, [1, 2, 4, 64], 4096, 4096),
        (16, 2, [1, 3, 6, 12, 24, 48], 32, 48),
        (16, 0, [1, 2, 4, 32], 16, 32),
        (8, 8, [], 32, 72),
        (16, 3, [64, 8192], 4096, 8192),
    ],
)
def test_dense_row_mask_capacity_includes_prefill_mtp_and_capture_budget(
    seqs, drafts, captures, prefill, expected
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts),
        scheduler_config=SimpleNamespace(
            max_num_seqs=seqs, max_num_batched_tokens=prefill
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=captures),
    )
    with mock.patch.object(df, "get_current_vllm_config_or_none", return_value=config):
        assert df.dense_row_mask_capacity() == expected


def test_dense_row_mask_capacity_without_speculation_or_config():
    config = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=None),
    )
    with mock.patch.object(df, "get_current_vllm_config_or_none", return_value=config):
        assert df.dense_row_mask_capacity() == 4096
    with mock.patch.object(df, "get_current_vllm_config_or_none", return_value=None):
        assert df.dense_row_mask_capacity() == 32
