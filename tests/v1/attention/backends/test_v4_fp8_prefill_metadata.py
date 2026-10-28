# SPDX-License-Identifier: Apache-2.0
"""Actual FP8 builder contracts with CPU tensors and mocked native operations."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.backends.test_ilu_sparse_swa_builder import (
    _make_builder,
    _make_cad,
)
from vllm_iluvatar.models.deepseek_v4 import sparse_swa
from vllm_iluvatar.models.deepseek_v4.fp8_attention import IluDeepseekV4Fp8Attention


def _builder(*, dspark=False):
    base = _make_builder(is_dspark=dspark, max_tokens=4096, window_size=128)
    builder = object.__new__(sparse_swa.IluDeepseekV4Fp8SWAMetadataBuilder)
    builder.__dict__.update(base.__dict__)
    builder._packed_swa_indices = torch.full((4096, 1, 256), -1, dtype=torch.int32)
    builder.noncausal_index_width = 256 if dspark else 0
    builder._decode_swa_indices_noncausal = (
        torch.full((4096, 1, 256), -1, dtype=torch.int32) if dspark else None
    )
    return builder


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("boundary", [0, 1024])
def test_explicit_replay_prefill_rejected_before_native_or_buffer_mutation(
    mixed, boundary
):
    builder = _builder()
    queries, lengths = ([1, 1024], [8192, 2048]) if mixed else ([1024], [2048])
    cad = _make_cad(query_lens=queries, seq_lens=lengths, causal=True)
    replay = torch.full((len(queries),), boundary, dtype=torch.int32)
    before = builder.decode_swa_indices.clone()
    with (
        patch.object(
            sparse_swa.ixfops, "ds4_compute_swa_decode_indices_and_lens"
        ) as decode,
        patch.object(
            sparse_swa.ixfops, "ds4_compute_swa_prefill_gather_lens"
        ) as gather,
        pytest.raises(ValueError, match="does not support explicit replay_start"),
    ):
        builder.build(0, cad, replay_start=replay)
    decode.assert_not_called()
    gather.assert_not_called()
    torch.testing.assert_close(before, builder.decode_swa_indices, atol=0, rtol=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_default_prefill_gathers_full_causal_window(mixed):
    builder = _builder()
    queries, lengths = ([1, 1024], [8192, 2048]) if mixed else ([1024], [2048])
    cad = _make_cad(query_lens=queries, seq_lens=lengths, causal=True)

    def gather(out, seq, qsl, npref, ndecode, window, *, replay_start):
        assert replay_start.data_ptr() == builder.no_replay_start.data_ptr()
        assert not torch.count_nonzero(replay_start)
        qlens = qsl[1:] - qsl[:-1]
        # Independent causal range: query suffix plus preceding window-1 keys.
        out.copy_(torch.minimum(seq, qlens + window - 1)[ndecode:])

    with (
        patch.object(sparse_swa.ixfops, "ds4_compute_swa_decode_indices_and_lens"),
        patch.object(
            sparse_swa.ixfops, "ds4_compute_swa_prefill_gather_lens", side_effect=gather
        ),
    ):
        md = builder.build(0, cad)
    assert md.num_decodes == int(mixed)
    assert md.num_prefills == 1
    assert md.prefill_gather_lens.tolist() == [1151]
    assert md.prefill_seq_lens.tolist() == [2048]
    # Mixed includes a longer decode request: the scheduler bound remains
    # conservative, while the sparse prefill selector only accepts pure batches.
    assert md.max_prefill_seq_len == (8192 if mixed else 2048)
    assert md.query_start_loc_cpu.tolist() == ([0, 1, 1025] if mixed else [0, 1024])
    assert md.all_swa_indices.stride(0) == 256


@pytest.mark.parametrize("boundary", [None, 0, 1024])
def test_pure_decode_keeps_persistent_buffers_and_ignores_replay(boundary):
    builder = _builder()
    cad = _make_cad(query_lens=[1], seq_lens=[8192], causal=True)
    replay = None if boundary is None else torch.tensor([boundary], dtype=torch.int32)
    with (
        patch.object(
            sparse_swa.ixfops, "ds4_compute_swa_decode_indices_and_lens"
        ) as decode,
        patch.object(
            sparse_swa.ixfops, "ds4_compute_swa_prefill_gather_lens"
        ) as gather,
    ):
        md = builder.build(0, cad, replay_start=replay)
    decode.assert_called_once()
    gather.assert_not_called()
    assert md.all_swa_indices.data_ptr() == builder._packed_swa_indices.data_ptr()
    assert md.all_swa_lens.data_ptr() == builder.decode_swa_lens.data_ptr()
    assert md.prefill_gather_lens is None
    assert md.max_prefill_seq_len == 0


def test_noncausal_dspark_keeps_existing_path_with_explicit_replay():
    builder = _builder(dspark=True)
    cad = _make_cad(query_lens=[5], seq_lens=[8192], causal=False)
    with patch.object(
        sparse_swa.ixfops, "ds4_compute_swa_prefill_gather_lens"
    ) as gather:
        md = builder.build(0, cad, replay_start=torch.tensor([1024], dtype=torch.int32))
    gather.assert_not_called()
    assert not md.causal
    assert md.prefill_gather_lens is None
    assert (
        md.all_swa_indices.data_ptr()
        == builder._decode_swa_indices_noncausal.data_ptr()
    )


@pytest.mark.parametrize("plain", [False, True])
def test_swa_and_compressed_cache_storage_contract(plain):
    cache_dtype = "fp8" if plain else "fp8_ds_mla"
    dtype = torch.float8_e4m3fn if plain else torch.uint8
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=256))
    swa = object.__new__(sparse_swa.IluDeepseekV4Fp8SWACache)
    torch.nn.Module.__init__(swa)
    swa.dtype = dtype
    swa.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    swa._block_size = 64
    swa.head_dim = 512
    swa.window_size = 128
    swa.bounded_replay = False
    swa.packed_bytes_per_token = 584
    spec = swa.get_kv_cache_spec(config)
    assert spec.dtype == dtype
    assert spec.state_content_size_bytes == (512 if plain else 584)
    assert spec.num_states == 64
    attn = object.__new__(IluDeepseekV4Fp8Attention)
    torch.nn.Module.__init__(attn)
    attn.compress_ratio = 4
    attn.head_dim = 512
    attn.kv_cache_dtype = cache_dtype
    attn.kv_cache_torch_dtype = dtype
    spec = attn.get_kv_cache_spec(config)
    assert spec.dtype == dtype
    assert spec.state_content_size_bytes == (512 if plain else 584)
    assert spec.tokens_per_state == 4


@pytest.mark.parametrize("compressed", [False, True])
def test_plain_attention_explicitly_masks_invalid_kv(monkeypatch, compressed):
    from vllm_iluvatar.models.deepseek_v4 import fp8_attention

    attn = object.__new__(IluDeepseekV4Fp8Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "mla"
    attn.compress_ratio = 4 if compressed else 1
    attn.use_prefill_sparse = False
    attn.use_fp8_ds_mla_layout = False
    attn.scale = 512**-0.5
    attn.attn_sink = torch.zeros(8)
    cache = torch.zeros((2, 64, 512), dtype=torch.float8_e4m3fn)
    attn.kv_cache = cache
    attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=cache)
    ids = torch.zeros((2, 256), dtype=torch.int32)
    lens = torch.tensor([1, 0], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_prefill_tokens=0,
        max_prefill_seq_len=0,
        num_prefills=0,
        causal=True,
        all_swa_indices=ids,
        all_swa_lens=lens,
    )
    monkeypatch.setattr(
        fp8_attention,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"swa": metadata, "mla": SimpleNamespace()}
        ),
    )
    attn._build_extra_topk = lambda *args: (ids, lens)
    attn._quant_plain_q = lambda q: (q.to(torch.float8_e4m3fn), torch.ones(q.shape[0]))
    calls = []

    def attention(**kwargs):
        calls.append(kwargs)
        kwargs["output"].fill_(3)

    monkeypatch.setattr(fp8_attention, "sparse_mla_fp8_pt", attention)
    q = torch.zeros((3, 8, 512), dtype=torch.bfloat16)
    output = torch.full_like(q, -7)
    attn.forward_mqa(q, None, torch.arange(3), output)
    assert len(calls) == 1 and calls[0]["mask_invalid_kv"] is True
    assert calls[0]["q"].shape[0] == 2
    assert calls[0]["extra_k_cache"] is (cache if compressed else None)
    assert torch.equal(output[:2], torch.full_like(output[:2], 3))
    assert torch.equal(output[2:], torch.full_like(output[2:], -7))
