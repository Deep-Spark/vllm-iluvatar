# SPDX-License-Identifier: Apache-2.0
"""Tests for HY4 PP predecessor-stage first full-indexer query-tail offload."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
    _lookup_activation_prequant,
    clear_activation_prequant,
    register_activation_prequant,
    sliced_activation_prequant,
)
from vllm_iluvatar.models.hy_v4.fp8_indexer import (
    compute_offload_split_t,
    hy4_fp8_indexer_forward,
    hy4_fp8_indexer_forward_offload_sender,
    slice_indexer_prefill_chunk,
)
from vllm_iluvatar.models.hy_v4.model import _compute_pp_first_indexer_offload_hops
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepSeekV32IndexerDecodeMetadata,
    IluDeepseekV32IndexerMetadata,
    IluDeepseekV32IndexerPrefillChunkMetadata,
    IluDeepseekV32IndexerPrefillMetadata,
    kv_spans_from_batches,
)


def _make_hy4_78_layer_config() -> SimpleNamespace:
    full_layers = {
        0,
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        29,
        33,
        37,
        41,
        45,
        49,
        53,
        57,
        61,
        65,
        69,
        73,
        77,
    }
    indexer_types = ["full" if i in full_layers else "shared" for i in range(78)]
    return SimpleNamespace(
        num_hidden_layers=78,
        index_topk=2048,
        q_lora_rank=1536,
        layer_types=["sparse_attention"] * 78,
        indexer_types=indexer_types,
    )


def test_compute_pp_first_indexer_offload_hops_hy4_pp16(monkeypatch):
    cfg = _make_hy4_78_layer_config()
    cache_cfg = SimpleNamespace(cache_dtype="fp8")

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4")
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_FIRST_INDEXER", "0")
    with patch(
        "vllm_iluvatar.models.hy_v4.model.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        assert _compute_pp_first_indexer_offload_hops(cfg, cache_cfg, 16) == set()

    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_FIRST_INDEXER", "1")
    with patch(
        "vllm_iluvatar.models.hy_v4.model.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        assert _compute_pp_first_indexer_offload_hops(cfg, cache_cfg, 16) == {
            1,
            5,
            9,
            13,
        }
        assert _compute_pp_first_indexer_offload_hops(cfg, cache_cfg, 1) == set()
        assert (
            _compute_pp_first_indexer_offload_hops(cfg, cache_cfg, 16, pcp_size=2)
            == set()
        )
        assert (
            _compute_pp_first_indexer_offload_hops(
                cfg, SimpleNamespace(cache_dtype="auto"), 16
            )
            == set()
        )

    with patch(
        "vllm_iluvatar.models.hy_v4.model.get_tensor_model_parallel_world_size",
        return_value=2,
    ):
        assert _compute_pp_first_indexer_offload_hops(cfg, cache_cfg, 16) == set()

    cfg_non_h32 = _make_hy4_78_layer_config()
    cfg_non_h32.index_n_heads = 16
    with patch(
        "vllm_iluvatar.models.hy_v4.model.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        assert (
            _compute_pp_first_indexer_offload_hops(cfg_non_h32, cache_cfg, 16) == set()
        )


def test_sliced_activation_prequant_preserves_and_restores():
    clear_activation_prequant()
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    x_fp8 = torch.zeros(32, 64, dtype=torch.float8_e4m3fn)
    x_scale = torch.arange(32, dtype=torch.float32)
    register_activation_prequant(x, x_fp8, x_scale)

    with sliced_activation_prequant(x, 0, 16) as head:
        hit = _lookup_activation_prequant(head)
        assert hit is not None
        assert hit[0].shape == (16, 64)
        assert torch.equal(hit[1], x_scale[:16])

    restored = _lookup_activation_prequant(x)
    assert restored is not None
    assert restored[0].shape == (32, 64)

    with sliced_activation_prequant(x, 16, 32) as tail:
        hit_tail = _lookup_activation_prequant(tail)
        assert hit_tail is not None
        assert hit_tail[0].shape == (16, 64)
        assert torch.equal(hit_tail[1], x_scale[16:32])

    assert _lookup_activation_prequant(x[16:32]) is None
    assert _lookup_activation_prequant(x) is not None
    clear_activation_prequant()


def _build_prefill_chunk(
    q_lens: list[int],
    seq_lens: list[int],
    block_table: torch.Tensor,
    token_offset: int,
    device: torch.device,
) -> IluDeepseekV32IndexerPrefillChunkMetadata:
    cu_q_cpu = [0]
    for q in q_lens:
        cu_q_cpu.append(cu_q_cpu[-1] + q)
    cu_q_t = torch.tensor(cu_q_cpu, dtype=torch.int64)
    seq_t = torch.tensor(seq_lens, dtype=torch.int32)
    ks, ke = kv_spans_from_batches(cu_q_t, seq_t, device, compress_ratio=1)
    cu_kv_cpu = [0]
    for s in seq_lens:
        cu_kv_cpu.append(cu_kv_cpu[-1] + s)
    return IluDeepseekV32IndexerPrefillChunkMetadata(
        block_table=block_table,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        valid_lens=ke - ks,
        cu_seq_lens=torch.tensor(cu_kv_cpu, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.tensor(cu_q_cpu, dtype=torch.int32, device=device),
        token_to_seq=torch.empty(0, dtype=torch.int32, device=device),
        total_seq_lens=sum(seq_lens),
        token_start=token_offset,
        token_end=token_offset + cu_q_cpu[-1],
        num_reqs=len(q_lens),
        max_context_len=max(seq_lens),
        max_q_len=max(q_lens),
        max_kv_len=max(seq_lens),
        cu_seqlens_q_cpu=cu_q_cpu,
        seq_lens_comp_cpu=list(seq_lens),
    )


def test_split_and_chunk_slicing_cpu(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_RATIO", "0.45")
    bt = torch.zeros((2, 16), dtype=torch.int32)
    chunk = _build_prefill_chunk(
        [1749, 811], [80789, 79851], bt, token_offset=0, device=torch.device("cpu")
    )
    md = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([80789, 79851], dtype=torch.int32),
        max_seq_len=80789,
        slot_mapping=torch.arange(2560, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        num_prefill_tokens=2560,
        prefill=IluDeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
    )
    split_range = compute_offload_split_t(md, 2560, 2048)
    assert split_range == (1408, 2560)

    head_sub = slice_indexer_prefill_chunk(chunk, 0, 1408)
    tail_sub = slice_indexer_prefill_chunk(chunk, 1408, 2560)
    assert head_sub is not None and tail_sub is not None
    assert head_sub.cu_seqlens_q_cpu == [0, 1408]
    assert head_sub.num_reqs == 1
    assert tail_sub.cu_seqlens_q_cpu == [0, 1749 - 1408, 2560 - 1408]
    assert tail_sub.num_reqs == 2
    assert torch.equal(
        torch.cat([head_sub.valid_lens, tail_sub.valid_lens]), chunk.valid_lens
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
@pytest.mark.parametrize("use_mrope", [False, True])
@pytest.mark.parametrize(
    "q_lens,prefix_lens,num_decodes",
    [
        ([2560], [79040], 0),
        ([1749, 811], [79040, 79040], 0),
        ([600, 705, 1255], [32768, 65536, 49152], 0),
        ([1524, 1020], [65536, 79040], 16),
    ],
)
def test_gpu_first_indexer_offload_bitwise_exact(
    monkeypatch, use_mrope, q_lens, prefix_lens, num_decodes
):
    import ixformer.inference.functions as ixfops
    from vllm.config import (
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm

    from vllm_iluvatar.models.hy_v4.attention import build_hy4_ropes

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_QA_NORM_QUANT", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_INDEXER_K", "1")
    monkeypatch.setenv("IXFORMER_DSA_FAST_RADIX_TOPK", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_FAST_RADIX_TOPK", "1")  # legacy alias
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_RATIO", "0.45")

    torch.manual_seed(20261124)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    block_size = 64
    hidden_size = 6144
    q_lora_rank = 1536
    kv_lora_and_rope = 576
    n_head = 32
    head_dim = 128
    rope_dim = 64
    topk_tokens = 2048

    prefill_tokens = sum(q_lens)
    total_tokens = num_decodes + prefill_tokens
    decode_seq_lens = [4096 + i * 64 for i in range(num_decodes)]
    prefill_seq_lens = [p + q for p, q in zip(prefix_lens, q_lens)]
    all_seq_lens = decode_seq_lens + prefill_seq_lens

    max_blocks_per_req = (max(all_seq_lens) + block_size - 1) // block_size
    total_blocks = sum((s + block_size - 1) // block_size for s in all_seq_lens) + 8
    block_table = torch.zeros(
        (len(all_seq_lens), max_blocks_per_req), dtype=torch.int32, device=device
    )
    next_blk = 1
    slot_list: list[int] = []
    pos_list: list[int] = []
    for r_idx, s_len in enumerate(all_seq_lens):
        n_blk = (s_len + block_size - 1) // block_size
        blks = torch.arange(
            next_blk, next_blk + n_blk, dtype=torch.int32, device=device
        )
        block_table[r_idx, :n_blk] = blks
        next_blk += n_blk
        q_len = 1 if r_idx < num_decodes else q_lens[r_idx - num_decodes]
        start_pos = s_len - q_len
        for p in range(start_pos, s_len):
            blk_id = int(blks[p // block_size].item())
            slot_list.append(blk_id * block_size + (p % block_size))
            pos_list.append(p)

    slot_mapping = torch.tensor(slot_list, dtype=torch.int64, device=device)
    pos_1d = torch.tensor(pos_list, dtype=torch.int64, device=device)
    if use_mrope:
        positions = torch.stack([pos_1d, pos_1d + 1, pos_1d + 2], dim=0)
    else:
        positions = pos_1d

    # Seed existing prefix KV cache identically across reference, sender, receiver.
    base_kv_cache = torch.randint(
        0,
        256,
        (total_blocks, block_size, head_dim + 4),
        dtype=torch.uint8,
        device=device,
    )
    kv_ref = base_kv_cache.clone()
    kv_sender = base_kv_cache.clone()
    kv_recv = base_kv_cache.clone()

    vllm_cfg = VllmConfig()

    rope_params = {
        "rope_type": "default",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 1.0,
        "mrope_section": [8, 12, 12],
        "mrope_interleaved": True,
    }
    with set_current_vllm_config(vllm_cfg):
        rotary_emb, mrope_emb = build_hy4_ropes(rope_dim, 131072, rope_params)
        rotary_emb = rotary_emb.to(device)
        if mrope_emb is not None:
            mrope_emb = mrope_emb.to(device)

        in_norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
        qa_norm = RMSNorm(q_lora_rank, eps=1e-6).to(device=device, dtype=dtype)
        k_norm = LayerNorm(head_dim, eps=1e-6).to(device=device, dtype=torch.float32)

    w_qkv = (
        torch.randn(
            q_lora_rank + kv_lora_and_rope, hidden_size, device=device, dtype=dtype
        )
        * 0.02
    )
    w_qkv_fp8, w_qkv_s = ixfops.fp8_per_token_dynamic_scaled(w_qkv)
    w_wq_b = (
        torch.randn(n_head * head_dim, q_lora_rank, device=device, dtype=dtype) * 0.02
    )
    w_wq_b_fp8, w_wq_b_s = ixfops.fp8_per_token_dynamic_scaled(w_wq_b)
    w_wk = (
        torch.randn(head_dim + n_head, hidden_size, device=device, dtype=dtype) * 0.02
    )
    w_wk_fp8, w_wk_s = ixfops.fp8_per_token_dynamic_scaled(w_wk)

    def _make_fp8_linear(w_fp8: torch.Tensor, w_s: torch.Tensor):
        def _run(x: torch.Tensor):
            hit = _lookup_activation_prequant(x)
            if hit is not None:
                a_fp8, a_s = hit
            else:
                a_fp8, a_s = ixfops.fp8_per_token_dynamic_scaled(x.contiguous())
            out = ixfops.fp8_per_channel_mm(a_fp8, w_fp8, a_s, w_s)
            return out, None

        return _run

    def _make_indexer_stub(kv_buf: torch.Tensor, topk_buf: torch.Tensor, prefix: str):
        return SimpleNamespace(
            n_head=n_head,
            head_dim=head_dim,
            rope_dim=rope_dim,
            softmax_scale=head_dim**-0.5,
            n_head_scale=n_head**-0.5,
            wq_b=_make_fp8_linear(w_wq_b_fp8, w_wq_b_s),
            wk_weights_proj=_make_fp8_linear(w_wk_fp8, w_wk_s),
            k_norm=k_norm,
            topk_tokens=topk_tokens,
            topk_indices_buffer=topk_buf,
            k_cache=SimpleNamespace(prefix=prefix, kv_cache=kv_buf),
            mrope_emb=mrope_emb,
            _offload_recv_wire=None,
        )

    prefill_chunk = _build_prefill_chunk(
        q_lens,
        prefill_seq_lens,
        block_table[num_decodes:],
        token_offset=num_decodes,
        device=device,
    )
    decode_md = None
    if num_decodes > 0:
        cu_d_q = torch.arange(num_decodes + 1, dtype=torch.int32, device=device)
        cu_d_kv = torch.tensor(
            [0] + list(torch.tensor(decode_seq_lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        d_ks = torch.zeros(num_decodes, dtype=torch.int32, device=device)
        d_ke = torch.tensor(decode_seq_lens, dtype=torch.int32, device=device)
        decode_md = IluDeepSeekV32IndexerDecodeMetadata(
            block_table=block_table[:num_decodes],
            seq_lens=d_ke,
            decode_lens=torch.ones(num_decodes, dtype=torch.int32, device=device),
            requires_padding=False,
            cu_seqlen_ks=d_ks,
            cu_seqlen_ke=d_ke,
            valid_lens=d_ke,
            cu_seqlens_kv=cu_d_kv,
            cu_seqlens_q=cu_d_q,
            max_context_len=max(decode_seq_lens),
            max_q_len=1,
            max_kv_len=max(decode_seq_lens),
        )

    md = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor(all_seq_lens, dtype=torch.int32, device=device),
        max_seq_len=max(all_seq_lens),
        slot_mapping=slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_prefills=len(q_lens),
        num_prefill_tokens=prefill_tokens,
        decode=decode_md,
        prefill=IluDeepseekV32IndexerPrefillMetadata(chunks=[prefill_chunk]),
    )

    raw_hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
    normed = in_norm(raw_hidden)
    normed_fp8, normed_s = ixfops.fp8_per_token_dynamic_scaled(normed)

    fused_qkv_fn = _make_fp8_linear(w_qkv_fp8, w_qkv_s)

    # 1. Reference unsplit run on Rank 2k.
    topk_ref = torch.full(
        (total_tokens, topk_tokens), -7, dtype=torch.int32, device=device
    )
    idx_ref = _make_indexer_stub(kv_ref, topk_ref, "layer9.k_cache")
    with set_forward_context({"layer9.k_cache": md}, vllm_cfg):
        register_activation_prequant(normed, normed_fp8, normed_s)
        qkv_all = fused_qkv_fn(normed)[0]
        q_c_all, _ = qkv_all.split([q_lora_rank, kv_lora_and_rope], dim=-1)
        qc_fp8_all, qc_s_all = ixfops.fp8_per_token_dynamic_scaled(
            q_c_all, rms_weight=qa_norm.weight, rms_eps=float(qa_norm.variance_epsilon)
        )
        q_c_view = qkv_all.view(-1)[: total_tokens * q_lora_rank].view(
            total_tokens, q_lora_rank
        )
        register_activation_prequant(q_c_view, qc_fp8_all, qc_s_all)
        hy4_fp8_indexer_forward(idx_ref, normed, q_c_view, positions, rotary_emb)
        clear_activation_prequant()

    # 2. Split run: Sender helper (Rank 2k-1) -> wire (T, 1024) -> Receiver (Rank 2k).
    wire_buf = torch.full(
        (total_tokens, topk_tokens // 2), -9, dtype=torch.int32, device=device
    )
    dummy_sender_topk = torch.empty((1, topk_tokens), dtype=torch.int32, device=device)
    idx_sender = _make_indexer_stub(kv_sender, dummy_sender_topk, "layer9.k_cache")

    topk_recv = torch.full(
        (total_tokens, topk_tokens), -7, dtype=torch.int32, device=device
    )
    idx_recv = _make_indexer_stub(kv_recv, topk_recv, "layer9.k_cache")

    with set_forward_context({"layer9.k_cache": md}, vllm_cfg):
        register_activation_prequant(normed, normed_fp8, normed_s)
        wire_sent = hy4_fp8_indexer_forward_offload_sender(
            indexer=idx_sender,
            normed_hidden_states=normed,
            fused_qkv_a_proj=fused_qkv_fn,
            q_a_layernorm=qa_norm,
            q_lora_rank=q_lora_rank,
            kv_lora_and_rope_dim=kv_lora_and_rope,
            supports_wq_b_prequant=True,
            positions=positions,
            rotary_emb=rotary_emb,
            wire_out=wire_buf,
        )
        clear_activation_prequant()

        idx_recv._offload_recv_wire = wire_sent.clone()
        register_activation_prequant(normed, normed_fp8, normed_s)
        qkv_recv = fused_qkv_fn(normed)[0]
        q_c_recv, _ = qkv_recv.split([q_lora_rank, kv_lora_and_rope], dim=-1)
        qc_fp8_r, qc_s_r = ixfops.fp8_per_token_dynamic_scaled(
            q_c_recv, rms_weight=qa_norm.weight, rms_eps=float(qa_norm.variance_epsilon)
        )
        q_c_recv_view = qkv_recv.view(-1)[: total_tokens * q_lora_rank].view(
            total_tokens, q_lora_rank
        )
        register_activation_prequant(q_c_recv_view, qc_fp8_r, qc_s_r)
        hy4_fp8_indexer_forward(idx_recv, normed, q_c_recv_view, positions, rotary_emb)
        idx_recv._offload_recv_wire = None
        clear_activation_prequant()

    assert torch.equal(kv_sender, kv_ref), (
        "Sender mirrored K-cache diverged from reference"
    )
    assert torch.equal(kv_recv, kv_ref), "Receiver K-cache diverged from reference"
    assert torch.equal(topk_recv, topk_ref), (
        "Offloaded + local topk_indices diverged from unsplit reference"
    )


@pytest.mark.parametrize("topk_cols", [0, 1024, 2048])
@pytest.mark.parametrize("num_tokens", [0, 1, 40, 128, 2560])
def test_pp_boundary_single_tensor_pack_unpack_roundtrip(
    topk_cols: int, num_tokens: int
) -> None:
    from vllm_iluvatar.models.hy_v4.model import (
        compute_pp_packed_row_bytes,
        pack_hy4_pp_boundary,
        unpack_hy4_pp_boundary,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_cols = 24576
    max_tokens = 2560
    row_bytes = compute_pp_packed_row_bytes(hidden_cols, topk_cols)

    send_buf = torch.full(
        (max_tokens, row_bytes), 0xAB, dtype=torch.uint8, device=device
    )
    recv_runner_buf = torch.full(
        (max_tokens, row_bytes), 0xCD, dtype=torch.uint8, device=device
    )

    q_src = torch.randint(
        0, 256, (num_tokens, hidden_cols), dtype=torch.uint8, device=device
    )
    scale_src = torch.randn((num_tokens,), dtype=torch.float32, device=device)
    tk_src = (
        torch.randint(
            -1, 65536, (num_tokens, topk_cols), dtype=torch.int32, device=device
        )
        if topk_cols > 0
        else None
    )

    # Exercise zero-copy wire_view path when topk_cols == 1024 (offload_first_topk).
    if topk_cols == 1024:
        packed_2d, wire_view = pack_hy4_pp_boundary(
            send_buf, q_src, scale_src, topk=None, topk_cols=topk_cols
        )
        assert wire_view is not None and wire_view.shape == (num_tokens, topk_cols)
        assert tk_src is not None
        wire_view.copy_(tk_src)
    else:
        packed_2d, wire_view = pack_hy4_pp_boundary(
            send_buf, q_src, scale_src, topk=tk_src, topk_cols=topk_cols
        )

    assert packed_2d.shape == (num_tokens, row_bytes)
    assert packed_2d.is_contiguous()

    # Simulate gpu_model_runner.sync_and_gather_intermediate_tensors:
    # self.intermediate_tensors[k][:num_tokens].copy_(v[:num_tokens])
    recv_runner_buf[:num_tokens].copy_(packed_2d[:num_tokens])
    recv_slice = recv_runner_buf[:num_tokens]

    q_dst, scale_dst, tk_dst = unpack_hy4_pp_boundary(
        recv_slice, hidden_cols, topk_cols
    )
    assert torch.equal(q_dst, q_src)
    assert torch.equal(scale_dst, scale_src)
    if topk_cols > 0:
        assert tk_dst is not None and tk_src is not None
        assert torch.equal(tk_dst, tk_src)
    else:
        assert tk_dst is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
def test_pp_fp8_boundary_operators():
    from vllm_iluvatar.models.hy_v4 import model

    tokens, hc, hidden = 8, 4, 6144
    x = torch.ones((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    x *= torch.arange(1, tokens + 1, device=x.device)[:, None]
    residual = torch.zeros((tokens, hc, hidden), device=x.device, dtype=x.dtype)
    post = torch.ones((tokens, hc), device=x.device)

    q, scale = model._ihc_post_quant_fp8_boundary(x, residual, post)
    expected = x[:, None, :].expand_as(residual)
    torch.testing.assert_close(scale, x[:, 0].float() / 448)
    assert torch.all(q.view(torch.float8_e4m3fn).float() == 448)
    decoded = model._dequant_pp_act(q, scale).view_as(residual)
    torch.testing.assert_close(decoded, expected)
    torch.testing.assert_close(
        model._ihc_post_fp8_boundary(x, q, scale, post), 2 * expected
    )
    q_next, scale_next = model._ihc_post_fp8io(x, q, scale, post)
    assert torch.equal(q_next, q)
    torch.testing.assert_close(scale_next, 2 * scale)

    # Zero projection gives pre gates of 1/2 and post gates of 1.
    hc_pre = SimpleNamespace(
        hc_mult=hc,
        hc_fn=SimpleNamespace(weight=x.new_zeros((2 * hc, hc * hidden))),
        hc_scale=post.new_ones(2),
        hc_base=post.new_zeros(2 * hc),
        layernorm_epsilon=1e-5,
        hc_eps=0.0,
        magnitude=2.0,
    )
    y, gates = model._ihc_pre_fp8_boundary(hc_pre, q, scale)
    torch.testing.assert_close(y, 2 * x)
    torch.testing.assert_close(gates, post)

    q_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_out = torch.empty_like(scale)
    normed, _ = model._ihc_pre_fp8_boundary(
        hc_pre,
        q,
        scale,
        rms_weight=torch.ones_like(x[0]),
        q_out=q_out,
        scale_out=scale_out,
    )
    torch.testing.assert_close(normed, torch.ones_like(x))
    torch.testing.assert_close(scale_out, torch.full_like(scale, 1 / 448))
    assert torch.all(q_out.float() == 448)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
def test_breakable_cudagraph_disables_fuse_qa_norm_quant(monkeypatch):
    """Review #1: breakable cudagraph must not pass a _PREQUANT placeholder q_c."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    from vllm_iluvatar.models.hy_v4 import attention as attn_mod
    from vllm_iluvatar.models.hy_v4.attention import HYV4MLAAttention

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_QA_NORM_QUANT", "1")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    tokens = 16
    hidden_size = 128
    q_lora_rank = 64
    kv_lora_rank = 32
    qk_rope_head_dim = 16
    num_heads = 4
    qk_nope_head_dim = 16
    v_head_dim = 16

    with set_current_vllm_config(VllmConfig()):
        qa_norm = RMSNorm(q_lora_rank, eps=1e-6).to(device=device, dtype=dtype)
    qkv_fixed = torch.randn(
        tokens,
        q_lora_rank + kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=dtype,
    )
    captured_qc: list[tuple[torch.Tensor, object]] = []

    stub = SimpleNamespace(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        num_local_heads=num_heads,
        v_head_dim=v_head_dim,
        gated_mla=False,
        linear_gate=None,
        config=SimpleNamespace(gating_type="elementwise"),
        mla_attn=SimpleNamespace(fuse_gate=False),
        indexer=SimpleNamespace(wq_b=object()),
        is_sparse=True,
        skip_topk=False,
        fused_qkv_a_proj=lambda h: (qkv_fixed.clone(), None),
        q_a_layernorm=qa_norm,
        q_b_proj=lambda qc: (
            torch.zeros(
                qc.shape[0],
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                device=device,
                dtype=dtype,
            ),
            None,
        ),
        o_proj=lambda x: (x, None),
        _supports_activation_prequant=lambda _proj: True,
        _prepare_mla_qkv=lambda q, kv_c, k_pe, pos: kv_c,
        _indexer_and_attn=lambda _h, q_c, _pos, _q, _kvc, _kpe, out: (
            captured_qc.append((q_c.clone(), _lookup_activation_prequant(q_c))),
            out.zero_(),
        ),
    )

    hidden = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    with patch.object(attn_mod, "is_breakable_cudagraph_enabled", return_value=True):
        HYV4MLAAttention.forward(stub, positions, hidden, None)

    assert len(captured_qc) == 1
    qc_seen, prequant_hit = captured_qc[0]
    assert prequant_hit is None
    expected_qc = qa_norm(qkv_fixed[:, :q_lora_rank].clone())
    assert torch.equal(qc_seen, expected_qc)


def _build_hc_and_norms_no_dist(cfg: SimpleNamespace, device: torch.device):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    from vllm_iluvatar.models.hy_v4.hc import HYV4HCLayer

    tp_stub = SimpleNamespace(rank_in_group=0, world_size=1)
    with (
        set_current_vllm_config(VllmConfig()),
        patch("vllm.distributed.parallel_state._TP", tp_stub),
    ):
        hc_attn = HYV4HCLayer(cfg, 0, prefix="layers.0.hc_attn_layer").to(device)
        hc_mlp = HYV4HCLayer(cfg, 0, prefix="layers.0.hc_mlp_layer").to(device)
        in_ln = RMSNorm(cfg.hidden_size, eps=1e-6).to(
            device=device, dtype=torch.bfloat16
        )
        post_ln = RMSNorm(cfg.hidden_size, eps=1e-6).to(
            device=device, dtype=torch.bfloat16
        )
    return hc_attn, hc_mlp, in_ln, post_ln


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
@pytest.mark.parametrize("hidden_size", [3072, 4096])
@pytest.mark.parametrize("tokens", [16, 160])
def test_ihc_fp8_narrow_widths_fallback_cleanly(
    monkeypatch, hidden_size: int, tokens: int
):
    """Review #2 & #3: Hy4-Flash (3072) and Hy4-Preview (4096) must not hit 6144-only ihc_fp8."""
    from vllm_iluvatar.models.hy_v4 import model as hy4_model

    monkeypatch.setenv("VLLM_ILU_HYV4_IHC_FP8_RES", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_ACT_FP8", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC_NORM", "1")

    device = torch.device("cuda:0")
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        enable_ihc=True,
    )
    assert not hy4_model._can_use_ihc_fp8_ops(cfg)

    hc_attn, hc_mlp, in_ln, post_ln = _build_hc_and_norms_no_dist(cfg, device)
    layer = SimpleNamespace(
        config=cfg,
        hc_attn_layer=hc_attn,
        hc_mlp_layer=hc_mlp,
        input_layernorm=in_ln,
        post_attention_layernorm=post_ln,
        self_attn=lambda positions, hidden_states: hidden_states * 0.5,
        mlp=lambda hidden_states: hidden_states * 0.5,
        _emit_pp_fp8=True,
        _is_stage_last_layer=True,
    )
    layer._run_ihc_pre_and_norm = (
        lambda hc_l, ln, h, fuse_fp8=True: hy4_model._run_ihc_pre_and_norm_impl(
            hc_l, ln, h, fuse_fp8=fuse_fp8
        )
    )
    layer._run_ihc_pre_fp8_boundary_and_norm = (
        lambda hc_pre, ln, qb, sc, fuse_fp8=True: (
            hy4_model._run_ihc_pre_fp8_boundary_and_norm_impl(
                hc_pre, ln, qb, sc, fuse_fp8=fuse_fp8
            )
        )
    )

    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    x_3d = torch.randn(tokens, 4, hidden_size, device=device, dtype=torch.bfloat16)
    out_from_bf16, res_none = hy4_model.HYV4DecoderLayer._forward_ihc(
        layer, positions, x_3d
    )
    assert res_none is None
    assert isinstance(out_from_bf16, torch.Tensor)
    assert out_from_bf16.shape == (tokens, 4, hidden_size)
    assert out_from_bf16.dtype == torch.bfloat16

    # Also test incoming PP FP8 (q, scale) tuple on narrow width.
    q_in, s_in = hy4_model._quant_pp_act(x_3d.flatten(1))
    out_from_fp8, _ = hy4_model.HYV4DecoderLayer._forward_ihc(
        layer, positions, (q_in, s_in)
    )
    assert isinstance(out_from_fp8, torch.Tensor)
    assert out_from_fp8.shape == (tokens, 4, hidden_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
def test_ihc_fp8_res_without_pp_act_fp8_keeps_bf16_pp_wire(monkeypatch):
    """Review #4: IHC_FP8_RES=1 + PP_ACT_FP8=0 must emit BF16 PP boundary matching receiver."""
    from vllm_iluvatar.models.hy_v4 import model as hy4_model

    monkeypatch.setenv("VLLM_ILU_HYV4_IHC_FP8_RES", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_ACT_FP8", "0")
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_IHC", "1")

    device = torch.device("cuda:0")
    tokens = 160
    hidden_size = 6144
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        enable_ihc=True,
        index_topk=2048,
    )
    assert hy4_model._can_use_ihc_fp8_ops(cfg)

    def _build_layer_stub(is_stage_last: bool):
        hc_attn, hc_mlp, in_ln, post_ln = _build_hc_and_norms_no_dist(cfg, device)
        stub = SimpleNamespace(
            config=cfg,
            hc_attn_layer=hc_attn,
            hc_mlp_layer=hc_mlp,
            input_layernorm=in_ln,
            post_attention_layernorm=post_ln,
            self_attn=lambda positions, hidden_states: hidden_states * 0.5,
            mlp=lambda hidden_states: hidden_states * 0.5,
            _emit_pp_fp8=False,
            _is_stage_last_layer=is_stage_last,
        )
        stub._run_ihc_pre_and_norm = (
            lambda hc_l, ln, h, fuse_fp8=True: hy4_model._run_ihc_pre_and_norm_impl(
                hc_l, ln, h, fuse_fp8=fuse_fp8
            )
        )
        stub._run_ihc_pre_fp8_boundary_and_norm = (
            lambda hc_pre, ln, qb, sc, fuse_fp8=True: (
                hy4_model._run_ihc_pre_fp8_boundary_and_norm_impl(
                    hc_pre, ln, qb, sc, fuse_fp8=fuse_fp8
                )
            )
        )
        return stub

    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    x_3d = torch.randn(tokens, 4, hidden_size, device=device, dtype=torch.bfloat16)

    # Mid-stage layer keeps intra-stage FP8 residual tuple.
    mid_out, _ = hy4_model.HYV4DecoderLayer._forward_ihc(
        _build_layer_stub(is_stage_last=False), positions, x_3d
    )
    assert isinstance(mid_out, tuple)

    # Stage-last layer with PP_ACT_FP8=0 converts back to BF16 matching receiver buffer.
    end_out, _ = hy4_model.HYV4DecoderLayer._forward_ihc(
        _build_layer_stub(is_stage_last=True), positions, mid_out
    )
    assert isinstance(end_out, torch.Tensor)
    assert end_out.dtype == torch.bfloat16
    assert end_out.shape == (tokens, 4, hidden_size)

    # Receiver's make_empty_intermediate_tensors allocates BF16 without scale key.
    recv_model_stub = SimpleNamespace(
        _use_pp_packed_wire=False,
        _pp_intermediate_keys=("hidden_states",),
        _pp_intermediate_hidden=4 * hidden_size,
        recv_topk=False,
        recv_offload_first_topk=False,
        config=cfg,
    )
    recv_empty = hy4_model.HYV4Model.make_empty_intermediate_tensors(
        recv_model_stub, tokens, torch.bfloat16, device
    )
    assert set(recv_empty.tensors.keys()) == {"hidden_states"}
    assert recv_empty["hidden_states"].dtype == torch.bfloat16
    assert recv_empty["hidden_states"].shape == (tokens, 4 * hidden_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
def test_gpu_chunked_prefill_small_first_chunk_syncs_helper_k_cache(monkeypatch):
    """Review #5 & #6: chunk 1 (128 tok, unsplit) must still write helper K-cache for chunk 2 (512 tok, split); also verify sender md=None fallback."""
    import ixformer.inference.functions as ixfops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm

    from vllm_iluvatar.models.hy_v4.attention import build_hy4_ropes

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_QA_NORM_QUANT", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_INDEXER_K", "1")
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_MIN_TOKENS", "256")
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_RATIO", "0.42")

    torch.manual_seed(20261127)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    block_size = 64
    hidden_size = 6144
    q_lora_rank = 1536
    kv_lora_and_rope = 576
    n_head = 32
    head_dim = 128
    rope_dim = 64
    topk_tokens = 2048

    chunk1_len = 128
    chunk2_len = 512
    total_seq = chunk1_len + chunk2_len
    n_blk = (total_seq + block_size - 1) // block_size
    block_table = torch.arange(1, 1 + n_blk, dtype=torch.int32, device=device).view(
        1, n_blk
    )

    base_kv = torch.randint(
        0, 256, (n_blk + 4, block_size, head_dim + 4), dtype=torch.uint8, device=device
    )
    kv_ref = base_kv.clone()
    kv_sender = base_kv.clone()
    kv_recv = base_kv.clone()

    vllm_cfg = VllmConfig()
    rope_params = {
        "rope_type": "default",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 1.0,
    }
    with set_current_vllm_config(vllm_cfg):
        rotary_emb, _ = build_hy4_ropes(rope_dim, 131072, rope_params)
        rotary_emb = rotary_emb.to(device)
        in_norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
        qa_norm = RMSNorm(q_lora_rank, eps=1e-6).to(device=device, dtype=dtype)
        k_norm = LayerNorm(head_dim, eps=1e-6).to(device=device, dtype=torch.float32)

    w_qkv_fp8, w_qkv_s = ixfops.fp8_per_token_dynamic_scaled(
        torch.randn(
            q_lora_rank + kv_lora_and_rope, hidden_size, device=device, dtype=dtype
        )
        * 0.02
    )
    w_wq_b_fp8, w_wq_b_s = ixfops.fp8_per_token_dynamic_scaled(
        torch.randn(n_head * head_dim, q_lora_rank, device=device, dtype=dtype) * 0.02
    )
    w_wk_fp8, w_wk_s = ixfops.fp8_per_token_dynamic_scaled(
        torch.randn(head_dim + n_head, hidden_size, device=device, dtype=dtype) * 0.02
    )

    def _fp8_lin(w_fp8: torch.Tensor, w_s: torch.Tensor):
        def _fn(x: torch.Tensor):
            hit = _lookup_activation_prequant(x)
            a_fp8, a_s = (
                hit
                if hit is not None
                else ixfops.fp8_per_token_dynamic_scaled(x.contiguous())
            )
            return ixfops.fp8_per_channel_mm(a_fp8, w_fp8, a_s, w_s), None

        return _fn

    def _stub(kv_buf: torch.Tensor, topk_buf: torch.Tensor):
        return SimpleNamespace(
            n_head=n_head,
            head_dim=head_dim,
            rope_dim=rope_dim,
            softmax_scale=head_dim**-0.5,
            n_head_scale=n_head**-0.5,
            wq_b=_fp8_lin(w_wq_b_fp8, w_wq_b_s),
            wk_weights_proj=_fp8_lin(w_wk_fp8, w_wk_s),
            k_norm=k_norm,
            topk_tokens=topk_tokens,
            topk_indices_buffer=topk_buf,
            k_cache=SimpleNamespace(prefix="layer9.k_cache", kv_cache=kv_buf),
            mrope_emb=None,
            _offload_recv_wire=None,
        )

    fused_qkv_fn = _fp8_lin(w_qkv_fp8, w_qkv_s)

    def _slots_for(start: int, end: int) -> torch.Tensor:
        out = [
            int(block_table[0, p // block_size].item()) * block_size + (p % block_size)
            for p in range(start, end)
        ]
        return torch.tensor(out, dtype=torch.int64, device=device)

    # --- Step 1: 128 tokens (below min_tokens=256 -> split_range is None) ---
    c1 = _build_prefill_chunk(
        [chunk1_len], [chunk1_len], block_table, token_offset=0, device=device
    )
    md1 = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([chunk1_len], dtype=torch.int32, device=device),
        max_seq_len=chunk1_len,
        slot_mapping=_slots_for(0, chunk1_len),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=chunk1_len,
        prefill=IluDeepseekV32IndexerPrefillMetadata(chunks=[c1]),
    )
    assert compute_offload_split_t(md1, chunk1_len, topk_tokens) is None

    pos1 = torch.arange(0, chunk1_len, device=device, dtype=torch.int64)
    normed1 = in_norm(torch.randn(chunk1_len, hidden_size, device=device, dtype=dtype))
    topk_ref1 = torch.full(
        (chunk1_len, topk_tokens), -7, dtype=torch.int32, device=device
    )
    topk_recv1 = torch.full(
        (chunk1_len, topk_tokens), -7, dtype=torch.int32, device=device
    )
    wire1 = torch.full(
        (chunk1_len, topk_tokens // 2), 99, dtype=torch.int32, device=device
    )

    idx_ref = _stub(kv_ref, topk_ref1)
    idx_sender = _stub(
        kv_sender, torch.empty((1, topk_tokens), dtype=torch.int32, device=device)
    )
    idx_recv = _stub(kv_recv, topk_recv1)

    with set_forward_context({"layer9.k_cache": md1}, vllm_cfg):
        qc1 = qa_norm(fused_qkv_fn(normed1)[0][:, :q_lora_rank])
        hy4_fp8_indexer_forward(idx_ref, normed1, qc1, pos1, rotary_emb)

        sent1 = hy4_fp8_indexer_forward_offload_sender(
            indexer=idx_sender,
            normed_hidden_states=normed1,
            fused_qkv_a_proj=fused_qkv_fn,
            q_a_layernorm=qa_norm,
            q_lora_rank=q_lora_rank,
            kv_lora_and_rope_dim=kv_lora_and_rope,
            supports_wq_b_prequant=True,
            positions=pos1,
            rotary_emb=rotary_emb,
            wire_out=wire1,
        )
        assert int(sent1.view(-1)[1].item()) == 0
        idx_recv._offload_recv_wire = sent1.clone()
        hy4_fp8_indexer_forward(idx_recv, normed1, qc1, pos1, rotary_emb)
        idx_recv._offload_recv_wire = None

    # Helper K-cache MUST already contain chunk 1's K entries!
    assert torch.equal(kv_sender, kv_ref)
    assert torch.equal(kv_recv, kv_ref)
    assert torch.equal(topk_recv1, topk_ref1)

    # --- Step 2: 512 tokens (positions [128, 640), split_range active!) ---
    c2 = _build_prefill_chunk(
        [chunk2_len], [total_seq], block_table, token_offset=0, device=device
    )
    md2 = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([total_seq], dtype=torch.int32, device=device),
        max_seq_len=total_seq,
        slot_mapping=_slots_for(chunk1_len, total_seq),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=chunk2_len,
        prefill=IluDeepseekV32IndexerPrefillMetadata(chunks=[c2]),
    )
    assert compute_offload_split_t(md2, chunk2_len, topk_tokens) is not None

    pos2 = torch.arange(chunk1_len, total_seq, device=device, dtype=torch.int64)
    normed2 = in_norm(torch.randn(chunk2_len, hidden_size, device=device, dtype=dtype))
    topk_ref2 = torch.full(
        (chunk2_len, topk_tokens), -7, dtype=torch.int32, device=device
    )
    topk_recv2 = torch.full(
        (chunk2_len, topk_tokens), -7, dtype=torch.int32, device=device
    )
    idx_ref.topk_indices_buffer = topk_ref2
    idx_recv.topk_indices_buffer = topk_recv2
    wire2 = torch.full(
        (chunk2_len, topk_tokens // 2), -99, dtype=torch.int32, device=device
    )

    with set_forward_context({"layer9.k_cache": md2}, vllm_cfg):
        qc2 = qa_norm(fused_qkv_fn(normed2)[0][:, :q_lora_rank])
        hy4_fp8_indexer_forward(idx_ref, normed2, qc2, pos2, rotary_emb)

        sent2 = hy4_fp8_indexer_forward_offload_sender(
            indexer=idx_sender,
            normed_hidden_states=normed2,
            fused_qkv_a_proj=fused_qkv_fn,
            q_a_layernorm=qa_norm,
            q_lora_rank=q_lora_rank,
            kv_lora_and_rope_dim=kv_lora_and_rope,
            supports_wq_b_prequant=True,
            positions=pos2,
            rotary_emb=rotary_emb,
            wire_out=wire2,
        )
        assert int(sent2.view(-1)[1].item()) > 0
        idx_recv._offload_recv_wire = sent2.clone()
        hy4_fp8_indexer_forward(idx_recv, normed2, qc2, pos2, rotary_emb)
        idx_recv._offload_recv_wire = None

    assert torch.equal(kv_sender, kv_ref)
    assert torch.equal(kv_recv, kv_ref)
    assert torch.equal(topk_recv2, topk_ref2)

    # --- Review #6: sender md=None on reused wire2 must zero header & trigger receiver full fallback ---
    kv_recv_fallback = kv_ref.clone()
    # Rewind chunk2 slots on kv_recv_fallback to step-1 state so fallback re-runs step 2 cleanly.
    topk_fallback = torch.full(
        (chunk2_len, topk_tokens), -55, dtype=torch.int32, device=device
    )
    idx_recv_fb = _stub(kv_recv_fallback, topk_fallback)

    # Sender runs with empty attn_metadata while wire2 still holds valid non-zero header
    # from above AND poisoned tail payload (-123) that would corrupt topk if copied!
    wire2.view(-1)[4:].fill_(-123)
    with set_forward_context({}, vllm_cfg):
        sent_missing = hy4_fp8_indexer_forward_offload_sender(
            indexer=idx_sender,
            normed_hidden_states=normed2,
            fused_qkv_a_proj=fused_qkv_fn,
            q_a_layernorm=qa_norm,
            q_lora_rank=q_lora_rank,
            kv_lora_and_rope_dim=kv_lora_and_rope,
            supports_wq_b_prequant=True,
            positions=pos2,
            rotary_emb=rotary_emb,
            wire_out=wire2,
        )
    assert int(sent_missing.view(-1)[1].item()) == 0

    with set_forward_context({"layer9.k_cache": md2}, vllm_cfg):
        idx_recv_fb._offload_recv_wire = sent_missing.clone()
        qc2_fb = qa_norm(fused_qkv_fn(normed2)[0][:, :q_lora_rank])
        hy4_fp8_indexer_forward(idx_recv_fb, normed2, qc2_fb, pos2, rotary_emb)
        idx_recv_fb._offload_recv_wire = None

    assert torch.equal(topk_fallback, topk_ref2)


@pytest.mark.parametrize("false_str", ["0", "false", "False", "no", "off", ""])
def test_canonical_env_switches_and_nan_ratio_handling(monkeypatch, false_str: str):
    """Pre-push audit: boolean/float env switches must accept false/off/empty and reject NaN."""
    from vllm_iluvatar import envs as ilu_envs
    from vllm_iluvatar.models.hy_v4 import flashmla_sparse as fmla_mod
    from vllm_iluvatar.models.hy_v4.fp8_indexer import compute_offload_split_t

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_MLA_REMAP", false_str)
    assert fmla_mod._hyv4_fuse_mla_remap() is False

    # Explicit empty primary env var must NOT fall through to legacy truthy alias.
    monkeypatch.setenv("VLLM_ILU_MXFP4_MOE_PREQUANT", false_str)
    monkeypatch.setenv("VLLM_ILU_HYV4_MOE_EXPAND_PREQUANT", "1")
    assert ilu_envs.VLLM_ILU_MXFP4_MOE_PREQUANT is False

    # NaN or invalid float ratio must disable offload cleanly instead of raising.
    bt = torch.zeros((1, 16), dtype=torch.int32)
    chunk = _build_prefill_chunk(
        [512], [512], bt, token_offset=0, device=torch.device("cpu")
    )
    md = IluDeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([512], dtype=torch.int32),
        max_seq_len=512,
        slot_mapping=torch.arange(512, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=512,
        prefill=IluDeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
    )
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_RATIO", "nan")
    assert compute_offload_split_t(md, 512, 2048) is None
    monkeypatch.setenv("VLLM_ILU_HYV4_PP_OFFLOAD_RATIO", "not_a_float")
    assert compute_offload_split_t(md, 512, 2048) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CoreX GPU")
def test_ivcore11_and_dtype_gates_fallback_cleanly(monkeypatch):
    """Pre-push audit: ivcore30-only kernels & BF16 rms_weight gates must fall back cleanly."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import MLAAttention

    from vllm_iluvatar.models.hy_v4 import attention as attn_mod
    from vllm_iluvatar.models.hy_v4 import fp8_indexer as idx_mod
    from vllm_iluvatar.models.hy_v4.hc import HYV4HCHeadLayer

    device = torch.device("cuda:0")

    # 1) HYV4MLAAttentionLayer.fuse_v_up must be False on ivcore11 or unsupported kv_lora_rank.
    def _init_mla_base(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.impl = SimpleNamespace(is_sparse=True, dcp_world_size=1)
        self.use_pcp = False
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self._vllm_config = SimpleNamespace(
            attention_config=SimpleNamespace(sparse_mla_force_mqa=True)
        )

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_KV_UP", "1")
    with (
        patch.object(MLAAttention, "__init__", _init_mla_base),
        patch.object(attn_mod, "is_ivcore30", return_value=False),
    ):
        layer_bi = attn_mod.HYV4MLAAttentionLayer()
    assert layer_bi.fuse_v_up is False
    assert layer_bi.fuse_gate is False

    monkeypatch.setenv("VLLM_ILU_HYV4_FUSE_KV_UP", "false")
    with (
        patch.object(MLAAttention, "__init__", _init_mla_base),
        patch.object(attn_mod, "is_ivcore30", return_value=True),
    ):
        layer_off = attn_mod.HYV4MLAAttentionLayer()
    assert layer_off.fuse_v_up is False

    # 2) can_fuse_qa_norm_quant rejects ivcore11, non-contiguous qkv_lora, or mismatched norm dtype.
    qkv = torch.randn(16, 2112, device=device, dtype=torch.bfloat16)
    qc = qkv[:, :1536]
    ln_bf16 = SimpleNamespace(
        weight=torch.ones(1536, device=device, dtype=torch.bfloat16)
    )
    ln_fp32 = SimpleNamespace(
        weight=torch.ones(1536, device=device, dtype=torch.float32)
    )
    with patch.object(attn_mod, "is_ivcore30", return_value=False):
        assert not attn_mod.HYV4MLAAttention.can_fuse_qa_norm_quant(qkv, qc, ln_bf16)
    assert not attn_mod.HYV4MLAAttention.can_fuse_qa_norm_quant(qkv, qc, ln_fp32)
    qkv_noncontig = qkv.t().contiguous().t()
    assert not attn_mod.HYV4MLAAttention.can_fuse_qa_norm_quant(
        qkv_noncontig, qkv_noncontig[:, :1536], ln_bf16
    )

    # 3) _can_fuse_indexer_k_insert rejects ivcore11 or RMSNorm (bias=None) on CUDA tensors.
    k_raw = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    cos_sin = torch.randn(16, 64, device=device, dtype=torch.float32)
    kv_cache = torch.zeros((4, 64, 132), device=device, dtype=torch.uint8)
    idx_stub = SimpleNamespace(
        k_norm=SimpleNamespace(
            weight=torch.ones(128, device=device, dtype=torch.float32),
            bias=torch.zeros(128, device=device, dtype=torch.float32),
        )
    )
    with patch.object(idx_mod, "is_ivcore30", return_value=False):
        assert not idx_mod._can_fuse_indexer_k_insert(
            idx_stub, k_raw, cos_sin, kv_cache
        )
    idx_no_bias = SimpleNamespace(
        k_norm=SimpleNamespace(
            weight=torch.ones(128, device=device, dtype=torch.float32),
            bias=None,
        )
    )
    assert not idx_mod._can_fuse_indexer_k_insert(idx_no_bias, k_raw, cos_sin, kv_cache)

    # 4) HYV4HCLayer / HYV4HCHeadLayer reject non-BF16 rms_weight and guard eager fallback.
    cfg = SimpleNamespace(
        hidden_size=4096,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        enable_ihc=True,
    )
    hc_attn, _, in_ln, _ = _build_hc_and_norms_no_dist(cfg, device)
    x = torch.randn(8, 4, 4096, device=device, dtype=torch.bfloat16)
    assert hc_attn.can_fuse_norm(x, in_ln.weight.data)
    assert not hc_attn.can_fuse_norm(x, in_ln.weight.data.float())

    tp_stub = SimpleNamespace(rank_in_group=0, world_size=1)
    with (
        set_current_vllm_config(VllmConfig()),
        patch("vllm.distributed.parallel_state._TP", tp_stub),
    ):
        hc_head = HYV4HCHeadLayer(cfg, cfg.hidden_size, prefix="hc_head").to(device)
    assert hc_head.can_fuse_norm(x, in_ln.weight.data)
    assert not hc_head.can_fuse_norm(x, in_ln.weight.data.float())
    with pytest.raises(RuntimeError, match="requires native iHC projection"):
        hc_head(x.float(), rms_weight=in_ln.weight.data, rms_eps=1e-6)
