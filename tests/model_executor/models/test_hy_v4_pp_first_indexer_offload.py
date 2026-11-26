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
    full_layers = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77}
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
            _compute_pp_first_indexer_offload_hops(
                cfg, cache_cfg, 16, pcp_size=2
            )
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
        CacheConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm

    from vllm_iluvatar.models.hy_v4.attention import build_hy4_ropes
    from vllm_iluvatar.models.hy_v4.model import (
        _quant_pp_act,
        _run_ihc_pre_fp8_boundary_and_norm_impl,
    )

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
        0, 256, (total_blocks, block_size, head_dim + 4), dtype=torch.uint8, device=device
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

    w_qkv = (torch.randn(q_lora_rank + kv_lora_and_rope, hidden_size, device=device, dtype=dtype) * 0.02)
    w_qkv_fp8, w_qkv_s = ixfops.fp8_per_token_dynamic_scaled(w_qkv)
    w_wq_b = (torch.randn(n_head * head_dim, q_lora_rank, device=device, dtype=dtype) * 0.02)
    w_wq_b_fp8, w_wq_b_s = ixfops.fp8_per_token_dynamic_scaled(w_wq_b)
    w_wk = (torch.randn(head_dim + n_head, hidden_size, device=device, dtype=dtype) * 0.02)
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
    topk_ref = torch.full((total_tokens, topk_tokens), -7, dtype=torch.int32, device=device)
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

    assert torch.equal(kv_sender, kv_ref), "Sender mirrored K-cache diverged from reference"
    assert torch.equal(kv_recv, kv_ref), "Receiver K-cache diverged from reference"
    assert torch.equal(
        topk_recv, topk_ref
    ), "Offloaded + local topk_indices diverged from unsplit reference"


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

