# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 lightning indexer for Hunyuan V4 (grouped fp8 K cache).

The BF16 indexer in ``attention.py`` keeps K in the checkpoint dtype and feeds
the cuinfer bf16 mqa-logits kernel. This variant runs the same three steps on
the fp8 lane that DeepSeek V4 uses -- ``ds4_fused_indexer_q_rope_fp8_quant``,
a grouped fp8 K insert, and ``dsa_indexer_mqa_logits_with_kvcache_fp8`` --
which measures 2.2~2.6x faster on hy4's prefill shapes and up to 9x on decode,
and halves the indexer K cache (132 B/token vs 256 B).

Three things make the reuse possible despite hy4 running ``index_n_heads=32``
against ds4's 64:

* ``index_head_dim`` is 128 on both. The shared FP8 facade dispatches to
  native H32/H64 specializations and honors explicit registry overrides.
  HY4 retains all 32 indexer heads on every TP rank.
* Both models build rotary embeddings with ``is_neox_style=False`` and keep
  the RoPE band in the last ``rope_dim`` dims, matching the fused q op's
  contract exactly.
* hy4 has no compressor. K is LayerNorm + tail RoPE + a grouped fp8
  scatter, fused as ``hy4_indexer_k_norm_rope_fp8_insert`` when that op is
  present (``VLLM_ILU_HYV4_FUSE_INDEXER_K=0`` keeps the three-kernel path).
"""

from __future__ import annotations

import math
from typing import Any

import ixformer.inference.functions as ixfops
import torch
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.envs import _ILU_CANONICAL_ONLY_ENV_VARS
from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.model_executor.kernels.linear.scaled_mm.ilu_fp8_block_scaled_mm import (
    register_activation_prequant,
    sliced_activation_prequant,
    unregister_activation_prequant,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerBackend,
    IluDeepseekV32IndexerMetadata,
    IluDeepseekV32IndexerPrefillChunkMetadata,
)

logger = init_logger(__name__)

# 4-byte fp32 ue8m0 scale per token (quant_block_size == index_head_dim).
_INDEXER_FP8_SCALE_BYTES = 4


class HYV4Fp8IndexerCache(torch.nn.Module, AttentionLayerBase):
    """Grouped fp8 indexer K cache: ``uint8 [num_blocks, block_size, D + 4]``.

    Within a block the ``[block_size, D]`` e4m3 K region and the
    ``[block_size]`` fp32 ue8m0 scale region are stored *grouped* rather than
    interleaved per token, so every K row stays 128B/16-aligned for the ixmma
    matrix loader.
    """

    def __init__(self, head_dim: int, prefix: str, cache_config: CacheConfig) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim + _INDEXER_FP8_SCALE_BYTES
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = torch.uint8
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV32IndexerBackend


def _mixed_mrope_cos_sin(
    mrope_emb: torch.nn.Module,
    positions: torch.Tensor,
    dtype_ref: torch.Tensor,
) -> torch.Tensor:
    """Per-token ``cos || sin`` for a ``[3, T]`` M-RoPE position.

    The fused indexer kernels index a 1-D table and apply GPT-J RoPE. Feeding
    them ``arange(T)`` plus this table is the same rotation ``MRotaryEmbedding``
    would apply to the three rows.
    """
    from vllm.model_executor.layers.rotary_embedding.mrope import (
        apply_interleaved_rope,
    )

    cache = mrope_emb._match_cos_sin_cache_dtype(dtype_ref)
    gathered = cache[positions.long()]
    half = gathered.shape[-1] // 2
    cos = gathered[..., :half]
    sin = gathered[..., half:]
    section = [int(v) for v in mrope_emb.mrope_section]
    if mrope_emb.mrope_interleaved:
        cos = apply_interleaved_rope(cos, section)
        sin = apply_interleaved_rope(sin, section)
    else:
        cos_parts = []
        sin_parts = []
        offset = 0
        for row, width in enumerate(section):
            cos_parts.append(cos[row, ..., offset : offset + width])
            sin_parts.append(sin[row, ..., offset : offset + width])
            offset += width
        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)
    return torch.cat((cos, sin), dim=-1).contiguous()


def _indexer_rope_table(
    indexer,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
    dtype_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(positions, cos_sin)`` the fused indexer kernels can consume.

    Three-row positions always use a mixed per-token table. This is also
    equivalent for text's identical rows and avoids a host read during capture.
    """
    if positions.ndim != 2:
        return positions.contiguous(), rotary_emb._match_cos_sin_cache_dtype(dtype_ref)
    prefetched = getattr(indexer, "_prefetched_mrope_table", None)
    if prefetched is not None:
        indexer._prefetched_mrope_table = None
        return prefetched
    mrope_emb = getattr(indexer, "mrope_emb", None)
    if mrope_emb is None or positions.shape[0] != 3:
        raise ValueError("HYV4 2-D positions require three rows and an M-RoPE module")
    section = getattr(mrope_emb, "_ixf_mrope_section", None)
    if (
        positions.is_cuda
        and positions.dtype == torch.int64
        and positions.stride(-1) == 1
        and positions.shape[1] <= 2560
        and section is not None
        and hasattr(ixfops, "mrope_cos_sin_table")
    ):
        return ixfops.mrope_cos_sin_table(
            positions,
            mrope_emb._match_cos_sin_cache_dtype(dtype_ref),
            section,
            mrope_emb.mrope_interleaved,
        )
    table = _mixed_mrope_cos_sin(mrope_emb, positions, dtype_ref)
    token_pos = torch.arange(table.shape[0], device=table.device, dtype=positions.dtype)
    return token_pos, table


def _can_fuse_indexer_k_insert(
    indexer: Any,
    k_raw: torch.Tensor,
    cos_sin: torch.Tensor,
    kv_cache: torch.Tensor,
) -> bool:
    if not _ILU_CANONICAL_ONLY_ENV_VARS[
        "VLLM_ILU_HYV4_FUSE_INDEXER_K"
    ]() or not hasattr(ixfops, "hy4_indexer_k_norm_rope_fp8_insert"):
        return False
    if not k_raw.is_cuda:
        # Host-only unit tests mock ixfops; real execution is always CUDA.
        return True
    k_norm_weight = getattr(indexer.k_norm, "weight", None)
    k_norm_bias = getattr(indexer.k_norm, "bias", None)
    return (
        is_ivcore30()
        and k_raw.dtype == torch.bfloat16
        and k_raw.ndim == 2
        and k_raw.shape[1] == 128
        and k_raw.stride(1) == 1
        and cos_sin.is_cuda
        and cos_sin.ndim == 2
        and cos_sin.shape[1] == 64
        and cos_sin.stride(1) == 1
        and cos_sin.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and k_norm_weight is not None
        and k_norm_bias is not None
        and k_norm_weight.is_cuda
        and k_norm_bias.is_cuda
        and k_norm_weight.is_contiguous()
        and k_norm_bias.is_contiguous()
        and k_norm_weight.dtype == torch.float32
        and k_norm_bias.dtype == torch.float32
        and k_norm_weight.shape == (128,)
        and k_norm_bias.shape == (128,)
        and kv_cache.is_cuda
        and kv_cache.ndim == 3
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[1] > 0
        and kv_cache.shape[2] == 132
        and kv_cache.stride(2) == 1
        and kv_cache.stride(1) == 132
    )


def _insert_indexer_k_cache(
    indexer,
    kw: torch.Tensor,
    indexer_positions: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
    md: IluDeepseekV32IndexerMetadata,
) -> None:
    kv_cache = indexer.k_cache.kv_cache
    slot_mapping = md.slot_mapping
    if getattr(md, "pcp_world_size", 1) > 1:
        raise NotImplementedError(
            "HY4 FP8 indexer requires PCP=1 on this vLLM baseline"
        )
    # Grid follows slot_mapping's (CUDA-graph padded) length; padding rows
    # carry slot_mapping == -1 and are skipped inside the kernel.
    num_tokens = slot_mapping.shape[0]
    n = min(num_tokens, kw.shape[0], int(indexer_positions.numel()))
    if n <= 0:
        return
    k_raw = kw[:n, : indexer.head_dim]
    pos = indexer_positions[:n].contiguous()
    if _can_fuse_indexer_k_insert(indexer, k_raw, cos_sin, kv_cache):
        if pos.dtype != torch.int64:
            pos = pos.to(torch.int64)
        slots = slot_mapping.reshape(-1)[:n].contiguous()
        if slots.dtype != torch.int64:
            slots = slots.to(torch.int64)
        ixfops.hy4_indexer_k_norm_rope_fp8_insert(
            k_raw,
            indexer.k_norm.weight,
            indexer.k_norm.bias,
            pos,
            cos_sin,
            kv_cache,
            slots,
            float(indexer.k_norm.eps),
        )
    else:
        k = indexer.k_norm(k_raw)
        nope_dim = indexer.head_dim - indexer.rope_dim
        mrope_emb = getattr(indexer, "mrope_emb", None)
        if positions.ndim == 2:
            assert mrope_emb is not None  # validated by _indexer_rope_table
            # Same three-row rotation as the fused insert. ``rotary_emb`` here
            # is the 1-D module and would ignore the mixed table.
            k_pe = k[:, nope_dim:]
            k_out, _ = mrope_emb(positions[:, :n], k_pe, k_pe)
            k = k.clone()
            k[:, nope_dim:] = k_out.reshape(k_pe.shape)
        else:
            k_pe = k[:, nope_dim:].unsqueeze(1)
            k_pe_out, _ = rotary_emb(pos, k_pe)
            if k_pe_out is not k_pe:
                k_pe.copy_(k_pe_out.reshape(k_pe.shape))
        slots = slot_mapping.reshape(-1)[:n].contiguous()
        ixfops.hy4_indexer_k_fp8_insert(k, kv_cache, slots)


# Leading 4 int32s (16 bytes, preserving 16-byte vector alignment for
# dsa_update_topk_indices) in the (N, index_topk // 2) offload wire:
#   header[0] = sent_split_t
#   header[1] = sent_tail_len   (0 when sender did not populate tail top-k)
#   header[2..3] = 0            (alignment padding)
_OFFLOAD_WIRE_HEADER_INTS = 4


def _clear_offload_wire_header(wire: torch.Tensor) -> None:
    flat = wire.view(-1)
    if flat.numel() >= _OFFLOAD_WIRE_HEADER_INTS:
        flat[:_OFFLOAD_WIRE_HEADER_INTS].zero_()


def _write_offload_wire_header(wire: torch.Tensor, split_t: int, tail_len: int) -> None:
    flat = wire.view(-1)
    assert flat.numel() >= _OFFLOAD_WIRE_HEADER_INTS
    flat[0].fill_(split_t)
    flat[1].fill_(tail_len)


def _verify_offload_wire_header(
    wire: torch.Tensor,
    expected_split_t: int,
    expected_tail_len: int,
    topk_tokens: int,
) -> tuple[bool, int, int]:
    flat = wire.view(-1)
    required = _OFFLOAD_WIRE_HEADER_INTS + expected_tail_len * topk_tokens
    if flat.numel() < required:
        return False, 0, 0
    sent_split_t, sent_tail_len = (int(x) for x in flat[:2].to(device="cpu").tolist())
    ok = (
        sent_tail_len > 0
        and sent_split_t == expected_split_t
        and sent_tail_len == expected_tail_len
    )
    return ok, sent_split_t, sent_tail_len


def compute_offload_split_t(
    md: IluDeepseekV32IndexerMetadata | None,
    total_tokens: int,
    index_topk: int,
) -> tuple[int, int] | None:
    """Deterministically split prefill query tokens between local head and offloaded tail.

    Returns ``(split_t, prefill_end)`` where the receiver rank computes
    ``[0, split_t)`` and the predecessor helper rank computes
    ``[split_t, prefill_end)``, or ``None`` when the batch is too small /
    decode-only. Uses CPU chunk metadata exclusively (zero GPU-CPU sync) and
    snaps ``split_t`` to an 8-row CTA boundary inside the split request so
    every ``dsa_indexer_mqa_logits_fp8_h32`` CTA stays bit-identical.
    """
    if (
        md is None
        or md.num_prefills <= 0
        or md.prefill is None
        or not md.prefill.chunks
        or index_topk <= 0
        or index_topk % 2 != 0
        or getattr(md, "pcp_world_size", 1) > 1
    ):
        return None

    chunks = md.prefill.chunks
    prefill_start = chunks[0].token_start
    prefill_end = chunks[-1].token_end
    prefill_tokens = prefill_end - prefill_start
    try:
        min_tokens = _ILU_CANONICAL_ONLY_ENV_VARS[
            "VLLM_ILU_HYV4_PP_OFFLOAD_MIN_TOKENS"
        ]()
        ratio = _ILU_CANONICAL_ONLY_ENV_VARS["VLLM_ILU_HYV4_PP_OFFLOAD_RATIO"]()
        min_half_tokens = _ILU_CANONICAL_ONLY_ENV_VARS[
            "VLLM_ILU_HYV4_PP_OFFLOAD_MIN_HALF_TOKENS"
        ]()
    except ValueError:
        return None
    if not math.isfinite(ratio):
        return None
    if total_tokens < 128 or prefill_end != total_tokens or prefill_tokens < min_tokens:
        return None

    wire_ints = total_tokens * (index_topk // 2)
    if wire_ints <= _OFFLOAD_WIRE_HEADER_INTS:
        return None
    max_wire_tokens = (wire_ints - _OFFLOAD_WIRE_HEADER_INTS) // index_topk
    ratio = min(max(ratio, 0.05), 0.50)
    desired_tail = min(int(round(prefill_tokens * ratio)), max_wire_tokens)
    if desired_tail <= 0:
        return None

    raw_split_t = prefill_end - desired_tail
    split_t = raw_split_t
    for chunk in chunks:
        if not (chunk.token_start <= raw_split_t < chunk.token_end):
            continue
        cu_q_cpu = (
            chunk.cu_seqlens_q_cpu
            if chunk.cu_seqlens_q_cpu is not None
            else [int(x) for x in chunk.cu_seqlens_q.cpu().tolist()]
        )
        rel = raw_split_t - chunk.token_start
        for b in range(chunk.num_reqs):
            q_s = cu_q_cpu[b]
            q_e = cu_q_cpu[b + 1]
            if q_s <= rel < q_e:
                aligned_rel = min(q_s + ((rel - q_s + 7) // 8) * 8, q_e)
                split_t = chunk.token_start + aligned_rel
                break
        break

    # Keep >10 Q CTAs (88+ rows at BQ=8) on both sides so neither half flips
    # into the small-M score specialization.
    head_prefill_tokens = split_t - prefill_start
    tail_tokens = prefill_end - split_t
    if (
        head_prefill_tokens < min_half_tokens
        or tail_tokens < min_half_tokens
        or tail_tokens > max_wire_tokens
    ):
        return None
    return split_t, prefill_end


def slice_indexer_prefill_chunk(
    chunk: IluDeepseekV32IndexerPrefillChunkMetadata,
    q_start: int,
    q_end: int,
) -> IluDeepseekV32IndexerPrefillChunkMetadata | None:
    """Restrict ``chunk`` to global query token range ``[q_start, q_end)``."""
    if chunk.token_end <= q_start or chunk.token_start >= q_end:
        return None
    if q_start <= chunk.token_start and chunk.token_end <= q_end:
        return chunk

    sub_start = max(chunk.token_start, q_start)
    sub_end = min(chunk.token_end, q_end)
    rel_s = sub_start - chunk.token_start
    rel_e = sub_end - chunk.token_start
    sub_m = rel_e - rel_s
    if sub_m <= 0:
        return None

    cu_q_cpu = (
        chunk.cu_seqlens_q_cpu
        if chunk.cu_seqlens_q_cpu is not None
        else [int(x) for x in chunk.cu_seqlens_q.cpu().tolist()]
    )
    b_lo = 0
    while b_lo < chunk.num_reqs and cu_q_cpu[b_lo + 1] <= rel_s:
        b_lo += 1
    b_hi = b_lo
    while b_hi < chunk.num_reqs and cu_q_cpu[b_hi] < rel_e:
        b_hi += 1
    sub_num_reqs = b_hi - b_lo
    if sub_num_reqs <= 0:
        return None

    sub_cu_q_cpu = [
        min(max(cu_q_cpu[b] - rel_s, 0), sub_m) for b in range(b_lo, b_hi + 1)
    ]
    sub_max_q = max(sub_cu_q_cpu[i + 1] - sub_cu_q_cpu[i] for i in range(sub_num_reqs))
    if b_lo == 0 and rel_s == 0 and cu_q_cpu[b_hi] == sub_m:
        sub_cu_q = chunk.cu_seqlens_q[: b_hi + 1]
    else:
        sub_cu_q = (chunk.cu_seqlens_q[b_lo : b_hi + 1] - rel_s).clamp_(0, sub_m)

    if b_lo == 0 and b_hi == chunk.num_reqs:
        sub_cu_kv = chunk.cu_seq_lens
        sub_bt = chunk.block_table
        sub_max_ctx = chunk.max_context_len
        sub_max_kv = chunk.max_kv_len
        sub_total_seq = chunk.total_seq_lens
        sub_seq_comp_cpu = chunk.seq_lens_comp_cpu
    else:
        if chunk.seq_lens_comp_cpu is not None:
            sub_seq_comp_cpu = chunk.seq_lens_comp_cpu[b_lo:b_hi]
        else:
            cu_kv_cpu = [int(x) for x in chunk.cu_seq_lens.cpu().tolist()]
            sub_seq_comp_cpu = [
                cu_kv_cpu[b + 1] - cu_kv_cpu[b] for b in range(b_lo, b_hi)
            ]
        if b_lo == 0:
            sub_cu_kv = chunk.cu_seq_lens[: b_hi + 1]
        else:
            sub_cu_kv = (
                chunk.cu_seq_lens[b_lo : b_hi + 1] - chunk.cu_seq_lens[b_lo : b_lo + 1]
            )
        sub_bt = chunk.block_table[b_lo:b_hi]
        sub_max_ctx = max(sub_seq_comp_cpu) if sub_seq_comp_cpu else 0
        sub_max_kv = sub_max_ctx
        sub_total_seq = sum(sub_seq_comp_cpu)

    return IluDeepseekV32IndexerPrefillChunkMetadata(
        block_table=sub_bt,
        cu_seqlen_ks=chunk.cu_seqlen_ks[rel_s:rel_e],
        cu_seqlen_ke=chunk.cu_seqlen_ke[rel_s:rel_e],
        valid_lens=chunk.valid_lens[rel_s:rel_e],
        cu_seq_lens=sub_cu_kv,
        cu_seqlens_q=sub_cu_q,
        token_to_seq=chunk.token_to_seq,
        total_seq_lens=sub_total_seq,
        token_start=sub_start,
        token_end=sub_end,
        num_reqs=sub_num_reqs,
        max_context_len=sub_max_ctx,
        max_q_len=sub_max_q,
        max_kv_len=sub_max_kv,
        skip_kv_gather=chunk.skip_kv_gather,
        context_lower_bound=0,
        cu_seqlens_q_cpu=sub_cu_q_cpu,
        seq_lens_comp_cpu=sub_seq_comp_cpu,
    )


def _run_prefill_chunks_range(
    chunks: list[IluDeepseekV32IndexerPrefillChunkMetadata],
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_tokens: int,
    out_buffer: torch.Tensor,
    q_start: int,
    q_end: int,
    q_base_offset: int,
) -> None:
    for chunk in chunks:
        sub = slice_indexer_prefill_chunk(chunk, q_start, q_end)
        if sub is None or sub.max_context_len == 0:
            continue
        local_s = sub.token_start - q_base_offset
        local_e = sub.token_end - q_base_offset
        logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
            q_fp8[local_s:local_e],
            sub.cu_seqlens_q,
            sub.cu_seq_lens,
            kv_cache,
            sub.block_table,
            weights[local_s:local_e],
            max_q_len=sub.max_q_len,
            max_kv_len=sub.max_kv_len,
            max_context_len=sub.max_context_len,
            valid_lens=sub.valid_lens,
            context_lower_bound=sub.context_lower_bound,
        )
        ixfops.dsa_update_topk_indices(
            logits,
            sub.cu_seqlen_ks,
            sub.cu_seqlen_ke,
            topk_tokens,
            out_buffer[local_s:local_e],
        )


def hy4_fp8_indexer_forward_offload_sender(
    indexer,
    normed_hidden_states: torch.Tensor,
    fused_qkv_a_proj: torch.nn.Module,
    q_a_layernorm: torch.nn.Module,
    q_lora_rank: int,
    kv_lora_and_rope_dim: int,
    supports_wq_b_prequant: bool,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
    wire_out: torch.Tensor,
) -> torch.Tensor:
    """Predecessor-stage helper: sync mirrored K-cache and compute offloaded tail top-k.

    ``wire_out`` is a physical ``(N, index_topk // 2)`` int32 plane. The
    leading 4 int32s store ``[sent_split_t, sent_tail_len, 0, 0]`` (with
    ``sent_tail_len == 0`` whenever the sender skipped tail computation), and
    the tail top-k is packed immediately after as a contiguous
    ``(tail_len, index_topk)`` view over ``flat[4 : 4 + tail_len * index_topk]``.
    """
    _clear_offload_wire_header(wire_out)

    attn_metadata = get_forward_context().attn_metadata
    md = (
        attn_metadata.get(indexer.k_cache.prefix)
        if isinstance(attn_metadata, dict)
        else None
    )
    if md is None:
        # Startup memory profiling often has empty attn_metadata; still
        # exercise the tail projection shapes so peak activation includes
        # the helper workspace. Outside profiling, header[1] == 0 tells the
        # receiver to fall back cleanly to a full local indexer.
        logger.warning_once(
            "HY4 PP offload sender missing indexer metadata for %s; "
            "marking wire tail_len=0 so receiver runs full local indexer.",
            getattr(indexer.k_cache, "prefix", "<unknown>"),
        )
        _ = indexer.wk_weights_proj(normed_hidden_states)
        total = normed_hidden_states.shape[0]
        if total >= 256:
            half_s = total - (total // 2)
            with sliced_activation_prequant(
                normed_hidden_states, half_s, total
            ) as sub_h:
                qkv_lora = fused_qkv_a_proj(sub_h)[0]
            q_c, _ = qkv_lora.split([q_lora_rank, kv_lora_and_rope_dim], dim=-1)
            q_c = q_a_layernorm(q_c)
            _ = indexer.wq_b(q_c)
        return wire_out

    assert isinstance(md, IluDeepseekV32IndexerMetadata)
    if getattr(md, "pcp_world_size", 1) > 1:
        return wire_out

    # Always synchronize the helper's mirrored K-cache even when this batch
    # does not split queries (e.g. small prefill chunks, decode steps, or mixed
    # steps) so subsequent offloaded prefill chunks see complete prior K slots.
    indexer_positions, cos_sin = _indexer_rope_table(
        indexer, positions, rotary_emb, normed_hidden_states
    )
    kw, _ = indexer.wk_weights_proj(normed_hidden_states)
    _insert_indexer_k_cache(
        indexer, kw, indexer_positions, cos_sin, positions, rotary_emb, md
    )

    split_range = compute_offload_split_t(
        md, normed_hidden_states.shape[0], indexer.topk_tokens
    )
    if split_range is None:
        return wire_out

    split_t, prefill_end = split_range
    tail_len = prefill_end - split_t
    with sliced_activation_prequant(
        normed_hidden_states, split_t, prefill_end
    ) as sub_hidden:
        qkv_lora = fused_qkv_a_proj(sub_hidden)[0]

    q_c, _ = qkv_lora.split([q_lora_rank, kv_lora_and_rope_dim], dim=-1)
    qc_prequant_registered = False
    try:
        from .attention import HYV4MLAAttention

        if supports_wq_b_prequant and HYV4MLAAttention.can_fuse_qa_norm_quant(
            qkv_lora,
            q_c,
            q_a_layernorm,
            allow_breakable_cudagraph=not torch.cuda.is_current_stream_capturing(),
        ):
            qc_fp8, qc_scale = ixfops.fp8_per_token_dynamic_scaled(
                q_c,
                rms_weight=q_a_layernorm.weight,
                rms_eps=float(q_a_layernorm.variance_epsilon),
            )
            q_c = qkv_lora.view(-1)[: q_c.shape[0] * q_lora_rank].view(
                q_c.shape[0], q_lora_rank
            )
            register_activation_prequant(q_c, qc_fp8, qc_scale)
            qc_prequant_registered = True
        else:
            q_c = q_a_layernorm(q_c)
        q_tail, _ = indexer.wq_b(q_c)
    finally:
        if qc_prequant_registered:
            unregister_activation_prequant(q_c)

    q_tail = q_tail.view(-1, indexer.n_head, indexer.head_dim)
    q_fp8_tail, weights_tail = ixfops.fused_indexer_q_rope_quant(
        q_tail,
        indexer_positions[split_t:prefill_end].contiguous(),
        cos_sin,
        kw[split_t:prefill_end, indexer.head_dim :].contiguous(),
        indexer.softmax_scale,
        indexer.n_head_scale,
        indexer.rope_dim,
    )

    topk_tokens = indexer.topk_tokens
    payload_start = _OFFLOAD_WIRE_HEADER_INTS
    payload_end = payload_start + tail_len * topk_tokens
    topk_tail_out = wire_out.view(-1)[payload_start:payload_end].view(
        tail_len, topk_tokens
    )
    assert md.prefill is not None
    _run_prefill_chunks_range(
        md.prefill.chunks,
        q_fp8_tail,
        weights_tail,
        indexer.k_cache.kv_cache,
        topk_tokens,
        topk_tail_out,
        q_start=split_t,
        q_end=prefill_end,
        q_base_offset=split_t,
    )
    _write_offload_wire_header(wire_out, split_t, tail_len)
    return wire_out


def hy4_fp8_indexer_forward(
    indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
) -> torch.Tensor:
    """fp8 replacement for ``Indexer.forward``; ``indexer`` is that module."""
    topk_indices_buffer = indexer.topk_indices_buffer
    attn_metadata = get_forward_context().attn_metadata
    md = (
        attn_metadata.get(indexer.k_cache.prefix)
        if isinstance(attn_metadata, dict)
        else None
    )
    offload_recv_wire = getattr(indexer, "_offload_recv_wire", None)
    if offload_recv_wire is not None and isinstance(md, IluDeepseekV32IndexerMetadata):
        split_range = compute_offload_split_t(
            md, hidden_states.shape[0], indexer.topk_tokens
        )
        if split_range is not None:
            split_t, prefill_end = split_range
            tail_len = prefill_end - split_t
            topk_tokens = indexer.topk_tokens
            wire_ok, sent_split_t, sent_tail_len = _verify_offload_wire_header(
                offload_recv_wire, split_t, tail_len, topk_tokens
            )
            if not wire_ok:
                logger.warning_once(
                    "HY4 PP offload receiver wire mismatch for %s "
                    "(expected split_t=%d tail_len=%d, got split_t=%d tail_len=%d); "
                    "falling back to full local indexer.",
                    getattr(indexer.k_cache, "prefix", "<unknown>"),
                    split_t,
                    tail_len,
                    sent_split_t,
                    sent_tail_len,
                )
            else:
                indexer_positions, cos_sin = _indexer_rope_table(
                    indexer, positions, rotary_emb, hidden_states
                )
                kw, _ = indexer.wk_weights_proj(hidden_states)
                _insert_indexer_k_cache(
                    indexer, kw, indexer_positions, cos_sin, positions, rotary_emb, md
                )

                with sliced_activation_prequant(qr, 0, split_t) as qr_head:
                    q_head, _ = indexer.wq_b(qr_head)
                q_head = q_head.view(-1, indexer.n_head, indexer.head_dim)
                q_fp8_head, weights_head = ixfops.fused_indexer_q_rope_quant(
                    q_head,
                    indexer_positions[:split_t].contiguous(),
                    cos_sin,
                    kw[:split_t, indexer.head_dim :].contiguous(),
                    indexer.softmax_scale,
                    indexer.n_head_scale,
                    indexer.rope_dim,
                )

                payload_start = _OFFLOAD_WIRE_HEADER_INTS
                payload_end = payload_start + tail_len * topk_tokens
                topk_indices_buffer[split_t:prefill_end].copy_(
                    offload_recv_wire.view(-1)[payload_start:payload_end].view(
                        tail_len, topk_tokens
                    )
                )
                kv_cache = indexer.k_cache.kv_cache
                assert md.prefill is not None
                _run_prefill_chunks_range(
                    md.prefill.chunks,
                    q_fp8_head,
                    weights_head,
                    kv_cache,
                    topk_tokens,
                    topk_indices_buffer,
                    q_start=0,
                    q_end=split_t,
                    q_base_offset=0,
                )

                if md.num_decodes > 0:
                    assert md.decode is not None
                    num_decode_tokens = md.num_decode_tokens
                    if md.decode.max_context_len == 0:
                        topk_indices_buffer[:num_decode_tokens].fill_(-1)
                        return topk_indices_buffer
                    logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
                        q_fp8_head[:num_decode_tokens],
                        md.decode.cu_seqlens_q,
                        md.decode.cu_seqlens_kv,
                        kv_cache,
                        md.decode.block_table,
                        weights_head[:num_decode_tokens],
                        max_q_len=md.decode.max_q_len,
                        max_kv_len=md.decode.max_kv_len,
                        max_context_len=md.decode.max_context_len,
                        valid_lens=md.decode.valid_lens,
                    )
                    ixfops.dsa_update_topk_indices(
                        logits,
                        md.decode.cu_seqlen_ks,
                        md.decode.cu_seqlen_ke,
                        topk_tokens,
                        topk_indices_buffer[:num_decode_tokens],
                    )
                return topk_indices_buffer

    indexer_positions, cos_sin = _indexer_rope_table(
        indexer, positions, rotary_emb, hidden_states
    )
    q, _ = indexer.wq_b(qr)
    q = q.view(-1, indexer.n_head, indexer.head_dim)

    kw, _ = indexer.wk_weights_proj(hidden_states)

    # Fused forward-RoPE + per-(token, head) ue8m0 fp8 quant. The op folds
    # softmax_scale, head_scale and the per-row q_scale into ``weights``, so
    # ``weights_in`` is the raw projection slice, *not* the pre-scaled
    # ``weights`` the bf16 path builds.
    q_fp8, weights = ixfops.fused_indexer_q_rope_quant(
        q,
        indexer_positions,
        cos_sin,
        kw[:, indexer.head_dim :].contiguous(),
        indexer.softmax_scale,
        indexer.n_head_scale,
        indexer.rope_dim,
    )

    if not isinstance(attn_metadata, dict) or md is None:
        return topk_indices_buffer
    assert isinstance(md, IluDeepseekV32IndexerMetadata)

    _insert_indexer_k_cache(
        indexer, kw, indexer_positions, cos_sin, positions, rotary_emb, md
    )
    kv_cache = indexer.k_cache.kv_cache

    topk_tokens = indexer.topk_tokens

    if md.num_prefills > 0:
        assert md.prefill is not None
        for chunk in md.prefill.chunks:
            if chunk.max_context_len == 0:
                continue
            logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
                q_fp8[chunk.token_start : chunk.token_end],
                chunk.cu_seqlens_q,
                chunk.cu_seq_lens,
                kv_cache,
                chunk.block_table,
                weights[chunk.token_start : chunk.token_end],
                max_q_len=chunk.max_q_len,
                max_kv_len=chunk.max_kv_len,
                max_context_len=chunk.max_context_len,
                valid_lens=chunk.valid_lens,
                context_lower_bound=chunk.context_lower_bound,
            )
            ixfops.dsa_update_topk_indices(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_tokens,
                topk_indices_buffer[chunk.token_start : chunk.token_end],
            )

    if md.num_decodes > 0:
        assert md.decode is not None
        num_decode_tokens = md.num_decode_tokens
        if md.decode.max_context_len == 0:
            # Early decode steps can have no KV entries yet: no top-k
            # candidates, and the logits kernel cannot materialize a matrix
            # with zero columns.
            topk_indices_buffer[:num_decode_tokens].fill_(-1)
            return topk_indices_buffer

        logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
            q_fp8[:num_decode_tokens],
            md.decode.cu_seqlens_q,
            md.decode.cu_seqlens_kv,
            kv_cache,
            md.decode.block_table,
            weights[:num_decode_tokens],
            max_q_len=md.decode.max_q_len,
            max_kv_len=md.decode.max_kv_len,
            max_context_len=md.decode.max_context_len,
            valid_lens=md.decode.valid_lens,
        )
        ixfops.dsa_update_topk_indices(
            logits,
            md.decode.cu_seqlen_ks,
            md.decode.cu_seqlen_ke,
            topk_tokens,
            topk_indices_buffer[:num_decode_tokens],
        )

    return topk_indices_buffer
