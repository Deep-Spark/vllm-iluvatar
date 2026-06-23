# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4 MLA attention layer (W4A8, BF16 KV cache).

Adapted from :mod:`vllm.model_executor.layers.deepseek_v4_attention` so the
ilu-fork W4A8 path can evolve independently. Stage 1 of the rollout (this
file): preserve the *entire* sub-module wiring required by
``load_weights`` -- ``fused_wqa_wkv`` / ``q_norm`` / ``wq_b`` / ``kv_norm``
/ ``wo_a`` / ``wo_b`` / ``attn_sink`` / ``q_head_norm`` /
``swa_cache_layer`` / ``mla_attn`` / ``compressor`` /
``IluDeepseekV4Indexer`` (with ``wq_b`` / ``weights_proj`` /
``compressor`` / ``k_cache``) -- so checkpoint init succeeds end-to-end.
Forward, prefill and decode methods all raise
``NotImplementedError("TODO(ilu_w4a8): ...")``: stage 2 will fill them in
once the bf16 MLA / sparse SWA / indexer kernels are stabilised.

Behavioural deltas vs the upstream V4 attention layer:

1. ``swa_cache_layer`` is :class:`IluDeepseekV4SWACache` with
   ``dtype=torch.bfloat16``.
2. ``IluDeepseekV4MLAAttention`` accepts any ``cache_config.cache_dtype`` and
   defaults to ``bfloat16``; the upstream "fp8 only" assertion is dropped.
   Its ``get_kv_cache_spec`` returns a bf16 :class:`MLAAttentionSpec` with
   ``cache_dtype_str="bfloat16"`` and ``model_version=None`` (the upstream
   ``"deepseek_v4"`` token is fp8-specific).
3. ``IluDeepseekV4IndexerCache`` is bf16 and pages without the fp8 scale
   padding (``head_size = config.index_head_dim`` directly).
4. ``IluDeepseekV4Indexer`` hardcodes ``use_fp4_kv = False`` and skips the
   ``SparseAttnIndexer`` op construction (it is a forward-only helper that
   would inspect the now-bf16 cache layout); forward raises TODO.
5. The ``deepseek_v4_attention`` / ``deepseek_v4_fp8_einsum`` custom ops are
   not re-registered: the wrapper's ``forward`` raises before they would be
   invoked. The upstream ones remain registered for the original V4 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import ixformer.inference.functions as ixfops
import torch
import torch.nn as nn
import vllm.envs as envs
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.compressor import (
    CompressorMetadata,
    DeepseekCompressor,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluDeepseekV4FlashMLASparseBackend,
    IluFlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV4IndexerBackend,
    IluDeepseekV32IndexerMetadata,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekV4SWACache,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


# Bound on bf16-gather workspace allocated at prefill.  Mirrors the
# upstream constant; kept here so stage-2 implementers don't have to
# rediscover the magic number when wiring ``_forward_prefill``.
PREFILL_CHUNK_SIZE = 4


def _swa_window_attention_torch(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    window_size: int,
    block_size: int,
    scale: float,
    attn_sink: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch sliding window attention for MLA prefill.

    MLA uses a shared latent for K and V (``head_dim``-wide), so the
    same ``swa_k_cache`` tensor serves as both K and V:
    ``score = q @ kv^T``, ``output = softmax(score) @ kv``.

    When ``return_lse=False`` (default), ``attn_sink`` is merged into the
    softmax denominator (matching ``ref_ds4_flash_mla_sparse_fwd``).

    When ``return_lse=True``, ``attn_sink`` is **not** applied (caller is
    responsible for merging it after combining partial results).  Returns
    ``(output, lse)`` where ``lse`` is ``[num_prefill_tokens, num_heads]``
    fp32 log-sum-exp **before** any sink adjustment.

    Args:
        q: Query tensor, ``[num_prefill_tokens, num_heads, head_dim]``.
        swa_k_cache: Paged KV cache, ``[num_blocks, block_size, head_dim]``.
        block_table: Per-sequence block mapping, ``[num_prefills, max_blocks]``.
        seq_lens: Total sequence lengths, ``[num_prefills]`` (GPU).
        gather_lens: Number of KV tokens to gather per seq, ``[num_prefills]`` (GPU).
        query_start_loc_cpu: Cumulative query counts, ``[num_prefills + 1]`` (CPU,
            relative to prefill start, i.e. first element is 0).
        window_size: Sliding window size.
        block_size: KV cache page block size.
        scale: Attention scale factor (typically ``head_dim ** -0.5``).
        attn_sink: Optional per-head sink logit, ``[num_heads]`` fp32.
            Ignored when ``return_lse=True``.
        return_lse: If True, return ``(output, lse)`` without applying
            ``attn_sink``.

    Returns:
        ``output`` when ``return_lse=False``;
        ``(output, lse)`` when ``return_lse=True``.
    """
    num_prefills = seq_lens.shape[0]
    num_heads = q.shape[1]
    num_tokens = q.shape[0]
    device = q.device

    output = torch.zeros_like(q)
    all_lse = (
        torch.full(
            (num_tokens, num_heads), float("-inf"), dtype=torch.float32, device=device
        )
        if return_lse
        else None
    )

    seq_lens_cpu = seq_lens.cpu()
    gather_lens_cpu = gather_lens.cpu()

    for i in range(num_prefills):
        s_len = seq_lens_cpu[i].item()
        g_len = gather_lens_cpu[i].item()
        q_start = query_start_loc_cpu[i].item()
        q_end = query_start_loc_cpu[i + 1].item()
        q_len = q_end - q_start

        q_i = q[q_start:q_end]  # [q_len, num_heads, head_dim]

        # Gather KV from paged cache: tokens at positions [s_len-g_len, s_len).
        start_pos = s_len - g_len
        positions = torch.arange(start_pos, s_len, device=device)
        blk_ids = block_table[i, positions // block_size]
        offsets = positions % block_size
        kv = swa_k_cache[blk_ids, offsets]  # [g_len, head_dim]

        # Absolute positions for mask construction.
        q_abs = torch.arange(s_len - q_len, s_len, device=device)
        k_abs = positions

        # Causal: q_pos >= k_pos.  Window: q_pos - k_pos < window_size.
        causal = q_abs.unsqueeze(1) >= k_abs.unsqueeze(0)
        window = (q_abs.unsqueeze(1) - k_abs.unsqueeze(0)) < window_size
        mask = causal & window  # [q_len, g_len]

        # fp32 accumulation for numerical stability.
        scores = torch.einsum("qhd,kd->qhk", q_i.float(), kv.float()) * scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        orig_lse = torch.logsumexp(scores, dim=-1)  # [q_len, num_heads]

        if return_lse:
            # Return raw LSE; caller handles attn_sink after merge.
            lse_for_w = orig_lse.clone()
            lse_for_w[lse_for_w == float("-inf")] = float("inf")
            attn_w = torch.exp(scores - lse_for_w.unsqueeze(-1))
            out_i = torch.einsum("qhk,kd->qhd", attn_w, kv.float())
            output[q_start:q_end] = out_i.to(q.dtype)
            all_lse[q_start:q_end] = orig_lse
        else:
            # Apply attn_sink inline (original SWA-only behaviour).
            if attn_sink is not None:
                sink = attn_sink.view(1, -1).float().to(device)
                lse_adj = torch.logsumexp(
                    torch.stack([orig_lse, sink.expand_as(orig_lse)], dim=0),
                    dim=0,
                )
            else:
                lse_adj = orig_lse
            lse_adj = lse_adj.clone()
            lse_adj[lse_adj == float("-inf")] = float("inf")
            attn_w = torch.exp(scores - lse_adj.unsqueeze(-1))
            out_i = torch.einsum("qhk,kd->qhd", attn_w, kv.float())
            output[q_start:q_end] = out_i.to(q.dtype)

    if return_lse:
        return output, all_lse
    return output


def _sparse_attention_with_lse(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA attention returning both output and raw LSE.

    Pure-torch reimplementation of ``ref_ds4_flash_mla_sparse_fwd`` that
    returns ``(output, lse)`` **without** applying ``attn_sink``.  The
    caller is expected to combine partial LSEs and apply the sink globally.

    Args:
        q:            ``[s_q, h_q, d]`` bf16/fp16.
        kv:           ``[s_kv, 1, d]`` same dtype as *q*.
        indices:      ``[s_q, 1, topk]`` int32; ``-1`` or ``>=s_kv`` is invalid.
        sm_scale:     softmax scale.
        topk_length:  ``[s_q]`` int32 optional; per-token k upper bound.

    Returns:
        ``(output, lse)`` where *output* is ``[s_q, h_q, d]`` (same dtype as
        *q*) and *lse* is ``[s_q, h_q]`` fp32.
    """
    s_q, h_q, d = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    device = q.device

    idx = indices.clone().squeeze(1)  # [s_q, topk]
    if topk_length is not None:
        col = torch.arange(topk, device=device).unsqueeze(0).expand(s_q, topk)
        oob = col >= topk_length.to(device).unsqueeze(1)
        idx = idx.clone()
        idx[oob] = -1

    invalid = (idx < 0) | (idx >= s_kv)
    idx_safe = idx.clone()
    idx_safe[invalid] = 0

    qf = q.float()
    kv_flat = kv.view(s_kv, d).float()
    gathered = kv_flat.index_select(0, idx_safe.flatten().to(torch.long)).view(
        s_q, topk, d
    )

    P = torch.einsum("shd,skd->shk", qf, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]

    lse_for_w = orig_lse.clone()
    lse_for_w[lse_for_w == float("-inf")] = float("inf")

    weights = torch.exp(P - lse_for_w.unsqueeze(-1))
    out_fp32 = torch.einsum("shk,skd->shd", weights, gathered)

    output = out_fp32.to(q.dtype)
    return output, orig_lse


def _merge_attention_outputs(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge two partial attention results via online softmax (log₂ LSE).

    Given ``(O1, lse1)`` from one KV segment and ``(O2, lse2)`` from
    another, produces the correct combined output as if the two KV
    segments had been attended to jointly.

    All LSE values are in **log₂ base** (hardware-native from V3).

    Optionally applies ``attn_sink`` (per-head extra softmax denominator
    logit, in natural log) after the merge.

    Args:
        o1:        ``[s_q, h_q, d]`` first partial output (any dtype).
        lse1:      ``[s_q, h_q]`` fp32 log₂-sum-exp of first segment.
        o2:        ``[s_q, h_q, d]`` second partial output.
        lse2:      ``[s_q, h_q]`` fp32 log₂-sum-exp of second segment.
        attn_sink: ``[h_q]`` fp32 optional sink logit (natural log).

    Returns:
        Combined output ``[s_q, h_q, d]``, same dtype as *o1*.
    """
    # log₂-based logsumexp: log2sumexp(a, b) = max(a,b) + log2(2^(a-max) + 2^(b-max))
    max_lse = torch.maximum(lse1, lse2)
    lse_combined = max_lse + torch.log2(
        torch.exp2(lse1 - max_lse) + torch.exp2(lse2 - max_lse)
    )  # [s_q, h_q]

    alpha1 = torch.exp2(lse1 - lse_combined).unsqueeze(-1)  # [s_q, h_q, 1]
    alpha2 = torch.exp2(lse2 - lse_combined).unsqueeze(-1)
    o_combined = alpha1 * o1.float() + alpha2 * o2.float()

    if attn_sink is not None:
        device = o1.device
        # Convert attn_sink from natural log to log₂
        _LOG2E = 1.4426950408889634
        sink_log2 = (attn_sink.view(1, -1).float().to(device) * _LOG2E).expand_as(
            lse_combined
        )
        max_final = torch.maximum(lse_combined, sink_log2)
        lse_final = max_final + torch.log2(
            torch.exp2(lse_combined - max_final) + torch.exp2(sink_log2 - max_final)
        )
        sink_scale = torch.exp2(lse_combined - lse_final).unsqueeze(-1)
        o_combined = sink_scale * o_combined

    return o_combined.to(o1.dtype)


def _dense_compressed_attention_torch(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    compress_ratio: int,
    c_block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense causal attention over all compressed KV tokens, returning LSE.

    For C128A layers every compressed token participates (no sparse
    top-k selection).  The causal constraint is that a query at absolute
    position ``pos`` can attend to compressed tokens ``0 .. (pos+1)//R - 1``
    where ``R = compress_ratio``.

    Args:
        q:                  ``[num_tokens, num_heads, head_dim]``.
        compressed_k_cache: Paged cache ``[num_blocks, c_block_size, head_dim]``.
        block_table:        ``[num_prefills, max_blocks]`` for compressed cache.
        seq_lens:           Total (uncompressed) seq lengths ``[num_prefills]`` (GPU).
        query_start_loc_cpu: Cumulative query counts ``[num_prefills + 1]`` (CPU,
            relative to prefill start, first element is 0).
        compress_ratio:     Compression factor (128).
        c_block_size:       ``block_size // compress_ratio`` (e.g. 2).
        scale:              Attention scale factor.

    Returns:
        ``(output, lse)`` where *output* is ``[num_tokens, num_heads, head_dim]``
        (same dtype as *q*) and *lse* is ``[num_tokens, num_heads]`` fp32.
        Tokens with no visible compressed KV get ``lse = -inf`` and ``output = 0``.
    """
    num_prefills = seq_lens.shape[0]
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    device = q.device

    output = torch.zeros_like(q)
    all_lse = torch.full(
        (num_tokens, num_heads),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    seq_lens_cpu = seq_lens.cpu()
    for i in range(num_prefills):
        s_len = seq_lens_cpu[i].item()
        c_seq_len = s_len // compress_ratio
        q_start = query_start_loc_cpu[i].item()
        q_end = query_start_loc_cpu[i + 1].item()
        q_len = q_end - q_start
        if q_len == 0 or c_seq_len == 0:
            continue

        q_i = q[q_start:q_end]  # [q_len, num_heads, head_dim]

        # Gather ALL compressed tokens from paged cache.
        c_positions = torch.arange(c_seq_len, device=device)
        c_blk_ids = block_table[i, c_positions // c_block_size]
        c_offsets = c_positions % c_block_size
        c_kv = compressed_k_cache[c_blk_ids, c_offsets]  # [c_seq_len, head_dim]

        # Causal mask: q at abs pos `pos` can see compressed kv 0..(pos+1)//R-1.
        q_abs = torch.arange(s_len - q_len, s_len, device=device)
        n_visible = (q_abs + 1) // compress_ratio  # [q_len]
        k_idx = torch.arange(c_seq_len, device=device)  # [c_seq_len]
        mask = k_idx.unsqueeze(0) < n_visible.unsqueeze(1)  # [q_len, c_seq_len]

        # Dense attention in fp32.
        scores = torch.einsum("qhd,kd->qhk", q_i.float(), c_kv.float()) * scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        lse_i = torch.logsumexp(scores, dim=-1)  # [q_len, num_heads]
        lse_for_w = lse_i.clone()
        lse_for_w[lse_for_w == float("-inf")] = float("inf")
        attn_w = torch.exp(scores - lse_for_w.unsqueeze(-1))
        out_i = torch.einsum("qhk,kd->qhd", attn_w, c_kv.float())

        output[q_start:q_end] = out_i.to(q.dtype)
        all_lse[q_start:q_end] = lse_i

    return output, all_lse


class IluDeepseekCompressor(DeepseekCompressor):
    """BF16 ilu fork of :class:`DeepseekCompressor`.

    Same parameters / state cache layout as upstream (so checkpoint loading,
    the ``ape`` / ``fused_wkv_wgate`` / ``norm`` weights and the paged fp32
    ``state_cache`` are inherited verbatim). Only :meth:`forward` is
    overridden to replace the two upstream Triton kernels with their
    ``ixfops`` equivalents:

    - ``_save_partial_states_kernel`` →
      ``ixfops.ds4_compressor_save_partial_states``
    - ``_fused_kv_compress_norm_rope_insert_*`` →
      ``ixfops.ds4_compressor_pool_norm_rope_insert``

    The rationale is twofold:

    1. The upstream Triton kernels pass ``launch_pdl=False`` (an SM90+
       programmatic-dependent-launch hint) which Iluvatar Triton rejects with
       ``KeyError: 'Keyword argument launch_pdl was specified but unrecognised'``.·
    2. The upstream second-step kernel writes the K cache in fp8_ds_mla
       packed layout (448B NoPE + 128B RoPE + 8B fp8 scale per token); the
       ilu fork stores raw bf16 paged ``[num_blocks, kv_block_size, head_dim]``
       and so collapses the three NV-side variants
       (``sparse_attn`` / ``indexer_attn`` / ``indexer_mxfp4_attn``) into a
       single ``ds4_compressor_pool_norm_rope_insert`` call.

    The op also takes care of the early-return semantics (PAD ``slot_mapping``,
    non-window-end positions, PAD ``kv_slot_mapping``) so the call site stays
    metadata-driven and CUDA-graph friendly.
    """

    def forward(
        self,
        x: torch.Tensor,  # [num_tokens, hidden_size]
        positions: torch.Tensor,  # [num_tokens]
        rotary_emb,
    ) -> None:
        # Step 1 — bf16 -> fp32 fused KV/score GEMM. Identical to upstream;
        # both ixformer ops below expect fp32 ``kv`` / ``score`` and an
        # fp32 paged state cache.
        kv_score = ixfops.gemm_bf16_bf16_fp32(
            x, self.fused_wkv_wgate.weight, format="TN"
        )
        # ``.split`` along the last dim yields non-contiguous views
        # (stride=(2W, 1), score with storage_offset=W). The ixformer op
        # ``ds4_compressor_save_partial_states`` accepts any row-contiguous
        # layout (stride(-1) == 1) so we hand the views straight through
        # without an extra .contiguous() copy.
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profiling / dummy run — no metadata, nothing to insert.
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # ``state_cache`` shape: [num_blocks, block_size, 2 * coff * head_dim]
        # (kv_state | score_state). fp32 by construction (see
        # ``CompressorStateCache``).
        state_cache = self.state_cache.kv_cache

        # Step 2 — write fp32 partial states (kv | score+ape) into the paged
        # state cache. PAD tokens (slot_mapping[t] < 0) are skipped inside the
        # op.
        ixfops.ds4_compressor_save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            slot_mapping=slot_mapping,
            state_cache=state_cache,
            block_size=block_size,
            compress_ratio=self.compress_ratio,
        )

        # Step 3 — pool window + RMSNorm + GPT-J RoPE + dense bf16/fp16 K
        # cache insert. Only window-right-end tokens
        # (``(pos + 1) % compress_ratio == 0``) actually write; the op
        # short-circuits the rest.
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache

        # ixformer accepts bf16/fp16 ``rms_norm_weight`` directly (T_W
        # templated, T2F-cast on load) and always-fp32 ``cos_sin_cache``.
        # The fp32 cos_sin contract matches DS-V4's RotaryEmbedding, which
        # registers the RoPE table as a single fp32 buffer shared across
        # all 60 layers (see DeepseekV4ScalingRotaryEmbedding "stored as
        # fp32 for higher precision RoPE"); down-casting it per-layer
        # would cost ~256MB temp alloc on the 1M-context path.
        ixfops.ds4_compressor_pool_norm_rope_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            rms_norm_weight=self.norm.weight,
            cos_sin_cache=cos_sin_cache,
            k_cache=kv_cache,
            kv_slot_mapping=k_cache_metadata.slot_mapping,
            compress_ratio=self.compress_ratio,
            rope_head_dim=self.rope_head_dim,
            rms_norm_eps=self.rms_norm_eps,
        )


@dataclass
class IluDeepseekV4MLAModules:
    """Modules used in DeepSeek-V4 MLA (ilu fork).

    Identical layout to the upstream ``DeepseekV4MLAModules`` so existing
    builders in the model code can pass a fully-populated instance through.
    """

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream: torch.cuda.Stream | None = None


@PluggableLayer.register("ilu_deepseek_v4_multi_head_latent_attention")
class IluDeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer for the ilu-fork W4A8 path."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: IluDeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config.
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepSeek-V4-specific attributes (num_heads is already TP-adjusted).
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules.
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a
        self.wo_b = mla_modules.wo_b

        # Hand the wo_a quant_method (W8A16 marlin) the geometry it needs at
        # process_weights_after_loading time. The flat ckpt weight
        # ``[n_local_groups * o_lora_rank, heads_per_group * head_dim]`` has to
        # be reshaped to a grouped 3D layout for marlin_w8_weight_repack; the
        # 4 numbers below are the only thing the W8A16 method can't recover
        # on its own (output_partition_sizes only carries the flat product).
        # Always-on (cheap) -- a BF16 wo_a path simply ignores the attribute.
        self.wo_a.bmm_geom = SimpleNamespace(
            n_local_groups=self.n_local_groups,
            o_lora_rank=self.o_lora_rank,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            head_dim=self.head_dim,
        )

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights).
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # head_bytes is forward-time data only; preserved so a stage-2 forward
        # can reuse the upstream bookkeeping when we wire bf16 kernels.
        self.head_bytes = self.head_dim * 2  # bf16 byte count

        self.aux_stream = mla_modules.aux_stream
        self.ln_events: list[torch.cuda.Event] = []  # populated at stage 2

        assert cache_config is not None, (
            "ilu DeepseekV4 attention requires cache_config"
        )
        # bf16 SWA cache (uint8 in upstream); ilu fork stage 1 always uses bf16.
        self.swa_cache_layer = IluDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = IluDeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=self.head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )

        compilation_config = mla_modules.vllm_config.compilation_config
        self.layer_name = prefix + ".ilu_deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # The compressor is only created on layers with compress_ratio > 1
        # (matching upstream). It owns its own bf16 nn.Parameters and must be
        # constructed even at stage 1 so load_weights can populate them.
        # ``IluDeepseekCompressor`` shares all parameters / state cache layout
        # with the upstream class but routes the two Triton kernels through
        # ``ixfops`` (see class docstring).
        self.compressor: DeepseekCompressor | None = None
        if self.compress_ratio > 1:
            self.compressor = IluDeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse MLA attention forward (BF16, ilu V4 W4A8).

        Drop-in bf16 replacement for the upstream V4 wrapper.forward
        (which uses ``fused_inv_rope_fp8_quant`` +
        ``vllm.deepseek_v4_fp8_einsum``). The bf16 path:

        1. ``qr_kv = fused_wqa_wkv(hidden_states)`` and split into
           ``qr`` (q_lora_rank) and ``kv`` (head_dim).
        2. Allocate ``o`` of shape ``(T, n_local_heads, head_dim)``;
           the sparse MLA op writes the attention result into this
           buffer in place via the custom op. (No head-count padding:
           ixformer's ``ds4_flash_mla_sparse_fwd`` /
           ``ds4_flash_mla_with_kvcache`` accept arbitrary ``h_q``.)
        3. ``torch.ops.vllm.ilu_deepseek_v4_attention(...)`` runs the
           full sparse MLA pipeline (norm, indexer, compressor,
           kv-insert, mla_attn) under the torch.compile boundary; the
           output is written into ``o``.
        4. ``ds4_inv_rope`` rotates the RoPE band of ``o`` in place
           (no FP8 quant; the bf16 result is consumed directly by the
           grouped einsum below).
        5. Reshape ``o`` to ``(T, n_local_groups, heads_per_group * head_dim)``
           and run the bf16 grouped GEMM ``z = einsum("bhr,hdr->bhd", o, w)``
           against ``wo_a.weight`` reshaped to
           ``(n_local_groups, o_lora_rank, heads_per_group * head_dim)``.
           This is the bf16 analogue of upstream's
           ``vllm.deepseek_v4_fp8_einsum`` (``is_bmm=True``,
           ``bmm_batch_size=n_local_groups``).
        6. ``self.wo_b(z.flatten(1))`` collapses the per-group lora
           outputs and runs the row-parallel projection back to
           ``hidden_size`` (with TP all-reduce inside ``RowParallelLinear``).
        """
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        num_tokens = hidden_states.shape[0]
        o = torch.empty(
            (num_tokens, self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.ilu_deepseek_v4_attention(
            hidden_states,
            qr,
            kv,
            positions,
            o,
            self.layer_name,
        )

        ixfops.ds4_inv_rope(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            inplace=True,
        )

        heads_per_group = self.n_local_heads // self.n_local_groups
        o_view = o.view(
            num_tokens, self.n_local_groups, heads_per_group * self.head_dim
        )
        if hasattr(self.wo_a, "W_repacked"):
            # v2 W8A16 marlin path (set by _IluV4W8A16BMMLinearMethod):
            # input stays bf16; weight is INT8 in Marlin layout +
            # per-(group,N) channel scale. The 3D batched signature mirrors
            # R1's _k_up_proj_w8a8 (vllm/v1/attention/backends/mla/ilu_mla.py).
            z = ixfops.marlin_w8a16(
                o_view,
                self.wo_a.W_repacked,
                self.wo_a.W_scale_repacked,
                group_size=-1,
                format=self.wo_a.W_format,
                batch_first=False,
            )
        else:
            # v1 BF16 path (and any future variant where wo_a is left
            # unquantised) -- keep the original grouped einsum verbatim.
            w_view = self.wo_a.weight.view(
                self.n_local_groups, self.o_lora_rank, heads_per_group * self.head_dim
            )
            z = torch.einsum("bhr,hdr->bhd", o_view, w_view)

        return self.wo_b(z.flatten(1))

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Sparse MLA attention impl (BF16, ilu V4 W4A8) — sequential.

        Drop-in replacement for the upstream V4 ``attention_impl`` with
        the multi-stream / overlap removed (per ilu fork policy: no
        ``maybe_execute_in_parallel`` aux-stream gymnastics — call
        ``indexer`` / ``compressor`` / ``_fused_qnorm_rope_kv_insert``
        sequentially on the default stream).

        Pipeline:
          1. ``qr = q_norm(qr)``, ``kv = kv_norm(kv)`` -- RMSNorm with
             learnable weight (bf16).
          2. ``q = wq_b(qr).view(-1, n_local_heads, head_dim)``.
          3. Layer-type dispatch:
               - C4A (``indexer is not None``):
                   * ``self.indexer(hidden_states, qr, positions,
                     indexer_rotary_emb)`` runs the indexer pipeline
                     end-to-end (it internally invokes its own
                     compressor over the **indexer** K cache).
                   * ``self.compressor(hidden_states, positions,
                     rotary_emb)`` writes the **main** MLA cache (the
                     wrapper's compressor; head_dim=512).
                   * ``self._fused_qnorm_rope_kv_insert(...)`` does the
                     fused Q RMSNorm + Q/KV RoPE + SWA cache insert.
               - C128A (``indexer is None`` but
                 ``self.compressor is not None``): main compressor +
                 fused QNorm/RoPE/insert (SWA cache).
               - SWA-only (``compressor is None``): only the fused
                 QNorm/RoPE/insert step.
          4. Profiling / dummy run: pre-warm the prefill workspace at
             the size the real prefill will request, zero ``out``, and
             return early.
          5. Call ``self.mla_attn(q, kv, positions, output=out)`` which
             writes the attention result into ``out``. No head-count
             padding is needed — ixformer's sparse MLA kernels accept
             arbitrary ``h_q``.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # 1) RMSNorm for both qr and kv.  Upstream uses a fused kernel;
        #    bf16 path here just calls the two RMSNorm modules. q_norm /
        #    kv_norm own learnable weights so we can't fold them into a
        #    weightless variant.

        # 1. origin impl
        # qr = self.q_norm(qr)
        # kv = self.kv_norm(kv)

        # 2. fused impl
        qr, kv = ixfops.rms_norm_qk(
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.eps,
        )

        # 2) Q projection: (T, q_lora_rank) -> (T, n_local_heads, head_dim)
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # 3) Per-layer-type sequencing.
        if self.indexer is not None:
            # C4A layer: indexer (own compressor + topk) + main compressor
            #            (writes main MLA cache) + SWA insert.
            self.indexer(
                hidden_states,
                qr,
                positions,
                self.indexer_rotary_emb,
            )
            assert self.compressor is not None
            self.compressor(hidden_states, positions, self.rotary_emb)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        elif self.compressor is not None:
            # C128A layer: main compressor + SWA insert. No indexer.
            self.compressor(hidden_states, positions, self.rotary_emb)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        else:
            # SWA-only layer: just the fused Q/KV norm+rope+insert.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # 4) Dummy / profiling run: reserve prefill workspace at the
        #    real-execution shape, then return zeroed output. Without
        #    this, the workspace stays "locked" at a smaller size.
        if not isinstance(attn_metadata, dict):
            sub = self.mla_attn
            swa_only = sub.compress_ratio <= 1
            if swa_only:
                N = 0
            else:
                N = (sub.max_model_len + sub.compress_ratio - 1) // sub.compress_ratio
            M = N + sub.window_size + sub.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            out.zero_()
            return

        # 5) Run the sparse MLA forward into the pre-allocated ``out`` buffer
        #    (shape ``(T, n_local_heads, head_dim)``). No head-count padding:
        #    ixformer kernels (``ds4_flash_mla_sparse_fwd`` /
        #    ``ds4_flash_mla_with_kvcache``) accept arbitrary ``h_q``.
        self.mla_attn(q, kv, positions, output=out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata,
    ) -> None:
        """BF16 fused Q RMSNorm + Q/KV RoPE + paged KV insert (ilu V4).

        Drop-in replacement for the upstream V4 fused kernel
        ``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``,
        with the FP8 quantization step removed so the SWA cache stores raw
        bf16 values.

        - ``q``: ``(num_tokens, n_local_heads, head_dim)`` bf16, modified
          in place. Per-head RMSNorm (no weight, ``self.eps``) is applied
          first; then GPT-J RoPE (interleaved, ``is_neox_style=False``) on
          the trailing ``rope_head_dim`` (last 64) dims, indexed by
          ``positions``.
        - ``kv``: ``(num_tokens, head_dim)`` bf16, read-only. The kernel
          rotates the trailing ``rope_head_dim`` of each row using the
          same cos/sin cache and writes the resulting full ``head_dim``
          row into ``swa_kv_cache_2d`` at ``swa_metadata.slot_mapping``.
        - Tokens with ``slot_mapping[t] < 0`` are skipped on the KV side
          (matches the kernel's pad-handling) but still get Q processed.

        Returns ``None``; ``q`` and ``self.swa_cache_layer.kv_cache`` are
        the only mutated tensors. Dummy / profiling runs (no metadata)
        return early without invoking the kernel.
        """
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "ilu V4 attention_impl: missing SWA metadata for prefix "
            f"{self.swa_cache_layer.prefix!r}; check that the SWA "
            "KVCacheGroup is registered alongside the main MLA cache."
        )

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # ``ds4_fused_qnorm_rope_kv_insert`` accepts both the 3D
        # ``(num_blocks, block_size, head_dim)`` and the flattened 2D
        # ``(num_blocks, block_size * head_dim)`` layouts; the upstream
        # vLLM op uses the flattened form, so we mirror it here for parity.
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        ixfops.ds4_fused_qnorm_rope_kv_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )


class IluDeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    """Sparse MLA attention layer for ilu fork (BF16 KV, W4A8 weights)."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: IluDeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix

        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = []

        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        assert swa_cache_layer is not None
        self.swa_cache_layer: IluDeepseekV4SWACache = swa_cache_layer

        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # ilu fork: bf16 KV cache by default. Reject explicit fp8 selection so
        # operators get a loud error instead of silently routing into the V4
        # fp8_ds_mla layout.
        kv_cache_dtype = (
            cache_config.cache_dtype
            if cache_config is not None and cache_config.cache_dtype not in (None, "")
            else "bfloat16"
        )
        if kv_cache_dtype == "auto":
            kv_cache_dtype = "bfloat16"
        if kv_cache_dtype.startswith("fp8"):
            raise ValueError(
                "IluDeepseekV4MLAAttention does not support fp8 KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}. Use "
                "--kv-cache-dtype bfloat16."
            )
        self.kv_cache_dtype = kv_cache_dtype
        if cache_config is not None:
            cache_config.cache_dtype = kv_cache_dtype

        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # SWA-only layers don't allocate their own cache (the SWA cache is a
        # separate KVCacheSpec owned by IluDeepseekV4SWACache).
        if self.compress_ratio <= 1:
            return None
        # ``model_version`` is left at None: with ``cache_dtype_str="bfloat16"``
        # MLAAttentionSpec.real_page_size_bytes already falls through to the
        # default ``storage_block_size * head_size * dtype_size`` formula,
        # which is what we want for the single-latent bf16 store. Mirroring
        # the SWA spec keeps both layers in the same KVCacheGroup at
        # ``page_size_bytes == 65536`` (storage_block_size=64 / head_size=512
        # for C4A), so they continue sharing one physical tensor.
        #
        # NOTE: no ``alignment``. Same rationale as
        # ``IluDeepseekV4SWACache.get_kv_cache_spec``: the ilu MLA path
        # dispatches through ixformer kernels (e.g. ``ds4_flash_mla_with_kvcache``
        # / ``ds4_fused_qnorm_rope_kv_insert``) which require a contiguous
        # paged layout. ``alignment=576`` would set ``page_size_padded`` and
        # take ``_reshape_kv_cache_tensors`` down the ``as_strided`` branch,
        # producing a non-contiguous tensor that the kernels' strict
        # ``is_contiguous()`` checks reject. The natural ``storage_block_size *
        # head_size * dtype_size`` is what every ilu kernel reads/writes
        # directly.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            cache_dtype_str="bfloat16",
            model_version=None,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse MLA forward (BF16, ilu V4 W4A8).

        Splits the batch into ``[decode | prefill]`` (the scheduler
        reorders this way) using ``swa_metadata.num_decode_tokens`` and
        delegates to :meth:`_forward_prefill` / :meth:`_forward_decode`.
        Both branches write into pre-sliced views of ``output`` so the
        caller's contiguous buffer is filled in place.

        ``kv`` itself is not consumed here -- the wrapper's
        ``_fused_qnorm_rope_kv_insert`` already wrote the rotated K rows
        into the SWA cache, and the sparse kernels read from
        ``self.swa_cache_layer.kv_cache`` plus (for ``compress_ratio >
        1``) the layer's compressed K cache. We accept ``kv`` only to
        match the upstream V4 attention signature, keeping the call
        site in :meth:`IluDeepseekV4MultiHeadLatentAttentionWrapper
        .attention_impl` symmetric with the upstream layout.
        """
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict), (
            "ilu V4 MLA forward requires a dict-shaped attn_metadata; got "
            f"{type(attn_metadata).__name__}. Profiling/dummy runs should "
            "be short-circuited by the wrapper before reaching here."
        )

        flashmla_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "Missing SWA metadata for prefix "
            f"{self.swa_cache_layer.prefix!r}; the SWA KVCacheGroup must "
            "be registered alongside the main MLA cache."
        )

        swa_only = self.compress_ratio <= 1
        # SWA-only layers don't allocate their own KV cache; ``self.kv_cache``
        # may be empty after profiling cleanup, so guard explicitly.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            prefill_q = q[num_decode_tokens:]
            prefill_out = output[num_decode_tokens:]

            if envs.VLLM_ILU_DS4_PREFILL_OPT:
                self._forward_prefill_opt(
                    q=prefill_q,
                    swa_k_cache=swa_kv_cache,
                    output=prefill_out,
                    swa_metadata=swa_metadata,
                    compressed_k_cache=self_kv_cache,
                    attn_metadata=flashmla_metadata if not swa_only else None,
                )
            else:
                self._forward_prefill(
                    q=prefill_q,
                    positions=positions[num_decode_tokens:],
                    compressed_k_cache=self_kv_cache,
                    swa_k_cache=swa_kv_cache,
                    output=prefill_out,
                    attn_metadata=flashmla_metadata if not swa_only else None,
                    swa_metadata=swa_metadata,
                )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata if not swa_only else None,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekSparseSWAMetadata,
        attn_metadata: IluFlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Decode-side sparse MLA attention (BF16, ilu V4 W4A8).

        Three-branch topk sourcing matches the upstream V4 layout:

        - ``swa_only`` (``compress_ratio <= 1``): no main-MLA cache, only
          the SWA topk window. ``topk_indices`` / ``topk_lens`` stay
          ``None`` so :func:`ds4_flash_mla_with_kvcache` runs the SWA-only
          single-path code path.
        - ``compress_ratio == 4`` (C4A): the indexer wrote per-token *local*
          (per-request) compressed-domain indices into
          ``self.topk_indices_buffer``. We map them to global compressed
          paged slots via ``ixfops.ds4_compute_global_topk_indices_and_lens``
          (the ixformer CUDA op that replaces the upstream Triton
          ``compute_global_topk_indices_and_lens``), using
          ``block_size = attn_metadata.block_size // compress_ratio``
          (256/4 = 64 for V4).
        - ``compress_ratio == 128`` (C128A): the metadata builder
          pre-computed ``c128a_global_decode_topk_indices`` /
          ``c128a_decode_topk_lens`` once per build, shared across all
          C128A layers.

        ``ds4_flash_mla_with_kvcache`` replaces the upstream FP8 FlashMLA
        op: it skips the tile_scheduler_metadata and the ``is_fp8_kvcache``
        flag because the kv cache is bf16 here. ixformer's kernel accepts
        arbitrary ``h_q``, so no head-count padding is required.
        """
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices: torch.Tensor | None = None
        topk_lens: torch.Tensor | None = None
        if not swa_only:
            assert attn_metadata is not None, (
                "FlashMLASparseMetadata is required when compress_ratio > 1; "
                f"prefix={self.prefix!r}"
            )
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: per-layer local topk indices written by the indexer.
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = (
                    ixfops.ds4_compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices[:num_decode_tokens],
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            elif self.compress_ratio == 128:
                # C128A: pre-computed once at metadata build time.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # Decode treats every query token as its own batch element so the
        # sparse MLA kernel can attend over per-token topk slices.
        q = q.unsqueeze(1)

        # ``ds4_flash_mla_with_kvcache`` needs the cache reshaped to
        # ``(num_blocks, page_block_size, 1, head_dim)``; ``unsqueeze(-2)``
        # preserves strides (handles padded blocks correctly). The kernel
        # accepts different ``page_block_size`` between ``k_cache`` (SWA, 64)
        # and ``extra_k_cache`` (C4A storage_block=64 / C128A storage_block=2)
        # — each cache flattens independently to ``[num_slots, d]`` and the
        # global slot id in ``extra_indices`` indexes the extra cache only.
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        ixfops.ds4_flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            indices=swa_indices,
            topk_length=swa_lens,
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices=topk_indices,
            extra_topk_length=topk_lens,
            output=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        """Prefill-side sparse MLA attention (BF16, ilu V4 W4A8).

        BF16 port of upstream V4's ``_forward_prefill``. The structural
        skeleton (chunk-by-``PREFILL_CHUNK_SIZE``, gather compressed +
        SWA segments into a shared bf16 workspace, combine topk indices,
        then run sparse MLA prefill) is identical; the substitutions are:

        - ``dequantize_and_gather_k_cache`` → :func:`ds4_gather_k_cache`
          (bf16 paged gather, no fp8 dequant).
        - ``combine_topk_swa_indices`` →
          :func:`ds4_combine_topk_swa_indices` (matches upstream layout
          and pads to ``ceil((topk + window) / 128) * 128`` columns).
        - ``flash_mla_sparse_fwd`` → :func:`ds4_flash_mla_sparse_fwd`.

        ``attn_metadata`` may be ``None`` for SWA-only layers (no main
        MLA cache); the loop then gathers only the SWA segment and
        passes ``compress_ratio=1, topk=0, N=0`` to the combiner so the
        layout collapses to pure SWA.
        """
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                # C4A: per-layer, per-token local indices written by the
                # indexer into the shared topk buffer; slice off the
                # decode prefix and clamp to actual prefill tokens.
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            elif self.compress_ratio == 128:
                # C128A: pre-computed by metadata builder.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
                assert topk_indices is not None
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )
            top_k = topk_indices.shape[-1]
            # Compressed segment must hold the full compressed pool, not just
            # ``top_k`` (which only bounds index count, not the source range).
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # ``topk_indices`` is unused for SWA-only layers (``top_k = 0``
            # tells the kernel to skip its content). We still need a 2D
            # ``[num_tokens, *]`` int32 buffer for the
            # ``ds4_combine_topk_swa_indices`` shape check; reuse the full
            # ``topk_indices_buffer[num_decode_tokens:]`` slice (dim-0 only,
            # so it stays contiguous). Slicing ``[:, :1]`` here would yield
            # strides ``(K, 1)`` on shape ``(N, 1)``, which fails the
            # kernel's ``is_contiguous()`` check when ``K > 1``.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        # Workspace row stride per batch: compressed pool + SWA window +
        # max query batch size (bounds the in-batch q rows fed to flash
        # mla sparse_fwd).
        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        (kv,) = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start

            # ------------------------------------------------------------
            # 1) Compressed K segment (only when this layer has its own
            #    main MLA cache; SWA-only layers skip).
            # ------------------------------------------------------------
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                ixfops.ds4_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # ------------------------------------------------------------
            # 2) SWA segment (always gathered; lives at offset ``N`` after
            #    the compressed pool inside the workspace row).
            # ------------------------------------------------------------
            swa_block_table = swa_metadata.block_table[num_decodes:]
            ixfops.ds4_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # ------------------------------------------------------------
            # 3) Combine local topk + SWA window into a single index
            #    list, in workspace coords (rows of stride ``M``).
            # ------------------------------------------------------------
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = ixfops.ds4_combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio if not swa_only else 1,
                top_k,
                M,
                N,
            )

            # ------------------------------------------------------------
            # 4) Sparse MLA prefill: read from the workspace KV view and
            #    write into the per-token output slice.
            # ------------------------------------------------------------
            ixfops.ds4_flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                output=output[query_start:query_end],
            )

    def _forward_prefill_opt(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        compressed_k_cache: torch.Tensor | None = None,
        attn_metadata: IluFlashMLASparseMetadata | None = None,
    ) -> None:
        """Optimised prefill dispatcher. Writes directly into ``output``.

        Routes to one of three sub-functions based on ``compress_ratio``:
        - SWA-only (ratio <= 1): sliding-window attention with attn_sink.
        - C4A (ratio == 4): split sparse-compressed + SWA, merge via LSE.
        - C128A (ratio == 128): split dense-compressed + SWA, merge via LSE.
        """
        if self.compress_ratio <= 1:
            self._prefill_swa_only(q, swa_k_cache, output, swa_metadata)
        elif self.compress_ratio == 4:
            assert compressed_k_cache is not None
            assert attn_metadata is not None
            self._prefill_c4a(
                q,
                swa_k_cache,
                output,
                swa_metadata,
                compressed_k_cache,
                attn_metadata,
            )
        elif self.compress_ratio == 128:
            assert compressed_k_cache is not None
            assert attn_metadata is not None
            self._prefill_c128a(
                q,
                swa_k_cache,
                output,
                swa_metadata,
                compressed_k_cache,
                attn_metadata,
            )
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio} "
                "in _forward_prefill_opt; expected 1, 4, or 128."
            )

    # ------------------------------------------------------------------
    # Sub-function: SWA-only (compress_ratio <= 1)
    # ------------------------------------------------------------------
    def _prefill_swa_only(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        """SWA-only sliding-window attention (with attn_sink) into ``output``."""
        ixfops.ds4_swa_attn_fwd(
            q=q,
            kv_cache=swa_k_cache,
            block_table=swa_metadata.prefill_swa_block_table,
            cu_seqlens_q=swa_metadata.prefill_cu_seqlens_q,
            cu_seqlens_k=swa_metadata.prefill_cu_seqlens_k,
            max_seqlen_q=swa_metadata.prefill_max_seqlen_q,
            max_seqlen_k=swa_metadata.prefill_max_seqlen_k,
            window_size=self.window_size,
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            output=output,
        )

    # ------------------------------------------------------------------
    # Sub-function: C4A (compress_ratio == 4)
    # ------------------------------------------------------------------
    def _prefill_c4a(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        compressed_k_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> None:
        """C4A: paged sparse-compressed + SWA attention, merged via LSE."""
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        cu_seqlens_q = swa_metadata.prefill_cu_seqlens_q
        cu_seqlens_k = swa_metadata.prefill_cu_seqlens_k
        prefill_seq_lens = swa_metadata.prefill_seq_lens
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        assert prefill_seq_lens is not None

        # SWA part (no attn_sink -- will be applied in merge)
        o_swa, lse_swa = ixfops.ds4_swa_attn_fwd(
            q=q,
            kv_cache=swa_k_cache,
            block_table=swa_metadata.prefill_swa_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=swa_metadata.prefill_max_seqlen_q,
            max_seqlen_k=swa_metadata.prefill_max_seqlen_k,
            window_size=self.window_size,
            sm_scale=self.scale,
            attn_sink=None,
        )

        # Compressed sparse attention -- single call, reads directly from paged cache
        c_block_table = attn_metadata.block_table[num_decodes:]
        c_block_size = attn_metadata.block_size // self.compress_ratio

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + num_prefill_tokens
        ]

        o_c4, lse_c4 = ixfops.ds4_paged_sparse_attn_fwd(
            q=q,
            kv_cache=compressed_k_cache,
            block_table=c_block_table,
            topk_indices=topk_indices,
            cu_seqlens_q=cu_seqlens_q,
            seq_lens=prefill_seq_lens,
            compress_ratio=self.compress_ratio,
            block_size=c_block_size,
            sm_scale=self.scale,
            return_lse=True,
        )

        # Merge compressed + SWA via fused kernel
        ixfops.ds4_merge_attn_outputs(
            output,
            o_c4,
            lse_c4,
            o_swa,
            lse_swa,
            attn_sink=self.attn_sink,
        )

    # ------------------------------------------------------------------
    # Sub-function: C128A (compress_ratio == 128)
    # ------------------------------------------------------------------
    def _prefill_c128a(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        swa_metadata: DeepseekSparseSWAMetadata,
        compressed_k_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> None:
        """C128A: dense-compressed + SWA attention, merged via LSE."""
        num_decodes = swa_metadata.num_decodes
        cu_seqlens_q = swa_metadata.prefill_cu_seqlens_q
        cu_seqlens_k = swa_metadata.prefill_cu_seqlens_k
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None

        # SWA part (no attn_sink -- will be applied in merge)
        o_swa, lse_swa = ixfops.ds4_swa_attn_fwd(
            q=q,
            kv_cache=swa_k_cache,
            block_table=swa_metadata.prefill_swa_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=swa_metadata.prefill_max_seqlen_q,
            max_seqlen_k=swa_metadata.prefill_max_seqlen_k,
            window_size=self.window_size,
            sm_scale=self.scale,
            attn_sink=None,
        )

        # Dense compressed attention
        c_block_table = attn_metadata.block_table[num_decodes:]
        o_compressed, lse_compressed = ixfops.ds4_dense_compressed_attn_fwd(
            q=q,
            kv_cache=compressed_k_cache,
            block_table=c_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            compress_ratio=self.compress_ratio,
            sm_scale=self.scale,
        )

        # Merge compressed + SWA via fused kernel
        ixfops.ds4_merge_attn_outputs(
            output,
            o_compressed,
            lse_compressed,
            o_swa,
            lse_swa,
            attn_sink=self.attn_sink,
        )


class IluDeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    """BF16 indexer K cache for ilu fork.

    Co-located with :class:`IluDeepseekV4Indexer` (matches upstream's choice
    of putting :class:`DeepseekV4IndexerCache` next to the ``Indexer`` module
    rather than in the backends directory).
    """

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # NOTE: Use ``cache_config.block_size`` (default 256) so the indexer
        # cache lives in the same ``KVCacheGroup`` as the main MLA cache,
        # matching the NV upstream allocator path
        # (``_get_kv_cache_config_deepseek_v4``). Both backends now advertise
        # ``[256]`` from ``get_supported_kernel_block_sizes()``, so the group
        # negotiates a uniform ``kernel_block_size = 256`` and
        # ``_reshape_kv_cache_tensors`` produces a contiguous
        # ``[num_blocks, storage_block_size, head_dim]`` view with no virtual
        # block splitting.
        #
        # The CUDA ``dsa_indexer_mqa_logits_with_blocks`` kernel itself only
        # supports block_size in {16, 32, 64}; with block_size=256 the Python
        # wrapper falls back to ``ref_dsa_indexer_mqa_logits_with_blocks``
        # automatically (see ixformer indexer.py). The fallback is slower but
        # functionally equivalent.
        #
        # No alignment padding: the indexer kernel reads/writes raw
        # ``[num_blocks, storage_block_size, head_dim]`` and does not require
        # the 576B FlashMLA-page alignment used by NV's main MLA cache.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV4IndexerBackend


class IluDeepseekV4Indexer(nn.Module):
    """Lightning indexer for ilu fork.

    Stage 1: builds ``wq_b`` / ``weights_proj`` / ``compressor`` /
    ``k_cache`` (everything that owns parameters reachable from
    ``load_weights``) but skips the forward-only :class:`SparseAttnIndexer`
    op construction; ``forward`` raises.

    Note: upstream's :class:`DeepseekV4Indexer` also constructs a
    ``self.k_norm = LayerNorm(head_dim)`` but never references it in
    ``forward``. The V4-Flash checkpoint has no corresponding tensor; we
    omit it here so the strict missing-param check stays clean.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio

        # Always FP8 path off in ilu fork; bf16 indexer K cache.
        self.use_fp4_kv = False
        logger.info_once(
            "Ilu DeepSeek V4 indexer using BF16 cache (fp4_indexer disabled)."
        )

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        # NOTE: upstream ``DeepseekV4Indexer`` constructs a ``self.k_norm =
        # LayerNorm(head_dim)`` here, but it is dead code -- the upstream
        # ``forward`` never calls it, and the V4-Flash checkpoint contains no
        # ``indexer.k_norm.{weight,bias}`` tensors. It only stays alive on the
        # upstream FP8 path because ``DefaultModelLoader`` skips the strict
        # missing-param check when ``model_config.quantization != None``,
        # silently accepting LayerNorm's default-init values. Our strict
        # check would (correctly) flag those 21*2=42 tensors as missing, so
        # we deliberately omit ``self.k_norm`` here. Leftover from porting
        # V3.2's indexer to V4.
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"  # only relevant once SparseAttnIndexer is wired
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Ilu DeepSeek V4 indexer requires cache_config"
        # No fp8 scale padding -- bf16 cache stores raw head_dim per token.
        self.k_cache = IluDeepseekV4IndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = IluDeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        # Stage 2: we do *not* register a SparseAttnIndexer custom op here.
        # The indexer's forward inlines the bf16 mqa-logits + topk kernels
        # because the wrapper's outer forward is itself a custom op
        # boundary, so there is no need to wrap a second custom op around
        # ixformer's stateless calls. The K-cache insertion is also
        # offloaded to ``self.compressor`` (which already writes the bf16
        # indexer cache), so the legacy ``ops.indexer_k_cache`` step from
        # ``IluSparseAttnIndexer`` is intentionally dropped.

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor | None:
        """Lightning indexer forward (bf16, ilu V4 W4A8).

        Mirrors the upstream ``DeepseekV4Indexer.forward`` skeleton but
        replaces the fp8 quant path with bf16 ixformer ops:

        1. ``q = wq_b(qr).view(-1, n_head, head_dim)`` — bf16.
        2. ``self.compressor(hidden_states, positions, rotary_emb)`` —
           compresses K, applies RMSNorm + RoPE, and writes the result
           into ``self.k_cache.kv_cache`` at the **compressed**
           ``slot_mapping`` (taken from ``attn_metadata[k_cache.prefix]``).
        3. ``weights, _ = weights_proj(hidden_states)`` — bf16
           ``[num_tokens, n_head]`` projection used as the per-token,
           per-head MQA weighting.
        4. ``ds4_fused_indexer_q_rope`` — in-place GPT-J RoPE on the last
           ``rope_dim`` of ``q`` plus the fused ``weights *
           softmax_scale * (n_head ** -0.5)`` written directly in
           ``q``'s dtype (kernel scales in fp32 and F2T-casts on store),
           matching the ``dsa_indexer_mqa_logits_with_blocks`` dtype
           contract for ``weights``.
        5. Per-prefill-chunk and (single) decode batch:
           ``dsa_indexer_mqa_logits_with_blocks`` (logits) +
           ``dsa_update_topk_indices`` (writes into
           ``self.topk_indices_buffer``).

        Returns ``self.topk_indices_buffer`` (mutated in place).  Dummy
        / profiling runs (when ``attn_metadata`` is not a dict) skip the
        kernel work and return the buffer untouched, matching the
        upstream V3.2 ilu pattern.
        """
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        # Compressor returns ``None`` for the V4 path (it just writes the
        # bf16 indexer K-cache).  Capturing the return value would shadow
        # the ``k`` symbol in upstream V4's forward; here we just call it.
        self.compressor(hidden_states, positions, rotary_emb)

        weights_in, _ = self.weights_proj(hidden_states)

        # ``ds4_fused_indexer_q_rope`` writes ``weights_out`` in q.dtype
        # directly (scaled in fp32 internally, F2T-cast on store), matching
        # the downstream ``dsa_indexer_mqa_logits_with_blocks`` dtype
        # contract -- no fp32 -> bf16 cast needed.
        weights = torch.empty(
            (q.shape[0], self.n_head),
            dtype=q.dtype,
            device=q.device,
        )
        ixfops.ds4_fused_indexer_q_rope(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights_in,
            weights,
            self.softmax_scale,
            self.n_head**-0.5,
            self.rope_dim,
            inplace_q=False,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profiling / dummy run: caller (wrapper.forward) handles the
            # workspace warm-up; the indexer just returns the (zero-init)
            # topk buffer.
            return self.topk_indices_buffer

        md = attn_metadata.get(self.k_cache.prefix)
        if md is None:
            return self.topk_indices_buffer
        assert isinstance(md, IluDeepseekV32IndexerMetadata)

        kv_cache = self.k_cache.kv_cache
        topk_buffer = self.topk_indices_buffer

        if md.num_prefills > 0:
            assert md.prefill is not None
            for chunk in md.prefill.chunks:
                if chunk.max_context_len == 0:
                    # CUDA graph warmup can produce all-padding chunks with no
                    # compressed KV entries. The indexer is a no-op in that case.
                    continue
                # bf16 native path: same signature as the v1 (fp32-logits /
                # cuinfer-backed) op, but logits are produced in bf16
                # directly.  ``dsa_update_topk_indices`` already accepts bf16
                # logits, so no extra cast is needed downstream.  The wrapper
                # over-allocates the row stride to an even number (required
                # by the bf16 vector loads in both this op's store path and
                # the downstream topk) and returns a strided view whose
                # ``size(1)`` matches the caller's ``max_context_len`` -- so
                # callers can pass the raw (possibly odd) token-count
                # metadata from ``seq_lens.max()`` as-is.
                logits = ixfops.dsa_indexer_mqa_logits_with_blocks_bf16(
                    q[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlens_q,
                    chunk.cu_seq_lens,
                    kv_cache,
                    chunk.block_table,
                    weights[chunk.token_start : chunk.token_end],
                    max_q_len=chunk.max_q_len,
                    max_kv_len=chunk.max_kv_len,
                    max_context_len=chunk.max_context_len,
                )
                ixfops.dsa_update_topk_indices(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    self.topk_tokens,
                    topk_buffer[chunk.token_start : chunk.token_end],
                )

        if md.num_decodes > 0:
            decode_md = md.decode
            assert decode_md is not None
            if decode_md.requires_padding:
                # ``requires_padding`` now flags only *genuinely ragged* decode
                # batches (real requests with differing query lengths, e.g.
                # variable accepted spec tokens). Uniform decode / uniform MTP
                # with CUDA-graph trailing zero-length padding rows are handled
                # natively (the metadata builder zero-fills padding spans and
                # the kernels no-op them), so they do NOT set this flag. Ragged
                # batches would need a pack/unpack path; surface loudly instead
                # of silently degrading.
                raise NotImplementedError(
                    "ilu DeepSeek-V4 indexer does not support ragged "
                    "(non-uniform query length) decode batches"
                )
            num_decode_tokens = md.num_decode_tokens
            if decode_md.max_context_len == 0:
                # Early decode steps can have no compressed C4A entries yet.
                # In that case there are no top-k candidates, and ixformer
                # cannot materialize a bf16 logits matrix with zero columns.
                topk_buffer[:num_decode_tokens].fill_(-1)
                return topk_buffer
            logits = ixfops.dsa_indexer_mqa_logits_with_blocks_bf16(
                q[:num_decode_tokens],
                decode_md.cu_seqlens_q,
                decode_md.cu_seqlens_kv,
                kv_cache,
                decode_md.block_table,
                weights[:num_decode_tokens],
                max_q_len=decode_md.max_q_len,
                max_kv_len=decode_md.max_kv_len,
                max_context_len=decode_md.max_context_len,
            )
            ixfops.dsa_update_topk_indices(
                logits,
                decode_md.cu_seqlen_ks,
                decode_md.cu_seqlen_ke,
                self.topk_tokens,
                topk_buffer[:num_decode_tokens],
            )

        return topk_buffer


# ---------------------------------------------------------------------------
# Custom op: vllm::ilu_deepseek_v4_attention
# ---------------------------------------------------------------------------
# This is the ilu-fork analogue of upstream's ``vllm::deepseek_v4_attention``.
# We register a *new* op rather than re-registering the same name so the
# fp8 path in ``deepseek_v4_attention.py`` keeps its registration intact and
# the dispatcher can dispatch to whichever module is loaded.
#
# The op is a thin wrapper around
# :meth:`IluDeepseekV4MultiHeadLatentAttentionWrapper.attention_impl`; the
# layer instance is looked up via ``forward_context.no_compile_layers``,
# keyed on the wrapper's ``layer_name`` (registered in
# ``compilation_config.static_forward_context`` during ``__init__``).
#
# ``mutates_args=["out"]`` informs torch.compile / Dynamo that the output
# buffer is mutated in place; the fake impl is a no-op to satisfy the
# meta-tensor tracing path (the op writes into ``out`` so meta tensors do
# not need a return value).


def ilu_deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom-op trampoline for the ilu V4 W4A8 sparse MLA path."""
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, qr, kv, positions, out)


def ilu_deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake/meta impl: ``out`` is declared as ``mutates_args``; we have
    no meta-time computation to do, so just no-op."""
    return None


direct_register_custom_op(
    op_name="ilu_deepseek_v4_attention",
    op_func=ilu_deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=ilu_deepseek_v4_attention_fake,
)
