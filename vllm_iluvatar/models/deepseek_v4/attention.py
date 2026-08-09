# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4 MLA attention layer (W4A8, BF16/FP8 KV cache).

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

1. ``swa_cache_layer`` is :class:`IluDeepseekV4SWACache`; its dtype is
   selected by ``cache_config.cache_dtype`` (``bfloat16`` or ``uint8``
   for ``fp8_ds_mla``).
2. ``IluDeepseekV4MLAAttention`` accepts any ``cache_config.cache_dtype`` and
   defaults to ``bfloat16``; ``fp8`` is mapped to ``fp8_ds_mla``.
   Its ``get_kv_cache_spec`` returns the appropriate :class:`MLAAttentionSpec`
   with matching ``cache_dtype_str`` and ``model_version``.
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

import os
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
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.compressor import (
    CompressorMetadata,
    DeepseekCompressor,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_iluvatar import _custom_ops as ilu_ops
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes import (
    IluCompressedTensorsW8A8Int8,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluDeepseekV4FlashMLASparseBackend,
    IluFlashMLASparseMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV4IndexerBackend,
    IluDeepseekV32IndexerMetadata,
    get_max_prefill_buffer_size,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_sparse_swa import (
    IluDeepseekV4SWACache,
)

logger = init_logger("vllm." + __name__)

DynamicInt8Input = tuple[torch.Tensor, torch.Tensor, torch.dtype]
_C4A_OUTER_Q_N = 16384
_C4A_INDEXER_Q_N = 8192
_C4A_Q_K = 1536
_DS4_COMPRESSOR_SAVE_POOL_FUSION = (
    os.environ.get("IXFORMER_DS4_COMPRESSOR_SAVE_POOL_FUSION", "1") == "1"
)
_DS4_QK_NORM_QUANT_FUSION = (
    os.environ.get("IXFORMER_DS4_QK_NORM_QUANT_FUSION", "1") == "1"
)
_DS4_C4A_MERGED_WQ_B = os.environ.get("IXFORMER_DS4_C4A_MERGED_WQ_B", "0") == "1"
_DS4_SHARED_QR_QUANT = os.environ.get("IXFORMER_DS4_SHARED_QR_QUANT", "1") == "1"


def _can_fuse_qr_norm_quant(
    wrapper: Any,
    qr: torch.Tensor,
    kv: torch.Tensor,
    attn_metadata: Any,
) -> bool:
    if not (
        _DS4_QK_NORM_QUANT_FUSION
        and wrapper.is_main_decoder_layer
        and qr.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and wrapper.q_norm.weight.dtype == torch.bfloat16
        and wrapper.kv_norm.weight.dtype == torch.bfloat16
        and qr.dim() == 2
        and kv.dim() == 2
        and 1 <= qr.shape[0] <= 8
        and qr.shape == (kv.shape[0], _C4A_Q_K)
        and kv.shape[1] == 512
        and qr.stride(-1) == 1
        and kv.stride(-1) == 1
        and isinstance(
            getattr(wrapper.wq_b, "scheme", None),
            IluCompressedTensorsW8A8Int8,
        )
        and isinstance(attn_metadata, dict)
    ):
        return False

    if wrapper.indexer is not None and not isinstance(
        getattr(wrapper.indexer.wq_b, "scheme", None),
        IluCompressedTensorsW8A8Int8,
    ):
        return False

    swa_metadata = attn_metadata.get(wrapper.swa_cache_layer.prefix)
    return swa_metadata is not None and swa_metadata.num_prefills == 0


def _can_merge_c4a_wq_b(
    wrapper: Any,
    qr_projection_input: torch.Tensor | DynamicInt8Input,
    attn_metadata: Any,
) -> bool:
    if not (
        _DS4_C4A_MERGED_WQ_B
        and wrapper.is_main_decoder_layer
        and wrapper.indexer is not None
        and wrapper._c4a_merged_wq_b_weight is not None
        and wrapper._c4a_merged_wq_b_scale is not None
        and isinstance(qr_projection_input, tuple)
        and isinstance(attn_metadata, dict)
    ):
        return False

    qr_int8, qr_scale, out_dtype = qr_projection_input
    if not (
        qr_int8.dtype == torch.int8
        and qr_int8.shape == (1, _C4A_Q_K)
        and qr_scale.dtype == torch.float32
        and qr_scale.numel() == 1
        and out_dtype == torch.bfloat16
    ):
        return False

    swa_metadata = attn_metadata.get(wrapper.swa_cache_layer.prefix)
    return swa_metadata is not None and swa_metadata.num_prefills == 0


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

    def k_cache_layer(self):
        """The attention layer this compressor writes its K cache into.

        ``DeepseekCompressor`` resolves it through the static forward context
        captured at construction time (``get_current_vllm_config()`` is not
        available in ``forward``). Exposed as a method so the C4A fused path,
        which drives a second compressor's cache, does not have to reach into
        another module's private attribute.
        """
        return self._static_forward_context[self.k_cache_prefix]

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
        # CUDA graph replay may pad hidden states while metadata only covers
        # actual tokens. ixformer compressor ops require all per-token inputs
        # to share the same leading dimension.
        num_tokens = min(
            kv.shape[0],
            positions.shape[0],
            state_metadata.slot_mapping.shape[0],
        )
        kv = kv[:num_tokens]
        score = score[:num_tokens]
        positions = positions[:num_tokens]
        token_to_req_indices = state_metadata.token_to_req_indices[:num_tokens]
        slot_mapping = state_metadata.slot_mapping[:num_tokens]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # Drop trailing data-parallel padding rows.
        #
        # The model forward runs on ``num_tokens_padded`` rows (so ``x`` /
        # ``positions`` / ``kv`` / ``score`` all have that length), but the
        # attention metadata is only padded to the CUDA-graph batch size when
        # the step is dispatched in FULL cudagraph mode (``pad_attn``). Under
        # data-parallel, ranks agree on a common per-step ``num_tokens_across_dp``
        # and one rank running a prefill / non-uniform-decode batch forces the
        # *whole* DP group onto the eager (non-FULL) path. On the decode ranks
        # that means ``x``/``positions`` are DP-padded while ``slot_mapping`` /
        # ``token_to_req_indices`` stay at the unpadded ``num_actual`` length,
        # so ``kv.size(0) != slot_mapping.size(0)`` and the strict ixformer op
        # ``ds4_compressor_save_partial_states`` aborts with
        # "slot_mapping shape must be [num_tokens]".
        #
        # The padding rows are no-op tokens, so slice ``kv``/``score``/
        # ``positions`` back down to ``num_actual`` to keep every compressor
        # kernel argument the same length. In FULL cudagraph mode
        # ``num_actual == kv.size(0)`` and these slices are captured no-ops.
        num_actual = slot_mapping.shape[0]
        if kv.shape[0] != num_actual:
            kv = kv[:num_actual]
            score = score[:num_actual]
            positions = positions[:num_actual]

        # ``state_cache`` shape: [num_blocks, block_size, 2 * coff * head_dim]
        # (kv_state | score_state). fp32 by construction (see
        # ``CompressorStateCache``).
        state_cache = self.state_cache.kv_cache

        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        # Use num_actual (not num_tokens) so pad tokens are excluded, matching
        # the indexer compressor. Required for MTP verify (q_len > 1).
        kv_slot_mapping = k_cache_metadata.slot_mapping[:num_actual]
        mla_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = mla_layer.kv_cache
        kv_cache_dtype = getattr(mla_layer, "kv_cache_dtype", "bfloat16")
        if kv_cache_dtype == "fp8_ds_mla":
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
            ixfops.ds4_compressor_pool_norm_rope_fp8_kv_insert(
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                rms_norm_weight=self.norm.weight,
                cos_sin_cache=cos_sin_cache,
                k_cache=kv_cache,
                kv_slot_mapping=kv_slot_mapping,
                compress_ratio=self.compress_ratio,
                rope_head_dim=self.rope_head_dim,
                rms_norm_eps=self.rms_norm_eps,
            )
        elif (
            _DS4_COMPRESSOR_SAVE_POOL_FUSION
            and num_tokens > 0
            and kv_cache.dtype == torch.bfloat16
        ):
            ixfops.ds4_compressor_save_pool_norm_rope_insert(
                kv=kv,
                score=score,
                ape=self.ape,
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                rms_norm_weight=self.norm.weight,
                cos_sin_cache=cos_sin_cache,
                k_cache=kv_cache,
                kv_slot_mapping=kv_slot_mapping,
                compress_ratio=self.compress_ratio,
                rope_head_dim=self.rope_head_dim,
                rms_norm_eps=self.rms_norm_eps,
            )
        else:
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
            ixfops.ds4_compressor_pool_norm_rope_insert(
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                rms_norm_weight=self.norm.weight,
                cos_sin_cache=cos_sin_cache,
                k_cache=kv_cache,
                kv_slot_mapping=kv_slot_mapping,
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
        layer_id = extract_layer_index(prefix)
        self.is_main_decoder_layer = 0 <= layer_id < config.num_hidden_layers
        self.register_buffer(
            "_c4a_merged_wq_b_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "_c4a_merged_wq_b_scale",
            None,
            persistent=False,
        )

        # Per-head RMS normalization for Q (no learnable weights).
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        self.aux_stream = mla_modules.aux_stream
        self.ln_events: list[torch.cuda.Event] = []  # populated at stage 2

        assert cache_config is not None, (
            "ilu DeepseekV4 attention requires cache_config"
        )
        kv_cache_dtype = cache_config.cache_dtype or "bfloat16"
        if kv_cache_dtype == "auto":
            kv_cache_dtype = "bfloat16"
        if kv_cache_dtype == "fp8":
            kv_cache_dtype = "fp8_ds_mla"

        self.head_bytes = 584 if kv_cache_dtype == "fp8_ds_mla" else self.head_dim * 2

        swa_dtype = torch.uint8 if kv_cache_dtype == "fp8_ds_mla" else torch.bfloat16
        self.swa_cache_layer = IluDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=swa_dtype,
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
            self.compressor.kv_cache_dtype = self.mla_attn.kv_cache_dtype

        self._c4a_fused_supported = (
            envs.VLLM_ILU_DS4_C4A_FUSED_INDEXCOMPRESS
            and self._c4a_fused_layer_eligible()
        )
        if self.indexer is not None and envs.VLLM_ILU_DS4_C4A_FUSED_INDEXCOMPRESS:
            logger.info_once(
                "ilu DeepSeek-V4: C4A fused index+compress %s.",
                "enabled" if self._c4a_fused_supported else "unsupported on this layer",
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self._ensure_c4a_wq_b_merge()

    def _ensure_c4a_wq_b_merge(self) -> None:
        if (
            not _DS4_C4A_MERGED_WQ_B
            or not self.is_main_decoder_layer
            or self.indexer is None
            or self._c4a_merged_wq_b_weight is not None
        ):
            return

        outer_weight = getattr(self.wq_b, "weight", None)
        outer_scale = getattr(self.wq_b, "weight_scale", None)
        indexer_weight = getattr(self.indexer.wq_b, "weight", None)
        indexer_scale = getattr(self.indexer.wq_b, "weight_scale", None)
        if any(
            value is None
            for value in (
                outer_weight,
                outer_scale,
                indexer_weight,
                indexer_scale,
            )
        ):
            return

        if outer_weight.shape != (_C4A_OUTER_Q_N, _C4A_Q_K) or indexer_weight.shape != (
            _C4A_INDEXER_Q_N,
            _C4A_Q_K,
        ):
            return

        self._c4a_merged_wq_b_weight = torch.cat(
            (outer_weight.detach(), indexer_weight.detach()),
            dim=0,
        )
        self._c4a_merged_wq_b_scale = torch.cat(
            (outer_scale.detach(), indexer_scale.detach()),
            dim=0,
        )

    def _apply_c4a_merged_wq_b(
        self,
        qr_projection_input: DynamicInt8Input,
    ) -> torch.Tensor:
        qr_int8, qr_scale, out_dtype = qr_projection_input
        return ixfops.w8a8(
            qr_int8,
            self._c4a_merged_wq_b_weight,
            qr_scale,
            self._c4a_merged_wq_b_scale,
            format="TN",
            out_dtype=out_dtype,
        )

    def _c4a_fused_layer_eligible(self) -> bool:
        """Whether this layer's geometry is the one the fused op is built for.

        ``ds4_c4a_indexcompress`` hardcodes the C4A shapes -- compress_ratio 4,
        coff 2, 64x128 index heads, a 512-wide main head and a 64-wide RoPE
        band, i.e. a ``[2624, hidden]`` concatenated projection. It is bf16
        only (its projection runs on ``gemm_bf16_bf16_fp32``), writes both K
        caches in that dtype, and takes a single cos/sin table for the indexer
        q RoPE and both compressor pools. So it cannot serve a C128A layer, an
        fp8_ds_mla main cache, or a model that gives the indexer its own rotary
        embedding.

        The C++ layer re-checks the resulting shapes, but that only fires once
        a batch is already in flight; deciding here keeps an unsupported
        checkpoint on the per-op path instead of aborting the first decode.
        """
        if self.indexer is None or self.compressor is None:
            return False
        if self.compress_ratio != 4 or self.rope_head_dim != 64:
            return False
        if self.mla_attn.kv_cache_dtype == "fp8_ds_mla":
            return False
        if self.compressor.norm.weight.dtype != torch.bfloat16:
            return False
        if self.indexer_rotary_emb is not self.rotary_emb:
            return False
        indexer_compressor = self.indexer.compressor
        return (
            self.indexer.n_head == 64
            and self.indexer.head_dim == 128
            and indexer_compressor.head_dim == 128
            and indexer_compressor.coff == 2
            and self.compressor.head_dim == 512
            and self.compressor.coff == 2
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        fused_wqa_wkv_input: DynamicInt8Input | None = None,
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
        projection_input = (
            hidden_states if fused_wqa_wkv_input is None else fused_wqa_wkv_input
        )
        qr_kv, _ = self.fused_wqa_wkv(projection_input)
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
            # v2 W8A16 marlin path (set by IluV4W8A16BMMLinearMethod):
            # input stays bf16; weight is INT8 in Marlin layout +
            # per-(group,N) channel scale. The 3D batched signature mirrors
            # R1's _k_up_proj_w8a8 (vllm_iluvatar/v1/attention/backends/mla/ilu_mla.py).
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

        qr_projection_input: torch.Tensor | DynamicInt8Input
        if _can_fuse_qr_norm_quant(self, qr, kv, attn_metadata):
            qr, kv, qr_int8, qr_scale = ixfops.rms_norm_qk_dynamic_int8(
                qr,
                kv,
                self.q_norm.weight,
                self.kv_norm.weight,
                self.eps,
            )
            qr_projection_input = (qr_int8, qr_scale, qr.dtype)
        else:
            qr, kv = ixfops.rms_norm_qk(
                qr,
                kv,
                self.q_norm.weight,
                self.kv_norm.weight,
                self.eps,
            )
            qr_projection_input = qr
            can_share_qr_quant = (
                _DS4_SHARED_QR_QUANT
                and self.indexer is not None
                and isinstance(
                    getattr(self.wq_b, "scheme", None),
                    IluCompressedTensorsW8A8Int8,
                )
                and isinstance(
                    getattr(self.indexer.wq_b, "scheme", None),
                    IluCompressedTensorsW8A8Int8,
                )
            )
            if can_share_qr_quant:
                qr_int8, qr_scale, _ = ilu_ops.scaled_int8_quant(qr.contiguous())
                qr_projection_input = (qr_int8, qr_scale, qr.dtype)

        # 2) Q projection: (T, q_lora_rank) -> (T, n_local_heads, head_dim)
        # ``ds4_c4a_indexcompress`` projects the indexer's q itself, in bf16,
        # and takes no pre-projected q, so merging wq_b ahead of it would be
        # dead work. Where both paths apply, the fused one wins.
        use_c4a_fused = self.indexer is not None and self._use_c4a_fused(attn_metadata)
        use_merged_c4a_wq_b = not use_c4a_fused and _can_merge_c4a_wq_b(
            self,
            qr_projection_input,
            attn_metadata,
        )
        indexer_q = None
        if use_merged_c4a_wq_b:
            assert isinstance(qr_projection_input, tuple)
            merged_q = self._apply_c4a_merged_wq_b(qr_projection_input)
            q, indexer_q = merged_q.split(
                [_C4A_OUTER_Q_N, _C4A_INDEXER_Q_N],
                dim=-1,
            )
            q = q.view(-1, self.n_local_heads, self.head_dim)
        else:
            q = self.wq_b(qr_projection_input).view(
                -1, self.n_local_heads, self.head_dim
            )

        # 3) Per-layer-type sequencing.
        if self.indexer is not None:
            # C4A layer: indexer (own compressor + topk) + main compressor
            #            (writes main MLA cache) + SWA insert.
            assert self.compressor is not None
            if use_c4a_fused:
                self.indexer.forward_c4a_fused(
                    hidden_states,
                    qr,
                    positions,
                    self.indexer_rotary_emb,
                    self.compressor,
                )
            elif indexer_q is None:
                self.indexer(
                    hidden_states,
                    qr_projection_input,
                    positions,
                    self.indexer_rotary_emb,
                )
                self.compressor(hidden_states, positions, self.rotary_emb)
            else:
                self.indexer(
                    hidden_states,
                    qr_projection_input,
                    positions,
                    self.indexer_rotary_emb,
                    projected_q=indexer_q,
                )
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

        # 5) Until this, all kv cache are wrote, do kv transfer if needed
        if has_kv_transfer_group():
            connector = get_kv_transfer_group()
            connector.save_kv_layer("layer_name", "kv_cache", "attn_metadata")

        # 6) Run the sparse MLA forward into the pre-allocated ``out`` buffer
        #    (shape ``(T, n_local_heads, head_dim)``). No head-count padding:
        #    ixformer kernels (``ds4_flash_mla_sparse_fwd`` /
        #    ``ds4_flash_mla_with_kvcache``) accept arbitrary ``h_q``.
        self.mla_attn(q, kv, positions, output=out)

    def _use_c4a_fused(self, attn_metadata) -> bool:
        """Whether this batch may go through ``ds4_c4a_indexcompress``.

        The fused post pass launches both compressors' ``save_partial`` and
        both ``pool_norm_rope_insert`` phases as sibling block groups of a
        single kernel with no grid barrier between them. That is safe because
        a pooling block never waits on a sibling: it takes *every* window row
        this batch produces straight from the projection output, and only rows
        written by an earlier step from the paged state cache. Resolving an
        in-batch row is positional -- the row at ``pos - d`` is token ``t - d``
        when that token is the same request sitting at that position -- so the
        only thing the op needs is vLLM's own batch layout, a request's query
        tokens contiguous and ascending. Plain decode (one token per request)
        and MTP / spec-decode verify steps (``next_n`` consecutive tokens) both
        satisfy it.

        Prefill stays on the per-op path. It would be correct by the same rule,
        but the launch geometry (five block groups of one block per token, and
        one whole block per pooled token) is sized for decode-length batches.

        Dummy / profiling runs are routed through the fused path as well so it
        can build its concatenated weight matrix and reserve the projection
        workspace before CUDA-graph capture.
        """
        if not self._c4a_fused_supported:
            return False
        if not isinstance(attn_metadata, dict):
            return True
        assert self.indexer is not None
        md = attn_metadata.get(self.indexer.k_cache.prefix)
        if not isinstance(md, IluDeepseekV32IndexerMetadata):
            return False
        return md.num_prefills == 0 and md.decode is not None

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

        if self.mla_attn.kv_cache_dtype == "fp8_ds_mla":
            # Pass the 3D uint8 cache directly; the CUDA insert kernel uses
            # stride(0) and must not flatten padded (non-contiguous) blocks.
            ixfops.ds4_fused_qnorm_rope_fp8_kv_insert(
                q,
                kv,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
        else:
            ixfops.ds4_fused_qnorm_rope_kv_insert(
                q,
                kv,
                swa_kv_cache,
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

        kv_cache_dtype = (
            cache_config.cache_dtype
            if cache_config is not None and cache_config.cache_dtype not in (None, "")
            else "bfloat16"
        )
        if kv_cache_dtype == "auto":
            kv_cache_dtype = "bfloat16"
        if kv_cache_dtype == "fp8":
            kv_cache_dtype = "fp8_ds_mla"
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
        if self.compress_ratio <= 1:
            return None

        if self.kv_cache_dtype == "fp8_ds_mla":
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                dtype=torch.uint8,
                compress_ratio=self.compress_ratio,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                # Ilu fp8 attention kernels require 64B-aligned block bases.
                # C128 pages (2*584=1168) pad to 1216; C4/SWA are already aligned.
                alignment=64,
            )

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

            # bf16-only experimental path: run prefill through the per-token
            # decode sparse MLA kernel (ds4_flash_mla_with_kvcache).
            prefill_as_decode = (
                envs.VLLM_ILU_DS4_PREFILL_AS_DECODE
                and self.kv_cache_dtype != "fp8_ds_mla"
            )
            if prefill_as_decode:
                self._forward_prefill_as_decode(
                    q=prefill_q,
                    kv_cache=self_kv_cache,
                    swa_metadata=swa_metadata,
                    attn_metadata=flashmla_metadata if not swa_only else None,
                    swa_only=swa_only,
                    output=prefill_out,
                )
            elif envs.VLLM_ILU_DS4_PREFILL_OPT:
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

        if self.kv_cache_dtype == "fp8_ds_mla":
            # fp8_ds_mla cache is already (num_blocks, block_size, 584);
            # the fp8 attention kernel accepts 3D directly.
            swa_cache = self.swa_cache_layer.kv_cache
            ixfops.ds4_flash_mla_with_fp8_kvcache(
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
        else:
            # bf16 cache: reshape to (num_blocks, page_block_size, 1, head_dim)
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

    def _gather_k_cache(
        self,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        gather_lens: torch.Tensor | None,
        block_table: torch.Tensor,
        block_size: int,
        offset: int,
    ) -> None:
        """Gather K cache into bf16 workspace, dispatching by cache format.

        When ``kv_cache_dtype == "fp8_ds_mla"``, uses
        ``ds4_gather_fp8_k_cache`` which dequantizes FP8 NoPE dims and
        copies bf16 RoPE dims into the bf16 output workspace.
        Otherwise, calls the standard ``ds4_gather_k_cache`` for bf16 cache.
        """
        if self.kv_cache_dtype == "fp8_ds_mla":
            ixfops.ds4_gather_fp8_k_cache(
                out=out,
                k_cache=k_cache,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                block_table=block_table,
                block_size=block_size,
                offset=offset,
            )
        else:
            ixfops.ds4_gather_k_cache(
                out=out,
                k_cache=k_cache,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                block_table=block_table,
                block_size=block_size,
                offset=offset,
            )

    def _forward_prefill_as_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekSparseSWAMetadata,
        attn_metadata: IluFlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Prefill-side sparse MLA via the per-token *decode* kernel (bf16 only).

        Experimental alternative to :meth:`_forward_prefill` /
        :meth:`_forward_prefill_opt` (gated by ``VLLM_ILU_DS4_PREFILL_AS_DECODE``):
        instead of gathering a shared bf16 workspace and calling
        ``ds4_flash_mla_sparse_fwd``, treat every prefill query token as its own
        batch element (``b = num_prefill_tokens``, ``s_q = 1``) and reuse the
        decode kernel ``ds4_flash_mla_with_kvcache``.

        This is well-defined because the per-token SWA window and compressed
        top-k slot sourcing are purely position-based and identical for decode
        and prefill tokens:

        - SWA indices/lens come from ``swa_metadata.prefill_swa_indices`` /
          ``prefill_swa_lens`` (the SWA metadata builder now computes these for
          the prefill token range when the flag is set), already in global
          paged-slot form.
        - C4A (``compress_ratio == 4``): the indexer wrote per-prefill-token
          *local* compressed indices into ``self.topk_indices_buffer``; we map
          them to global paged slots with
          ``ixfops.ds4_compute_global_topk_indices_and_lens`` -- the same op the
          decode path uses.
        - C128A (``compress_ratio == 128``): the FlashMLA sparse builder
          pre-computed per-prefill-token *local* compressed indices in
          ``attn_metadata.c128a_prefill_topk_indices``; we map them to global
          paged slots the same way.

        Unlike the decode path (where request ids are 0-based within the decode
        prefix), prefill ``token_to_req_indices`` carry *global* request ids, so
        the full (unsliced) ``block_table`` is handed to the global-index op.

        Only the bf16 KV cache path is supported; the caller guards on
        ``kv_cache_dtype != "fp8_ds_mla"``.
        """
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        if num_prefill_tokens == 0:
            return

        swa_indices = swa_metadata.prefill_swa_indices
        swa_lens = swa_metadata.prefill_swa_lens
        assert swa_indices is not None and swa_lens is not None, (
            "prefill_swa_indices / prefill_swa_lens must be populated when "
            "VLLM_ILU_DS4_PREFILL_AS_DECODE is enabled; check the SWA metadata "
            "builder."
        )

        token_end = num_decode_tokens + num_prefill_tokens

        topk_indices: torch.Tensor | None = None
        topk_lens: torch.Tensor | None = None
        if not swa_only:
            assert attn_metadata is not None, (
                "FlashMLASparseMetadata is required when compress_ratio > 1; "
                f"prefix={self.prefix!r}"
            )
            assert swa_metadata.is_valid_token is not None
            assert swa_metadata.token_to_req_indices is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[num_decode_tokens:token_end]
            # Prefill tokens carry *global* request ids, so pass the full
            # block_table (rows are indexed by the global req id directly).
            token_to_req = swa_metadata.token_to_req_indices[
                num_decode_tokens:token_end
            ]
            if self.compress_ratio == 4:
                # C4A: per-layer local topk indices written by the indexer.
                assert self.topk_indices_buffer is not None
                local_topk = self.topk_indices_buffer[num_decode_tokens:token_end]
            elif self.compress_ratio == 128:
                # C128A: per-prefill-token local compressed indices from the
                # FlashMLA sparse builder.
                local_topk = attn_metadata.c128a_prefill_topk_indices
                assert local_topk is not None
                local_topk = local_topk[:num_prefill_tokens]
            else:
                raise ValueError(
                    f"Unsupported compress_ratio={self.compress_ratio}; "
                    "expected 1, 4, or 128."
                )
            global_indices, topk_lens = ixfops.ds4_compute_global_topk_indices_and_lens(
                local_topk,
                token_to_req,
                attn_metadata.block_table,
                block_size,
                is_valid,
            )
            topk_indices = global_indices.view(num_prefill_tokens, 1, -1)

        # Each prefill query token is its own batch element (s_q = 1), matching
        # the decode kernel's per-token layout.
        q = q.unsqueeze(1)

        # bf16 cache: reshape to (num_blocks, page_block_size, 1, head_dim).
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
                self._gather_k_cache(
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
            self._gather_k_cache(
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
        # Concatenated projection matrix for ``forward_c4a_fused``; built on
        # the first fused call, once the checkpoint weights are in place.
        self._cmp_a: torch.Tensor | None = None

        # Stage 2: we do *not* register a SparseAttnIndexer custom op here.
        # The indexer's forward inlines the bf16 mqa-logits + topk kernels
        # because the wrapper's outer forward is itself a custom op
        # boundary, so there is no need to wrap a second custom op around
        # ixformer's stateless calls. The K-cache insertion is also
        # offloaded to ``self.compressor`` (which already writes the bf16
        # indexer cache), so the legacy ``ops.indexer_k_cache`` step from
        # ``SparseAttnIndexer`` is intentionally dropped.

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor | DynamicInt8Input,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        projected_q: torch.Tensor | None = None,
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
        if projected_q is None:
            q, _ = self.wq_b(qr)
        else:
            q = projected_q
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

        return self._topk_tail(q, weights)

    def _topk_tail(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """MQA logits + top-k over the compressed indexer K cache.

        Shared by :meth:`forward` and :meth:`forward_c4a_fused`; both only
        differ in how ``q`` / ``weights`` are produced.
        """
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
            # Ragged (non-uniform query length) decode batches -- e.g. K>=2 MTP
            # where requests carry a variable number of accepted/proposed spec
            # tokens -- are supported natively. ``dsa_indexer_mqa_logits_with_blocks_bf16``
            # iterates requests via ``cu_seqlens_q`` (the same variable-q path
            # prefill uses) and ``cu_seqlen_ks/ke`` carry one causal span per
            # query token, so each decode token is an independent logits/topk
            # row. No pack/unpack is required (that is only needed by the
            # community paged-MQA kernel which expects a uniform ``[B, next_n]``
            # layout). ``requires_padding`` is kept on the metadata only as a
            # diagnostic flag.
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

    def _build_cmp_a(self, main_compressor: IluDeepseekCompressor) -> torch.Tensor:
        """Concatenate the three projections ``ds4_c4a_indexcompress`` expects
        into one ``[2624, hidden_size]`` matrix: indexer kv|score (512),
        indexer weights (64), main kv|score (2048).

        The source parameters are re-pointed at row views of the result so the
        weights live in one buffer instead of being stored twice. Both
        compressors use ``disable_tp=True`` and ``weights_proj`` is replicated,
        so the row count is TP-invariant.
        """
        ic_w = self.compressor.fused_wkv_wgate.weight
        wt_w = self.weights_proj.weight
        main_w = main_compressor.fused_wkv_wgate.weight
        assert ic_w.dtype == wt_w.dtype == main_w.dtype, (
            "C4A fused index+compress needs one dtype across the three "
            f"projections, got {ic_w.dtype}/{wt_w.dtype}/{main_w.dtype}; a "
            "quantized compressor or weights_proj is not supported"
        )
        assert ic_w.shape[1] == wt_w.shape[1] == main_w.shape[1], (
            "C4A fused index+compress needs the three projections to share a "
            f"reduction dim, got {ic_w.shape}/{wt_w.shape}/{main_w.shape}"
        )
        cmp_a = torch.cat([ic_w, wt_w, main_w], dim=0)
        n_ic, n_wt = ic_w.shape[0], wt_w.shape[0]
        self.compressor.fused_wkv_wgate.weight.data = cmp_a[:n_ic]
        self.weights_proj.weight.data = cmp_a[n_ic : n_ic + n_wt]
        main_compressor.fused_wkv_wgate.weight.data = cmp_a[n_ic + n_wt :]
        return cmp_a

    def forward_c4a_fused(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        main_compressor: IluDeepseekCompressor,
    ) -> torch.Tensor | None:
        """Fused replacement for ``indexer(...)`` plus the C4A layer's main
        ``compressor(...)``: one concatenated GEMM and one task-split paged
        post pass instead of two GEMMs and five per-op launches.

        ``ds4_c4a_indexcompress`` covers the indexer ``save_partial`` +
        ``pool_norm_rope_insert``, ``weights_proj`` + its scale, the indexer q
        RoPE, and the main compressor's ``save_partial`` +
        ``pool_norm_rope_insert``. MQA logits and top-k stay on the per-op
        path (:meth:`_topk_tail`).

        Eligibility is decided by the caller
        (``IluDeepseekV4MultiHeadLatentAttentionWrapper._use_c4a_fused``); this
        method assumes a decode batch whose query tokens are contiguous and
        ascending per request (plain decode or an MTP verify step) and a bf16
        main KV cache.
        """
        if self._cmp_a is None:
            # Concatenating reallocates the three checkpoint weights, so this
            # cannot happen under graph capture. The dummy run is routed here
            # (see ``_use_c4a_fused``) precisely so that it happens first;
            # assert rather than let capture record a stale pointer.
            assert not torch.cuda.is_current_stream_capturing(), (
                "C4A fused index+compress: the concatenated projection matrix "
                "must be built before CUDA-graph capture, but no dummy run "
                "reached forward_c4a_fused"
            )
            self._cmp_a = self._build_cmp_a(main_compressor)

        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        num_tokens = q.shape[0]

        # Shared, monotonically growing workspace: the projection output is
        # dead once the post pass has run, so every layer reuses the same
        # bytes. Requested on dummy runs too, so the buffer is already at full
        # size before CUDA-graph capture locks it.
        (cmp_out,) = current_workspace_manager().get_simultaneous(
            ((num_tokens, self._cmp_a.shape[0]), torch.float32),
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer

        ic_md = cast(
            CompressorMetadata,
            attn_metadata[self.compressor.state_cache.prefix],
        )
        main_md = cast(
            CompressorMetadata,
            attn_metadata[main_compressor.state_cache.prefix],
        )
        idx_k_md = cast(Any, attn_metadata[self.compressor.k_cache_prefix])
        main_k_md = cast(Any, attn_metadata[main_compressor.k_cache_prefix])
        main_k = main_compressor.k_cache_layer().kv_cache

        # The op requires every per-token argument to have the same leading
        # dim. Under data parallelism the model forward runs on DP-padded rows
        # while the metadata stays at ``num_actual``, so trim to the shortest
        # (same reasoning as ``IluDeepseekCompressor.forward``; in FULL
        # cudagraph mode these slices are captured no-ops).
        num_actual = min(
            num_tokens,
            positions.shape[0],
            ic_md.slot_mapping.shape[0],
            main_md.slot_mapping.shape[0],
            idx_k_md.slot_mapping.shape[0],
            main_k_md.slot_mapping.shape[0],
        )

        # Sized like ``q`` rather than like the op's input, because
        # ``_topk_tail`` slices both with the same token counts. Only the first
        # ``num_actual`` rows are written; the DP-padding tail is never read,
        # exactly as for ``q``.
        weights = torch.empty(
            (num_tokens, self.n_head), dtype=q.dtype, device=q.device
        )
        ixfops.ds4_c4a_indexcompress(
            cmp_a=self._cmp_a,
            cmp_x=hidden_states[:num_actual],
            cmp_out=cmp_out[:num_actual],
            index_q=q[:num_actual],
            weights=weights[:num_actual],
            positions=positions[:num_actual],
            ape_ic=self.compressor.ape,
            ape_main=main_compressor.ape,
            rms_ic=self.compressor.norm.weight,
            rms_main=main_compressor.norm.weight,
            cos_sin=rotary_emb.cos_sin_cache,
            ic_state=self.compressor.state_cache.kv_cache,
            main_state=main_compressor.state_cache.kv_cache,
            idx_k=self.k_cache.kv_cache,
            main_k=main_k,
            token_to_req_indices=ic_md.token_to_req_indices[:num_actual],
            ic_block_table=ic_md.block_table,
            main_block_table=main_md.block_table,
            ic_state_slot_mapping=ic_md.slot_mapping[:num_actual],
            main_state_slot_mapping=main_md.slot_mapping[:num_actual],
            idx_k_slot_mapping=idx_k_md.slot_mapping[:num_actual],
            main_k_slot_mapping=main_k_md.slot_mapping[:num_actual],
            ic_block_size=ic_md.block_size,
            main_block_size=main_md.block_size,
            rms_norm_eps=self.compressor.rms_norm_eps,
        )
        return self._topk_tail(q, weights)


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
