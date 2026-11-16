# SPDX-License-Identifier: Apache-2.0
"""Iluvatar DeepSeek-V4-Flash sparse-MLA attention for the raw FP8 checkpoint.

Packed KV uses ``fp8_ds_mla``; plain direct KV uses E4M3 with scale one.
The heavy ops reuse IxFormer ``ds4_*`` /
``dsa_*`` kernels; input/output projections run on the block-fp8 LinearMethod
(``wo_a`` uses the fp8 grouped-einsum path, see ``_o_proj``). ``forward_mqa``
routes large causal TP1 prefill through gather/dequantize + BF16 sparse prefill
v2. Decode, small fragments and lower-head paths retain the packed paged
operation. The SWA/main-MLA caches and compressor use the ilu
variants (community ones build NV-shaped metadata or launch Triton kernels the
Iluvatar stack rejects). The lightning indexer is the self-contained
``IluDeepseekV4Fp8Indexer`` with a grouped fp8 K cache.

Deviations from the NV public version: no head padding, single-stream, no
custom op.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import ixformer.inference.functions as ixfops
import torch
import torch.nn as nn
import vllm.envs as envs
from ixformer.inference.backend_registry import registry
from ixformer.inference.functions.models.deepseek_v4.mla_sparse.float.prefill_q_fp8_round_bf16 import (
    prefill_q_fp8_round_bf16,
)
from ixformer.inference.functions.models.deepseek_v41.mla_sparse.fp8.quant_mla_q_fp8_pt import (
    quant_mla_q_fp8_pt,
)
from ixformer.inference.functions.models.deepseek_v41.mla_sparse.fp8.sparse_mla_fp8_pt import (
    sparse_mla_fp8_pt,
)
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_iluvatar.models.deepseek_v4.attention import (
    PREFILL_CHUNK_SIZE,
    IluDeepseekCompressor,
    IluDeepseekV4MLAAttention,
)
from vllm_iluvatar.models.deepseek_v4.flashmla_sparse import (
    IluDeepseekV4Fp8FlashMLASparseBackend,
)
from vllm_iluvatar.models.deepseek_v4.sparse_swa import (
    IluDeepseekV4Fp8SWACache,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV4IndexerBackend,
    IluDeepseekV32IndexerMetadata,
)

logger = init_logger(__name__)

# 4-byte fp32 ue8m0 scale per token (quant_block_size == index_head_dim).
_INDEXER_FP8_SCALE_BYTES = 4
# ds4_sparse_mla_fp8_decode: h_q is a kernel bucket; indices.stride(0) % 256 == 0.
_SPARSE_DECODE_H_Q = (1, 8, 16, 32, 64, 128)
_SPARSE_DECODE_INDEX_ALIGN = 256

# flash_mla_sparse_fwd/ixformer_perf: h_q is a kernel bucket, and the backend
# raises rather than falling back on anything outside it.
_SPARSE_PREFILL_H_Q = (8, 16, 32, 64, 128)

# Both prefill paths are legal for every bucket the two kernels share, so the
# split-prefill gate is a throughput choice, not a capability one. Measured on
# ivcore30 at index_topk 512 + window 128 (640 index columns), head_dim 512,
# s_kv 164352, per 8192 prefill tokens -- flash_mla_sparse_fwd/ixformer_perf
# versus ds4_sparse_mla_fp8_decode fed the same tokens:
#
#   h_q    8     16     32     64
#   fwd   10.9   7.5    7.0    7.4   ms
#   dec    4.9   4.7    8.1   13.4   ms
#
# flash_mla_sparse_fwd is flat in h_q (it is bound by the topk KV gather), while
# the decode kernel scales with h_q, so they cross between 16 and 32. Below the
# crossover the paged kernel also skips the bf16 KV workspace materialisation
# entirely, so keep those head counts on it. Ratios held at 1k/2k/4k tokens too.
_SPARSE_PREFILL_MIN_H_Q = 32


def _as_sparse_decode_indices(indices: torch.Tensor) -> torch.Tensor:
    """``(T, 1, K)`` or ``(T, K)`` → ``(T, K)`` with ``stride(0) % 256 == 0``."""
    if indices.dim() == 3:
        indices = indices.squeeze(1)
    if (
        indices.dim() != 2
        or indices.stride(1) != 1
        or indices.stride(0) % _SPARSE_DECODE_INDEX_ALIGN != 0
    ):
        raise ValueError(
            "ds4_sparse_mla_fp8_decode requires 2D indices with stride(1)==1 "
            f"and stride(0)%{_SPARSE_DECODE_INDEX_ALIGN}==0; got "
            f"shape={tuple(indices.shape)} stride={indices.stride()}"
        )
    return indices


class IluDeepseekV4Fp8IndexerCache(nn.Module, AttentionLayerBase):
    """FP8 grouped indexer K cache.

    uint8 paged cache with ``head_size = index_head_dim + 4``. Per block the
    ``[block_size, 128]`` e4m3 K region and ``[block_size]`` fp32 ue8m0 scale
    region are stored *grouped* (not interleaved per token) so each K row is
    128B/16-aligned, which the ixmma matrix loader on ix30 requires.
    """

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        # head_size = head_dim fp8 bytes + 4B fp32 ue8m0 scale (quant_block ==
        # head_dim); per block stored grouped (K region then scale region).
        self.head_dim = head_dim + _INDEXER_FP8_SCALE_BYTES
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = torch.uint8
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # vLLM v0.31.0 (db9527a468) KV-cache layout: allocate_kv_cache binds a 4D
        # [B, H=1, N, C] tensor; squeeze the singleton head dim -> 3D [B, N, C]
        # as required by the ilu ixformer indexer kernels (ndim 2 or 3 only).
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV4IndexerBackend


class IluDeepseekV4Fp8IndexerCompressor(IluDeepseekCompressor):
    """Indexer compressor that writes the grouped fp8 K layout.

    Overrides only the final K insert (grouped fp8) of the shared
    ``IluDeepseekCompressor``; the GEMM / split / ``save_partial_states``
    prologue is reused.
    """

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        kv_score = ixfops.gemm_bf16_bf16_fp32(
            x, self.fused_wkv_wgate.weight, format="TN"
        )
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = attn_metadata[self.state_cache.prefix]
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

        # Drop trailing DP padding rows so every per-token arg matches
        # ``slot_mapping`` length.
        num_actual = slot_mapping.shape[0]
        if kv.shape[0] != num_actual:
            kv = kv[:num_actual]
            score = score[:num_actual]
            positions = positions[:num_actual]

        state_cache = self.state_cache.kv_cache

        ixfops.save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            slot_mapping=slot_mapping,
            state_cache=state_cache,
            block_size=block_size,
            compress_ratio=self.compress_ratio,
        )

        # pool + RMSNorm + GPT-J RoPE + grouped fp8 insert.
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = attn_metadata[self.k_cache_prefix]
        kv_slot_mapping = k_cache_metadata.slot_mapping[:num_actual]
        mla_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = mla_layer.kv_cache

        ixfops.ds4_indexer_compressor_pool_norm_rope_fp8_kv_insert(
            state_cache,
            token_to_req_indices[:num_actual],
            positions,
            slot_mapping,
            block_table,
            self.norm.weight,
            cos_sin_cache,
            kv_cache,
            kv_slot_mapping,
            self.compress_ratio,
            self.rope_head_dim,
            self.rms_norm_eps,
        )


class IluDeepseekV4Fp8Indexer(nn.Module):
    """Self-contained fp8 lightning indexer (grouped fp8 K cache).

    Sub-module names/shapes match the W4A8 ``IluDeepseekV4Indexer`` so weight
    loading is unchanged. All three compute steps (q rope+quant, K insert, paged
    mqa-logits) run on ixformer fp8 kernels.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config,
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
        self.softmax_scale = self.head_dim**-0.5
        self.topk_indices_buffer = topk_indices_buffer
        self.prefix = prefix

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

        assert cache_config is not None, (
            "Ilu DeepSeek V4 fp8 indexer needs cache_config"
        )
        self.k_cache = IluDeepseekV4Fp8IndexerCache(
            head_dim=self.head_dim,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = IluDeepseekV4Fp8IndexerCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
        )
        logger.info_once(
            "Ilu DeepSeek V4 fp8 indexer: grouped fp8 K cache, ixformer "
            "q-rope-quant / K-insert / paged mqa-logits kernels."
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor | None:
        """fp8 lightning-indexer forward: writes per-token top-k slots into
        ``topk_indices_buffer`` and returns it."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        # K: compress + norm + rope + grouped fp8 insert.
        self.compressor(hidden_states, positions, rotary_emb)

        weights_in, _ = self.weights_proj(hidden_states)
        # Fused forward-RoPE + per-(token,head) ue8m0 fp8 quant (q_scale folded
        # into weights).
        q_fp8, weights = ixfops.fused_indexer_q_rope_quant(
            q,
            positions,
            rotary_emb.cos_sin_cache,
            weights_in,
            self.softmax_scale,
            self.n_head**-0.5,
            self.rope_dim,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer

        md = attn_metadata.get(self.k_cache.prefix)
        if md is None:
            return self.topk_indices_buffer
        assert isinstance(md, IluDeepseekV32IndexerMetadata)

        kv_cache = self.k_cache.kv_cache
        topk_buffer = self.topk_indices_buffer

        # Only C4A feeds this buffer into ``_build_extra_topk``; C128A takes its
        # prefill indices from the metadata builder and cr == 1 never calls
        # ``_build_extra_topk`` at all. Under C128A the skip below fires for
        # every prompt shorter than 128 tokens and the chunk can span the whole
        # batch, so resetting there would cost ``index_topk`` int32 per token,
        # per layer, for rows nobody reads.
        #
        # The reset is a stopgap. ixformer a9d172e2 (branch yd_opt_fusion, not
        # yet on the ixformer mainline this plugin builds against) gives
        # ``compute_global_topk_indices_and_lens`` optional ``positions`` /
        # ``compress_ratio`` args and bounds each row by
        # ``min((positions[t] + 1) / compress_ratio, topk)``. That makes a
        # never-written row unrepresentable at the point of use instead of
        # asking the caller to scrub a buffer whose lifetime it does not own,
        # and it is faster for short contexts. Once it lands: thread
        # ``positions`` (already a ``forward_mqa`` arg, int64 is accepted)
        # through ``_build_extra_topk`` and delete both this flag and the
        # reset below.
        topk_buffer_is_read = self.compress_ratio == 4

        if md.num_prefills > 0:
            assert md.prefill is not None
            for chunk in md.prefill.chunks:
                if chunk.max_context_len == 0:
                    # No compressed KV for any request in this chunk (C4A:
                    # every seq_len <= 3), so the kernels below are skipped.
                    # The buffer is torch.empty and only ever rewritten in
                    # place, so leaving these rows alone keeps the previous
                    # step's slots; ``compute_global_topk_indices_and_lens``
                    # would then index this step's block_table with them. The
                    # is_valid gate does not cover it -- these are real tokens
                    # with real slots, unlike the all-padding warmup chunks.
                    # -1 is the invalid sentinel that makes topk_lens zero.
                    if topk_buffer_is_read:
                        topk_buffer[chunk.token_start : chunk.token_end].fill_(-1)
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
            num_decode_tokens = md.num_decode_tokens
            if decode_md.max_context_len == 0:
                topk_buffer[:num_decode_tokens].fill_(-1)
                return topk_buffer
            logits = ixfops.dsa_indexer_mqa_logits_with_kvcache_fp8(
                q_fp8[:num_decode_tokens],
                decode_md.cu_seqlens_q,
                decode_md.cu_seqlens_kv,
                kv_cache,
                decode_md.block_table,
                weights[:num_decode_tokens],
                max_q_len=decode_md.max_q_len,
                max_kv_len=decode_md.max_kv_len,
                max_context_len=decode_md.max_context_len,
                valid_lens=decode_md.valid_lens,
            )
            ixfops.dsa_update_topk_indices(
                logits,
                decode_md.cu_seqlen_ks,
                decode_md.cu_seqlen_ke,
                self.topk_tokens,
                topk_buffer[:num_decode_tokens],
            )

        return topk_buffer


class IluDeepseekV4Fp8Attention(nn.Module, AttentionLayerBase):
    """DeepSeek-V4-Flash sparse-MLA attention for the raw-FP8 checkpoint.

    No head padding: ``attn_sink`` and ``o`` carry exactly ``n_local_heads``.
    ``backend_cls`` is bound directly instead of going through the platform
    selector.
    """

    # Bound directly: the platform selector returns V3.2's FLASHMLA_SPARSE for
    # (use_mla and use_sparse), not the fp8 DSV4 backend.
    backend_cls: ClassVar[type[AttentionBackend]] = (
        IluDeepseekV4Fp8FlashMLASparseBackend
    )
    # UE8M0 block-scaled fp8 packed as uint8 (`fp8_ds_mla`).
    use_fp8_ds_mla_layout: ClassVar[bool] = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)
        self.attn_mode = envs.VLLM_ILU_DS4_ATTN_MODE
        self.use_fp8_ds_mla_layout = self.attn_mode == "packed"
        if not self.use_fp8_ds_mla_layout and tp_size not in (1, 2, 4, 8):
            raise ValueError("DeepSeek V4 plain attention requires TP1/2/4/8")

        self.prefix = prefix  # the compressor keys its k-cache off this
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        if not self.use_fp8_ds_mla_layout and self.n_heads != 64:
            raise ValueError(
                "DeepSeek V4 plain attention requires Flash with 64 total heads"
            )
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # compress_ratios has num_hidden_layers + num_nextn_predict_layers
        # entries; dense layers and MTP store 0, which degrades to 1.
        if layer_id < len(config.compress_ratios):
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        # Wide shards take the split-prefill path; narrow ones are faster on the
        # paged decode kernel (see _SPARSE_PREFILL_MIN_H_Q). Both kernels bucket
        # h_q, so a shard that is not a bucket has no split-prefill kernel at all.
        self.use_prefill_sparse = (
            envs.VLLM_ILU_DS4_FP8_PREFILL_SPARSE
            and self.use_fp8_ds_mla_layout
            and self.n_local_heads >= _SPARSE_PREFILL_MIN_H_Q
            and self.n_local_heads in _SPARSE_PREFILL_H_Q
        )

        if not self.use_fp8_ds_mla_layout:
            for op in (
                "quant_mla_q_fp8_pt",
                "sparse_mla_fp8_pt",
            ):
                if "ixformer_native" not in registry.available_backends(op):
                    raise RuntimeError(f"plain attention requires {op}/ixformer_native")

        if self.use_prefill_sparse:
            for op, backend in (
                ("prefill_q_fp8_round_bf16", "ixformer_native"),
                ("flash_mla_sparse_fwd", "ixformer_perf"),
            ):
                if backend not in registry.available_backends(op):
                    raise RuntimeError(f"Sparse prefill requires {op}/{backend}")

        # No head padding: packed decode buckets ``h_q`` exactly (no pad);
        # ``attn_sink`` carries ``n_local_heads`` values.
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        # ---- quantised linears (community Fp8LinearMethod, zero changes) ----
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        # The wo_a quant_method (``ilu_fp8_block_scaled_mm``) requantizes the
        # block-fp8 weight to per-channel fp8 and attaches the grouped views
        # ``bmm_b_fp8`` [G, D, R] + ``bmm_bs`` [G, D] at load time, so
        # ``_o_proj`` does zero weight work in forward. ``bmm_geom`` carries the
        # reshape geometry the quant_method cannot recover from the flat
        # ``output_partition_sizes``.
        self.wo_a.bmm_geom = SimpleNamespace(
            n_local_groups=self.n_local_groups,
            o_lora_rank=self.o_lora_rank,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            head_dim=self.head_dim,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        # cos_sin_cache only; no kernel launch at build time.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer

        # Only C4A layers run sparse attention and therefore own an indexer.
        self.indexer: IluDeepseekV4Fp8Indexer | None = None
        if self.compress_ratio == 4:
            self.indexer = self._build_indexer(
                vllm_config,
                config=config,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.indexer",
            )

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"

        if not self.use_fp8_ds_mla_layout:
            if cache_config.cache_dtype not in ("auto", "fp8"):
                raise ValueError("plain attention requires auto or fp8 cache dtype")
            cache_config.cache_dtype = "fp8"
            logger.info_once(
                "DeepSeek V4 attention mode=%s, plain E4M3 KV scale=1", self.attn_mode
            )

        # `--kv-cache-dtype fp8_ds_mla` is a hard precondition; the resolver
        # asserts it and normalises the string back onto cache_config.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = IluDeepseekV4Fp8SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with the compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Built after the attention setup so the compressor sees self.prefix.
        # ``IluDeepseekCompressor`` (not community ``DeepseekCompressor``, whose
        # Triton kernels the Iluvatar stack rejects) writes the packed 584B
        # main-MLA cache when ``kv_cache_dtype == "fp8_ds_mla"``.
        self.compressor: IluDeepseekCompressor | None = None
        if self.compress_ratio > 1:
            self.compressor = IluDeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
            )
            self.compressor.kv_cache_dtype = self.kv_cache_dtype

    def _build_indexer(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        quant_config,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> IluDeepseekV4Fp8Indexer:
        """Build the lightning indexer for a ``compress_ratio == 4`` layer."""
        return IluDeepseekV4Fp8Indexer(
            vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            quant_config=quant_config,
            cache_config=cache_config,
            topk_indices_buffer=self.topk_indices_buffer,
            compress_ratio=self.compress_ratio,
            prefix=prefix,
        )

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Whether this instance stores fp8 KV in the ``fp8_ds_mla`` layout."""
        return self.use_fp8_ds_mla_layout

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # vLLM v0.31.0 (db9527a468) KV-cache layout: allocate_kv_cache binds a 4D
        # [B, H=1, N, C] tensor; squeeze the singleton head dim -> 3D [B, N, C]
        # as required by the ilu ixformer attention kernels (ndim 2 or 3 only).
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Main MLA spec.

        Reads only ``compress_ratio`` / ``head_dim`` / ``kv_cache_dtype`` /
        ``kv_cache_torch_dtype`` so it stays callable against a duck-typed
        ``self`` (see ``test_ilu_dsv4_fp8_kv_spec.py``).
        """
        if self.compress_ratio <= 1:
            # SWA-only layer: its cache is allocated by DeepseekV4SWACache.
            return None
        # ``alignment=64`` (not NV's 576): the ixformer ds4_* kernels require
        # 64B-aligned block bases, matching the sibling W4A8 fp8 path.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=64 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            # 448B NoPE + 128B RoPE + 8B fp8 scale per token; ``head_size``
            # stays semantic (512). v0.30.0 (ced6857afa) takes C from this
            # field rather than from ``cache_dtype_str``.
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse MLA attention forward (raw-FP8 checkpoint, fp8_ds_mla KV).

        Calls ``attention_impl`` directly rather than dispatching through a
        torch custom op.
        """
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        num_tokens = hidden_states.shape[0]
        # No head padding: ``o`` carries exactly ``n_local_heads`` heads.
        o = torch.empty(
            (num_tokens, self.n_local_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self.attention_impl(hidden_states, qr, kv, positions, o)

        return self._o_proj(o, positions)

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Sequential sparse-MLA body.

        No aux-stream overlap: indexer / compressor / kv-insert run in order
        on the default stream.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        qr, kv = ixfops.rms_norm_qk(
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.eps,
        )

        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        if self.indexer is not None:
            # C4A: indexer + main compressor + SWA fp8 insert.
            self.indexer(hidden_states, qr, positions, self.indexer_rotary_emb)
            assert self.compressor is not None
            self.compressor(hidden_states, positions, self.rotary_emb)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        elif self.compressor is not None:
            # C128A: main compressor + SWA fp8 insert (no indexer).
            self.compressor(hidden_states, positions, self.rotary_emb)
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        else:
            # SWA-only: just the fused Q/KV norm+rope+insert.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # Reserve the real prefill peak before workspace locking and KV sizing.
        if not isinstance(attn_metadata, dict):
            if self.use_prefill_sparse:
                N = (
                    0
                    if self.compress_ratio <= 1
                    else (
                        (self.max_model_len + self.compress_ratio - 1)
                        // self.compress_ratio
                    )
                )
                M = N + self.window_size + self.max_num_batched_tokens
                current_workspace_manager().get_simultaneous(
                    ((PREFILL_CHUNK_SIZE, M, self.head_dim), torch.bfloat16),
                )
                # Hold the real additional Q allocation while the KV workspace
                # is reserved so memory profiling sizes the cache correctly.
                _prefill_q = prefill_q_fp8_round_bf16(q, backend="ixformer_native")
            elif not self.use_fp8_ds_mla_layout:
                _plain_q = self._quant_plain_q(q)
            out.zero_()
            return

        # 5) All KV caches are now written -- push to a KV connector if present.
        if has_kv_transfer_group():
            connector = get_kv_transfer_group()
            if connector.has_connector_metadata():
                connector.save_kv_layer(self.prefix, self.kv_cache, attn_metadata)

        # 6) Sparse MLA forward into the pre-allocated ``out`` (T, H, head_dim).
        self.forward_mqa(q, kv, positions, out)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse MLA forward: BF16 sparse prefill and packed FP8 decode.

        Prefill on a wide shard preserves FP8 NoPE Q quantization in a fused
        BF16 materialization. The decode operation block also handles small
        prefills and narrow shards, where it is the faster of the two.

        ``kv`` is unused (the rotated K rows were already written by the
        compressor / ``_fused_qnorm_rope_kv_insert`` step); it is kept for the
        NV signature.

        CUDA-graph / padding contract: ``q`` / ``output`` arrive padded (to the
        DP width in eager/piecewise, or the captured uniform batch under a full
        graph). We trim them to ``swa_metadata.num_actual_tokens`` so the op's
        batch dim matches its ``indices`` / ``topk_length``. Under full-graph
        capture ``num_actual`` equals the padded max, so the slice is a
        full-length no-op and pad rows at replay are neutralised by their zeroed
        ``all_swa_lens``.
        """
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict), (
            "fp8 V4 MLA forward_mqa requires a dict-shaped attn_metadata; got "
            f"{type(attn_metadata).__name__}. Profiling/dummy runs are "
            "short-circuited by attention_impl before reaching here."
        )

        flashmla_metadata = attn_metadata.get(self.prefix)
        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "Missing SWA metadata for prefix "
            f"{self.swa_cache_layer.prefix!r}; the SWA KVCacheGroup must be "
            "registered alongside the main MLA cache."
        )

        swa_only = self.compress_ratio <= 1
        # Real (scheduled) token count; trims padding to match the
        # num_actual-length SWA/extra index buffers.
        num_actual = swa_metadata.num_actual_tokens
        q = q[:num_actual]
        output = output[:num_actual]

        # Avoid gathering long pools for small prefill fragments. Mixed batches
        # retain the paged all-token launch because its reduction geometry
        # depends on the complete token count. Noncausal batches use that path too.
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        min_prefill_tokens = (
            512 if 0 < swa_metadata.max_prefill_seq_len <= 8192 else 1024
        )
        split_prefill = (
            self.use_prefill_sparse
            and swa_metadata.num_decode_tokens == 0
            and num_prefill_tokens
            >= min_prefill_tokens * max(1, swa_metadata.num_prefills)
            and swa_metadata.causal
        )
        if split_prefill:
            num_decode_tokens = swa_metadata.num_decode_tokens
            if num_decode_tokens + num_prefill_tokens != num_actual:
                raise ValueError("FP8 SWA decode/prefill token split is inconsistent")
            self._forward_prefill_fp8(
                q[num_decode_tokens:],
                positions[num_decode_tokens:],
                output[num_decode_tokens:],
                None if swa_only else flashmla_metadata,
                swa_metadata,
            )
            return

        # Contiguous all-token (decode + prefill) SWA slots.
        swa_indices = swa_metadata.all_swa_indices
        swa_lens = swa_metadata.all_swa_lens
        assert swa_indices is not None and swa_lens is not None, (
            "fp8 unified forward_mqa needs all_swa_indices/all_swa_lens; ensure "
            "IluDeepseekV4Fp8SWAMetadataBuilder is the active SWA builder."
        )

        extra_k_cache: torch.Tensor | None = None
        extra_indices: torch.Tensor | None = None
        extra_topk_length: torch.Tensor | None = None
        if not swa_only:
            assert flashmla_metadata is not None, (
                "FlashMLASparseMetadata is required when compress_ratio > 1; "
                f"prefix={self.prefix!r}"
            )
            extra_indices, extra_topk_length = self._build_extra_topk(
                swa_metadata, flashmla_metadata, num_actual
            )
            extra_k_cache = self.kv_cache

        h_q = q.shape[1]
        assert h_q in _SPARSE_DECODE_H_Q, (
            f"ds4_sparse_mla_fp8_decode requires h_q in {_SPARSE_DECODE_H_Q}, got {h_q}"
        )
        indices = _as_sparse_decode_indices(swa_indices)
        extra_idx = (
            None if extra_indices is None else _as_sparse_decode_indices(extra_indices)
        )
        if not self.use_fp8_ds_mla_layout:
            q_fp8, q_scale = self._quant_plain_q(q)
            sparse_mla_fp8_pt(
                q=q_fp8,
                q_scale=q_scale,
                k_cache=self.swa_cache_layer.kv_cache,
                indices=indices,
                output=output,
                sm_scale=self.scale,
                kv_scale=1.0,
                topk_length=swa_lens,
                attn_sink=self.attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_idx,
                extra_topk_length=extra_topk_length,
                backend="ixformer_native",
                mask_invalid_kv=True,
            )
            return
        q_packed, q_scales = ixfops.ds4_dynamic_quant_mla_q_packed_fp8(q)
        ixfops.ds4_sparse_mla_fp8_decode(
            q_packed=q_packed,
            q_scales=q_scales,
            k_cache=self.swa_cache_layer.kv_cache,
            indices=indices,
            sm_scale=self.scale,
            topk_length=swa_lens,
            attn_sink=self.attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices=extra_idx,
            extra_topk_length=extra_topk_length,
            output=output,
        )

    def _gather_k_cache(
        self,
        kv,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    ) -> None:
        # DSA's masked loads use global scratch slot 0. For initial contexts
        # with zero compressed keys it is otherwise uninitialized; initialize
        # before gather, which overwrites the slot when a real key 0 exists.
        if self.compress_ratio > 1 and offset == 0:
            kv[0, 0].zero_()
        IluDeepseekV4MLAAttention._gather_k_cache(
            self,
            kv,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
        )

    def _forward_prefill_fp8(
        self,
        q,
        positions,
        output,
        attn_metadata,
        swa_metadata,
    ) -> None:
        if not swa_metadata.causal:
            raise ValueError("Sparse BF16 prefill requires causal metadata")
        if self.compress_ratio > 1 and attn_metadata is None:
            raise ValueError("Compressed FP8 prefill requires main MLA metadata")
        # Preserve the old FP8 NoPE Q quantization before BF16 sparse attention.
        # RoPE stays unchanged; the helper consumes local C4/C128 topk directly.
        q = prefill_q_fp8_round_bf16(q, backend="ixformer_native")
        IluDeepseekV4MLAAttention._forward_prefill(
            self,
            q=q,
            positions=positions,
            compressed_k_cache=self.kv_cache if self.compress_ratio > 1 else None,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            output=output,
            attn_metadata=attn_metadata,
            swa_metadata=swa_metadata,
            # Pinned rather than left to the registry default (ixinfer), whose
            # rationale is an ivcore11 measurement. On ivcore30 the two are
            # within 5% for every bucket this path runs (h_q >= 32), and the
            # registry default can move without this path noticing.
            prefill_backend="ixformer_perf",
        )

    def _quant_plain_q(self, q):
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        q_scale = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)
        workspace = torch.empty(256, dtype=torch.float32, device=q.device)
        return quant_mla_q_fp8_pt(
            q, q_fp8, q_scale, workspace, backend="ixformer_native"
        )

    def _build_extra_topk(
        self,
        swa_metadata,
        attn_metadata: IluFlashMLASparseMetadata,
        num_actual: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build global compressed top-k indices/lens for the real tokens.

        Returns ``(extra_indices [num_actual, 1, topk] int32, extra_topk_length
        [num_actual] int32)``; ``forward_mqa`` squeezes extra_indices to 2D
        for ``ds4_sparse_mla_fp8_decode``.

        - **C4A**: globalize the indexer's per-token local slots over the whole
          ``[0:num_actual]`` range in one call.
        - **C128A**: the fp8 builder pre-merges the all-token global top-k once
          per step, so here we just slice it back.
        """
        if self.compress_ratio == 128:
            # Step-level merged global top-k; layer-invariant, so read directly.
            global_indices = attn_metadata.c128a_global_topk_indices
            topk_lens = attn_metadata.c128a_topk_lens
            assert global_indices is not None and topk_lens is not None
            return (
                global_indices[:num_actual].unsqueeze(1),
                topk_lens[:num_actual],
            )

        assert swa_metadata.is_valid_token is not None
        assert swa_metadata.token_to_req_indices is not None
        is_valid = swa_metadata.is_valid_token[:num_actual]
        token_to_req = swa_metadata.token_to_req_indices[:num_actual]

        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            global_indices, topk_lens = ixfops.compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_actual],
                token_to_req,
                attn_metadata.block_table,
                block_size,
                is_valid,
            )
            return global_indices.view(num_actual, 1, -1), topk_lens

        raise ValueError(
            f"Unsupported compress_ratio={self.compress_ratio}; expected 1, 4, or 128."
        )

    def _require_bmm_weights(self) -> None:
        """Check the grouped fp8 views ``_o_proj`` needs actually got attached.

        Only ``IluFp8BlockScaledMMKernel.process_weights_after_loading`` attaches
        them, and only when it sees ``is_bmm`` with a non-empty ``bmm_geom``. A
        different arch, linear backend or LinearMethod leaves them off, and the
        bare attribute access below would surface as an AttributeError deep in
        the first forward with no hint about kernel selection.
        """
        missing = [
            name
            for name in ("bmm_b_fp8", "bmm_bs")
            if getattr(self.wo_a, name, None) is None
        ]
        if missing:
            raise RuntimeError(
                f"{self.prefix}.wo_a is missing {missing}, so the fused fp8 "
                "o-projection cannot run. These are attached by "
                "IluFp8BlockScaledMMKernel.process_weights_after_loading; make "
                "sure it is the active linear kernel (ivcore30 + "
                "VLLM_ILU_LINEAR_BACKEND selecting the block-scaled fp8 path)."
            )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Fused inverse-RoPE + fp8 grouped ``wo_a`` + row-parallel ``wo_b``.

        ``fused_inv_rope_fp8_quant`` fuses the inverse RoPE of ``o`` with a
        per-(group, token) fp8 dynamic quant (activation ``a`` [G, T, R] fp8 +
        scale ``as_scale`` [G, T] fp32); ``fp8_per_channel_einsum`` contracts it
        with the load-time per-channel fp8 weight into ``z`` [T, G, D] bf16.
        """
        self._require_bmm_weights()
        a, as_scale = ixfops.fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.n_local_groups,
        )
        z = ixfops.fp8_per_channel_einsum(
            a, self.wo_a.bmm_b_fp8, as_scale, self.wo_a.bmm_bs
        )
        return self.wo_b(z.flatten(1))

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        """Fused per-head q RMSNorm + q/kv RoPE + fp8 KV paged insert (SWA).

        ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` writes the 584B packed SWA cache:
        q gets weightless per-head RMSNorm + GPT-J RoPE, kv NoPE -> fp8 E4M3 +
        UE8M0 scale and RoPE -> bf16. Tokens with ``slot_mapping[t] < 0`` are
        skipped. ``q`` and the SWA cache are mutated in place.
        """
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        assert swa_metadata is not None, (
            "fp8 V4 attention_impl: missing SWA metadata for prefix "
            f"{self.swa_cache_layer.prefix!r}; check that the SWA KVCacheGroup "
            "is registered alongside the main MLA cache."
        )

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # 3D uint8 cache passed directly: the fp8 insert kernel uses stride(0)
        # and must not flatten padded (non-contiguous) blocks.
        ixfops.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
