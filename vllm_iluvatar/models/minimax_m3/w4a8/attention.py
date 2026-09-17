# SPDX-License-Identifier: Apache-2.0
"""Iluvatar MiniMax-M3 W4A8 attention layers (dense + block-sparse).

Self-contained wiring: local indexer/sparse backends, Gemma norm, and fused
QK-norm/RoPE/KV insert via ixformer (same pattern as DeepSeek-V4).

BF16 / mxfp8 attention lives in ``minimax_m3.mxfp8.attention``.
"""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_iluvatar.models.minimax_m3.w4a8.indexer import (
    IluMiniMaxM3Indexer,
)
from vllm_iluvatar.models.minimax_m3.w4a8.sparse_attention import (
    IluMiniMaxM3SparseBackend,
    IluMiniMaxM3SparseImpl,
    resolve_sparse_attention_cfg,
)


class IluMiniMaxM3Attention(nn.Module):
    """Dense attention with per-head QK norm and partial RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} must be "
                f"divisible by tensor_parallel_size={tp_size}"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads={self.total_num_kv_heads} must be "
                    f"divisible by tensor_parallel_size={tp_size}"
                )
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError(
                f"tensor_parallel_size={tp_size} must be divisible by "
                f"num_key_value_heads={self.total_num_kv_heads}"
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via model._fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # Fused per-head Gemma QK-norm + partial NeoX RoPE. Writes Q/K back
        # into ``qkv`` in place (ixformer fused op contract).
        ixfops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class IluMiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention layer with the lightning-indexer branch.

    This is a merged attention layer: it owns the projections (qkv + index
    q/k), per-head QK norms and RoPE, *and* the attention-backend wiring that a
    generic ``Attention`` layer would normally provide — it binds the
    ``IluMiniMaxM3SparseBackend`` + main impl, registers the main paged K/V cache,
    and owns the lightning indexer (``IluMiniMaxM3Indexer``), which holds the
    index-key side cache.

    The index branch (index_{q,k}_proj + index_{q,k}_norm) feeds the sparse
    top-k block selection. M3 always disables the index value/output
    projections (``sparse_disable_index_value`` set for every sparse layer), so
    ``index_{v,o}_proj`` are never created.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} must be "
                f"divisible by tensor_parallel_size={tp_size}"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads={self.total_num_kv_heads} must be "
                    f"divisible by tensor_parallel_size={tp_size}"
                )
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError(
                f"tensor_parallel_size={tp_size} must be divisible by "
                f"num_key_value_heads={self.total_num_kv_heads}"
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Sparse "index" branch dims. index_q has the same head count as the KV
        # heads (sparse_num_index_heads == num_key_value_heads), so it shards
        # identically -- including replication when tp_size > num_key_value_heads.
        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        if self.total_idx_heads != self.total_num_kv_heads:
            raise ValueError(
                "MiniMax-M3 fused QKV+index projection requires "
                f"sparse_num_index_heads={self.total_idx_heads} == "
                f"num_key_value_heads={self.total_num_kv_heads}"
            )
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim

        # Single fused projection: q, k, v, index_q, index_k in one GEMM.
        # q/k/v/index_q/index_k are pre-quantized INT8 checkpoint shards and
        # loaded directly into the fused W8A8 projection.
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via model._fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)

        # Attention-backend wiring.
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # Indexer side-cache dtype, mirroring --kv-cache-dtype for the main
        # cache (--attention-config '{"indexer_kv_dtype": ...}').
        self.indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype

        self.attn_backend = IluMiniMaxM3SparseBackend
        sparse_resolved = resolve_sparse_attention_cfg(config)
        topk_blocks = sparse_resolved["topk_blocks"]
        init_blocks = sparse_resolved["init_blocks"]
        local_blocks = sparse_resolved["local_blocks"]
        # Single Iluvatar sparse impl (ixformer registry selects backend).
        self.impl: IluMiniMaxM3SparseImpl = IluMiniMaxM3SparseImpl(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_resolved["block_size"],
        )
        if topk_indices_buffer is None:
            raise ValueError(
                "MiniMax-M3 sparse attention requires a shared "
                "topk_indices_buffer from the model"
            )
        self.indexer = IluMiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_resolved["block_size"],
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            cache_config=cache_config,
            indexer_kv_dtype=self.indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
        )

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache
        # Page size must equal sparse block size (one selected block == one page).
        self.sparse_block_size = sparse_resolved["block_size"]

    def get_attn_backend(self) -> type[IluMiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Main GQA K/V cache. Page size must equal sparse block size.
        block_size = vllm_config.cache_config.block_size
        if block_size != self.sparse_block_size:
            raise ValueError(
                f"MiniMax-M3 requires cache block_size == "
                f"sparse_block_size ({self.sparse_block_size}), got {block_size}. "
                f"Start the engine with --block-size {self.sparse_block_size}."
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse attention forward (aligned with Ilu DeepSeek-V4).

        1. ``qkv = qkv_proj(hidden_states)``.
        2. ``attn_out = empty(T, q_size)`` like DSV4's ``o``; the custom op
           writes fused insert + indexer + sparse attend into it.
        3. ``o_proj(attn_out)`` (outside the custom op, same as V4 ``wo_*``).
        """
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]
        attn_out = qkv.new_empty((num_tokens, self.q_size))
        torch.ops.vllm.ilu_minimax_m3_sparse_attention(
            qkv,
            positions,
            attn_out,
            self.layer_name,
        )
        proj_out, _ = self.o_proj(attn_out)
        return proj_out

    def attention_impl(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Fused insert + indexer + sparse attend; writes into ``out``.

        ``q`` / ``index_q`` are per-call ``empty`` (same as DSV4 ``o``). Dummy
        / profiling still allocates them, reserves TCU scratch, zeros ``out``,
        and returns — fused insert needs a bound NHD KV cache.
        """
        num_tokens = qkv.shape[0]
        q = qkv.new_empty((num_tokens, self.q_size))
        index_q = qkv.new_empty((num_tokens, self.index_q_size))

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        if self.kv_cache.numel() == 0 or not isinstance(attn_metadata, dict):
            self.impl.reserve_tcu_workspace(qkv)
            out.zero_()
            return

        fwd_slot_mapping = fwd_ctx.slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            raise RuntimeError(
                "MiniMax-M3 sparse attention is missing slot_mapping for "
                f"{self.layer_name!r} after KV cache bind"
            )
        main_slot_mapping = fwd_slot_mapping[self.layer_name]
        index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]

        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon

        if self.kv_cache.dim() != 5:
            raise RuntimeError(
                f"Unexpected main KV cache rank {self.kv_cache.dim()}, "
                f"shape={tuple(self.kv_cache.shape)}"
            )
        # Native sparse kernels require NHD [B, 2, block, H, D].
        if self.kv_cache.shape[2] != self.sparse_block_size:
            raise RuntimeError(
                f"Main KV cache shape {tuple(self.kv_cache.shape)} is not NHD "
                f"with block_size={self.sparse_block_size} "
                f"(HND is not supported by Iluvatar MiniMax-M3 sparse kernels)"
            )
        page_size = int(self.kv_cache.shape[2])

        ixfops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            rotary_dim,
            eps,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            main_slot_mapping,
            index_slot_mapping,
            self.kv_cache,
            self.indexer.index_cache.kv_cache,
            page_size,
            q,
            index_q,
            self.kv_cache_dtype,
        )

        self.indexer(index_q)
        self.impl.forward(self, q, self.kv_cache, out)


# ---------------------------------------------------------------------------
# Custom op: vllm::ilu_minimax_m3_sparse_attention
# ---------------------------------------------------------------------------
# Ilu-fork analogue of ``vllm::ilu_deepseek_v4_attention``. Opaque boundary for
# fused insert + indexer + sparse attend so torch.compile / Dynamo see a single
# mutating op. Layer lookup via ``forward_context.no_compile_layers`` (filled
# from ``compilation_config.static_forward_context`` in ``__init__``).
#
# ``mutates_args`` lists ``out`` (attend) and ``qkv`` (fused QK-norm/RoPE
# writes Q/K/index_q/index_k in place in the packed row, and also copies
# Q / index_q into separate temps).


def ilu_minimax_m3_sparse_attention(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom-op trampoline for the Ilu MiniMax-M3 sparse attention path."""
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(qkv, positions, out)


def ilu_minimax_m3_sparse_attention_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake/meta impl: ``out`` and ``qkv`` are in ``mutates_args``."""
    return None


direct_register_custom_op(
    op_name="ilu_minimax_m3_sparse_attention",
    op_func=ilu_minimax_m3_sparse_attention,
    mutates_args=["out", "qkv"],
    fake_impl=ilu_minimax_m3_sparse_attention_fake,
)
