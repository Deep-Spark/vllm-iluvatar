# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Upstream reference: vLLM v0.27.1 (6e448d0),
# vllm/models/minimax_m3/nvidia/model.py.
# Modified by Shanghai Iluvatar CoreX Semiconductor Co., Ltd. for the ixaite backend.
"""MiniMax-M3 attention for the ixaite backend.

Dense and sparse layers construct projections / norms / RoPE. Sparse attend
and fused QK-norm/RoPE dispatch to ixaite FlyDSL (no ixformer / NVIDIA MSA).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_iluvatar.backends.ixaite.models.minimax_m3.indexer import (
    IxaiteMiniMaxM3Indexer,
)
from vllm_iluvatar.backends.ixaite.models.minimax_m3.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseIxaiteImpl,
)

# Re-export for registration / gate tests.
__all__ = (
    "MiniMAXGemmaRMSNorm",
    "MiniMaxM3Attention",
    "MiniMaxM3SparseAttention",
    "MiniMaxM3SparseBackend",
    "MiniMaxM3SparseIxaiteImpl",
    "fused_minimax_m3_qknorm_rope_kv_insert",
)

# GemmaRMSNorm is OOT-replaced by IluGemmaRMSNorm under the ixaite provider.
MiniMAXGemmaRMSNorm = GemmaRMSNorm

SUPPORTED_KV_CACHE_DTYPES = ("bfloat16",)
SUPPORTED_KV_CACHE_TORCH_DTYPES = (torch.bfloat16,)
# vLLM AttentionConfig may spell BF16 as "bf16" (alias of "bfloat16").
SUPPORTED_INDEXER_KV_DTYPES = ("bfloat16", "bf16", "auto")

logger = init_logger(__name__)


def _gemma_unit_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (weight.data.to(torch.float32) + 1.0).to(dtype=dtype).contiguous()


def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    epsilon: float,
    kv_cache_dtype: str = "auto",
) -> None:
    """In-place Gemma QK RMSNorm + NeoX RoPE on the Q/K slices of ``qkv``.

    KV cache write stays with the Attention / sparse layer; this helper only
    replaces the host QK prep so MiniMax never depends on ixformer fused insert.
    """
    del kv_cache_dtype
    if not torch.cuda.is_available():
        raise RuntimeError("ixaite MiniMax fused QK-norm/RoPE requires CUDA/Iluvatar")
    import ixaite

    head_dim = int(q_weight.shape[0])
    q_size = int(num_heads) * head_dim
    kv_size = int(num_kv_heads) * head_dim
    q = qkv[..., :q_size].contiguous()
    k = qkv[..., q_size : q_size + kv_size].contiguous()
    qw = _gemma_unit_weight(q_weight, q.dtype)
    kw = _gemma_unit_weight(k_weight, q.dtype)
    q_out, k_out = ixaite.fused_qk_rmsnorm_rope(
        q,
        k,
        qw,
        kw,
        cos_sin_cache,
        positions,
        float(epsilon),
        int(num_heads),
        int(num_kv_heads),
        head_dim,
        int(rotary_dim),
    )
    qkv[..., :q_size].copy_(q_out)
    qkv[..., q_size : q_size + kv_size].copy_(k_out)


def _require_unquantized_m3_caches(
    kv_cache_dtype: str,
    kv_cache_torch_dtype: torch.dtype,
    indexer_kv_dtype: str,
) -> None:
    if kv_cache_dtype not in ("auto", *SUPPORTED_KV_CACHE_DTYPES):
        raise NotImplementedError(
            "ixaite MiniMax-M3 sparse attention requires kv_cache_dtype in "
            f"{('auto', *SUPPORTED_KV_CACHE_DTYPES)}, got {kv_cache_dtype!r}."
        )
    if kv_cache_torch_dtype not in SUPPORTED_KV_CACHE_TORCH_DTYPES:
        raise NotImplementedError(
            "ixaite MiniMax-M3 sparse attention requires a KV cache in "
            f"{sorted(str(d) for d in SUPPORTED_KV_CACHE_TORCH_DTYPES)}; "
            f"kv_cache_dtype={kv_cache_dtype!r} resolved to "
            f"{kv_cache_torch_dtype}."
        )
    if indexer_kv_dtype not in SUPPORTED_INDEXER_KV_DTYPES:
        raise NotImplementedError(
            "ixaite MiniMax-M3 indexer requires indexer_kv_dtype in "
            f"{SUPPORTED_INDEXER_KV_DTYPES}, got {indexer_kv_dtype!r}."
        )


class MiniMaxM3Attention(nn.Module):
    """Dense attention with fused QK normalization and cache insertion."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        del layer_id
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
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
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
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
        fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            kv_cache_dtype="auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


def _insert_nhd_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter ``[T, Hkv, D]`` into MiniMax NHD ``[blocks, 2, page, Hkv, D]``."""
    if kv_cache.numel() == 0 or k.numel() == 0:
        return
    if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
        raise RuntimeError(
            "ixaite MiniMax sparse expects NHD KV "
            f"[blocks, 2, page, Hkv, D], got {tuple(kv_cache.shape)}"
        )
    page = int(kv_cache.shape[2])
    # Graph-safe: no host sync and no data-dependent boolean gather (both are
    # illegal under CUDA stream capture). PAD_SLOT_ID (-1) clamps to 0; callers
    # pass a length-T mapping of real tokens, so pad only appears in capture /
    # profile dummies where corrupting block 0 is harmless.
    slots = slot_mapping.long().clamp(min=0)
    blocks = torch.div(slots, page, rounding_mode="floor")
    offs = slots % page
    kv_cache[blocks, 0, offs] = k
    kv_cache[blocks, 1, offs] = v


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Sparse attention with ixaite FlyDSL decode/prefill(q=1) attend."""

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
        del layer_id
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self.sparse_block_size = int(sparse_cfg["sparse_block_size"])

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
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        self.index_q_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )

        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.calculate_kv_scales = False
        set_default_quant_scales(self, register_buffer=True)
        self.indexer_kv_dtype = vllm_config.attention_config.indexer_kv_dtype
        _require_unquantized_m3_caches(
            self.kv_cache_dtype,
            self.kv_cache_torch_dtype,
            self.indexer_kv_dtype,
        )

        self.topk_indices_buffer = topk_indices_buffer
        self.attn_backend = MiniMaxM3SparseBackend
        self.impl = MiniMaxM3SparseIxaiteImpl(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=int(sparse_cfg["sparse_topk_blocks"]),
            sparse_block_size=self.sparse_block_size,
        )
        self.impl.init_blocks = int(sparse_cfg.get("sparse_init_block", 0))
        self.impl.local_blocks = int(sparse_cfg.get("sparse_local_block", 1))
        self.impl.idx_scale = float(self.scaling)
        # Community MiniMaxM3IndexerCache rejects "auto"; treat it as bf16.
        indexer_cache_dtype = (
            "bf16"
            if self.indexer_kv_dtype in ("auto", "bfloat16")
            else self.indexer_kv_dtype
        )
        self.indexer = IxaiteMiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=int(sparse_cfg["sparse_topk_blocks"]),
            sparse_block_size=self.sparse_block_size,
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=int(sparse_cfg.get("sparse_init_block", 0)),
            local_blocks=int(sparse_cfg.get("sparse_local_block", 1)),
            score_type=str(sparse_cfg.get("sparse_score_type", "max")),
            cache_config=cache_config,
            indexer_kv_dtype=indexer_cache_dtype,
            topk_indices_buffer=topk_indices_buffer,
        )
        logger.info_once(
            "[IluMiniMaxSparse] layer=%s backend=%s block=%s topk=%s",
            self.layer_name,
            MiniMaxM3SparseBackend.get_name(),
            sparse_cfg["sparse_block_size"],
            sparse_cfg["sparse_topk_blocks"],
        )

        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
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
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound.
            return qkv.new_zeros((num_tokens, self.hidden_size))

        main_slot_mapping = fwd_slot_mapping[self.layer_name]
        index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]

        # Packed row: [q | k | v | index_q | index_k].
        q_end = self.q_size
        k_end = q_end + self.kv_size
        v_end = k_end + self.kv_size
        iq_end = v_end + self.index_q_size
        q = qkv[:, :q_end].contiguous()
        k = qkv[:, q_end:k_end].contiguous()
        v = qkv[:, k_end:v_end].contiguous()
        index_q = qkv[:, v_end:iq_end].contiguous()
        index_k = qkv[:, iq_end : iq_end + self.idx_head_dim].contiguous()

        fused_minimax_m3_qknorm_rope_kv_insert(
            qkv[:, :v_end],
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        q = qkv[:, :q_end].contiguous()
        k = qkv[:, q_end:k_end].contiguous()
        v = qkv[:, k_end:v_end].contiguous()

        # Index branch: Gemma RMSNorm + NeoX RoPE (same rotary as main).
        index_q = self.index_q_norm(index_q.view(-1, self.idx_head_dim)).view(
            num_tokens, self.index_q_size
        )
        index_k = self.index_k_norm(index_k)
        # RoPE expects [T, H, D]; index_k is a single head.
        iq_heads = index_q.view(num_tokens, self.num_idx_heads, self.idx_head_dim)
        ik_heads = index_k.view(num_tokens, 1, self.idx_head_dim)
        iq_heads, ik_heads = self.rotary_emb(positions, iq_heads, ik_heads)
        index_q = iq_heads.reshape(num_tokens, self.index_q_size)
        index_k = ik_heads.reshape(num_tokens, self.idx_head_dim)

        _insert_nhd_kv(
            k.view(num_tokens, self.num_kv_heads, self.head_dim),
            v.view(num_tokens, self.num_kv_heads, self.head_dim),
            self.kv_cache,
            main_slot_mapping,
        )
        self.indexer.insert_keys(index_k, index_slot_mapping)

        output = torch.empty_like(q)
        # Stash index tensors so multi-q sparse prefill can recompute causal
        # top-k without a second host gather (SGLang-aligned path).
        self._ilu_idx_q = iq_heads
        self._ilu_idx_k_cache = self.indexer.index_cache.kv_cache
        self.indexer(index_q)
        self.impl.forward(self, q, self.kv_cache, output)
        proj_out, _ = self.o_proj(output)
        return proj_out
