# SPDX-License-Identifier: Apache-2.0
"""GLM5Next projections and complete Iluvatar Kpool indexing."""

import os

import ixformer.inference.functions as ixfops
import torch
import triton
import triton.language as tl
import vllm.envs as envs
from ixformer.inference.functions.models.glm5next import (
    fwht128_quant_int8_with_weights,
)
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
from vllm.models.glm5next.sparse_indexer import SparseAttnIndexerKpool
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.utils.deep_gemm import PAGED_MQA_PAGE_SIZES
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.kv_cache_interface import KpoolTailSpec, MLAAttentionSpec

from vllm_iluvatar.ilu_util.arch import is_ivcore30
from vllm_iluvatar.model_executor.layers.quantization.fp8 import IluFp8Config


def resolve_glm5next_indexer_kv_dtype(
    quant_config: QuantizationConfig | None = None,
) -> str:
    if "VLLM_ILU_INDEXER_KV_DTYPE" in os.environ:
        return envs.VLLM_ILU_INDEXER_KV_DTYPE
    if is_ivcore30() and isinstance(quant_config, IluFp8Config):
        return "fp8"
    return "bf16"


@triton.jit
def _scale_cast_bf16_kernel(
    in_ptr,
    out_ptr,
    scale,
    T,
    H,
    in_s0,
    in_s1,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_H)
    mask = col < H
    x = tl.load(in_ptr + row * in_s0 + col * in_s1, mask=mask, other=0.0)
    tl.store(out_ptr + row * H + col, (x * scale).to(tl.bfloat16), mask=mask)


def _scale_to_bf16(weights: torch.Tensor, scale: float) -> torch.Tensor:
    T, H = weights.shape
    out = torch.empty((T, H), dtype=torch.bfloat16, device=weights.device)
    _scale_cast_bf16_kernel[(T,)](
        weights,
        out,
        scale,
        T,
        H,
        weights.stride(0),
        weights.stride(1),
        BLOCK_H=next_power_of_2(H),
    )
    return out


def glm5next_uses_compact_pool_indices(vllm_config: VllmConfig, kpool: int) -> bool:
    return (
        kpool == 4
        and vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        and vllm_config.model_config.dtype == torch.bfloat16
        and getattr(torch, "corex_arch", None) == "ivcore11"
    )


def glm5next_topk_buffer_width(
    vllm_config: VllmConfig, topk_tokens: int, kpool: int
) -> int:
    if glm5next_uses_compact_pool_indices(vllm_config, kpool):
        return topk_tokens // kpool
    width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
    return (width + 127) // 128 * 128


class IluGlm5NextIndexerCache(DeepseekV32IndexerCache):
    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
        scale_fmt: str,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "IluGlm5NextIndexerCache expects index_kpool > 1"
        assert cache_config.block_size % index_kpool == 0, (
            "IluGlm5NextIndexerCache: cache_config.block_size "
            f"({cache_config.block_size}) must be a multiple of index_kpool "
            f"({index_kpool}) so chunked-prefill boundaries stay pool-aligned."
        )
        self._index_kpool = index_kpool
        self._scale_fmt = scale_fmt

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from dataclasses import replace

        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        spec = replace(spec, tokens_per_state=self._index_kpool)

        storage_block_size = spec.block_size // self._index_kpool
        assert (
            spec.block_size % self._index_kpool == 0 and storage_block_size % 32 == 0
        ), (
            "IluGlm5NextIndexerCache: kpool indexer requires cache block_size to "
            f"be a multiple of index_kpool * 32 ({self._index_kpool * 32}) so "
            "that DeepGEMM paged-MQA pool pages (32 or 64 entries) tile the "
            f"storage block, got block_size={spec.block_size} -> "
            f"storage_block_size={storage_block_size}."
        )
        max_page_size = max(PAGED_MQA_PAGE_SIZES)
        min_page_size = min(PAGED_MQA_PAGE_SIZES)
        if storage_block_size <= max_page_size:
            page_size = storage_block_size
        elif storage_block_size % max_page_size == 0:
            page_size = max_page_size
        else:
            page_size = min_page_size
        assert self._scale_fmt == "None" or page_size in (64, 256), (
            "IluGlm5NextIndexerCache: the quantized index cache needs a pool page of "
            f"64 or 256, got {page_size} from block_size={spec.block_size} and "
            f"index_kpool={self._index_kpool}. Launch with --block-size "
            f"{64 * self._index_kpool} (or any multiple of it)."
        )
        return replace(
            spec,
            storage_block_size=page_size * self._index_kpool,
        )

    def get_attn_backend(self):
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            IluDeepseekV32IndexerBackend,
        )

        return IluDeepseekV32IndexerBackend


class IluGlm5NextTailCache(DeepseekV32IndexerCache):
    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(
            head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config
        )
        assert index_kpool > 1, "IluGlm5NextTailCache expects index_kpool > 1"
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        span = self._index_kpool + vllm_config.num_speculative_tokens
        ring = self._index_kpool * next_power_of_2(cdiv(span, self._index_kpool))
        assert self.cache_config.block_size % ring == 0, (
            f"IluGlm5NextTailCache: cache_config.block_size "
            f"({self.cache_config.block_size}) must be a multiple of the "
            f"tail ring ({ring})"
        )
        return KpoolTailSpec(
            block_size=ring,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=ring,
        )

    def get_attn_backend(self):
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            IluKpoolTailBackend,
        )

        return IluKpoolTailBackend


class IluIndexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank

        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16)
        )

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.quant_block_size = 128
        indexer_kv_dtype = resolve_glm5next_indexer_kv_dtype(quant_config)
        if indexer_kv_dtype == "bf16":
            self.k_cache_dtype = torch.bfloat16
            self.k_cache_head_dim = self.head_dim
            self.scale_fmt = "None"
        elif indexer_kv_dtype == "int8":
            self.k_cache_dtype = torch.uint8
            self.k_cache_head_dim = self.head_dim + 4
            self.scale_fmt = "float32"
        else:
            self.k_cache_dtype = torch.uint8
            self.k_cache_head_dim = (
                self.head_dim + self.head_dim // self.quant_block_size * 4
            )
            self.scale_fmt = "ue8m0"
        self.topk_indices_buffer = topk_indices_buffer
        self._wp_fp32: torch.Tensor | None = None
        self._k_norm_cast: tuple[torch.Tensor, torch.Tensor] | None = None

        self.k_cache = IluGlm5NextIndexerCache(
            head_dim=self.k_cache_head_dim,
            dtype=self.k_cache_dtype,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
            scale_fmt=self.scale_fmt,
        )
        self.tail_cache = IluGlm5NextTailCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.tail_cache",
            cache_config=cache_config,
            index_kpool=self.index_kpool,
        )
        self.max_pool_len = vllm_config.model_config.max_model_len // self.index_kpool
        self.prefix = prefix
        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.compact_pool_indices = glm5next_uses_compact_pool_indices(
            vllm_config, self.index_kpool
        )
        self.indexer_op = SparseAttnIndexerKpool(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_pool_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            tail_cache=self.tail_cache,
        )
        self.indexer_op.compact_pool_indices = self.compact_pool_indices

    def _fused_indexer_k_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dim: int,
        eps: float,
    ) -> torch.Tensor:
        if self._k_norm_cast is None or self._k_norm_cast[0].dtype != x.dtype:
            self._k_norm_cast = (weight.to(x.dtype), bias.to(x.dtype))
        cast_weight, cast_bias = self._k_norm_cast
        return ixfops.layer_norm(x, [dim], cast_weight, cast_bias, eps)

    def _fused_indexer_weight_scale(
        self, weights: torch.Tensor, q_scale: torch.Tensor, scale: float
    ) -> torch.Tensor:
        return (weights.unsqueeze(-1) * q_scale * scale).squeeze(-1)

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        kw = ixfops.linear(
            hidden_states,
            self.wk_weights_proj.weight,
        )
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self._fused_indexer_k_norm(
            k, self.k_norm.weight, self.k_norm.bias, self.head_dim, self.k_norm.eps
        )
        if self.rope_dim > 0:
            q, k = rotary_emb(positions, q, k.unsqueeze(1))

        assert self.head_dim == 128 and self.quant_block_size == 128
        weight_scale = self.softmax_scale * self.n_head**-0.5
        if self.scale_fmt == "ue8m0":
            q = q.view(-1, self.head_dim)
            q_fp8, q_scale = fwht128_quant_fp8(q)
            q = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head, 1)
            weights = self._fused_indexer_weight_scale(weights, q_scale, weight_scale)
        elif self.scale_fmt == "float32":
            q, weights = fwht128_quant_int8_with_weights(q, weights, weight_scale)
        else:
            weights = _scale_to_bf16(weights, weight_scale)

        gate_score = ixfops.linear(hidden_states, self.index_kpool_compress_gate)

        return self.indexer_op(
            hidden_states,
            q,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )


class IluGlm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        skip_rope: bool | None = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if not skip_rope:
            assert config.rope_parameters is not None
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_llama_scaling"
                    if config.rope_parameters.get("attention_factor") == 1.0
                    else "deepseek_yarn"
                )

            self.rotary_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )

            if (
                config.rope_parameters["rope_type"] != "default"
                and config.rope_parameters["rope_type"] == "deepseek_yarn"
            ):
                mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
                scaling_factor = config.rope_parameters["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        self.is_v32 = config.index_topk is not None

        if self.is_v32:
            self.indexer_rope_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not config.indexer_rope_interleave,
            )
            assert q_lora_rank is not None
            self.indexer: IluIndexer | None = IluIndexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            None,
            prefix,
            skip_topk=False,
            fuse_qkv_rmsnorm=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)
