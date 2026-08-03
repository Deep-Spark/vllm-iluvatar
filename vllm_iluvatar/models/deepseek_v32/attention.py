# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar DeepSeek V3.2 sparse MLA attention."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm import _custom_ops as ops
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_v2 import yarn_get_mscale
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_iluvatar.model_executor.layers.sparse_attn_indexer import (
    IluSparseAttnIndexer,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse import (
    IluFlashMLASparseBackend,
    IluFlashMLASparseMetadata,
)
from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
    IluDeepseekV32IndexerBackend,
)


class IluDeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    """KV-cache owner for the DeepSeek V3.2 sparse indexer."""

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str="origin",
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluDeepseekV32IndexerBackend


class IluDeepseekV32Indexer(nn.Module):
    """Iluvatar sparse-attention indexer for DeepSeek V3.2."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
        is_inplace_rope: bool = False,
    ):
        super().__init__()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

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
        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        assert cache_config is not None, "DeepSeek V3.2 indexer requires cache_config"
        self.k_cache = IluDeepseekV32IndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm_iluvatar.v1.attention.backends.mla.ilu_indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = IluSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )
        self.is_inplace_rope = is_inplace_rope
        self.n_head_scale = self.n_head**-0.5
        self.use_fused_indexer_q = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        weights = weights.unsqueeze(-1) * self.softmax_scale * self.n_head_scale
        return self.indexer_op(hidden_states, q, k, weights.squeeze(-1))


class IluDeepseekV32SparseMLA(nn.Module, AttentionLayerBase):
    """Self-contained sparse MLA inner attention for DeepSeek V3.2.

    This is the V3.2 counterpart of ``glm_moe_dsa``'s self-contained sparse
    MLA, but uses a BF16 absorbed ``kv_b_proj`` split and ``torch.bmm`` instead
    of the GLM W8A16 marlin path. It owns KV-cache registration and calls the
    sparse FlashMLA kernel directly, so the runtime path does not go through
    community MLA wrappers.
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        rotary_emb: nn.Module,
        topk_indices_buffer: torch.Tensor | None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads  # local (TP-sharded) head count
        self.num_kv_heads = 1
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        # MLA latent K/V share a single 576-wide vector (kv_lora + rope).
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.softmax_scale = float(scale)
        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb
        self.topk_indices_buffer = topk_indices_buffer
        self.prefix = prefix

        # ilu fork: BF16 KV cache. Reject explicit fp8 selection loudly.
        kv_cache_dtype = (
            cache_config.cache_dtype
            if cache_config is not None and cache_config.cache_dtype not in (None, "")
            else "auto"
        )
        if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
            raise ValueError(
                "IluGlmMoeDsaSparseMLA does not support fp8 KV cache; "
                f"got kv_cache_dtype={kv_cache_dtype!r}. Use --kv-cache-dtype "
                "bfloat16."
            )
        self.kv_cache_dtype = kv_cache_dtype

        # BF16 cache -> concat_and_cache_mla scale is a no-op 1.0.
        self.register_buffer(
            "_k_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False
        )

        self.kv_cache = torch.tensor([])
        self.W_UK_T: torch.Tensor | None = None
        self.W_UV: torch.Tensor | None = None
        self._vllm_config = get_current_vllm_config()
        compilation_config = self._vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return IluFlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )

    def _prepare_bf16_up_proj_weights(self, act_dtype: torch.dtype) -> None:
        if self.W_UK_T is not None and self.W_UV is not None:
            return
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, {self.kv_lora_rank=}, "
            f"{self.num_heads=}, {self.qk_nope_head_dim=}, {self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        self.W_UV = W_UV.transpose(0, 1).contiguous()

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self._prepare_bf16_up_proj_weights(act_dtype)

    def _k_up_proj(self, q_nope: torch.Tensor) -> torch.Tensor:
        self._prepare_bf16_up_proj_weights(q_nope.dtype)
        assert self.W_UK_T is not None
        q_t = q_nope.transpose(0, 1)
        return torch.bmm(q_t, self.W_UK_T).transpose(0, 1)

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        self._prepare_bf16_up_proj_weights(x.dtype)
        assert self.W_UV is not None
        x_t = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        return torch.bmm(x_t, self.W_UV).transpose(0, 1)

    @staticmethod
    def flash_mla_sparse_prefill(
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse attention prefill kernel

        Args:
        - q: [s_q, h_q, d_qk], bfloat16
        - kv: [s_kv, h_kv, d_qk], bfloat16
        - indices: [s_q, h_kv, topk], int32.
            Invalid indices should be set to -1 or numbers >= s_kv
        - sm_scale: float
        - d_v: The dimension of value vectors. Can only be 512

        Returns:
        - (output, max_logits, lse)
            About the definition of output,
            max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
        """
        results = ops.sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)
        return results

    def _sparse_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: IluFlashMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        topk_indices = self.topk_indices_buffer[:num_tokens]
        # Per-request indices -> global paged slots.
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        return self.flash_mla_sparse_prefill(
            q, kv_c_and_k_pe_cache, topk_indices, self.softmax_scale
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse MLA forward. Returns ``(T, num_heads * v_head_dim)``.

        ``q``: ``(T, num_heads, qk_head_dim)``; ``kv_c_normed``: ``(T, kv_lora)``;
        ``k_pe``: ``(T, qk_rope_head_dim)``. Rope is applied here (not in the
        wrapper), matching the fork ``forward_full`` contract.
        """
        num_tokens = q.shape[0]
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.prefix)

        if attn_metadata is None:
            # Profiling / dummy run: reserve a comparable workspace so peak
            # memory is measured, then return zeros (matches fork forward_full).
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadataBuilder,
            )

            ws = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                self._vllm_config
            )
            _ = torch.empty(
                (ws, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            return q.new_zeros((num_tokens, self.num_heads * self.v_head_dim))

        num_actual_toks = attn_metadata.num_actual_tokens
        kv_cache = self.kv_cache

        k_pe = k_pe.unsqueeze(1)
        q = q[:num_actual_toks]
        kv_c_normed = kv_c_normed[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]
        positions = positions[:num_actual_toks]

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        # Absorbed k-up: (B, N, P) -> (B, N, kv_lora_rank), concat with rope.
        q_nope = self._k_up_proj(q_nope)
        q_nope = q_nope.view(-1, self.num_heads, self.kv_lora_rank)
        q = torch.cat([q_nope, q_pe], dim=-1)

        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                kv_c_normed,
                k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=self._k_scale,
            )

        attn_out = self._sparse_mla_kernel(q, kv_cache, attn_metadata)

        output = q.new_empty((num_tokens, self.num_heads, self.v_head_dim))
        output[:num_actual_toks] = self._v_up_proj(attn_out)
        return output.view(num_tokens, self.num_heads * self.v_head_dim)


class IluDeepseekV32Attention(nn.Module):
    """DeepSeek V3.2 MLA wrapper with Iluvatar DSA indexer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.topk_indices_buffer = topk_indices_buffer

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        assert self.q_lora_rank is not None, (
            "DeepseekV32Attention requires q_lora_rank; the no-lora path is "
            "not supported for DSA."
        )
        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
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
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        skip_topk = False
        layer_id = extract_layer_index(prefix)
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_topk_pattern = getattr(config, "index_topk_pattern", None)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        if index_topk_pattern is None:
            skip_topk = (
                max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
            )
        elif 0 <= layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = num_hidden_layers is not None and layer_id >= num_hidden_layers

        if not skip_topk or is_mtp_layer:
            self.indexer = IluDeepseekV32Indexer(
                vllm_config,
                config,
                self.hidden_size,
                self.q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                prefix=f"{prefix}.indexer",
                is_inplace_rope=False,
            )
        else:
            self.indexer = None

        self.skip_topk = skip_topk and not is_mtp_layer
        self.mla_attn = IluDeepseekV32SparseMLA(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            topk_indices_buffer=topk_indices_buffer,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    @property
    def skip_topk(self) -> bool:
        return self._skip_topk

    @skip_topk.setter
    def skip_topk(self, skip: bool) -> None:
        self._skip_topk = skip

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_latent_kpe = self.fused_qkv_a_proj(hidden_states)[0]
        output_padding_size = getattr(self.fused_qkv_a_proj, "output_padding_size", 0)
        q_c, kv_a, k_pe, _ = q_latent_kpe.split(
            [
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                output_padding_size,
            ],
            dim=1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv_c_normed = self.kv_a_layernorm(kv_a)

        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.rotary_emb)

        attn_out = self.mla_attn(q, kv_c_normed, k_pe, positions)
        return self.o_proj(attn_out)[0]
