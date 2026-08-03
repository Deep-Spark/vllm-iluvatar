# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar OOT wrapper for vLLM's generic MLA pluggable layer.

Corresponds to: vllm/model_executor/layers/mla.py

Target   : vllm.model_executor.layers.mla.MultiHeadLatentAttentionWrapper
Baseline : main_61ac3680210
Why      : Iluvatar MLA backends apply rope and cache updates inside
           ``MLAAttention`` using ``positions`` and the layer ``rotary_emb``.
How      : @MultiHeadLatentAttentionWrapper.register_oot subclass. Community
           initialization is preserved; the OOT wrapper only forwards
           ``rotary_emb`` into the inner attention layer and owns the Iluvatar
           forward variants.
Scope    : instantiated on Iluvatar through the official PluggableLayer OOT
           extension point during ``apply_oot_registrations()``.
Remove   : when the community MLA wrapper exposes a backend-controlled
           pre/post-processing path for rope and fused q/kv projection.
"""

from __future__ import annotations

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


@MultiHeadLatentAttentionWrapper.register_oot()
class IluMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """MultiHeadLatentAttentionWrapper with Iluvatar MLA preprocessing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            scale,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk,
        )
        # [ILU override] MLAAttention.rotary_emb - delta: plugin backends apply
        # rope inside MLAAttention when positions are passed through forward().
        self.mla_attn.rotary_emb = self.rotary_emb

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            output_padding_size = getattr(
                self.fused_qkv_a_proj, "output_padding_size", 0
            )
            q_c, kv_c, k_pe, _ = qkv_lora.split(
                [
                    self.q_lora_rank,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                    output_padding_size,
                ],
                dim=1,
            )
            q_c = self.q_a_layernorm(q_c)
            q_proj_layer = self.q_b_proj
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=1
            )
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)
        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            position=positions,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            q_dcp_replicated=q_dcp_replicated,
        )

        return self.o_proj(attn_out)[0]
