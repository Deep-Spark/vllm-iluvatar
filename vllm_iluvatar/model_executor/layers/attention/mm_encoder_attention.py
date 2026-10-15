# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Iluvatar OOT override for multimodal encoder attention.

Corresponds to: vllm/model_executor/layers/attention/mm_encoder_attention.py

Target   : vllm.model_executor.layers.attention.mm_encoder_attention.
           MMEncoderAttention
Baseline : main_6e448d0
Why      : CoreX ViT FlashAttention benefits from padded head dimensions and
           needs optional ixformer/sageattention INT8 paths controlled by
           Iluvatar env vars.
How      : @MMEncoderAttention.register_oot subclass. Community initialization
           and non-FA backends are preserved; only FlashAttention routing and
           helpers are owned here.
Scope    : multimodal encoder attention instantiated on Iluvatar.
Remove   : when upstream exposes backend hooks for ViT FA padding/quantization.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
import vllm.envs as envs
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)

_VIT_HEAD_PAD_TO = 128
_BLOCK_IDX_RE = re.compile(r"\.blocks\.(\d+)\.")


@MMEncoderAttention.register_oot()
class IluMMEncoderAttention(MMEncoderAttention):
    """CoreX-specific MM encoder attention override."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )
        match = _BLOCK_IDX_RE.search(prefix or "")
        self._vit_block_idx = int(match.group(1)) if match else -1

    @property
    def padded_head_dim(self) -> int:
        if self.is_flash_attn_backend:
            return _VIT_HEAD_PAD_TO * (
                (self.head_size + _VIT_HEAD_PAD_TO - 1) // _VIT_HEAD_PAD_TO
            )
        return self.head_size

    def _effective_vit_opt_level(self) -> int:
        opt_level = envs.VLLM_ILU_VIT_ATTN_OPT_LEVEL
        if opt_level == 0:
            return 0
        layer_filter = envs.VLLM_ILU_VIT_ATTN_LAYER
        if not layer_filter:
            return opt_level
        try:
            allowed = {
                int(layer.strip()) for layer in layer_filter.split(",") if layer.strip()
            }
        except ValueError:
            return opt_level
        return opt_level if self._vit_block_idx in allowed else 0

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4
        if max_seqlen is None:
            eff_max_q = q_len
            eff_max_kv = kv_len
        else:
            eff_max = (
                int(max_seqlen.item())
                if isinstance(max_seqlen, torch.Tensor)
                else int(max_seqlen)
            )
            eff_max_q = eff_max
            eff_max_kv = eff_max

        head_size = self.head_size
        target_head = self.padded_head_dim
        in_head = head_size if query.dim() == 3 else query.size(-1)
        assert in_head in (head_size, target_head), (
            f"input head_dim={in_head} must be either {head_size} (raw) "
            f"or {target_head} (pre-padded)."
        )
        pad = target_head - in_head

        query_4d = query.view(bsz, q_len, self.num_heads, in_head)
        key_4d = key.view(bsz, kv_len, self.num_kv_heads, in_head)
        value_4d = value.view(bsz, kv_len, self.num_kv_heads, in_head)
        if pad > 0:
            query_4d = F.pad(query_4d, [0, pad], value=0)
            key_4d = F.pad(key_4d, [0, pad], value=0)
            value_4d = F.pad(value_4d, [0, pad], value=0)
        padded_head = target_head

        opt_level = self._effective_vit_opt_level()
        if opt_level == 2:
            out = self._forward_fa_qk_int8_v_bf16(
                query_4d, key_4d, value_4d, bsz, q_len, kv_len
            )
        elif opt_level == 1:
            out = self._forward_fa_all_int8(
                query_4d,
                key_4d,
                value_4d,
                bsz,
                q_len,
                kv_len,
                cu_seqlens,
                eff_max_q,
                eff_max_kv,
            )
        else:
            out = self._forward_fa_bf16(
                query_4d,
                key_4d,
                value_4d,
                bsz,
                q_len,
                kv_len,
                cu_seqlens,
                eff_max_q,
                eff_max_kv,
            )
        if padded_head > head_size:
            out = out[..., :head_size]
        if is_reshaped:
            out = out.reshape(bsz, q_len, -1)
        return out

    def _forward_fa_bf16(
        self,
        query_4d: torch.Tensor,
        key_4d: torch.Tensor,
        value_4d: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
        cu_seqlens: torch.Tensor | None,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        import ixformer.contrib.vllm_flash_attn as ops

        padded_head = query_4d.size(-1)
        q3d = query_4d.reshape(bsz * q_len, self.num_heads, padded_head)
        k3d = key_4d.reshape(bsz * kv_len, self.num_kv_heads, padded_head)
        v3d = value_4d.reshape(bsz * kv_len, self.num_kv_heads, padded_head)

        if cu_seqlens is not None:
            cu_q = cu_seqlens
            cu_kv = cu_seqlens
        else:
            cu_q = torch.arange(
                0,
                (bsz + 1) * q_len,
                q_len,
                dtype=torch.int32,
                device=q3d.device,
            )
            cu_kv = (
                cu_q
                if kv_len == q_len
                else torch.arange(
                    0,
                    (bsz + 1) * kv_len,
                    kv_len,
                    dtype=torch.int32,
                    device=q3d.device,
                )
            )
        out = ops.flash_attn_varlen_func(
            q3d,
            k3d,
            v3d,
            cu_q,
            cu_kv,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
        )
        return out.view(bsz, q_len, self.num_heads, padded_head)

    def _forward_fa_qk_int8_v_bf16(
        self,
        query_4d: torch.Tensor,
        key_4d: torch.Tensor,
        value_4d: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        from ixformer.contrib.sageattention import (
            ixinfer_flash_attn_unpad_int8_vf16,
        )
        from ixformer.contrib.sageattention.core import quant_q

        i8_q, q_scale = quant_q(query_4d)
        i8_k, k_scale = quant_q(key_4d)
        out, _ = ixinfer_flash_attn_unpad_int8_vf16(
            i8_q,
            i8_k,
            value_4d,
            q_scale,
            k_scale,
            is_causal=False,
            atten_scale=self.scale,
        )
        return out

    def _forward_fa_all_int8(
        self,
        query_4d: torch.Tensor,
        key_4d: torch.Tensor,
        value_4d: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
        cu_seqlens: torch.Tensor | None,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        from ixformer.contrib.sageattention import sageattn_allint8

        padded_head = query_4d.size(-1)
        if cu_seqlens is not None:
            cu_q = cu_seqlens
            cu_kv = cu_seqlens
        else:
            cu_q = torch.arange(
                0,
                (bsz + 1) * q_len,
                q_len,
                dtype=torch.int32,
                device=query_4d.device,
            )
            cu_kv = (
                cu_q
                if kv_len == q_len
                else torch.arange(
                    0,
                    (bsz + 1) * kv_len,
                    kv_len,
                    dtype=torch.int32,
                    device=query_4d.device,
                )
            )

        out, _ = sageattn_allint8(
            query_4d,
            key_4d,
            value_4d,
            is_causal=False,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
        )
        return out.view(bsz, q_len, self.num_heads, padded_head)
