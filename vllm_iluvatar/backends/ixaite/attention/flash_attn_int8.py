# SPDX-License-Identifier: Apache-2.0
"""KINT8/KVINT8 forwards for the ixaite flash-attn backend.

Quant and cache writers come from ``kv_cache`` (ixaite). Attention kernels stay
on ixformer (``flash_attn_varlen_int8_func`` / paged INT8). Baseline
``ilu_flash_attn.py`` is not modified.
"""

from __future__ import annotations

import ixformer.inference.functions as ixf_ops
import torch
from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_int8_func

from vllm_iluvatar.backends.ixaite.attention import kv_cache
from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import (
    IluFlashAttentionMetadata,
    _get_k_int8_cache_views,
)


class IxaiteFlashAttentionInt8Mixin:
    """INT8 forward methods mixed into ``IxaiteFlashAttentionImpl``."""

    def _forward_kv_int8(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported")
        early_output = self._maybe_forward_encoder(
            layer, query, key, value, attn_metadata, output
        )
        if early_output is not None:
            return early_output
        if attn_metadata.use_cascade:
            raise NotImplementedError("Cascade attention is not supported with KVINT8")
        if self.dcp_world_size > 1:
            raise NotImplementedError("DCP attention is not supported with KVINT8")
        kv_scale_cache = getattr(layer, "kv_cache_scale", None)
        assert kv_scale_cache is not None

        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_only = attn_metadata.num_decodes > 0 and attn_metadata.num_prefills == 0
        prefill_output = output[num_decode_tokens:]
        decode_output = output[:num_decode_tokens]

        i8_key_cache, i8_value_cache = kv_cache_tensor.unbind(0)
        num_blocks, num_kv_heads, block_size, head_size = i8_key_cache.shape
        key_scale_cache, value_scale_cache = kv_scale_cache
        assert (
            key_scale_cache.shape
            == (
                num_blocks,
                num_kv_heads,
                block_size,
            )
            and key_scale_cache.dtype == torch.float32
        ), (
            f"key_scale_cache.shape {key_scale_cache.shape} != "
            f"{(num_blocks, num_kv_heads, block_size)} or "
            f"key_scale_cache.dtype {key_scale_cache.dtype} != torch.float32"
        )
        assert (
            value_scale_cache.shape
            == (
                num_kv_heads,
                head_size,
            )
            and value_scale_cache.dtype == torch.float32
        ), (
            f"value_scale_cache.shape {value_scale_cache.shape} != "
            f"{(num_kv_heads, head_size)} or "
            f"value_scale_cache.dtype {value_scale_cache.dtype} != torch.float32"
        )

        int8_query, query_scale = kv_cache.scaled_int8_quant_for_attn(
            query, 2, transpose_scale=not decode_only
        )
        i8_key, key_scale = kv_cache.scaled_int8_quant_for_attn(
            key, 2, transpose_scale=False
        )
        i8_value, _value_scale = kv_cache.scaled_int8_quant_for_attn(
            value, 0, transpose_scale=False, scale=value_scale_cache
        )

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and attn_metadata.num_prefills > 0
        ):
            kv_cache.reshape_and_cache_with_scale(
                i8_key,
                i8_value,
                key_scale,
                i8_key_cache,
                i8_value_cache,
                key_scale_cache,
                attn_metadata.slot_mapping,
            )

        if attn_metadata.num_prefills > 0 and attn_metadata.prefill is not None:
            prefill_q = query[num_decode_tokens : attn_metadata.num_actual_tokens]
            sliding_window_size = (
                list(self.sliding_window)
                if self.sliding_window is not None
                else (-1, -1)
            )
            flash_attn_varlen_int8_func(
                q=int8_query[num_decode_tokens:],
                k=i8_key_cache,
                v=i8_value_cache,
                q_scale=query_scale[:, num_decode_tokens:],
                k_scale=key_scale_cache,
                v_scale=value_scale_cache,
                cu_seqlens_q=attn_metadata.prefill.query_start_loc,
                cu_seqlens_k=attn_metadata.prefill.key_start_loc,
                max_seqlen_q=attn_metadata.prefill.max_query_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=sliding_window_size,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
                out=prefill_output[: prefill_q.shape[0]],
                block_table=attn_metadata.prefill.block_table,
                output_dtype=query.dtype,
            )

        if attn_metadata.num_decodes > 0 and attn_metadata.decode is not None:
            para_dict = dict(
                output=decode_output,
                query=int8_query[:num_decode_tokens],
                key_cache=i8_key_cache,
                query_scale=query_scale[:num_decode_tokens]
                if decode_only
                else query_scale[:, :num_decode_tokens].t().contiguous(),
                key_scale_cache=key_scale_cache,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_tables=attn_metadata.decode.block_table,
                context_lens=attn_metadata.decode.seq_lens,
                block_size=i8_key_cache.shape[-2],
                softcap=self.logits_soft_cap,
                alibi_slopes=self.alibi_slopes,
                causal=True,
                window_left=self.sliding_window[0],
                window_right=self.sliding_window[1],
                use_cuda_graph=attn_metadata.decode.use_graph if decode_only else False,
                max_context_len=attn_metadata.decode.max_decode_seq_len,
                cu_query_lens=attn_metadata.decode.query_start_loc,
                max_query_len=attn_metadata.decode.max_query_len,
                value_cache=i8_value_cache,
                value_scale_cache=value_scale_cache,
            )
            if decode_only:
                para_dict.update(
                    dict(
                        save_key=i8_key[:num_decode_tokens],
                        save_value=i8_value[:num_decode_tokens],
                        save_key_scale=key_scale[:num_decode_tokens],
                    )
                )
            ixf_ops.vllm_paged_attention_int8(**para_dict)
        return output

    def _forward_k_int8(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported")
        early_output = self._maybe_forward_encoder(
            layer, query, key, value, attn_metadata, output
        )
        if early_output is not None:
            return early_output
        if attn_metadata.use_cascade:
            raise NotImplementedError("Cascade attention is not supported with KINT8")
        if self.dcp_world_size > 1:
            raise NotImplementedError("DCP attention is not supported with KINT8")

        kv_scale_cache = getattr(layer, "kv_cache_scale", None)
        assert kv_scale_cache is not None

        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_only = attn_metadata.num_decodes > 0 and attn_metadata.num_prefills == 0
        prefill_output = output[num_decode_tokens:]
        decode_output = output[:num_decode_tokens]

        i8_key_cache, value_cache = _get_k_int8_cache_views(
            kv_cache_tensor, query.dtype
        )
        key_scale_cache = kv_scale_cache[0]

        int8_query, query_scale = kv_cache.scaled_int8_quant_for_attn(
            query, 2, transpose_scale=not decode_only
        )
        i8_key, key_scale = kv_cache.scaled_int8_quant_for_attn(
            key, 2, transpose_scale=False
        )

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and attn_metadata.num_prefills > 0
        ):
            kv_cache.reshape_and_cache_mix(
                i8_key,
                value,
                key_scale,
                i8_key_cache,
                value_cache,
                key_scale_cache,
                attn_metadata.slot_mapping,
            )

        if attn_metadata.num_prefills > 0 and attn_metadata.prefill is not None:
            prefill_q = query[num_decode_tokens : attn_metadata.num_actual_tokens]
            sliding_window_size = (
                list(self.sliding_window)
                if self.sliding_window is not None
                else (-1, -1)
            )
            flash_attn_varlen_int8_func(
                q=int8_query[num_decode_tokens:],
                k=i8_key_cache,
                v=value_cache,
                q_scale=query_scale[:, num_decode_tokens:],
                k_scale=key_scale_cache,
                v_scale=None,
                cu_seqlens_q=attn_metadata.prefill.query_start_loc,
                cu_seqlens_k=attn_metadata.prefill.key_start_loc,
                max_seqlen_q=attn_metadata.prefill.max_query_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=sliding_window_size,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
                out=prefill_output[: prefill_q.shape[0]],
                block_table=attn_metadata.prefill.block_table,
                output_dtype=query.dtype,
            )

        if attn_metadata.num_decodes > 0 and attn_metadata.decode is not None:
            para_dict = dict(
                output=decode_output,
                query=int8_query[:num_decode_tokens],
                key_cache=i8_key_cache,
                query_scale=query_scale[:num_decode_tokens]
                if decode_only
                else query_scale[:, :num_decode_tokens].t().contiguous(),
                key_scale_cache=key_scale_cache,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_tables=attn_metadata.decode.block_table,
                context_lens=attn_metadata.decode.seq_lens,
                block_size=i8_key_cache.shape[-2],
                softcap=self.logits_soft_cap,
                alibi_slopes=self.alibi_slopes,
                causal=True,
                window_left=self.sliding_window[0],
                window_right=self.sliding_window[1],
                use_cuda_graph=attn_metadata.decode.use_graph if decode_only else False,
                max_context_len=attn_metadata.decode.max_decode_seq_len,
                cu_query_lens=attn_metadata.decode.query_start_loc,
                max_query_len=attn_metadata.decode.max_query_len,
                value_cache=value_cache,
            )
            if decode_only:
                para_dict.update(
                    dict(
                        save_key=i8_key[:num_decode_tokens],
                        save_value=value[:num_decode_tokens].contiguous(),
                        save_key_scale=key_scale[:num_decode_tokens],
                    )
                )
            ixf_ops.vllm_paged_attention_mix(**para_dict)
        return output
