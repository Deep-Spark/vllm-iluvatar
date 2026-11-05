# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only worker: shared ixaite cache with baseline INT8 Attention.

Select this class explicitly with --worker-cls. Production provider selection
and its native INT8 Attention gates remain unchanged.
"""

import os

from vllm.logger import init_logger

from vllm_iluvatar.backends.ixaite.attention.kv_cache import (
    bind_int8_kv_caches,
    load_ixaite_static_v_scales,
)
from vllm_iluvatar.v1.attention.backends import ilu_flash_attn as baseline
from vllm_iluvatar.v1.worker.gpu_worker import IluWorker

logger = init_logger("vllm." + __name__)


class CacheReadAttention(baseline.IluFlashAttentionImpl):
    """Keep native Q quantization and Attention; write new KV through ixaite."""

    def _forward_cache(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("benchmark does not support output quantization")
        early = self._maybe_forward_encoder(
            layer, query, key, value, attn_metadata, output
        )
        if early is not None:
            return early
        if attn_metadata.use_cascade or self.dcp_world_size != 1:
            raise NotImplementedError("benchmark requires ordinary MHA/GQA without DCP")
        if self.kv_sharing_target_layer_name is not None:
            raise NotImplementedError("benchmark does not support layer KV sharing")
        cache = layer._ixaite_int8_cache
        decode_tokens = attn_metadata.num_decode_tokens
        decode_only = attn_metadata.num_decodes > 0 and attn_metadata.num_prefills == 0
        quant_query, query_scale = baseline.ixf_ops.scaled_int8_quant_for_attn(
            query, 2, transpose_scale=not decode_only
        )
        cache.write(key, value, attn_metadata.slot_mapping)

        if attn_metadata.num_prefills > 0 and attn_metadata.prefill is not None:
            prefill = attn_metadata.prefill
            baseline.flash_attn_varlen_int8_func(
                q=quant_query[decode_tokens:],
                k=cache.key,
                v=cache.value,
                q_scale=query_scale[:, decode_tokens:],
                k_scale=cache.key_scale,
                v_scale=cache.value_scale,
                cu_seqlens_q=prefill.query_start_loc,
                cu_seqlens_k=prefill.key_start_loc,
                max_seqlen_q=prefill.max_query_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
                out=output[decode_tokens : attn_metadata.num_actual_tokens],
                block_table=prefill.block_table,
                output_dtype=query.dtype,
            )

        if attn_metadata.num_decodes > 0 and attn_metadata.decode is not None:
            decode = attn_metadata.decode
            arguments = dict(
                output=output[:decode_tokens],
                query=quant_query[:decode_tokens],
                key_cache=cache.key,
                value_cache=cache.value,
                query_scale=query_scale[:decode_tokens]
                if decode_only
                else query_scale[:, :decode_tokens].t().contiguous(),
                key_scale_cache=cache.key_scale,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_tables=decode.block_table,
                context_lens=decode.seq_lens,
                block_size=cache.key.shape[-2],
                max_context_len=decode.max_decode_seq_len,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
                causal=True,
                window_left=self.sliding_window[0],
                window_right=self.sliding_window[1],
                use_cuda_graph=decode.use_graph if decode_only else False,
                cu_query_lens=decode.query_start_loc,
                max_query_len=decode.max_query_len,
            )
            # The independent writer already appended KV; omit fused save_* args.
            if cache.value_scale is None:
                baseline.ixf_ops.vllm_paged_attention_mix(**arguments)
            else:
                baseline.ixf_ops.vllm_paged_attention_int8(
                    **arguments, value_scale_cache=cache.value_scale
                )
        return output

    _forward_k_int8 = _forward_cache
    _forward_kv_int8 = _forward_cache


class CacheReadWorker(IluWorker):
    """Install the bridge only in the explicitly selected benchmark worker."""

    def __init__(self, *args, **kwargs):
        if os.environ.get("VLLM_ILU_KERNEL_PROVIDER", "baseline") != "baseline":
            raise ValueError("cache benchmark keeps other operators on baseline")
        mode = os.environ.get("VLLM_ILU_FLASH_ATTN_QUANT_TYPE")
        if mode not in ("KINT8", "KVINT8"):
            raise ValueError("cache benchmark requires KINT8 or KVINT8")
        baseline.IluFlashAttentionBackend.get_impl_cls = staticmethod(
            lambda: CacheReadAttention
        )
        from vllm_iluvatar.v1.worker import gpu_model_runner

        gpu_model_runner.load_static_v_scales = load_ixaite_static_v_scales
        super().__init__(*args, **kwargs)

    def init_device(self):
        """Bind profiling's temporary caches before its throwaway graph capture."""
        super().init_device()
        runner = self.model_runner
        initialize = runner._init_minimal_kv_cache_for_profiling
        cleanup = runner._cleanup_profiling_kv_cache

        def initialize_profile():
            initialize()
            bind_int8_kv_caches(runner)

        def cleanup_profile():
            # Drop views before the host frees profiling-only backing storage.
            for layer in runner.compilation_config.static_forward_context.values():
                if hasattr(layer, "_ixaite_int8_cache"):
                    del layer._ixaite_int8_cache
            runner._ixaite_int8_caches = ()
            cleanup()

        runner._init_minimal_kv_cache_for_profiling = initialize_profile
        runner._cleanup_profiling_kv_cache = cleanup_profile

    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        bind_int8_kv_caches(self.model_runner)
        for (
            layer
        ) in self.model_runner.compilation_config.static_forward_context.values():
            if hasattr(layer, "_ixaite_int8_cache"):
                assert isinstance(layer.impl, CacheReadAttention), type(layer.impl)
        logger.info(
            "INT8 cache benchmark: %d ixaite bindings; baseline Q/prefill/decode; independent KV writes",
            len(self.model_runner._ixaite_int8_caches),
        )
