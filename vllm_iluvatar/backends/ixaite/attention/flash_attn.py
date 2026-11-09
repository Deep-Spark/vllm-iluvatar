# SPDX-License-Identifier: Apache-2.0
# Source: ilu_flash_attn.py at 47fb804; independent BF16 execution and cache writes.
"""Ixaite attention with baseline metadata storage and no vendor execution path."""

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import torch
import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType

from vllm_iluvatar.backends.ixaite import ops
from vllm_iluvatar.backends.ixaite.attention.flash_attn_int8 import (
    IxaiteFlashAttentionInt8Mixin,
)
from vllm_iluvatar.backends.ixaite.attention.kv_cache import (
    configure_int8_attention_providers,
)
from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
from vllm_iluvatar.v1.attention.backends.ilu_flash_attn import (
    IluFlashAttentionBackend,
    IluFlashAttentionImpl,
    IluFlashAttentionMetadata,
    IluFlashAttentionMetadataBuilder,
    QueryLenSupport,
    use_cascade_attention,
)

_SUPPORTED_INT8_QUANT_TYPES = frozenset({"KINT8", "KVINT8"})


def _flash_attn_quant_type() -> str:
    return envs.VLLM_ILU_FLASH_ATTN_QUANT_TYPE or ""


def _is_int8_flash_attn_quant(quant_type: str | None = None) -> bool:
    return (quant_type if quant_type is not None else _flash_attn_quant_type()) in (
        _SUPPORTED_INT8_QUANT_TYPES
    )

if TYPE_CHECKING:
    from vllm.v1.attention.selector import AttentionSelectorConfig

_FlashAttnOps = tuple[
    Callable[..., object], Callable[..., object], Callable[..., object]
]
_SELECTED_FLASH_ATTN_OPS: _FlashAttnOps | None = None


# Fixed speculative graph configuration must fail before state allocation.
def _uses_speculative_full_graph(vllm_config) -> bool:
    spec = getattr(vllm_config, "speculative_config", None)
    compilation = getattr(vllm_config, "compilation_config", None)
    return (
        spec is not None
        and bool(getattr(spec, "num_speculative_tokens", 0))
        and compilation is not None
        and compilation.cudagraph_mode.has_full_cudagraphs()
    )


# The whole attention contract is available before any stateful layer is built.
def configure_flash_attn_provider() -> None:
    global _SELECTED_FLASH_ATTN_OPS
    selected, reason = _load_ixaite_flash_attn_ops()
    if selected is None:
        raise RuntimeError(f"ixaite flash_attn is unavailable: {reason}")
    _SELECTED_FLASH_ATTN_OPS = selected
    from vllm_iluvatar.backends.ixaite.attention.mm_encoder_attention import (
        bind_bf16_flash_attn_varlen,
    )

    bind_bf16_flash_attn_varlen(selected[0], provider="ixaite")


# Platform selection rejects fixed configuration before allocating cache metadata.
def get_flash_attn_backend_class_path(
    attn_selector_config: "AttentionSelectorConfig | None" = None,
) -> str:
    reason = _ixaite_flash_attn_selector_rejection_reason(attn_selector_config)
    if reason is not None:
        raise RuntimeError(f"ixaite requested flash_attn, but {reason}")
    return (
        "vllm_iluvatar.backends.ixaite.attention.flash_attn.IxaiteFlashAttentionBackend"
    )


# Attention, paged decode and cascade merge form one initialization contract.
def _load_ixaite_flash_attn_ops() -> tuple[_FlashAttnOps | None, str]:
    """Load the two Python-ABI attention entry points as one provider."""
    implementations: list[Callable[..., object]] = []
    reasons: list[str] = []
    for func_name in (
        "flash_attn_varlen_func",
        "flash_attn_qwen35_decode",
        "merge_attn_states",
    ):
        implementation, reason = load_ixaite_python_op(
            "flash_attn",
            func_name,
        )
        if implementation is None:
            return None, reason
        implementations.append(implementation)
        reasons.append(reason)
    return (implementations[0], implementations[1], implementations[2]), "; ".join(
        dict.fromkeys(reasons)
    )


# Use the kernel library's geometry table instead of duplicating supported ratios.
def _ixaite_attention_geometry_rejection_reason(
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> str | None:
    try:
        from ixaite.ops.attention.flash_attn_paged_kv import (
            SUPPORTED_D256_GQA_RATIOS,
            SUPPORTED_PAGED_HEAD_DIMS,
            is_supported_d256_attention_geometry,
        )
    except ImportError:
        return "ixaite attention geometry helper is unavailable"
    if is_supported_d256_attention_geometry(num_heads, num_kv_heads, head_size):
        return None
    dims = ", ".join(str(dim) for dim in SUPPORTED_PAGED_HEAD_DIMS)
    ratios = ", ".join(str(ratio) for ratio in SUPPORTED_D256_GQA_RATIOS)
    return (
        f"TP-local attention geometry must have head_size in {{{dims}}} and "
        f"num_heads / num_kv_heads in {{{ratios}}}; got "
        f"({num_heads}, {num_kv_heads}, {head_size})"
    )


# Global options are known before a backend's metadata buffers exist.
def _ixaite_flash_attn_selector_rejection_reason(
    attn_selector_config: "AttentionSelectorConfig | None",
) -> str | None:
    """Return why the fixed configuration is unsupported."""
    if attn_selector_config is None:
        return "attention selector configuration is unavailable"

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "no active VllmConfig is available"
    if _uses_speculative_full_graph(vllm_config):
        return "multi-token graph decode has an unresolved ixaite numerical failure"

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    if getattr(parallel_config, "decode_context_parallel_size", 1) != 1:
        return "decode context parallelism is not supported"
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        return "prefill context parallelism is not supported"

    num_heads = model_config.get_num_attention_heads(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    geometry_reason = _ixaite_attention_geometry_rejection_reason(
        num_heads,
        num_kv_heads,
        attn_selector_config.head_size,
    )
    if geometry_reason is not None:
        return geometry_reason
    if attn_selector_config.dtype != torch.bfloat16:
        return "only BF16 attention is supported"
    if attn_selector_config.kv_cache_dtype not in (None, "auto", "bfloat16"):
        return "only unquantized BF16 KV cache is supported"
    if attn_selector_config.block_size not in (None, 16):
        return "only KV-cache block size 16 is supported"
    if attn_selector_config.attn_type != AttentionType.DECODER:
        return "only standard decoder attention is supported"
    if attn_selector_config.has_sink:
        return "attention sinks are not supported"
    if attn_selector_config.has_sliding_window:
        return "sliding-window attention is not supported"
    if attn_selector_config.use_non_causal:
        return "non-causal attention is not supported"
    hf_config = getattr(model_config, "hf_config", None)
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config
    if hf_config is not None:
        if any(
            bool(getattr(hf_config, name, False)) for name in ("alibi", "use_alibi")
        ):
            return "ALiBi is not supported"
        if any(
            float(getattr(hf_config, name, 0.0) or 0.0) != 0.0
            for name in (
                "attn_logit_softcapping",
                "attention_logit_softcapping",
                "logits_soft_cap",
            )
        ):
            return "logits soft cap is not supported"
        query_pre_attn_scalar = getattr(hf_config, "query_pre_attn_scalar", None)
        if query_pre_attn_scalar is not None and not math.isclose(
            float(query_pre_attn_scalar),
            float(attn_selector_config.head_size),
        ):
            return "non-default attention scale is not supported"
        attention_multiplier = getattr(hf_config, "attention_multiplier", None)
        if attention_multiplier is not None and not math.isclose(
            float(attention_multiplier),
            float(attn_selector_config.head_size) ** -0.5,
        ):
            return "non-default attention multiplier is not supported"
    if (
        attn_selector_config.use_mla
        or attn_selector_config.use_sparse
        or attn_selector_config.use_mm_prefix
        or attn_selector_config.use_per_head_quant_scales
        or attn_selector_config.use_batch_invariant
        or attn_selector_config.use_kv_connector
        or attn_selector_config.use_pcp
    ):
        return "the requested attention mode is not supported"
    quant_type = _flash_attn_quant_type()
    if quant_type and quant_type not in _SUPPORTED_INT8_QUANT_TYPES:
        return f"attention quant type {quant_type!r} is not supported"
    if envs.VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH:
        return "FP8 attention scales are not supported"
    if envs.VLLM_ILU_FLASH_ATTN_V_SCALE_PATH and quant_type != "KVINT8":
        return "INT8 post-quantization is not supported"
    return None


# Layer options include the final scale and cache contract after TP sharding.
def _ixaite_flash_attn_config_rejection_reason(
    *,
    vllm_config: VllmConfig | None,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    attn_type: AttentionType,
    kv_cache_dtype: str,
    quant_type: str,
    post_quant_kv_int8: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    sinks: torch.Tensor | None,
    kv_sharing_target_layer_name: str | None,
) -> str | None:
    """Return why this layer is outside the validated ixaite envelope."""
    if vllm_config is None:
        return "no active VllmConfig is available"
    if _uses_speculative_full_graph(vllm_config):
        return "multi-token graph decode has an unresolved ixaite numerical failure"

    parallel_config = vllm_config.parallel_config
    geometry_reason = _ixaite_attention_geometry_rejection_reason(
        num_heads,
        num_kv_heads,
        head_size,
    )
    if geometry_reason is not None:
        return geometry_reason
    if not math.isclose(scale, 1.0 / math.sqrt(head_size)):
        return "non-default attention scale is not supported"
    if getattr(parallel_config, "decode_context_parallel_size", 1) != 1:
        return "decode context parallelism is not supported"
    # Weight-only/W8A8 model quantization is orthogonal to this operator. Gate
    # only the attention/KV-cache quantization modes below.
    if quant_type and quant_type not in _SUPPORTED_INT8_QUANT_TYPES:
        return f"attention quant type {quant_type!r} is not supported"
    if post_quant_kv_int8:
        return "INT8 post-quantization is not supported"
    if kv_cache_dtype not in ("auto", "bfloat16"):
        return "only BF16 attention and KV cache dtypes are supported"
    if attn_type != AttentionType.DECODER:
        return "only standard decoder attention is supported"
    if alibi_slopes is not None:
        return "ALiBi is not supported"
    if sliding_window != (-1, -1):
        return "sliding-window attention is not supported"
    if logits_soft_cap != 0:
        return "logits soft cap is not supported"
    if sinks is not None:
        return "attention sinks are not supported"
    if kv_sharing_target_layer_name is not None:
        return "KV-cache sharing is not supported"
    return None


# Keep existing metadata addresses and ordering while applying ixaite graph limits.
class IxaiteFlashAttentionMetadataBuilder(IluFlashAttentionMetadataBuilder):
    """Metadata policy for the capability-gated ixaite sibling backend."""

    # Spec verify is uniform K+1 tokens per request. The ixaite multi-token
    # decode plan covers the full paged-cache capacity and derives active
    # splits from cache_seqlens at replay time.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    # Reject incompatible physical pages before allocating persistent metadata.
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if _uses_speculative_full_graph(vllm_config):
            raise RuntimeError("ixaite multi-token graph decode is not supported")
        if kv_cache_spec.block_size != 16:
            raise RuntimeError("ixaite flash_attn requires KV-cache pages of size 16")
        quant_type = _flash_attn_quant_type()
        if _is_int8_flash_attn_quant(quant_type):
            # KINT8/KVINT8 keep the model dtype on the spec; physical planes are
            # int8 (and BF16 value planes for KINT8). Page size 16 is required.
            pass
        elif kv_cache_spec.dtype != torch.bfloat16:
            raise RuntimeError(
                "ixaite flash_attn requires BF16 KV-cache pages of size 16"
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    # Mixed prefill batches must not use a uniform decode graph plan.
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> IluFlashAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        decode = metadata.decode
        if decode is not None and decode.use_graph and decode.max_query_len > 1:
            raise RuntimeError("ixaite multi-token graph decode is not supported")
        if decode is not None and metadata.num_prefills > 0:
            decode.use_graph = False
        return metadata

    # The baseline's scheduling heuristic only inspects metadata and is reusable.
    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


# Reuse metadata types while keeping the executable layer independent.
class IxaiteFlashAttentionImpl(IxaiteFlashAttentionInt8Mixin, IluFlashAttentionImpl):
    provider_name: ClassVar[str] = "ixaite"
    supports_ilu_external_fp8_scales = False

    # Validate fixed options before binding kernels or allocating layer state.
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        sinks=None,
    ):
        quant_type = _flash_attn_quant_type()
        # V_SCALE_PATH is required for KVINT8 inference, not post-quant.
        post_quant_kv_int8 = (not quant_type) and bool(
            envs.VLLM_ILU_FLASH_ATTN_V_SCALE_PATH
            or envs.VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH
        )
        reason = _ixaite_flash_attn_config_rejection_reason(
            vllm_config=get_current_vllm_config_or_none(),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            scale=float(scale),
            attn_type=attn_type,
            kv_cache_dtype=kv_cache_dtype,
            quant_type=quant_type,
            post_quant_kv_int8=post_quant_kv_int8,
            alibi_slopes=alibi_slopes,
            sliding_window=(-1, -1)
            if sliding_window is None
            else (sliding_window - 1, 0),
            logits_soft_cap=logits_soft_cap or 0,
            sinks=sinks,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )
        if reason is not None:
            raise RuntimeError(f"ixaite flash_attn configuration rejected: {reason}")
        if quant_type == "KVINT8":
            v_scale_path = envs.VLLM_ILU_FLASH_ATTN_V_SCALE_PATH
            if envs.VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH:
                raise RuntimeError(
                    "KVINT8 inference must not set VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH"
                )
            if not v_scale_path or not os.path.isfile(v_scale_path):
                raise RuntimeError(
                    "KVINT8 inference requires an existing "
                    "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH"
                )
        if _is_int8_flash_attn_quant(quant_type):
            configure_int8_attention_providers()
            self._ops = None
        else:
            if _SELECTED_FLASH_ATTN_OPS is None:
                configure_flash_attn_provider()
            self._ops = _SELECTED_FLASH_ATTN_OPS
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.attn_type = attn_type
        self.kv_cache_dtype = "auto"
        self.kv_sharing_target_layer_name = None
        self.sliding_window = (-1, -1)
        self.alibi_slopes = None
        self.logits_soft_cap = 0
        self.sinks = None
        self.is_fp8 = False
        self.is_post_quant = False
        self.quant_type = quant_type if _is_int8_flash_attn_quant(quant_type) else ""
        self.dcp_world_size = 1
        self.supports_quant_query_input = False
        self.supports_per_head_quant_scales = False
        self.vllm_flash_attn_version = None
        self._forward_fn = {
            "": self._forward_unquantized,
            "KINT8": self._forward_k_int8,
            "KVINT8": self._forward_kv_int8,
        }[self.quant_type]

    # Warmup obtains the same bound functions that inference calls directly.
    def _flash_attn_op(self, op_index):
        if self._ops is None:
            raise RuntimeError(
                "ixaite BF16 flash_attn ops are not bound for INT8 attention"
            )
        return self._ops[op_index]

    # Dynamic checks precede writes to either output or KV state.
    def forward(
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
            raise NotImplementedError("fused output quantization is not yet supported")
        if (
            query.ndim != 3
            or query.dtype != torch.bfloat16
            or query.shape[-2:]
            != (
                self.num_heads,
                self.head_size,
            )
            or query.stride(-1) != 1
        ):
            raise RuntimeError(
                "ixaite flash_attn requires BF16 query tensors with the configured head geometry"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if not attn_metadata.causal:
            raise RuntimeError("ixaite flash_attn requires causal decoder metadata")
        decode = getattr(attn_metadata, "decode", None)
        # shortcut: the current SIMT verify kernel can produce NaN for empty causal pages; enable after the library fix passes replay checks.
        if decode is not None and decode.use_graph and decode.max_query_len > 1:
            raise RuntimeError("ixaite multi-token graph decode is not supported")
        return self._forward_fn(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    # Cache writing precedes segment reads, including a continuation's new suffix.
    def _forward_unquantized(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported")
        key_cache, value_cache = kv_cache.unbind(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        prefill_output = output[num_decode_tokens:]
        decode_output = output[:num_decode_tokens]

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if attn_metadata.use_cascade:
            return self._forward_cascade(
                query, key_cache, value_cache, attn_metadata, output
            )

        if attn_metadata.num_prefills > 0 and attn_metadata.prefill is not None:
            prefill_q = query[num_decode_tokens:num_actual_tokens]
            prefill_out_slice = prefill_output[: prefill_q.shape[0]]
            sliding_window_size = (
                list(self.sliding_window)
                if self.sliding_window is not None
                else (-1, -1)
            )
            self._ops[0](
                q=prefill_q,
                k=key_cache,
                v=value_cache,
                out=prefill_out_slice,
                cu_seqlens_q=attn_metadata.prefill.query_start_loc,
                cu_seqlens_k=attn_metadata.prefill.key_start_loc,
                max_seqlen_q=attn_metadata.prefill.max_query_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                sinks=self.sinks,
                window_size=sliding_window_size,
                block_table=attn_metadata.prefill.block_table,
                softcap=self.logits_soft_cap,
            )

        if attn_metadata.num_decodes > 0 and attn_metadata.decode is not None:
            decode_q = query[:num_decode_tokens]
            self._ops[1](
                q=decode_q.unsqueeze(1),
                k_cache=key_cache,
                v_cache=value_cache,
                block_table=attn_metadata.decode.block_table,
                cache_seqlens=attn_metadata.decode.seq_lens,
                max_query_len=attn_metadata.decode.max_query_len,
                cu_query_lens=attn_metadata.decode.query_start_loc,
                softmax_scale=self.scale,
                causal=True,
                window_size=self.sliding_window,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
                sinks=self.sinks,
                out=decode_output.unsqueeze(1),
                use_cuda_graph=attn_metadata.decode.use_graph,
                max_context_len=attn_metadata.decode.max_decode_seq_len,
            )
        return output

    # Both prefix and suffix use the same bound functions as ordinary attention.
    def _forward_cascade(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: IluFlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            cu_prefix_kv_lens=attn_metadata.cu_prefix_kv_lens,
            cu_suffix_kv_lens=attn_metadata.cu_suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            s_aux=self.sinks,
            varlen_op=self._ops[0],
            merge_op=self._ops[2],
        )
        return output

    # Separate cache-update callers must retain the independently bound writer.
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ):
            return

        key_cache, value_cache = kv_cache.unbind(0)

        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )


# Reuse cache shape helpers while publishing only the supported execution contract.
class IxaiteFlashAttentionBackend(IluFlashAttentionBackend):
    """Provider-specific backend selected before metadata/state allocation."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
    SUPPORTED_HEAD_SIZES = (128, 256)

    # Physical hybrid blocks are split into the kernel's page-16 views.
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]

    # General encoder attention uses the separate MM implementation.
    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    # Decoder metadata is the only contract accepted by this paged backend.
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    # The writer and paged kernels require heads before token slots.
    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType:
        return "HND"

    # Construct executable state from the selected backend only.
    @staticmethod
    def get_impl_cls() -> type[IxaiteFlashAttentionImpl]:
        return IxaiteFlashAttentionImpl

    # Keep the builder paired with the selected implementation.
    @staticmethod
    def get_builder_cls() -> type[IxaiteFlashAttentionMetadataBuilder]:
        return IxaiteFlashAttentionMetadataBuilder


# Merge two independently normalized segments without reading unwritten KV rows.
def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    cu_prefix_kv_lens: torch.Tensor,
    cu_suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    s_aux: torch.Tensor | None = None,
    *,
    varlen_op: Callable,
    merge_op: Callable,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-2]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0

    prefix_output, prefix_lse = varlen_op(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        cu_seqlens_k=cu_prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
    )

    suffix_output, suffix_lse = varlen_op(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
    )

    merge_op(prefix_output, prefix_lse, suffix_output, suffix_lse, output)
