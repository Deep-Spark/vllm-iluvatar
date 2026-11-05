# SPDX-License-Identifier: Apache-2.0
# Source: _custom_ops.py at 47fb804; the selected ABI has no vendor fallback.
from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps

import torch

from vllm_iluvatar import _custom_ops as baseline
from vllm_iluvatar.backends.ixaite.runtime import (
    load_ixaite_python_op,
    load_ixaite_torch_op,
)

_ProviderOp = Callable[..., object]
_ILU_OVERRIDES = baseline._ILU_OVERRIDES
_ILU_ADDED = baseline._ILU_ADDED + [
    'ds4_compressor_pool_norm_rope_insert',
    'ds4_compressor_save_partial_states',
    'ds4_fused_qnorm_rope_kv_insert',
    'ds4_inv_rope',
    'ds4_combine_topk_swa_indices',
    'ds4_compute_global_topk_indices_and_lens',
    'ds4_compute_swa_decode_indices_and_lens',
    'ds4_compute_swa_prefill_gather_lens',
    'ds4_dense_compressed_attn_fwd',
    'ds4_flash_mla_sparse_fwd',
    'ds4_flash_mla_with_kvcache',
    'ds4_fused_indexer_q_rope',
    'ds4_gather_k_cache',
    'ds4_merge_attn_outputs',
    'ds4_paged_sparse_attn_fwd',
    'ds4_swa_attn_fwd',
    'dsa_indexer_mqa_logits_with_blocks_bf16',
    'dsa_update_topk_indices',
    'ds4_hc_head',
    'ds4_mhc_post',
    'ds4_mhc_pre',
    'ds4_mhc_pre_v2',
    'ds4_mhc_pre_v3',
    'gemm_bf16_bf16_fp32',
    'gemma_rms_norm',
    'layer_norm',
    'linear',
    'marlin_w8a16',
    'rms_norm_gated',
]
_ROUTED_OP_NAMES = (
    "silu_and_mul",
    "linear",
    "rms_norm_gated",
    "rms_norm",
    "fused_add_rms_norm",
    "gemma_rms_norm",
    "layer_norm",
    "rms_norm_qk",
    "ds4_inv_rope",
    "ds4_fused_qnorm_rope_kv_insert",
    "gemm_bf16_bf16_fp32",
    "marlin_w8a16",
    "ds4_compressor_save_partial_states",
    "ds4_compressor_pool_norm_rope_insert",
    "ds4_mhc_pre",
    "ds4_mhc_pre_v2",
    "ds4_mhc_pre_v3",
    "ds4_mhc_post",
    "ds4_compute_global_topk_indices_and_lens",
    "ds4_combine_topk_swa_indices",
    "ds4_compute_swa_decode_indices_and_lens",
    "ds4_compute_swa_prefill_gather_lens",
    "ds4_gather_k_cache",
    "ds4_merge_attn_outputs",
    "ds4_fused_indexer_q_rope",
    "dsa_indexer_mqa_logits_with_blocks_bf16",
    "dsa_update_topk_indices",
    "ds4_flash_mla_with_kvcache",
    "ds4_flash_mla_sparse_fwd",
    "ds4_swa_attn_fwd",
    "ds4_paged_sparse_attn_fwd",
    "ds4_dense_compressed_attn_fwd",
    "ds4_hc_head",
    "scaled_int8_quant",
    "reshape_and_cache_flash",
)


def _ixaite_ds4_decode(
    implementation, *args, positions=None, cos_sin_cache=None, **kwargs
):
    if positions is not None or cos_sin_cache is not None:
        raise ValueError("ixaite decode requires a separate inverse RoPE operation")
    return implementation(*args, **kwargs)


def _ixaite_mqa_logits(implementation, *args, backend=None, **kwargs):
    del backend
    return implementation(*args, **kwargs)


def _load_ixaite_routed_op(op_name: str) -> tuple[_ProviderOp | None, str]:
    # Python entry keeps the CUDA-graph pin plan; torch.ops skips it.
    # Python entry accepts optional residual/weight; torch.ops requires both.
    # scaled_int8_quant returns a tuple and has a static/dynamic split.
    if op_name in (
        "reshape_and_cache_flash",
        "rms_norm",
        "rms_norm_qk",
        "fused_add_rms_norm",
        "gemma_rms_norm",
        "layer_norm",
        "scaled_int8_quant",
        "ds4_fused_qnorm_rope_kv_insert",
        "ds4_inv_rope",
        "ds4_compressor_save_partial_states",
        "ds4_compressor_pool_norm_rope_insert",
        "gemm_bf16_bf16_fp32",
        "marlin_w8a16",
        "ds4_mhc_pre",
        "ds4_mhc_pre_v2",
        "ds4_mhc_pre_v3",
        "ds4_mhc_post",
        "ds4_hc_head",
        "ds4_compute_global_topk_indices_and_lens",
        "ds4_combine_topk_swa_indices",
        "ds4_compute_swa_decode_indices_and_lens",
        "ds4_compute_swa_prefill_gather_lens",
        "ds4_gather_k_cache",
        "ds4_merge_attn_outputs",
        "ds4_fused_indexer_q_rope",
        "dsa_indexer_mqa_logits_with_blocks_bf16",
        "dsa_update_topk_indices",
        "ds4_flash_mla_with_kvcache",
        "ds4_flash_mla_sparse_fwd",
        "ds4_swa_attn_fwd",
        "ds4_paged_sparse_attn_fwd",
        "ds4_dense_compressed_attn_fwd",
    ):
        _DS4_PY_MOD = {
            "ds4_compute_global_topk_indices_and_lens": "ds4_compute_global_topk",
            "ds4_combine_topk_swa_indices": "ds4_combine_topk_swa",
            "ds4_compute_swa_decode_indices_and_lens": "ds4_compute_swa_decode_indices",
            "ds4_compute_swa_prefill_gather_lens": "ds4_compute_swa_prefill_gather_lens",
            "dsa_indexer_mqa_logits_with_blocks_bf16": "dsa_indexer_mqa_logits_bf16",
        }
        if op_name in (
            "ds4_flash_mla_with_kvcache",
            "dsa_indexer_mqa_logits_with_blocks_bf16",
        ):
            implementation, reason = load_ixaite_python_op(
                _DS4_PY_MOD.get(op_name, op_name), op_name
            )
            if implementation is None:
                return None, reason
            adapter = (
                _ixaite_ds4_decode
                if op_name == "ds4_flash_mla_with_kvcache"
                else _ixaite_mqa_logits
            )
            return partial(adapter, implementation), reason
        if op_name in _DS4_PY_MOD:
            return load_ixaite_python_op(_DS4_PY_MOD[op_name], op_name)
        if op_name in (
            "ds4_mhc_pre",
            "ds4_mhc_pre_v2",
            "ds4_mhc_pre_v3",
            "ds4_mhc_post",
            "ds4_hc_head",
        ):
            return load_ixaite_python_op("ds4_mhc", op_name)
        if op_name in (
            "ds4_compressor_save_partial_states",
            "ds4_compressor_pool_norm_rope_insert",
        ):
            return load_ixaite_python_op("ds4_compressor", op_name)
        if op_name == "gemm_bf16_bf16_fp32":
            return load_ixaite_python_op("gemm_bf16_bf16_fp32", op_name)
        if op_name == "ds4_inv_rope":
            return load_ixaite_python_op("ds4_inv_rope", op_name)
        if op_name == "marlin_w8a16":
            return load_ixaite_python_op("marlin_w8a16", op_name)
        return load_ixaite_python_op(op_name, op_name)
    return load_ixaite_torch_op(op_name)


# Missing catalog entries remain explicit failures until an affected model uses them.
def _unsupported(name, template):
    def reject(*args, **kwargs):
        raise RuntimeError(f"ixaite does not support operator {name}")

    return wraps(template)(reject) if template is not None else reject


# Resolve aliases once before the community import hook publishes copied references.
def _bind_exports() -> None:
    pure_torch = {
        "apply_repetition_penalties",
        "gelu_fast",
        "gelu_new",
        "gelu_quick",
        "fatrelu_and_mul",
        "mul_and_silu",
        "cutlass_scaled_mm_supports_fp8",
        "cutlass_scaled_mm_supports_block_fp8",
        "ref_mqa_logits",
        "ref_paged_mqa_logits",
    }
    for name in _ILU_OVERRIDES + _ILU_ADDED:
        template = getattr(baseline, name, None)
        if name in _ROUTED_OP_NAMES:
            implementation, reason = _load_ixaite_routed_op(name)
            if implementation is None:
                implementation = _unsupported(f"{name}: {reason}", template)
            globals()[name] = implementation
        elif name in pure_torch:
            globals()[name] = template
        else:
            globals()[name] = _unsupported(name, template)


_bind_exports()
_raw_reshape_and_cache_flash = globals()["reshape_and_cache_flash"]


# The unquantized community ABI carries tensor scales; this writer uses unit scales.
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    _raw_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, 1.0, 1.0
    )


# ixaite decode keeps inverse RoPE as a separate operation.
def ds4_flash_mla_with_kvcache_supports_inv_rope(**kwargs) -> bool:
    return False


# Keep the router's output dtype and TN layout contract.
def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return globals()["gemm_bf16_bf16_fp32"](input, weight, format="TN")


# Bind the fused top-k entry once; padding masking stays in the plugin ABI.
def _bind_topk():
    implementation, reason = load_ixaite_python_op("moe_fused_topk_bias", "apply")
    return implementation or _unsupported(
        f"moe_fused_topk_bias: {reason}", baseline.topk_softmax
    )


_fused_topk = _bind_topk()
_mask_topk_padding = baseline._mask_topk_padding


# Preserve the published top-k ABI, including padding rows.
def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
) -> None:
    _fused_topk(
        gating_output=gating_output,
        topk=topk_weights.shape[-1],
        scoring_func="softmax",
        routed_scaling_factor=1,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
    )
    _mask_topk_padding(topk_weights, topk_ids, is_padding)


# Preserve the published top-k ABI, including padding rows.
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> None:
    _fused_topk(
        gating_output=gating_output,
        topk=topk_weights.shape[-1],
        scoring_func="sigmoid",
        routed_scaling_factor=routed_scaling_factor,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
    )
    _mask_topk_padding(topk_weights, topk_ids, is_padding)
