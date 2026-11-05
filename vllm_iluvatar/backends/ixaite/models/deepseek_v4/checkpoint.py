# SPDX-License-Identifier: Apache-2.0
"""Classify a DeepSeek-V4 HuggingFace config as FP8 vs W4A8.

Community arch name ``DeepseekV4ForCausalLM`` is used by both the raw FP8
release and FAE W4A8 compressed-tensors packs (DeepSeek-V4-Flash-0731). The
plugin must pick the model class from the checkpoint quant blob, not from
``expert_dtype`` (0731 still says ``fp4``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

_FP8_METHODS = frozenset(
    {
        "fp8",
        "deepseek_v4_fp8",
        "deepseek-v4-fp8",
    }
)
_W4A8_METHODS = frozenset(
    {
        "ilu_deepseek_v4_w4a8",
        "ilu-deepseek-v4-w4a8",
    }
)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict()
        except Exception:
            converted = None
        if isinstance(converted, Mapping):
            return converted
    return None


def _quant_blob(hf_config: Any) -> Mapping[str, Any] | None:
    if isinstance(hf_config, Mapping):
        for key in ("quantization_config", "compression_config"):
            blob = _as_mapping(hf_config.get(key))
            if blob is not None:
                return blob
        return None
    for key in ("quantization_config", "compression_config"):
        blob = _as_mapping(getattr(hf_config, key, None))
        if blob is not None:
            return blob
    return None


def _norm_method(method: Any) -> str:
    return str(method or "").strip().lower().replace("_", "-")


def _has_w4a8_groups(qc: Mapping[str, Any] | None) -> bool:
    if not qc:
        return False
    groups = qc.get("config_groups") or {}
    if not isinstance(groups, Mapping):
        return "w4a8" in str(qc).lower()
    for name, group in groups.items():
        if "w4a8" in str(name).lower():
            return True
        if not isinstance(group, Mapping):
            continue
        weights = group.get("weights") or {}
        acts = group.get("input_activations") or {}
        if not isinstance(weights, Mapping) or not isinstance(acts, Mapping):
            continue
        try:
            w_bits = int(weights.get("num_bits") or 0)
            a_bits = int(acts.get("num_bits") or 0)
        except (TypeError, ValueError):
            continue
        if w_bits == 4 and a_bits == 8:
            return True
    return False


def classify_deepseek_v4_checkpoint(hf_config: Any) -> str:
    """Return ``fp8``, ``w4a8``, or ``unknown``.

    ``unknown`` is fail-closed: do not default to the FP8 OOT model.
    """
    qc = _quant_blob(hf_config)
    method = _norm_method((qc or {}).get("quant_method"))
    method_underscore = method.replace("-", "_")
    if method_underscore in _W4A8_METHODS or method in {
        "ilu-deepseek-v4-w4a8",
    }:
        return "w4a8"
    if method in {"fp8", "deepseek-v4-fp8"} or method_underscore in _FP8_METHODS:
        return "fp8"
    iluvatar_quant = None
    if isinstance(hf_config, Mapping):
        iluvatar_quant = hf_config.get("iluvatar_quant")
    else:
        iluvatar_quant = getattr(hf_config, "iluvatar_quant", None)
    if _has_w4a8_groups(qc) or iluvatar_quant:
        return "w4a8"
    return "unknown"


def classify_deepseek_v4_vllm_config(vllm_config: Any) -> str:
    """Prefer the HuggingFace blob; fall back to the resolved QuantizationConfig name."""
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    kind = classify_deepseek_v4_checkpoint(hf_config)
    if kind != "unknown":
        return kind
    quant_config = getattr(vllm_config, "quant_config", None)
    get_name = getattr(quant_config, "get_name", None)
    if not callable(get_name):
        return "unknown"
    name = _norm_method(get_name())
    name_underscore = name.replace("-", "_")
    if name in {"fp8", "deepseek-v4-fp8"} or name_underscore in _FP8_METHODS:
        return "fp8"
    if name_underscore in _W4A8_METHODS:
        return "w4a8"
    return "unknown"


# Grouped layouts must be recognized before dense loading mutates them.
def is_ds4_wo_a_bmm_layer(layer: torch.nn.Module) -> bool:
    """True for DeepSeek-V4 ``attn.wo_a`` (not GLM ``kv_b_proj``).

    The attention wrapper always attaches ``bmm_geom`` with
    ``n_local_groups`` / ``o_lora_rank``. GLM's absorbed MLA up-proj uses
    a different geom (``num_heads`` / ``kv_lora_rank``) and must stay on
    its own split-repack path.
    """
    geom = getattr(layer, "bmm_geom", None)
    return geom is not None and hasattr(geom, "n_local_groups") and hasattr(
        geom, "o_lora_rank"
    )


from vllm_iluvatar.backends.ixaite.models.support import require_model_ops
from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op


# Preserve the grouped-load guard without invoking a vendor repacker.
def maybe_repack_ds4_wo_a_w8a16(layer, *, gemm_format="k16n16") -> bool:
    if not is_ds4_wo_a_bmm_layer(layer):
        return False
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dtype != torch.int8:
        return False
    repack_ds4_wo_a_w8a16(layer, gemm_format=gemm_format)
    return True


# Cache layout is fixed before any attention cache or output can be modified.
def validate_attention_contract(cache_config) -> None:
    dtype = None if cache_config is None else cache_config.cache_dtype
    if dtype not in {None, "auto", "bfloat16"}:
        raise RuntimeError(f"ixaite DeepSeek-V4 cache layout is unavailable: {dtype}")


# Reject fixed routing/activation dependencies before constructing parameter owners.
def validate_model_contract(vllm_config, *, draft=False) -> None:
    model_config = (
        vllm_config.speculative_config.draft_model_config
        if draft
        else vllm_config.model_config
    )
    config = model_config.hf_config
    validate_attention_contract(vllm_config.cache_config)
    if (
        getattr(config, "num_hash_layers", 0)
        or getattr(config, "scoring_func", "sqrtsoftplus") == "sqrtsoftplus"
    ):
        require_model_ops("topk_hash_softplus_sqrt")
    if getattr(config, "swiglu_limit", None) is not None:
        raise RuntimeError(
            "ixaite DeepSeek-V4 routed MoE requires unavailable clamped activation quantization"
        )
    require_model_ops("ds4_mhc_pre", "ds4_mhc_post", "ds4_hc_head")


# A missing repacker must fail before deleting or changing checkpoint parameters.
def repack_ds4_wo_a_w8a16(layer, *, gemm_format="k16n16") -> None:
    repack, reason = load_ixaite_python_op(
        "marlin_w8_weight_repack", "marlin_w8_weight_repack"
    )
    if repack is None:
        raise RuntimeError(f"ixaite DeepSeek-V4 requires Marlin W8 repack: {reason}")
    raise RuntimeError(
        "ixaite DeepSeek-V4 Marlin W8 repack integration has not been validated"
    )
