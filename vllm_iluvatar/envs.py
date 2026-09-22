# SPDX-License-Identifier: Apache-2.0
"""
Iluvatar-specific environment variable definitions and community-default
overrides.  Implementation layer — no side effects on import.

Corresponds to: vllm/envs.py
Applied by:      vllm_iluvatar.register_platform()

Naming policy:
  - Canonical name: VLLM_ILU_<suffix> (Iluvatar namespace, kept apart from
    real community vars).
  - Legacy alias:   VLLM_<suffix> (the original, non-standard fork name).
  - Both `envs.VLLM_ILU_X` and `envs.VLLM_X` resolve to the same value, and
    both OS env keys are honoured (VLLM_ILU_X preferred, then VLLM_X, then
    default).  Existing call sites (envs.VLLM_X) and launch scripts keep
    working unchanged; new code should migrate to VLLM_ILU_X.
  - NIXL community vars are used directly with their community names.
  - Iluvatar-only NIXL vars use VLLM_ILU_NIXL_* only and do not register
    legacy aliases.

Contents:
  1. _ILU_DEFAULTS    — community vars whose default differs for Iluvatar
                        (NOT renamed; they are real community vars).
  2. _ILU_ENV_VARS    — new Iluvatar env vars with legacy aliases.
  3. _ILU_CANONICAL_ONLY_ENV_VARS — Iluvatar-only vars without aliases.
  4. _ILU_ENV_ALIASES — legacy VLLM_* names mapped to aliased readers.
  5. apply()          — inject defaults + register env readers.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

_TRUE_VALS = ("1", "true", "yes", "on")


def _bool_reader(name: str, default: str) -> Callable[[], bool]:
    return lambda: os.environ.get(name, default).strip().lower() in _TRUE_VALS


def _int_reader(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> Callable[[], int]:
    def read() -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", name, raw, default)
            return default
        if minimum is not None:
            return max(minimum, value)
        return value

    return read


# Community default -> Iluvatar default. These are real community vars, so they
# are NOT renamed; we only change their default via os.environ.setdefault.
#   VLLM_USE_FLASHINFER_SAMPLER       : "1"   -> "0"
#   VLLM_DISABLE_SHARED_EXPERTS_STREAM: "0"   -> "1"
#   VLLM_MOE_SKIP_PADDING             : "1"   -> "0"
_ILU_DEFAULTS: dict[str, str] = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
    "VLLM_MOE_SKIP_PADDING": "0",
}


def _get(new: str, old: str, default: str) -> str:
    """Read OS env, preferring the canonical VLLM_ILU_* name, then the legacy
    VLLM_* name, then the default."""
    val = os.environ.get(new)
    if val is not None:
        return val
    return os.environ.get(old, default)


def _maybe_stripped(value: str) -> str | None:
    value = value.strip()
    return value or None


def _hybrid_dp_sync_prefills_default() -> bool:
    return (
        os.environ.get("VLLM_ILU_DECODE_EP_BACKEND", "").strip() == "deepep_low_latency"
        and os.environ.get("VLLM_ILU_DP_ROUND_ROBIN_LB", "0").strip().lower()
        in _TRUE_VALS
    )


def _gdn_attn_backend_default() -> str:
    """Enable the tuned ixformer GDN path by default on CoreX30."""
    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    return "1" if is_ivcore30() else "0"


if TYPE_CHECKING:
    # Canonical names (new code should use these).
    VLLM_ILU_MLA_CUSTOMIZE: bool = True
    VLLM_ILU_MIX_QUANTIZATION_TYPE: str = ""
    VLLM_ILU_FLASH_ATTN_QUANT_TYPE: str = ""
    VLLM_ILU_FLASH_ATTN_V_SCALE_PATH: str = ""
    VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH: str = ""
    VLLM_ILU_MLA_ATTN_QUANT_TYPE: str = ""
    VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE: int = 0
    VLLM_ILU_USE_LORA_FUSION: bool = False
    VLLM_ILU_VIT_ATTN_OPT_LEVEL: int = 0
    VLLM_ILU_VIT_ATTN_LAYER: str = ""
    VLLM_ILU_GDN_ATTN_BACKEND: bool = False
    VLLM_ILU_DS4_PREFILL_OPT: bool = False
    VLLM_ILU_DS4_PREFILL_AS_DECODE: bool = False
    VLLM_ILU_DS4_C4A_FUSED_INDEXCOMPRESS: bool = False
    VLLM_ILU_DS4_MQA_QUANT_TYPE: str = ""
    VLLM_ILU_FUSE_NORM_QUANT: bool = True
    VLLM_ILU_FUSE_MOE_W13: bool = True
    VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV: bool = True
    VLLM_ILU_GLM_DSA_FP8_ABSORB_UP: bool = True
    VLLM_ILU_GLM_DSA_TOPK_CONVERT_CACHE: bool = True
    VLLM_ILU_GLM_DSA_NATIVE_MQA_LOGITS: bool = False
    VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER: bool = False
    VLLM_ILU_NIXL_HOST_DTYPE: str = "auto"
    VLLM_ILU_NIXL_PROXY_ADDR: str = ""
    VLLM_ILU_NIXL_SCHEDULER_ADDR: str = ""
    VLLM_ILU_NIXL_FORCE_PER_RANK_SEND: bool = False
    VLLM_ILU_NIXL_SERVER_ADDR: str = ""
    VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER: bool = False
    VLLM_ILU_LINEAR_BACKEND: str = ""
    VLLM_ILU_DP_LB_EXACT_INFLIGHT: bool = False
    VLLM_ILU_DEEPEP_USE_INT8: int = 0
    VLLM_ILU_DISABLE_DEEPEP: bool = False
    VLLM_ILU_DEEPEP_USE_SCATTER: bool = True
    VLLM_ILU_LL_FUSED_AG_AR: bool = True
    VLLM_ILU_DECODE_EP_BACKEND: str | None = None
    VLLM_ILU_DP_ROUND_ROBIN_LB: bool = False
    VLLM_ILU_DP_ROUND_ROBIN_LB_FILE: str = ""
    VLLM_ILU_DP_ENABLE_SYNC_PREFILLS: bool = False
    VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS: int = 8
    VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS: float = 0.0
    VLLM_ILU_MOE_EP_LL_MOE_ALGO: str = "GEMV"
    VLLM_ILU_ENABLE_PP_MTP: bool = False
    VLLM_ILU_ENABLE_V2_PP_MTP: bool = False
    VLLM_ILU_MXFP4_MOE_BACKEND: str = "ixmm"
    VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES: int = 128 * 1024 * 1024
    # GLM DSA fp8 KV: single float shared by ALL layers. Prefer PATH for
    # per-layer tables; empty -> online amax/448 on first eager insert.
    VLLM_ILU_GLM_DSA_FP8_KV_SCALE: str = ""
    VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH: str = ""

# Canonical (VLLM_ILU_*) readers. Each lambda preserves the exact parsing the
# fork used in vllm/envs.py, and reads VLLM_ILU_* first then the legacy VLLM_*.
_ILU_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_MLA_CUSTOMIZE": lambda: (
        _get("VLLM_ILU_MLA_CUSTOMIZE", "VLLM_MLA_CUSTOMIZE", "1").lower()
        in ("1", "true", "yes")
    ),
    "VLLM_ILU_MIX_QUANTIZATION_TYPE": lambda: _get(
        "VLLM_ILU_MIX_QUANTIZATION_TYPE", "VLLM_MIX_QUANTIZATION_TYPE", ""
    ).upper(),
    "VLLM_ILU_USE_LORA_FUSION": lambda: (
        _get("VLLM_ILU_USE_LORA_FUSION", "VLLM_USE_LORA_FUSION", "0").lower()
        in ("1", "true")
    ),
    # ViT attention quantization level
    "VLLM_ILU_VIT_ATTN_OPT_LEVEL": lambda: int(
        _get("VLLM_ILU_VIT_ATTN_OPT_LEVEL", "VLLM_VIT_ATTN_OPT_LEVEL", "0")
    ),
    # Comma-separated ViT block indices on which VLLM_VIT_ATTN_OPT_LEVEL
    # takes effect (e.g. "0,1,2,5,10").
    "VLLM_ILU_VIT_ATTN_LAYER": lambda: _get(
        "VLLM_ILU_VIT_ATTN_LAYER", "VLLM_VIT_ATTN_LAYER", ""
    ).upper(),
    # Gate the Iluvatar GDN attention backend. CoreX30 defaults to ixformer;
    # other arches keep the community path. Explicit env values always win.
    "VLLM_ILU_GDN_ATTN_BACKEND": lambda: (
        _get(
            "VLLM_ILU_GDN_ATTN_BACKEND",
            "VLLM_GDN_ATTN_BACKEND",
            _gdn_attn_backend_default(),
        ).lower()
        in _TRUE_VALS
    ),
    "VLLM_ILU_DS4_PREFILL_OPT": lambda: (
        _get("VLLM_ILU_DS4_PREFILL_OPT", "VLLM_DS4_PREFILL_OPT", "0").lower()
        in ("1", "true")
    ),
    # Route DeepSeek-V4 prefill through the per-token *decode* sparse MLA
    # kernel (ds4_flash_mla_with_kvcache) instead of the workspace-gather
    # prefill kernel. bf16 KV cache only. See
    # IluDeepseekV4MLAAttention._forward_prefill_as_decode.
    "VLLM_ILU_DS4_PREFILL_AS_DECODE": lambda: (
        _get(
            "VLLM_ILU_DS4_PREFILL_AS_DECODE", "VLLM_DS4_PREFILL_AS_DECODE", "0"
        ).lower()
        in ("1", "true")
    ),
    # Collapse a C4A layer's indexer compressor + weights_proj + main
    # compressor into the single ``ds4_c4a_indexcompress`` op (one concatenated
    # GEMM plus one task-split paged post). Only eligible on decode batches
    # (plain or MTP verify) with a bf16 main KV cache; see
    # IluDeepseekV4MultiHeadLatentAttentionWrapper._use_c4a_fused.
    "VLLM_ILU_DS4_C4A_FUSED_INDEXCOMPRESS": lambda: (
        _get(
            "VLLM_ILU_DS4_C4A_FUSED_INDEXCOMPRESS",
            "VLLM_DS4_C4A_FUSED_INDEXCOMPRESS",
            "0",
        ).lower()
        in ("1", "true")
    ),
}


_ILU_CANONICAL_ONLY_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_FLASH_ATTN_QUANT_TYPE": lambda: os.environ.get(
        "VLLM_ILU_FLASH_ATTN_QUANT_TYPE", ""
    ).upper(),
    "VLLM_ILU_FLASH_ATTN_TOTAL_KV_HEADS": lambda: int(
        os.environ.get("VLLM_ILU_FLASH_ATTN_TOTAL_KV_HEADS", "-1")
    ),
    "VLLM_ILU_MLA_ATTN_QUANT_TYPE": lambda: os.environ.get(
        "VLLM_ILU_MLA_ATTN_QUANT_TYPE", ""
    ).upper(),
    # Cap for MLA chunked-prefill workspace rows. 0 = keep community size.
    "VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE": lambda: int(
        os.environ.get("VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE", "0")
    ),
    "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH": lambda: os.environ.get(
        "VLLM_ILU_FLASH_ATTN_V_SCALE_PATH", ""
    ),
    "VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH": lambda: os.environ.get(
        "VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH", ""
    ),
    "VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER": _bool_reader(
        "VLLM_ILU_NIXL_WRITE_USE_HOST_BUFFER", "0"
    ),
    "VLLM_ILU_NIXL_HOST_DTYPE": lambda: os.environ.get(
        "VLLM_ILU_NIXL_HOST_DTYPE", "auto"
    ).lower(),
    "VLLM_ILU_NIXL_PROXY_ADDR": lambda: os.environ.get("VLLM_ILU_NIXL_PROXY_ADDR", ""),
    "VLLM_ILU_NIXL_SCHEDULER_ADDR": lambda: os.environ.get(
        "VLLM_ILU_NIXL_SCHEDULER_ADDR", ""
    ),
    "VLLM_ILU_NIXL_FORCE_PER_RANK_SEND": _bool_reader(
        "VLLM_ILU_NIXL_FORCE_PER_RANK_SEND", "0"
    ),
    "VLLM_ILU_NIXL_SERVER_ADDR": lambda: os.environ.get(
        "VLLM_ILU_NIXL_SERVER_ADDR", ""
    ),
    "VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER": _bool_reader(
        "VLLM_ILU_NIXL_DS32_QUANT_AFTER_TRANSFER", "0"
    ),
    "VLLM_ILU_LINEAR_BACKEND": lambda: (
        os.environ.get("VLLM_ILU_LINEAR_BACKEND", "").strip().lower().replace("-", "_")
    ),
    "VLLM_ILU_DP_LB_EXACT_INFLIGHT": _bool_reader("VLLM_ILU_DP_LB_EXACT_INFLIGHT", "0"),
    "VLLM_ILU_DEEPEP_USE_INT8": lambda: int(
        os.environ.get("VLLM_ILU_DEEPEP_USE_INT8", "0")
    ),
    "VLLM_ILU_DISABLE_DEEPEP": _bool_reader("VLLM_ILU_DISABLE_DEEPEP", "0"),
    "VLLM_ILU_DEEPEP_USE_SCATTER": _bool_reader("VLLM_ILU_DEEPEP_USE_SCATTER", "1"),
    "VLLM_ILU_LL_FUSED_AG_AR": _bool_reader("VLLM_ILU_LL_FUSED_AG_AR", "1"),
    "VLLM_ILU_DECODE_EP_BACKEND": lambda: _maybe_stripped(
        os.environ.get("VLLM_ILU_DECODE_EP_BACKEND", "")
    ),
    "VLLM_ILU_DP_ROUND_ROBIN_LB": _bool_reader("VLLM_ILU_DP_ROUND_ROBIN_LB", "0"),
    "VLLM_ILU_DP_ROUND_ROBIN_LB_FILE": lambda: os.environ.get(
        "VLLM_ILU_DP_ROUND_ROBIN_LB_FILE", ""
    ),
    "VLLM_ILU_DP_ENABLE_SYNC_PREFILLS": _bool_reader(
        "VLLM_ILU_DP_ENABLE_SYNC_PREFILLS",
        "1" if _hybrid_dp_sync_prefills_default() else "0",
    ),
    "VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS": lambda: max(
        0,
        int(
            os.environ.get(
                "VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS",
                "8",
            )
        ),
    ),
    "VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS": lambda: max(
        0.0,
        float(
            os.environ.get(
                "VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS",
                "5" if _hybrid_dp_sync_prefills_default() else "0",
            )
        ),
    ),
    "VLLM_ILU_MOE_EP_LL_MOE_ALGO": lambda: os.environ.get(
        "VLLM_ILU_MOE_EP_LL_MOE_ALGO", "GEMV"
    ).upper(),
    # Opt-in gate for the PP + MTP / speculative-decode compatibility
    # patch (ilu_patch/general_patches/patch_pp_mtp.py). Default off so the
    # plugin leaves community vLLM untouched unless explicitly enabled.
    "VLLM_ILU_ENABLE_PP_MTP": _bool_reader("VLLM_ILU_ENABLE_PP_MTP", "0"),
    # Opt-in gate for V2 ModelRunner PP + MTP compatibility
    # (ilu_patch/general_patches/patch_v2_pp_mtp.py). Kept separate from the
    # legacy V1 PP+MTP patch because the patched code paths are different.
    "VLLM_ILU_ENABLE_V2_PP_MTP": _bool_reader("VLLM_ILU_ENABLE_V2_PP_MTP", "0"),
    # Selects the ilu_mxfp4_fused_experts compute backend. Defaults to "ixmm";
    # "auto" -> ixmm when available else torch; "torch" forces the pure-torch
    # fallback; "ixformer_native" is the correctness-only native kernel.
    "VLLM_ILU_MXFP4_MOE_BACKEND": lambda: os.environ.get(
        "VLLM_ILU_MXFP4_MOE_BACKEND", "ixmm"
    )
    .strip()
    .lower(),
    "VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES": _int_reader(
        "VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES",
        128 * 1024 * 1024,
        minimum=1,
    ),
    # DeepSeek-V4 fp8 MQA Q quant mode. Empty = bf16 Q (default).
    # "Q_NOPE_FP8" = dynamically quantize Q NoPE [0:448] to fp8 then dequant
    # before ds4_flash_mla_with_fp8_kvcache.
    "VLLM_ILU_DS4_MQA_QUANT_TYPE": lambda: os.environ.get(
        "VLLM_ILU_DS4_MQA_QUANT_TYPE", ""
    ).upper(),
    # IluGlm residual/RMSNorm+INT8 fusion. Default on (TP8 Graph ITL benefit);
    # set 0 to force the split RMSNorm path. Fail-closed at the call site.
    "VLLM_ILU_FUSE_NORM_QUANT": _bool_reader("VLLM_ILU_FUSE_NORM_QUANT", "1"),
    # Compact-row W13+SwiGLU+INT8 fusion (native ixformer). Default on after
    # TP8 Graph ITL A/B showed benefit; set 0 for the split GEMV path.
    # Fail-closed for unsupported shapes / runtime errors.
    "VLLM_ILU_FUSE_MOE_W13": _bool_reader("VLLM_ILU_FUSE_MOE_W13", "1"),
    # Fuse kv_a RMSNorm + Q/K RoPE + FP8 KV pack/insert for GLM DSA.
    # Default on; requires FP8 KV + frozen scale + ixformer op. Set 0 to disable.
    "VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV": _bool_reader(
        "VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV", "1"
    ),
    # GLM DSA official-FP8 checkpoint: run the absorbed k-up / v-up as fp8
    # (fp8_grouped_per_token_quant + fp8_per_channel_einsum) instead of BF16
    # grouped einsums off a dequantised kv_b_proj. Default on; set 0 for the
    # BF16 A/B baseline. Ignored by the W4A8 (marlin) path.
    "VLLM_ILU_GLM_DSA_FP8_ABSORB_UP": _bool_reader(
        "VLLM_ILU_GLM_DSA_FP8_ABSORB_UP", "1"
    ),
    # Indexer mqa-logits: 1 = ixformer_native narrow tile, 0 = ixinfer (16 slots).
    "VLLM_ILU_GLM_DSA_NATIVE_MQA_LOGITS": _bool_reader(
        "VLLM_ILU_GLM_DSA_NATIVE_MQA_LOGITS", "0"
    ),
    # Cache request-local -> global top-k conversion across shared DSA layers.
    "VLLM_ILU_GLM_DSA_TOPK_CONVERT_CACHE": _bool_reader(
        "VLLM_ILU_GLM_DSA_TOPK_CONVERT_CACHE", "1"
    ),
    # GLM DSA 640B fp8 packed KV scale. A non-empty float applies the SAME
    # scale to every layer (not per-layer). Prefer PATH for per-layer JSON
    # ``{prefix: {"kv": float}}``. Empty + empty PATH -> online amax/448.
    "VLLM_ILU_GLM_DSA_FP8_KV_SCALE": lambda: os.environ.get(
        "VLLM_ILU_GLM_DSA_FP8_KV_SCALE", ""
    ),
    "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH": lambda: os.environ.get(
        "VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH", ""
    ),
    # Optional override for MessageQueue max_chunks (see patch_message_queue).
    # Empty / unset → auto: broadcast max(10, 2*n_reader+8), response max 48.
    "VLLM_ILU_MQ_MAX_CHUNKS": lambda: os.environ.get("VLLM_ILU_MQ_MAX_CHUNKS", ""),
}


def _canonical_to_legacy(name: str) -> str:
    """VLLM_ILU_FOO -> VLLM_FOO."""
    return "VLLM_" + name[len("VLLM_ILU_") :]


# Legacy VLLM_* aliases -> same reader as the canonical name. Keeps existing
# `envs.VLLM_X` call sites working.
_ILU_ENV_ALIASES: dict[str, Callable[[], Any]] = {
    _canonical_to_legacy(canonical): reader
    for canonical, reader in _ILU_ENV_VARS.items()
}


# Iluvatar plugin control vars that are read directly via os.environ where they
# are used (NOT through this module) — registered here only so that
# vllm.envs.validate_environ() recognises them and does not warn (or, with
# fail_on_environ_validation, hard-fail) about "Unknown vLLM environment
# variable". Sources:
#   VLLM_ILU_PATCH_CHECK_SIGNATURE / _CHECK_FINGERPRINT / _VERBOSE
#       -> vllm_iluvatar/ilu_patch/general_patch.py
#   VLLM_ILUVATAR_ENABLE_STARTUP_PATCH
#       -> vllm_iluvatar/__init__.py (STARTUP_PATCH_ENV)
_ILU_GUARD_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_PATCH_CHECK_SIGNATURE": _bool_reader(
        "VLLM_ILU_PATCH_CHECK_SIGNATURE", "1"
    ),
    "VLLM_ILU_PATCH_CHECK_FINGERPRINT": _bool_reader(
        "VLLM_ILU_PATCH_CHECK_FINGERPRINT", "1"
    ),
    "VLLM_ILU_PATCH_VERBOSE": _bool_reader("VLLM_ILU_PATCH_VERBOSE", "0"),
    "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH": _bool_reader(
        "VLLM_ILUVATAR_ENABLE_STARTUP_PATCH", "0"
    ),
}


def apply() -> None:
    """Inject Iluvatar env defaults and register env readers."""
    for key, value in _ILU_DEFAULTS.items():
        os.environ.setdefault(key, value)
        logger.debug("ILU envs: setdefault %s=%s", key, value)

    import vllm.envs as envs

    envs.environment_variables.update(_ILU_ENV_VARS)
    envs.environment_variables.update(_ILU_ENV_ALIASES)
    envs.environment_variables.update(_ILU_CANONICAL_ONLY_ENV_VARS)
    envs.environment_variables.update(_ILU_GUARD_ENV_VARS)
    logger.debug(
        "ILU envs: registered %d canonical + %d legacy-alias + "
        "%d canonical-only + %d guard env vars",
        len(_ILU_ENV_VARS),
        len(_ILU_ENV_ALIASES),
        len(_ILU_CANONICAL_ONLY_ENV_VARS),
        len(_ILU_GUARD_ENV_VARS),
    )
