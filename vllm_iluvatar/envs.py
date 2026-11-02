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


def _step5_bool(suffix: str, default: str) -> bool:
    """Step5 switches accept exactly 0/1, including through legacy aliases."""
    name = f"VLLM_ILU_STEP5_{suffix}"
    old = f"VLLM_STEP5_{suffix}"
    value = _get(name, old, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} (legacy {old}) must be 0 or 1, got {value!r}")
    return value == "1"


def _step5_fp8_backend() -> str:
    value = _get(
        "VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND",
        "VLLM_STEP5_FP8_ATTENTION_BACKEND",
        "ixformer_native",
    )
    if value not in ("ixformer_native", "ixattn"):
        raise ValueError(
            "VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND must be ixformer_native or ixattn"
        )
    return value


def _maybe_stripped(value: str) -> str | None:
    value = value.strip()
    return value or None


def _decode_ep_backend() -> str | None:
    value = _maybe_stripped(os.environ.get("VLLM_ILU_DECODE_EP_BACKEND", ""))
    if value not in (None, "naive", "deepep_low_latency"):
        raise ValueError(
            f"Invalid VLLM_ILU_DECODE_EP_BACKEND: {value!r}; "
            "expected 'naive' or 'deepep_low_latency'."
        )
    return value


def _hybrid_dp_sync_prefills_default() -> bool:
    return (
        os.environ.get("VLLM_ILU_DECODE_EP_BACKEND", "").strip() == "deepep_low_latency"
        and os.environ.get("VLLM_ILU_DP_ROUND_ROBIN_LB", "0").strip().lower()
        in _TRUE_VALS
    )


def _glm_sampling_mode() -> str:
    """Read the GLM sampling mode through the registered environment reader."""
    value = os.environ.get("VLLM_ILU_GLM_SAMPLING_MODE", "off").strip().lower()
    choices = ("off", "sorted", "radix")
    if value not in choices:
        raise ValueError(
            f"VLLM_ILU_GLM_SAMPLING_MODE must be one of {choices}, got {value!r}"
        )
    return value


def _ds4_attn_mode() -> str:
    value = os.environ.get("VLLM_ILU_DS4_ATTN_MODE", "packed").strip().lower()
    if value not in ("packed", "plain_direct"):
        raise ValueError(f"invalid VLLM_ILU_DS4_ATTN_MODE: {value!r}")
    return value


def _moe_runtime_layout() -> str | None:
    value = os.environ.get("VLLM_ILU_MOE_RUNTIME_LAYOUT")
    if value is None:
        return None
    value = value.strip().upper()
    if value not in ("NN", "TN"):
        raise ValueError("VLLM_ILU_MOE_RUNTIME_LAYOUT must be NN or TN")
    return value


def _gdn_attn_backend_default() -> str:
    """Enable the tuned ixformer GDN path by default on CoreX30."""
    from vllm_iluvatar.ilu_util.arch import is_ivcore30

    return "1" if is_ivcore30() else "0"


if TYPE_CHECKING:
    # Canonical names (new code should use these).
    VLLM_ILU_STEP5_SPARSE_ATTENTION: bool = True
    VLLM_ILU_STEP5_SPARSE_FP8: bool = False
    VLLM_ILU_STEP5_FP32_TP: bool = False
    VLLM_ILU_STEP5_FP32_EP: bool = False
    VLLM_ILU_STEP5_FP8_KV_SCALE_PATH: str = ""
    VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND: str = "ixformer_native"
    VLLM_ILU_STEP5_FP4_SMALL_TILES: bool = False
    VLLM_ILU_MLA_CUSTOMIZE: bool = True
    VLLM_ILU_MIX_QUANTIZATION_TYPE: str = ""
    VLLM_ILU_FLASH_ATTN_QUANT_TYPE: str = ""
    VLLM_ILU_FLASH_ATTN_V_SCALE_PATH: str = ""
    VLLM_ILU_FLASH_ATTN_FP8_SCALE_PATH: str = ""
    VLLM_ILU_MLA_ATTN_QUANT_TYPE: str = ""
    VLLM_ILU_MLA_CHUNKED_PREFILL_MAX_WS_SIZE: int = 0
    VLLM_ILU_HYV4_FUSE_IHC: bool = True
    VLLM_ILU_USE_LORA_FUSION: bool = False
    VLLM_ILU_VIT_ATTN_OPT_LEVEL: int = 0
    VLLM_ILU_VIT_ATTN_LAYER: str = ""
    VLLM_ILU_GDN_ATTN_BACKEND: bool = False
    VLLM_ILU_CUDAGRAPH_ASYNC_REPLAY: bool = False
    VLLM_ILU_COMPACT_RAGGED_GATHER: bool = False
    VLLM_ILU_DP_VOCAB_GROUP_SIZE: int = 0
    VLLM_ILU_FUSE_EP_COMBINE: bool = False
    VLLM_ILU_E25_COMPACT_ENGRAM: bool = False
    VLLM_ILU_DS41_DECODE_KV_RESTORE: bool = False
    VLLM_ILU_DS41_GROUPED_INT8_O_PROJ: bool = True
    VLLM_ILU_DS41_WOA_INPUT_MODE: str = "reader_int8"
    VLLM_ILU_DS41_PRODUCER_TAIL_FFN: bool = False
    VLLM_ILU_DS4_CANDIDATE_FILTER: bool = False
    VLLM_ILU_DS4_PREFILL_OPT: bool = False
    VLLM_ILU_DS4_PREFILL_AS_DECODE: bool = False
    VLLM_ILU_DS4_FP8_PREFILL_SPARSE: bool = True
    VLLM_ILU_DS4_ATTN_MODE: str = "packed"
    VLLM_ILU_DS4_C4A_FUSED_INDEXCOMPRESS: bool = False
    VLLM_ILU_WO_A_QUANT: str = "marlin"
    VLLM_ILU_WO_A_MSD_MIN_TOKENS: int = 256
    VLLM_ILU_DS4_MQA_QUANT_TYPE: str = ""
    VLLM_ILU_DS41_INDEXER_KV_LAYOUT: str = "legacy"
    VLLM_ILU_DS41_COMPRESSED_KV_FORMAT: str = "none"
    VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED: bool = False
    VLLM_ILU_ONCHIP_CHECK: bool = False
    VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK: int = 0
    VLLM_ILU_ONCHIP_CHECK_KMSG: bool = False
    VLLM_ILU_ONCHIP_CHECK_STARTUP_RETRIES: int = 2
    VLLM_ILU_ONCHIP_CHECK_RETRY_DELAY_MS: float = 100.0
    VLLM_ILU_ONCHIP_CHECK_INCOMPLETE_LOG_INTERVAL: int = 100
    VLLM_ILU_HBM_ECC_CHECK: bool = False
    VLLM_ILU_HBM_ECC_POLL_INTERVAL_MS: float = 1000.0
    VLLM_ILU_HBM_ECC_HEARTBEAT_TIMEOUT_MS: float = 5000.0
    VLLM_ILU_HBM_ECC_MAX_QUERY_FAILURES: int = 3
    VLLM_ILU_HBM_ECC_KMSG: bool = False
    VLLM_ILU_HBM_ECC_KILL_PROCESS_GROUP: bool = False
    VLLM_ILU_FUSE_NORM_QUANT: bool = True
    VLLM_ILU_FUSE_MOE_W13: bool = True
    VLLM_ILU_GLM_SAMPLING_MODE: str = "off"
    VLLM_ILU_GLM_MOE_PREPARE_FUSION: bool = False
    VLLM_ILU_GLM_DSA_PREPARE_FUSION: bool = False
    VLLM_ILU_GLM_DSA_RMS_ROPE_FP8_KV: bool = True
    VLLM_ILU_GLM_DSA_FP8_ABSORB_UP: bool = True
    VLLM_ILU_GLM_DSA_TOPK_CONVERT_CACHE: bool = True
    VLLM_ILU_GLM_DSA_NATIVE_MQA_LOGITS: bool = False
    VLLM_ILU_INDEXER_KV_DTYPE: str = "bf16"
    VLLM_ILU_GLM5NEXT_MLA_POOL4_DECODE: bool = True
    VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK: bool = False
    VLLM_ILU_GLM5NEXT_DRAFT_HEAD_W8A8: bool = True
    VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8: bool = True
    VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8: bool = True
    VLLM_ILU_GLM5NEXT_GATHER_BEFORE_GATE: bool = True
    VLLM_ILU_GLM_LL_SHARED_DISPATCH_OVERLAP: bool = False
    VLLM_ILU_MOE_AG_SHARED_OVERLAP: bool = False
    VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP: bool = True
    VLLM_ILU_GLM_DSA_INDEXER_PREPARE_FUSION: bool = True
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
    VLLM_ILU_DP_SYNC_READY_REPLAY: bool = False
    VLLM_ILU_DP_PREFILL_REARM: bool = True
    VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS: int = 8
    VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS: float = 0.0
    VLLM_ILU_PP_PREFILL_COALESCE_MS: float = 0.0
    VLLM_ILU_PP_MAX_CONCURRENT_BATCHES: int = 0
    VLLM_ILU_ENABLE_V2_PP_TOKEN_P2P: bool = False

    VLLM_ILU_MOE_RUNTIME_LAYOUT: str | None = None
    VLLM_ILU_MOE_W4A8_GEMV_PERF: bool = False
    VLLM_ILU_MOE_EP_LL_MOE_ALGO: str = "GEMV"
    VLLM_ILU_MOE_EP_RS_TP_AG: bool = False
    VLLM_ILU_MXFP4_MOE_BACKEND: str = "ixmm"
    VLLM_ILU_REJECTION_SAMPLER_MAX_CHUNK_BYTES: int = 128 * 1024 * 1024
    # GLM DSA fp8 KV: single float shared by ALL layers. Prefer PATH for
    # per-layer tables; empty -> online amax/448 on first eager insert.
    VLLM_ILU_GLM_DSA_FP8_KV_SCALE: str = ""
    VLLM_ILU_GLM_DSA_FP8_KV_SCALE_PATH: str = ""
    # Cap num_cpus passed to ray.init() for fresh local cluster starts, so the
    # raylet does not pre-start host-count workers and thrash the driver connect
    # in containers without a cgroup CPU quota. See
    # ilu_patch/general_patches/patch_ray_init_num_cpus.py.
    VLLM_ILU_RAY_NUM_CPUS: int = 16

# Canonical (VLLM_ILU_*) readers. Each lambda preserves the exact parsing the
# fork used in vllm/envs.py, and reads VLLM_ILU_* first then the legacy VLLM_*.
_ILU_ENV_VARS: dict[str, Callable[[], Any]] = {
    "VLLM_ILU_DS4_ATTN_MODE": _ds4_attn_mode,
    "VLLM_ILU_DS4_FP8_PREFILL_SPARSE": lambda: (
        os.environ.get("VLLM_ILU_DS4_FP8_PREFILL_SPARSE", "1").lower() in ("1", "true")
    ),
    # Step5 retains strict 0/1 parsing and canonical-first legacy compatibility.
    "VLLM_ILU_STEP5_SPARSE_ATTENTION": lambda: _step5_bool("SPARSE_ATTENTION", "1"),
    "VLLM_ILU_STEP5_SPARSE_FP8": lambda: _step5_bool("SPARSE_FP8", "0"),
    "VLLM_ILU_STEP5_FP32_TP": lambda: _step5_bool("FP32_TP", "0"),
    "VLLM_ILU_STEP5_FP32_EP": lambda: _step5_bool("FP32_EP", "0"),
    "VLLM_ILU_STEP5_FP8_KV_SCALE_PATH": lambda: _get(
        "VLLM_ILU_STEP5_FP8_KV_SCALE_PATH", "VLLM_STEP5_FP8_KV_SCALE_PATH", ""
    ),
    "VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND": _step5_fp8_backend,
    "VLLM_ILU_STEP5_FP4_SMALL_TILES": lambda: _step5_bool("FP4_SMALL_TILES", "0"),
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
    # attn.wo_a grouped BMM backend: marlin | msd | w8a8. Read directly from
    # os.environ in models/deepseek_v4/attention.py (module-level constant),
    # and through this reader by IluV4W8A16BMMLinearMethod at weight loading.
    "VLLM_ILU_WO_A_QUANT": lambda: (
        os.environ.get("VLLM_ILU_WO_A_QUANT", "marlin").strip().lower()
    ),
    "VLLM_ILU_WO_A_MSD_MIN_TOKENS": _int_reader(
        "VLLM_ILU_WO_A_MSD_MIN_TOKENS", 256, minimum=0
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
    "VLLM_ILU_MOE_RUNTIME_LAYOUT": _moe_runtime_layout,
    # Opt in to supported NN MoE W4A8 GEMV perf kernels; layout is separate.
    "VLLM_ILU_MOE_W4A8_GEMV_PERF": _bool_reader("VLLM_ILU_MOE_W4A8_GEMV_PERF", "0"),
    # Quantize BF16 GLM5Next projections after loading; KDA gates stay BF16.
    "VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK": _bool_reader(
        "VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK", "0"
    ),
    "VLLM_ILU_GLM5NEXT_DRAFT_HEAD_W8A8": _bool_reader(
        "VLLM_ILU_GLM5NEXT_DRAFT_HEAD_W8A8", "1"
    ),
    "VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8": _bool_reader(
        "VLLM_ILU_GLM5NEXT_MLA_PROJ_W8A8", "1"
    ),
    "VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8": _bool_reader(
        "VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8", "1"
    ),
    # Compact-pool BF16 decode by default; set 0 to use paged decode.
    "VLLM_ILU_GLM5NEXT_MLA_POOL4_DECODE": _bool_reader(
        "VLLM_ILU_GLM5NEXT_MLA_POOL4_DECODE", "1"
    ),
    # Gather hidden states before the replicated GLM-5.3 MoE gate on decode
    # batches (IluMoERunner._dispatch_before_gate). Default on: the benchmarks
    # and serving were measured with it enabled. Set 0 for the baseline
    # gate-then-dispatch path.
    "VLLM_ILU_GLM5NEXT_GATHER_BEFORE_GATE": _bool_reader(
        "VLLM_ILU_GLM5NEXT_GATHER_BEFORE_GATE", "1"
    ),
    "VLLM_ILU_INDEXER_KV_DTYPE": lambda: (
        os.environ.get("VLLM_ILU_INDEXER_KV_DTYPE", "bf16").strip().lower()
    ),
    "VLLM_ILU_HYV4_FUSE_IHC": _bool_reader("VLLM_ILU_HYV4_FUSE_IHC", "1"),
    "VLLM_ILU_CUDAGRAPH_ASYNC_REPLAY": _bool_reader(
        "VLLM_ILU_CUDAGRAPH_ASYNC_REPLAY", "0"
    ),
    "VLLM_ILU_COMPACT_RAGGED_GATHER": _bool_reader(
        "VLLM_ILU_COMPACT_RAGGED_GATHER", "0"
    ),
    # AG/RS DP+EP only. Collapses the MoE runner's five collectives per layer
    # (3x dispatch all-gather + DP reduce-scatter + TP all-reduce) into two:
    # one all-gather of hidden_states, and one EP all-reduce that subsumes the
    # reduce-scatter, the TP reduction, and the shared-expert add. Opt in with
    # VLLM_ILU_FUSE_EP_COMBINE=1 after validating workload performance.
    # See IluMoERunner._ilu_fused_ep_combine.
    "VLLM_ILU_FUSE_EP_COMBINE": _bool_reader("VLLM_ILU_FUSE_EP_COMBINE", "0"),
    "VLLM_ILU_E25_COMPACT_ENGRAM": _bool_reader("VLLM_ILU_E25_COMPACT_ENGRAM", "0"),
    "VLLM_ILU_ONCHIP_CHECK": _bool_reader("VLLM_ILU_ONCHIP_CHECK", "0"),
    "VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK": _int_reader(
        "VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK", 0
    ),
    "VLLM_ILU_ONCHIP_CHECK_KMSG": _bool_reader("VLLM_ILU_ONCHIP_CHECK_KMSG", "0"),
    "VLLM_ILU_ONCHIP_CHECK_STARTUP_RETRIES": _int_reader(
        "VLLM_ILU_ONCHIP_CHECK_STARTUP_RETRIES", 2, minimum=0
    ),
    "VLLM_ILU_ONCHIP_CHECK_RETRY_DELAY_MS": lambda: max(
        0.0, float(os.environ.get("VLLM_ILU_ONCHIP_CHECK_RETRY_DELAY_MS", "100"))
    ),
    "VLLM_ILU_ONCHIP_CHECK_INCOMPLETE_LOG_INTERVAL": _int_reader(
        "VLLM_ILU_ONCHIP_CHECK_INCOMPLETE_LOG_INTERVAL", 100, minimum=1
    ),
    "VLLM_ILU_HBM_ECC_CHECK": _bool_reader("VLLM_ILU_HBM_ECC_CHECK", "0"),
    "VLLM_ILU_HBM_ECC_POLL_INTERVAL_MS": lambda: max(
        0.0, float(os.environ.get("VLLM_ILU_HBM_ECC_POLL_INTERVAL_MS", "1000"))
    ),
    "VLLM_ILU_HBM_ECC_HEARTBEAT_TIMEOUT_MS": lambda: max(
        0.0, float(os.environ.get("VLLM_ILU_HBM_ECC_HEARTBEAT_TIMEOUT_MS", "5000"))
    ),
    "VLLM_ILU_HBM_ECC_MAX_QUERY_FAILURES": _int_reader(
        "VLLM_ILU_HBM_ECC_MAX_QUERY_FAILURES", 3, minimum=1
    ),
    "VLLM_ILU_HBM_ECC_KMSG": _bool_reader("VLLM_ILU_HBM_ECC_KMSG", "0"),
    "VLLM_ILU_HBM_ECC_KILL_PROCESS_GROUP": _bool_reader(
        "VLLM_ILU_HBM_ECC_KILL_PROCESS_GROUP", "0"
    ),
    "VLLM_ILU_DS41_DECODE_KV_RESTORE": _bool_reader(
        "VLLM_ILU_DS41_DECODE_KV_RESTORE", "0"
    ),
    "VLLM_ILU_DS41_GROUPED_INT8_O_PROJ": _bool_reader(
        "VLLM_ILU_DS41_GROUPED_INT8_O_PROJ", "1"
    ),
    "VLLM_ILU_DS41_WOA_INPUT_MODE": lambda: os.environ.get(
        "VLLM_ILU_DS41_WOA_INPUT_MODE", "reader_int8"
    ),
    "VLLM_ILU_DS41_PRODUCER_TAIL_FFN": _bool_reader(
        "VLLM_ILU_DS41_PRODUCER_TAIL_FFN", "0"
    ),
    "VLLM_ILU_DS4_CANDIDATE_FILTER": _bool_reader("VLLM_ILU_DS4_CANDIDATE_FILTER", "0"),
    "VLLM_ILU_DS41_COMPRESSED_KV_FORMAT": lambda: os.environ.get(
        "VLLM_ILU_DS41_COMPRESSED_KV_FORMAT", "none"
    ),
    "VLLM_ILU_DS41_INDEXER_KV_LAYOUT": lambda: os.environ.get(
        "VLLM_ILU_DS41_INDEXER_KV_LAYOUT", "legacy"
    ),
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
    "VLLM_ILU_DECODE_EP_BACKEND": _decode_ep_backend,
    "VLLM_ILU_DP_ROUND_ROBIN_LB": _bool_reader("VLLM_ILU_DP_ROUND_ROBIN_LB", "0"),
    "VLLM_ILU_DP_ROUND_ROBIN_LB_FILE": lambda: os.environ.get(
        "VLLM_ILU_DP_ROUND_ROBIN_LB_FILE", ""
    ),
    "VLLM_ILU_DP_ENABLE_SYNC_PREFILLS": _bool_reader(
        "VLLM_ILU_DP_ENABLE_SYNC_PREFILLS",
        "1" if _hybrid_dp_sync_prefills_default() else "0",
    ),
    "VLLM_ILU_DP_SYNC_READY_REPLAY": _bool_reader(
        "VLLM_ILU_DP_SYNC_READY_REPLAY",
        "0",
    ),
    "VLLM_ILU_DP_PREFILL_REARM": _bool_reader("VLLM_ILU_DP_PREFILL_REARM", "1"),
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
    "VLLM_ILU_MOE_EP_RS_TP_AG": _bool_reader("VLLM_ILU_MOE_EP_RS_TP_AG", "0"),
    "VLLM_ILU_MOE_EP_LL_MOE_ALGO": lambda: os.environ.get(
        "VLLM_ILU_MOE_EP_LL_MOE_ALGO", "GEMV"
    ).upper(),
    # Total host wait budget while initially filling a PP pipeline. Opt-in:
    # sparse arrivals can trade up to this delay for fuller prefill batches.
    "VLLM_ILU_PP_PREFILL_COALESCE_MS": lambda: max(
        0.0, float(os.environ.get("VLLM_ILU_PP_PREFILL_COALESCE_MS", "0"))
    ),
    "VLLM_ILU_PP_MAX_CONCURRENT_BATCHES": _int_reader(
        "VLLM_ILU_PP_MAX_CONCURRENT_BATCHES", 0, minimum=0
    ),
    # Opt-in gate for relaying V2 PP sampled/draft tokens with ixfd
    # send/recv instead of a broadcast (ilu_patch/general_patches/
    # patch_v2_pp_token_p2p.py). IXCCL has no copy-engine broadcast, so the
    # community collective leaves a resident RING_LL kernel pending for the
    # whole PP rank skew; per SWDRV-5168 that pins a compute kernel until it
    # retires. Needs NCCL_PRIM_P2P_LEVEL=1 for the copy-engine send/recv path.
    "VLLM_ILU_ENABLE_V2_PP_TOKEN_P2P": _bool_reader(
        "VLLM_ILU_ENABLE_V2_PP_TOKEN_P2P", "0"
    ),
    # Selects the ilu_mxfp4_fused_experts compute backend. Defaults to "ixmm";
    # "auto" -> ixmm when available else torch; "torch" forces the pure-torch
    # fallback; "ixformer_native" is the correctness-only native kernel.
    "VLLM_ILU_MXFP4_MOE_BACKEND": lambda: (
        os.environ.get("VLLM_ILU_MXFP4_MOE_BACKEND", "ixmm").strip().lower()
    ),
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
    # GLM shared-expert SwiGLU + INT8 quant has no environment switch:
    # silu_and_mul_quant is selected for a dynamic symmetric W8A8 down_proj;
    # other quantization schemes keep separate activation and quantization.
    #
    # Fuse kv_a RMSNorm + Q/K RoPE + FP8 KV pack/insert for GLM DSA.
    # Default 1; set 0 for the split path. Requires local Q heads in
    # {8, 16, 32, 64}, FP8 KV cache, a frozen KV scale, and ixformer's native
    # fused_glm_dsa_rmsnorm_rope_fp8_kv_insert op. Unsupported cases fall back
    # to the split path. Online scale calibration uses the split path first;
    # fusion becomes eligible after the scale freezes. Read at model
    # construction: restart the model process after changing this switch.
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
    # GLM-5.2 LL: shared MLP on an auxiliary stream overlaps dispatch.
    "VLLM_ILU_GLM_LL_SHARED_DISPATCH_OVERLAP": _bool_reader(
        "VLLM_ILU_GLM_LL_SHARED_DISPATCH_OVERLAP", "0"
    ),
    # V4.1 decode NVSM/LL: shared MLP runs on the main stream between
    # dispatch_begin and dispatch_finish by default; set 1 to move it to the
    # auxiliary stream.
    "VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP": _bool_reader(
        "VLLM_ILU_V41_LL_SHARED_DISPATCH_OVERLAP", "0"
    ),
    "VLLM_ILU_DP_VOCAB_GROUP_SIZE": _int_reader(
        "VLLM_ILU_DP_VOCAB_GROUP_SIZE", 0, minimum=0
    ),
    "VLLM_ILU_GLM_SAMPLING_MODE": _glm_sampling_mode,
    # Opt-in local routing and INT8 quant+expand; restart workers to change.
    "VLLM_ILU_GLM_MOE_PREPARE_FUSION": _bool_reader(
        "VLLM_ILU_GLM_MOE_PREPARE_FUSION", "0"
    ),
    "VLLM_ILU_GLM_DSA_PREPARE_FUSION": _bool_reader(
        "VLLM_ILU_GLM_DSA_PREPARE_FUSION", "0"
    ),
    # MoERunner AG/RS: overlap local shared MLP with input dispatch.
    "VLLM_ILU_MOE_AG_SHARED_OVERLAP": _bool_reader(
        "VLLM_ILU_MOE_AG_SHARED_OVERLAP", "0"
    ),
    # GLM indexer preparation fusion. Default 1; set 0 for the split path:
    # - glm_dsa_indexer_q_rope_weights: in-place RoPE on Q's first 64 dims
    #   plus softmax_scale and head_scale multiplication of indexer weights.
    # - glm_dsa_indexer_k_norm_rope_cache: K LayerNorm + first-64-dim RoPE
    #   + cache insert, skipping the later standalone cache insert.
    # Requires index_head_dim=128, qk_rope_head_dim=64 and interleaved RoPE.
    # Supports prefill/decode/mixed batches and graph padding slots (-1 does
    # not write cache); dummy/profiling calls without attention metadata use
    # the split preparation path. Missing either Python or native ixformer
    # API raises at initialization.
    # Read at model construction: restart after changing this switch.
    "VLLM_ILU_GLM_DSA_INDEXER_PREPARE_FUSION": _bool_reader(
        "VLLM_ILU_GLM_DSA_INDEXER_PREPARE_FUSION", "1"
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
    # Cap num_cpus for ray.init() fresh local cluster starts (see
    # patch_ray_init_num_cpus). 0/1 are clamped up to 1.
    "VLLM_ILU_RAY_NUM_CPUS": _int_reader("VLLM_ILU_RAY_NUM_CPUS", 16, minimum=1),
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
    "VLLM_ILU_TOP_P_BF16_SORT": _bool_reader("VLLM_ILU_TOP_P_BF16_SORT", "1"),
    "VLLM_ILU_TOP_P_BF16_HIST": _bool_reader("VLLM_ILU_TOP_P_BF16_HIST", "0"),
    "VLLM_ILU_PD_PREFILL_KV_ONLY": _bool_reader("VLLM_ILU_PD_PREFILL_KV_ONLY", "1"),
    "VLLM_ILU_DSPARK_FUSED_MARKOV": _bool_reader("VLLM_ILU_DSPARK_FUSED_MARKOV", "1"),
    "VLLM_ILU_DSPARK_BACKBONE_CG": _bool_reader("VLLM_ILU_DSPARK_BACKBONE_CG", "1"),
    "VLLM_ILU_MERGE_DP_SYNC": _bool_reader("VLLM_ILU_MERGE_DP_SYNC", "1"),
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
    "VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED": _bool_reader(
        "VLLM_ILU_PP_CPU_OFFLOAD_REQUIRED", "0"
    ),
}


def __getattr__(name: str) -> Any:
    """Use the same readers without importing vLLM during host-only checks."""
    for readers in (
        _ILU_ENV_VARS,
        _ILU_ENV_ALIASES,
        _ILU_CANONICAL_ONLY_ENV_VARS,
        _ILU_GUARD_ENV_VARS,
    ):
        if name in readers:
            return readers[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
