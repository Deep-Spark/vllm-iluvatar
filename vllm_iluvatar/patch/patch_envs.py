"""Inject iluvatar-specific environment variables into ``vllm.envs``.

``vllm.envs`` exposes variables through a module-level ``__getattr__`` that
evaluates ``environment_variables[name]()`` lazily on first access. We can
therefore add brand-new variables and override the default of a handful of
existing ones simply by mutating that dict -- as long as we do it before the
value is first read (we run first in ``adapt_patch``).

The internal iluvatar fork adds ~18 new ``VLLM_*`` knobs (W4A8/W8A8 MoE,
linear/attn opt levels, MLA customization, ...) and changes the default of
three existing ones.
"""

from __future__ import annotations

import os

_PATCHED = False


def _b(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def _new_vars() -> dict:
    return {
        # --- W4A8 / W8A8 MoE knobs (iluvatar DeepSeek int4/int8) ---------
        "VLLM_W8A8_MOE_USE_W4A8": lambda: _b("VLLM_W8A8_MOE_USE_W4A8"),
        "VLLM_WNA16_MOE_USE_W4A8": lambda: _b("VLLM_WNA16_MOE_USE_W4A8"),
        "VLLM_W4A8_FORMAT": lambda: os.environ.get("VLLM_W4A8_FORMAT", "TN").upper(),
        "VLLM_W4A8_VERSION": lambda: int(os.environ.get("VLLM_W4A8_VERSION", "2")),
        "VLLM_W8A8_FORMAT": lambda: os.environ.get("VLLM_W8A8_FORMAT", "TN").upper(),
        "VLLM_USE_INT8_MLA": lambda: _b("VLLM_USE_INT8_MLA"),
        "VLLM_MLA_CUSTOMIZE": lambda: _b("VLLM_MLA_CUSTOMIZE", "1"),
        "VLLM_MIX_QUANTIZATION_TYPE": lambda: os.environ.get(
            "VLLM_MIX_QUANTIZATION_TYPE", ""
        ).upper(),
        # --- cuda graph -------------------------------------------------
        "VLLM_ENFORCE_CUDA_GRAPH": lambda: bool(
            int(os.getenv("VLLM_ENFORCE_CUDA_GRAPH", "0"))
        ),
        # --- linear / moe / attn opt levels -----------------------------
        "VLLM_MOE_OPT_LEVEL": lambda: int(os.getenv("VLLM_MOE_OPT_LEVEL", "0")),
        "VLLM_LINEAR_OPT_LEVEL": lambda: int(os.getenv("VLLM_LINEAR_OPT_LEVEL", "0")),
        "VLLM_OPT_EXCLUDE_LAYERS": lambda: os.environ.get(
            "VLLM_OPT_EXCLUDE_LAYERS", ""
        ).upper(),
        "VLLM_LINEAR_SPECIFIED_LAYERS": lambda: os.environ.get(
            "VLLM_LINEAR_SPECIFIED_LAYERS", ""
        ).upper(),
        "VLLM_LINEAR_SPECIFIED_KEYS": lambda: os.environ.get(
            "VLLM_LINEAR_SPECIFIED_KEYS", ""
        ).lower(),
        "VLLM_LINEAR_SPECIFIED_OPT_LEVEL": lambda: int(
            os.getenv("VLLM_LINEAR_SPECIFIED_OPT_LEVEL", "0")
        ),
        "VLLM_ATTN_OPT_LEVEL": lambda: int(os.environ.get("VLLM_ATTN_OPT_LEVEL", "0")),
        "VLLM_ATTN_STATIC_QUANT_SCALE_FILE_PATH": lambda: os.environ.get(
            "VLLM_ATTN_STATIC_QUANT_SCALE_FILE_PATH", ""
        ),
        # --- misc -------------------------------------------------------
        "VLLM_USE_LORA_FUSION": lambda: _b("VLLM_USE_LORA_FUSION"),
    }


def _overridden_defaults() -> dict:
    """Existing vars whose iluvatar default differs from community."""
    return {
        "VLLM_USE_FLASHINFER_SAMPLER": lambda: (
            bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
            if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ
            else False
        ),
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM": lambda: bool(
            int(os.getenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1"))
        ),
        "VLLM_USE_V2_MODEL_RUNNER": lambda: _maybe_bool(
            os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        ),
    }


def _maybe_bool(value):
    if value is None:
        return None
    return bool(int(value))


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import vllm.envs as envs
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    table = getattr(envs, "environment_variables", None)
    if not isinstance(table, dict):
        logger.warning(
            "vllm-iluvatar: vllm.envs.environment_variables not found; "
            "skipping env injection"
        )
        return

    new = _new_vars()
    overrides = _overridden_defaults()
    table.update(new)
    # Only override defaults the user has NOT explicitly set, to avoid
    # surprising users who rely on community defaults.
    for name, fn in overrides.items():
        table[name] = fn

    # Drop any values already materialized on the module so __getattr__
    # re-evaluates through the updated table.
    for name in list(new) + list(overrides):
        if name in envs.__dict__:
            del envs.__dict__[name]

    logger.info(
        "vllm-iluvatar: injected %d env vars, overrode %d defaults",
        len(new),
        len(overrides),
    )
