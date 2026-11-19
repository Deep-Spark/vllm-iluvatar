# SPDX-License-Identifier: Apache-2.0
"""Load-time execution preferences, independent of checkpoint quantization."""

from vllm.config import get_current_vllm_config


def compute_options(section: str) -> dict:
    config = get_current_vllm_config()
    raw = (config.additional_config or {}).get("ilu_compute", {})
    if not isinstance(raw, dict):
        raise ValueError("ilu_compute must be an object")
    if set(raw) - {"schema_version", "linear", "moe"}:
        raise ValueError("Unknown ilu_compute field")
    if (
        type(raw.get("schema_version", 1)) is not int
        or raw.get("schema_version", 1) != 1
    ):
        raise ValueError("ilu_compute.schema_version must be 1")
    allowed = {"linear": {"fp8_backend"}, "moe": {"mode"}}
    for key, fields in allowed.items():
        value = raw.get(key, {})
        if not isinstance(value, dict) or set(value) - fields:
            raise ValueError(f"Invalid ilu_compute.{key} fields")
    return raw.get(section, {})


def fp8_linear_backend() -> str:
    from vllm_iluvatar.ilu_util.arch import cuda_arch

    backend = compute_options("linear").get("fp8_backend", "auto")
    if backend not in ("auto", "fp8", "w8a8_int8"):
        raise ValueError(f"Unsupported FP8 Linear backend: {backend!r}")
    arch = cuda_arch()
    if backend == "auto":
        return "w8a8_int8" if arch == "ivcore11" else "fp8"
    if backend == "fp8" and arch != "ivcore30":
        raise ValueError(f"Explicit FP8 Linear backend is not supported on {arch}")
    if backend == "w8a8_int8" and arch not in ("ivcore11", "ivcore30"):
        raise ValueError(f"INT8 W8A8 Linear backend is not supported on {arch}")
    return backend


# Public four-bit expert execution modes. Backend/layout mechanics remain internal; source
# compatibility and the supported platform matrix belong to the Method policy.
_EXPERT_MODES = {
    "ivcore11": {
        "ixinfer_w4a8_nn_v2": ("ixinfer_w4a8", "NN"),
        "ixinfer_w4a8_tn_v2": ("ixinfer_w4a8", "TN"),
    },
    "ivcore30": {
        "ixmm_fp4fp8": ("ixmm_fp4fp8", "fp4fp8"),
        "ixmm_w4a8_nn": ("ixmm_w4a8", "NN"),
    },
}
_EXPERT_DEFAULTS = {"ivcore11": "ixinfer_w4a8_nn_v2", "ivcore30": "ixmm_fp4fp8"}


def select_four_bit_expert_compute(arch, mode, *, source):
    """Resolve one public mode before loading; reject unsupported overrides."""
    from vllm_iluvatar.model_executor.layers.fused_moe.experts.ilu_compute_backend import (
        select_moe_backend,
    )

    modes = _EXPERT_MODES.get(arch, {})
    if mode == "auto":
        mode = _EXPERT_DEFAULTS.get(arch)
    if not isinstance(mode, str) or mode not in modes:
        raise ValueError(
            f"Unsupported {source} mode {mode!r} on {arch}; supported: {tuple(modes)}"
        )
    name, layout = modes[mode]
    return select_moe_backend(arch, name=name, layout=layout)
