#!/usr/bin/env python3
"""Standalone sanity probe for vllm-iluvatar.

Run on a machine that already has the iluvatar fork of vLLM installed
(branch ``iluvatar_main``, NO ``support_dsv4`` patches applied) and the
plugin installed via ``pip install -e .``.

Checks:

1. The plugin entry point is discoverable through Python's standard
   ``importlib.metadata.entry_points``.
2. ``vllm.plugins.load_general_plugins()`` triggers our
   :func:`vllm_iluvatar.register_model`.
3. The new architecture and quant method are registered.
4. The 6 ``sys.modules`` redirects loaded the support_dsv4 versions
   (we spot-check ``IluDeepseekV4FlashMLASparseBackend`` which is the
   new class only present in the support_dsv4 version).
5. ``vllm._custom_ops.silu_and_mul_with_clamp`` is now bound to our
   Python wrapper (proxy for: all 4 ``_custom_ops`` patches applied).
6. ``ModelConfig.__post_init__`` is wrapped (proxy: the tokenizer-mode
   patch is in place).

The script does NOT instantiate a model or attention backend so it can
run on a CPU-only / no-checkpoint box. Anything more end-to-end is the
job of ``run_mmlu_accuracy.sh``.
"""

from __future__ import annotations

import os
import sys

# Avoid the cwd-shadowing trap: if the user runs this from
# /Users/jian.wang/Documents/wj/work/llm/ where there's a sibling
# directory called ``vllm_iluvatar`` (with underscore, unrelated April
# scratch dir), CPython picks that up as a namespace package and shadows
# our real plugin install. Strip "" from sys.path so we always go
# through the installed package.
sys.path = [p for p in sys.path if p not in ("", os.getcwd())]


def _expect(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    failures: list[str] = []

    # 1. Discover plugin via entry_points.
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="vllm.general_plugins")
        names = {ep.name for ep in eps}
        if not _expect(
            "entry_point 'vllm_iluvatar' discoverable",
            "vllm_iluvatar" in names,
            detail=f"found: {sorted(names) or '(none)'}",
        ):
            failures.append("entry_point")
    except Exception as e:
        _expect("entry_point discovery", False, detail=str(e))
        failures.append("entry_point")

    # 2 + 3. Trigger plugin loading and verify registrations.
    try:
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        from vllm import ModelRegistry

        archs = set(ModelRegistry.get_supported_archs())
        if not _expect(
            "IluDeepseekV4ForCausalLM registered",
            "IluDeepseekV4ForCausalLM" in archs,
        ):
            failures.append("model_arch")

        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

        if not _expect(
            "quant method 'ilu_deepseek_v4_w4a8' registered",
            "ilu_deepseek_v4_w4a8" in QUANTIZATION_METHODS,
        ):
            failures.append("quant_method")
    except Exception as e:
        _expect("plugin load + registration", False, detail=repr(e))
        failures.append("plugin_load")

    # 4. sys.modules redirect spot-check.
    try:
        from vllm.v1.attention.backends.mla import ilu_flashmla_sparse

        if not _expect(
            "ilu_flashmla_sparse has IluDeepseekV4FlashMLASparseBackend",
            hasattr(ilu_flashmla_sparse, "IluDeepseekV4FlashMLASparseBackend"),
            detail=f"module loaded from {ilu_flashmla_sparse.__file__}",
        ):
            failures.append("redirect_ilu_flashmla")
    except Exception as e:
        _expect("ilu_flashmla_sparse import", False, detail=repr(e))
        failures.append("redirect_ilu_flashmla")

    # 5. vllm._custom_ops patch spot-check.
    try:
        import vllm._custom_ops as _co

        # The Python wrapper lives in vllm_iluvatar.ops; we just check the
        # attribute exists (the upstream module doesn't ship it).
        if not _expect(
            "vllm._custom_ops.silu_and_mul_with_clamp present",
            hasattr(_co, "silu_and_mul_with_clamp"),
        ):
            failures.append("custom_ops")
        if not _expect(
            "vllm._custom_ops.router_gemm_bf16_fp32 present",
            hasattr(_co, "router_gemm_bf16_fp32"),
        ):
            failures.append("router_gemm")
    except Exception as e:
        _expect("vllm._custom_ops patch", False, detail=repr(e))
        failures.append("custom_ops")

    # 6. ModelConfig.__post_init__ wrapper marker.
    try:
        from vllm.config.model import ModelConfig

        if not _expect(
            "ModelConfig.__post_init__ is wrapped",
            getattr(ModelConfig.__post_init__, "_ilu_patched", False),
        ):
            failures.append("model_config_patch")
    except Exception as e:
        _expect("model config patch", False, detail=repr(e))
        failures.append("model_config_patch")

    print()
    if failures:
        print(f"FAIL: {len(failures)} probe(s) failed: {failures}")
        return 1
    print("All probes passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
