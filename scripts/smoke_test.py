#!/usr/bin/env python3
"""Standalone sanity probe for vllm-iluvatar.

Run on a machine that already has **community** vLLM installed (branch
``main_plugin_969aec4bc84``) and this plugin installed via ``pip install -e .``.

Checks (all run on a CPU-only / no-checkpoint box):

1. The plugin entry point is discoverable.
2. ``vllm.plugins.load_general_plugins()`` triggers our
   :func:`vllm_iluvatar.register_model`.
3. The new ``IluDeepseekV4ForCausalLM`` architecture is registered and the
   ``ilu_deepseek_v4_w4a8`` quant method is known.
4. The override finder is installed and discovers the mirror-tree modules;
   ``find_spec`` for an overridden module resolves into ``_vllm_overrides``.
5. The three ilu attention backends are added to ``AttentionBackendEnum``.
6. The new iluvatar env vars are injected into ``vllm.envs``.
7. ``vllm._custom_ops`` has the ilu ops (full splat or 4-op fallback).
8. ``ModelConfig.__post_init__`` is wrapped and ``get_quantization_config``
   is remapped.

The script does NOT instantiate a model / attention backend, so it never
imports ixformer. End-to-end accuracy / perf is the job of the modelzoo
wrappers.
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

    # 4. Override finder installed + discovers the mirror tree, and
    #    find_spec for an overridden module resolves into _vllm_overrides.
    try:
        import sys as _sys

        from vllm_iluvatar.patch import patch_sys_modules as psm

        names = set(psm._iter_override_module_names(psm._overrides_root()))
        if not _expect(
            "override finder discovers mirror-tree modules",
            len(names) > 100,
            detail=f"{len(names)} modules",
        ):
            failures.append("override_count")

        finder = next(
            (f for f in _sys.meta_path if isinstance(f, psm._IluRedirectFinder)),
            None,
        )
        if not _expect(
            "override finder installed on sys.meta_path", finder is not None
        ):
            failures.append("override_finder")
        elif finder is not None:
            spec = finder.find_spec("vllm.model_executor.models.qwen3")
            ok = bool(spec and spec.origin and "_vllm_overrides" in spec.origin)
            if not _expect(
                "qwen3 resolves into _vllm_overrides",
                ok,
                detail=(spec.origin if spec else "no spec"),
            ):
                failures.append("override_resolve")

            # Comm stack overlay capability (served only when the .pth
            # bootstrap installed the finder before `import vllm`).
            cspec = finder.find_spec("vllm.distributed.parallel_state")
            if not _expect(
                "parallel_state resolvable into _vllm_overrides (comm overlay)",
                bool(cspec and cspec.origin and "_vllm_overrides" in cspec.origin),
                detail=(cspec.origin if cspec else "no spec"),
            ):
                failures.append("comm_overlay_capability")

        # Informational: did the bootstrap actually take effect this run?
        try:
            import vllm.distributed.parallel_state as ps

            active = "_vllm_overrides" in (getattr(ps, "__file__", "") or "")
            _expect(
                "comm optimization active (bootstrap .pth installed)",
                True,
                detail=(
                    "ixformer overlay in use"
                    if active
                    else "community NCCL (run scripts/install_bootstrap.py to enable)"
                ),
            )
        except Exception:
            pass
    except Exception as e:
        _expect("override finder", False, detail=repr(e))
        failures.append("override_finder")

    # 5. ilu attention backends added to AttentionBackendEnum.
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        members = set(AttentionBackendEnum.__members__)
        want = {"ILU_FLASH_ATTN", "ILU_MLA", "ILU_FLASHMLA_SPARSE"}
        if not _expect(
            "ilu attention backends registered",
            want.issubset(members),
            detail=f"missing: {sorted(want - members) or 'none'}",
        ):
            failures.append("attn_backends")
    except Exception as e:
        _expect("attention backend registration", False, detail=repr(e))
        failures.append("attn_backends")

    # 6. iluvatar env vars injected.
    try:
        import vllm.envs as envs

        ok = envs.VLLM_W4A8_FORMAT in ("TN", "NN") and isinstance(
            envs.VLLM_MOE_OPT_LEVEL, int
        )
        if not _expect(
            "iluvatar env vars injected (VLLM_W4A8_FORMAT/VLLM_MOE_OPT_LEVEL)", ok
        ):
            failures.append("envs")
    except Exception as e:
        _expect("env injection", False, detail=repr(e))
        failures.append("envs")

    # 7. vllm._custom_ops patch spot-check.
    try:
        import vllm._custom_ops as _co

        if not _expect(
            "vllm._custom_ops.silu_and_mul_with_clamp present",
            hasattr(_co, "silu_and_mul_with_clamp"),
        ):
            failures.append("custom_ops")
    except Exception as e:
        _expect("vllm._custom_ops patch", False, detail=repr(e))
        failures.append("custom_ops")

    # 8. ModelConfig wrap + quant remap markers.
    try:
        from vllm.config.model import ModelConfig

        if not _expect(
            "ModelConfig.__post_init__ is wrapped",
            getattr(ModelConfig.__post_init__, "_ilu_patched", False),
        ):
            failures.append("model_config_patch")

        import vllm.model_executor.layers.quantization as q

        if not _expect(
            "get_quantization_config is remapped",
            getattr(q.get_quantization_config, "_ilu_wrapped", False),
        ):
            failures.append("quant_remap")
    except Exception as e:
        _expect("model config / quant remap", False, detail=repr(e))
        failures.append("model_config_patch")

    print()
    if failures:
        print(f"FAIL: {len(failures)} probe(s) failed: {failures}")
        return 1
    print("All probes passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
