# SPDX-License-Identifier: Apache-2.0
"""Iluvatar override for ``vllm.model_executor.warmup.jit_warmup_triton_helper._triton_key_deriver``.

Corresponds to: vllm/model_executor/warmup/jit_warmup_triton_helper.py
Target   : _triton_key_deriver
Baseline : v0.31.0 (db9527a468)
Why      : CoreX Triton commit ``8e6f69a0c9d18fee94d1726b5ed6c3f3c594a63e``
           (``[SWFW-3370] Keep CQ fp32 dots off the TCU``) made
           ``triton.runtime.jit.compute_cache_key`` require a trailing
           ``backend`` argument and key specialization through
           ``backend.cache_key_specialization``. Stock / community callers
           (including vLLM JIT warmup) still use the 3-arg form, which raises
           ``TypeError: compute_cache_key() missing 1 required positional
           argument: 'backend'`` on first Triton JIT warmup.
How      : Capture ``backend`` from ``kernel.device_caches[device]`` and forward
           it to ``compute_cache_key``. Inspect the live ``compute_cache_key``
           signature once and fall back to the stock 3-arg call when ``backend``
           is not required, so the override works on both CoreX and stock Triton.
Scope    : every V1 Triton JIT kernel warmup on Iluvatar.
Remove   : when CoreX Triton restores a community-compatible
           ``compute_cache_key`` API (``backend`` optional / 3-arg callable).
           **Currently waiting on that Triton fix.**
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Mapping
from typing import Any, cast

from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonJitKey
from vllm.triton_utils import triton


def _triton_key_deriver(
    kernel: Any,
) -> Callable[[Mapping[str, Any]], set[TritonJitKey]]:
    """Prepare Triton's key derivation once for one kernel and device."""
    knobs = triton.knobs
    Autotuner = triton.runtime.autotuner.Autotuner
    Heuristics = triton.runtime.autotuner.Heuristics
    driver = triton.runtime.driver
    JITFunction = triton.runtime.jit.JITFunction
    compute_cache_key = triton.runtime.jit.compute_cache_key

    # >>> ILU MODIFICATION >>>
    # CoreX Triton 8e6f69a0 made ``backend`` required on compute_cache_key;
    # stock Triton does not. Decide once which call form to use. Waiting on
    # Triton to restore a community-compatible (optional / 3-arg) API.
    try:
        _needs_backend = "backend" in inspect.signature(compute_cache_key).parameters
    except (ValueError, TypeError):
        _needs_backend = False
    # <<< ILU MODIFICATION <<<

    if isinstance(kernel, Heuristics):
        derive_inner = _triton_key_deriver(kernel.fn)

        def derive_heuristic(kwargs: Mapping[str, Any]) -> set[TritonJitKey]:
            heuristic_kwargs = dict(kwargs)
            for name, heuristic in kernel.values.items():
                heuristic_kwargs[name] = heuristic(heuristic_kwargs)
            return derive_inner(heuristic_kwargs)

        return derive_heuristic

    if isinstance(kernel, Autotuner):
        derive_inner = _triton_key_deriver(kernel.fn)

        def derive_autotuned(kwargs: Mapping[str, Any]) -> set[TritonJitKey]:
            previous_nargs = getattr(kernel, "nargs", None)
            kernel.nargs = {}
            try:
                configs = kernel.prune_configs(dict(kwargs))
            finally:
                kernel.nargs = previous_nargs
            keys: set[TritonJitKey] = set()
            for config in configs:
                conflicts = kwargs.keys() & config.kwargs.keys()
                if conflicts:
                    names = ", ".join(sorted(conflicts))
                    raise ValueError(f"Conflicting autotune parameters: {names}")
                keys.update(derive_inner(dict(kwargs) | config.all_kwargs()))
            return keys

        return derive_autotuned

    if not isinstance(kernel, JITFunction):
        raise TypeError(f"Unsupported Triton kernel wrapper: {type(kernel).__name__}")

    device = cast(Hashable, driver.active.get_current_device())
    # >>> ILU MODIFICATION >>>
    # Capture ``backend`` (4th element) when the installed Triton needs it.
    if _needs_backend:
        _, kernel_key_cache, _, backend, binder = kernel.device_caches[device]
    else:
        _, kernel_key_cache, _, _, binder = kernel.device_caches[device]
    # <<< ILU MODIFICATION <<<
    jit_function_id = id(kernel)
    jit_function_key = cast(Hashable, kernel.cache_key)
    debug = kernel.debug or knobs.runtime.debug
    instrumentation_mode = knobs.compilation.instrumentation_mode
    fpsan_homomorphic_casts = getattr(
        knobs.compilation, "fpsan_homomorphic_casts", None
    )
    inspection_hook = knobs.runtime.add_stages_inspection_hook

    def derive_jit(kwargs: Mapping[str, Any]) -> set[TritonJitKey]:
        binder_kwargs = dict(kwargs)
        binder_kwargs["debug"] = binder_kwargs.get("debug", debug) or debug
        binder_kwargs["instrumentation_mode"] = instrumentation_mode
        if fpsan_homomorphic_casts is not None:
            binder_kwargs["fpsan_homomorphic_casts"] = fpsan_homomorphic_casts
        _, specialization, options = binder(**binder_kwargs)
        if inspection_hook is not None:
            _, inspection_hash = inspection_hook()
            specialization.append(f'("custom_pipeline", {inspection_hash})')
        # >>> ILU MODIFICATION >>>
        # Forward ``backend`` for CoreX Triton 8e6f69a0+ (required arg).
        if _needs_backend:
            cache_key = cast(
                Hashable,
                compute_cache_key(kernel_key_cache, specialization, options, backend),
            )
        else:
            cache_key = cast(
                Hashable,
                compute_cache_key(kernel_key_cache, specialization, options),
            )
        # <<< ILU MODIFICATION <<<
        return {
            TritonJitKey(
                jit_function_id,
                jit_function_key,
                device,
                cache_key,
            )
        }

    return derive_jit
