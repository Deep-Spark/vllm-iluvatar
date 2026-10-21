# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar JIT-warmup ``compute_cache_key`` override.

Corresponds to: vllm/model_executor/warmup/jit_warmup_triton_helper.py
Target   : vllm.model_executor.warmup.jit_warmup_triton_helper._triton_key_deriver
Baseline : v0.31.0 (db9527a468)
Why      : CoreX Triton commit ``8e6f69a0c9d18fee94d1726b5ed6c3f3c594a63e``
           (``[SWFW-3370] Keep CQ fp32 dots off the TCU``) changed
           ``triton.runtime.jit.compute_cache_key`` from the stock 3-arg form
           ``(kernel_key_cache, specialization, options)`` to a 4-arg form that
           requires ``backend``
           (``compute_cache_key(kernel_key_cache, specialization, options, backend)``)
           so specialization can be keyed via ``backend.cache_key_specialization``.
           Community vLLM's JIT-warmup ``_triton_key_deriver`` still calls the
           stock 3-arg API, so ``derive_jit`` raises
           ``TypeError: compute_cache_key() missing 1 required positional
           argument: 'backend'`` on the first Triton JIT warmup. That surfaces
           as ``RuntimeError: Worker failed with error 'compute_cache_key()
           missing 1 required positional argument: 'backend''`` from
           ``compile_or_warm_up_model`` during KV-cache init (e.g. DeepSeek /
           fused-MoE / sampling kernels under DP+EP).
How      : Fingerprint the live community ``_triton_key_deriver`` source against
           the baseline snapshot below, then rebind to the override in
           ``vllm_iluvatar.model_executor.warmup.jit_warmup_triton_helper``,
           which captures ``backend`` from ``kernel.device_caches[device]`` and
           forwards it. The override inspects the live ``compute_cache_key``
           signature and falls back to the stock 3-arg call when ``backend`` is
           not required, so it works against both CoreX and stock Triton.
Scope    : every V1 Triton JIT kernel warmup on Iluvatar.
Remove   : when CoreX Triton restores a community-compatible
           ``compute_cache_key`` API (e.g. ``backend`` optional / 3-arg callable).
           **Currently waiting on that Triton fix**; do not drop this patch
           until the installed CoreX Triton signature is compatible again.
Fingerprint: the live community ``_triton_key_deriver`` source is checked against
           ``_TRITON_KEY_DERIVER`` before rebinding; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of ``_triton_key_deriver`` (module-level
# free function), used as the drift anchor. Keep in sync with
# vllm/vllm/model_executor/warmup/jit_warmup_triton_helper.py.
_TRITON_KEY_DERIVER = '''\
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
    _, kernel_key_cache, _, _, binder = kernel.device_caches[device]
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
        cache_key = cast(
            Hashable,
            compute_cache_key(kernel_key_cache, specialization, options),
        )
        return {
            TritonJitKey(
                jit_function_id,
                jit_function_key,
                device,
                cache_key,
            )
        }

    return derive_jit
'''


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="compute_cache_key backend arg for CoreX Triton",
    )


def _apply() -> None:
    from vllm.model_executor.warmup import jit_warmup_triton_helper as _m

    import vllm_iluvatar.model_executor.warmup.jit_warmup_triton_helper as impl

    p.check_fingerprint(
        "vllm.model_executor.warmup.jit_warmup_triton_helper._triton_key_deriver",
        inspect.getsource(_m._triton_key_deriver),
        _TRITON_KEY_DERIVER,
        resync_hint=(
            "Refresh _TRITON_KEY_DERIVER to the new community source, then "
            "review whether the backend-arg override in "
            "vllm_iluvatar/model_executor/warmup/jit_warmup_triton_helper.py "
            "still applies."
        ),
    )
    p.check_signatures(
        _m,
        impl,
        ["_triton_key_deriver"],
        strict=True,
    )

    # [ILU override] _triton_key_deriver — delta: forward ``backend`` to
    # ``compute_cache_key`` after CoreX Triton 8e6f69a0 made it required.
    # Waiting on Triton to restore a community-compatible signature.
    p.rebind_module_func(_m, "_triton_key_deriver", impl._triton_key_deriver)
