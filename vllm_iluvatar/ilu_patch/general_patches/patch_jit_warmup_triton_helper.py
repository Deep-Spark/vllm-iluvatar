# SPDX-License-Identifier: Apache-2.0
"""Wiring for the Iluvatar JIT-warmup ``compute_cache_key`` override.

Corresponds to: vllm/model_executor/warmup/jit_warmup_triton_helper.py
Target   : vllm.model_executor.warmup.jit_warmup_triton_helper._triton_key_deriver
Baseline : v0.30.0 (ced6857afa)
Why      : The CoreX (Iluvatar) Triton build extends
           ``triton.runtime.jit.compute_cache_key`` with an extra required
           ``backend`` positional argument
           (``compute_cache_key(kernel_key_cache, specialization, options, backend)``).
           The community JIT-warmup helper was written against stock Triton,
           whose ``compute_cache_key`` takes only
           ``(kernel_key_cache, specialization, options)``, so its ``derive_jit``
           closure raises
           ``TypeError: compute_cache_key() missing 1 required positional
           argument: 'backend'`` the first time any Triton JIT kernel is warmed
           up.  This surfaces as
           ``RuntimeError: Worker failed with error 'compute_cache_key() missing
           1 required positional argument: 'backend''`` from
           ``compile_or_warm_up_model`` during KV-cache init, e.g. DeepSeek-V2-Lite
           under Elastic EP / DPMoE (DP+EP) which exercises the standard
           fused-MoE / sampling Triton kernels.
How      : Fingerprint the live community ``_triton_key_deriver`` source against
           the baseline snapshot below, then rebind the function to the
           override in ``vllm_iluvatar.model_executor.warmup.jit_warmup_triton_helper``,
           which captures ``backend`` from ``kernel.device_caches[device]`` and
           forwards it.  The override inspects the live ``compute_cache_key``
           signature and falls back to the stock 3-arg call when ``backend`` is
           not required, so it works against both CoreX and stock Triton.
Scope    : every V1 Triton JIT kernel warmup on Iluvatar.
Remove   : when community's JIT-warmup helper forwards ``backend`` to
           ``compute_cache_key`` (or CoreX Triton drops the ``backend`` arg).
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
    # ``compute_cache_key`` for the CoreX Triton build (extra trailing arg).
    p.rebind_module_func(_m, "_triton_key_deriver", impl._triton_key_deriver)
