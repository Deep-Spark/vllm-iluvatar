# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for vLLM JIT monitor on older Triton.

Corresponds to: vllm/utils/jit_monitor.py

Target   : vllm.utils.jit_monitor.{_setup_triton_autotuning_print,
           _setup_triton_jit_hook}
Baseline : main_6e448d0
Why      : community jit_monitor imports ``triton.knobs`` to register Triton
           autotune/JIT hooks. Older Triton builds used in the Iluvatar runtime
           do not expose ``knobs``, so worker startup fails during
           ``activate_jit_monitor``.
How      : replace only the Triton setup helpers with guarded versions. Missing
           ``triton.knobs`` disables Triton JIT monitoring, while CuTeDSL and
           TileLang monitors remain active.
Scope    : JIT monitor setup only.
Remove   : when the Iluvatar runtime ships a Triton version exposing
           ``triton.knobs``.
"""

from __future__ import annotations

from vllm_iluvatar.ilu_patch import general_patch as p


def _get_triton_knobs(jit_monitor):
    try:
        from triton import knobs  # type: ignore[import-untyped]
    except ImportError:
        jit_monitor.logger.debug(
            "Triton knobs API is unavailable; skipping Triton JIT monitor hooks."
        )
        return None
    return knobs


def _make_setup_triton_autotuning_print(jit_monitor):
    def _setup_triton_autotuning_print() -> None:
        if not jit_monitor.HAS_TRITON:
            return
        knobs = _get_triton_knobs(jit_monitor)
        if knobs is None:
            return

        user_val = jit_monitor.os.environ.get("TRITON_PRINT_AUTOTUNING")
        if user_val == "0":
            jit_monitor.logger.debug(
                "TRITON_PRINT_AUTOTUNING=0 set by user; "
                "autotuning messages will stay suppressed."
            )
            return

        knobs.autotuning.print = True

    return _setup_triton_autotuning_print


def _make_setup_triton_jit_hook(jit_monitor):
    def _setup_triton_jit_hook() -> None:
        if not jit_monitor.HAS_TRITON:
            return
        knobs = _get_triton_knobs(jit_monitor)
        if knobs is None:
            return

        existing_hook = knobs.runtime.jit_post_compile_hook

        def _on_jit_compile(**kwargs):
            fn = kwargs.get("fn")
            fn_name = getattr(fn, "name", "<unknown>")
            jit_monitor._log_triton_jit_compile(fn_name, kwargs)
            if existing_hook is not None:
                return existing_hook(**kwargs)
            return None

        knobs.runtime.jit_post_compile_hook = _on_jit_compile

    return _setup_triton_jit_hook


def _apply() -> None:
    from vllm.utils import jit_monitor

    jit_monitor._setup_triton_autotuning_print = (  # type: ignore[attr-defined]
        _make_setup_triton_autotuning_print(jit_monitor)
    )
    jit_monitor._setup_triton_jit_hook = _make_setup_triton_jit_hook(  # type: ignore[attr-defined]
        jit_monitor
    )


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="guard jit_monitor when Triton knobs API is unavailable",
    )
