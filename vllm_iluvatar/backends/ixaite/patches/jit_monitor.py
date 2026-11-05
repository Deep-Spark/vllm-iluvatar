# Source: patch_jit_monitor_flydsl.py at 47fb804; baseline never installs FlyDSL hooks.
# SPDX-License-Identifier: Apache-2.0
"""Hook FlyDSL ``flyc.compile`` into the community JIT monitor.

Corresponds to: vllm/utils/jit_monitor.py

Target   : ``vllm.utils.jit_monitor.activate`` + ``flydsl.compiler.compile``
Baseline : community jit_monitor covers Triton / CuTeDSL / TileLang only
Why      : ixaite launches through ``flyc.compile``. A request-window miss
           (new prefill bucket or small-M launcher) serializes EngineCore
           for hundreds of milliseconds and shows up as a TTFT cliff, while
           the existing ``_launch_*`` / Triton hooks stay silent.
How      : after warmup activates the monitor, wrap ``flyc.compile`` so a
           later call logs (or raises in ``error`` mode) with launcher name
           and operand shapes.
Scope    : FlyDSL compile entry only; warmup / CUDA-graph capture still
           happen before ``activate()``.
"""

from __future__ import annotations

from typing import Any

from vllm_iluvatar.ilu_patch import general_patch as p

_ILU_ACTIVATE_ATTR = "_ilu_flydsl_monitor"
_ILU_COMPILE_ATTR = "_ilu_flydsl_compile_hook"


def _flydsl_fn_name(fn: object) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if name:
        return str(name)
    return type(fn).__name__


def _format_flydsl_detail(args: tuple[object, ...]) -> str | None:
    shapes: list[object] = []
    scalars: list[int] = []
    for arg in args:
        shape = getattr(arg, "shape", None)
        if shape is not None:
            try:
                shapes.append(tuple(int(dim) for dim in shape))
            except (TypeError, ValueError):
                shapes.append(tuple(shape))
            continue
        if isinstance(arg, bool):
            continue
        if isinstance(arg, int):
            scalars.append(arg)
    parts: list[str] = []
    if shapes:
        parts.append(f"operands={shapes}")
    if scalars:
        parts.append(f"scalars={scalars}")
    return "; ".join(parts) if parts else None


def _install_flydsl_compile_hook(jit_monitor: Any) -> None:
    if getattr(jit_monitor, "_flydsl_hook_installed", False):
        return

    try:
        import flydsl.compiler as flyc
    except Exception:
        jit_monitor.logger.debug(
            "FlyDSL is not available; skipping FlyDSL JIT monitor hook."
        )
        jit_monitor._flydsl_hook_installed = True
        return

    inner = flyc.compile
    if getattr(inner, _ILU_COMPILE_ATTR, None) is True:
        jit_monitor._flydsl_hook_installed = True
        return

    def compile(fn, *args, **kwargs):
        jit_monitor._flydsl_compile_count = (
            getattr(jit_monitor, "_flydsl_compile_count", 0) + 1
        )
        jit_monitor._handle_jit_event(
            backend="FlyDSL",
            event="flyc.compile",
            fn_name=_flydsl_fn_name(fn),
            detail=_format_flydsl_detail(args),
        )
        return inner(fn, *args, **kwargs)

    compile.__name__ = "compile"
    compile.__qualname__ = "compile"
    setattr(compile, _ILU_COMPILE_ATTR, True)
    flyc.compile = compile
    jit_monitor._flydsl_hook_installed = True
    jit_monitor._flydsl_compile_count = getattr(jit_monitor, "_flydsl_compile_count", 0)


def _apply() -> None:
    from vllm.utils import jit_monitor

    orig_activate = jit_monitor.activate
    if getattr(orig_activate, _ILU_ACTIVATE_ATTR, False):
        return

    def activate(*, mode: str = "warn", verbose: bool = False) -> None:
        orig_activate(mode=mode, verbose=verbose)
        _install_flydsl_compile_hook(jit_monitor)

    setattr(activate, _ILU_ACTIVATE_ATTR, True)
    jit_monitor.activate = activate  # type: ignore[method-assign]


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="monitor FlyDSL flyc.compile after warmup",
    )
