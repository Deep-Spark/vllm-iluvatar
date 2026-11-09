# SPDX-License-Identifier: Apache-2.0
"""Cached operator bindings for the ixaite integration."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import cache

import torch
from vllm.logger import init_logger

from vllm_iluvatar.backends import provider_mode
from vllm_iluvatar.ilu_util.arch import cuda_arch

logger = init_logger("vllm." + __name__)

_CATALOG_MISS = "is not in this catalog"


# Import and initialize the optional kernel library once per worker process.
@cache
def _load_ixaite_runtime() -> tuple[str | None, str]:
    try:
        import ixaite
    except (ImportError, ModuleNotFoundError) as error:
        return None, f"ixaite cannot be imported: {error}"

    arch = cuda_arch()
    if arch is None:
        return None, "the Iluvatar GPU architecture cannot be detected"
    try:
        ixaite.initialize(arch)
    except (ImportError, ModuleNotFoundError) as error:
        return None, f"the ixaite FlyDSL backend is unavailable: {error}"
    return arch, f"initialized for {arch}"


# Import one ixaite op module and confirm it supports the initialized arch.
def _resolve_ixaite_op_module(op_name: str) -> tuple[object | None, str]:
    arch, reason = _load_ixaite_runtime()
    if arch is None:
        return None, reason

    try:
        op_module = importlib.import_module(f"ixaite.ops.{op_name}")
    except ModuleNotFoundError as error:
        missing = error.name or ""
        expected = f"ixaite.ops.{op_name}"
        if missing in {expected, op_name}:
            return None, f"ixaite operator {op_name} {_CATALOG_MISS}: {error}"
        return None, f"ixaite operator {op_name} cannot be imported: {error}"
    except ImportError as error:
        return None, f"ixaite operator {op_name} cannot be imported: {error}"

    supported, reason = op_module.is_supported(arch)
    if not supported:
        return None, reason or f"{op_name} is not supported on {arch}"
    return op_module, f"supported on {arch}"


# Resolve one framework-neutral ixaite torch op without per-op loader functions.
@cache
def load_ixaite_torch_op(
    op_name: str,
) -> tuple[Callable[..., object] | None, str]:
    op_module, reason = _resolve_ixaite_op_module(op_name)
    if op_module is None:
        return None, reason

    try:
        implementation = getattr(torch.ops.ixaite, op_name).default
    except AttributeError:
        return None, f"torch.ops.ixaite.{op_name} is not registered"
    return implementation, reason


# Resolve a module-level ixaite Python callable for ops whose ABI carries
# non-tensor arguments (e.g. attention metadata) that a torch custom-op
# schema cannot express. The callable still enters the FlyDSL custom op
# internally, so Dynamo never traces the JIT launch path.
@cache
def load_ixaite_python_op(
    op_name: str, func_name: str
) -> tuple[Callable[..., object] | None, str]:
    op_module, reason = _resolve_ixaite_op_module(op_name)
    if op_module is None:
        return None, reason

    implementation = getattr(op_module, func_name, None)
    if implementation is None:
        return None, f"ixaite.ops.{op_name}.{func_name} is not defined"
    return implementation, reason


def is_ixaite_op_enabled(op_name: str) -> bool:
    """Return whether startup provider selection enables an ixaite operator."""
    if provider_mode() == "baseline":
        return False
    implementation, _ = load_ixaite_torch_op(op_name)
    return implementation is not None


def is_ixaite_python_op_enabled(op_name: str, func_name: str) -> bool:
    """Return whether selection enables an ixaite Python operator."""
    if provider_mode() == "baseline":
        return False
    implementation, _ = load_ixaite_python_op(op_name, func_name)
    return implementation is not None


__all__ = [
    "is_ixaite_op_enabled",
    "is_ixaite_python_op_enabled",
    "load_ixaite_python_op",
    "load_ixaite_torch_op",
]
