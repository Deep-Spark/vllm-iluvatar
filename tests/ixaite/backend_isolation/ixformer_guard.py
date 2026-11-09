# SPDX-License-Identifier: Apache-2.0
"""Deny Python and dispatcher ixformer calls during strict functional tests.

Install before importing plugin consumers so copied function references are
covered. Native launches and graph replay additionally require profiler evidence.
"""

from __future__ import annotations

import importlib
import inspect
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from torch.utils._python_dispatch import TorchDispatchMode

_NAMESPACES = frozenset({"ixf_ops", "_ixformer_C", "ixformer"})
_MODULES = ("ixformer.inference.functions", "ixformer.distributed")


# Raise before a forbidden callable can mutate weights, caches, or GPU state.
def _forbidden(name, violations):
    def reject(*args, **kwargs):
        message = f"strict ixaite test attempted ixformer call: {name}"
        violations.append(message)
        raise AssertionError(message)

    return reject


# Catch direct torch.ops calls in addition to Python facade calls.
class _NoIxformerDispatch(TorchDispatchMode):
    # Keep attempts visible even if warmup catches the immediate assertion.
    def __init__(self, violations):
        super().__init__()
        self.violations = violations

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if name.split("::", 1)[0] in _NAMESPACES:
            message = f"strict ixaite test attempted ixformer op: {name}"
            self.violations.append(message)
            raise AssertionError(message)
        return func(*args, **(kwargs or {}))


# Restore every patched facade and dispatch mode even when the check fails.
@contextmanager
def forbid_ixformer_calls(modules=None):
    if modules is None:
        modules = [importlib.import_module(name) for name in _MODULES]
        modules.append(importlib.import_module("ixformer._C").infer)
    violations = []
    with ExitStack() as stack:
        for module in modules:
            for name, value in vars(module).items():
                if inspect.isroutine(value):
                    stack.enter_context(
                        patch.object(
                            module,
                            name,
                            _forbidden(f"{module.__name__}.{name}", violations),
                        )
                    )
        stack.enter_context(_NoIxformerDispatch(violations))
        try:
            yield
        finally:
            if violations:
                raise AssertionError(violations[0])


# Replay bypasses Python facades; require GPU events and reject native vendor symbols.
def assert_ixaite_kernel_events(events):
    kernels = {event["name"] for event in events if event.get("cat") == "kernel"}
    if not kernels:
        raise AssertionError("profiler contains no GPU kernel events")
    native = sorted(
        name
        for name in kernels
        if "ixformer" in name.lower() or "cuinfer" in name.lower()
    )
    if native:
        raise AssertionError(f"profiler contains native vendor kernels: {native}")
    return kernels
