# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for cached ixaite operator bindings."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import vllm_iluvatar.backends.ixaite.runtime as kernel_provider
from vllm_iluvatar.backends.ixaite.runtime import (
    is_ixaite_op_enabled,
    is_ixaite_python_op_enabled,
    load_ixaite_torch_op,
)


# Check all stateless operators share one lazy ixaite initialization per worker.
def test_ixaite_torch_op_loader_is_cached(monkeypatch) -> None:
    implementation = object()
    fake_ixaite = ModuleType("ixaite")
    fake_ixaite.__path__ = []
    fake_ixaite.initialize = Mock()
    fake_ops = ModuleType("ixaite.ops")
    fake_ops.__path__ = []
    fake_op = ModuleType("ixaite.ops.silu_and_mul")
    fake_op.is_supported = Mock(return_value=(True, None))

    monkeypatch.setitem(sys.modules, "ixaite", fake_ixaite)
    monkeypatch.setitem(sys.modules, "ixaite.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "ixaite.ops.silu_and_mul", fake_op)
    monkeypatch.setattr(kernel_provider, "cuda_arch", lambda: "ivcore11")
    monkeypatch.setattr(
        kernel_provider,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                ixaite=SimpleNamespace(
                    silu_and_mul=SimpleNamespace(default=implementation)
                )
            )
        ),
    )
    kernel_provider._load_ixaite_runtime.cache_clear()
    load_ixaite_torch_op.cache_clear()

    first = load_ixaite_torch_op("silu_and_mul")
    second = load_ixaite_torch_op("silu_and_mul")

    assert first == second == (implementation, "supported on ivcore11")
    fake_ixaite.initialize.assert_called_once_with("ivcore11")
    fake_op.is_supported.assert_called_once_with("ivcore11")

    load_ixaite_torch_op.cache_clear()
    kernel_provider._load_ixaite_runtime.cache_clear()


# Python and torch availability probes avoid optional imports in baseline mode.
def test_baseline_availability_checks_are_lazy(monkeypatch):
    monkeypatch.delenv("VLLM_ILU_KERNEL_PROVIDER", raising=False)
    torch_loader, python_loader = Mock(), Mock()
    monkeypatch.setattr(kernel_provider, "load_ixaite_torch_op", torch_loader)
    monkeypatch.setattr(kernel_provider, "load_ixaite_python_op", python_loader)
    assert not is_ixaite_op_enabled("linear")
    assert not is_ixaite_python_op_enabled("video_evs", "compute_retention_mask")
    torch_loader.assert_not_called()
    python_loader.assert_not_called()
