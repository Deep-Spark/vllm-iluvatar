# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for compilation backend compatibility patches."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_iluvatar.ilu_patch.general_patches import patch_compilation_backends


def test_vllm_backend_call_accepts_torch_compile_options(monkeypatch):
    calls = []
    wrap_calls = []

    class VllmBackend:
        def __call__(self, graph, example_inputs):
            calls.append((self, graph, example_inputs))
            return "compiled"

    def wrap_with_cudagraph_if_needed(*args):
        wrap_calls.append(args)
        return "wrapped"

    community = ModuleType("vllm.compilation.backends")
    community.VllmBackend = VllmBackend
    community.wrap_with_cudagraph_if_needed = wrap_with_cudagraph_if_needed

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.compilation", ModuleType("vllm.compilation"))
    monkeypatch.setitem(sys.modules, community.__name__, community)

    patch_compilation_backends._apply()

    backend = VllmBackend()
    graph = SimpleNamespace(name="graph")
    example_inputs = [object()]

    assert backend(graph, example_inputs, options={"mode": "ignored"}) == "compiled"
    assert calls == [(backend, graph, example_inputs)]


def test_piecewise_cudagraph_uses_graph_forward_for_ixformer_torch_ops(
    monkeypatch,
):
    wrap_calls = []

    class VllmBackend:
        def __call__(self, graph, example_inputs):
            return "compiled"

    def wrap_with_cudagraph_if_needed(*args):
        wrap_calls.append(args)
        return "wrapped"

    community = ModuleType("vllm.compilation.backends")
    community.VllmBackend = VllmBackend
    community.wrap_with_cudagraph_if_needed = wrap_with_cudagraph_if_needed

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.compilation", ModuleType("vllm.compilation"))
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setenv("IXFORMER_USE_TORCH_OPS", "1")

    patch_compilation_backends._apply()

    graph_forward = object()
    piecewise_backend = SimpleNamespace(graph=SimpleNamespace(forward=graph_forward))
    result = community.wrap_with_cudagraph_if_needed(
        piecewise_backend,
        "vllm_config",
        "compilation_config",
        True,
        False,
    )

    assert result == "wrapped"
    assert wrap_calls == [
        (graph_forward, "vllm_config", "compilation_config", True, False)
    ]


def test_piecewise_cudagraph_falls_back_without_graph_forward(
    monkeypatch,
):
    wrap_calls = []

    class VllmBackend:
        def __call__(self, graph, example_inputs):
            return "compiled"

    def wrap_with_cudagraph_if_needed(*args):
        wrap_calls.append(args)
        return "wrapped"

    community = ModuleType("vllm.compilation.backends")
    community.VllmBackend = VllmBackend
    community.wrap_with_cudagraph_if_needed = wrap_with_cudagraph_if_needed

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.compilation", ModuleType("vllm.compilation"))
    monkeypatch.setitem(sys.modules, community.__name__, community)
    monkeypatch.setenv("IXFORMER_USE_TORCH_OPS", "1")

    patch_compilation_backends._apply()

    piecewise_backend = SimpleNamespace(graph=None)
    result = community.wrap_with_cudagraph_if_needed(
        piecewise_backend,
        "vllm_config",
        "compilation_config",
        True,
        True,
    )

    assert result is piecewise_backend
    assert wrap_calls == []
