# SPDX-License-Identifier: Apache-2.0
"""Wiring for vLLM compilation backend compatibility.

Corresponds to: vllm/compilation/backends.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

Target   : vllm.compilation.backends.VllmBackend.__call__,
           vllm.compilation.backends.wrap_with_cudagraph_if_needed
Baseline : main_6e448d0
Why      : Some CoreX Torch builds invoke custom torch.compile backends with
           extra keyword arguments such as ``options``. Upstream vLLM's backend
           callable accepts only ``(graph, example_inputs)``. Also, with
           ``IXFORMER_USE_TORCH_OPS=1``, replaying piecewise CUDA graphs through
           ``PiecewiseBackend`` can produce incorrect outputs, while capturing
           the FX graph ``forward`` directly is correct on CoreX.
How      : wrap the class method and discard backend kwargs before delegating to
           the upstream implementation. Wrap the cudagraph helper to capture
           ``piecewise_backend.graph.forward`` under ixformer torch-ops
           mode. No upstream method body is copied.
Scope    : torch.compile backend invocation on CoreX Torch.
Remove   : when paired vLLM accepts backend ``**kwargs`` or CoreX stops passing
           them to callable backends, and when piecewise cudagraph capture is
           correct with ixformer torch-ops mode.
"""

from __future__ import annotations

import os
from functools import wraps

from vllm_iluvatar.ilu_patch import general_patch as p


def register() -> None:
    """Register VllmBackend callable compatibility wrapper."""
    p.register_once(
        __name__,
        _apply,
        detail="VllmBackend kwargs and piecewise cudagraph compatibility",
    )


def _ixformer_torch_ops_enabled() -> bool:
    if os.environ.get("IXFORMER_USE_TORCH_OPS") == "1":
        return True
    try:
        from ixformer.core import config as ixf_config

        return bool(ixf_config.IXFORMER_USE_TORCH_OPS)
    except Exception:
        return False


def _apply() -> None:
    import vllm.compilation.backends as _m

    original_call = _m.VllmBackend.__call__

    @wraps(original_call)
    def __call__(self, graph, example_inputs, **kwargs):
        # [ILU override] VllmBackend.__call__ - delta: CoreX Torch may pass
        # backend kwargs (for example ``options``); upstream ignores them.
        return original_call(self, graph, example_inputs)

    _m.VllmBackend.__call__ = __call__

    original_wrap = _m.wrap_with_cudagraph_if_needed

    @wraps(original_wrap)
    def wrap_with_cudagraph_if_needed(
        piecewise_backend,
        vllm_config,
        compilation_config,
        is_first_graph,
        is_last_graph,
    ):
        if not _ixformer_torch_ops_enabled():
            return original_wrap(
                piecewise_backend,
                vllm_config,
                compilation_config,
                is_first_graph,
                is_last_graph,
            )

        # [ILU override] wrap_with_cudagraph_if_needed - delta: capture the FX
        # graph forward directly with ixformer torch ops. This avoids replaying
        # through PiecewiseBackend, which produced incorrect outputs with
        # IXFORMER_USE_TORCH_OPS=1 in CoreX validation.
        graph = getattr(piecewise_backend, "graph", None)
        graph_forward = getattr(graph, "forward", None)
        if graph_forward is None:
            return piecewise_backend
        return original_wrap(
            graph_forward,
            vllm_config,
            compilation_config,
            is_first_graph,
            is_last_graph,
        )

    _m.wrap_with_cudagraph_if_needed = wrap_with_cudagraph_if_needed
