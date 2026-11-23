# SPDX-License-Identifier: Apache-2.0
"""What the DSpark context-KV fast path requires of the ixformer build.

Not quant-bound, and deliberately outside ``mxfp8/``: the case these cover is
ivcore11, where that directory is not collected.
"""

from types import SimpleNamespace

import torch

from vllm_iluvatar.models.minimax_m3 import dspark
from vllm_iluvatar.models.minimax_m3.dspark import IluMiniMaxDSparkModel


def test_the_ixmm_probe_follows_the_registry(monkeypatch):
    import ixformer.inference.backend_registry as registry

    for backends, expected in ((["torch"], False), (["torch", "ixmm"], True)):
        monkeypatch.setattr(
            registry.registry, "available_backends", lambda op, b=backends: b
        )
        dspark._ixmm_fp8_per_channel_available.cache_clear()
        assert dspark._ixmm_fp8_per_channel_available() is expected
    dspark._ixmm_fp8_per_channel_available.cache_clear()


def test_context_kv_fast_path_needs_the_ixmm_fp8_gemm(monkeypatch):
    """The fast path bypasses the QKV layers' quant method for a raw ixmm GEMM.

    ivcore11 publishes only the ``torch`` backend for that op, so there the
    projection has to stay with the layers -- which run the MXFP8 source as
    INT8 and would otherwise never be consulted.
    """
    model = IluMiniMaxDSparkModel.__new__(IluMiniMaxDSparkModel)
    torch.nn.Module.__init__(model)
    model._context_kv_weights = model._context_kv_scales = None
    # Shaped so every other guard in the condition passes: the backend probe
    # must be the only thing that can send this to the fallback.
    projection = SimpleNamespace(
        weight=torch.empty(4, 32, dtype=torch.float8_e4m3fn),
        weight_scale=torch.empty(4, 1),
    )
    monkeypatch.setattr(
        type(model).__mro__[1],
        "_build_context_kv_buffers",
        lambda self, attn, bias: setattr(self, "_context_qkv_projs", (projection,)),
    )
    monkeypatch.setattr(dspark, "_ixmm_fp8_per_channel_available", lambda: False)

    model._build_context_kv_buffers([], False)

    assert model._context_kv_weights is None
