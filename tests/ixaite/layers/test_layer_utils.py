# SPDX-License-Identifier: Apache-2.0
"""Verify the private unquantized GEMM binding."""

from types import SimpleNamespace

import torch


# The own utility directly calls its selected export, with no vendor shape fallback.
def test_own_unquantized_gemm_uses_bound_linear(monkeypatch):
    from unittest.mock import Mock

    from vllm_iluvatar.backends.ixaite.layers import utils as own

    kernel = Mock(return_value="own output")
    monkeypatch.setattr(own.ilu_ops, "linear", kernel)
    x = torch.empty(1, 2, dtype=torch.bfloat16)
    weight = torch.empty(3, 2, dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=weight)
    assert own.default_unquantized_gemm(layer, x, weight) == "own output"
    kernel.assert_called_once_with(x, weight, None)
