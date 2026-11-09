# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.5 attention-output sigmoid-gate provider routing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from conftest import requires_cuda

from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5


def test_startup_binds_ixaite_when_supported(monkeypatch) -> None:
    ixaite_op = Mock(return_value="ixaite")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    loader = Mock(return_value=(ixaite_op, "supported on ivcore11"))
    monkeypatch.setattr(ilu_qwen3_5, "load_ixaite_python_op", loader)

    ilu_qwen3_5.configure_attn_output_sigmoid_gate_provider()

    assert ilu_qwen3_5._ATTN_OUTPUT_SIGMOID_GATE is ixaite_op
    loader.assert_called_once_with(
        "attn_output_sigmoid_gate",
        "attn_output_sigmoid_gate",
    )


def test_forced_ixaite_raises_when_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        ilu_qwen3_5,
        "load_ixaite_python_op",
        Mock(return_value=(None, "not validated on this arch")),
    )

    with pytest.raises(RuntimeError, match="not validated on this arch"):
        ilu_qwen3_5.configure_attn_output_sigmoid_gate_provider()


@requires_cuda
def test_ixaite_routed_op_matches_torch_inplace(monkeypatch) -> None:
    pytest.importorskip("ixaite")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")

    import vllm_iluvatar.backends.ixaite.runtime as kernel_provider

    kernel_provider._load_ixaite_runtime.cache_clear()
    kernel_provider.load_ixaite_python_op.cache_clear()
    try:
        ilu_qwen3_5.configure_attn_output_sigmoid_gate_provider()

        torch.manual_seed(0)
        attn_output = torch.randn(
            37,
            4096,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = torch.randn_like(attn_output)
        expected = (attn_output.float() * torch.sigmoid(gate.float())).to(
            attn_output.dtype
        )

        returned = ilu_qwen3_5._ATTN_OUTPUT_SIGMOID_GATE(
            attn_output,
            gate,
            attn_output,
        )
        torch.cuda.synchronize()

        assert returned is attn_output
        torch.testing.assert_close(attn_output, expected, rtol=2e-2, atol=2e-2)
    finally:
        kernel_provider.load_ixaite_python_op.cache_clear()
        kernel_provider._load_ixaite_runtime.cache_clear()
