# SPDX-License-Identifier: Apache-2.0
"""Tests for fused QK RMSNorm(RoPE)gate provider routing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401
from conftest import requires_cuda

from vllm_iluvatar.backends.ixaite.models import qwen3_5 as ilu_qwen3_5


def test_startup_binds_ixaite_when_supported(monkeypatch) -> None:
    rope = Mock(return_value="ixaite_rope")
    gate = Mock(return_value="ixaite_gate")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        ilu_qwen3_5,
        "load_ixaite_torch_op",
        Mock(
            side_effect=lambda op_name: {
                "fused_qk_rmsnorm_rope_gate": (rope, "supported on ivcore11"),
                "fused_qk_rmsnorm_gate": (gate, "supported on ivcore11"),
            }[op_name]
        ),
    )

    ilu_qwen3_5.configure_fused_qk_provider()

    assert ilu_qwen3_5._FUSED_QK_RMSNORM_ROPE_GATE is rope
    assert ilu_qwen3_5._FUSED_QK_RMSNORM_GATE is gate


def test_forced_ixaite_raises_when_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        ilu_qwen3_5,
        "load_ixaite_torch_op",
        Mock(return_value=(None, "not validated on this arch")),
    )

    with pytest.raises(RuntimeError, match="not validated on this arch"):
        ilu_qwen3_5.configure_fused_qk_provider()


@requires_cuda
def test_ixaite_routed_ops_match_plugin(monkeypatch) -> None:
    pytest.importorskip("ixaite")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")

    import vllm_iluvatar.backends.ixaite.runtime as kernel_provider

    kernel_provider._load_ixaite_runtime.cache_clear()
    kernel_provider.load_ixaite_torch_op.cache_clear()
    try:
        ilu_qwen3_5.configure_fused_qk_provider()

        torch.manual_seed(0)
        dtype = torch.bfloat16
        tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim = 4, 8, 2, 128, 64
        q_gate = torch.randn(
            tokens, num_q_heads * 2 * head_dim, device="cuda", dtype=dtype
        )
        k = torch.randn(tokens, num_kv_heads * head_dim, device="cuda", dtype=dtype)
        q_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
        k_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
        cache = torch.randn(32, rotary_dim, device="cuda", dtype=dtype)
        positions = torch.randint(0, 32, (tokens,), device="cuda")

        plugin_rope = torch.ops._C.fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            cache,
            positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
        ixaite_rope = ilu_qwen3_5._FUSED_QK_RMSNORM_ROPE_GATE(
            q_gate,
            k,
            q_weight,
            k_weight,
            cache,
            positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
        )
        for actual, expected in zip(ixaite_rope[:2], plugin_rope[:2]):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=5e-3)
        torch.testing.assert_close(ixaite_rope[2], plugin_rope[2], atol=0, rtol=0)

        plugin_gate = torch.ops._C.fused_qk_rmsnorm_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
        ixaite_gate = ilu_qwen3_5._FUSED_QK_RMSNORM_GATE(
            q_gate,
            k,
            q_weight,
            k_weight,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
        for actual, expected in zip(ixaite_gate[:2], plugin_gate[:2]):
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=6e-3)
        torch.testing.assert_close(ixaite_gate[2], plugin_gate[2], atol=0, rtol=0)
    finally:
        kernel_provider.load_ixaite_torch_op.cache_clear()
        kernel_provider._load_ixaite_runtime.cache_clear()
