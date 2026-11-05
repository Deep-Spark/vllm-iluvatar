# SPDX-License-Identifier: Apache-2.0
"""Tests for causal conv1d provider routing (plugin Triton vs ixaite FlyDSL)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from conftest import requires_cuda

import vllm_iluvatar.backends.ixaite.runtime as kernel_provider
from vllm_iluvatar.backends.ixaite.layers import causal_conv1d as ilu_causal_conv1d
from vllm_iluvatar.model_executor.layers.mamba.ops import (
    ilu_causal_conv1d as baseline_conv,
)


@requires_cuda
def test_forced_ixaite_rejects_incomplete_apc_metadata(
    monkeypatch, cuda_device
) -> None:
    """Reject APC calls that do not provide the required block table."""
    pytest.importorskip("ixaite")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    kernel_provider._load_ixaite_runtime.cache_clear()
    kernel_provider.load_ixaite_python_op.cache_clear()
    try:
        ilu_causal_conv1d.configure_causal_conv1d_provider()

        dim, width = 2, 2
        x = torch.zeros((dim, 1), device=cuda_device, dtype=torch.bfloat16)
        weight = torch.zeros((dim, width), device=cuda_device, dtype=torch.bfloat16)
        conv_state = torch.zeros(
            (1, dim, width - 1), device=cuda_device, dtype=torch.bfloat16
        )
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=cuda_device)

        with pytest.raises(ValueError, match=r"\[batch, blocks\]"):
            ilu_causal_conv1d.causal_conv1d_fn(
                x,
                weight,
                None,
                conv_state,
                query_start_loc,
                block_idx_last_scheduled_token=torch.zeros(
                    1, dtype=torch.int32, device=cuda_device
                ),
            )
    finally:
        kernel_provider.load_ixaite_python_op.cache_clear()
        kernel_provider._load_ixaite_runtime.cache_clear()


# Check FP32 convolution results, cache updates, and in-place decode routing.
@requires_cuda
def test_ixaite_routed_dispatchers_match_fp32_reference(
    monkeypatch, cuda_device
) -> None:
    """End to end through the dispatchers with the real ixaite kernels."""
    pytest.importorskip("ixaite")
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    kernel_provider._load_ixaite_runtime.cache_clear()
    kernel_provider.load_ixaite_python_op.cache_clear()

    ilu_causal_conv1d.configure_causal_conv1d_provider()
    torch.manual_seed(0)
    dtype = torch.bfloat16
    dim, width, slots = 256, 4, 4
    weight = torch.randn((dim, width), device=cuda_device, dtype=dtype)
    bias = torch.randn((dim,), device=cuda_device, dtype=dtype)
    # Plugin conv_state layout: physical (slots, state_len, dim) transposed.
    conv_state = torch.randn(
        (slots, width - 1, dim), device=cuda_device, dtype=dtype
    ).transpose(1, 2)

    # Prefill through the routed dispatcher.
    x = torch.randn((dim, 9), device=cuda_device, dtype=dtype)
    query_start_loc = torch.tensor([0, 4, 9], dtype=torch.int32, device=cuda_device)
    cache_indices = torch.tensor([1, 3], dtype=torch.int32, device=cuda_device)

    triton_state = conv_state.clone()
    # The Triton kernel dereferences has_initial_state, so pass explicit zeros
    # (same semantics as None) to both implementations.
    no_init = torch.zeros(2, dtype=torch.int32, device=cuda_device)
    # Start each sequence with zero history, compute depthwise convolution and
    # SiLU in FP32 on CPU, then round only the final result to the output dtype.
    # Triton remains the cache oracle; its BF16 products use different rounding.
    expected_parts = []
    for sequence in x.float().cpu().split([4, 5], dim=-1):
        padded = F.pad(sequence.unsqueeze(0), (width - 1, 0))
        reference = F.conv1d(
            padded, weight.float().cpu().unsqueeze(1), bias.float().cpu(), groups=dim
        )
        expected_parts.append(F.silu(reference).squeeze(0))
    expected = torch.cat(expected_parts, dim=-1).to(x)
    baseline_conv.causal_conv1d_fn(
        x,
        weight,
        bias,
        triton_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=no_init,
        activation="silu",
    )
    out = ilu_causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=no_init,
        activation="silu",
    )
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(conv_state, triton_state, rtol=1e-2, atol=1e-2)

    # Decode through the routed dispatcher (in-place on x).
    xd = torch.randn((2, dim), device=cuda_device, dtype=dtype)
    indices = torch.tensor([1, 3], dtype=torch.int32, device=cuda_device)
    xd_triton = xd.clone()
    triton_state2 = conv_state.clone()
    # Append the decode token to its cached history before either kernel mutates
    # the cache, then use the same FP32 convolution and final BF16 rounding.
    window = torch.cat([conv_state[indices], xd.unsqueeze(-1)], dim=-1).float().cpu()
    reference_d = F.conv1d(
        window, weight.float().cpu().unsqueeze(1), bias.float().cpu(), groups=dim
    )
    expected_d = F.silu(reference_d).squeeze(-1).to(xd)
    baseline_conv.causal_conv1d_update(
        xd_triton,
        triton_state2,
        weight,
        bias,
        "silu",
        conv_state_indices=indices,
    )
    out_d = ilu_causal_conv1d.causal_conv1d_update(
        xd,
        conv_state,
        weight,
        bias,
        "silu",
        conv_state_indices=indices,
    )
    assert out_d.data_ptr() == xd.data_ptr()
    torch.testing.assert_close(out_d, expected_d, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(conv_state, triton_state2, rtol=1e-2, atol=1e-2)

    kernel_provider.load_ixaite_python_op.cache_clear()
    kernel_provider._load_ixaite_runtime.cache_clear()


# Preserve APC metadata when dispatching to the selected implementation.
def test_forced_ixaite_forwards_apc_metadata(monkeypatch) -> None:
    selected_fn = Mock(return_value=object())
    selected_update = Mock()
    monkeypatch.setenv("VLLM_ILU_KERNEL_PROVIDER", "ixaite")
    monkeypatch.setattr(
        ilu_causal_conv1d,
        "load_ixaite_python_op",
        Mock(
            side_effect=lambda op_name, func_name: {
                "causal_conv1d_fn": (selected_fn, "supported on ivcore11"),
                "causal_conv1d_update": (selected_update, "supported on ivcore11"),
            }[func_name]
        ),
    )
    ilu_causal_conv1d.configure_causal_conv1d_provider()

    x, weight, state, query_start_loc = (object() for _ in range(4))
    last_scheduled = object()
    result = ilu_causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        None,
        state,
        query_start_loc,
        block_idx_last_scheduled_token=last_scheduled,
    )

    assert result is selected_fn.return_value
    selected_fn.assert_called_once_with(
        x,
        weight,
        None,
        state,
        query_start_loc,
        block_idx_last_scheduled_token=last_scheduled,
    )
    selected_update.assert_not_called()
