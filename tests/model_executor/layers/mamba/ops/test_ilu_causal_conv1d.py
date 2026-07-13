# SPDX-License-Identifier: Apache-2.0
"""Tests for Iluvatar causal conv1d wrapper metadata and kernel behavior."""

from __future__ import annotations

import torch
from conftest import requires_cuda

from vllm_iluvatar.model_executor.layers.mamba.ops import ilu_causal_conv1d


class _FakeKernel:
    def __init__(self) -> None:
        self.grid = None
        self.args = None
        self.meta = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args, **meta):
            self.args = args
            self.meta = meta

        return launch


def test_causal_conv1d_update_normalizes_activation_and_reuses_input(monkeypatch):
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(ilu_causal_conv1d, "_causal_conv1d_update_kernel", fake_kernel)
    x = torch.ones(3, 4, device="cpu")
    conv_state = torch.zeros(3, 4, 2, device="cpu")
    weight = torch.ones(4, 3, device="cpu")

    out = ilu_causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        activation=True,
    )

    assert out is x
    assert fake_kernel.meta["SILU_ACTIVATION"] is True
    assert fake_kernel.meta["IS_VARLEN"] is False
    assert fake_kernel.meta["IS_SPEC_DECODING"] is False
    assert fake_kernel.meta["USE_INPUT_INDEX"] is False
    assert fake_kernel.meta["KERNEL_WIDTH"] == 3


def test_causal_conv1d_update_allocates_indexed_output(monkeypatch):
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(ilu_causal_conv1d, "_causal_conv1d_update_kernel", fake_kernel)
    x = torch.ones(3, 4, device="cpu")
    conv_state = torch.zeros(3, 4, 2, device="cpu")
    weight = torch.ones(4, 3, device="cpu")
    input_index = torch.tensor([2, 0], dtype=torch.int32, device="cpu")

    out = ilu_causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        input_index=input_index,
    )

    assert out is not x
    assert out.shape == (2, 4)
    assert fake_kernel.meta["USE_INPUT_INDEX"] is True


def test_causal_conv1d_fn_sets_indexed_launch_metadata(monkeypatch):
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(ilu_causal_conv1d, "_causal_conv1d_fwd_kernel", fake_kernel)
    x = torch.ones(4, 5, device="cpu")
    weight = torch.ones(4, 3, device="cpu")
    conv_states = torch.zeros(2, 2, 4, device="cpu").transpose(1, 2)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device="cpu")
    input_index = torch.tensor([4, 3, 2], dtype=torch.int32, device="cpu")

    out = ilu_causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        bias=None,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        input_index=input_index,
        activation="swish",
    )

    assert out.shape == (4, 3)
    assert fake_kernel.meta["USE_INPUT_INDEX"] is True
    assert fake_kernel.meta["HAS_BIAS"] is False
    assert fake_kernel.meta["SILU_ACTIVATION"] is True
    assert fake_kernel.meta["KERNEL_WIDTH"] == 3


def _causal_conv1d_update_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = weight.shape[1]
    state_len = width - 1
    conv_window = torch.cat([conv_state[:, :, :state_len], x.unsqueeze(-1)], dim=-1)
    out = (conv_window * weight.unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        out = out + bias
    if activation in ("silu", "swish"):
        out = out * torch.sigmoid(out)

    next_state = conv_state.clone()
    next_state[:, :, :state_len] = conv_window[:, :, 1:]
    return out, next_state


@requires_cuda
def test_causal_conv1d_update_matches_torch_reference(cuda_device):
    torch.manual_seed(0)
    batch = 3
    dim = 8
    width = 4
    dtype = torch.float32
    x = torch.randn(batch, dim, device=cuda_device, dtype=dtype)
    conv_state = torch.randn(batch, width - 1, dim, device=cuda_device, dtype=dtype)
    conv_state = conv_state.transpose(1, 2)
    weight = torch.randn(dim, width, device=cuda_device, dtype=dtype)
    bias = torch.randn(dim, device=cuda_device, dtype=dtype)
    conv_state_indices = torch.arange(batch, device=cuda_device, dtype=torch.int32)
    expected_out, expected_state = _causal_conv1d_update_reference(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
    )

    out = ilu_causal_conv1d.causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        validate_data=True,
    )

    torch.testing.assert_close(out, expected_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(conv_state, expected_state, atol=1e-4, rtol=1e-4)
