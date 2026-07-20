# SPDX-License-Identifier: Apache-2.0
"""Tests for the Iluvatar FLA l2norm wrapper and kernel behavior."""

from __future__ import annotations

import importlib

from conftest import requires_cuda

from vllm_iluvatar.third_party.flash_linear_attention.ops import ilu_l2norm

torch = importlib.import_module("torch")


class _FakeKernel:
    def __init__(self, name: str, calls: list[tuple[str, object, dict]]) -> None:
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, grid, kwargs))

        return launch


def test_l2norm_uses_grouped_kernel_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(ilu_l2norm, "USE_DEFAULT_FLA_NORM", False)
    monkeypatch.setattr(ilu_l2norm, "l2norm_fwd_kernel1", _FakeKernel("k1", calls))
    monkeypatch.setattr(ilu_l2norm, "l2norm_fwd_kernel2", _FakeKernel("k2", calls))
    x = torch.ones(2, 4, 3, device="cpu")
    x1 = torch.ones(2, 4, 3, device="cpu")

    out, out1 = ilu_l2norm.l2norm_fwd_inplace(x, x1)

    assert out.shape == x.shape
    assert out1.shape == x1.shape
    assert out.data_ptr() == x.data_ptr()
    assert out1.data_ptr() == x1.data_ptr()
    assert calls == [("k2", (2, 1), {})]


def test_l2norm_uses_default_kernel_when_env_requests_it(monkeypatch):
    calls = []
    monkeypatch.setattr(ilu_l2norm, "USE_DEFAULT_FLA_NORM", True)
    monkeypatch.setattr(ilu_l2norm, "l2norm_fwd_kernel1", _FakeKernel("k1", calls))
    monkeypatch.setattr(ilu_l2norm, "l2norm_fwd_kernel2", _FakeKernel("k2", calls))
    x = torch.ones(2, 4, 3, device="cpu")
    x1 = torch.ones(2, 4, 3, device="cpu")

    ilu_l2norm.l2norm_fwd_inplace(x, x1)

    assert len(calls) == 1
    name, grid, kwargs = calls[0]
    assert name == "k1"
    assert grid == (2, 4)
    assert kwargs == {}


def _l2norm_reference(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    square_sum = (x_float * x_float).sum(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(square_sum + eps)).to(x.dtype)


@requires_cuda
def test_l2norm_matches_torch_reference(cuda_device, monkeypatch):
    monkeypatch.setattr(ilu_l2norm, "USE_DEFAULT_FLA_NORM", False)
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(3, 4, 16, device=cuda_device, dtype=torch.float32)
    x1 = torch.randn(3, 4, 16, device=cuda_device, dtype=torch.float32)
    expected_x = _l2norm_reference(x, eps)
    expected_x1 = _l2norm_reference(x1, eps)

    out, out1 = ilu_l2norm.l2norm_fwd_inplace(x, x1, eps=eps)

    torch.testing.assert_close(out, expected_x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out1, expected_x1, atol=1e-5, rtol=1e-5)
