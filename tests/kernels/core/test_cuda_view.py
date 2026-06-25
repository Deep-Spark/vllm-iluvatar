# SPDX-License-Identifier: Apache-2.0
"""Behavioural tests for torch.ops._C.weak_ref_tensor /
get_cuda_view_from_cpu_tensor.

These are the two utility ops the reverted community vllm/utils/torch_utils.py
calls. They are self-contained (no ixformer kernel) but still need a CUDA device
because both deal with CUDA storage / mapped host memory.
"""

from __future__ import annotations

import pytest

# Native extension is a MANDATORY dep: a missing build is an environment error
# (collection-time ImportError), never a silent skip.
import torch
import vllm_iluvatar._C  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("Iluvatar/CUDA device required", allow_module_level=True)

DEVICE = "cuda:0"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_weak_ref_tensor_aliases_storage(dtype: torch.dtype) -> None:
    x = torch.randn(16, 32, dtype=dtype, device=DEVICE)
    ref = torch.ops._C.weak_ref_tensor(x)

    assert ref.data_ptr() == x.data_ptr()
    assert ref.shape == x.shape
    assert ref.stride() == x.stride()
    assert ref.dtype == x.dtype
    assert ref.device == x.device

    # It is a true alias: in-place edits on the source are visible via the view.
    x.add_(1.0)
    torch.testing.assert_close(ref, x)


def test_weak_ref_tensor_rejects_cpu() -> None:
    x = torch.randn(4, 4)
    with pytest.raises(Exception):  # noqa: B017 - torch raises a RuntimeError
        torch.ops._C.weak_ref_tensor(x)


def test_get_cuda_view_empty_tensor() -> None:
    cpu = torch.empty(0, dtype=torch.float32)
    view = torch.ops._C.get_cuda_view_from_cpu_tensor(cpu)
    assert view.is_cuda
    assert view.numel() == 0


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_get_cuda_view_matches_host_values(pinned: bool, dtype: torch.dtype) -> None:
    cpu = torch.randn(64, 128, dtype=dtype)
    if pinned:
        cpu = cpu.pin_memory()

    view = torch.ops._C.get_cuda_view_from_cpu_tensor(cpu)

    assert view.is_cuda
    assert view.shape == cpu.shape
    assert view.dtype == cpu.dtype
    torch.testing.assert_close(view.cpu(), cpu)


def test_get_cuda_view_rejects_cuda_input() -> None:
    x = torch.randn(4, 4, device=DEVICE)
    with pytest.raises(Exception):  # noqa: B017
        torch.ops._C.get_cuda_view_from_cpu_tensor(x)
