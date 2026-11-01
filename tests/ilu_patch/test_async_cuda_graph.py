# SPDX-License-Identifier: Apache-2.0
"""Validate the opt-in replay policy without loading a graph or GPU model."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_iluvatar.ilu_patch.general_patches import patch_async_cuda_graph


@pytest.mark.parametrize(
    "setting,architecture,registration,cumem,expected",
    [
        (None, "ivcore11", None, None, "off"),
        ("0", "ivcore30", "1", "1", "off"),
        ("1", "ivcore11", "0", None, "on"),
        ("1", "ivcore30", "0", None, "error"),
        ("1", "sm90", "0", None, "error"),
        ("1", "ivcore11", None, None, "error"),
        ("1", "ivcore11", "1", None, "error"),
        ("1", "ivcore11", "0", "0", "error"),
        ("1", "ivcore11", "0", "1", "error"),
        ("yes", "ivcore11", "0", None, "error"),
    ],
)
def test_replay_policy(
    monkeypatch, setting, architecture, registration, cumem, expected
):
    for key, value in (
        ("VLLM_ILU_CUDAGRAPH_ASYNC_REPLAY", setting),
        ("NCCL_GRAPH_REGISTER", registration),
        ("NCCL_CUMEM_ENABLE", cumem),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    old, new = object(), object()
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        CUDAGraph=old, graphs=SimpleNamespace(CUDAGraph=old)
    )
    ixformer = ModuleType("ixformer")
    ixformer.__path__ = []
    ixformer.cuda = ModuleType("ixformer.cuda")
    ixformer.cuda.cuda_arch = architecture
    ixformer.cuda.AsyncCUDAGraph = new
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ixformer", ixformer)
    monkeypatch.setitem(sys.modules, "ixformer.cuda", ixformer.cuda)

    if expected == "error":
        with pytest.raises((RuntimeError, ValueError)):
            patch_async_cuda_graph._apply()
        assert torch.cuda.CUDAGraph is old
    else:
        patch_async_cuda_graph._apply()
        assert torch.cuda.CUDAGraph is (new if expected == "on" else old)
        patch_async_cuda_graph._apply()
    assert torch.cuda.graphs.CUDAGraph is torch.cuda.CUDAGraph
