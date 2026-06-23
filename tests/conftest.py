# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures / skip markers for the vllm_iluvatar test suite.

Two tiers of tests (see plan §6):
  * schema/signature smoke — CPU-only, just needs the compiled ``vllm_iluvatar._C``
    module importable; gated by ``requires_native_ops``.
  * numerical correctness — needs an Iluvatar GPU + ixformer kernels; gated by
    ``requires_cuda`` (and ``requires_native_ops``).

All gating is done at collection time so the suite degrades gracefully on hosts
without the native build / CUDA rather than erroring.
"""

from __future__ import annotations

import importlib

import pytest

NATIVE_OPS = ("rms_norm", "weak_ref_tensor", "get_cuda_view_from_cpu_tensor")


def _native_ops_loaded() -> bool:
    try:
        importlib.import_module("vllm_iluvatar._C")
    except Exception:
        return False
    try:
        import torch

        return all(hasattr(torch.ops._C, name) for name in NATIVE_OPS)
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_native_ops = pytest.mark.skipif(
    not _native_ops_loaded(),
    reason="vllm_iluvatar._C native ops are not built/importable",
)

requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="an Iluvatar/CUDA device is required for this test",
)


@pytest.fixture()
def cuda_device() -> str:
    if not _cuda_available():
        pytest.skip("CUDA device required")
    return "cuda:0"
