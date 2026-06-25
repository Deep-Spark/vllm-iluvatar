# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures / skip markers for the vllm_iluvatar test suite.

Policy: this is an Iluvatar build, so its dependencies (the compiled
``vllm_iluvatar._C`` native ops, ``ixformer``, ``vllm``) are MANDATORY. A missing
dependency is an environment error, NOT a silent skip — tests import them
directly so collection fails loudly when the build/env is broken.

The only thing legitimately gated via skip is the physical CUDA device: a
numerical/kernel test genuinely cannot run on a host with no Iluvatar GPU.
"""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="an Iluvatar/CUDA device is required for this test",
)


@pytest.fixture()
def cuda_device() -> str:
    if not _cuda_available():
        pytest.skip("CUDA device required")
    return "cuda:0"
