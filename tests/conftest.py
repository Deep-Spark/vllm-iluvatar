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


def _compiled_cuda_arch() -> str | None:
    """The ivcore arch ``vllm_iluvatar._C`` was compiled for, e.g. "ivcore30".

    Arch-gated kernels are selected at compile time, so the extension itself is
    asked rather than the runtime device: that is exactly what decides whether
    an op runs or raises. Tests gate on the returned value. None means there is
    no device, in which case they are skipped by ``requires_cuda`` anyway; with
    a device, a failing import/attribute lookup is an environment error and is
    left to propagate per the policy above.
    """
    if not _cuda_available():
        return None

    import torch
    import vllm_iluvatar._C  # noqa: F401

    return str(torch.ops._C.compiled_cuda_arch())


COMPILED_CUDA_ARCH = _compiled_cuda_arch()


def _apply_triton_jit_compat_patch() -> None:
    """Drop the unsupported Triton ``do_not_specialize_on_alignment`` kwarg.

    Community kernels (e.g. ``vllm.model_executor.layers.mamba.ops.causal_conv1d``)
    call ``@triton.jit(do_not_specialize_on_alignment=...)`` at import time, which
    the Iluvatar Triton build rejects. In production this shim runs via
    ``register_general()``; tests skip that boot path, so apply it at conftest
    import (before any test module is collected/imported).
    """
    from vllm_iluvatar.ilu_patch.general_patches import patch_triton_jit_compat

    patch_triton_jit_compat.register()


_apply_triton_jit_compat_patch()


@pytest.fixture(autouse=True)
def reset_default_torch_device():
    """Restore the default torch device after each test.

    Some CUDA kernel tests call ``torch.set_default_device("cuda:0")`` without
    restoring it; when the full suite runs, that leaks into CPU-only tests
    (e.g. MXFP4) which then allocate on a stale CUDA default and fail with
    "Expected a 'cuda' device type for generator but found 'cpu'". Mirrors the
    upstream vLLM ``tests/kernels/conftest.py`` fixture.
    """
    yield
    import torch

    torch.set_default_device(None)


@pytest.fixture()
def cuda_device() -> str:
    if not _cuda_available():
        pytest.skip("CUDA device required")
    return "cuda:0"


@pytest.fixture()
def default_vllm_config():
    """Set a default VllmConfig for tests that directly construct CustomOps.

    CustomOp.__init__ calls get_current_vllm_config(), which asserts a config
    context is active. Mirrors the upstream vllm tests/conftest.py fixture.
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config
