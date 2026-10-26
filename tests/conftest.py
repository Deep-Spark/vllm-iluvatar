# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures / skip markers for the vllm_iluvatar test suite.

Policy: this is an Iluvatar build, so its dependencies (the compiled
``vllm_iluvatar._C`` native ops, ``ixformer``, ``vllm``) are MANDATORY. A missing
dependency is an environment error, NOT a silent skip — tests import them
directly so collection fails loudly when the build/env is broken.

Device tests may skip when there is no Iluvatar GPU or the architecture is
unsupported. CoreX30-only kernels skip on CoreX11; shared CPU tests still run.
Missing dependencies on a supported architecture remain environment errors.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-legacy-drift",
        action="store_true",
        default=False,
        help="Include optional legacy numerical compatibility diagnostics.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-legacy-drift"):
        return
    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker("legacy_drift") is not None:
            deselected.append(item)
        else:
            selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


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

    This identifies the installed artifact, not the device capability. Device
    support and build mismatches are checked separately below. A missing native
    dependency with a device is an environment error and propagates.
    """
    if not _cuda_available():
        return None

    import torch
    import vllm_iluvatar._C  # noqa: F401

    return str(torch.ops._C.compiled_cuda_arch())


COMPILED_CUDA_ARCH = _compiled_cuda_arch()


def _corex30_available() -> bool:
    if not _cuda_available():
        return False
    import torch
    from ixformer.cuda import cuda_arch

    name = torch.cuda.get_device_name()
    device_arch = (
        "ivcore30" if "V300" in name else "ivcore11" if "V150" in name else None
    )
    if device_arch is not None and (
        device_arch != COMPILED_CUDA_ARCH or cuda_arch != device_arch
    ):
        raise RuntimeError(
            f"{name} requires {device_arch}; plugin build={COMPILED_CUDA_ARCH}, "
            f"ixformer build={cuda_arch}"
        )
    return device_arch == "ivcore30"


requires_corex30 = pytest.mark.skipif(
    not _corex30_available(),
    reason="requires a CoreX30 device",
)


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
