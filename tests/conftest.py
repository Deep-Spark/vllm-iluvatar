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


@pytest.fixture(autouse=True, scope="session")
def _inject_ilu_module_aliases():
    """Inject sys.modules aliases that ``register_general()`` would set up.

    In production the plugin's ``register_general()`` installs these before any
    model import.  Tests skip that boot path, so the aliases must be injected
    here to prevent ModuleNotFoundError when community code (e.g.
    deepseek_v2.py) transitively imports migrated ilu_* modules via the old
    ``vllm.*`` paths.
    """
    from vllm_iluvatar import _inject_ilu_module_aliases as _inject
    _inject()


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
