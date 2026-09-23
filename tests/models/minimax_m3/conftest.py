# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 test layout.

Quant-specific cases live in ``w4a8/`` and ``mxfp8/``.
Shared (not quant-bound) cases stay in this directory.

ivcore11 collects w4a8 (compressed-tensors) only; ivcore30 collects both.
"""

from __future__ import annotations

import pytest

from vllm_iluvatar.ilu_util.arch import is_ivcore11

# Do not collect MiniMax mxfp8 cases on ivcore11. w4a8 (compressed-tensors) runs
# on both ivcore11 and ivcore30.
collect_ignore = ["mxfp8"] if is_ivcore11() else []

# NOTE: pytest imports non-package conftest.py files under the bare module
# name ``conftest``, which shadows the root tests/conftest.py in sys.modules.
# Tests collected after this dir that do ``from conftest import ...`` then
# resolve to *this* conftest, so the shared marks must be redefined here too.
# Keep in sync with tests/conftest.py and tests/models/minimax_m3/mxfp8/conftest.py.


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
    if not _cuda_available():
        return None
    import torch
    import vllm_iluvatar._C  # noqa: F401

    return str(torch.ops._C.compiled_cuda_arch())


COMPILED_CUDA_ARCH = _compiled_cuda_arch()
