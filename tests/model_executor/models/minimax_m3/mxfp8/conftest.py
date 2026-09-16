# SPDX-License-Identifier: Apache-2.0
"""Skip MiniMax-M3 mxfp8 cases on ivcore11 (W4A8-only)."""

from __future__ import annotations

import pytest

from vllm_iluvatar.ilu_util.arch import is_ivcore11

_SKIP_IVCORE11 = pytest.mark.skip(
    reason="MiniMax-M3 mxfp8 tests run on ivcore30, not ivcore11",
)


def pytest_collection_modifyitems(config, items):
    if not is_ivcore11():
        return
    for item in items:
        item.add_marker(_SKIP_IVCORE11)
