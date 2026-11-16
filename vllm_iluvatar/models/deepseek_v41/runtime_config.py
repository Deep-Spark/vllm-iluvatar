# SPDX-License-Identifier: Apache-2.0
"""Worker-local runtime policies selected before V4.1 warmup and graph capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def configure_w8a8_input_rows(vllm_config: VllmConfig) -> bool:
    """Disable copied row padding; the quantization producer owns GEMM tiles.

    The target is constructed before its draft and before graph capture, so
    both use the same worker-local ixformer setting. Clear the legacy option
    so an inherited flag cannot add a second copy after producer allocation.
    Storage safety is independent of the sequence limit and PD role.
    """
    from ixformer.core import config as ixformer_config

    enabled = False
    ixformer_config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS = enabled
    return enabled
