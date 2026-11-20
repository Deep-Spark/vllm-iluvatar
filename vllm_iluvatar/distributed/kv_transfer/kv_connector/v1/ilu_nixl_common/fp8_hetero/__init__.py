# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenMP host-to-host scatter kernel with automatic vectorized PyTorch fallback."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_EXT_LOCK = threading.Lock()
_EXT_STATE: dict[str, Any] = {"loaded": False, "mod": None}


def get_h2h_scatter_extension() -> Any | None:
    """Lazily JIT-compile ``h2h_scatter.cpp`` or return None when unavailable."""
    if os.environ.get("VLLM_ILU_DISABLE_H2H_SCATTER_EXT", "0") == "1":
        return None
    if _EXT_STATE["loaded"]:
        return _EXT_STATE["mod"]
    with _EXT_LOCK:
        if _EXT_STATE["loaded"]:
            return _EXT_STATE["mod"]
        try:
            from torch.utils.cpp_extension import load

            source = os.path.join(os.path.dirname(__file__), "h2h_scatter.cpp")
            mod = load(
                name="ilu_fp8_h2h_scatter",
                sources=[source],
                extra_cflags=["-O3", "-fopenmp"],
                extra_ldflags=["-fopenmp"],
            )
            _EXT_STATE["mod"] = mod
        except Exception as exc:
            logger.debug(
                "Falling back to vectorized PyTorch span scatter (C++ h2h_scatter "
                "extension unavailable: %s)",
                exc,
            )
            _EXT_STATE["mod"] = None
        _EXT_STATE["loaded"] = True
        return _EXT_STATE["mod"]


def scatter_staging_rows_to_pool(
    host_staging: torch.Tensor,
    wire_rows: torch.Tensor,
    source_block_ids: Sequence[int],
    spans: Sequence[tuple[int, int]],
    *,
    num_threads: int = 4,
) -> None:
    """Scatter ``spans`` from CPU ``host_staging`` rows into ``wire_rows[source_block_ids]``."""
    n_blocks = len(source_block_ids)
    if n_blocks == 0 or not spans:
        return
    ids_cpu = torch.as_tensor(source_block_ids, dtype=torch.int64, device="cpu")
    ext = (
        get_h2h_scatter_extension()
        if host_staging.device.type == "cpu" and wire_rows.device.type == "cpu"
        else None
    )
    if ext is not None and n_blocks >= 2:
        offsets = torch.tensor([s[0] for s in spans], dtype=torch.int64, device="cpu")
        lengths = torch.tensor(
            [s[1] - s[0] for s in spans], dtype=torch.int64, device="cpu"
        )
        ext.h2h_scatter(
            wire_rows,
            host_staging,
            ids_cpu,
            offsets,
            offsets,
            lengths,
            int(max(1, num_threads)),
        )
        return

    # Vectorized PyTorch fallback: one index_copy_ per merged span instead of
    # O(blocks * spans) elementwise loops!
    for start, end in spans:
        wire_rows[:, start:end].index_copy_(
            0, ids_cpu, host_staging[:n_blocks, start:end]
        )
