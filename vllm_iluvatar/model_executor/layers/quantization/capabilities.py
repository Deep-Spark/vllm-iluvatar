# SPDX-License-Identifier: Apache-2.0
"""Quantization capability checks shared by configs, online paths and backends."""

from __future__ import annotations

from vllm_iluvatar.ilu_util.arch import (
    IVCORE11,
    IVCORE30,
    cuda_arch,
    is_ivcore11,
    is_ivcore30,
)


def require_ivcore30(feature: str) -> None:
    """Reject an ivcore30-only feature with a consistent diagnostic."""
    arch = cuda_arch()
    if arch != IVCORE30:
        raise RuntimeError(
            f"{feature} requires ivcore30, but the detected architecture is "
            f"{arch or 'unknown'}"
        )


__all__ = [
    "IVCORE11",
    "IVCORE30",
    "cuda_arch",
    "is_ivcore11",
    "is_ivcore30",
    "require_ivcore30",
]
