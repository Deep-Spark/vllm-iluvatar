# SPDX-License-Identifier: Apache-2.0
"""Shared Iluvatar architecture detection.

Single source of truth for ixformer ``cuda_arch`` queries.  All arch-specific
dispatch across the plugin should go through this module instead of importing
``ixformer`` directly.

The ``ixformer`` import is kept inside the function body so that importing
this module at auto-discovery time never pulls in ixformer eagerly.
"""
from __future__ import annotations

IVCORE30 = "ivcore30"
IVCORE11 = "ivcore11"

_cuda_arch: str | None = None
_resolved = False


def cuda_arch() -> str | None:
    """Cached ixformer arch string; None if ixformer unavailable."""
    global _cuda_arch, _resolved
    if not _resolved:
        _resolved = True
        try:
            from ixformer import cuda
            _cuda_arch = cuda.cuda_arch
        except ImportError:
            _cuda_arch = None
    return _cuda_arch


def is_ivcore30() -> bool:
    return cuda_arch() == IVCORE30


def is_ivcore11() -> bool:
    return cuda_arch() == IVCORE11
