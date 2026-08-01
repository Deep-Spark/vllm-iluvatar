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


# ``IluPlatform.device_name`` (vllm_iluvatar/platforms/ilu_platform.py:115).
ILU_DEVICE_NAME = "iluvatar"


def is_iluvatar_platform() -> bool:
    """True when the resolved vLLM platform is Iluvatar.

    The gate every general patch and OOT override uses to decide "is this run
    ours?" before changing community behaviour.

    Deliberately a ``device_name`` string compare:

    * ``isinstance(current_platform, IluPlatform)`` would need to import
      ``platforms/ilu_platform.py``, whose import stubs ``vllm._C``, builds the
      platform class and touches ``torch.backends``;
    * ``current_platform.is_cuda()`` is unusable -- ``IluPlatform`` subclasses
      ``CudaPlatform`` and keeps ``_enum == CUDA``, so it is true on real NVIDIA
      runs too and a gate built on it would hijack them.

    ``current_platform`` is imported lazily because ``vllm.platforms.__getattr__``
    resolves the platform on attribute access, which must not happen at plugin
    import time.
    """
    from vllm.platforms import current_platform

    return getattr(current_platform, "device_name", None) == ILU_DEVICE_NAME
