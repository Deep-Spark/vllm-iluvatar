# SPDX-License-Identifier: Apache-2.0
"""
IluPlatform — Iluvatar GPU platform for vLLM.

Inherits from the concrete CudaPlatform (NvmlCudaPlatform or
NonNvmlCudaPlatform, resolved at import time in cuda.py) so that all
NVML-based device queries and helper methods work out of the box.

The platform class path is returned by ``vllm_iluvatar.register_platform``
(the ``vllm.platform_plugins`` entry point).

All Iluvatar-specific platform overrides (formerly scattered in cuda.py /
interface.py) are centralised in the IluPlatform class below, so the
community platform files stay byte-identical to upstream.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)

_ATTN_BACKENDS_REGISTERED = False


def register_attention_backends() -> None:
    """Override community AttentionBackendEnum paths with Iluvatar backends.

    Backend classes report community enum names via ``get_name()`` so
    ``AttentionBackendEnum[backend.get_name()]`` in attention.py keeps working
    without custom ILU_* enum members in the community registry.
    """
    global _ATTN_BACKENDS_REGISTERED
    if _ATTN_BACKENDS_REGISTERED:
        return

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "vllm.v1.attention.backends.ilu_flash_attn.IluFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        "vllm.v1.attention.backends.mla.ilu_mla.IluMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        "vllm.v1.attention.backends.mla.ilu_flashmla_sparse.IluFlashMLASparseBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        "vllm.v1.attention.backends.mla.ilu_flashmla_sparse."
        "IluDeepseekV4FlashMLASparseBackend",
    )
    _ATTN_BACKENDS_REGISTERED = True


def _make_iluvatar_platform():
    """Build IluPlatform at first import of this module."""

    # Pre-stub NVIDIA compiled extensions that may not exist on Iluvatar.
    # Community cuda.py imports these at module level; without them the import
    # would fail with ImportError before we can define our class.
    for _ext in ("vllm._C", "vllm._C_stable_libtorch"):
        if _ext not in sys.modules:
            try:
                __import__(_ext)
            except ImportError:
                sys.modules[_ext] = types.ModuleType(_ext.split(".")[-1])
                logger.debug("ILU: stubbed missing extension %s", _ext)

    from vllm.platforms.cuda import CudaPlatform
    from vllm.platforms.interface import DeviceCapability

    # Undo the cudnn sdp disable that community cuda.py applies at module level.
    try:
        import torch

        torch.backends.cuda.enable_cudnn_sdp(True)
        logger.debug("ILU: re-enabled cudnn sdp (undoing cuda.py disable)")
    except Exception:
        pass

    class IluPlatform(CudaPlatform):
        """Iluvatar GPU platform.

        Inherits device enumeration and NVML helpers from CudaPlatform.
        Every override below maps 1:1 to a hunk the fork made in
        vllm/platforms/cuda.py or interface.py.
        """

        device_name: str = "iluvatar"

        @classmethod
        def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
            # fork cuda.py: Nvml/NonNvml get_device_capability hardcode (9, 0).
            return DeviceCapability(major=9, minor=0)

        @classmethod
        def has_device_capability(
            cls,
            capability: tuple[int, int] | int,
            device_id: int = 0,
        ) -> bool:
            # fork interface.py: Platform.has_device_capability -> return True.
            return True

        @classmethod
        def import_kernels(cls) -> None:
            # fork interface.py: Platform.import_kernels -> pass.
            pass

        @classmethod
        def opaque_attention_op(cls) -> bool:
            # fork cuda.py: True -> False.
            return False

        @classmethod
        def use_custom_op_collectives(cls) -> bool:
            # fork cuda.py: True -> False.
            return False

        @classmethod
        def get_device_communicator_cls(cls) -> str:
            return (
                "vllm_iluvatar.distributed.device_communicators"
                ".ilu_communicator.IluCommunicator"
            )

        @classmethod
        def get_attn_backend_cls(
            cls,
            selected_backend: AttentionBackendEnum | None,
            attn_selector_config: AttentionSelectorConfig,
            num_heads: int | None = None,
        ) -> str:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            register_attention_backends()

            if selected_backend is not None:
                logger.info("Using %s backend (user-selected).", selected_backend)
                return selected_backend.get_path()

            use_mla = attn_selector_config.use_mla
            use_sparse = getattr(attn_selector_config, "use_sparse", False)

            if use_mla:
                backend = (
                    AttentionBackendEnum.FLASHMLA_SPARSE
                    if use_sparse
                    else AttentionBackendEnum.FLASHMLA
                )
            else:
                backend = AttentionBackendEnum.FLASH_ATTN

            logger.info_once("Using %s attention backend.", backend.name)
            return backend.get_path()

    IluPlatform.__name__ = "IluPlatform"
    IluPlatform.__qualname__ = "IluPlatform"
    return IluPlatform


IluPlatform = _make_iluvatar_platform()
