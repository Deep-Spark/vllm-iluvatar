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
_ILU_MLA_PREFILL_FLASH_ATTN_BACKEND = (
    "vllm_iluvatar.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend"
)


def register_attention_backends() -> None:
    """Override community AttentionBackendEnum paths with Iluvatar backends.

    Backend classes report community enum names via ``get_name()`` so
    ``AttentionBackendEnum[backend.get_name()]`` in attention.py keeps working
    without custom ILU_* enum members in the community registry.

    GDN is env-gated because the Iluvatar implementation requires plugin-owned
    metadata. When the gate is disabled, ``GDN_ATTN`` keeps the community path.
    """
    global _ATTN_BACKENDS_REGISTERED
    if _ATTN_BACKENDS_REGISTERED:
        return

    import vllm.envs as envs
    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        MambaAttentionBackendEnum,
        register_backend,
    )

    register_mla_prefill_backend(
        MLAPrefillBackendEnum.FLASH_ATTN,
        _ILU_MLA_PREFILL_FLASH_ATTN_BACKEND,
    )

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "vllm_iluvatar.v1.attention.backends.ilu_flash_attn.IluFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        "vllm_iluvatar.v1.attention.backends.mla.ilu_mla.IluMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        "vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse.IluFlashMLASparseBackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
        "vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse."
        "IluDeepseekV4FlashMLASparseBackend",
    )
    # Reuse the official GDN_ATTN enum member; do not introduce an ILU_* enum.
    if envs.VLLM_ILU_GDN_ATTN_BACKEND:
        register_backend(
            MambaAttentionBackendEnum.GDN_ATTN,
            "vllm_iluvatar.v1.attention.backends.ilu_gdn_attn.IluGDNAttentionBackend",
            is_mamba=True,
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
        def import_ir_kernels(cls) -> None:
            # fork interface.py: keep the community providers (vllm_c backs the
            # rms_norm fallback via torch.ops._C) AND register the Iluvatar
            # provider. This is the official hook that makes "iluvatar" visible
            # before KernelConfig.ir_op_priority is validated/applied.
            import vllm.kernels  # noqa: F401

            import vllm_iluvatar.kernels  # noqa: F401  registers "iluvatar"

        @classmethod
        def get_default_ir_op_priority(cls, vllm_config):
            # fork cuda.py get_default_ir_op_priority: prepend the "iluvatar"
            # provider so ir.ops.{rms_norm,fused_add_rms_norm} dispatch to
            # ixformer first. Under inductor compilation, defer to native (let
            # inductor fuse), mirroring the community CUDA default.
            from vllm.config.compilation import CompilationMode
            from vllm.config.kernel import IrOpPriorityConfig

            cc = vllm_config.compilation_config
            using_inductor = (
                cc.backend == "inductor" and cc.mode != CompilationMode.NONE
            )
            if using_inductor:
                return IrOpPriorityConfig.with_default(["native"])
            # rms_norm keeps vllm_c as a fallback (its torch.ops._C.rms_norm is
            # provided by vllm_iluvatar._C); fused_add_rms_norm drops vllm_c
            # because community vllm_c.fused_add_rms_norm calls
            # torch.ops._C.fused_add_rms_norm, which the plugin _C does not
            # register.
            return IrOpPriorityConfig.with_default(
                ["iluvatar", "native"],
                rms_norm=["iluvatar", "vllm_c", "native"],
                fused_add_rms_norm=["iluvatar", "native"],
            )

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
        def check_and_update_config(cls, vllm_config) -> None:
            super().check_and_update_config(vllm_config)

            pc = getattr(vllm_config, "parallel_config", None)
            import sys

            print(
                f"[ILU platform] check_and_update_config: "
                f"pc={pc is not None}, "
                f"worker_cls={getattr(pc, 'worker_cls', '<none>')!r}",
                file=sys.stderr,
                flush=True,
            )
            if pc is not None and pc.worker_cls == "vllm.v1.worker.gpu_worker.Worker":
                pc.worker_cls = "vllm_iluvatar.v1.worker.gpu_worker.IluWorker"
                print(
                    "[ILU platform] set worker_cls -> IluWorker",
                    file=sys.stderr,
                    flush=True,
                )

            # Community ModelConfig only auto-selects the deepseek_v4 tokenizer
            # for the upstream "DeepseekV4ForCausalLM" arch. The Iluvatar W4A8
            # arch is OOT-registered under "IluDeepseekV4ForCausalLM", so set its
            # tokenizer_mode here (runs in VllmConfig.__post_init__, before the
            # tokenizer is built).
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is None:
                return

            try:
                archs = model_config.architectures or []
            except Exception:
                archs = []
            if (
                "IluDeepseekV4ForCausalLM" in archs
                and model_config.tokenizer_mode == "auto"
            ):
                model_config.tokenizer_mode = "deepseek_v4"
                logger.info(
                    "ILU: set tokenizer_mode='deepseek_v4' for IluDeepseekV4ForCausalLM"
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
