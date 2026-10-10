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

import importlib
import logging
import os
import sys
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)

_ATTN_BACKENDS_REGISTERED = False
_ILU_BACKEND_MODULES_SMOKED = False
_ILU_MLA_PREFILL_FLASH_ATTN_BACKEND = (
    "vllm_iluvatar.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend"
)

# ILU override backend modules to smoke-import at registration time so that
# import-time name drift (e.g. a community symbol rename consumed at module
# scope, such as ``FA4_MLA_PREFILL_KERNEL`` -> ``_FA4_MLA_PREFILL_KERNEL``) is
# surfaced at startup instead of deep inside model loading. Only the module is
# imported — no backend class is instantiated — so this stays cheap and side
# effect free relative to a normal model run, which would import these anyway.
_ILU_BACKEND_MODULE_PATHS = (
    "vllm_iluvatar.v1.attention.backends.mla.prefill.flash_attn",
    "vllm_iluvatar.v1.attention.backends.ilu_flash_attn",
    "vllm_iluvatar.v1.attention.backends.mla.ilu_mla",
    "vllm_iluvatar.v1.attention.backends.mla.ilu_flashmla_sparse",
)
# GDN is env-gated; only smoke-import when the gate is on (checked at call time).
_ILU_GDN_BACKEND_MODULE_PATH = "vllm_iluvatar.v1.attention.backends.ilu_gdn_attn"


def _smoke_import_ilu_backend_modules() -> None:
    """Import every ILU override backend module to fail fast on import drift.

    The fingerprint guard inside each backend class only runs when an instance
    is created, so a renamed community symbol imported at module scope fails
    much later (during model load) as an opaque ``ImportError``. Importing the
    module here — right after registration, before any model loads — turns that
    into an immediate, located startup error. Disable with
    ``VLLM_ILU_BACKEND_IMPORT_GUARD=0`` for debugging.
    """
    global _ILU_BACKEND_MODULES_SMOKED
    if _ILU_BACKEND_MODULES_SMOKED:
        return
    if os.environ.get("VLLM_ILU_BACKEND_IMPORT_GUARD", "1") != "1":
        _ILU_BACKEND_MODULES_SMOKED = True
        return

    import vllm.envs as envs

    paths = list(_ILU_BACKEND_MODULE_PATHS)
    if envs.VLLM_ILU_GDN_ATTN_BACKEND:
        paths.append(_ILU_GDN_BACKEND_MODULE_PATH)

    _ILU_BACKEND_MODULES_SMOKED = True
    for module_path in paths:
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - surface as a located guard error
            raise RuntimeError(
                f"ILU backend import guard: failed to import override module "
                f"{module_path!r} at startup. This usually means a community "
                f"symbol consumed at module scope was renamed/removed upstream; "
                f"re-derive the ILU override against the current community "
                f"source. Underlying error: {exc!r}"
            ) from exc


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
                # gelu_and_mul_sparse has no "iluvatar" impl; mirror the
                # community CUDA default (triton + native fallback).
                gelu_and_mul_sparse=["triton", "native"],
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

            # JIT warmup derives its cache key through ``triton.knobs``, which
            # the CoreX Triton build does not expose. Warmup is a startup-time
            # optimisation only, so drop it rather than half-mirror the helper.
            kernel_config = getattr(vllm_config, "kernel_config", None)
            if kernel_config is not None and kernel_config.enable_jit_warmup:
                try:
                    from triton import knobs as _triton_knobs  # noqa: F401
                except ImportError:
                    kernel_config.enable_jit_warmup = False
                    logger.info(
                        "ILU: disabled enable_jit_warmup; this Triton build "
                        "has no `knobs` API."
                    )

            # Community ModelConfig only auto-selects the deepseek_v4 tokenizer
            # for the upstream "DeepseekV4ForCausalLM" arch. The Iluvatar W4A8
            # arch is OOT-registered under "IluDeepseekV4ForCausalLM", so set its
            # tokenizer_mode here (runs in VllmConfig.__post_init__, before the
            # tokenizer is built). V4.1's upstream name is "DeepseekV41ForCausalLM"
            # (mode deepseek_v41); the W4A8 arch is "IluDeepseekV41ForCausalLM".
            model_config = getattr(vllm_config, "model_config", None)
            if model_config is None:
                return

            from vllm_iluvatar.models.deepseek_v41.engram_storage import (
                validate_engram_config,
            )

            validate_engram_config(vllm_config)

            try:
                archs = model_config.architectures or []
            except Exception:
                archs = []
            if model_config.tokenizer_mode == "auto":
                if "IluDeepseekV41ForCausalLM" in archs:
                    model_config.tokenizer_mode = "deepseek_v41"
                    logger.info(
                        "ILU: set tokenizer_mode='deepseek_v41' for "
                        "IluDeepseekV41ForCausalLM"
                    )
                elif "IluDeepseekV4ForCausalLM" in archs:
                    model_config.tokenizer_mode = "deepseek_v4"
                    logger.info(
                        "ILU: set tokenizer_mode='deepseek_v4' for "
                        "IluDeepseekV4ForCausalLM"
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
            _smoke_import_ilu_backend_modules()

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
