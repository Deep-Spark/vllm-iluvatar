# SPDX-License-Identifier: Apache-2.0
"""Platform choices for an ixaite process."""

from vllm.logger import init_logger

from vllm_iluvatar.backends import communicator_class_path, worker_class_path
from vllm_iluvatar.platforms.ilu_platform import IluPlatform

logger = init_logger(__name__)

_MINIMAX_FLASH_MODEL_TYPES = frozenset({"minimax_m3", "minimax_m3_vl"})


def is_minimax_flash_hf_config(hf_config) -> bool:
    """Whether this checkpoint is MiniMax-M3.1-Flash (MXFP8 dense + NVFP4 MoE).

    Ships either as the VL wrapper (``minimax_m3_vl``) or the text model.
    ``sparse_attention_config`` plus NVFP4 is not unique on its own, so the
    model type has to match as well.
    """
    inner = getattr(hf_config, "text_config", hf_config)
    model_type = getattr(hf_config, "model_type", None)
    if model_type not in _MINIMAX_FLASH_MODEL_TYPES:
        model_type = getattr(inner, "model_type", None)
    if model_type not in _MINIMAX_FLASH_MODEL_TYPES:
        return False
    if getattr(inner, "sparse_attention_config", None) is None:
        return False
    quant_cfg = getattr(hf_config, "quantization_config", None)
    if not isinstance(quant_cfg, dict):
        return False
    return str(quant_cfg.get("moe_quant_algo", "")).upper() == "NVFP4"


def apply_minimax_flash_piecewise_default(vllm_config) -> None:
    """Default MiniMax-M3.1-Flash decode to PIECEWISE breakable CUDA graphs.

    FULL mode captures the whole decode step into a single opaque replay
    that costs ~1.5x the breakable PIECEWISE routine at TP8 c1. Only the
    community default is replaced, so an explicit ``cudagraph_mode`` or
    ``enforce_eager`` request still wins.
    """
    from vllm.config.compilation import CUDAGraphMode

    model_config = getattr(vllm_config, "model_config", None)
    cc = getattr(vllm_config, "compilation_config", None)
    if model_config is None or cc is None:
        return
    if getattr(model_config, "enforce_eager", False):
        return
    if cc.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE:
        return
    if not is_minimax_flash_hf_config(model_config.hf_config):
        return
    cc.cudagraph_mode = CUDAGraphMode.PIECEWISE
    logger.info(
        "MiniMax-M3.1-Flash decode defaults to PIECEWISE breakable "
        "CUDA graphs (set cudagraph_mode explicitly to override)"
    )


# Reuse hardware queries while selecting ixaite objects before construction.
class IxaitePlatform(IluPlatform):
    # Piecewise fullgraph compilation can trace only the custom-op collectives.
    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True

    # ViT D=72 is padded by the ixaite encoder and must not use decoder widths.
    @classmethod
    def get_vit_attn_backend(cls, head_size, dtype, backend=None):
        import torch
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if backend is not None or dtype == torch.float32:
            return super().get_vit_attn_backend(head_size, dtype, backend)
        return AttentionBackendEnum.FLASH_ATTN

    # Fixed attention options are checked before the builder owns cache state.
    @classmethod
    def get_attn_backend_cls(
        cls, selected_backend, attn_selector_config, num_heads=None
    ):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if attn_selector_config.use_mla or selected_backend not in (
            None,
            AttentionBackendEnum.FLASH_ATTN,
        ):
            raise RuntimeError("ixaite does not support this attention backend")
        from vllm_iluvatar.backends.ixaite.attention.flash_attn import (
            get_flash_attn_backend_class_path,
        )

        return get_flash_attn_backend_class_path(attn_selector_config)

    # Ordinary and stateless groups share the launch-time choice.
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return communicator_class_path()

    # Reject formats whose vendor loaders can repack weights before execution.
    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        if quant not in (
            "compressed-tensors",
            "ilu_deepseek_v4_w4a8",
            "ilu_minimax_m3_flash",
        ):
            raise ValueError(f"ixaite does not support quantization={quant!r}")

    # Preserve common configuration before replacing the default worker path.
    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        if vllm_config.lora_config is not None:
            raise ValueError("ixaite does not support LoRA kernels")
        if "ilu_quant" in vllm_config.additional_config:
            raise ValueError("ixaite does not support runtime ilu_quant conversion")
        super().check_and_update_config(vllm_config)
        pc = vllm_config.parallel_config
        if pc.worker_cls == "vllm_iluvatar.v1.worker.gpu_worker.IluWorker":
            pc.worker_cls = worker_class_path()
        elif pc.worker_cls != worker_class_path():
            from vllm.utils.import_utils import resolve_obj_by_qualname

            from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker

            # A custom worker must retain the owned warmup and graph lifecycle.
            if not issubclass(resolve_obj_by_qualname(pc.worker_cls), IxaiteWorker):
                raise ValueError("ixaite custom worker must derive from IxaiteWorker")
        from vllm_iluvatar.backends.ixaite.envs import configure_peer_access

        configure_peer_access(getattr(pc, "nnodes", 1))

        # MiniMax-M3.1-Flash decode defaults to PIECEWISE breakable graphs;
        # the helper keeps the explicit-mode and enforce-eager contract.
        apply_minimax_flash_piecewise_default(vllm_config)

    # Register only the selected provider; vllm_c RMSNorm enters a vendor native kernel.
    @classmethod
    def import_ir_kernels(cls) -> None:
        import vllm.kernels  # noqa: F401

        import vllm_iluvatar.backends.ixaite.ir  # noqa: F401

    # Independent native Torch formulas remain valid compiler targets.
    @classmethod
    def get_default_ir_op_priority(cls, vllm_config):
        from vllm.config.compilation import CompilationMode
        from vllm.config.kernel import IrOpPriorityConfig

        cc = vllm_config.compilation_config
        if cc.backend == "inductor" and cc.mode != CompilationMode.NONE:
            return IrOpPriorityConfig.with_default(["native"])
        return IrOpPriorityConfig.with_default(["ixaite", "native"])
