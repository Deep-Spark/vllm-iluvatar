# SPDX-License-Identifier: Apache-2.0
"""Default ivcore11 execution policy for native BF16/FP16 checkpoints."""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_iluvatar.config.ilu_compute import (
    native_linear_backend,
    native_moe_compute,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.native.linear import (
    IluNativeW8A8Int8LinearMethod,
)
from vllm_iluvatar.model_executor.layers.quantization.backends.native.moe import (
    IluNativeToW4A8MoEMethod,
)

logger = init_logger(__name__)


class IluNativeQuantConfig(QuantizationConfig):
    def get_name(self):
        return "ilu_native_auto"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict):
        if config:
            raise ValueError("Native execution takes no checkpoint quantization config")
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            backend = native_linear_backend()
            # Attention can read these projection weights directly for grouped
            # BMM rather than invoking the Linear method.
            if (
                backend == "original"
                or prefix.rsplit(".", 1)[-1] in ("kv_b_proj", "wo_a")
                or layer.params_dtype not in (torch.bfloat16, torch.float16)
            ):
                return UnquantizedLinearMethod()
            logger.info_once("ivcore11 native Linear compute: w8a8_int8")
            return IluNativeW8A8Int8LinearMethod()

        if isinstance(layer, RoutedExperts):
            if layer.params_dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(
                    f"{prefix}: native MoE dispatch requires BF16/FP16 weights, "
                    f"got {layer.params_dtype}"
                )
            selection = native_moe_compute()
            if selection is None:
                layer._ilu_require_native_expert_weights = True
                return None
            logger.info_once(
                "ivcore11 native MoE compute: %s_%s",
                selection.name,
                selection.layout.lower(),
            )
            return IluNativeToW4A8MoEMethod(layer.moe_config, selection)

        return None


def native_quant_config(vllm_config):
    """The native dispatch for this deployment, or ``None`` to leave vLLM's.

    Only an unquantized BF16/FP16 checkpoint on ivcore11 is claimed. An explicit
    ``ilu_quant`` keeps its own entry point, and a pre-converted sharded-state
    load arrives with nothing left to convert.
    """
    from vllm_iluvatar.ilu_util.arch import is_ivcore11

    model_config = vllm_config.model_config
    if (
        model_config is None
        or model_config.quantization is not None
        or vllm_config.quant_config is not None
        or model_config.dtype not in (torch.bfloat16, torch.float16)
        or (vllm_config.additional_config or {}).get("ilu_quant")
        or vllm_config.load_config.load_format
        in ("sharded_state", "runai_streamer_sharded")
        or not is_ivcore11()
    ):
        return None
    return IluNativeQuantConfig()
