# SPDX-License-Identifier: Apache-2.0
"""Configuration and backend selection for Iluvatar AWQ."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_awq import (
    AutoAWQConfig,
    AutoAWQLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_marlin_supports_layer,
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)


@register_quantization_config("auto_awq")
@register_quantization_config("awq")
@register_quantization_config("awq_marlin")
class IluAutoAWQConfig(AutoAWQConfig):
    """Iluvatar AWQ config registered out-of-tree.

    Uses Iluvatar-owned method names so community ``awq`` / ``auto_awq`` /
    ``awq_marlin`` keep their upstream behavior unless a checkpoint or user
    explicitly asks for the ``ilu_*`` methods.
    """

    def __repr__(self) -> str:
        return (
            f"IluAutoAWQConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "auto_awq"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IluAutoAWQConfig:
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config,
            ["modules_to_not_convert"],
            None,
        )
        # Ensure full_config uses "awq" as quant_method for MoE fallback compatibility.
        # MoeWNA16Config only accepts "gptq" or "awq", so we normalize here.
        full_config = config.copy()
        full_config["quant_method"] = "awq"
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            full_config,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> LinearMethodBase | QuantizeMethodBase | None:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                match_mode="substring",
            ):
                return UnquantizedLinearMethod()
            # Check if Marlin is supported and not using batch invariant mode
            # (Marlin kernels are not batch invariant)
            use_marlin = (
                not envs.VLLM_BATCH_INVARIANT
                and current_platform.is_cuda()
                and check_marlin_supported(
                    self.quant_type, self.group_size, self.zero_point
                )
            )

            if use_marlin:
                # tile-misaligned shapes are fixed by padding at weight prep
                if not check_marlin_supports_layer(
                    layer, self.group_size, allow_tile_padding=True
                ):
                    logger.warning_once(
                        "Layer '%s' is not supported by AutoAWQMarlin. "
                        "Falling back to unoptimized AWQ kernels.",
                        prefix,
                    )
                    return AutoAWQLinearMethod(self)
                from ..backends.awq.linear import IluAWQMarlinLinearMethod

                quant_method = IluAWQMarlinLinearMethod(self)
                quant_method.input_dtype = get_marlin_input_dtype(prefix)
                return quant_method

            return AutoAWQLinearMethod(self)

        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix,
                getattr(self, "modules_to_not_convert", []),
                match_mode="substring",
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)

            if not check_moe_marlin_supports_layer(
                layer, self.group_size, allow_tile_padding=True
            ):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AutoAWQMoEMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )

            from ..backends.awq.moe import IluAutoAWQMoEMethod

            return IluAutoAWQMoEMethod(self, layer.moe_config)

        return None


__all__ = [
    "IluAutoAWQConfig",
]
